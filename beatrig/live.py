"""Live link: a persistent Remote Control session from beatrig to a running Unreal editor.

    from beatrig.live import Link
    link = Link("127.0.0.1:30010")
    link.push(bake, "/Game/Show/LS_Main")      # rewrites only the outputs whose keys changed since the last push
    link.scan()                                # the scene.json dict (actors, components, material params, MPCs, camera)
    link.playhead_get()                        # {"sequence", "frame", "playing", ...} of the sequence open in Sequencer
    link.playhead_set(120, playing=True)

How it talks to the editor. The first call bootstraps once: one ExecuteStatement puts the repo's ue/ folder on the
editor's sys.path and imports ue/beatrig_apply.py as the module `beatrig_apply` (reloaded only when the files change).
Every later call is one short ExecuteStatement, `beatrig_apply.rpc("<payload.json>")`: the request goes in a temp JSON
file, the result comes back in <payload.json>.out. One keep-alive HTTP connection is reused, and a lock makes threads
take turns (the editor runs every call on its game thread anyway).

Speed. With the editor in the foreground a call costs ~10 ms plus the editor's work. With it in the background and
"Use Less CPU when in Background" on (the default), the editor ticks a few times a second and every call waits for a
tick (~330 ms here). Link(keep_awake=True), the default, turns that setting off in the editor's memory for the session
and restores it on close() or interpreter exit. It is never written to the ini.

Needs Remote Control (and the Python Editor Script Plugin) with PythonScriptLibrary.ExecutePythonCommandEx allowed;
see docs/unreal.md. The editor must see the same temp folder, so it runs on the same machine.
"""
import atexit
import hashlib
import http.client
import importlib.util
import json
import os
import socket
import sys
import tempfile
import threading
import time

DEFAULT_HOST = "127.0.0.1:30010"
_RC_PATH = "/remote/object/call"
_PY_LIB = "/Script/PythonScriptPlugin.Default__PythonScriptLibrary"
_STALE_CONN = (http.client.RemoteDisconnected, http.client.BadStatusLine, ConnectionResetError, ConnectionAbortedError,
               BrokenPipeError)


class LiveError(RuntimeError):
    """the editor couldn't be reached, refused the call, or the call failed (the message says what to do)"""


def ue_dir():
    """the folder holding beatrig_apply.py: ue/ in a clone, beatrig/_ue in an installed wheel"""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.join(os.path.dirname(here), "ue"), os.path.join(here, "_ue")):
        if os.path.exists(os.path.join(d, "beatrig_apply.py")):
            return d
    raise LiveError("ue/beatrig_apply.py not found next to the package (install from a clone of the repo)")


_KEYS = None


def _keys():
    """ue/keys.py (pure, no unreal import), loaded by path under the same unique name the editor uses"""
    global _KEYS
    if _KEYS is None:
        spec = importlib.util.spec_from_file_location("beatrig_keys", os.path.join(ue_dir(), "keys.py"))
        _KEYS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_KEYS)
    return _KEYS


# ------------------------------------------------------------------------------------------------ bake shaping
def _load_bake(bake):
    if isinstance(bake, dict):
        return bake
    with open(os.fspath(bake), encoding="utf-8") as fh:
        return json.load(fh)


def _slim_output(o):
    """what the applier needs from an output: id, target, value_type and per-channel keys (no per-frame values)"""
    s = {"id": o["id"], "target": o.get("target") or {},
         "channels": [{"actor": c.get("actor"), "keys": c.get("keys") or []} for c in o.get("channels", [])]}
    if o.get("value_type"):
        s["value_type"] = o["value_type"]
    return s


def slim_bake(bake, ids=None):
    """a bake.json dict reduced to what the editor reads (keys only), optionally just the outputs in ids"""
    return {"version": bake.get("version", 1), "fps": bake.get("fps", 30), "frames": bake.get("frames"),
            "outputs": [_slim_output(o) for o in bake.get("outputs", []) if ids is None or o.get("id") in ids]}


def output_hash(output, fps, start=0):
    """content hash of everything the applier writes for one output: target, value type and keys (not the per-frame
    values, the min/max or the rig hash), plus the bake fps and the start frame"""
    s = _slim_output(output)
    s["fps"], s["start"] = fps, int(start)
    return hashlib.sha1(json.dumps(s, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def track_count(output):
    """how many tracks the applier writes for an output (one per channel with keys)"""
    return sum(1 for c in output.get("channels", []) if c.get("keys"))


# ------------------------------------------------------------------------------------------------ the link
class Link:
    """A persistent session with one editor. Thread-safe; calls are serialised.

    Every method returns the editor's result dict (always with "ok"), plus "rt_ms" (the round trip) and "editor_ms"
    (time spent inside the editor). Failures to reach or use the editor raise LiveError."""

    def __init__(self, host=None, ue_dir_=None, timeout=120.0, keep_awake=True):
        host = host or os.environ.get("BEATRIG_UE_HOST") or DEFAULT_HOST
        h, _, p = str(host).replace("http://", "").rstrip("/").partition(":")
        self.host, self.port = h or "127.0.0.1", int(p or 30010)
        self.ue_dir = os.path.abspath(ue_dir_ or ue_dir())
        self.timeout = float(timeout)
        self.keep_awake = bool(keep_awake)
        self._conn = None
        self._lock = threading.RLock()
        self._booted = False
        self._restore_throttle = None
        self._src = (None, None)
        self._pushed = {}                          # (sequence, start) -> {output id: [hash, track count]}
        self._payload = os.path.join(tempfile.gettempdir(),
                                     f"beatrig-live-{os.getpid()}-{id(self):x}.json").replace("\\", "/")
        self.last = {}                             # last call: op, rt_ms, editor_ms
        self.connects = 0                          # TCP connections opened (1 while keep-alive holds)
        self.calls = 0
        atexit.register(self.close)

    # ---------------------------------------------------------------- transport
    @property
    def url(self):
        return f"http://{self.host}:{self.port}{_RC_PATH}"

    def _exec(self, command, mode="ExecuteStatement"):
        """one ExecutePythonCommandEx call; returns the parsed reply. Reconnects once if a kept-alive socket died."""
        body = json.dumps({"objectPath": _PY_LIB, "functionName": "ExecutePythonCommandEx",
                           "parameters": {"PythonCommand": command, "ExecutionMode": mode,
                                          "FileExecutionScope": "Public"},
                           "generateTransaction": False}).encode()
        for attempt in (0, 1):
            reused = self._conn is not None
            if self._conn is None:
                conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
                try:
                    conn.connect()              # refused or timed out here = no editor (Windows refuses after ~2 s)
                except OSError as e:
                    conn.close()
                    raise LiveError(f"no Unreal editor answering at {self.url} ({e}). Open the project with the "
                                    f"Remote Control and Python plugins enabled, or pass host=") from None
                self._conn = conn
                self.connects += 1
            conn = self._conn
            try:
                conn.request("PUT", _RC_PATH, body=body, headers={"Content-Type": "application/json"})
                resp = conn.getresponse()
                data = resp.read()
            except _STALE_CONN as e:
                self._drop()
                if attempt == 0 and reused:
                    continue
                raise LiveError(f"the editor at {self.url} dropped the connection ({type(e).__name__}). If it crashed, "
                                f"check its Saved/Logs before retrying.") from None
            except socket.timeout:
                self._drop()
                raise LiveError(f"the editor at {self.url} did not answer within {self.timeout:g} s; the call may still "
                                f"be running on its game thread") from None
            except OSError as e:
                self._drop()
                raise LiveError(f"no Unreal editor answering at {self.url} ({e}). Open the project with the Remote "
                                f"Control and Python plugins enabled, or pass host=") from None
            self.calls += 1
            if resp.getheader("Connection", "").lower() == "close":
                self._drop()
            if resp.status in (403, 404):
                raise LiveError(f"Remote Control at {self.url} returned HTTP {resp.status}: allowlist "
                                f"PythonScriptLibrary.ExecutePythonCommandEx and restart the editor (docs/unreal.md)")
            if resp.status != 200:
                raise LiveError(f"Remote Control at {self.url} returned HTTP {resp.status}: {data[:200]!r}")
            try:
                reply = json.loads(data.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                raise LiveError(f"unexpected Remote Control reply: {data[:200]!r}") from None
            if reply.get("errorMessage"):
                raise LiveError(f"Remote Control: {reply['errorMessage']}")
            return reply

    def _drop(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None

    @staticmethod
    def _log(reply, n=8):
        lines = [str(e.get("Output", "")).strip() for e in reply.get("LogOutput") or []]
        return " | ".join(x for x in lines[-n:] if x)[:900]

    def _src_hash(self):
        """source hash of ue/*.py, recomputed only when a file's size or mtime changes"""
        names = sorted(n for n in os.listdir(self.ue_dir) if n.endswith(".py"))
        stamp = tuple((n, os.stat(os.path.join(self.ue_dir, n)).st_mtime_ns, os.stat(os.path.join(self.ue_dir, n)).st_size)
                      for n in names)
        if self._src[0] != stamp:
            self._src = (stamp, _keys().source_hash(self.ue_dir))
        return self._src[1]

    def _boot(self):
        """load ue/beatrig_apply.py in the editor as `beatrig_apply` (from exactly this folder; reloaded when its
        source hash differs) and put the folder on sys.path"""
        h = self._src_hash()
        d = self.ue_dir.replace("\\", "/")
        code = (
            "import sys, os, importlib.util\n"
            f"d = {d!r}\n"
            "if d not in sys.path: sys.path.append(d)\n"
            "m = sys.modules.get('beatrig_apply')\n"
            "same = m is not None and os.path.normcase(os.path.dirname(os.path.abspath(m.__file__))) == os.path.normcase(os.path.abspath(d))\n"
            f"if not (same and getattr(m, 'SRC_HASH', None) == {h!r}):\n"
            "    spec = importlib.util.spec_from_file_location('beatrig_apply', os.path.join(d, 'beatrig_apply.py'))\n"
            "    m = importlib.util.module_from_spec(spec)\n"
            "    sys.modules['beatrig_apply'] = m\n"
            "    spec.loader.exec_module(m)\n"
            "print('BEATRIG_BOOT ' + m.SRC_HASH)\n")
        reply = self._exec(f"exec({code!r}, {{}})")
        log = self._log(reply)
        if f"BEATRIG_BOOT {h}" not in log:
            raise LiveError(f"could not load beatrig_apply in the editor: {log or reply}")
        self._booted = True
        if self.keep_awake and self._restore_throttle is None:
            r = self._rpc("throttle", {"enabled": False}, retry=False)
            if r.get("was"):
                self._restore_throttle = True

    def _rpc(self, op, args=None, retry=True):
        with self._lock:
            if not self._booted:
                self._boot()
            req = {"op": op, "args": args or {}, "src": self._src_hash()}
            out = self._payload + ".out"
            if os.path.exists(out):
                os.remove(out)
            with open(self._payload, "w", encoding="utf-8") as fh:
                json.dump(req, fh, separators=(",", ":"))
            t0 = time.perf_counter()
            reply = self._exec(f"__import__('beatrig_apply').rpc({self._payload!r})")
            rt = (time.perf_counter() - t0) * 1000
            if not os.path.exists(out):
                log = self._log(reply)
                if retry:                          # editor restarted or module gone: boot again, once
                    self._booted = False
                    return self._rpc(op, args, retry=False)
                raise LiveError(f"the editor ran no beatrig call for {op!r}: {log or reply}")
            with open(out, encoding="utf-8") as fh:
                res = json.load(fh)
            os.remove(out)
            if res.get("stale_module") and retry:
                self._booted = False
                return self._rpc(op, args, retry=False)
            res["rt_ms"] = round(rt, 2)
            self.last = {"op": op, "rt_ms": res["rt_ms"], "editor_ms": res.get("editor_ms")}
            return res

    # ---------------------------------------------------------------- API
    def hello(self):
        """engine version, open level, the sequence open in Sequencer, the module hash, and whether the editor is
        throttled in the background"""
        return self._rpc("hello")

    def apply(self, bake, sequence, only=None, start=0, save=False, refresh=True):
        """write a bake (dict or path) into a Level Sequence; only = output ids. refresh re-evaluates the sequence if
        it is the one open in Sequencer."""
        b = slim_bake(_load_bake(bake), only)
        fps = b.get("fps", 30)
        hashes = {o["id"]: output_hash(o, fps, start) for o in b["outputs"]}
        res = self._rpc("apply", {"sequence": sequence, "bake": b, "only": None, "start": int(start),
                                  "save": bool(save), "refresh": bool(refresh), "hashes": hashes})
        self._remember(sequence, start, b, res)
        return res

    def check(self, bake, sequence, only=None, start=0):
        """read the keys back from the sequence and compare them with the bake, key for key"""
        return self._rpc("check", {"sequence": sequence, "bake": slim_bake(_load_bake(bake), only), "start": int(start)})

    def undo(self, sequence, output_id=None, save=False, refresh=True):
        """remove beatrig's tracks (all, or one output's) and the now-empty bindings beatrig created"""
        res = self._rpc("undo", {"sequence": sequence, "output_id": output_id, "save": bool(save),
                                 "refresh": bool(refresh)})
        self.forget(sequence, output_id)
        return res

    def push(self, bake, sequence, only_changed=True, start=0, refresh=True, focus=False):
        """Write the bake's outputs into the sequence, skipping outputs whose content hash matches the last push of
        this Link to this sequence. Outputs pushed earlier but gone from the bake are removed. One editor call (two if
        the editor reports an output it no longer holds as pushed, e.g. after an undo elsewhere). focus=True opens
        the sequence in Sequencer first. Returns {"ok", "pushed", "skipped", "removed", "errors", "keys", "ms",
        "rt_ms", "editor_ms", "write_ms", "calls", "refreshed"} (pushed/skipped/removed are output ids)."""
        t0 = time.perf_counter()
        b = _load_bake(bake)
        fps = b.get("fps", 30)
        cur = {o["id"]: [output_hash(o, fps, start), track_count(o)] for o in b.get("outputs", [])}
        with self._lock:
            cache = self._pushed.setdefault((sequence, int(start)), {})
            changed = [i for i in cur if not only_changed or cache.get(i) != cur[i]]
            expect = {i: cur[i] for i in cur if i not in changed}
            gone = [i for i in cache if i not in cur]
            res = self._push_call(b, sequence, changed, cur, expect, gone, start, refresh, focus)
            calls, rt, ed = 1, res.get("rt_ms", 0), res.get("editor_ms", 0)
            stale = res.get("stale") or []
            if stale:
                res2 = self._push_call(b, sequence, stale, cur, {}, [], start, refresh, False)
                calls, rt, ed = 2, rt + res2.get("rt_ms", 0), ed + res2.get("editor_ms", 0)
                res["written"] = res.get("written", []) + res2.get("written", [])
                res["errors"] = res.get("errors", []) + res2.get("errors", [])
                res["keys"] = res.get("keys", 0) + res2.get("keys", 0)
                res["write_ms"] = res.get("write_ms", 0) + res2.get("write_ms", 0)
                res["refreshed"] = res2.get("refreshed", res.get("refreshed"))
                res["ok"] = res.get("ok", False) and res2.get("ok", False)
                changed = changed + stale
            failed = {e.get("output") for e in res.get("errors", [])}
            for i in changed:
                if i in failed:
                    cache.pop(i, None)
                else:
                    cache[i] = cur[i]
            for i in gone:
                cache.pop(i, None)
        pushed = [i for i in cur if i in changed and i not in failed]
        return {"ok": bool(res.get("ok", False)), "pushed": pushed, "skipped": [i for i in cur if i not in changed],
                "removed": gone, "errors": res.get("errors", []), "keys": res.get("keys", 0),
                "ms": round((time.perf_counter() - t0) * 1000, 2), "rt_ms": round(rt, 2), "editor_ms": round(ed, 2),
                "write_ms": res.get("write_ms"), "refresh_ms": res.get("refresh_ms"), "calls": calls,
                "refreshed": bool(res.get("refreshed")), "bindings": res.get("bindings")}

    def _push_call(self, b, sequence, ids, cur, expect, gone, start, refresh, focus):
        return self._rpc("push", {"sequence": sequence, "bake": slim_bake(b, set(ids)),
                                  "hashes": {i: cur[i][0] for i in ids}, "expect": expect, "remove": gone,
                                  "start": int(start), "refresh": bool(refresh), "focus": bool(focus)})

    def _remember(self, sequence, start, bake, res):
        """after a plain apply, treat the applied outputs as pushed (so the next push can skip them)"""
        fps = bake.get("fps", 30)
        cache = self._pushed.setdefault((sequence, int(start)), {})
        failed = {e.get("output") for e in res.get("errors", [])}
        for o in bake.get("outputs", []):
            if o["id"] in failed:
                cache.pop(o["id"], None)
            else:
                cache[o["id"]] = [output_hash(o, fps, start), track_count(o)]

    def forget(self, sequence=None, output_id=None):
        """drop the push cache (all, one sequence, or one output in it): the next push rewrites those outputs"""
        with self._lock:
            for key in list(self._pushed):
                if sequence is None or key[0] == sequence:
                    if output_id is None:
                        del self._pushed[key]
                    else:
                        self._pushed[key].pop(output_id, None)

    def scan(self, paths=None):
        """the scene.json dict for the open level (see docs/design.md); paths = content folders searched for MPCs
        and Level Sequences (default ["/Game"])"""
        res = self._rpc("scan", {"paths": list(paths) if paths else None})
        if not res.get("ok"):
            raise LiveError(f"scan failed in the editor: {res.get('error')}")
        scene = res["scene"]
        scene.setdefault("scan_ms", {})["rt"] = res.get("rt_ms")
        return scene

    def playhead_get(self):
        """{"sequence", "frame" (display frames), "playing", "fps", "range"} of the sequence open in Sequencer
        (sequence None when nothing is open)"""
        return self._rpc("playhead_get")

    def playhead_set(self, frame=None, playing=None, sequence=None):
        """move the playhead (display frames) and/or play (True) / pause (False). sequence: open it in Sequencer first
        if it isn't the open one (this replaces whatever Sequencer was showing)."""
        return self._rpc("playhead_set", {"frame": frame, "playing": playing, "sequence": sequence})

    def ensure_sequence(self, path, fps=30, frames=None):
        """create a Level Sequence at path if missing (an existing one is left untouched)"""
        return self._rpc("ensure_sequence", {"path": path, "fps": fps, "frames": frames})

    def throttle(self, enabled=None):
        """read or set the editor's 'Use Less CPU when in Background' (in memory only)"""
        return self._rpc("throttle", {"enabled": enabled})

    def status(self):
        """what the Link knows without calling the editor"""
        return {"host": f"{self.host}:{self.port}", "booted": self._booted, "connected": self._conn is not None,
                "connects": self.connects, "calls": self.calls, "last": self.last, "keep_awake": self.keep_awake,
                "sequences": sorted({k[0] for k in self._pushed})}

    def close(self):
        """restore the editor's background throttle (if this Link turned it off) and drop the connection"""
        with self._lock:
            if self._restore_throttle is not None and self._booted:
                try:
                    self._rpc("throttle", {"enabled": True}, retry=False)
                except Exception:
                    pass
            self._restore_throttle = None
            self._drop()
            for p in (self._payload, self._payload + ".out"):
                try:
                    os.remove(p)
                except OSError:
                    pass
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------------------------------------------ CLI: beatrig scan
def summarize(scene):
    """one screen of text about a scene.json dict"""
    kinds = {}
    for name, a in scene.get("actors", {}).items():
        for c in a.get("components", []):
            kinds.setdefault(c["kind"], []).append(name)
    lines = [f"level     {scene.get('level')}  ({scene.get('engine', '')})",
             f"actors    {len(scene.get('actors', {}))}"]
    for k in ("light", "mesh", "niagara", "camera"):
        names = sorted(set(kinds.get(k, [])))
        if names:
            more = f" ... (+{len(names) - 12})" if len(names) > 12 else ""
            lines.append(f"  {k:8} {len(names):4}  {', '.join(names[:12])}{more}")
    for path, m in sorted(scene.get("mpcs", {}).items()):
        lines.append(f"mpc       {path}  scalars {m['scalars']}  vectors {m['vectors']}")
    lines.append(f"sequences {len(scene.get('sequences', []))}")
    cam = scene.get("camera")
    if cam:
        lines.append(f"camera    {cam['actor']} ({cam.get('projection')}, dome {cam.get('dome_angle')}) at {cam['location']}")
    ms = scene.get("scan_ms", {})
    lines.append(f"scan      {ms.get('total')} ms in the editor, {ms.get('rt')} ms round trip")
    return "\n".join(lines)


def cli_scan(a):
    """`beatrig scan [--out scene.json] [--host H] [--paths /Game ...]`; a = argparse namespace or argv list"""
    if a is None or isinstance(a, (list, tuple)):
        import argparse
        ap = argparse.ArgumentParser(prog="beatrig scan")
        _scan_args(ap)
        a = ap.parse_args(list(a or []))
    try:
        with Link(getattr(a, "host", None), keep_awake=False) as link:
            scene = link.scan(getattr(a, "paths", None))
    except LiveError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = getattr(a, "out", None) or "scene.json"
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(scene, indent=2, ensure_ascii=False))
        fh.write("\n")
    os.replace(tmp, out)
    print(summarize(scene))
    print(f"wrote     {os.path.abspath(out)}")
    return 0


def _scan_args(p):
    p.add_argument("--out", default="scene.json", help="where to write scene.json (default ./scene.json)")
    p.add_argument("--host", default=None, help="Remote Control host:port (default BEATRIG_UE_HOST or 127.0.0.1:30010)")
    p.add_argument("--paths", nargs="*", default=None, help="content folders searched for MPCs and sequences (default /Game)")


def add_cli(sub):
    """register `scan` on beatrig's CLI subparsers (cli.py: live.add_cli(sub))"""
    p = sub.add_parser("scan", help="scan the open Unreal level into scene.json (needs a running editor)")
    _scan_args(p)
    p.set_defaults(fn=cli_scan)
    return p


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m beatrig.live")
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_cli(sub)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
