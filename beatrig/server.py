"""beatrig preview server: http://127.0.0.1:4700 (localhost only).

Serves the page in ../web and the JSON/binary API (docs/design.md "Preview server" and "Server endpoints added"):
  GET  /              page
  GET  /api/state     metadata only: rig, notes, analysis summary (beats, bpm map; no peaks), bake outputs with key
                      counts (no values, no key lists), band ids, scene.json, live status, flash-safety flags
  GET  /api/curves    ?output=<id> | ?band=<id>, &lod=<n>: binary curve data (format below)
  GET  /api/peaks     ?lod=<n>: binary waveform peaks (uint8 max |mono|, 200 per second at lod 0)
  GET  /api/events    Server-Sent Events: rev (state changed + agent activity), live (push status), playhead (the
                      editor's, <= 30 Hz, only while a page has "link playhead" on). Heartbeat comment every 15 s.
  GET  /api/ops       the engine's op specs (params + defaults, required params, input arity, kind)
  GET  /api/audio     the rig's audio file, with HTTP Range support (the <audio> element seeks with it)
  POST /api/rig       full rig JSON -> validate + re-bake from memory, returns the bake metadata (or 400 + error)
  POST /api/notes     full notes JSON -> re-bake from memory, returns the bake metadata (or 400 + error)
  POST /api/edit      {"rig": ..., "notes": ...} (either may be left out): both in one re-bake, for edits that are only
                      valid together (renaming a section that a layer key names)
  POST /api/save      write rig.json / notes.json to disk (a .bak copy of each first)
  POST /api/notify    {"source": "mcp", "action": "bake", "detail": "..."}: an agent (beatrig/mcp_server.py) tells the
                      page it changed something. If rig.json/notes.json changed on disk they are reloaded and re-baked.
  GET/POST /api/live  {enabled, sequence, link_playhead}: push changed outputs to the editor after every re-bake
  GET/POST /api/playhead   {frame, playing}: the editor's Sequencer playhead (read / set)
  POST /api/scan      refresh scene.json from the editor (beatrig.live.Link.scan)

Binary format (/api/curves, /api/peaks): uint32 little-endian header length H, then H bytes of JSON header (space
padded so the data starts on a 4-byte boundary), then the data. Curves are little-endian float32:
  lod 0       values  [channels][frames]          (colour: [channels][frames][4] linear RGBA)
  lod n > 0   minmax  [channels][buckets][2]      (colour: rgba_mean [channels][buckets][4]), bucket = 4**n frames
Peaks are uint8 (0..255 = 0..1), one per 5 ms * 4**lod.

Every state change bumps `rev` and is announced on /api/events. POST /api/rig and /api/notes accept ?source=<name> so
an agent calling them directly is also logged in the activity strip. Edits stay in memory until /api/save.

The page always sees and saves notes.json as written: pattern placements ({"pattern", "at_beat", "repeat"}) stay
placements (the engine expands them when it evaluates), so editing a pattern moves every copy.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import mimetypes
import os
import queue
import re
import shutil
import sys
import threading
import time
import traceback
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

_HERE = Path(__file__).resolve().parent
# source checkout: <repo>/web ; installed wheel: beatrig/_web
WEB_DIR = next((d for d in (_HERE.parent / "web", _HERE / "_web") if (d / "index.html").is_file()), _HERE.parent / "web")
PEAKS_PER_SEC = 200        # analysis peak grid (5 ms)
CHUNK = 256 * 1024
MAX_LOD = 8
HEARTBEAT_S = 15
PLAYHEAD_HZ = 20           # editor playhead poll while linked (contract: <= 30 Hz)
OFFLINE_RETRY_S = 3.0      # after a failed editor call, wait this long before trying again on our own
SERVER_ID = f"{os.getpid()}-{int(time.time())}"   # lets a page tell a restarted server from a reconnect

mimetypes.add_type("audio/wav", ".wav")
mimetypes.add_type("audio/flac", ".flac")
mimetypes.add_type("text/javascript", ".js")


class RigError(Exception):
    """A validation/bake problem the page should show inline (400, not 500)."""


# --------------------------------------------------------------------------------------------- helpers

def notes_path_for(rig: dict, rig_path: Path) -> Path:
    return (rig_path.parent / rig.get("notes", "notes.json")).resolve()


def audio_path_for(rig: dict, rig_path: Path) -> Path:
    return (rig_path.parent / rig["audio"]).resolve()


def scene_path_for(rig: dict, rig_path: Path) -> Path:
    """rig.scene, else scene.json beside the rig"""
    return (rig_path.parent / (rig.get("scene") or "scene.json")).resolve()


def read_notes(path: Path) -> dict:
    """notes.json as written (pattern references kept); a missing file is empty notes"""
    if not path.is_file():
        return {"bpm_map": [], "grid_offset": 0.0, "tracks": {}}
    n = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(n, dict):
        raise RigError(f"{path.name}: must be a JSON object")
    n.setdefault("bpm_map", [])
    n.setdefault("grid_offset", 0.0)
    n.setdefault("tracks", {})
    return n


def read_scene(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError):
        return None


def binary(header: dict, data: bytes) -> bytes:
    """uint32 LE header length + JSON header padded to a 4-byte boundary + data"""
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    h += b" " * ((-(4 + len(h))) % 4)
    return len(h).to_bytes(4, "little") + h + data


def lod_array(a: np.ndarray, lod: int):
    """a: float32 [C, N] or [C, N, 4] -> (data, layout, bucket). lod 0 = as is; else min/max (or mean RGBA) per 4**lod."""
    if lod <= 0:
        return a, ("rgba" if a.ndim == 3 else "values"), 1
    B = 4 ** lod
    C, N = a.shape[0], a.shape[1]
    M = max(1, -(-N // B))
    pad = M * B - N
    if pad:
        a = np.concatenate([a, np.repeat(a[:, -1:], pad, axis=1)], axis=1)
    if a.ndim == 3:
        return a.reshape(C, M, B, 4).mean(axis=2, dtype=np.float64).astype(np.float32), "rgba_mean", B
    r = a.reshape(C, M, B)
    return np.stack([r.min(axis=2), r.max(axis=2)], axis=2), "minmax", B


def jsonable(v):
    """op-spec defaults and other engine values -> plain JSON (unknown objects become None)"""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, np.generic):
        return v.item()
    return None


# --------------------------------------------------------------------------------------------- engine adapter

class Engine:
    """Adapter over the beatrig package (docs/design.md "Python API")."""

    def __init__(self):
        from . import Rig, RigError as ERigError, OpError, analyze, bake, check, evaluate, load_notes
        from .ops import SPECS, REQ
        self.Rig, self.errors = Rig, (ERigError, OpError)
        self.analyze, self.bake_fn, self.check_fn, self.evaluate, self.load_notes = analyze, bake, check, evaluate, load_notes
        self.specs, self.REQ = SPECS, REQ
        try:
            from .check import ENUMS
            self.enums = dict(ENUMS)
        except ImportError:
            self.enums = {}
        self.safety = None
        try:
            self.safety = importlib.import_module("beatrig.safety")
        except ImportError:
            pass

    def load(self, rig_path: Path):
        rig = json.loads(rig_path.read_text(encoding="utf-8"))
        if not isinstance(rig, dict):
            raise RigError(f"{rig_path.name}: must be a JSON object")
        return rig, read_notes(notes_path_for(rig, rig_path))

    def bake(self, rig: dict, notes: dict, base_dir: Path) -> dict:
        """validate + evaluate + bake once; band nodes are kept as float32 arrays for the band lanes"""
        try:
            r = self.Rig.from_dict(rig, str(base_dir))
            issues = self.check_fn(r, notes)
            errors = [i for i in issues if i["level"] == "error"]
            if errors:
                raise RigError("; ".join(f"{e['where']}: {e['msg']}" for e in errors))
            an = self.analyze(r) if r.audio_path else None
            sigs = self.evaluate(r, an, notes)
            result = self.bake_fn(r, an, notes, signals=sigs)
        except self.errors as e:
            raise RigError(str(e)) from e
        bands = {}
        for n in r.nodes:
            if n.get("op") == "band" and n.get("id") in sigs:
                bands[n["id"]] = {"lo": n.get("lo"), "hi": n.get("hi"), "values": np.ascontiguousarray(sigs[n["id"]].v[:1], np.float32)}
        result["_bands"] = bands
        result["_warnings"] = [f"{i['where']}: {i['msg']}" for i in issues if i["level"] != "error"]
        result["_flags"] = self.flags(r, result)
        return result

    def flags(self, rig, result):
        """flash-safety windows per output when beatrig.safety exists: {output id: [[t0, t1], ...]} in audio-file seconds,
        every channel (and the mean of all channels) merged. Uses flash_report directly, since check_bake caps the list."""
        s = self.safety
        if s is None or not hasattr(s, "flash_report"):
            return None
        try:
            cfg, _ = s.settings(rig.data)
            fps = float(result.get("fps", 30))
            # report windows count seconds from frame 0; audio time = that - offset / fps + audio_in_s
            off = float(result.get("audio_offset_frames", 0)) / fps - float(rig.data.get("audio_in_s") or 0.0)
            by_id = {o.get("id"): o for o in rig.data.get("outputs", [])}
            out = {}
            for o in result.get("outputs", []):
                if not s.monitored(dict(by_id.get(o["id"], {}), target=o["target"]), cfg["outputs"]):
                    continue
                wins = []
                for _label, vals, lo, hi in s.channel_series(o):
                    wins += s.flash_report(vals, fps, lo, hi, cfg["max_flashes_per_s"], cfg["min_swing"])["windows"]
                merged = []
                for a, b in sorted(wins):
                    if merged and a <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], b)
                    else:
                        merged.append([a, b])
                if merged:
                    out[o["id"]] = [[round(a - off, 3), round(b - off, 3)] for a, b in merged]
            return out
        except Exception as e:                          # the adapter must never break a bake
            return {"_error": f"{type(e).__name__}: {e}"}

    def analysis(self, rig: dict, rig_path: Path):
        """(summary without peaks, peaks float array at 200/s)"""
        try:
            a = self.analyze(self.Rig.from_dict(rig, str(rig_path.parent)))
        except self.errors as e:
            raise RigError(str(e)) from e
        if a is None:
            return {}, np.zeros(0, np.float32)
        m = a.summary(1)                                # 1 peak per second: cheap; the real peaks come from the arrays
        arrays = getattr(a, "arrays", None)
        peaks = arrays["peaks"] if arrays is not None and "peaks" in arrays else None
        if peaks is None:
            peaks = np.asarray(a.summary(PEAKS_PER_SEC)["peaks"], np.float32)
        meta = {"duration": m["duration"], "sample_rate": m["sr"],
                "beats": [round(float(b["time"] if isinstance(b, dict) else b), 4) for b in m["beats"]],
                "bpm": m["bpm"], "bpm_map": m["bpm_map"], "grid_offset": m["grid_offset"],
                "pulse": (m.get("pulse") or {}).get("label"), "peaks_per_s": PEAKS_PER_SEC}
        return meta, np.asarray(peaks, np.float32)

    def op_params(self) -> dict:
        """{op: {param: default}} for numeric defaults, so the page can offer sliders for params the rig leaves unset"""
        return {name: {k: v for k, v in spec.params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
                for name, spec in self.specs.items()}

    def ops(self) -> dict:
        """the op registry as JSON for the page's graph editor"""
        out = {}
        for name, spec in sorted(self.specs.items()):
            params, req = {}, []
            for k, v in spec.params.items():
                if v is self.REQ:
                    req.append(k)
                    params[k] = None
                else:
                    params[k] = jsonable(v)
            enums = {k: list(v) for (o, k), v in self.enums.items() if o == name}
            out[name] = {"params": params, "required": req, "inputs": spec.inputs, "enums": enums,
                         "kind": spec.kind if isinstance(spec.kind, str) else None,
                         "one_of": [list(g) for g in (spec.one_of or ())],
                         "doc": ((spec.fn.__doc__ or "").strip().splitlines() or [""])[0]}
        return out


# --------------------------------------------------------------------------------------------- events (SSE)

class Events:
    """Fan-out of server-sent events to every open /api/events stream. publish() never blocks: a client that falls
    behind loses events (each client queue is bounded) and resyncs from the next rev."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clients = set()
        self.changed = threading.Event()

    def subscribe(self):
        q = queue.Queue(maxsize=512)
        with self.lock:
            self.clients.add(q)
        self.changed.set()
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.discard(q)
        self.changed.set()

    def count(self):
        with self.lock:
            return len(self.clients)

    @staticmethod
    def frame(event: str, data) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode("utf-8")

    def publish(self, event: str, data):
        msg = self.frame(event, data)
        with self.lock:
            clients = list(self.clients)
        for q in clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass


# --------------------------------------------------------------------------------------------- live link

class Live:
    """Thin adapter over beatrig.live.Link (docs/design.md "Live link"). Every editor call goes through self.call(), one
    at a time. Pushes run on a worker thread, coalesced to the newest bake, so a re-bake never waits for the editor.
    After a failure the editor counts as offline and is retried on its own only every OFFLINE_RETRY_S seconds;
    explicit actions (enable, scan, set playhead) always try."""

    def __init__(self, events: Events):
        self.events = events
        self.enabled, self.sequence, self.link_playhead = False, "", False
        self.link = None
        self.state = "off"                  # off | ok | offline | unavailable
        self.error = None
        self.last = None                    # {pushed, skipped, ms, at}
        self.editor = None                  # last editor playhead {frame, playing}
        self.call_lock = threading.Lock()
        self.cv = threading.Condition()
        self.pending = None
        self.retry_at = 0.0
        threading.Thread(target=self._push_worker, daemon=True, name="beatrig-live-push").start()
        threading.Thread(target=self._playhead_worker, daemon=True, name="beatrig-live-playhead").start()

    # -- plumbing
    def status(self) -> dict:
        return {"ok": True, "enabled": self.enabled, "sequence": self.sequence, "link_playhead": self.link_playhead,
                "state": self.state, "error": self.error, "last": self.last, "pending": self.pending is not None}

    def announce(self):
        self.events.publish("live", self.status())

    def _module(self):
        try:
            return importlib.import_module("beatrig.live")
        except ImportError as e:
            raise RigError(f"live link not available: {e}") from e

    def _connect(self):
        if self.link is None:
            self.link = self._module().Link(os.environ.get("BEATRIG_UE_HOST"))   # None = the Link's default host
        return self.link

    def _close_if_idle(self):
        """drop the editor session when neither push nor playhead link needs it (Link.close restores the editor's
        background-CPU setting that the Link turned off)"""
        if self.enabled or self.link_playhead or self.link is None:
            return
        with self.call_lock:
            link, self.link = self.link, None
            try:
                link.close()
            except Exception:
                pass

    def call(self, fn_name, *args, force=False, **kw):
        """run one Link method; marks the editor offline on failure and raises RigError"""
        if not force and self.state == "offline" and time.monotonic() < self.retry_at:
            raise RigError(f"editor offline: {self.error}")
        with self.call_lock:
            try:
                link = self._connect()
                out = getattr(link, fn_name)(*args, **kw)
            except RigError as e:
                self.state, self.error = "unavailable", str(e)
                self.announce()
                raise
            except Exception as e:                      # connection refused, timeouts, editor-side errors
                # the Link is kept: it reconnects and re-boots its editor module by itself (also after an editor restart)
                self.state, self.error = "offline", f"{type(e).__name__}: {e}"
                self.retry_at = time.monotonic() + OFFLINE_RETRY_S
                self.announce()
                raise RigError(f"editor offline: {self.error}") from e
        if self.state != "ok":
            self.state, self.error = "ok", None
            self.announce()
        return out

    # -- push
    def queue_push(self, bake: dict):
        if not self.enabled or not bake:
            return
        with self.cv:
            self.pending = bake
            self.cv.notify()

    def _push_worker(self):
        while True:
            with self.cv:
                while self.pending is None:
                    self.cv.wait()
                bake, self.pending = self.pending, None
            if not self.enabled or not self.sequence:
                continue
            try:
                clean = {k: v for k, v in bake.items() if not k.startswith("_")}
                res = self.call("push", clean, self.sequence, only_changed=True, force=True) or {}
                errs = res.get("errors") or []
                self.last = {"pushed": jsonable(res.get("pushed")), "skipped": jsonable(res.get("skipped")),
                             "removed": jsonable(res.get("removed")), "keys": jsonable(res.get("keys")),
                             "ms": jsonable(res.get("ms")), "at": time.strftime("%H:%M:%S"), "ok": bool(res.get("ok", True)),
                             "errors": [str(e.get("error") or e) if isinstance(e, dict) else str(e) for e in errs][:5]}
            except RigError:
                pass
            except Exception as e:
                self.state, self.error = "offline", f"{type(e).__name__}: {e}"
            self.announce()

    # -- playhead (editor -> page), only while a page asked for it and is listening
    def _playhead_worker(self):
        last = None
        while True:
            if not (self.link_playhead and self.events.count() > 0):
                self.events.changed.wait(1.0)
                self.events.changed.clear()
                last = None
                continue
            t0 = time.monotonic()
            try:
                ph = self.call("playhead_get")
                ph = jsonable(ph) if isinstance(ph, dict) else {"frame": jsonable(ph)}
                key = (ph.get("frame"), ph.get("playing"))
                if key != last:
                    last = key
                    self.editor = ph
                    self.events.publish("playhead", ph)
            except RigError:
                time.sleep(OFFLINE_RETRY_S)
            time.sleep(max(0.0, 1.0 / PLAYHEAD_HZ - (time.monotonic() - t0)))

    # -- page requests
    def configure(self, body: dict, current_bake):
        if not isinstance(body, dict):
            raise RigError("live body must be a JSON object")
        if "sequence" in body:
            seq = body["sequence"]
            if seq is not None and not isinstance(seq, str):
                raise RigError("sequence must be a string")
            self.sequence = (seq or "").strip()
        if "link_playhead" in body:
            self.link_playhead = bool(body["link_playhead"])
            self.events.changed.set()
        if "enabled" in body:
            on = bool(body["enabled"])
            if on and not self.sequence:
                raise RigError("live needs a sequence path, e.g. /Game/Show/LS_Main")
            if on:
                self._module()                          # fail fast with a clear message when the module is missing
            self.enabled = on
            if on:
                self.state = "ok" if self.state in ("off", "unavailable") else self.state
                self.queue_push(current_bake)
        if not self.enabled and not self.link_playhead:
            self.state, self.error = "off", None
            self._close_if_idle()
        self.announce()
        return self.status()


# --------------------------------------------------------------------------------------------- state

class State:
    def __init__(self, rig_path: Path, engine: Engine, events: Events):
        self.rig_path = rig_path.resolve()
        self.base_dir = self.rig_path.parent
        self.engine, self.events = engine, events
        self.lock = threading.Lock()
        self.rig, self.notes = engine.load(self.rig_path)
        self.dirty = False
        self.bake = None
        self.bake_ms = None
        self.bake_id = 0
        self.arrays = {}                    # ("output"|"band", id) -> float32 array, for the current bake
        self.hashes = {}
        self.cache = {}                     # (kind, id, lod) -> encoded bytes
        self._an_key = None
        self._an_meta, self._peaks = {}, np.zeros(0, np.float32)
        self._peak_cache = {}
        self.rev = 1
        self.activity = deque(maxlen=20)
        self.live = Live(events)
        self.scene = read_scene(scene_path_for(self.rig, self.rig_path))
        self.rebake(self.rig, self.notes)
        self._disk = self._disk_stamp()

    def _disk_stamp(self):
        out = []
        for p in (self.rig_path, notes_path_for(self.rig, self.rig_path)):
            try:
                st = p.stat()
                out.append((st.st_size, st.st_mtime_ns))
            except OSError:
                out.append(None)
        return out

    def rev_event(self) -> dict:
        return {"rev": self.rev, "server": SERVER_ID, "dirty": self.dirty, "activity": list(self.activity)}

    def bump(self):
        self.rev += 1
        self.events.publish("rev", self.rev_event())

    def log(self, source, action, detail=""):
        self.activity.append({"rev": self.rev + 1, "time": time.strftime("%H:%M:%S"), "source": source,
                              "action": action, "detail": detail})
        self.bump()

    def notify(self, source, action, detail):
        """An agent says it changed something. Reload from disk when the files changed; never silently
        drop unsaved page edits (then it's logged as a conflict instead)."""
        stamp = self._disk_stamp()
        if stamp != self._disk:
            if self.dirty:
                detail = (detail + " | " if detail else "") + \
                    "rig/notes changed on disk but the page has unsaved edits: not reloaded (Save would overwrite)"
            else:
                try:
                    rig, notes = self.engine.load(self.rig_path)
                    self.rebake(rig, notes)
                except (RigError, ValueError, OSError) as e:
                    self.log(source, action, f"reload from disk failed: {e}")
                    raise RigError(f"reload from disk failed: {e}") from e
                self._disk = stamp
                detail = (detail + " | " if detail else "") + "reloaded from disk"
        self.scene = read_scene(scene_path_for(self.rig, self.rig_path))
        self.log(source, action, detail)

    # Every engine call goes through here, under the lock; on failure the previous good state is kept.
    def rebake(self, rig: dict, notes: dict) -> dict:
        if not isinstance(rig, dict) or not isinstance(notes, dict):
            raise RigError("rig and notes must be JSON objects")
        t0 = time.perf_counter()
        try:
            self.analysis_meta(rig)                     # fails early and clearly when the audio is missing
            bake = self.engine.bake(rig, notes, self.base_dir)
        except RigError:
            raise
        except (ValueError, KeyError, TypeError, FileNotFoundError) as e:
            raise RigError(f"{type(e).__name__}: {e}") from e
        arrays, hashes = {}, {}
        for o in bake.get("outputs", []):
            a = np.asarray([ch.get("values") or [] for ch in o.get("channels", [])], np.float32)
            arrays[("output", o["id"])] = a
            hashes[o["id"]] = hashlib.blake2b(a.tobytes() + str(a.shape).encode(), digest_size=8).hexdigest()
        for bid, b in bake.get("_bands", {}).items():
            arrays[("band", bid)] = b["values"]
            hashes["band:" + bid] = hashlib.blake2b(b["values"].tobytes(), digest_size=8).hexdigest()
        self.bake_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.rig, self.notes, self.bake = rig, notes, bake
        self.arrays, self.hashes, self.cache = arrays, hashes, {}
        self.bake_id += 1
        self.live.queue_push(bake)
        return bake

    def analysis_meta(self, rig=None) -> dict:
        rig = rig or self.rig
        if not rig.get("audio"):
            self._an_key, self._an_meta, self._peaks = None, {}, np.zeros(0, np.float32)
            return self._an_meta
        ap = audio_path_for(rig, self.rig_path)
        if not ap.is_file():
            raise RigError(f"audio not found: {ap.name}")
        st = ap.stat()
        key = (str(ap), st.st_size, st.st_mtime_ns, json.dumps(rig.get("analysis"), sort_keys=True),
               json.dumps(rig.get("channel_weights")))
        if key != self._an_key:
            self._an_meta, self._peaks = self.engine.analysis(rig, self.rig_path)
            self._peak_cache = {}
            self._an_key = key
        return self._an_meta

    def bake_meta(self) -> dict:
        """the bake without per-frame values or key lists: outputs, channels (actor + key count) and content hashes"""
        b = self.bake or {}
        out = {k: v for k, v in b.items() if k != "outputs" and not k.startswith("_")}
        outs = []
        for o in b.get("outputs", []):
            oc = {k: v for k, v in o.items() if k != "channels"}
            a = self.arrays.get(("output", o["id"]))
            oc["components"] = 4 if a is not None and a.ndim == 3 else 1
            oc["hash"] = self.hashes.get(o["id"])
            oc["channels"] = [dict({k: v for k, v in ch.items() if k not in ("values", "keys")},
                                   key_count=len(ch.get("keys") or [])) for ch in o.get("channels", [])]
            outs.append(oc)
        out["outputs"] = outs
        out["bake_id"] = self.bake_id
        return out

    def bake_payload(self) -> dict:
        b = self.bake or {}
        return {"bake": self.bake_meta(),
                "bands": [{"id": k, "lo": v["lo"], "hi": v["hi"], "hash": self.hashes.get("band:" + k)} for k, v in b.get("_bands", {}).items()],
                "warnings": b.get("_warnings", []), "flags": b.get("_flags"), "bake_ms": self.bake_ms,
                "bake_id": self.bake_id}

    def snapshot(self) -> dict:
        return {
            "rev": self.rev,
            "server": SERVER_ID,
            "activity": list(self.activity),
            "rig_path": str(self.rig_path),
            "notes_path": str(notes_path_for(self.rig, self.rig_path)),
            "scene_path": str(scene_path_for(self.rig, self.rig_path)),
            "rig": self.rig,
            "notes": self.notes,
            "dirty": self.dirty,
            "analysis": self.analysis_meta(),
            "scene": self.scene,
            "live": self.live.status(),
            "op_params": self.engine.op_params(),
            "safety": self.engine.safety is not None,
            **self.bake_payload(),
        }

    def curves(self, kind: str, cid: str, lod: int) -> bytes:
        key = (kind, cid, lod)
        if key in self.cache:
            return self.cache[key]
        a = self.arrays.get((kind, cid))
        if a is None:
            raise KeyError(f"no {kind} {cid!r} in the current bake")
        data, layout, bucket = lod_array(a, lod)
        hdr = {"kind": kind, "id": cid, "bake_id": self.bake_id, "lod": lod, "bucket": bucket, "layout": layout,
               "channels": int(a.shape[0]), "frames": int(a.shape[1]), "count": int(data.shape[1]),
               "components": 4 if a.ndim == 3 else 1, "hash": self.hashes.get(cid if kind == "output" else "band:" + cid),
               "fps": (self.bake or {}).get("fps"), "audio_offset_frames": (self.bake or {}).get("audio_offset_frames", 0)}
        blob = binary(hdr, np.ascontiguousarray(data, "<f4").tobytes())
        self.cache[key] = blob
        return blob

    def peaks(self, lod: int) -> bytes:
        if lod in self._peak_cache:
            return self._peak_cache[lod]
        p = np.clip(self._peaks, 0, 1)
        B = 4 ** lod
        if B > 1 and len(p):
            M = -(-len(p) // B)
            p = np.concatenate([p, np.zeros(M * B - len(p), p.dtype)]).reshape(M, B).max(axis=1)
        q = np.round(p * 255).astype(np.uint8)
        blob = binary({"lod": lod, "bucket": B, "rate": PEAKS_PER_SEC / B, "count": int(len(q))}, q.tobytes())
        self._peak_cache[lod] = blob
        return blob

    def save(self):
        saved = []
        targets = [(self.rig_path, self.rig), (notes_path_for(self.rig, self.rig_path), self.notes)]
        for path, obj in targets:
            if path.exists():
                shutil.copy2(path, path.with_name(path.name + ".bak"))
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, path)
            saved.append(str(path))
        self.dirty = False
        self._disk = self._disk_stamp()
        return saved

    def write_scene(self, scene: dict) -> Path:
        path = scene_path_for(self.rig, self.rig_path)
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(scene, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        self.scene = scene
        return path


# --------------------------------------------------------------------------------------------- HTTP

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


class Handler(BaseHTTPRequestHandler):
    server_version = "beatrig"
    state: State = None  # set in run()

    def log_message(self, fmt, *args):
        if os.environ.get("BEATRIG_LOG"):
            super().log_message(fmt, *args)

    # -- responses
    def _json(self, obj, status=200):
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self._bytes(body, "application/json; charset=utf-8", status)

    def _bytes(self, body: bytes, ctype: str, status=200, head=False):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            try:
                self.wfile.write(body)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass

    def _error(self, msg, status=400, **extra):
        self._json({"ok": False, "error": msg, **extra}, status)

    def _file(self, path: Path, head=False):
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng and rng.strip().startswith("bytes="):   # other units are ignored (RFC 9110): full 200
            m = RANGE_RE.match(rng.strip())
            if not m or (m.group(1) == "" and m.group(2) == ""):
                return self._range_unsatisfiable(size)
            a, b = m.group(1), m.group(2)
            if a == "":                     # suffix: last N bytes
                n = int(b)
                if n == 0:
                    return self._range_unsatisfiable(size)
                start, end = max(0, size - n), size - 1
            else:
                start = int(a)
                end = min(int(b), size - 1) if b else size - 1
                if start >= size or end < start:
                    return self._range_unsatisfiable(size)
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if head:
            return
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    buf = f.read(min(CHUNK, left))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    left -= len(buf)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass  # the <audio> element drops connections when it seeks

    def _range_unsatisfiable(self, size):
        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _static(self, rel: str, head=False):
        root = WEB_DIR.resolve()
        p = (WEB_DIR / rel).resolve()
        if (root not in p.parents and p != root) or not p.is_file():
            return self._error("not found", 404)
        self._file(p, head)

    def _events(self):
        """GET /api/events: hold the connection open and stream server-sent events until the client goes away"""
        st = self.state
        q = st.events.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            with st.lock:
                first = Events.frame("rev", st.rev_event())
            self.wfile.write(b"retry: 1500\n\n" + first + Events.frame("live", st.live.status()))
            if st.live.link_playhead and st.live.editor:
                self.wfile.write(Events.frame("playhead", st.live.editor))
            self.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=HEARTBEAT_S)
                except queue.Empty:
                    msg = b": hb\n\n"
                self.wfile.write(msg)
                self.wfile.flush()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
            pass
        finally:
            st.events.unsubscribe(q)

    # -- routes
    def do_HEAD(self):
        self._get(head=True)

    def do_GET(self):
        self._get(head=False)

    def _get(self, head):
        url = urlparse(self.path)
        path = url.path
        qs = parse_qs(url.query)
        st = self.state
        try:
            if path == "/" or path == "/index.html":
                return self._static("index.html", head)
            if path == "/api/state":
                since = qs.get("since", [None])[0]
                with st.lock:
                    if since is not None and since == str(st.rev):
                        return self._json({"ok": True, "rev": st.rev, "unchanged": True, "activity": list(st.activity)})
                    snap = st.snapshot()
                return self._json({"ok": True, **snap})
            if path == "/api/curves":
                lod = max(0, min(MAX_LOD, int(qs.get("lod", ["0"])[0])))
                kind, cid = ("band", qs["band"][0]) if "band" in qs else ("output", (qs.get("output") or [""])[0])
                if not cid:
                    raise RigError("curves needs ?output=<id> or ?band=<id>")
                with st.lock:
                    try:
                        blob = st.curves(kind, cid, lod)
                    except KeyError as e:
                        return self._error(str(e.args[0]), 404)
                return self._bytes(blob, "application/octet-stream", head=head)
            if path == "/api/peaks":
                lod = max(0, min(MAX_LOD, int(qs.get("lod", ["0"])[0])))
                with st.lock:
                    blob = st.peaks(lod)
                return self._bytes(blob, "application/octet-stream", head=head)
            if path == "/api/events":
                return self._events()
            if path == "/api/ops":
                return self._json({"ok": True, "ops": st.engine.ops()})
            if path == "/api/live":
                return self._json(st.live.status())
            if path == "/api/playhead":
                try:
                    ph = st.live.call("playhead_get", force=True)
                finally:
                    st.live._close_if_idle()
                ph = jsonable(ph) if isinstance(ph, dict) else {"frame": jsonable(ph)}
                return self._json({"ok": True, **ph})
            if path == "/api/audio":
                with st.lock:
                    ap = audio_path_for(st.rig, st.rig_path)
                if not ap.is_file():
                    return self._error(f"audio not found: {ap.name}", 404)
                return self._file(ap, head)
            if path.startswith("/api/"):
                return self._error("unknown endpoint", 404)
            return self._static(path.lstrip("/"), head)
        except RigError as e:
            return self._error(str(e), 400)
        except ValueError as e:
            return self._error(f"bad request: {e}", 400)
        except Exception as e:
            traceback.print_exc()
            return self._error(f"{type(e).__name__}: {e}", 500)

    def _body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw.decode("utf-8") or "null")
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise RigError(f"invalid JSON: {e}") from e

    def do_POST(self):
        url = urlparse(self.path)
        path = url.path
        source = (parse_qs(url.query).get("source", ["page"])[0] or "page")[:40]
        st = self.state
        try:
            if path in ("/api/rig", "/api/notes", "/api/edit"):
                obj = self._body_json()
                with st.lock:
                    if path == "/api/edit":
                        if not isinstance(obj, dict) or not ({"rig", "notes"} & set(obj)):
                            raise RigError('edit needs {"rig": {...}, "notes": {...}} (either may be left out)')
                        rig, notes = obj.get("rig", st.rig), obj.get("notes", st.notes)
                    else:
                        rig, notes = (obj, st.notes) if path == "/api/rig" else (st.rig, obj)
                    st.rebake(rig, notes)
                    if path != "/api/notes":
                        st.scene = read_scene(scene_path_for(st.rig, st.rig_path))
                    st.dirty = True
                    if source == "page":
                        st.bump()
                    else:
                        st.log(source, path.rsplit("/", 1)[1], f"re-baked in {st.bake_ms} ms")
                    return self._json({"ok": True, "rev": st.rev, "dirty": st.dirty, **st.bake_payload()})
            if path == "/api/save":
                with st.lock:
                    saved = st.save()
                    if source == "page":
                        st.bump()
                    else:
                        st.log(source, "save", ", ".join(Path(p).name for p in saved))
                    return self._json({"ok": True, "rev": st.rev, "saved": saved, "dirty": False})
            if path == "/api/notify":
                obj = self._body_json()
                if not isinstance(obj, dict):
                    raise RigError("notify body must be a JSON object")
                fields = {k: obj.get(k, "") for k in ("source", "action", "detail")}
                if not all(isinstance(v, str) for v in fields.values()) or not fields["action"]:
                    raise RigError('notify needs string fields: {"source", "action", "detail"}')
                with st.lock:
                    st.notify(fields["source"][:40] or "agent", fields["action"][:40], fields["detail"][:300])
                    return self._json({"ok": True, "rev": st.rev})
            if path == "/api/live":
                obj = self._body_json()
                with st.lock:
                    bake = st.bake
                return self._json(st.live.configure(obj, bake))
            if path == "/api/playhead":
                obj = self._body_json()
                if not isinstance(obj, dict) or not isinstance(obj.get("frame"), (int, float)):
                    raise RigError('playhead needs {"frame": <number>, "playing": <bool>}')
                try:
                    # with a live sequence set, only ever move that sequence's playhead, never another one that
                    # happens to be open in Sequencer
                    want = st.live.sequence.split(".")[0]
                    if want:
                        cur = st.live.call("playhead_get", force=True) or {}
                        have = (cur.get("sequence") or "").split(".")[0]
                        if have and have != want:
                            raise RigError(f"Sequencer shows {have}, not the live sequence {want}: playhead not moved")
                    res = st.live.call("playhead_set", int(obj["frame"]), bool(obj.get("playing", False)), force=True)
                finally:
                    st.live._close_if_idle()
                res = jsonable(res) if isinstance(res, dict) else {}
                if res.get("ok") is False:
                    raise RigError(f"editor: {res.get('error') or 'playhead not set'}")
                return self._json({"ok": True, **{k: v for k, v in res.items() if k != "ok"}})
            if path == "/api/scan":
                try:
                    scene = st.live.call("scan", force=True)
                finally:
                    st.live._close_if_idle()
                if not isinstance(scene, dict):
                    raise RigError("scan returned no scene")
                with st.lock:
                    p = st.write_scene(scene)
                    detail = f"{len(scene.get('actors') or {})} actors -> {p.name}"
                    if not st.rig.get("scene"):
                        # the engine reads positions only from rig.scene: reference the file (unsaved, like any edit)
                        try:
                            st.rebake(dict(st.rig, scene=os.path.relpath(p, st.base_dir).replace(os.sep, "/")), st.notes)
                            st.dirty = True
                            detail += f"; rig.scene set to {p.name} (unsaved)"
                        except RigError as e:
                            detail += f"; rig.scene not set: {e}"
                    st.log(source, "scan", detail)
                    return self._json({"ok": True, "rev": st.rev, "scene": scene, "path": str(p), "dirty": st.dirty})
            return self._error("unknown endpoint", 404)
        except RigError as e:
            return self._error(str(e), 400)
        except Exception as e:
            traceback.print_exc()
            return self._error(f"{type(e).__name__}: {e}", 500)


def run(rig_path: str, port: int = 4700):
    """Serve the preview page for `rig_path` on http://127.0.0.1:<port> until Ctrl+C."""
    events = Events()
    Handler.state = State(Path(rig_path), Engine(), events)
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        hint = "Is another beatrig already running?"
        if getattr(e, "winerror", None) == 10013:
            # Windows reserves dynamic port blocks (Hyper-V/WinNAT); see `netsh int ipv4 show excludedportrange protocol=tcp`
            hint = "Windows has this port in an excluded range (netsh int ipv4 show excludedportrange protocol=tcp); try --port."
        sys.exit(f"beatrig: cannot bind 127.0.0.1:{port} ({e}). {hint}")
    httpd.daemon_threads = True
    print(f"beatrig preview: http://127.0.0.1:{port}  rig={Handler.state.rig_path}  bake {Handler.state.bake_ms} ms",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
