"""beatrig preview server: http://127.0.0.1:4700 (localhost only).

Serves the page in ../web and the JSON API from docs/design.md "Preview server":
  GET  /              page
  GET  /api/state     rig, notes, analysis summary (peaks, beats, bpm map, bands), bake (downsampled)
  GET  /api/audio     the rig's audio file, with HTTP Range support (the <audio> element seeks with it)
  POST /api/rig       full rig JSON -> validate + re-bake from memory, returns the bake (or 400 + error)
  POST /api/notes     full notes JSON -> re-bake from memory, returns the bake (or 400 + error)
  POST /api/save      write rig.json / notes.json to disk (a .bak copy of each first)
  POST /api/notify    {"source": "mcp", "action": "bake", "detail": "..."}: an agent (beatrig/mcp_server.py) tells the
                      page it changed something. If rig.json/notes.json changed on disk they are reloaded and re-baked.

Every state change bumps `rev`. The page polls GET /api/state?since=<rev>; an unchanged rev returns only
{rev, unchanged, activity}, so polling is cheap. POST /api/rig and /api/notes accept ?source=<name> so an agent
calling them directly is also logged in the activity strip.

Edits stay in memory until /api/save; GET /api/state reports `dirty` so a reload keeps the unsaved marker.
Stdlib only, plus numpy when it's importable (the engine needs it anyway) for fast waveform peaks.

Engine: the real package (`from . import ...`). Set BEATRIG_FAKE=1 (or run(..., fake=True)) to use the small
fake engine at the bottom of this file instead; it exists only to develop the page without the engine.
"""
from __future__ import annotations

import json
import math
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import traceback
import wave
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_HERE = Path(__file__).resolve().parent
# source checkout: <repo>/web ; installed wheel: beatrig/_web
WEB_DIR = next((d for d in (_HERE.parent / "web", _HERE / "_web") if (d / "index.html").is_file()), _HERE.parent / "web")
MAX_POINTS = 2000          # per channel, docs/design.md
PEAKS_PER_SEC = 200        # waveform peak pairs per second, docs/design.md
CHUNK = 256 * 1024

mimetypes.add_type("audio/wav", ".wav")
mimetypes.add_type("audio/flac", ".flac")
mimetypes.add_type("text/javascript", ".js")


class RigError(Exception):
    """A validation/bake problem the page should show inline (400, not 500)."""


# --------------------------------------------------------------------------------------------- helpers

def downsample(values, max_points=MAX_POINTS):
    """Values for one channel -> (points, stride). Above max_points it returns min/max pairs per bucket
    (interleaved min,max,min,max...), so hits survive; stride = frames per bucket."""
    n = len(values)
    if n <= max_points:
        return [round(float(v), 6) for v in values], 1
    stride = math.ceil(n / (max_points // 2))
    out = []
    for i in range(0, n, stride):
        seg = values[i:i + stride]
        out.append(round(float(min(seg)), 6))
        out.append(round(float(max(seg)), 6))
    return out, stride


def slim_bake(bake: dict) -> dict:
    """bake.json -> the /api/state form: per-channel values downsampled (keys kept whole)."""
    out = {k: v for k, v in bake.items() if k != "outputs" and not k.startswith("_")}
    outs = []
    for o in bake.get("outputs", []):
        oc = {k: v for k, v in o.items() if k != "channels"}
        chans = []
        for ch in o.get("channels", []):
            c = {k: v for k, v in ch.items() if k != "values"}
            vals = ch.get("values")
            if vals is not None:
                pts, stride = downsample(list(vals))
                c["values"] = pts
                c["stride"] = stride
                c["minmax"] = stride > 1
            c["keys"] = [[int(f), round(float(v), 6)] for f, v in ch.get("keys", [])]
            chans.append(c)
        oc["channels"] = chans
        outs.append(oc)
    out["outputs"] = outs
    return out


def wav_peaks(path: Path, rate=PEAKS_PER_SEC):
    """Downmixed min/max peaks as int8 pairs (-127..127), `rate` pairs per second. WAV only (stdlib reader)."""
    with wave.open(str(path), "rb") as w:
        nch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    hop = max(1, sr // rate)
    try:
        import numpy as np
        if sw == 2:
            a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
        elif sw == 4:
            a = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
        elif sw == 3:
            b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
            i = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
            a = np.where(i >= 1 << 23, i - (1 << 24), i).astype(np.float32) / 8388608.0
        elif sw == 1:
            a = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128.0
        else:
            raise RigError(f"unsupported wav sample width {sw}")
        a = a.reshape(-1, nch).mean(axis=1)
        m = (len(a) // hop) * hop
        tail = a[m:]
        blocks = a[:m].reshape(-1, hop)
        mins, maxs = blocks.min(axis=1), blocks.max(axis=1)
        if len(tail):
            mins = np.append(mins, tail.min())
            maxs = np.append(maxs, tail.max())
        data = np.empty(len(mins) * 2, np.int16)
        data[0::2] = np.clip(np.round(mins * 127), -127, 127)
        data[1::2] = np.clip(np.round(maxs * 127), -127, 127)
        data = data.tolist()
    except ImportError:
        import array
        if sw != 2:
            raise RigError("waveform without numpy needs 16-bit wav")
        a = array.array("h")
        a.frombytes(raw)
        data = []
        step = hop * nch
        for i in range(0, len(a), step):
            seg = a[i:i + step]
            mono = [sum(seg[j:j + nch]) / nch for j in range(0, len(seg), nch)]
            data += [round(min(mono) / 32768 * 127), round(max(mono) / 32768 * 127)]
    return {"rate": sr / hop, "data": data}, n / sr, sr


def notes_path_for(rig: dict, rig_path: Path) -> Path:
    return (rig_path.parent / rig.get("notes", "notes.json")).resolve()


def audio_path_for(rig: dict, rig_path: Path) -> Path:
    return (rig_path.parent / rig["audio"]).resolve()


# --------------------------------------------------------------------------------------------- engine adapter

class RealEngine:
    """Adapter over the beatrig package (docs/design.md "Python API" + bake_from_objects)."""
    name = "real"

    def __init__(self):
        from . import (Rig, RigError as ERigError, OpError, analyze, bake, check, evaluate, load_notes)
        from .ops import SPECS
        self.Rig, self.errors = Rig, (ERigError, OpError)
        self.analyze, self.bake_fn, self.check_fn, self.evaluate, self.load_notes = analyze, bake, check, evaluate, load_notes
        self.specs = SPECS

    def load(self, rig_path: Path):
        rig = json.loads(rig_path.read_text(encoding="utf-8"))
        return rig, self.load_notes(str(notes_path_for(rig, rig_path)))

    def check(self, rig: dict, base_dir: Path):
        return []  # bake() runs the engine's check() and raises on errors

    def bake(self, rig: dict, notes: dict, base_dir: Path) -> dict:
        """Same as bake_from_objects(), but evaluates once itself so the band nodes can be previewed too."""
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
        bands = []
        for n in r.nodes:
            if n.get("op") == "band" and n.get("id") in sigs:
                pts, stride = downsample(sigs[n["id"]].v[0].tolist())
                bands.append({"id": n["id"], "lo": n.get("lo"), "hi": n.get("hi"), "values": pts, "stride": stride,
                              "minmax": stride > 1})
        result["_bands"] = bands
        result["_warnings"] = [f"{i['where']}: {i['msg']}" for i in issues if i["level"] != "error"]
        return result

    def analysis_summary(self, rig: dict, rig_path: Path) -> dict:
        try:
            a = self.analyze(self.Rig.from_dict(rig, str(rig_path.parent)))
        except self.errors as e:
            raise RigError(str(e)) from e
        if a is None:
            return {}
        m = a.summary(PEAKS_PER_SEC)
        q = [int(round(min(1.0, p) * 127)) for p in m["peaks"]]   # max |mono| per bucket -> symmetric int8 pairs
        data = [0] * (2 * len(q))
        data[0::2] = [-v for v in q]
        data[1::2] = q
        return {"duration": m["duration"], "sample_rate": m["sr"], "peaks": {"rate": m["peaks_per_s"], "data": data},
                "beats": [round(float(b["time"] if isinstance(b, dict) else b), 4) for b in m["beats"]],
                "bpm": m["bpm"], "bpm_map": m["bpm_map"], "grid_offset": m["grid_offset"],
                "pulse": (m.get("pulse") or {}).get("label")}

    def op_params(self) -> dict:
        """{op: {param: default}} for numeric defaults, so the page can offer sliders for params the rig leaves unset."""
        out = {}
        for name, spec in self.specs.items():
            out[name] = {k: v for k, v in spec.params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        return out


# --------------------------------------------------------------------------------------------- state

class State:
    def __init__(self, rig_path: Path, engine):
        self.rig_path = rig_path.resolve()
        self.base_dir = self.rig_path.parent
        self.engine = engine
        self.lock = threading.Lock()
        self.rig, self.notes = engine.load(self.rig_path)
        self.dirty = False
        self.bake = None
        self.bake_ms = None
        self._peaks_key = None
        self._analysis = None
        self.rev = 1
        self.activity = deque(maxlen=20)
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

    def log(self, source, action, detail=""):
        self.rev += 1
        self.activity.append({"rev": self.rev, "time": time.strftime("%H:%M:%S"), "source": source,
                              "action": action, "detail": detail})

    def notify(self, source, action, detail):
        """An agent says it changed something. Reload from disk when the files changed; never silently
        drop unsaved page edits (then it's logged as a conflict instead)."""
        stamp = self._disk_stamp()
        if stamp != self._disk:
            if self.dirty:
                detail = (detail + " | " if detail else "") +                     "rig/notes changed on disk but the page has unsaved edits: not reloaded (Save would overwrite)"
            else:
                try:
                    rig, notes = self.engine.load(self.rig_path)
                    self.rebake(rig, notes)
                except (RigError, ValueError, OSError) as e:
                    self.log(source, action, f"reload from disk failed: {e}")
                    raise RigError(f"reload from disk failed: {e}") from e
                self._disk = stamp
                detail = (detail + " | " if detail else "") + "reloaded from disk"
        self.log(source, action, detail)

    # Every engine call goes through here, under the lock; on failure the previous good state is kept.
    def rebake(self, rig: dict, notes: dict) -> dict:
        if not isinstance(rig, dict) or not isinstance(notes, dict):
            raise RigError("rig and notes must be JSON objects")
        errs = self.engine.check(rig, self.base_dir)
        if errs:
            raise RigError("; ".join(errs))
        t0 = time.perf_counter()
        try:
            bake = self.engine.bake(rig, notes, self.base_dir)
        except RigError:
            raise
        except (ValueError, KeyError, TypeError, FileNotFoundError) as e:
            raise RigError(f"{type(e).__name__}: {e}") from e
        self.bake_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.rig, self.notes, self.bake = rig, notes, bake
        return bake

    def analysis(self) -> dict:
        ap = audio_path_for(self.rig, self.rig_path)
        st = ap.stat()
        key = (str(ap), st.st_size, st.st_mtime_ns)
        if key != self._peaks_key:
            self._analysis = self.engine.analysis_summary(self.rig, self.rig_path)
            self._peaks_key = key
        return self._analysis

    def snapshot(self) -> dict:
        b = self.bake or {}
        return {
            "rev": self.rev,
            "activity": list(self.activity),
            "engine": self.engine.name,
            "rig_path": str(self.rig_path),
            "notes_path": str(notes_path_for(self.rig, self.rig_path)),
            "rig": self.rig,
            "notes": self.notes,
            "dirty": self.dirty,
            "bake_ms": self.bake_ms,
            "analysis": self.analysis(),
            "bake": slim_bake(b),
            "bands": b.get("_bands", []),
            "warnings": b.get("_warnings", []),
            "op_params": self.engine.op_params(),
        }

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
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

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
        p = (WEB_DIR / rel).resolve()
        if WEB_DIR.resolve() not in p.parents and p != WEB_DIR.resolve() or not p.is_file():
            return self._error("not found", 404)
        self._file(p, head)

    # -- routes
    def do_HEAD(self):
        self._get(head=True)

    def do_GET(self):
        self._get(head=False)

    def _get(self, head):
        url = urlparse(self.path)
        path = url.path
        st = self.state
        try:
            if path == "/" or path == "/index.html":
                return self._static("index.html", head)
            if path == "/api/state":
                since = parse_qs(url.query).get("since", [None])[0]
                with st.lock:
                    if since is not None and since == str(st.rev):
                        return self._json({"ok": True, "rev": st.rev, "unchanged": True,
                                           "activity": list(st.activity)})
                    snap = st.snapshot()
                return self._json({"ok": True, **snap})
            if path == "/api/audio":
                with st.lock:
                    ap = audio_path_for(st.rig, st.rig_path)
                if not ap.is_file():
                    return self._error(f"audio not found: {ap}", 404)
                return self._file(ap, head)
            if path.startswith("/api/"):
                return self._error("unknown endpoint", 404)
            return self._static(path.lstrip("/"), head)
        except RigError as e:
            return self._error(str(e), 400)
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
            if path in ("/api/rig", "/api/notes"):
                obj = self._body_json()
                with st.lock:
                    if path == "/api/rig":
                        st.rebake(obj, st.notes)
                    else:
                        st.rebake(st.rig, obj)
                    st.dirty = True
                    if source == "page":
                        st.rev += 1
                    else:
                        st.log(source, "rig" if path == "/api/rig" else "notes", f"re-baked in {st.bake_ms} ms")
                    return self._json({"ok": True, "rev": st.rev, "bake": slim_bake(st.bake),
                                       "bands": st.bake.get("_bands", []), "warnings": st.bake.get("_warnings", []),
                                       "bake_ms": st.bake_ms, "dirty": st.dirty})
            if path == "/api/save":
                with st.lock:
                    saved = st.save()
                    if source == "page":
                        st.rev += 1
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
            return self._error("unknown endpoint", 404)
        except RigError as e:
            return self._error(str(e), 400)
        except Exception as e:
            traceback.print_exc()
            return self._error(f"{type(e).__name__}: {e}", 500)


def run(rig_path: str, port: int = 4700, fake: bool | None = None):
    """Serve the preview page for `rig_path` on http://127.0.0.1:<port> until Ctrl+C."""
    if fake is None:
        fake = os.environ.get("BEATRIG_FAKE") == "1"
    engine = FakeEngine() if fake else RealEngine()
    Handler.state = State(Path(rig_path), engine)
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        hint = "Is another beatrig already running?"
        if getattr(e, "winerror", None) == 10013:
            # Windows reserves dynamic port blocks (Hyper-V/WinNAT); see `netsh int ipv4 show excludedportrange protocol=tcp`
            hint = "Windows has this port in an excluded range (netsh int ipv4 show excludedportrange protocol=tcp); try --port."
        sys.exit(f"beatrig: cannot bind 127.0.0.1:{port} ({e}). {hint}")
    httpd.daemon_threads = True
    print(f"beatrig preview ({engine.name} engine): http://127.0.0.1:{port}  rig={Handler.state.rig_path}"
          f"  bake {Handler.state.bake_ms} ms", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


# --------------------------------------------------------------------------------------------- fake engine
# Development stand-in only (BEATRIG_FAKE=1). Crude, but bake-shaped, so every view has something to draw.

class FakeEngine:
    name = "fake"
    OPS = {"band", "normalize", "smooth", "notes", "envelope", "chase", "math", "lfo", "const"}

    def load(self, rig_path: Path):
        rig = json.loads(rig_path.read_text(encoding="utf-8"))
        return rig, json.loads(notes_path_for(rig, rig_path).read_text(encoding="utf-8"))

    def check(self, rig, base_dir):
        errs = []
        ids = set()
        for n in rig.get("nodes", []):
            if n.get("op") not in self.OPS:
                errs.append(f"node '{n.get('id')}': unknown op '{n.get('op')}'")
            ids.add(n.get("id"))
        for o in rig.get("outputs", []):
            if o.get("signal") not in ids:
                errs.append(f"output '{o.get('id')}': unknown signal '{o.get('signal')}'")
        return errs

    def analysis_summary(self, rig, rig_path):
        peaks, dur, sr = wav_peaks(audio_path_for(rig, rig_path))
        return {"duration": dur, "sample_rate": sr, "peaks": peaks, "beats": [], "bpm": None}

    def op_params(self):
        return {}

    def bake(self, rig, notes, base_dir):
        fps = rig["fps"]
        ap = (Path(base_dir) / rig["audio"]).resolve()
        peaks, dur, _ = wav_peaks(ap, rate=fps)
        frames = rig.get("frames") or int(dur * fps)
        off = rig.get("audio_offset_frames", 0)
        pk = peaks["data"]
        env = [0.0] * frames
        for f in range(frames):
            i = (f - off) * 2
            if 0 <= i < len(pk):
                env[f] = (pk[i + 1] - pk[i]) / 254.0
        bpm = (notes.get("bpm_map") or [{"bpm": 120}])[0]["bpm"]
        fpb = fps * 60.0 / bpm
        sig, trig = {}, set()
        for n in rig["nodes"]:
            op, nid = n["op"], n["id"]
            src = lambda k="in": sig[n[k]]
            if op == "band":
                sig[nid] = [env[:]]
            elif op == "normalize":
                ch = src()
                hi = max(max(c) for c in ch) or 1.0
                sig[nid] = [[min(1.0, v / hi) for v in c] for c in ch]
            elif op == "smooth":
                a = math.exp(-1000.0 / fps / max(1e-3, n.get("attack_ms", 10)))
                r = math.exp(-1000.0 / fps / max(1e-3, n.get("release_ms", 150)))
                out = []
                for c in src():
                    y, o = 0.0, []
                    for v in c:
                        k = a if v > y else r
                        y = k * y + (1 - k) * v
                        o.append(y)
                    out.append(o)
                sig[nid] = out
            elif op == "notes":
                c = [0.0] * frames
                for h in notes.get("tracks", {}).get(n["track"], []):
                    f = math.ceil(h["time"] * fps - 1e-9) + off
                    if 0 <= f < frames:
                        c[f] = max(c[f], h.get("vel", 1.0))
                        trig.add(f)
                sig[nid] = [c]
            elif op == "envelope":
                L = max(1.0, n.get("length_beats", 0.5) * fpb)
                out = []
                for c in src():
                    o, cur = [0.0] * frames, 0.0
                    last, vel = None, 0.0
                    for f in range(frames):
                        if c[f] > 0:
                            last, vel = f, c[f]
                        if last is not None:
                            t = (f - last) / L
                            cur = vel * max(0.0, 1 - t) ** 2 if t < 1 else 0.0
                        o[f] = cur
                    out.append(o)
                sig[nid] = out
            elif op == "chase":
                step = n.get("step_beats", 0.125) * fpb
                base = src()[0]
                sig[nid] = [[base[f - int(round(i * step))] if f - int(round(i * step)) >= 0 else 0.0
                             for f in range(frames)] for i in range(int(n.get("channels", 8)))]
            elif op == "math":
                a, b = sig[n["in"][0]], sig[n["in"][1]]
                N = max(len(a), len(b))
                fn = {"max": max, "min": min, "add": lambda x, y: x + y, "mul": lambda x, y: x * y,
                      "sub": lambda x, y: x - y}.get(n.get("fn", "max"), max)
                sig[nid] = [[fn(a[i % len(a)][f], b[i % len(b)][f]) for f in range(frames)] for i in range(N)]
            elif op == "lfo":
                rate = n.get("rate_hz") or (bpm / 60.0 / n.get("rate_beats", 1))
                sig[nid] = [[0.5 + 0.5 * math.sin(2 * math.pi * rate * f / fps + n.get("phase", 0) * 2 * math.pi)
                             for f in range(frames)]]
            elif op == "const":
                sig[nid] = [[float(n.get("value", 0))] * frames]
        outputs = []
        tol = rig.get("bake", {}).get("key_tolerance", 0.002)
        for o in rig["outputs"]:
            lo, hi = o.get("min", 0.0), o.get("max", 1.0)
            ch = sig[o["signal"]]
            actors = o["target"].get("actors") or [None]
            if len(ch) == 1 and len(actors) > 1:
                ch = ch * len(actors)
            chans = []
            for i, c in enumerate(ch):
                vals = [lo + v * (hi - lo) for v in c]
                chans.append({"actor": actors[i] if i < len(actors) else None, "values": vals,
                              "keys": _rdp(vals, tol * abs(hi - lo) or 1e-9, trig)})
            outputs.append({"id": o["id"], "target": o["target"], "min": lo, "max": hi, "channels": chans})
        bands = []
        for n in rig["nodes"]:
            if n["op"] == "band":
                pts, stride = downsample(sig[n["id"]][0])
                bands.append({"id": n["id"], "lo": n.get("lo"), "hi": n.get("hi"), "values": pts, "stride": stride,
                              "minmax": stride > 1})
        return {"version": 1, "fps": fps, "frames": frames, "audio": rig["audio"], "rig_hash": "fake",
                "outputs": outputs, "_bands": bands}


def _rdp(vals, eps, keep):
    n = len(vals)
    if n < 3:
        return [[i, v] for i, v in enumerate(vals)]
    mark = [False] * n
    mark[0] = mark[-1] = True
    for f in keep:
        if 0 <= f < n:
            mark[f] = True
            if f > 0:
                mark[f - 1] = True
    fixed = [i for i in range(n) if mark[i]]
    stack = list(zip(fixed, fixed[1:]))
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        va, vb = vals[a], vals[b]
        best, bi = -1.0, -1
        for i in range(a + 1, b):
            d = abs(vals[i] - (va + (vb - va) * (i - a) / (b - a)))
            if d > best:
                best, bi = d, i
        if best > eps:
            mark[bi] = True
            stack.append((a, bi))
            stack.append((bi, b))
    return [[i, vals[i]] for i in range(n) if mark[i]]
