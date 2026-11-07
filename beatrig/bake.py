"""Evaluate a rig and bake it: per-frame values per target channel plus RDP-simplified linear keys (bake.json)."""
import json
import logging
import math
import os
import time
from typing import Optional

import numpy as np

from .analysis import Analysis, open_analysis
from .check import check, order_nodes
from .notes import load_notes, normalise
from .ops import Ctx, OpError, SPECS, input_ids
from .rig import Rig, RigError, sha256
from .signal import Signal
from .tempo import Tempo

log = logging.getLogger("beatrig")
BAKE_VERSION = 1


# ---------- setup
def analysis_reqs(rig: Rig) -> list:
    """every band/loudness power series the rig needs, so one audio pass computes them all"""
    reqs = []
    for n in rig.nodes:
        if n.get("op") == "band":
            reqs.append(("band", n.get("lo"), n.get("hi"), n.get("weights")))
        elif n.get("op") == "loudness":
            reqs.append(("loud", None, None, n.get("weights")))
        elif n.get("op") == "onsets" and n.get("in") is not None:
            ref = n["in"]
            src = ref if isinstance(ref, dict) else rig.node(ref) or {}
            if src.get("lo") is not None or src.get("hi") is not None:
                reqs.append(("band", src.get("lo"), src.get("hi"), src.get("weights")))
    return reqs


def analyze(rig: Rig, use_cache: bool = True) -> Optional[Analysis]:
    """Build or load the analysis cache (analysis.json/.npz next to the rig) and compute every band the rig uses.
    Returns None for a rig without audio. The cache is invalidated by the audio's size/mtime or the analysis settings."""
    if not rig.audio_path:
        return None
    if not os.path.exists(rig.audio_path):
        hint = " (generate the demo track with: python examples/make_demo_audio.py)" if "demo_110bpm" in rig.audio_path else ""
        raise RigError(f"audio not found: {rig.audio_path}{hint}")
    a = open_analysis(rig.audio_path, rig.base_dir, rig.data.get("analysis"), rig.data.get("channel_weights"), use_cache)
    a.ensure(analysis_reqs(rig))
    a.save()
    return a


def frame_count(rig: Rig, analysis: Optional[Analysis]) -> int:
    if rig.data.get("frames") is not None:
        return int(rig.data["frames"])
    if analysis is None:
        raise RigError("rig has no audio: set frames")
    return int(math.ceil(round(analysis.duration * rig.fps, 6))) + rig.offset


def resolve_tempo(rig: Rig, analysis: Optional[Analysis], notes: Optional[dict]) -> Tempo:
    """rig.tempo.map (frames) > rig.tempo.bpm > notes bpm_map > analysis bpm map > 120. Grid offset: rig.tempo.grid_offset >
    notes grid_offset (when notes carry a bpm map) > analysis > 0."""
    t = rig.data.get("tempo") or {}
    notes_map = (notes or {}).get("bpm_map") or []
    if t.get("map"):
        segs = [((m["frame"] - rig.offset) / rig.fps, m["bpm"]) for m in t["map"]]
    elif t.get("bpm"):
        segs = [(0.0, t["bpm"])]
    elif notes_map:
        segs = [(m["time"], m["bpm"]) for m in notes_map]
    elif analysis is not None and analysis.bpm_map:
        segs = [(m["time"], m["bpm"]) for m in analysis.bpm_map]
    else:
        segs = [(0.0, 120.0)]
    if t.get("grid_offset") is not None:
        off = t["grid_offset"]
    elif notes_map:
        off = notes.get("grid_offset", 0.0)
    elif analysis is not None:
        off = analysis.grid_offset
    else:
        off = 0.0
    return Tempo(segs, off)


def make_ctx(rig, analysis, notes):
    ctx = Ctx(rig.fps, frame_count(rig, analysis), rig.offset, resolve_tempo(rig, analysis, notes), analysis, notes, rig.seed,
              rig.data.get("channel_weights"))
    ctx.rig_nodes = rig.nodes
    return ctx


def evaluate(rig: Rig, analysis: Optional[Analysis] = None, notes: Optional[dict] = None, ctx: Optional[Ctx] = None) -> dict:
    """Evaluate every node in dependency order. Returns {node_id: Signal} (Signal.v is float32 [channels, frames])."""
    if notes is None:
        notes = load_notes(rig.notes_path)
    ctx = ctx or make_ctx(rig, analysis, notes)
    out = {}
    for n in order_nodes(rig.nodes):
        spec = SPECS.get(n.get("op"))
        if spec is None:
            raise OpError(n.get("id"), f"unknown op {n.get('op')!r}")
        ins = [out[i] for i in input_ids(n)] if spec.inputs != "ref" else []
        try:
            s = spec.fn(n, ins, ctx)
        except OpError:
            raise
        except (ValueError, KeyError, TypeError) as e:
            raise OpError(n.get("id"), str(e)) from None
        if s.frames != ctx.frames:
            raise OpError(n.get("id"), f"produced {s.frames} frames, expected {ctx.frames}")
        out[n["id"]] = s
    return out


# ---------- layers
def ease(name, x):
    x = np.clip(x, 0, 1)
    if name == "in":
        return x * x
    if name == "out":
        return 1 - (1 - x) ** 2
    if name == "linear":
        return x
    return x * x * (3 - 2 * x)                               # in_out: zero velocity at both ends


def layer_curve(keys: list, frames: int) -> np.ndarray:
    """intensity per frame. A key {frame, value, ease, fade_frames} starts its fade AT `frame` and reaches `value` at
    frame + fade_frames (0 = a cut). Before the first key the first value holds."""
    v = np.ones(frames)
    if not keys:
        return v
    keys = sorted(keys, key=lambda k: k["frame"])
    f = np.arange(frames, dtype=np.float64)
    cur = float(keys[0]["value"])
    v[:] = cur
    for k in keys[1:]:
        f0, fade, tgt = float(k["frame"]), float(k.get("fade_frames") or 0), float(k["value"])
        x = (f - f0) / fade if fade > 0 else (f >= f0).astype(np.float64)
        seg = cur + (tgt - cur) * ease(k.get("ease", "in_out"), x)
        v = np.where(f >= f0, seg, v)
        cur = tgt
    return v


# ---------- keys
def rdp(v: np.ndarray, tol: float, forced=None) -> np.ndarray:
    """indices of linear keys whose reconstruction (linear interpolation) stays within tol of v everywhere. The first
    and last frames and every forced frame are kept; RDP runs between consecutive kept frames (vertical error)."""
    n = len(v)
    keep = np.zeros(n, bool)
    if n == 0:
        return np.zeros(0, np.int64)
    keep[0] = keep[-1] = True
    if forced is not None:
        keep[np.asarray(forced, np.int64)] = True
    v = np.asarray(v, np.float64)
    idx = np.flatnonzero(keep)
    stack = list(zip(idx[:-1].tolist(), idx[1:].tolist()))
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        t = np.arange(1, b - a, dtype=np.float64) / (b - a)
        err = np.abs(v[a + 1:b] - (v[a] + (v[b] - v[a]) * t))
        k = int(np.argmax(err))
        if err[k] > tol:
            m = a + 1 + k
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return np.flatnonzero(keep)


def target_channels(target):
    if target.get("kind") == "mpc":
        return [None]
    return list(target.get("actors") or [])


def bake(rig: Rig, analysis: Optional[Analysis] = None, notes: Optional[dict] = None, out: Optional[str] = None,
         signals: Optional[dict] = None) -> dict:
    """Validate, evaluate and bake. Returns the bake.json structure; writes it to `out` if given. Raises RigError with
    every check error if the rig is invalid. Same inputs -> byte-identical file."""
    if notes is None:
        notes = load_notes(rig.notes_path)
    errors = [i for i in check(rig, notes) if i["level"] == "error"]
    if errors:
        raise RigError("; ".join(f"{e['where']}: {e['msg']}" for e in errors))
    if analysis is None and rig.audio_path:
        analysis = analyze(rig)
    t0 = time.perf_counter()
    ctx = make_ctx(rig, analysis, notes)
    sigs = signals if signals is not None else evaluate(rig, analysis, notes, ctx)
    layers = {L["id"]: layer_curve(L.get("intensity", []), ctx.frames) for L in rig.layers}
    bk = rig.data.get("bake") or {}
    tol_rel, dec = float(bk.get("key_tolerance", 0.002)), int(bk.get("decimals", 4))
    outs = []
    for o in rig.outputs:
        s = sigs[o["signal"]]
        lo, hi = float(o.get("min", 0.0)), float(o.get("max", 1.0))
        inten = layers[o["layer"]] if o.get("layer") else np.ones(ctx.frames)
        names = target_channels(o["target"])
        s = s.broadcast(len(names))
        tol = tol_rel * abs(hi - lo) if hi != lo else tol_rel
        chans = []
        for c, name in enumerate(names):
            vals = np.round(lo + (s.v[c].astype(np.float64) * inten) * (hi - lo), dec)
            keys = rdp(vals, tol, np.flatnonzero(s.hits[c]))
            ch = {} if name is None else {"actor": name}
            ch["values"] = vals.tolist()
            ch["keys"] = [[int(k), float(vals[k])] for k in keys]
            chans.append(ch)
        entry = {"id": o["id"]}
        if o.get("layer"):
            entry["layer"] = o["layer"]
        entry.update({"signal": o["signal"], "target": o["target"], "min": lo, "max": hi, "channels": chans})
        outs.append(entry)
    result = {"version": BAKE_VERSION, "fps": rig.data.get("fps", 30), "frames": ctx.frames, "audio": rig.data.get("audio"),
              "audio_offset_frames": rig.offset, "rig_hash": rig.hash, "notes_hash": sha256(normalise(notes)), "outputs": outs}
    log.info("baked %d outputs, %d frames in %.2f s", len(outs), ctx.frames, time.perf_counter() - t0)
    if out:
        write_bake(out, result)
    return result


def write_bake(path: str, result: dict) -> None:
    """compact JSON, '\\n' line endings on every OS (part of the byte-identical guarantee)"""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
        fh.write("\n")


def bake_from_objects(rig_dict: dict, notes_dict: Optional[dict], base_dir: str) -> dict:
    """Bake an unsaved rig + notes held in memory (preview server / MCP). Relative paths in rig_dict resolve from
    base_dir; the analysis cache there is reused. notes_dict None = read the rig's notes file. Nothing is written."""
    rig = Rig.from_dict(rig_dict, base_dir)
    notes = load_notes(rig.notes_path) if notes_dict is None else notes_dict
    return bake(rig, None, notes)


def bake_stats(result: dict) -> list:
    """[(output id, channels, min, max, mean, total keys, frames)] for printing"""
    rows = []
    for o in result["outputs"]:
        v = np.array([c["values"] for c in o["channels"]], np.float64)
        k = sum(len(c["keys"]) for c in o["channels"])
        rows.append((o["id"], len(o["channels"]), float(v.min()), float(v.max()), float(v.mean()), k, v.size))
    return rows
