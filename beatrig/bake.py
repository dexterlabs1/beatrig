"""Evaluate a rig and bake it: per-frame values per target channel plus RDP-simplified linear keys (bake.json)."""
import gc
import json
import logging
import math
import os
import time
from typing import Optional

import numpy as np

from .analysis import Analysis, open_analysis
from .check import check, order_nodes
from .notes import load_notes, normalise, section_frames
from .ops import Ctx, OpError, SPECS, input_ids
from .rig import Rig, RigError, sha256
from .signal import Signal
from .tempo import Tempo
from . import safety as safety_mod

log = logging.getLogger("beatrig")
BAKE_VERSION = 1
DEFAULT_KEY_TOLERANCE = 0.002                               # rig bake.key_tolerance: RDP error / |max - min|


# ---------- setup
def analysis_reqs(rig: Rig, stem=None) -> list:
    """every band/loudness power series the rig needs from the mix (stem None) or from one stem, so one audio pass
    computes them all"""
    reqs = []
    for n in rig.nodes:
        src_stem = n.get("stem")
        if n.get("op") == "band" and src_stem == stem:
            reqs.append(("band", n.get("lo"), n.get("hi"), n.get("weights")))
        elif n.get("op") == "loudness" and src_stem == stem:
            reqs.append(("loud", None, None, n.get("weights")))
        elif n.get("op") == "onsets" and n.get("in") is not None:
            ref = n["in"]
            src = ref if isinstance(ref, dict) else rig.node(ref) or {}
            if (src_stem or src.get("stem")) == stem and (src.get("lo") is not None or src.get("hi") is not None):
                reqs.append(("band", src.get("lo"), src.get("hi"), src.get("weights")))
    return reqs


def analyze(rig: Rig, use_cache: bool = True) -> Optional[Analysis]:
    """Build or load the analysis cache (analysis.json/.npz in rig.cache_dir, default next to the rig) and compute every
    band the rig uses. Returns None for a rig without audio. The cache is invalidated by the audio's size/mtime or the
    analysis settings.
    Stems (analysis.stems = {name: Analysis}, each cached in its own folder next to the stem wavs): with
    analysis.stems "auto" (the default) cached stems are always used, and a missing separation runs once when the
    [stems] extra and a CUDA GPU are there; "on" separates on any device; "off" reads only the stems that nodes name.
    analysis.stems_info says what was used, or why the mix stands in."""
    if not rig.audio_path:
        return None
    if not os.path.exists(rig.audio_path):
        hint = " (generate the demo track with: python examples/make_demo_audio.py)" if "demo_110bpm" in rig.audio_path else ""
        raise RigError(f"audio not found: {rig.audio_path}{hint}")
    a = open_analysis(rig.audio_path, rig.cache_dir, rig.data.get("analysis"), rig.data.get("channel_weights"), use_cache)
    a.stems, a.stems_info = open_stems(rig, a, use_cache)
    reqs = analysis_reqs(rig)
    if "drums" not in a.stems or any(n.get("op") == "drum" and n.get("source") == "mix" for n in rig.nodes):
        reqs += drum_reqs(rig)                                # drum parts read the mix
    a.ensure(reqs)
    a.save()
    return a


def drum_reqs(rig: Rig) -> list:
    """the drum-part bands, when the rig has a drum node"""
    from .drums import part_bands
    if not any(n.get("op") == "drum" for n in rig.nodes):
        return []
    return [("band", lo, hi, None) for lo, hi in part_bands().values()]


def _stem_mode(a, rig):
    return (rig.data.get("analysis") or {}).get("stems", "auto") if a is None else a.settings.get("stems", "auto")


def open_stems(rig: Rig, a: Optional[Analysis] = None, use_cache: bool = True):
    """({stem: Analysis}, info) for the stems the rig reads: the ones its nodes name, and with analysis.stems auto/on
    all four (drum parts and song structure read them). info = {"used", "why"} (+ the separation's model, device,
    seconds and folder when used)."""
    from . import neural
    from . import stems as S
    mode = _stem_mode(a, rig)
    names = set(S.used_stems(rig.nodes))
    paths = S.cached(rig.audio_path, digest=a.content_hash() if a is not None else None)
    why = ""
    if paths is None:
        if mode == "off":
            why = "analysis.stems is off"
        elif mode == "auto" and neural.disabled():
            why = "BEATRIG_EXTRAS=off"
        else:
            ok, why = S.available()
            if ok and mode == "auto" and not S.cuda_available():
                ok, why = False, "no CUDA GPU (beatrig stems <rig> separates on the CPU, slowly; or set analysis.stems to \"on\")"
            if ok:
                paths = S.separate(rig.audio_path, log=log.info)
        if paths is None:
            if names:
                raise RigError(f"nodes read stems {sorted(names)} but they are not separated yet ({why}): "
                               f"run 'beatrig stems {rig.path or '<rig>'}'")
            return {}, {"used": False, "why": why}
    unknown = sorted(n for n in names if n not in paths)
    if unknown:
        raise RigError(f"unknown stem {unknown[0]!r} (one of {S.STEMS})")
    use = sorted(names if mode == "off" else names | set(S.STEMS))
    settings = dict(rig.data.get("analysis") or {}, beats="none")
    out = {}
    for name in use:
        sa = open_analysis(paths[name], S.stem_cache_dir(paths[name]), settings, None, use_cache)
        reqs = analysis_reqs(rig, name)
        if name == "drums":
            from .drums import part_bands
            reqs += [("band", lo, hi, None) for lo, hi in part_bands().values()]
        if mode != "off":
            reqs.append(("loud", None, None, None))           # stem levels for the song structure
        sa.ensure(reqs)
        sa.save()
        out[name] = sa
    if mode == "off":
        return out, {"used": False, "why": "analysis.stems is off" + (f" (nodes read {', '.join(use)})" if use else "")}
    meta = S.info(paths)
    return out, {"used": True, "why": "", "dir_name": os.path.basename(os.path.dirname(paths["drums"])),
                 **{k: meta[k] for k in ("model", "device", "seconds") if k in meta}}


def frame_count(rig: Rig, analysis: Optional[Analysis]) -> int:
    """frames baked = rig.frames, else from bake.start_frame (default 0) to the end of the audio window (audio_in_s ..
    audio_out_s or the end of the file, which ends at sequence frame offset + its length)"""
    if rig.data.get("frames") is not None:
        return int(rig.data["frames"])
    if analysis is None:
        raise RigError("rig has no audio: set frames")
    end = analysis.duration if rig.audio_out_s is None else min(rig.audio_out_s, analysis.duration)
    span = end - rig.audio_in_s
    if span <= 0:
        raise RigError(f"audio window {rig.audio_in_s}..{end} s is empty (the file is {analysis.duration:.3f} s)")
    end = int(math.ceil(round(span * rig.fps, 6))) + rig.offset
    if end <= rig.start_frame:
        raise RigError(f"bake.start_frame {rig.start_frame} is at or past the end of the audio window (frame {end})")
    return end - rig.start_frame


def resolve_tempo(rig: Rig, analysis: Optional[Analysis], notes: Optional[dict]) -> Tempo:
    """rig.tempo.map (frames) > rig.tempo.bpm > notes bpm_map > analysis bpm map > 120. Grid offset: rig.tempo.grid_offset >
    notes grid_offset (when notes carry a bpm map) > analysis > 0."""
    t = rig.data.get("tempo") or {}
    notes_map = (notes or {}).get("bpm_map") or []
    if t.get("map"):
        segs = [((m["frame"] - rig.offset) / rig.fps + rig.audio_in_s, m["bpm"]) for m in t["map"]]
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
    sset, _ = safety_mod.settings(rig.data)
    ctx = Ctx(rig.fps, frame_count(rig, analysis), rig.offset - rig.start_frame, resolve_tempo(rig, analysis, notes), analysis,
              notes, rig.seed, rig.data.get("channel_weights"), t0=rig.audio_in_s, scene=rig.load_scene(), safety=sset,
              stems=getattr(analysis, "stems", None) or {}, start=rig.start_frame)
    ctx.rig_nodes = rig.resolved_nodes()
    return ctx


def evaluate(rig: Rig, analysis: Optional[Analysis] = None, notes: Optional[dict] = None, ctx: Optional[Ctx] = None) -> dict:
    """Evaluate every node in dependency order. Returns {node_id: Signal} (Signal.v is float32 [channels, frames]; frame
    i is sequence frame bake.start_frame + i)."""
    if notes is None:
        notes = load_notes(rig.notes_path)
    ctx = ctx or make_ctx(rig, analysis, notes)
    out = {}
    for n in order_nodes(rig.resolved_nodes()):
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


def resolve_layer_keys(keys: list, sections: dict) -> list:
    """layer keys that name a section ({"section": "breakdown", ...}) -> frame keys (the fade starts at the first
    frame at or after the section's time). sections: {name: frame} from notes.section_frames."""
    out = []
    for k in keys or []:
        if k.get("section") is not None and k.get("frame") is None:
            if k["section"] not in sections:
                raise RigError(f"layer key names section {k['section']!r}, which is not in notes.json sections")
            k = dict(k, frame=sections[k["section"]])
        out.append(k)
    return out


def layer_curve(keys: list, frames: int, start: int = 0) -> np.ndarray:
    """intensity per frame for sequence frames start .. start + frames - 1. A key {frame, value, ease, fade_frames}
    (sequence frames) starts its fade AT `frame` and reaches `value` at frame + fade_frames (0 = a cut). Before the
    first key the first value holds."""
    v = np.ones(frames)
    if not keys:
        return v
    keys = sorted(keys, key=lambda k: k["frame"])
    f = np.arange(start, start + frames, dtype=np.float64)
    cur = float(keys[0]["value"])
    v[:] = cur
    rest = keys[1:]
    starts = [int(np.searchsorted(f, float(k["frame"]), side="left")) for k in rest] + [frames]
    for j, k in enumerate(rest):                             # a key rules from its frame until the next key's frame
        f0, fade, tgt = float(k["frame"]), float(k.get("fade_frames") or 0), float(k["value"])
        s0, s1 = starts[j], starts[j + 1]
        if s1 > s0:
            fs = f[s0:s1]
            x = (fs - f0) / fade if fade > 0 else (fs >= f0).astype(np.float64)
            v[s0:s1] = cur + (tgt - cur) * ease(k.get("ease", "in_out"), x)
        cur = tgt
    return v


# ---------- keys
def rdp(v: np.ndarray, tol: float, forced=None) -> np.ndarray:
    """indices of linear keys whose reconstruction (linear interpolation) stays within tol of v everywhere. The first
    and last frames and every forced frame are kept; RDP runs between consecutive kept frames (vertical error).
    v may be [N] or [N, K] (a colour channel): the error is then the largest component error.
    Every open span is split in the same numpy pass (level by level), so the cost is a few array operations per level
    instead of one Python iteration per key. A span's split point depends only on that span (the largest error, the
    first one on ties), so this keeps exactly the frames a one-span-at-a-time RDP keeps."""
    n = len(v)
    keep = np.zeros(n, bool)
    if n == 0:
        return np.zeros(0, np.int64)
    keep[0] = keep[-1] = True
    if forced is not None:
        keep[np.asarray(forced, np.int64)] = True
    v = np.asarray(v, np.float64)
    cols = [v] if v.ndim == 1 else [np.ascontiguousarray(v[:, k]) for k in range(v.shape[1])]
    idx = np.flatnonzero(keep)
    a, b = idx[:-1], idx[1:]
    open_ = b - a >= 2
    a, b = a[open_], b[open_]
    while len(a):
        span = b - a                                         # n per span
        inner = span - 1                                     # interior frames per span
        first = np.cumsum(inner) - inner                     # where each span's interior starts in the flat arrays
        seg = np.repeat(np.arange(len(a)), inner)            # span of each interior frame
        i = np.arange(len(seg)) - first[seg] + 1             # 1 .. n-1 inside each span
        t = i / span[seg]                                    # float64 i / n, as np.arange(1, n) / n
        j = a[seg] + i
        err = None
        for col in cols:                                     # colour: the largest component error, one column at a time
            va = col[a]
            e = np.abs(col[j] - (va[seg] + (col[b] - va)[seg] * t))
            err = e if err is None else np.maximum(err, e)
        top = np.maximum.reduceat(err, first)
        split = top > tol
        if not split.any():
            break
        at = np.flatnonzero((err == top[seg]) & split[seg])  # the largest error(s) of each span that splits
        s_at = seg[at]
        at = at[np.concatenate(([True], s_at[1:] != s_at[:-1]))]   # the first one per span, in span order
        m = a[seg[at]] + i[at]
        keep[m] = True
        a2 = np.concatenate((a[split], m))
        b2 = np.concatenate((m, b[split]))
        open_ = b2 - a2 >= 2
        a, b = a2[open_], b2[open_]
    return np.flatnonzero(keep)


def target_channels(target):
    if target.get("kind") == "mpc":
        return [None]
    return list(target.get("actors") or [])


def bake(rig: Rig, analysis: Optional[Analysis] = None, notes: Optional[dict] = None, out: Optional[str] = None,
         signals: Optional[dict] = None, strict: bool = True) -> dict:
    """Validate, evaluate and bake. Returns the bake.json structure; writes it to `out` if given. Raises RigError with
    every check error if the rig is invalid. Same inputs -> byte-identical file.
    Flash safety runs on the baked values: with a rig "safety" block a violation is an error (RigError unless
    strict=False); without one it is reported in result["safety"] only.
    rig.bake.start_frame S (default 0): only sequence frames S .. S + frames - 1 are evaluated and baked; values and key
    frames index that range, and the result carries "start_frame": S when S > 0."""
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
    secs = section_frames(notes, rig.fps, rig.offset, rig.audio_in_s)
    layers = {L["id"]: layer_curve(resolve_layer_keys(L.get("intensity", []), secs), ctx.frames, ctx.start) for L in rig.layers}
    bk = rig.data.get("bake") or {}
    tol_rel, dec = float(bk.get("key_tolerance", DEFAULT_KEY_TOLERANCE)), int(bk.get("decimals", 4))
    hashes = {"rig_hash": rig.hash, "notes_hash": sha256(normalise(notes))}
    gc_on = gc.isenabled()
    gc.disable()        # millions of small value/key lists: the cyclic GC would re-scan them over and over (they hold no cycles)
    try:
        outs, arrays = _bake_outputs(rig, sigs, layers, ctx, tol_rel, dec)
        result = {"version": BAKE_VERSION, "fps": rig.data.get("fps", 30), "frames": ctx.frames, "audio": rig.data.get("audio"),
                  "audio_offset_frames": rig.offset}
        if ctx.start:                                        # values[0] / key frame 0 = this sequence frame (0: left out)
            result["start_frame"] = ctx.start
        if tol_rel != DEFAULT_KEY_TOLERANCE:                 # the applier re-simplifies blended keys with it (default: left out)
            result["key_tolerance"] = tol_rel
        if rig.data.get("audio_in_s") is not None or rig.data.get("audio_out_s") is not None:
            result["audio_in_s"] = rig.audio_in_s
            result["audio_out_s"] = rig.audio_out_s
        result.update(hashes)
        result["outputs"] = outs
        issues, summary = safety_mod.check_bake(rig.data, result, arrays)
        result["safety"] = summary
        errs = [i for i in issues if i["level"] == "error"]
        if errs and strict:
            raise RigError("; ".join(f"{e['where']}: {e['msg']}" for e in errs))
        log.info("baked %d outputs, %d frames in %.2f s", len(outs), ctx.frames, time.perf_counter() - t0)
        if out:
            write_bake(out, result)
        return result
    finally:
        if gc_on:
            gc.enable()


def _bake_outputs(rig, sigs, layers, ctx, tol_rel, dec):
    """(bake.json outputs, {output id: per-channel value arrays}) for every rig output. An output with blend "multiply"
    or "add" bakes the same way (its values are gains or offsets on the target's own animation, applied in the editor)
    and carries "blend" in bake.json; "replace" (the default) is left implicit."""
    outs, arrays = [], {}
    for o in rig.outputs:
        s = sigs[o["signal"]]
        target = rig.resolved_target(o["target"])
        inten = layers[o["layer"]] if o.get("layer") else np.ones(ctx.frames)
        names = target_channels(target)
        s = s.broadcast(len(names))
        entry = {"id": o["id"]}
        if o.get("layer"):
            entry["layer"] = o["layer"]
        blend = o.get("blend") or "replace"
        if s.kind == "color":
            chans, arrays[o["id"]] = color_channels(s, names, inten, tol_rel, dec, with_arrays=True)
            entry.update({"signal": o["signal"], "value_type": "color", "target": dict(target, value_type="color"),
                          "channels": chans})
            if blend != "replace":
                entry["blend"] = blend
            outs.append(entry)
            continue
        lo, hi = float(o.get("min", 0.0)), float(o.get("max", 1.0))
        tol = tol_rel * abs(hi - lo) if hi != lo else tol_rel
        chans, arrays[o["id"]] = [], []
        for c, name in enumerate(names):
            vals = np.round(lo + (s.v[c].astype(np.float64) * inten) * (hi - lo), dec)
            keys = rdp(vals, tol, np.flatnonzero(s.hits[c]))
            ch = {} if name is None else {"actor": name}
            ch["values"] = vals.tolist()
            ch["keys"] = [[f, x] for f, x in zip(keys.tolist(), vals[keys].tolist())]
            chans.append(ch)
            arrays[o["id"]].append(vals)
        entry.update({"signal": o["signal"], "target": target, "min": lo, "max": hi, "channels": chans})
        if blend != "replace":                               # "replace" stays implicit: v0.2 rigs bake byte-identically
            entry["blend"] = blend
        outs.append(entry)
    return outs, arrays


def color_channels(s: Signal, names: list, inten: np.ndarray, tol_rel: float, dec: int, with_arrays: bool = False):
    """colour output channels: values [[r, g, b, a], ...] (RGB scaled by the layer intensity, alpha kept) and keys
    [[frame, r, g, b, a], ...] from a 4-vector RDP whose tolerance is key_tolerance x the channel's largest component
    range. with_arrays: also return the per-channel [N, 4] value arrays."""
    chans, arrs = [], []
    for c, name in enumerate(names):
        v = s.v[c].astype(np.float64).copy()
        v[:, :3] *= inten[:, None]
        vals = np.round(v, dec)
        rng = float((vals.max(axis=0) - vals.min(axis=0)).max()) if len(vals) else 0.0
        keys = rdp(vals, tol_rel * rng if rng > 0 else tol_rel, np.flatnonzero(s.hits[c]))
        ch = {} if name is None else {"actor": name}
        ch["values"] = vals.tolist()
        ch["keys"] = [[f] + row for f, row in zip(keys.tolist(), vals[keys].tolist())]
        chans.append(ch)
        arrs.append(vals)
    return (chans, arrs) if with_arrays else chans


def write_bake(path: str, result: dict) -> None:
    """compact JSON, '\\n' line endings on every OS (part of the byte-identical guarantee)"""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
        fh.write("\n")


def bake_from_objects(rig_dict: dict, notes_dict: Optional[dict], base_dir: str, strict: bool = True) -> dict:
    """Bake an unsaved rig + notes held in memory (preview server / MCP). Relative paths in rig_dict resolve from
    base_dir; the analysis cache there is reused. notes_dict None = read the rig's notes file. Nothing is written."""
    rig = Rig.from_dict(rig_dict, base_dir)
    notes = load_notes(rig.notes_path) if notes_dict is None else notes_dict
    return bake(rig, None, notes, strict=strict)


def bake_stats(result: dict) -> list:
    """[(output id, channels, min, max, mean, total keys, frames)] for printing (colour: over all components)"""
    rows = []
    for o in result["outputs"]:
        v = np.array([c["values"] for c in o["channels"]], np.float64)
        k = sum(len(c["keys"]) for c in o["channels"])
        rows.append((o["id"], len(o["channels"]), float(v.min()), float(v.max()), float(v.mean()), k, v.size))
    return rows
