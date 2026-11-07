"""beatrig MCP server (stdio): lets an agent drive the whole loop, from analysing a song through editing the rig and its
notes to baking keys and applying them to a running Unreal editor.

    beatrig-mcp                       # stdio transport; register with:  claude mcp add beatrig -- beatrig-mcp

Design rules for the tools:
  - Results are small, compact JSON. Big arrays (per-frame values) stay in files; tools return stats and paths.
  - Errors say what to do next.
  - Every mutating tool validates the edited rig with beatrig.check before writing (nothing invalid reaches disk),
    keeps a .bak of the previous file, and returns a diff summary.
  - Things are addressed by id (node id, output id, track name, hit time), never by list index.
  - After a change, the preview page (http://127.0.0.1:4700) is told via POST /api/notify, best effort.
  - stdout is the protocol: every engine call runs with stdout redirected to stderr.

The `mcp` package is optional (`pip install beatrig[mcp]`). This module imports without it; only main() needs it, so
the tool functions can also be called directly from Python and tests.
"""
import contextlib
import functools
import json
import os
import shutil
import struct
import subprocess
import sys
import urllib.error
import urllib.request
import zlib

import numpy as np

PREVIEW_URL = os.environ.get("BEATRIG_PREVIEW_URL", "http://127.0.0.1:4700")
UE_HOST = os.environ.get("BEATRIG_UE_HOST", "127.0.0.1:30010")

# Candidate frequency ranges for analyze/describe_audio. These are ranges where a kind of sound usually sits, not
# detected instruments: the score says how clearly a range carries separate hits in THIS song.
BAND_LADDER = [
    ("sub", 30, 60, "sub bass, 808 tails"),
    ("kick", 40, 120, "kick drum body"),
    ("bass", 80, 250, "bass line, toms"),
    ("lowmid", 250, 800, "snare body, chords, vocals"),
    ("mid", 800, 2500, "snare crack, claps, leads"),
    ("presence", 2500, 6000, "claps, rim, vocal consonants"),
    ("air", 6000, 14000, "hi-hats, cymbals, shakers"),
]


class ToolFailure(Exception):
    """an expected failure with an actionable message (converted to an MCP tool error)"""


def _j(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=_jdefault)


def _jdefault(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _r(x, n=3):
    return None if x is None else round(float(x), n)


def tool(fn):
    """engine output never reaches stdout; known failures become clean messages"""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        with contextlib.redirect_stdout(sys.stderr):
            try:
                return fn(*a, **kw)
            except ToolFailure:
                raise
            except Exception as e:                       # engine errors (RigError, OpError, ...) carry good messages
                raise ToolFailure(f"{type(e).__name__}: {e}") from e
    wrapper.is_tool = True
    return wrapper


# ------------------------------------------------------------------------------------------------ files and engine
def _eng():
    import beatrig
    return beatrig


def _rig_path(rig_path):
    p = os.path.abspath(os.path.expanduser(str(rig_path)))
    if os.path.isdir(p):
        p = os.path.join(p, "rig.json")
    if not os.path.exists(p):
        raise ToolFailure(f"rig not found: {p} (pass the path to a rig.json)")
    return p


def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as e:
            raise ToolFailure(f"{path}: invalid JSON ({e})") from None


def _write_json(path, obj):
    """.bak of the old file, then an atomic replace"""
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=2, ensure_ascii=False))
        fh.write("\n")
    os.replace(tmp, path)


def _load(rig_path):
    """-> (path, rig dict, Rig)"""
    b = _eng()
    p = _rig_path(rig_path)
    data = _read_json(p)
    if not isinstance(data, dict):
        raise ToolFailure(f"{p}: a rig must be a JSON object")
    return p, data, b.Rig.from_dict(data, os.path.dirname(p))


def _notes(rig):
    return _eng().load_notes(rig.notes_path)


def _issues(rig_dict, base_dir, notes=None):
    b = _eng()
    r = b.Rig.from_dict(rig_dict, base_dir)
    return b.check(r, notes if notes is not None else b.load_notes(r.notes_path))


def _commit_rig(path, old, new, action, diff):
    """validate, write, notify. Refuses (and writes nothing) on any check error."""
    base = os.path.dirname(path)
    issues = _issues(new, base)
    errors = [i for i in issues if i["level"] == "error"]
    if errors:
        raise ToolFailure("rejected, rig.json unchanged: " + "; ".join(f"{e['where']}: {e['msg']}" for e in errors[:8]))
    if new == old:
        return {"ok": True, "changed": False, "diff": {}, "warnings": _fmt_issues(issues)}
    _write_json(path, new)
    preview = _notify(action, f"{os.path.basename(base)}: {_short(diff)}")
    return {"ok": True, "changed": True, "diff": diff, "warnings": _fmt_issues(issues), "preview": preview}


def _fmt_issues(issues):
    return [f"{i['where']}: {i['msg']}" for i in issues if i["level"] != "error"][:12]


def _short(diff):
    s = _j(diff)
    return s if len(s) <= 240 else s[:237] + "..."


def _dict_diff(old, new, prefix=""):
    """{"key": [old, new]} for changed keys (nested dicts flattened with dots); missing = null"""
    out = {}
    for k in sorted(set(old) | set(new), key=str):
        a, b = old.get(k), new.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            out.update(_dict_diff(a, b, f"{prefix}{k}."))
        elif a != b:
            out[f"{prefix}{k}"] = [a, b]
    return out


def _merge(base, fields):
    """shallow merge where a null value removes the key; nested dicts (e.g. target) merge one level down"""
    out = dict(base)
    for k, v in fields.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _notify(action, detail):
    """tell the preview page what the agent did; returns a one-word status, never raises"""
    body = json.dumps({"source": "mcp", "action": action[:40], "detail": detail[:300]}).encode()
    req = urllib.request.Request(PREVIEW_URL + "/api/notify", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            return "notified" if resp.status == 200 else f"http {resp.status}"
    except urllib.error.HTTPError as e:
        return f"http {e.code}"
    except Exception:
        return "not running"


def _node_ids(rig_dict):
    return [n.get("id") for n in rig_dict.get("nodes", [])]


def _find(items, key, value, what):
    for i, it in enumerate(items):
        if it.get(key) == value:
            return i
    known = ", ".join(str(it.get(key)) for it in items) or "none"
    raise ToolFailure(f"no {what} {value!r} (known: {known})")


def _ref_ids(x):
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        return [i for i in x if isinstance(i, str)]
    return []


# ------------------------------------------------------------------------------------------------ analysis tools
def _analysis(rig):
    a = _eng().analyze(rig)
    if a is None:
        raise ToolFailure("this rig has no audio; set \"audio\" in rig.json to a wav/flac path")
    return a


def _band_scores(a, ladder, beats):
    """per band: hit count, contrast of hits over the background, beat alignment, relative level"""
    import scipy.signal as sg
    a.ensure([("band", lo, hi, None) for _, lo, hi, _ in ladder])
    beat_t = np.array([b["time"] for b in beats]) if beats else np.zeros(0)
    rows, levels = [], []
    for name, lo, hi, typical in ladder:
        f = a.onset_fn(lo, hi).astype(np.float64)
        env = a.envelope("band", lo, hi, 10, 150)
        levels.append(float(np.sqrt(np.mean(env.astype(np.float64) ** 2)) + 1e-12))
        pk, _ = sg.find_peaks(f, distance=16)             # 80 ms at the 5 ms onset hop
        pk = pk[f[pk] > 0]
        strong = np.zeros(0, int)
        contrast = 0.0
        if len(pk):
            rel = f[pk] / (np.percentile(f[pk], 95) + 1e-12)
            strong = pk[rel >= 0.3]
            if len(strong):
                contrast = 20 * np.log10(np.median(f[strong]) / (np.percentile(f, 90) + 1e-12) + 1e-12)
        times = (strong + 1) * 0.005
        on_beat = None
        if len(times) and len(beat_t):
            d = np.min(np.abs(times[:, None] - beat_t[None, :]), axis=1)
            on_beat = float(np.mean(d <= 0.06))
        rows.append({"band": name, "lo": lo, "hi": hi, "typical": typical, "hits": int(len(times)),
                     "hits_per_beat": _r(len(times) / len(beat_t), 2) if len(beat_t) else None,
                     "on_beat": _r(on_beat, 2), "contrast_db": _r(contrast, 1),
                     "first_hits_s": [_r(t, 3) for t in times[:4]]})
    top = max(levels)
    for row, lv in zip(rows, levels):
        row["level_db"] = _r(20 * np.log10(lv / top), 1)
    return rows


@tool
def analyze(rig_path: str) -> str:
    """Analyse the rig's audio (cached next to the rig). Returns duration, tempo, beat count, pulse confidence and
    suggested_bands: which frequency ranges carry clear, separate hits in this song (contrast_db = how far hits stand
    above the background; on_beat = share of hits within 60 ms of a beat; hits_per_beat ~1 = on the beat, ~2 = eighths).
    The bands are candidate ranges, not detected instruments. Use them in band/onsets nodes or notes_init(source="band:lo-hi")."""
    p, data, rig = _load(rig_path)
    a = _analysis(rig)
    m = a.meta
    bands = _band_scores(a, BAND_LADDER, m.get("beats"))
    good = sorted((b for b in bands if b["hits"] >= 4 and (b["contrast_db"] or 0) >= 12),
                  key=lambda b: -b["contrast_db"])
    return _j({
        "audio": os.path.basename(a.audio), "duration_s": _r(m["duration"]), "sr": a.sr, "channels": a.channels,
        "bpm": m["bpm_map"][0]["bpm"] if m.get("bpm_map") else None, "bpm_map": m.get("bpm_map"),
        "grid_offset_s": _r(m.get("grid_offset")), "beats": len(m.get("beats") or []),
        "pulse": m.get("pulse"), "low_confidence_tempo": bool(m.get("fallback")),
        "passages": [{k: p_.get(k) for k in ("start", "end", "bpm", "confidence")} for p_ in m.get("passages") or []][:12],
        "cache": "hit" if a.loaded_from_cache else "built",
        "suggested_bands": [b["band"] for b in good][:3],
        "bands": bands,
        "next": "describe_audio for structure; notes_init(track, source='band:LO-HI') to make an editable hit lane",
    })


@tool
def describe_audio(rig_path: str, start_s: float = 0.0, end_s: float | None = None, window_s: float | None = None) -> str:
    """Describe the song so you can reason about its structure without hearing it: per time window (default one bar),
    the level of each band (0..1, dB-scaled over the whole song) and its hit count, plus candidate section boundaries
    where the energy changes sharply. Boundaries are energy changes, not labelled sections."""
    import scipy.signal as sg
    p, data, rig = _load(rig_path)
    a = _analysis(rig)
    m = a.meta
    dur = float(m["duration"])
    t0 = max(0.0, float(start_s))
    t1 = min(dur, float(end_s)) if end_s is not None else dur
    if t1 <= t0:
        raise ToolFailure(f"empty range {t0}-{t1} s (song is {dur:.2f} s)")
    bpm = m["bpm_map"][0]["bpm"] if m.get("bpm_map") else None
    bar = 4 * 60.0 / bpm if bpm else 2.0
    w = float(window_s) if window_s else bar
    while (t1 - t0) / w > 48:                             # keep the answer small: at most 48 rows
        w *= 2
    edges = np.arange(t0, t1, w)
    a.ensure([("band", lo, hi, None) for _, lo, hi, _ in BAND_LADDER])
    levels, hits = [], []
    for name, lo, hi, _ in BAND_LADDER:
        env = a.envelope("band", lo, hi, 10, 150).astype(np.float64)        # 1 ms grid
        db = 20 * np.log10(env + 1e-9)
        lo_db, hi_db = np.percentile(db, 10), np.percentile(db, 99.5)
        row = []
        for e in edges:
            seg = env[int(e * 1000):int(min(e + w, t1) * 1000)]
            v = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) if len(seg) else lo_db
            row.append(float(np.clip((v - lo_db) / max(hi_db - lo_db, 1e-6), 0, 1)))
        levels.append(row)
        f = a.onset_fn(lo, hi).astype(np.float64)
        pk, _ = sg.find_peaks(f, distance=16)
        pk = pk[f[pk] > 0]
        if len(pk):
            pk = pk[f[pk] / (np.percentile(f[pk], 95) + 1e-12) >= 0.3]
        ht = (pk + 1) * 0.005
        hits.append([int(np.sum((ht >= e) & (ht < e + w))) for e in edges])
    names = [b[0] for b in BAND_LADDER]
    L = np.array(levels)                                  # [bands, windows]
    rows = [{"t": [_r(e, 2), _r(min(e + w, t1), 2)], "level": [_r(x, 2) for x in L[:, i]],
             "hits": [h[i] for h in hits]} for i, e in enumerate(edges)]
    changes = []
    for i in range(1, L.shape[1]):
        d = L[:, i] - L[:, i - 1]
        if np.abs(d).sum() >= 0.9 or np.abs(d).max() >= 0.35:
            changes.append({"time_s": _r(edges[i], 2),
                            "rose": [names[k] for k in np.flatnonzero(d >= 0.2)],
                            "fell": [names[k] for k in np.flatnonzero(d <= -0.2)]})
    beats = [b["time"] for b in m.get("beats") or [] if t0 <= b["time"] < t1]
    return _j({"range_s": [_r(t0, 2), _r(t1, 2)], "window_s": _r(w, 3), "bpm": bpm, "beats_in_range": len(beats),
               "bands": names, "rows": rows, "candidate_boundaries": changes,
               "legend": "level per band 0..1 (dB over the whole song), hits = clear onsets in that band per window"})


# ------------------------------------------------------------------------------------------------ rig tools
@tool
def list_ops() -> str:
    """Every node op with its params (defaults; "REQUIRED" marks required ones), its input arity and what it outputs.
    Read this before add_node/set_node."""
    from . import ops
    out = {}
    for name, s in sorted(ops.SPECS.items()):
        out[name] = {"inputs": s.inputs, "kind": s.kind if isinstance(s.kind, str) else "same as input",
                     "params": {k: ("REQUIRED" if v is ops.REQ else v) for k, v in s.params.items()}}
        if s.one_of:
            out[name]["one_of"] = [list(g) if isinstance(g, (list, tuple)) else g for g in s.one_of]
    return _j({"ops": out, "common_keys": sorted(ops.COMMON),
               "inputs_legend": {"one": "in = node id", "many": "in = [node ids]", "ref": "in = band node id or {lo, hi}",
                                 "null": "source, no in"}})


@tool
def get_rig(rig_path: str) -> str:
    """The rig as JSON (nodes, layers, outputs, settings), plus the node graph, the notes tracks and any check issues."""
    p, data, rig = _load(rig_path)
    notes = _notes(rig)
    issues = _eng().check(rig, notes)
    graph = {n.get("id"): _ref_ids(n.get("in")) for n in data.get("nodes", [])}
    return _j({"path": p, "rig": data, "graph": graph,
               "notes_tracks": {k: len(v) for k, v in notes.get("tracks", {}).items()},
               "errors": [f"{i['where']}: {i['msg']}" for i in issues if i["level"] == "error"],
               "warnings": _fmt_issues(issues)})


@tool
def set_node(rig_path: str, node_id: str, params: dict) -> str:
    """Change params of one node by id (merge; a null value removes the param and restores its default). The edited rig is
    validated before writing; an invalid edit is rejected and nothing changes. Returns the diff. Rename = add_node + remove_node."""
    p, data, _ = _load(rig_path)
    nodes = data.get("nodes", [])
    i = _find(nodes, "id", node_id, "node")
    if "id" in params and params["id"] != node_id:
        raise ToolFailure("set_node can't rename; add a node with the new id, repoint references, then remove_node")
    new = json.loads(json.dumps(data))
    new["nodes"][i] = _merge(nodes[i], {k: v for k, v in params.items() if k != "id"})
    diff = {"node": node_id, "changed": _dict_diff(nodes[i], new["nodes"][i])}
    return _j(_commit_rig(p, data, new, "set_node", diff))


@tool
def add_node(rig_path: str, node: dict, after: str | None = None) -> str:
    """Add a node ({"id", "op", ...params}; see list_ops). Placed after node `after` (by id) or at the end. Validated
    before writing."""
    p, data, _ = _load(rig_path)
    if not isinstance(node, dict) or not node.get("id") or not node.get("op"):
        raise ToolFailure('node needs at least {"id": "...", "op": "..."}')
    if node["id"] in _node_ids(data):
        raise ToolFailure(f"node id {node['id']!r} already exists; use set_node to change it")
    new = json.loads(json.dumps(data))
    nodes = new.setdefault("nodes", [])
    at = _find(nodes, "id", after, "node") + 1 if after else len(nodes)
    nodes.insert(at, node)
    return _j(_commit_rig(p, data, new, "add_node", {"added": node, "position_after": after}))


@tool
def remove_node(rig_path: str, node_id: str) -> str:
    """Remove a node by id. Refused while other nodes or outputs still use it (the error names them)."""
    p, data, _ = _load(rig_path)
    nodes = data.get("nodes", [])
    i = _find(nodes, "id", node_id, "node")
    users = [n["id"] for n in nodes if node_id in _ref_ids(n.get("in"))]
    users += [f"output {o.get('id')}" for o in data.get("outputs", []) if o.get("signal") == node_id]
    if users:
        raise ToolFailure(f"{node_id!r} is used by {', '.join(users)}; repoint or remove those first")
    new = json.loads(json.dumps(data))
    removed = new["nodes"].pop(i)
    return _j(_commit_rig(p, data, new, "remove_node", {"removed": removed}))


@tool
def set_output(rig_path: str, output_id: str, fields: dict, create: bool = False) -> str:
    """Change an output by id (signal, min, max, layer, target...; merge, null removes, target merges one level down).
    create=true adds a new output (needs signal and target). Validated before writing. Returns the diff."""
    p, data, _ = _load(rig_path)
    outs = data.get("outputs", [])
    new = json.loads(json.dumps(data))
    if any(o.get("id") == output_id for o in outs):
        i = _find(outs, "id", output_id, "output")
        new["outputs"][i] = _merge(outs[i], {k: v for k, v in fields.items() if k != "id"})
        diff = {"output": output_id, "changed": _dict_diff(outs[i], new["outputs"][i])}
    elif create:
        o = dict({"id": output_id}, **{k: v for k, v in fields.items() if v is not None and k != "id"})
        if "signal" not in o or "target" not in o:
            raise ToolFailure("a new output needs signal (node id) and target ({kind, actors|asset, param|property})")
        new.setdefault("outputs", []).append(o)
        diff = {"added_output": o}
    else:
        _find(outs, "id", output_id, "output")
    return _j(_commit_rig(p, data, new, "set_output", diff))


@tool
def remove_output(rig_path: str, output_id: str) -> str:
    """Remove an output by id."""
    p, data, _ = _load(rig_path)
    i = _find(data.get("outputs", []), "id", output_id, "output")
    new = json.loads(json.dumps(data))
    removed = new["outputs"].pop(i)
    return _j(_commit_rig(p, data, new, "remove_output", {"removed_output": removed}))


@tool
def set_layer(rig_path: str, layer_id: str, intensity: list) -> str:
    """Set (or create) a layer's intensity keys: [{"frame", "value", "ease": in_out|in|out|linear, "fade_frames"}].
    Outputs with "layer": layer_id are scaled by it, so two layers with opposite keys switch looks between sections."""
    p, data, _ = _load(rig_path)
    new = json.loads(json.dumps(data))
    layers = new.setdefault("layers", [])
    i = next((k for k, L in enumerate(layers) if L.get("id") == layer_id), None)
    before = None if i is None else layers[i].get("intensity")
    if i is None:
        layers.append({"id": layer_id, "intensity": intensity})
    else:
        layers[i]["intensity"] = intensity
    diff = {"layer": layer_id, "intensity": [before, intensity]}
    return _j(_commit_rig(p, data, new, "set_layer", diff))


@tool
def check(rig_path: str) -> str:
    """Validate the rig: unknown ops/params, missing references, cycles, channel counts vs actors, and the timing rules
    (nothing ahead of the audio, only trigger envelopes may move instantly). No audio needed."""
    p, data, rig = _load(rig_path)
    issues = _eng().check(rig, _notes(rig))
    errors = [f"{i['where']}: {i['msg']}" for i in issues if i["level"] == "error"]
    return _j({"ok": not errors, "errors": errors, "warnings": _fmt_issues(issues)})


# ------------------------------------------------------------------------------------------------ notes tools
def _save_notes(rig, notes):
    if os.path.exists(rig.notes_path):
        shutil.copy2(rig.notes_path, rig.notes_path + ".bak")
    return _eng().save_notes(rig.notes_path, notes)


def _track_summary(hits):
    if not hits:
        return {"hits": 0}
    t = [h["time"] for h in hits]
    return {"hits": len(hits), "first_s": _r(min(t)), "last_s": _r(max(t)),
            "mean_vel": _r(sum(h.get("vel", 1.0) for h in hits) / len(hits), 2)}


@tool
def get_notes(rig_path: str, track: str | None = None, start_s: float = 0.0, end_s: float | None = None,
              limit: int = 64) -> str:
    """Read notes.json: every track's summary, and the hits of `track` in [start_s, end_s) (up to `limit`).
    Hits are addressed by time in seconds of audio."""
    p, data, rig = _load(rig_path)
    n = _notes(rig)
    out = {"bpm_map": n.get("bpm_map"), "grid_offset": n.get("grid_offset"),
           "tracks": {k: _track_summary(v) for k, v in n.get("tracks", {}).items()}}
    if track is not None:
        if track not in n["tracks"]:
            raise ToolFailure(f"no track {track!r} (tracks: {', '.join(n['tracks']) or 'none'})")
        hi = float("inf") if end_s is None else float(end_s)
        sel = [h for h in n["tracks"][track] if float(start_s) <= h["time"] < hi]
        out["hits"] = sel[:int(limit)]
        out["truncated"] = len(sel) > int(limit)
    return _j(out)


@tool
def notes_init(rig_path: str, track: str, source: str, threshold: float | None = None, replace: bool = False) -> str:
    """Fill a notes track from the audio. source: "band:LO-HI" (onsets in that Hz range, e.g. "band:40-120"),
    "onsets:<node id>", "beats" (the verified beat grid) or "beats:<node id>". Refuses to overwrite an existing track
    unless replace=true. Seeds the notes tempo map from the analysis if it has none."""
    from .bake import make_ctx
    from .ops import beat_times, detect_onsets
    p, data, rig = _load(rig_path)
    notes = _notes(rig)
    if track in notes["tracks"] and notes["tracks"][track] and not replace:
        raise ToolFailure(f"track {track!r} already has {len(notes['tracks'][track])} hits; pass replace=true to overwrite")
    a = _analysis(rig)
    ctx = make_ctx(rig, a, notes)
    kind, _, arg = source.partition(":")
    if kind == "band":
        try:
            lo, hi = (float(x) for x in arg.split("-"))
        except ValueError:
            raise ToolFailure('band source looks like "band:40-120"') from None
        node = {"id": f"_{track}", "op": "onsets", "in": {"lo": lo, "hi": hi}}
        a.ensure([("band", lo, hi, None)])
    elif kind == "onsets":
        node = rig.node(arg)
        if node is None or node.get("op") != "onsets":
            raise ToolFailure(f"onsets:<id> needs an onsets node id, got {arg!r}")
        node = dict(node)
    elif kind == "beats":
        node = rig.node(arg) if arg else {"id": "_beats", "op": "beats"}
        if node is None or node.get("op") != "beats":
            raise ToolFailure(f"beats:<id> needs a beats node id, got {arg!r}")
    else:
        raise ToolFailure('source must be "band:LO-HI", "onsets:<node>", "beats" or "beats:<node>"')
    if threshold is not None and node.get("op") == "onsets":
        node = dict(node, threshold=float(threshold))
    if node["op"] == "onsets":
        times, vels = detect_onsets(node, ctx, rig.nodes)
    else:
        times, vels = beat_times(node, ctx)
    if not notes["bpm_map"]:
        notes["bpm_map"], notes["grid_offset"] = a.bpm_map, a.grid_offset
    before = len(notes["tracks"].get(track, []))
    notes["tracks"][track] = [{"time": float(t), "vel": round(float(v), 3)} for t, v in zip(times, vels)]
    _save_notes(rig, notes)
    diff = {"track": track, "source": source, "hits": [before, len(times)]}
    preview = _notify("notes_init", f"{track}: {len(times)} hits from {source}")
    return _j({"ok": True, "diff": diff, "summary": _track_summary(notes["tracks"][track]), "preview": preview,
               "next": f"add a node {{\"op\": \"notes\", \"track\": \"{track}\"}} to use it"})


def _match(hits, t, tol):
    d = [abs(h["time"] - t) for h in hits]
    near = [i for i, x in enumerate(d) if x <= tol]
    if not near:
        closest = min(hits, key=lambda h: abs(h["time"] - t))["time"] if hits else None
        raise ToolFailure(f"no hit within {tol * 1000:.0f} ms of {t} s (closest: {closest})")
    if len(near) > 1:
        raise ToolFailure(f"{len(near)} hits within {tol * 1000:.0f} ms of {t} s; give a more exact time or a smaller tolerance_ms")
    return near[0]


@tool
def notes_edit(rig_path: str, track: str, edits: list, tolerance_ms: float = 10.0) -> str:
    """Edit hits in one track, all-or-nothing. Each edit is one of:
      {"action": "add", "time": s, "vel": 0..1, ["len_beats", "shape"]}
      {"action": "move", "time": s, "to": s}
      {"action": "delete", "time": s}
      {"action": "velocity", "time": s, "vel": 0..1}
    Existing hits are matched by time within tolerance_ms (a hit's id is its time). Creates the track on first add."""
    p, data, rig = _load(rig_path)
    notes = _notes(rig)
    hits = [dict(h) for h in notes["tracks"].get(track, [])]
    tol = float(tolerance_ms) / 1000.0
    log = []
    for n, e in enumerate(edits):
        act = e.get("action")
        try:
            t = float(e["time"])
        except (KeyError, TypeError, ValueError):
            raise ToolFailure(f"edit {n}: needs a numeric time (seconds)") from None
        if act == "add":
            vel = float(e.get("vel", 1.0))
            if not 0 < vel <= 1:
                raise ToolFailure(f"edit {n}: vel must be in (0, 1]")
            if t < 0:
                raise ToolFailure(f"edit {n}: time must be >= 0")
            h = {"time": t, "vel": vel}
            for k in ("len_beats", "shape"):
                if e.get(k) is not None:
                    h[k] = e[k]
            hits.append(h)
            log.append(["add", _r(t, 4), vel])
        elif act in ("move", "delete", "velocity"):
            i = _match(hits, t, tol)
            old = hits[i]["time"]
            if act == "move":
                to = float(e["to"])
                if to < 0:
                    raise ToolFailure(f"edit {n}: to must be >= 0")
                hits[i]["time"] = to
                log.append(["move", _r(old, 4), _r(to, 4)])
            elif act == "delete":
                hits.pop(i)
                log.append(["delete", _r(old, 4)])
            else:
                vel = float(e["vel"])
                if not 0 < vel <= 1:
                    raise ToolFailure(f"edit {n}: vel must be in (0, 1]")
                log.append(["velocity", _r(old, 4), hits[i].get("vel"), vel])
                hits[i]["vel"] = vel
        else:
            raise ToolFailure(f"edit {n}: action must be add, move, delete or velocity")
    notes["tracks"][track] = hits
    _save_notes(rig, notes)
    preview = _notify("notes_edit", f"{track}: {len(log)} edit(s)")
    return _j({"ok": True, "track": track, "applied": log, "summary": _track_summary(hits), "preview": preview})


@tool
def notes_snap(rig_path: str, track: str, grid: str = "1/16", strength: float = 1.0) -> str:
    """Quantise a track's hits to the beat grid ("1/4" = beats, "1/8", "1/16", or a number of beats). strength 0..1 moves
    part of the way. Reports how many moved and the largest move earlier: snapping earlier can put a visual ahead of
    its sound, so keep max_earlier_ms small (under ~20 ms) or lower strength."""
    from .notes import notes_tempo, parse_grid, snap
    from .tempo import Tempo
    p, data, rig = _load(rig_path)
    notes = _notes(rig)
    if track not in notes["tracks"]:
        raise ToolFailure(f"no track {track!r} (tracks: {', '.join(notes['tracks']) or 'none'})")
    tempo = notes_tempo(notes)
    if tempo is None:
        a = _analysis(rig)
        tempo = Tempo([(m["time"], m["bpm"]) for m in a.bpm_map], a.grid_offset) if a.bpm_map else Tempo([(0, 120)])
    hits = notes["tracks"][track]
    notes["tracks"][track], st = snap(hits, tempo, parse_grid(grid), float(strength))
    _save_notes(rig, notes)
    preview = _notify("notes_snap", f"{track}: snapped to {grid}, {st['moved']} moved")
    return _j({"ok": True, "track": track, "grid": grid, "strength": strength, "moved": st["moved"], "of": len(hits),
               "max_earlier_ms": st["max_earlier_ms"], "preview": preview})


# ------------------------------------------------------------------------------------------------ bake
def _bake_all(rig):
    """(analysis, signals, bake dict); one evaluation shared by bake and plot"""
    b = _eng()
    notes = _notes(rig)
    errors = [i for i in b.check(rig, notes) if i["level"] == "error"]
    if errors:
        raise ToolFailure("rig has errors: " + "; ".join(f"{e['where']}: {e['msg']}" for e in errors[:8]))
    a = b.analyze(rig)
    sigs = b.evaluate(rig, a, notes)
    return a, sigs, b.bake(rig, a, notes, signals=sigs)


@tool
def bake(rig_path: str, out: str | None = None) -> str:
    """Evaluate the rig and write bake.json (default next to the rig). Returns per-output stats (channels, min, max,
    mean, key count) and the bake path; the per-frame values stay in the file."""
    import time
    p, data, rig = _load(rig_path)
    t0 = time.perf_counter()
    a, sigs, res = _bake_all(rig)
    path = os.path.abspath(out) if out else rig.resolve("bake.json")
    _eng().write_bake(path, res)
    outs = []
    for o in res["outputs"]:
        v = np.array([c["values"] for c in o["channels"]], np.float64)
        outs.append({"id": o["id"], "target": o["target"].get("kind"), "channels": len(o["channels"]),
                     "min": _r(v.min(), 4), "max": _r(v.max(), 4), "mean": _r(v.mean(), 4),
                     "keys": sum(len(c["keys"]) for c in o["channels"]),
                     "hits": int(sigs[o["signal"]].hits.sum())})
    preview = _notify("bake", f"{os.path.basename(path)}: {len(outs)} outputs, {res['frames']} frames")
    return _j({"ok": True, "bake_path": path, "fps": res["fps"], "frames": res["frames"],
               "seconds": _r(res["frames"] / float(res["fps"]), 2), "rig_hash": res["rig_hash"][:12],
               "outputs": outs, "elapsed_s": _r(time.perf_counter() - t0, 2), "preview": preview})


# ------------------------------------------------------------------------------------------------ plot (PNG, no deps)
# 5x7 bitmap font: each glyph is 7 rows of 5 bits (bit 4 = leftmost), hex encoded. Lowercase renders as uppercase.
_FONT = {
    "0": "0E11131519110E", "1": "040C040404040E", "2": "0E11010204081F", "3": "1F02040201110E",
    "4": "02060A121F0202", "5": "1F101E0101110E", "6": "0608101E11110E", "7": "1F010204080808",
    "8": "0E11110E11110E", "9": "0E11110F01020C", "A": "0E11111F111111", "B": "1E11111E11111E",
    "C": "0E11101010110E", "D": "1C12111111121C", "E": "1F10101E10101F", "F": "1F10101E101010",
    "G": "0E11101711110F", "H": "1111111F111111", "I": "0E04040404040E",
    "J": "0702020202120C", "K": "11121418141211", "L": "1010101010101F", "M": "111B1515111111",
    "N": "11111915131111", "O": "0E11111111110E", "P": "1E11111E101010", "Q": "0E11111115120D",
    "R": "1E11111E141211", "S": "0F10100E01011E", "T": "1F040404040404", "U": "1111111111110E",
    "V": "11111111110A04", "W": "1111111515150A", "X": "11110A040A1111",
    "Y": "1111110A040404", "Z": "1F01020408101F", " ": "00000000000000", ".": "00000000000C0C",
    "-": "0000001F000000", ":": "000C0C000C0C00", "_": "0000000000001F", "/": "00010204081000",
    "(": "02040808080402", ")": "08040202020408", "+": "0004041F040400", "=": "00001F001F0000",
    "%": "18190204081303", ",": "000000000C0408", "|": "04040404040404",
}
_GLYPH = {}


def _glyph(ch):
    g = _GLYPH.get(ch)
    if g is None:
        h = _FONT.get(ch.upper(), _FONT[" "])
        rows = [int(h[i:i + 2], 16) for i in range(0, 14, 2)]
        g = np.array([[(r >> (4 - c)) & 1 for c in range(5)] for r in rows], bool)
        _GLYPH[ch] = g
    return g


def _text(img, x, y, s, color):
    """draw s with its top-left at (x, y); 6 px per character"""
    H, W, _ = img.shape
    for i, ch in enumerate(str(s)):
        g = _glyph(ch)
        gx = x + 6 * i
        if gx + 5 > W or y + 7 > H or gx < 0 or y < 0:
            continue
        img[y:y + 7, gx:gx + 5][g] = color


def _png(img):
    """uint8 [H, W, 3] -> PNG bytes (zlib only)"""
    H, W, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(H))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _hex(c):
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)


C_BG, C_LANE, C_GRID, C_BAR, C_WAVE = _hex("#FAFAF7"), _hex("#F0F1F4"), _hex("#E2E4E9"), _hex("#C3C8D1"), _hex("#9AA1AD")
C_FILL, C_CURVE, C_HIT, C_KEY, C_TEXT = _hex("#D6E2F5"), _hex("#1F5FBF"), _hex("#E8590C"), _hex("#0B2E66"), _hex("#2B2F36")
C_FLASH = _hex("#C92A2A")
LUMA = np.array([0.2126, 0.7152, 0.0722])                       # Rec. 709 luminance of linear RGB (colour outputs)


def _time_step(span):
    for s in (0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600):
        if span / s <= 12:
            return s
    return 1200


def _fmt_t(t):
    if t >= 60:
        return f"{int(t // 60)}:{t % 60:04.1f}" if t % 1 else f"{int(t // 60)}:{int(t % 60):02d}"
    return f"{t:g}s"


def render_plot(output, hits, keys_on, peaks, peaks_per_s, beats, fps, offset, t0, t1, width=1200, flash=None):
    """Draw one bake output: waveform lane on top, then one lane per channel with the value curve (filled), hits as
    orange ticks, keys as dark dots, beat lines through every lane. flash: per channel, [[s0, s1], ...] spans (bake
    seconds) that break the flash-safety limit, drawn as a red bar along the top of the lane. Returns PNG bytes."""
    chans = output["channels"]
    n = len(chans)
    lane_h = int(np.clip(480 // max(n, 1), 36, 110))
    left, right, top, wave_h, gap, bottom = 96, 12, 26, 64, 6, 22
    W = int(width)
    H = top + wave_h + gap + n * (lane_h + gap) + bottom
    img = np.empty((H, W, 3), np.uint8)
    img[:] = C_BG
    pw = W - left - right
    xs_t = t0 + (np.arange(pw) + 0.5) / pw * (t1 - t0)            # audio time at each pixel column centre

    def x_of(t):
        return left + int((t - t0) / (t1 - t0) * pw)

    lanes = [(top, wave_h)] + [(top + wave_h + gap + i * (lane_h + gap), lane_h) for i in range(n)]
    for y, h in lanes:
        img[y:y + h, left:left + pw] = C_LANE
    # beat and bar lines (bar = every 4th beat from the first beat)
    for k, bt in enumerate(beats):
        if t0 <= bt < t1:
            x = x_of(bt)
            col = C_BAR if k % 4 == 0 else C_GRID
            for y, h in lanes:
                img[y:y + h, x] = col
    # waveform: max |amplitude| per column, mirrored
    if len(peaks):
        y, h = lanes[0]
        mid = y + h // 2
        a0 = np.clip((xs_t - (t1 - t0) / pw / 2) * peaks_per_s, 0, len(peaks)).astype(int)
        a1 = np.clip((xs_t + (t1 - t0) / pw / 2) * peaks_per_s, 0, len(peaks)).astype(int)
        top_pk = max(float(np.max(peaks)), 1e-6)
        for c in range(pw):
            seg = peaks[a0[c]:max(a1[c], a0[c] + 1)]
            if len(seg):
                amp = int(min(float(seg.max()) / top_pk, 1.0) * (h // 2 - 2))
                img[mid - amp:mid + amp + 1, left + c] = C_WAVE
        _text(img, 6, y + 4, "AUDIO", C_TEXT)
    lo, hi = float(output.get("min", 0)), float(output.get("max", 1))
    span = (hi - lo) or 1.0
    fr_of = lambda t: t * fps + offset                              # noqa: E731  audio s -> bake frame
    for i, ch in enumerate(chans):
        y, h = lanes[i + 1]
        vals = np.asarray(ch["values"], np.float64)
        N = len(vals)
        f0 = np.clip(fr_of(xs_t - (t1 - t0) / pw / 2), 0, N - 1)
        f1 = np.clip(fr_of(xs_t + (t1 - t0) / pw / 2), 0, N - 1)
        norm = np.clip((vals - min(lo, hi)) / abs(span), 0, 1)
        frames_idx = np.arange(N)
        prev = None
        for c in range(pw):
            # Sequencer interpolates linearly between keys, so draw the line through the frame values, not steps
            inner = np.arange(int(np.ceil(f0[c])), int(np.floor(f1[c])) + 1)
            seg = np.interp(np.concatenate(([f0[c]], inner, [f1[c]])), frames_idx, norm)
            vmin, vmax = float(seg.min()), float(seg.max())
            ytop = y + h - 2 - int(vmax * (h - 5))
            ybot = y + h - 2 - int(vmin * (h - 5))
            img[ytop:y + h - 1, left + c] = C_FILL
            y_a, y_b = (min(ytop, prev), max(ybot, prev)) if prev is not None else (ytop, ybot)
            img[max(y_a - 1, y):min(y_b + 2, y + h), left + c] = C_CURVE
            prev = y + h - 2 - int(float(seg[-1]) * (h - 5))
        # keys (dark dots) when they are sparse enough to read
        kf = [k for k in ch.get("keys", []) if t0 <= (k[0] - offset) / fps < t1]
        if keys_on and len(kf) <= pw // 4:
            for f, v in kf:
                x = x_of((f - offset) / fps)
                yy = y + h - 2 - int(np.clip((v - min(lo, hi)) / abs(span), 0, 1) * (h - 5))
                img[max(yy - 1, y):yy + 2, max(x - 1, left):x + 2] = C_KEY
        # hits: orange ticks along the lane floor and a thin line up the lane
        for f in np.flatnonzero(hits[i]) if i < len(hits) else []:
            t = (f - offset) / fps
            if t0 <= t < t1:
                x = x_of(t)
                img[y + h - 7:y + h, max(x - 1, left):x + 1] = C_HIT
                img[y:y + h - 7:3, x] = C_HIT
        # flash-safety violations: a red bar along the lane top
        for s0, s1 in (flash[i] if flash and i < len(flash) else []):
            a0, a1 = max(s0 - offset / fps, t0), min(s1 - offset / fps, t1)
            if a1 > a0:
                img[y:y + 4, x_of(a0):max(x_of(a1), x_of(a0) + 1)] = C_FLASH
        label = ch.get("actor") or output.get("target", {}).get("param") or output["id"]
        _text(img, 6, y + 4, str(label)[:15], C_TEXT)
        _text(img, 6, y + h - 9, f"{lo:g}-{hi:g}"[:15], C_WAVE)
    # time axis
    st = _time_step(t1 - t0)
    t = np.ceil(t0 / st) * st
    while t < t1:
        x = x_of(t)
        img[H - bottom:H - bottom + 4, x] = C_TEXT
        _text(img, max(x - 9, left), H - bottom + 7, _fmt_t(round(float(t), 2)), C_TEXT)
        t += st
    tgt = output.get("target", {})
    title = (f"{output['id']}  {tgt.get('kind', '')}  {t0:.2f}-{t1:.2f}s  {n} ch   "
             f"grey=audio  lines=beats  orange=hits  blue=value  dots=keys" + ("  red=flash limit" if flash else ""))
    _text(img, 6, 8, title[: (W - 12) // 6], C_TEXT)
    return _png(img)


def _plot_data(rig_path, output_id, start_s, end_s, channels):
    p, data, rig = _load(rig_path)
    a, sigs, res = _bake_all(rig)
    o = next((x for x in res["outputs"] if x["id"] == output_id), None)
    if o is None:
        raise ToolFailure(f"no output {output_id!r} (outputs: {', '.join(x['id'] for x in res['outputs'])})")
    s = sigs[o["signal"]].broadcast(len(o["channels"]))
    idx = list(range(len(o["channels"])))
    if channels:
        names = [c.get("actor") for c in o["channels"]]
        idx = [names.index(c) if isinstance(c, str) and c in names else int(c) for c in channels]
    idx = idx[:16]
    o = dict(o, channels=[o["channels"][i] for i in idx])
    if o.get("value_type") == "color" or (o["channels"] and o["channels"][0]["values"]
                                          and isinstance(o["channels"][0]["values"][0], list)):
        o = _as_luminance(o)
    hits = [s.hits[i] for i in idx]
    fps, off = float(res["fps"]), int(res.get("audio_offset_frames", 0))
    dur = (res["frames"] - off) / fps
    t0 = max(0.0, float(start_s))
    t1 = min(dur, float(end_s)) if end_s is not None else dur
    if t1 <= t0:
        raise ToolFailure(f"empty range {t0}-{t1} s (bake covers 0-{dur:.2f} s of audio)")
    peaks = a.arrays["peaks"].astype(np.float64) if a is not None else np.zeros(0)
    beats = [b["time"] for b in a.beats] if a is not None else []
    return o, hits, peaks, beats, fps, off, t0, t1, data


def _as_luminance(o):
    """a colour output drawn as one luminance curve per channel (0..the output's brightest), keys likewise"""
    chans, top = [], 0.0
    for c in o["channels"]:
        lum = np.asarray(c["values"], np.float64)[:, :3] @ LUMA
        top = max(top, float(lum.max()) if len(lum) else 0.0)
        keys = [[k[0], float(np.dot(k[1:4], LUMA))] for k in c.get("keys", [])]
        chans.append(dict(c, values=lum.tolist(), keys=keys))
    return dict(o, channels=chans, min=0.0, max=top or 1.0, luminance=True)


def _flash_safety(rig_data, o, fps):
    """(per-channel violation spans in bake seconds, caption block) from beatrig.safety.flash_report, or (None, None)
    when that module isn't installed. Colour outputs are judged on luminance, as the bake check does."""
    try:
        from . import safety
        report = safety.flash_report
    except (ImportError, AttributeError):
        return None, None
    try:
        s = safety.settings(rig_data)[0] if hasattr(safety, "settings") else {"max_flashes_per_s": 3.0, "min_swing": 0.1}
        lo, hi = float(o.get("min", 0.0)), float(o.get("max", 1.0))
        spans, rows = [], []
        for c in o["channels"]:
            r = report(c["values"], fps, lo, hi, s["max_flashes_per_s"], s["min_swing"])
            spans.append(r.get("windows") or [])
            rows.append({"actor": c.get("actor"), "ok": r.get("ok"), "max_per_s": r.get("max_per_s"),
                         "windows_s": (r.get("windows") or [])[:8]})
        return spans, {"limit_per_s": s["max_flashes_per_s"], "min_swing": s["min_swing"],
                       "ok": all(x["ok"] for x in rows), "channels": rows}
    except Exception as e:                                     # an engine change must never break plot
        return None, {"unavailable": f"{type(e).__name__}: {e}"}


def plot_png(rig_path, output_id, start_s=0.0, end_s=None, channels=None, width=1200, keys=True):
    """(png bytes, caption dict); the image the plot tool returns"""
    o, hits, peaks, beats, fps, off, t0, t1, rig_data = _plot_data(rig_path, output_id, start_s, end_s, channels)
    spans, safety_cap = _flash_safety(rig_data, o, fps)
    png = render_plot(o, hits, keys, peaks, 200.0, beats, fps, off, t0, t1, int(np.clip(width, 480, 2400)),
                      flash=spans if spans and any(spans) else None)
    f0, f1 = int(t0 * fps) + off, int(np.ceil(t1 * fps)) + off
    cap = {"output": o["id"], "range_s": [_r(t0, 2), _r(t1, 2)], "frames": [f0, f1], "channels": []}
    for ch, h in zip(o["channels"], hits):
        v = np.asarray(ch["values"][f0:f1], np.float64)
        cap["channels"].append({"actor": ch.get("actor"), "min": _r(v.min(), 4) if len(v) else None,
                                "max": _r(v.max(), 4) if len(v) else None, "hits": int(h[f0:f1].sum()),
                                "keys": sum(1 for k in ch["keys"] if f0 <= k[0] < f1)})
    cap["legend"] = "top lane: audio waveform; grey lines: beats (darker = bar); per channel: blue value curve " \
                    "(lane floor = output min, top = max), orange ticks = trigger hits, dark dots = baked keys"
    if o.get("luminance"):
        cap["legend"] += "; colour output: curves show luminance (Rec. 709 of linear RGB)"
    if safety_cap is not None:
        cap["flash_safety"] = safety_cap
        if safety_cap.get("channels"):
            cap["legend"] += "; red bar on a lane top = more flashes per second than the safety limit"
    return png, cap


@tool
def plot(rig_path: str, output_id: str, start_s: float = 0.0, end_s: float | None = None,
         channels: list | None = None, width: int = 1200):
    """SEE the animation: a PNG of one output's baked curves over the audio waveform, with beats, trigger hits and keys
    marked, one lane per channel (up to 16; pick some with channels=[actor names or indexes]). Use a short range
    (4-16 s) to judge timing and easing; the whole song to judge structure. Returns the image plus a JSON caption."""
    png, cap = plot_png(rig_path, output_id, start_s, end_s, channels, width)
    return [_j(cap), _image(png)]


def _image(png):
    Image = _mcp_image_class()
    return Image(data=png, format="png")


# ------------------------------------------------------------------------------------------------ preview page
@tool
def preview_url() -> str:
    """Is the preview page (waveform, note lanes, curves, light ring) running? Returns its URL for the human to open."""
    try:
        with urllib.request.urlopen(urllib.request.Request(PREVIEW_URL + "/", method="HEAD"), timeout=0.5) as r:
            return _j({"running": True, "url": PREVIEW_URL + "/", "status": r.status})
    except Exception:
        return _j({"running": False, "url": PREVIEW_URL + "/",
                   "hint": "start it with preview_start(rig_path) or `beatrig serve <rig.json>`"})


@tool
def preview_start(rig_path: str, port: int = 4700) -> str:
    """Start the preview page for a rig in the background (localhost only) and return its URL. It runs as a child of
    this MCP server, so it stops when the server does; for a long-lived page run `beatrig serve <rig.json>` yourself."""
    p = _rig_path(rig_path)
    url = f"http://127.0.0.1:{int(port)}/"
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=0.5)
        return _j({"running": True, "url": url, "started": False,
                   "note": "a preview server is already listening there; it may be showing another rig"})
    except Exception:
        pass
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    proc = subprocess.Popen([sys.executable, "-m", "beatrig", "serve", p, "--port", str(int(port))],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=flags, cwd=os.path.dirname(p))
    import time
    for _ in range(40):
        time.sleep(0.25)
        if proc.poll() is not None:
            raise ToolFailure(f"preview server exited with code {proc.returncode}; run `beatrig serve {p}` to see why")
        try:
            urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=0.5)
            return _j({"running": True, "url": url, "started": True, "pid": proc.pid})
        except Exception:
            continue
    return _j({"running": False, "url": url, "started": True, "pid": proc.pid, "note": "not answering yet"})


# ------------------------------------------------------------------------------------------------ Unreal (Remote Control)
_LINKS = {}


def _link(host=None):
    """one persistent beatrig.live.Link per editor host for the life of this server (keep-alive connection, applier
    imported once, push cache). It leaves the editor's background throttle alone (keep_awake=False)."""
    from .live import Link
    key = host or UE_HOST
    if key not in _LINKS:
        _LINKS[key] = Link(key, keep_awake=False)
    return _LINKS[key]


def _live(fn, *a, **kw):
    from .live import LiveError
    try:
        return fn(*a, **kw)
    except LiveError as e:
        raise ToolFailure(str(e)) from None


def _bake_file(bake_path):
    bp = os.path.abspath(bake_path)
    if not os.path.exists(bp):
        raise ToolFailure(f"bake not found: {bp} (run bake first)")
    return bp


def _ue_result(res, keep=("ok", "sequence", "keys", "replaced", "removed", "errors", "refreshed", "saved", "bindings",
                          "pushed", "skipped", "calls", "ms", "rt_ms", "editor_ms", "write_ms")):
    """the editor's result without the per-track list (a count instead)"""
    out = {k: res[k] for k in keep if k in res}
    if "written" in res:
        out["tracks"] = len(res["written"])
    if isinstance(out.get("bindings"), dict) and isinstance(out["bindings"].get("pruned"), list):
        out["bindings"] = dict(out["bindings"], pruned=len(out["bindings"]["pruned"]))
    return out


@tool
def ue_apply(bake_path: str, sequence_path: str, host: str | None = None, start_frame: int = 0,
             only: list | None = None, save: bool = False) -> str:
    """Write a bake.json into a Level Sequence in a running Unreal editor (Remote Control on localhost:30010).
    sequence_path is the asset path, e.g. "/Game/Show/LS_Main". Replaces only the tracks beatrig wrote before for these
    outputs (tagged "beatrig:<output id>"). The bake's fps must equal the sequence's display rate. save=true saves the
    asset. Tested on UE 5.8.3; run ue_check afterwards to compare key for key."""
    res = _live(_link(host).apply, _bake_file(bake_path), sequence_path, only or None, int(start_frame), bool(save))
    _notify("ue_apply", f"{sequence_path}: {res.get('keys', 0)} keys, ok={res.get('ok')}")
    return _j(_ue_result(res))


@tool
def ue_undo(sequence_path: str, output_id: str | None = None, host: str | None = None, save: bool = False) -> str:
    """Remove the tracks beatrig wrote into a Level Sequence (all of them, or one output's), plus the bindings beatrig
    created that are now empty. Bindings that were already in the sequence are never removed."""
    res = _live(_link(host).undo, sequence_path, output_id, bool(save))
    _notify("ue_undo", f"{sequence_path}: removed {len(res.get('removed', []))} track(s)")
    return _j(_ue_result(res))


@tool
def ue_check(bake_path: str, sequence_path: str, host: str | None = None, start_frame: int = 0,
             only: list | None = None) -> str:
    """Read the keys back from the Level Sequence and compare them with the bake: per track, expected vs actual key
    counts, missing/extra keys and the largest value error (colour tracks: all four channels; Niagara tracks: also the
    parameter name)."""
    res = _live(_link(host).check, _bake_file(bake_path), sequence_path, only or None, int(start_frame))
    bad = [t for t in res.get("tracks", []) if not t.get("ok")]
    return _j({"ok": res.get("ok"), "sequence": res.get("sequence"), "tracks": len(res.get("tracks", [])),
               "keys": res.get("keys"), "failed": bad[:20], "rt_ms": res.get("rt_ms")})


@tool
def live_push(bake_path: str, sequence_path: str, only_changed: bool = True, start_frame: int = 0,
              host: str | None = None) -> str:
    """Push a bake into a Level Sequence, rewriting only the outputs whose keys changed since this server's last push
    to that sequence (only_changed=false rewrites all). Outputs pushed before but gone from the bake are removed. If the
    sequence is open in Sequencer it is re-evaluated, so the viewport shows the change. Returns pushed/skipped/removed
    output ids and timings (ms)."""
    res = _live(_link(host).push, _bake_file(bake_path), sequence_path, bool(only_changed), int(start_frame))
    _notify("live_push", f"{sequence_path}: pushed {res.get('pushed')}, skipped {len(res.get('skipped', []))}")
    return _j(_ue_result(res, keep=("ok", "pushed", "skipped", "removed", "errors", "keys", "refreshed", "calls", "ms",
                                    "rt_ms", "editor_ms", "write_ms")))


@tool
def ue_scan(out: str | None = None, rig_path: str | None = None, host: str | None = None,
            paths: list | None = None) -> str:
    """Scan the level open in the editor: every actor (class, location, rotation, mobility, components of kind light/
    mesh/niagara/camera, each mesh's material slots with their scalar/vector parameter names), the MPC assets with
    their parameters, the Level Sequences, and the camera (an OWL 360 dome camera: projection, dome_angle). Writes
    scene.json to `out`, or next to `rig_path` (the rig's "scene" file if it names one), and returns a summary. Use the
    names it lists in outputs (actors, params, MPC paths). paths = content folders searched for assets (default /Game)."""
    scene = _live(_link(host).scan, paths or None)
    path = None
    if out:
        path = os.path.abspath(out)
    elif rig_path:
        rp = _rig_path(rig_path)
        path = os.path.join(os.path.dirname(rp), _read_json(rp).get("scene") or "scene.json")
    if path:
        _write_json(path, scene)
    kinds, params = {}, {}
    for name, a in scene.get("actors", {}).items():
        for c in a.get("components", []):
            if name not in kinds.setdefault(c["kind"], []):
                kinds[c["kind"]].append(name)
            for m in c.get("materials", []):
                for p in m.get("scalars", []) + m.get("vectors", []):
                    params[p] = params.get(p, 0) + 1
    return _j({"scene_path": path, "level": scene.get("level"), "engine": scene.get("engine"),
               "actors": len(scene.get("actors", {})),
               "by_kind": {k: {"count": len(v), "names": v[:40]} for k, v in kinds.items()},
               "material_params": params, "mpcs": scene.get("mpcs"), "sequences": scene.get("sequences", [])[:40],
               "camera": scene.get("camera"), "scan_ms": scene.get("scan_ms"),
               "next": "add \"scene\": \"scene.json\" to the rig so check can warn about unknown actors and params"})


@tool
def playhead_get(host: str | None = None) -> str:
    """The sequence open in Sequencer, its playhead in display frames, whether it is playing, its fps and range."""
    return _j(_live(_link(host).playhead_get))


@tool
def playhead_set(frame: float | None = None, playing: bool | None = None, sequence_path: str | None = None,
                 host: str | None = None) -> str:
    """Move the Sequencer playhead (display frames) and/or play (true) / pause (false). sequence_path opens that
    sequence in Sequencer first if it is not the open one; that replaces whatever Sequencer was showing."""
    return _j(_live(_link(host).playhead_set, frame, playing, sequence_path))


# ------------------------------------------------------------------------------------------------ server
TOOLS = [analyze, describe_audio, list_ops, get_rig, set_node, add_node, remove_node, set_output, remove_output,
         set_layer, check, get_notes, notes_init, notes_edit, notes_snap, bake, plot, preview_url, preview_start,
         ue_scan, ue_apply, ue_undo, ue_check, live_push, playhead_get, playhead_set]

INSTRUCTIONS = """beatrig turns music into deterministic animation keys. Files: rig.json (motion graph: nodes -> outputs),
notes.json (editable hit lanes, times in audio seconds), bake.json (per-frame values + linear keys per target).
Typical loop: analyze -> describe_audio -> notes_init (from a suggested band) -> list_ops -> add_node/set_node/
set_output -> check -> bake -> plot (look at the result, adjust, repeat) -> ue_apply -> ue_check.
With an editor open: ue_scan first (real actor, parameter and MPC names); live_push re-sends only changed outputs.
Rules the engine enforces: nothing moves ahead of its sound; continuous motion is smoothed; only trigger envelopes may
jump instantly; same inputs give the same bake. Paths are absolute or relative to the server's working directory."""


def _mcp_image_class():
    try:
        from mcp.server.mcpserver import Image
    except ImportError:
        from mcp.server.fastmcp import Image
    return Image


def build_server():
    try:
        from mcp.server.mcpserver import MCPServer as Server
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as Server
            from mcp.server.fastmcp.exceptions import ToolError
        except ImportError:
            raise SystemExit("beatrig-mcp needs the MCP SDK: pip install \"beatrig[mcp]\"") from None
    from . import __version__
    srv = Server("beatrig", instructions=INSTRUCTIONS)
    try:
        srv._mcp_server.version = __version__          # cosmetic; attribute name differs across SDK versions
    except Exception:
        pass

    def wrap(fn):
        @functools.wraps(fn)
        def call(*a, **kw):
            try:
                return fn(*a, **kw)
            except ToolFailure as e:
                raise ToolError(str(e)) from None
        return call

    for fn in TOOLS:
        srv.add_tool(wrap(fn), name=fn.__name__, description=(fn.__doc__ or "").strip())
    return srv


def main(argv=None):
    build_server().run()


if __name__ == "__main__":
    main()
