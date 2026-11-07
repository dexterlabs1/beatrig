"""notes.json: the editable hit timeline. Times are seconds of audio (never frames), so changing fps never moves hits.

v0.2: `sections` ([{name, time}], ordered) and `patterns` ({name: {length_beats, hits: [{beat, vel, ...}]}}). A track
entry {"pattern", "at_beat", "repeat"} is a linked loop: it expands, on the notes' bpm map, into ordinary hits that
carry `from_pattern`. Expansion is idempotent (expanded hits are dropped and rebuilt from the placements), happens
at load time and again at evaluation, and saving writes only the placements, so editing a pattern moves every copy.
"""
import json
import os
from typing import Optional

import numpy as np

from .tempo import Tempo


def empty_notes():
    return {"bpm_map": [], "grid_offset": 0.0, "tracks": {}}


def load_notes(path: Optional[str]) -> dict:
    """missing file -> empty notes. Pattern placements are expanded (expanded hits carry from_pattern)."""
    if not path or not os.path.exists(path):
        return empty_notes()
    with open(path, encoding="utf-8") as fh:
        n = json.load(fh)
    n.setdefault("bpm_map", [])
    n.setdefault("grid_offset", 0.0)
    n.setdefault("tracks", {})
    return expand_patterns(n)


def _extras(h, out):
    for k in ("len_beats", "shape"):
        if h.get(k) is not None:
            out[k] = round(float(h[k]), 4) if k == "len_beats" else str(h[k])
    return out


def clean_hit(h):
    return _extras(h, {"time": round(float(h["time"]), 6), "vel": round(float(h.get("vel", 1.0)), 4)})


def clean_ref(r):
    return {"pattern": str(r["pattern"]), "at_beat": round(float(r.get("at_beat", 0.0)), 6), "repeat": int(r.get("repeat", 1))}


def clean_pattern(p):
    hits = sorted((_extras(h, {"beat": round(float(h["beat"]), 6), "vel": round(float(h.get("vel", 1.0)), 4)})
                   for h in p.get("hits", [])), key=lambda h: h["beat"])
    return {"length_beats": round(float(p.get("length_beats", 4.0)), 6), "hits": hits}


def is_ref(h):
    return isinstance(h, dict) and "pattern" in h and "time" not in h


def normalise(notes):
    """sorted hits, rounded, known keys in a fixed order (keeps diffs and round-trips stable). Expanded pattern hits
    are dropped (their placements are kept), so a save never freezes a linked loop into plain hits."""
    out = {"bpm_map": [{"time": round(float(b["time"]), 6), "bpm": round(float(b["bpm"]), 4)} for b in notes.get("bpm_map", [])],
           "grid_offset": round(float(notes.get("grid_offset", 0.0)), 6)}
    if notes.get("sections"):
        out["sections"] = sorted(({"name": str(x["name"]), "time": round(float(x["time"]), 6)} for x in notes["sections"]),
                                 key=lambda x: x["time"])
    if notes.get("patterns"):
        out["patterns"] = {str(k): clean_pattern(v) for k, v in notes["patterns"].items()}
    out["tracks"] = {}
    for name, hits in notes.get("tracks", {}).items():
        plain = sorted((clean_hit(h) for h in hits if not is_ref(h) and not h.get("from_pattern")), key=lambda h: h["time"])
        refs = sorted((clean_ref(h) for h in hits if is_ref(h)), key=lambda r: r["at_beat"])
        out["tracks"][name] = plain + refs
    return out


def _pattern_tempo(notes):
    return notes_tempo(notes) or Tempo([(0.0, 120.0)], float(notes.get("grid_offset", 0.0) or 0.0))


def expand_track(notes, name, tempo=None):
    """one track's hits with its placements expanded: plain hits + one hit per pattern hit per repeat, sorted by time.
    at_beat counts on the notes' bpm map (120 bpm from grid_offset when the notes have none)."""
    tempo = tempo or _pattern_tempo(notes)
    pats = notes.get("patterns") or {}
    items = (notes.get("tracks") or {}).get(name, [])
    out = [h for h in items if not is_ref(h) and not h.get("from_pattern")]
    for r in (h for h in items if is_ref(h)):
        p = pats.get(r["pattern"])
        if not p:
            continue
        L = float(p.get("length_beats", 4.0))
        for k in range(max(0, int(r.get("repeat", 1)))):
            base = float(r.get("at_beat", 0.0)) + k * L
            for h in p.get("hits", []):
                hit = _extras(h, {"time": round(float(tempo.time_of_beat(base + float(h["beat"]))), 6),
                                  "vel": round(float(h.get("vel", 1.0)), 4)})
                hit["from_pattern"] = {"pattern": r["pattern"], "at_beat": float(r.get("at_beat", 0.0)), "copy": k}
                out.append(hit)
    return sorted(out, key=lambda h: float(h["time"]))


def track_hits(notes, name):
    """the hits a notes op plays: plain + freshly expanded pattern hits (never stale copies)"""
    return expand_track(notes or {}, name)


def expand_patterns(notes):
    """expand every track's placements in place (idempotent); placements stay at the end of each track list"""
    tempo = _pattern_tempo(notes)
    for name in list((notes.get("tracks") or {})):
        refs = [h for h in notes["tracks"][name] if is_ref(h)]
        notes["tracks"][name] = expand_track(notes, name, tempo) + refs
    return notes


def section_frames(notes, fps, offset=0, t0=0.0):
    """{section name: first frame at or after its time}"""
    out = {}
    for x in (notes or {}).get("sections") or []:
        out[str(x["name"])] = int(np.ceil(np.round((float(x["time"]) - float(t0)) * float(fps), 6))) + int(offset)
    return out


def make_pattern(notes, name, length_beats, hits=None, from_track=None, at_beat=0.0):
    """add/replace pattern `name`: explicit hits [{beat, vel}], or the plain hits of `from_track` within
    [at_beat, at_beat + length_beats) (beats on the notes' bpm map, made relative to at_beat). Returns the pattern."""
    L = float(length_beats)
    if L <= 0:
        raise ValueError("length_beats must be > 0")
    if hits is None:
        tempo = _pattern_tempo(notes)
        hits = []
        for h in (notes.get("tracks") or {}).get(from_track, []):
            if is_ref(h) or h.get("from_pattern"):
                continue
            b = float(tempo.beats_at(float(h["time"]))) - float(at_beat)
            if -1e-6 <= b < L - 1e-6:
                hits.append(_extras(h, {"beat": round(max(0.0, b), 6), "vel": h.get("vel", 1.0)}))
    notes.setdefault("patterns", {})[name] = clean_pattern({"length_beats": L, "hits": hits})
    return notes["patterns"][name]


def place_pattern(notes, track, name, at_beat, repeat=1, replace=False):
    """add a linked placement of pattern `name` to `track`; replace=True removes the plain hits it covers"""
    p = (notes.get("patterns") or {}).get(name)
    if p is None:
        raise ValueError(f"no pattern {name!r}")
    items = notes.setdefault("tracks", {}).setdefault(track, [])
    if replace:
        tempo = _pattern_tempo(notes)
        a, b = float(at_beat), float(at_beat) + int(repeat) * float(p["length_beats"])
        items[:] = [h for h in items if is_ref(h) or h.get("from_pattern")
                    or not a - 1e-6 <= float(tempo.beats_at(float(h["time"]))) < b - 1e-6]
    items.append({"pattern": name, "at_beat": float(at_beat), "repeat": int(repeat)})
    return expand_patterns(notes)


def save_notes(path: str, notes: dict, backup: bool = False) -> dict:
    """write notes.json in a stable layout (sorted, rounded hits); returns the normalised notes. backup: keep <path>.bak"""
    if backup and os.path.exists(path):
        os.replace(path, path + ".bak")
    n = normalise(notes)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("{\n")
        fh.write(f' "bpm_map": {json.dumps(n["bpm_map"])},\n "grid_offset": {json.dumps(n["grid_offset"])},\n')
        if n.get("sections"):
            fh.write(' "sections": [' + ",".join("\n  " + json.dumps(x) for x in n["sections"]) + "\n ],\n")
        if n.get("patterns"):
            pats = list(n["patterns"].items())
            fh.write(' "patterns": {' + ",".join(f"\n  {json.dumps(k)}: {json.dumps(v)}" for k, v in pats) + "\n },\n")
        fh.write(' "tracks": {')
        names = list(n["tracks"])
        for i, name in enumerate(names):
            fh.write(f'\n  {json.dumps(name)}: [')
            hits = n["tracks"][name]
            for j, h in enumerate(hits):
                fh.write("\n   " + json.dumps(h) + ("," if j < len(hits) - 1 else ""))
            fh.write("\n  ]" if hits else "]")
            fh.write("," if i < len(names) - 1 else "")
        fh.write("\n }\n}\n" if names else "}\n}\n")
    return n


def notes_tempo(notes):
    if not notes or not notes.get("bpm_map"):
        return None
    return Tempo([(b["time"], b["bpm"]) for b in notes["bpm_map"]], notes.get("grid_offset", 0.0))


def parse_grid(grid):
    """'1/8' -> 0.5 beats (a quarter note is one beat); a plain number is beats"""
    g = str(grid)
    if "/" in g:
        a, b = g.split("/")
        return 4.0 * float(a) / float(b)
    return float(g)


def snap(hits, tempo, grid_beats, strength=1.0):
    """quantise hit times to the nearest grid line (in beats from the grid offset); strength 0..1 moves part way.
    Returns (new hits, stats). Grid lines are beat times, which are themselves causal onset times."""
    out, moved, earlier = [], 0, 0.0
    for h in hits:
        t = float(h["time"])
        b = float(tempo.beats_at(t))
        tq = float(tempo.time_of_beat(np.round(b / grid_beats) * grid_beats))
        nt = t + strength * (tq - t)
        if abs(nt - t) > 1e-9:
            moved += 1
            earlier = max(earlier, t - nt)
        out.append(dict(h, time=round(nt, 6)))
    return out, dict(moved=moved, max_earlier_ms=round(earlier * 1000, 1))
