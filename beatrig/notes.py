"""notes.json: the editable hit timeline. Times are seconds of audio (never frames), so changing fps never moves hits."""
import json
import os
from typing import Optional

import numpy as np

from .tempo import Tempo


def empty_notes():
    return {"bpm_map": [], "grid_offset": 0.0, "tracks": {}}


def load_notes(path: Optional[str]) -> dict:
    """missing file -> empty notes"""
    if not path or not os.path.exists(path):
        return empty_notes()
    with open(path, encoding="utf-8") as fh:
        n = json.load(fh)
    n.setdefault("bpm_map", [])
    n.setdefault("grid_offset", 0.0)
    n.setdefault("tracks", {})
    return n


def clean_hit(h):
    out = {"time": round(float(h["time"]), 6), "vel": round(float(h.get("vel", 1.0)), 4)}
    for k in ("len_beats", "shape"):
        if h.get(k) is not None:
            out[k] = round(float(h[k]), 4) if k == "len_beats" else str(h[k])
    return out


def normalise(notes):
    """sorted hits, rounded, known keys in a fixed order (keeps diffs and round-trips stable)"""
    out = {"bpm_map": [{"time": round(float(b["time"]), 6), "bpm": round(float(b["bpm"]), 4)} for b in notes.get("bpm_map", [])],
           "grid_offset": round(float(notes.get("grid_offset", 0.0)), 6), "tracks": {}}
    for name, hits in notes.get("tracks", {}).items():
        out["tracks"][name] = sorted((clean_hit(h) for h in hits), key=lambda h: h["time"])
    return out


def save_notes(path: str, notes: dict, backup: bool = False) -> dict:
    """write notes.json in a stable layout (sorted, rounded hits); returns the normalised notes. backup: keep <path>.bak"""
    if backup and os.path.exists(path):
        os.replace(path, path + ".bak")
    n = normalise(notes)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("{\n")
        fh.write(f' "bpm_map": {json.dumps(n["bpm_map"])},\n "grid_offset": {json.dumps(n["grid_offset"])},\n "tracks": {{')
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
