"""Command line: analyze | notes init/snap/show | bake | check | serve."""
import argparse
import logging
import sys
import time

import numpy as np

from .bake import analyze, bake, bake_stats, make_ctx
from .check import check
from .notes import load_notes, notes_tempo, parse_grid, save_notes, snap
from .ops import OpError, beat_times, detect_onsets
from .rig import RigError, load_rig
from .tempo import Tempo


def cmd_analyze(a):
    rig = load_rig(a.rig)
    t0 = time.perf_counter()
    an = analyze(rig, use_cache=not a.no_cache)
    if an is None:
        print("rig has no audio")
        return 0
    m = an.meta
    print(f"audio     {an.audio}")
    print(f"format    {an.sr} Hz, {an.channels} ch, {m['duration']:.3f} s")
    print(f"cache     {'hit' if an.loaded_from_cache else 'built'} ({time.perf_counter() - t0:.2f} s)")
    bpms = ", ".join(f"{b['bpm']} @ {b['time']:.2f}s" for b in m["bpm_map"])
    print(f"bpm       {bpms}" + ("  (LOW CONFIDENCE: no proven pulse)" if m["fallback"] else ""))
    print(f"grid      offset {m['grid_offset']:.3f} s")
    print(f"beats     {len(m['beats'])}")
    print(f"pulse     {m['pulse']['label']} (confidence {m['pulse']['confidence']})")
    for p in m["passages"]:
        print(f"  passage {p['start']:.2f}-{p['end']:.2f}s {p['bpm']} bpm conf {p['confidence']} hit {p['hit_rate']} jitter {p['jitter']} agree {p['agree']} vs {p['chance']}")
    for p in m["rejected_passages"]:
        print(f"  rejected {p['start']:.2f}-{p['end']:.2f}s {p['bpm']} bpm {p.get('reason', '')}")
    return 0


def cmd_notes(a):
    rig = load_rig(a.rig)
    path = rig.notes_path
    notes = load_notes(path)
    if a.action == "show":
        print(f"notes     {path}")
        print(f"bpm_map   {notes['bpm_map']}  grid_offset {notes['grid_offset']}")
        for name, hits in notes["tracks"].items():
            if a.track and name != a.track:
                continue
            print(f"track {name}: {len(hits)} hits")
            for h in hits[:a.limit]:
                print(f"  {h}")
        return 0
    if not a.track:
        print("--track is required", file=sys.stderr)
        return 2
    if a.action == "init":
        an = analyze(rig)
        ctx = make_ctx(rig, an, notes)
        src, _, nid = (a.source or "").partition(":")
        if src == "onsets":
            node = rig.node(nid)
            if node is None or node.get("op") != "onsets":
                print(f"--from onsets:<node> needs an onsets node id, got {nid!r}", file=sys.stderr)
                return 2
            times, vels = detect_onsets(node, ctx, rig.nodes)
        elif src == "beats":
            node = rig.node(nid) if nid else {"id": "_beats", "op": "beats"}
            if node is None or node.get("op") != "beats":
                print(f"--from beats:<node> needs a beats node id, got {nid!r}", file=sys.stderr)
                return 2
            times, vels = beat_times(node, ctx)
        else:
            print("--from must be onsets:<node> or beats[:<node>]", file=sys.stderr)
            return 2
        if not notes["bpm_map"] and an is not None:
            notes["bpm_map"] = an.bpm_map
            notes["grid_offset"] = an.grid_offset
        notes["tracks"][a.track] = [{"time": float(t), "vel": round(float(v), 3)} for t, v in zip(times, vels)]
        save_notes(path, notes)
        print(f"{a.track}: {len(times)} hits from {a.source} -> {path}")
        return 0
    if a.action == "snap":
        tempo = notes_tempo(notes)
        if tempo is None:
            an = analyze(rig)
            tempo = Tempo([(m["time"], m["bpm"]) for m in an.bpm_map], an.grid_offset) if an else Tempo([(0, 120)])
        hits = notes["tracks"].get(a.track)
        if hits is None:
            print(f"no track {a.track!r}", file=sys.stderr)
            return 2
        notes["tracks"][a.track], st = snap(hits, tempo, parse_grid(a.grid), a.strength)
        save_notes(path, notes)
        print(f"{a.track}: snapped to {a.grid} (strength {a.strength}): {st['moved']} of {len(hits)} moved, "
              f"latest-to-earlier shift {st['max_earlier_ms']} ms")
        return 0
    return 2


def cmd_bake(a):
    rig = load_rig(a.rig)
    out = a.out or rig.resolve("bake.json")
    t0 = time.perf_counter()
    an = analyze(rig)
    t1 = time.perf_counter()
    res = bake(rig, an, out=out)
    t2 = time.perf_counter()
    print(f"bake      {out}")
    print(f"frames    {res['frames']} @ {res['fps']} fps   analysis {t1 - t0:.2f} s, bake {t2 - t1:.2f} s")
    print(f"{'output':16} {'ch':>3} {'min':>10} {'max':>10} {'mean':>10} {'keys':>7} {'values':>8}")
    for oid, ch, lo, hi, mean, k, n in bake_stats(res):
        print(f"{oid:16} {ch:>3} {lo:>10.4g} {hi:>10.4g} {mean:>10.4g} {k:>7} {n:>8}")
    return 0


def cmd_check(a):
    rig = load_rig(a.rig)
    issues = check(rig, load_notes(rig.notes_path))
    for i in issues:
        print(f"{i['level']:7} {i['where']}: {i['msg']}")
    n = sum(i["level"] == "error" for i in issues)
    print(f"{n} error(s), {len(issues) - n} warning(s)")
    return 1 if n else 0


def cmd_serve(a):
    try:
        from .server import run
    except ImportError as e:
        print(f"preview server not available ({e})", file=sys.stderr)
        return 1
    run(a.rig, a.port)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="beatrig", description="offline, causal music -> animation baking")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyze", help="build/refresh the analysis cache and print a summary")
    p.add_argument("rig")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("notes", help="notes init|snap|show")
    p.add_argument("action", choices=["init", "snap", "show"])
    p.add_argument("rig")
    p.add_argument("--track")
    p.add_argument("--from", dest="source", help="onsets:<node> | beats[:<node>]")
    p.add_argument("--grid", default="1/16")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(fn=cmd_notes)
    p = sub.add_parser("bake", help="evaluate and write bake.json")
    p.add_argument("rig")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_bake)
    p = sub.add_parser("check", help="validate the rig")
    p.add_argument("rig")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("serve", help="local preview page")
    p.add_argument("rig")
    p.add_argument("--port", type=int, default=4700)
    p.set_defaults(fn=cmd_serve)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")
    try:
        return a.fn(a)
    except (RigError, OpError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
