"""Command line: analyze | notes (init, snap, show, sections, pattern, import-midi) | bake | check | stems | timeline | serve."""
import argparse
import logging
import sys
import time

from .bake import analyze, bake, bake_stats, make_ctx
from .check import check
from .notes import load_notes, make_pattern, notes_tempo, parse_grid, place_pattern, save_notes, snap
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
    print(f"beats     {len(m['beats'])} from the {an.tracker_label()}")
    bars = an.bars_summary()
    if bars:
        print(f"bars      {bars['meter']} beats per bar, {bars['bars']} bars, first downbeat {bars['first_downbeat']:.3f} s "
              f"(confidence {bars['confidence']})")
    print(f"stems     stem features read {an.stems_label()}")
    print(f"pulse     {m['pulse']['label']} (confidence {m['pulse']['confidence']})")
    for p in m["passages"]:
        print(f"  passage {p['start']:.2f}-{p['end']:.2f}s {p['bpm']} bpm conf {p['confidence']} hit {p['hit_rate']} jitter {p['jitter']} agree {p['agree']} vs {p['chance']}")
    for p in m["rejected_passages"]:
        print(f"  rejected {p['start']:.2f}-{p['end']:.2f}s {p['bpm']} bpm {p.get('reason', '')}")
    cands = m.get("tempo_candidates") or []
    if cands:
        print("tempo     candidates (pin one with rig.tempo.bpm):")
        for c in cands:
            print(f"  {c['ratio']:>5} {c['bpm']:>8.2f} bpm  confidence {c['confidence']:.3f}  (support {c['support']})")
    return 0


# ---------- notes
def _need_track(a):
    if not a.track:
        print("--track is required", file=sys.stderr)
        return False
    return True


def cmd_notes_show(a):
    rig = load_rig(a.rig)
    path = rig.notes_path
    notes = load_notes(path)
    print(f"notes     {path}")
    print(f"bpm_map   {notes['bpm_map']}  grid_offset {notes['grid_offset']}")
    for s in notes.get("sections") or []:
        print(f"section   {s['time']:9.3f}s  {s['name']}")
    for name, p in (notes.get("patterns") or {}).items():
        print(f"pattern   {name}: {p['length_beats']} beats, {len(p['hits'])} hits")
    for name, hits in notes["tracks"].items():
        if a.track and name != a.track:
            continue
        plain = [h for h in hits if "time" in h]
        refs = [h for h in hits if "time" not in h]
        print(f"track {name}: {len(plain)} hits" + (f" ({sum(1 for h in plain if h.get('from_pattern'))} from {len(refs)} pattern placements)" if refs else ""))
        for h in plain[:a.limit]:
            print(f"  {h}")
    return 0


def cmd_notes_init(a):
    if not _need_track(a):
        return 2
    rig = load_rig(a.rig)
    path = rig.notes_path
    notes = load_notes(path)
    an = analyze(rig)
    ctx = make_ctx(rig, an, notes)
    src, _, nid = (a.source or "").partition(":")
    if a.min_conf is not None and src not in ("beats", "downbeats"):
        print("--min-conf applies to --from beats or downbeats", file=sys.stderr)
        return 2
    if src == "onsets":
        node = rig.node(nid)
        if node is None or node.get("op") != "onsets":
            print(f"--from onsets:<node> needs an onsets node id, got {nid!r}", file=sys.stderr)
            return 2
        times, vels = detect_onsets(node, ctx, rig.nodes)
    elif src in ("beats", "downbeats"):
        node = rig.node(nid) if nid and src == "beats" else {"id": "_beats", "op": "beats"}
        if node is None or node.get("op") != "beats":
            print(f"--from beats:<node> needs a beats node id, got {nid!r}", file=sys.stderr)
            return 2
        if src == "downbeats":                               # bar lines: every meter-th beat on the bar grid
            node = dict(node, every=int(((an.meta.get("bars") or {}) if an else {}).get("meter") or 4), anchor="bar")
        if a.min_conf is not None:
            if not 0 <= a.min_conf <= 1:
                print("--min-conf must be within 0..1", file=sys.stderr)
                return 2
            node = dict(node, min_confidence=a.min_conf)
        times, vels = beat_times(node, ctx)
    elif src == "drum":
        from .drums import PARTS, drum_times
        if nid not in PARTS:
            print(f"--from drum:<part> needs one of {list(PARTS)}, got {nid!r}", file=sys.stderr)
            return 2
        times, vels, read = drum_times({"id": "_drum", "op": "drum", "part": nid}, ctx)
        print(f"{nid}: read from the {'drums stem' if read == 'stem' else 'mix (' + an.stems_label() + ')'}")
    else:
        print("--from must be onsets:<node>, beats[:<node>], downbeats or drum:<kick|snare|hats>", file=sys.stderr)
        return 2
    if not notes["bpm_map"] and an is not None:
        notes["bpm_map"] = an.bpm_map
        notes["grid_offset"] = an.grid_offset
    notes["tracks"][a.track] = [{"time": float(t), "vel": round(float(v), 3)} for t, v in zip(times, vels)]
    save_notes(path, notes)
    print(f"{a.track}: {len(times)} hits from {a.source} -> {path}")
    return 0


def cmd_notes_snap(a):
    if not _need_track(a):
        return 2
    rig = load_rig(a.rig)
    path = rig.notes_path
    notes = load_notes(path)
    tempo = notes_tempo(notes)
    if tempo is None:
        an = analyze(rig)
        tempo = Tempo([(m["time"], m["bpm"]) for m in an.bpm_map], an.grid_offset) if an else Tempo([(0, 120)])
    hits = notes["tracks"].get(a.track)
    if hits is None:
        print(f"no track {a.track!r}", file=sys.stderr)
        return 2
    plain = [h for h in hits if "time" in h and not h.get("from_pattern")]
    keep = [h for h in hits if "time" not in h]
    snapped, st = snap(plain, tempo, parse_grid(a.grid), a.strength)
    notes["tracks"][a.track] = snapped + keep
    save_notes(path, notes)
    print(f"{a.track}: snapped to {a.grid} (strength {a.strength}): {st['moved']} of {len(plain)} moved, "
          f"latest-to-earlier shift {st['max_earlier_ms']} ms")
    return 0


def cmd_sections(a):
    from .analysis import name_sections, section_boundaries
    rig = load_rig(a.rig)
    path = rig.notes_path
    notes = load_notes(path)
    if a.sub == "list":
        for s in notes.get("sections") or []:
            print(f"{s['time']:9.3f}s  {s['name']}")
        return 0
    if notes.get("sections") and not a.replace:
        print(f"notes already have {len(notes['sections'])} sections (use --replace to overwrite)", file=sys.stderr)
        return 2
    an = analyze(rig)
    if an is None:
        print("rig has no audio", file=sys.stderr)
        return 2
    secs, _ = section_boundaries(an, a.window_s, a.min_bars)
    named = name_sections(secs)
    notes["sections"] = named
    if not notes["bpm_map"]:
        notes["bpm_map"], notes["grid_offset"] = an.bpm_map, an.grid_offset
    save_notes(path, notes)
    for s, n in zip(secs, named):
        ch = (" rose " + ",".join(s["rose"]) if s["rose"] else "") + (" fell " + ",".join(s["fell"]) if s["fell"] else "")
        print(f"{n['time']:9.3f}s  {n['name']:<12} level {s['level']}{ch}")
    print(f"{len(named)} sections -> {path} (names are a guess from energy; edit them freely)")
    return 0


def parse_hits(spec):
    """'0:1,1.5:0.7' -> [{beat, vel}]"""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            b, _, v = part.partition(":")
            out.append({"beat": float(b), "vel": float(v) if v else 1.0})
    return out


def cmd_pattern(a):
    rig = load_rig(a.rig)
    path = rig.notes_path
    notes = load_notes(path)
    if a.sub == "list":
        for name, p in (notes.get("patterns") or {}).items():
            uses = [(t, h["at_beat"], h.get("repeat", 1)) for t, hs in notes["tracks"].items() for h in hs if h.get("pattern") == name and "time" not in h]
            print(f"{name}: {p['length_beats']} beats, hits {[(h['beat'], h['vel']) for h in p['hits']]}")
            for t, at, rep in uses:
                print(f"  placed on {t} at beat {at} x{rep}")
        return 0
    if not a.name:
        print("--name is required", file=sys.stderr)
        return 2
    if a.sub == "make":
        if a.hits is None and not a.from_track:
            print("give --hits '0:1,1.5:0.7' or --from-track TRACK --at-beat B", file=sys.stderr)
            return 2
        p = make_pattern(notes, a.name, a.length_beats, parse_hits(a.hits) if a.hits is not None else None, a.from_track, a.at_beat)
        save_notes(path, notes)
        print(f"pattern {a.name}: {p['length_beats']} beats, {len(p['hits'])} hits -> {path}")
        return 0
    if a.sub == "place":
        if not _need_track(a):
            return 2
        try:
            place_pattern(notes, a.track, a.name, a.at_beat, a.repeat, a.replace)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        save_notes(path, notes)
        n = sum(1 for h in notes["tracks"][a.track] if (h.get("from_pattern") or {}).get("pattern") == a.name)
        print(f"{a.track}: placed {a.name} at beat {a.at_beat} x{a.repeat} ({n} linked hits on this track) -> {path}")
        return 0
    return 2


def cmd_import_midi(a):
    from .midi import MidiError, import_midi
    rig = load_rig(a.rig)
    path = rig.notes_path
    notes = load_notes(path)
    try:
        res = import_midi(a.file, a.map, a.channel, a.offset_s, a.lengths)
    except (MidiError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for name, hits in res["tracks"].items():
        if a.replace or name not in notes["tracks"]:
            notes["tracks"][name] = hits
        else:
            notes["tracks"][name] = notes["tracks"][name] + hits
    if not a.keep_tempo:
        notes["bpm_map"], notes["grid_offset"] = res["bpm_map"], res["grid_offset"]
    save_notes(path, notes)
    st = res["stats"]
    print(f"midi      format {st['format']}, {st['notes']} notes ({st['skipped']} unmapped/filtered), {st['tempo_changes']} tempo segment(s)")
    if not a.keep_tempo:
        print(f"bpm_map   {res['bpm_map'][:4]}{' ...' if len(res['bpm_map']) > 4 else ''}")
    for name, hits in res["tracks"].items():
        print(f"track {name}: {len(hits)} hits" + (f", {hits[0]['time']:.3f}-{hits[-1]['time']:.3f}s" if hits else ""))
    print(f"-> {path}")
    return 0


# ---------- bake / check / stems / timeline / serve
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
    for v in res.get("safety", {}).get("violations", []):
        print(f"warning   flash safety: {v['output']} {v['channel']} {v['max_per_s']} flashes/s at {v['windows'][:3]}")
    return 0


def cmd_check(a):
    rig = load_rig(a.rig)
    notes = load_notes(rig.notes_path)
    issues = check(rig, notes)
    if not a.static and not any(i["level"] == "error" for i in issues) and (rig.audio_path or rig.data.get("frames")):
        try:                                                  # flash safety needs the baked values
            res = bake(rig, None, notes, strict=False)
            issues = check(rig, notes, bake=res)
        except (RigError, OpError) as e:
            issues.append({"level": "error", "where": "bake", "msg": str(e)})
    for i in issues:
        print(f"{i['level']:7} {i['where']}: {i['msg']}")
    n = sum(i["level"] == "error" for i in issues)
    print(f"{n} error(s), {len(issues) - n} warning(s)")
    return 1 if n else 0


def cmd_stems(a):
    from . import stems as S
    rig = load_rig(a.rig)
    if not rig.audio_path:
        print("rig has no audio", file=sys.stderr)
        return 2
    ok, why = S.available()
    if not ok:
        print(f"error: {why}", file=sys.stderr)
        return 1
    paths = S.separate(rig.audio_path, device=a.device)
    for name, p in paths.items():
        print(f"{name:7} {p}")
    return 0


def cmd_timeline(a):
    from .timeline import import_timeline
    res = import_timeline(a.file, a.audio, a.out, fps=a.fps, start_tc=a.start_tc, track=a.track, handles=a.handles,
                          source_start_tc=a.source_start_tc)
    print(f"{'#':>3} {'mix in':>10} {'mix out':>10} {'src in':>7}  clip")
    for c in res["clips"]:
        tr = "".join(f"  {side} {t['name']} {t['frames']}f" for side, t in (("in:", c["transition_in"]), ("out:", c["transition_out"])) if t)
        print(f"{c['index']:3} {c['audio_in_s']:10.3f} {c['audio_out_s']:10.3f} {c['audio_offset_frames']:>7}  {c['name']}{tr}  -> {c['rig']}")
    for w in res["warnings"]:
        print(f"warning   {w}")
    print(f"{len(res['clips'])} clip rig(s) -> {res['index']}")
    return 0


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
    p = sub.add_parser("analyze", help="build/refresh the analysis cache and print a summary (with tempo candidates)")
    p.add_argument("rig")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(fn=cmd_analyze)

    pn = sub.add_parser("notes", help="notes init|snap|show|sections|pattern|import-midi")
    ns = pn.add_subparsers(dest="action", required=True)
    p = ns.add_parser("init", help="fill a track from detected onsets or beats")
    p.add_argument("rig")
    p.add_argument("--track")
    p.add_argument("--from", dest="source", help="onsets:<node> | beats[:<node>] | downbeats (bar lines) | drum:kick|snare|hats")
    p.add_argument("--min-conf", type=float, default=None,
                   help="beats: keep only beats whose analysis confidence is at least this (0..1)")
    p.set_defaults(fn=cmd_notes_init)
    p = ns.add_parser("snap", help="quantise a track's hits to a grid")
    p.add_argument("rig")
    p.add_argument("--track")
    p.add_argument("--grid", default="1/16")
    p.add_argument("--strength", type=float, default=1.0)
    p.set_defaults(fn=cmd_notes_snap)
    p = ns.add_parser("show", help="print the notes")
    p.add_argument("rig")
    p.add_argument("--track")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(fn=cmd_notes_show)
    p = ns.add_parser("sections", help="sections init (from the analysis) | list")
    p.add_argument("sub", choices=["init", "list"])
    p.add_argument("rig")
    p.add_argument("--window-s", type=float, default=None, help="analysis window (default: one bar)")
    p.add_argument("--min-bars", type=int, default=2, help="shortest section, in windows")
    p.add_argument("--replace", action="store_true")
    p.set_defaults(fn=cmd_sections)
    p = ns.add_parser("pattern", help="pattern make | place | list (linked loops)")
    p.add_argument("sub", choices=["make", "place", "list"])
    p.add_argument("rig")
    p.add_argument("--name")
    p.add_argument("--length-beats", type=float, default=4.0)
    p.add_argument("--hits", help="make: '0:1,1.5:0.7' (beat:vel)")
    p.add_argument("--from-track", help="make: capture this track's hits from --at-beat for --length-beats")
    p.add_argument("--at-beat", type=float, default=0.0)
    p.add_argument("--track", help="place: the track to place on")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--replace", action="store_true", help="place: remove the plain hits the placement covers")
    p.set_defaults(fn=cmd_pattern)
    p = ns.add_parser("import-midi", help="import a .mid: mapped notes become tracks, the tempo map becomes the bpm map")
    p.add_argument("rig")
    p.add_argument("file")
    p.add_argument("--map", required=True, help="NOTE:track,... e.g. 36:kick,38:snare (numbers or names like C1)")
    p.add_argument("--channel", type=int, default=None, help="keep only this MIDI channel (1-16)")
    p.add_argument("--offset-s", type=float, default=0.0, help="audio time of MIDI time 0")
    p.add_argument("--lengths", action="store_true", help="also write len_beats from each note's length")
    p.add_argument("--keep-tempo", action="store_true", help="keep the notes' bpm map instead of the file's tempo map")
    p.add_argument("--replace", action="store_true", help="replace existing tracks instead of adding to them")
    p.set_defaults(fn=cmd_import_midi)

    p = sub.add_parser("bake", help="evaluate and write bake.json")
    p.add_argument("rig")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_bake)
    p = sub.add_parser("check", help="validate the rig (and, unless --static, bake in memory for the flash-safety rule)")
    p.add_argument("rig")
    p.add_argument("--static", action="store_true", help="skip the bake (no audio needed)")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("stems", help="separate drums/bass/vocals/other (needs the [stems] extra); cached next to the audio")
    p.add_argument("rig")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda when available)")
    p.set_defaults(fn=cmd_stems)
    pt = sub.add_parser("timeline", help="timeline import: one stub rig per clip of an FCPXML or EDL")
    ts = pt.add_subparsers(dest="action", required=True)
    p = ts.add_parser("import", help="the show mix must line up with the timeline: audio_in_s = the clip's timeline position, "
                                     "audio_offset_frames = its source in-point")
    p.add_argument("file", help=".fcpxml or .edl")
    p.add_argument("--audio", required=True, help="the full mix, starting at the timeline's start")
    p.add_argument("--out", required=True, help="folder for the clip rigs (shares one analysis cache)")
    p.add_argument("--fps", type=float, default=None, help="rig fps (FCPXML: from the sequence format; EDL: default 30)")
    p.add_argument("--start-tc", default=None, help="EDL: timecode of the mix's first sample (default: the hour of the first event)")
    p.add_argument("--track", default="V", help="EDL: track letter to read (default V)")
    p.add_argument("--handles", type=int, default=0,
                   help="frames baked before each clip's in point and after its out point, for dissolves (default 0)")
    p.add_argument("--source-start-tc", default=None,
                   help="EDL: timecode of the source media's first frame (default 00:00:00:00)")
    p.set_defaults(fn=cmd_timeline)
    p = sub.add_parser("serve", help="local preview page")
    p.add_argument("rig")
    p.add_argument("--port", type=int, default=4700)
    p.set_defaults(fn=cmd_serve)
    from . import live
    live.add_cli(sub)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")
    try:
        return a.fn(a)
    except (RigError, OpError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
