"""Timeline import: one stub rig per clip of an edit (FCPXML or CMX3600 EDL), each a window into one long show mix.

Assumption: the mix (--audio) starts at the timeline's start, so a clip's audio_in_s is its TIMELINE position (FCPXML
`offset` minus the sequence's `tcStart`; EDL record-in minus the start timecode).

v0.3: each clip also records its SOURCE in-point (where the clip starts inside its media: FCPXML `start` minus the
asset's `start`; EDL source-in minus the source start timecode) and becomes the rig's audio_offset_frames, so the rig's
frames are media frames: frame source_in plays the clip's first timeline frame. The bake covers the clip plus `handles`
frames on each side (bake.start_frame = source_in - handles, clamped at the media's start); handle frames read the mix
beyond the cut, which is what plays under a dissolve. The transitions on either side of each clip are recorded with
the handle frames they need.

FCPXML times are rational seconds ("1001/30000s", "10s"). EDL timecodes are HH:MM:SS:FF counted at the nominal rate
(30 for 29.97); drop-frame timecode (';' or "FCM: DROP FRAME") is converted to a real frame count first. Frame counts
become seconds at the real rate (30000/1001 for 29.97).
"""
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from fractions import Fraction

CLIP_TAGS = {"asset-clip", "clip", "ref-clip", "sync-clip", "mc-clip", "video", "title", "audio"}
TC = r"(\d{1,2})[:;.](\d{2})[:;.](\d{2})([:;.])(\d{2})"
EVENT = re.compile(r"^\s*(\d{1,6})\s+(\S+)\s+(\S+)\s+(C|D|W\d+|K\S*|KB|KO)\s*(\d+)?\s+" + TC + r"\s+" + TC + r"\s+" + TC + r"\s+" + TC)
EDL_KINDS = {"C": "cut", "D": "dissolve", "W": "wipe", "K": "key"}


class TimelineError(ValueError):
    pass


def rational_seconds(s):
    """FCPXML time "1001/30000s" / "10s" / "3600/1s" -> Fraction seconds"""
    if s is None or s == "":
        return Fraction(0)
    t = str(s).strip()
    if t.endswith("s"):
        t = t[:-1]
    try:
        if "/" in t:
            a, b = t.split("/")
            return Fraction(int(a), int(b))
        return Fraction(t)
    except (ValueError, ZeroDivisionError):
        raise TimelineError(f"bad FCPXML time {s!r}") from None


def _source_in(el, assets, media):
    """where an FCPXML spine element starts inside its media (Fraction seconds from the media's own start):
    asset-clip / video / audio: start - the asset's start; ref-clip (a compound or multicam clip in <resources>):
    start - that media's sequence tcStart; clip / sync-clip: through the first child that references media, mapped
    from the clip's local timeline (child start + clip start - child offset). No start = the media's start (0); a title
    (generator) has no media and gets 0."""
    rs = rational_seconds
    start = rs(el.get("start"))
    if el.tag == "title":                                          # a generator: no media to start inside
        return Fraction(0)
    if el.get("start") is None and el.tag != "clip" and el.tag != "sync-clip":
        return Fraction(0)                                         # no start: the clip begins at its media's start
    if el.tag in ("asset-clip", "video", "audio"):
        a = assets.get(el.get("ref"))
        return start - (rs(a.get("start")) if a is not None else 0)
    if el.tag in ("ref-clip", "mc-clip"):
        m = media.get(el.get("ref"))
        inner = None if m is None else (m.find("sequence") if m.find("sequence") is not None else m.find("multicam"))
        return start - (rs(inner.get("tcStart")) if inner is not None else 0)
    if el.tag in ("clip", "sync-clip"):
        kids = [ch for ch in el if ch.get("ref") and ch.tag in ("video", "asset-clip", "ref-clip", "audio")]
        kids.sort(key=lambda ch: ch.tag != "video")                # picture first
        if kids:
            return _source_in(kids[0], assets, media) + start - rs(kids[0].get("offset"))
    return start


def parse_fcpxml(text):
    """-> {"fps": float | None, "clips": [{"name", "start_s", "end_s", "source_in" (Fraction s), "transition_in",
    "transition_out"}]} for the primary storyline of the edit's sequence. A transition is {"name", "start_s", "end_s"}
    (timeline seconds) when a <transition> sits right before (in) or after (out) the clip in the spine."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise TimelineError(f"not valid XML: {e}") from None
    formats = {f.get("id"): f for f in root.iter("format")}
    assets = {a.get("id"): a for a in root.iter("asset")}
    media = {m.get("id"): m for m in root.iter("media")}
    # the edit itself is <library>/<event>/<project>/<sequence>; a compound clip's <sequence> lives in <resources>/<media>
    # and may come first in the document, so prefer a project's sequence and only then fall back to the first one
    seq = next((p_.find("sequence") for p_ in root.iter("project") if p_.find("sequence") is not None), None)
    if seq is None:
        seq = next(root.iter("sequence"), None)
    if seq is None:
        raise TimelineError("no <sequence> in the FCPXML")
    fmt = formats.get(seq.get("format"))
    fps = None
    if fmt is not None and fmt.get("frameDuration"):
        fd = rational_seconds(fmt.get("frameDuration"))
        fps = round(float(1 / fd), 3) if fd else None
    tc0 = rational_seconds(seq.get("tcStart"))
    spine = seq.find("spine")
    if spine is None:
        raise TimelineError("the sequence has no <spine>")
    clips, prev = [], None                                         # prev = the last spine item: ("clip", c) / ("transition", t) / other
    for el in spine:
        off, dur = rational_seconds(el.get("offset")), rational_seconds(el.get("duration"))
        if el.tag == "transition":
            t = {"name": el.get("name") or "transition", "start_s": float(off - tc0), "end_s": float(off - tc0 + dur)}
            if prev is not None and prev[0] == "clip":
                prev[1]["transition_out"] = t
            prev = ("transition", t)
            continue
        if el.tag not in CLIP_TAGS:
            prev = ("other", None)
            continue
        name = el.get("name") or (assets.get(el.get("ref")).get("name") if assets.get(el.get("ref")) is not None else None)
        c = {"name": name or f"clip_{len(clips) + 1}", "start_s": float(off - tc0), "end_s": float(off - tc0 + dur),
             "source_in": _source_in(el, assets, media), "transition_in": None, "transition_out": None}
        if prev is not None and prev[0] == "transition":
            c["transition_in"] = prev[1]
        clips.append(c)
        prev = ("clip", c)
    return {"fps": fps, "clips": clips}


def real_rate(fps):
    f = float(fps)
    for nom in (24, 30, 48, 60, 120):
        if abs(f - nom * 1000 / 1001) < 5e-3:
            return Fraction(nom * 1000, 1001), nom
    return Fraction(f).limit_denominator(1001), int(round(f))


def tc_frames(h, m, s, f, nominal, drop):
    """timecode -> frame count (drop-frame: two frame numbers per minute skipped, except every tenth minute, at 30;
    four at 60)"""
    n = ((int(h) * 60 + int(m)) * 60 + int(s)) * nominal + int(f)
    if drop:
        d = 2 * nominal // 30
        mins = int(h) * 60 + int(m)
        n -= d * (mins - mins // 10)
    return n


def parse_edl(text, fps=30.0, track="V"):
    """-> {"events": [{"num", "name", "src_in", "rec_in", "rec_out" (frames), "kind", "trans_frames"}], "nominal",
    "drop"} (one entry per event number: the incoming clip of a dissolve, with the dissolve's type and length)"""
    _, nominal = real_rate(fps)
    drop_all = bool(re.search(r"FCM:\s*DROP", text, re.I))
    events, last = {}, None
    for line in text.splitlines():
        m = EVENT.match(line)
        if m:
            g = m.groups()
            num, trk = g[0], g[2]
            if track.upper() not in trk.upper():
                last = None
                continue
            src_in, rec_in, rec_out = g[5:10], g[15:20], g[20:25]
            drop = drop_all or rec_in[3] == ";"
            ri = tc_frames(rec_in[0], rec_in[1], rec_in[2], rec_in[4], nominal, drop)
            ro = tc_frames(rec_out[0], rec_out[1], rec_out[2], rec_out[4], nominal, drop)
            si = tc_frames(src_in[0], src_in[1], src_in[2], src_in[4], nominal, drop_all or src_in[3] == ";")
            prev = events.get(num)
            if prev is None or (ro - ri) >= (prev["rec_out"] - prev["rec_in"]):
                events[num] = {"num": int(num), "name": prev["name"] if prev else None, "src_in": si, "rec_in": ri,
                               "rec_out": ro, "drop": drop, "kind": EDL_KINDS.get(g[3][0], g[3]),
                               "trans_frames": int(g[4]) if g[4] and g[3] != "C" else 0}
            last = num
            continue
        cm = re.match(r"^\s*\*\s*(FROM|TO) CLIP NAME:\s*(.+?)\s*$", line, re.I)
        if cm and last in events:
            if cm.group(1).upper() == "TO" or not events[last]["name"]:
                events[last]["name"] = cm.group(2)
    evs = sorted((e for e in events.values() if e["rec_out"] > e["rec_in"]), key=lambda e: e["rec_in"])
    return {"events": evs, "nominal": nominal, "drop": drop_all or any(e["drop"] for e in evs)}


def parse_tc(tc, nominal, drop=False):
    m = re.fullmatch(TC, tc.strip())
    if not m:
        raise TimelineError(f"bad timecode {tc!r} (want HH:MM:SS:FF)")
    h, mi, s, sep, f = m.groups()
    return tc_frames(h, mi, s, f, nominal, drop or sep == ";")


def edl_clips(text, fps=30.0, start_tc=None, track="V", source_start_tc=None):
    """clips with timeline seconds, source_in (Fraction s from the source start timecode, default 00:00:00:00) and the
    dissolve / wipe on either side: an event's transition starts at its record-in and runs `frames` into it (CMX3600),
    so the outgoing clip plays on under it"""
    rate, nominal = real_rate(fps)
    e = parse_edl(text, fps, track)
    if not e["events"]:
        raise TimelineError(f"no {track} events in the EDL")
    if start_tc:
        start = parse_tc(start_tc, nominal, e["drop"])
    else:                                                    # the hour the first event sits in (01:00:00:00 usually)
        first = e["events"][0]["rec_in"]
        hour = _hour_of(first, nominal) if e["drop"] else first // (3600 * nominal)
        start = tc_frames(hour, 0, 0, 0, nominal, e["drop"])
    src0 = parse_tc(source_start_tc, nominal, e["drop"]) if source_start_tc else 0
    clips = []
    for ev in e["events"]:
        c = {"name": ev["name"] or f"event_{ev['num']:03d}", "start_s": float((ev["rec_in"] - start) / rate),
             "end_s": float((ev["rec_out"] - start) / rate), "source_in": Fraction(ev["src_in"] - src0) / rate,
             "transition_in": None, "transition_out": None}
        if ev["trans_frames"]:
            t = {"name": ev["kind"], "start_s": c["start_s"], "end_s": float((ev["rec_in"] + ev["trans_frames"] - start) / rate),
                 "frames": ev["trans_frames"]}
            c["transition_in"] = t
            out = next((p for p in reversed(clips) if abs(p["end_s"] - c["start_s"]) < 1e-9), None)
            if out is not None:
                out["transition_out"] = t
        clips.append(c)
    return {"fps": float(fps), "clips": clips}


def _hour_of(frames, nominal):
    """the timecode hour of a drop-frame frame count (an hour holds 107892 frames at 29.97 DF)"""
    per_hour = 3600 * nominal - (2 * nominal // 30) * 54
    return frames // per_hour


def slug(name):
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return (s or "clip")[:48]


def handles_needed(clip, rate):
    """[before, after] frames of media a clip's transitions use beyond its edit points"""
    def frames(sec):
        return max(0, int(math.ceil(round(sec * float(rate), 6))))
    t_in, t_out = clip.get("transition_in"), clip.get("transition_out")
    return [frames(clip["start_s"] - t_in["start_s"]) if t_in else 0, frames(t_out["end_s"] - clip["end_s"]) if t_out else 0]


def stub_rig(clip, audio_rel, fps, notes_name, offset=0, meta=None, start_frame=0):
    rig = {"description": f"timeline clip {clip['name']!r} ({clip['start_s']:.3f}-{clip['end_s']:.3f} s of the mix"
                          + (f", source in at frame {offset}" if offset else "") + ")",
           "audio": audio_rel, "fps": fps, "audio_in_s": round(clip["start_s"], 6), "audio_out_s": round(clip["out_s"], 6),
           "frames": None, "audio_offset_frames": int(offset), "notes": notes_name, "seed": 1,
           "nodes": [{"id": "loud", "op": "loudness", "attack_ms": 30, "release_ms": 400},
                     {"id": "loudN", "op": "normalize", "in": "loud", "range_db": 30},
                     {"id": "glow", "op": "smooth", "in": "loudN", "attack_ms": 150, "release_ms": 600}],
           "layers": [], "outputs": []}
    if start_frame:
        rig["bake"] = {"start_frame": int(start_frame)}
    if meta is not None:
        rig["_timeline"] = meta
    return rig


def import_timeline(path, audio, out_dir, fps=None, start_tc=None, track="V", handles=0, source_start_tc=None):
    """read an .fcpxml / .edl and write one stub rig (+ its own notes file name) per clip into out_dir, plus
    timeline.json listing them. All rigs share the mix and therefore one analysis cache in out_dir.
    Each rig: audio_in_s = the clip's timeline position, audio_offset_frames = its source in-point in frames,
    bake.start_frame = source in - `handles` (>= 0), audio_out_s = its out point + `handles` frames, so it bakes the clip
    and its handles only; "_timeline" records the source in-point, the transitions on either side and the handles they
    need. Returns {"clips", "index", "fps", "warnings"}."""
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        text = fh.read()
    handles = int(handles or 0)
    if handles < 0:
        raise TimelineError("handles must be >= 0 frames")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fcpxml", ".xml") or text.lstrip().startswith("<"):
        tl = parse_fcpxml(text)
    else:
        tl = edl_clips(text, fps or 30.0, start_tc, track, source_start_tc)
    rig_fps = fps or tl["fps"] or 30.0
    rate, _ = real_rate(rig_fps)
    os.makedirs(out_dir, exist_ok=True)
    try:
        audio_rel = os.path.relpath(os.path.abspath(audio), os.path.abspath(out_dir)).replace("\\", "/")
    except ValueError:                                        # another drive on Windows
        audio_rel = os.path.abspath(audio)
    clips, warnings = [], []
    for i, c in enumerate(tl["clips"], 1):
        if c["end_s"] <= c["start_s"] or c["end_s"] <= 0:
            continue
        src = Fraction(c.get("source_in") or 0)
        if c["start_s"] < 0:                                  # starts before the mix: the window and the media start later
            src += Fraction(-c["start_s"]).limit_denominator(10 ** 9)
        exact = src * rate
        offset = max(0, int(round(exact)))
        if abs(exact - round(exact)) > Fraction(1, 1000):
            warnings.append(f"{c['name']}: source in {float(src):.6f} s is not on a frame at {float(rate):.3f} fps; rounded to frame {offset}")
        if src < 0:
            warnings.append(f"{c['name']}: source in {float(src):.6f} s is before the media's start; using frame 0")
        c = dict(c, start_s=max(0.0, c["start_s"]))
        c["out_s"] = c["end_s"] + float(Fraction(handles) / rate)
        need = handles_needed(c, rate)
        start = max(0, offset - handles)
        base = f"{i:02d}_{slug(c['name'])}"
        meta = {"clip": c["name"], "timeline_in_s": round(c["start_s"], 6), "timeline_out_s": round(c["end_s"], 6),
                "source_in_s": round(float(Fraction(offset) / rate), 6), "source_in_frames": offset,
                "transition_in": _trans(c["transition_in"], rate), "transition_out": _trans(c["transition_out"], rate),
                "handles": {"before": offset - start, "after": handles}, "handles_needed": need}
        rig = stub_rig(c, audio_rel, rig_fps, base + ".notes.json", offset, meta, start)
        rp = os.path.join(out_dir, base + ".rig.json")
        with open(rp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rig, fh, indent=1)
            fh.write("\n")
        if need[0] > offset:
            warnings.append(f"{c['name']}: its incoming transition needs {need[0]} frames before the in point, the media has {offset}")
        elif need[0] > handles:
            warnings.append(f"{c['name']}: its incoming transition needs {need[0]} frames before the in point (--handles {need[0]})")
        if need[1] > handles:
            warnings.append(f"{c['name']}: its outgoing transition needs {need[1]} frames after the out point (--handles {need[1]})")
        clips.append({"index": i, "name": c["name"], "audio_in_s": rig["audio_in_s"], "audio_out_s": rig["audio_out_s"],
                      "audio_offset_frames": offset, "start_frame": start, "rig": os.path.basename(rp), **{k: meta[k] for k in
                      ("source_in_s", "transition_in", "transition_out", "handles", "handles_needed")}})
    idx = os.path.join(out_dir, "timeline.json")
    with open(idx, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"source": os.path.basename(path), "audio": audio_rel, "fps": rig_fps, "handles": handles, "clips": clips},
                  fh, indent=1)
        fh.write("\n")
    return {"clips": clips, "index": idx, "fps": rig_fps, "warnings": warnings}


def _trans(t, rate):
    """a transition for the rig / timeline.json: name, timeline span and length in frames"""
    if not t:
        return None
    frames = t.get("frames")
    if frames is None:
        frames = int(round(round((t["end_s"] - t["start_s"]) * float(rate), 6)))
    return {"name": t["name"], "start_s": round(t["start_s"], 6), "end_s": round(t["end_s"], 6), "frames": int(frames)}
