"""Timeline import: one stub rig per clip of an edit (FCPXML or CMX3600 EDL), each a window into one long show mix.

Assumption: the mix (--audio) starts at the timeline's start, so a clip's audio_in_s is its TIMELINE position (FCPXML
`offset` minus the sequence's `tcStart`; EDL record-in minus the start timecode), not its source in-point.

FCPXML times are rational seconds ("1001/30000s", "10s"). EDL timecodes are HH:MM:SS:FF counted at the nominal rate
(30 for 29.97); drop-frame timecode (';' or "FCM: DROP FRAME") is converted to a real frame count first. Frame counts
become seconds at the real rate (30000/1001 for 29.97).
"""
import json
import os
import re
import xml.etree.ElementTree as ET
from fractions import Fraction

CLIP_TAGS = {"asset-clip", "clip", "ref-clip", "sync-clip", "mc-clip", "video", "title", "audio"}
TC = r"(\d{1,2})[:;.](\d{2})[:;.](\d{2})([:;.])(\d{2})"
EVENT = re.compile(r"^\s*(\d{1,6})\s+(\S+)\s+(\S+)\s+(C|D|W\d+|K\S*|KB|KO)\s*(\d+)?\s+" + TC + r"\s+" + TC + r"\s+" + TC + r"\s+" + TC)


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


def parse_fcpxml(text):
    """-> {"fps": float | None, "clips": [{"name", "start_s", "end_s"}]} for the primary storyline of the first sequence"""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise TimelineError(f"not valid XML: {e}") from None
    formats = {f.get("id"): f for f in root.iter("format")}
    assets = {a.get("id"): a for a in root.iter("asset")}
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
    clips = []
    for el in spine:
        if el.tag not in CLIP_TAGS:
            continue
        off, dur = rational_seconds(el.get("offset")), rational_seconds(el.get("duration"))
        name = el.get("name") or (assets.get(el.get("ref")).get("name") if assets.get(el.get("ref")) is not None else None)
        clips.append({"name": name or f"clip_{len(clips) + 1}", "start_s": float(off - tc0), "end_s": float(off - tc0 + dur)})
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
    """-> {"events": [{"num", "name", "rec_in", "rec_out"} in frames], "nominal", "drop"} (one entry per event number:
    the incoming clip of a dissolve)"""
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
            rec_in = g[15:20]
            rec_out = g[20:25]
            drop = drop_all or rec_in[3] == ";"
            ri = tc_frames(rec_in[0], rec_in[1], rec_in[2], rec_in[4], nominal, drop)
            ro = tc_frames(rec_out[0], rec_out[1], rec_out[2], rec_out[4], nominal, drop)
            prev = events.get(num)
            if prev is None or (ro - ri) >= (prev["rec_out"] - prev["rec_in"]):
                events[num] = {"num": int(num), "name": prev["name"] if prev else None, "rec_in": ri, "rec_out": ro, "drop": drop}
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


def edl_clips(text, fps=30.0, start_tc=None, track="V"):
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
    clips = [{"name": ev["name"] or f"event_{ev['num']:03d}", "start_s": float((ev["rec_in"] - start) / rate),
              "end_s": float((ev["rec_out"] - start) / rate)} for ev in e["events"]]
    return {"fps": float(fps), "clips": clips}


def _hour_of(frames, nominal):
    """the timecode hour of a drop-frame frame count (an hour holds 107892 frames at 29.97 DF)"""
    per_hour = 3600 * nominal - (2 * nominal // 30) * 54
    return frames // per_hour


def slug(name):
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return (s or "clip")[:48]


def stub_rig(clip, audio_rel, fps, notes_name):
    return {"description": f"timeline clip {clip['name']!r} ({clip['start_s']:.3f}-{clip['end_s']:.3f} s of the mix)",
            "audio": audio_rel, "fps": fps, "audio_in_s": round(clip["start_s"], 6), "audio_out_s": round(clip["end_s"], 6),
            "frames": None, "audio_offset_frames": 0, "notes": notes_name, "seed": 1,
            "nodes": [{"id": "loud", "op": "loudness", "attack_ms": 30, "release_ms": 400},
                      {"id": "loudN", "op": "normalize", "in": "loud", "range_db": 30},
                      {"id": "glow", "op": "smooth", "in": "loudN", "attack_ms": 150, "release_ms": 600}],
            "layers": [], "outputs": []}


def import_timeline(path, audio, out_dir, fps=None, start_tc=None, track="V"):
    """read an .fcpxml / .edl and write one stub rig (+ its own notes file name) per clip into out_dir, plus
    timeline.json listing them. All rigs share the mix and therefore one analysis cache in out_dir."""
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        text = fh.read()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fcpxml", ".xml") or text.lstrip().startswith("<"):
        tl = parse_fcpxml(text)
    else:
        tl = edl_clips(text, fps or 30.0, start_tc, track)
    rig_fps = fps or tl["fps"] or 30.0
    os.makedirs(out_dir, exist_ok=True)
    try:
        audio_rel = os.path.relpath(os.path.abspath(audio), os.path.abspath(out_dir)).replace("\\", "/")
    except ValueError:                                        # another drive on Windows
        audio_rel = os.path.abspath(audio)
    clips = []
    for i, c in enumerate(tl["clips"], 1):
        if c["end_s"] <= c["start_s"] or c["end_s"] <= 0:
            continue
        c = dict(c, start_s=max(0.0, c["start_s"]))
        base = f"{i:02d}_{slug(c['name'])}"
        rig = stub_rig(c, audio_rel, rig_fps, base + ".notes.json")
        rp = os.path.join(out_dir, base + ".rig.json")
        with open(rp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rig, fh, indent=1)
            fh.write("\n")
        clips.append({"index": i, "name": c["name"], "audio_in_s": rig["audio_in_s"], "audio_out_s": rig["audio_out_s"],
                      "rig": os.path.basename(rp)})
    idx = os.path.join(out_dir, "timeline.json")
    with open(idx, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"source": os.path.basename(path), "audio": audio_rel, "fps": rig_fps, "clips": clips}, fh, indent=1)
        fh.write("\n")
    return {"clips": clips, "index": idx, "fps": rig_fps}
