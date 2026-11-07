"""MIDI file import (pure-Python SMF), long-file windows (audio_in_s/out_s), 5.1 channel weights, FCPXML/EDL timeline
import, and tempo candidates."""
import json
import math
import os
import struct
from fractions import Fraction

import numpy as np
import soundfile as sf

from helpers import FPS, SR, click_times, rig_folder, tmpdir, write_clicks, write_loop
import beatrig
from beatrig import analysis as A
from beatrig.cli import main as cli
from beatrig.midi import MidiError, import_midi, note_number, read_smf
from beatrig.timeline import edl_clips, parse_fcpxml, rational_seconds


# ---------- MIDI
def vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def chunk(tag, body):
    return tag + struct.pack(">I", len(body)) + body


def write_mid(path, smpte=False):
    """format 1: track 0 = tempo map (120 bpm, then 60 bpm at beat 4) + a text meta; track 1 = drums on channel 10
    with running status, note-on velocity 0 as note-off, a 0x89 note-off, a sysex, and one note on channel 1"""
    tempo = lambda us: b"\xff\x51\x03" + us.to_bytes(3, "big")
    t0 = vlq(0) + tempo(500000) + vlq(0) + b"\xff\x01\x04test" + vlq(1920) + tempo(1000000) + vlq(0) + b"\xff\x2f\x00"
    t1 = (vlq(0) + b"\x99\x24\x7f"          # kick, vel 127
          + vlq(0) + b"\x90\x24\x50"        # channel 1 kick (filtered by --channel 10)
          + vlq(240) + b"\x89\x24\x00"      # kick off (explicit note-off) at tick 240
          + vlq(240) + b"\x99\x26\x40"      # snare on at 480, vel 64
          + vlq(0) + b"\xf0\x03\x7e\x7f\xf7"  # a sysex in the middle
          + vlq(120) + b"\x99\x26\x00"      # snare off (note-on vel 0) at 600
          + vlq(1320) + b"\x99\x24\x64"     # kick at 1920 (beat 4), vel 100
          + vlq(480) + b"\x24\x00"          # running status: kick off at 2400
          + vlq(0) + b"\x99\x24\x7f"        # kick at 2400 (beat 5: 2 s + 1 s at 60 bpm)
          + vlq(480) + b"\x24\x00"
          + vlq(0) + b"\x80\x24\x00"
          + vlq(0) + b"\xff\x2f\x00")
    div = struct.pack(">BB", 256 - 25, 40) if smpte else struct.pack(">H", 480)
    data = chunk(b"MThd", struct.pack(">HH", 1, 2) + div) + chunk(b"MTrk", t0) + chunk(b"MTrk", t1)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def test_smf_parser_and_tempo_map():
    d = tmpdir()
    p = write_mid(os.path.join(d, "drums.mid"))
    smf = read_smf(open(p, "rb").read())
    assert smf["format"] == 1 and smf["division"] == 480 and smf["tempos"] == [(0, 500000), (1920, 1000000)]
    notes = [(on, off, ch, n, v) for on, off, ch, n, v in smf["notes"]]
    assert (0, 240, 9, 36, 127) in notes and (480, 600, 9, 38, 64) in notes and (1920, 2400, 9, 36, 100) in notes
    res = import_midi(p, "36:kick,38:snare", channel=10, offset_s=1.0, lengths=True)
    assert [h["time"] for h in res["tracks"]["kick"]] == [1.0, 3.0, 4.0]
    assert [h["vel"] for h in res["tracks"]["kick"]] == [1.0, round(100 / 127, 4), 1.0]
    assert res["tracks"]["snare"] == [{"time": 1.5, "vel": round(64 / 127, 4), "len_beats": 0.25}]
    assert res["tracks"]["kick"][0]["len_beats"] == 0.5
    assert res["bpm_map"] == [{"time": 1.0, "bpm": 120.0}, {"time": 3.0, "bpm": 60.0}] and res["grid_offset"] == 1.0
    every = import_midi(p, {"C2": "kick"})
    assert len(every["tracks"]["kick"]) == 4                                     # channel 1's kick included
    assert note_number("C2") == 36 and note_number("D2") == 38 and note_number("F#2") == 42 and note_number("C-1") == 0
    sm = import_midi(write_mid(os.path.join(d, "smpte.mid"), smpte=True), "36:kick", channel=10)
    assert [h["time"] for h in sm["tracks"]["kick"]] == [0.0, 1.92, 2.4]         # 25 fps x 40 ticks = 1000 ticks/s
    try:
        read_smf(b"RIFF....")
        raise AssertionError("parsed a non-MIDI file")
    except MidiError:
        pass


def test_import_midi_cli_and_bake():
    d = tmpdir()
    mid = write_mid(os.path.join(d, "drums.mid"))
    with open(os.path.join(d, "rig.json"), "w", encoding="utf-8") as fh:
        json.dump({"frames": 180, "fps": 30, "nodes": [{"id": "k", "op": "notes", "track": "kick"}], "outputs": []}, fh)
    assert cli(["notes", "import-midi", os.path.join(d, "rig.json"), mid, "--map", "36:kick,38:snare", "--channel", "10"]) == 0
    n = beatrig.load_notes(os.path.join(d, "notes.json"))
    assert [h["time"] for h in n["tracks"]["kick"]] == [0.0, 2.0, 3.0] and n["bpm_map"][1] == {"time": 2.0, "bpm": 60.0}
    rig = beatrig.load_rig(os.path.join(d, "rig.json"))
    assert np.flatnonzero(beatrig.evaluate(rig, None, n)["k"].v[0]).tolist() == [0, 60, 90]


# ---------- long files, 5.1
def test_audio_window_matches_the_full_file():
    times = click_times(24.0)
    full_d = rig_folder({"fps": FPS, "nodes": [{"id": "low", "op": "band", "lo": 40, "hi": 120}, {"id": "on", "op": "onsets"}],
                         "outputs": []}, times=times)
    A._MEMO.clear()
    full_rig = beatrig.load_rig(os.path.join(full_d, "rig.json"))
    full = beatrig.evaluate(full_rig, beatrig.analyze(full_rig), {})
    win = json.loads(json.dumps(full_rig.data))
    win.update(audio_in_s=10.0, audio_out_s=16.0)
    wr = beatrig.Rig.from_dict(win, full_d)
    an = beatrig.analyze(wr)
    sig = beatrig.evaluate(wr, an, {})
    assert sig["low"].frames == 180
    assert np.array_equal(sig["low"].v[0], full["low"].v[0, 300:480])            # same causal values, shifted
    assert np.flatnonzero(sig["on"].v[0]).tolist() == [f - 300 for f in np.flatnonzero(full["on"].v[0]) if 300 <= f < 480]
    notes = {"tracks": {"k": [{"time": 9.99, "vel": 1}, {"time": 10.01, "vel": 1}, {"time": 15.99, "vel": 1}]}}
    wr2 = beatrig.Rig.from_dict(dict(win, nodes=[{"id": "k", "op": "notes", "track": "k"}]), full_d)
    k = beatrig.evaluate(wr2, an, notes)["k"]
    assert np.flatnonzero(k.v[0]).tolist() == [0, 1]                             # 9.99 s lands on frame 0 (first frame at or after); 15.99 s is past the end
    off = beatrig.Rig.from_dict(dict(win, audio_offset_frames=5, frames=None), full_d)
    assert beatrig.bake(off, an, {})["frames"] == 185 and beatrig.bake(off, an, {})["audio_in_s"] == 10.0


def test_channel_weights_5_1():
    d = tmpdir()
    n = int(8 * SR)
    x = np.zeros((n, 6))
    for t in (1.01, 2.51, 4.01, 5.51):
        i = int(t * SR)
        x[i:i + 2400, 3] = np.sin(2 * np.pi * 60 * np.arange(2400) / SR) * np.exp(-np.arange(2400) / 600)   # LFE only
    sf.write(os.path.join(d, "mix51.wav"), x * 0.8, SR, subtype="FLOAT")
    base = {"audio": "mix51.wav", "fps": FPS, "nodes": [{"id": "low", "op": "band", "lo": 40, "hi": 120}], "outputs": []}
    A._MEMO.clear()
    on = beatrig.Rig.from_dict(dict(base, channel_weights=[1, 1, 1, 0.5, 0.7, 0.7]), d)
    s_on = beatrig.evaluate(on, beatrig.analyze(on), {})["low"].v[0]
    off = beatrig.Rig.from_dict(dict(base, channel_weights=[1, 1, 1, 0, 0.7, 0.7]), d)
    a_off = beatrig.analyze(off)
    s_off = beatrig.evaluate(off, a_off, {})["low"].v[0]
    assert s_on.max() > 0.01 and s_off.max() == 0
    assert not a_off.loaded_from_cache                                             # the weights are part of the cache key
    assert [i for i in beatrig.check(beatrig.Rig.from_dict(dict(base, channel_weights=[1, -1]), d)) if i["level"] == "error"]


# ---------- timeline import
FCPXML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.10">
 <resources>
  <format id="r1" name="FFVideoFormat1080p2997" frameDuration="1001/30000s" width="1920" height="1080"/>
  <asset id="r2" name="Scene_A_render" start="0s" duration="600600/30000s" hasVideo="1" format="r1"/>
 </resources>
 <library><event name="Show"><project name="Show Edit">
  <sequence format="r1" duration="900900/30000s" tcStart="3600/1s" tcFormat="NDF">
   <spine>
    <asset-clip ref="r2" offset="3600/1s" name="Scene A" start="3003/30000s" duration="300300/30000s" format="r1"/>
    <gap name="Gap" offset="108300300/30000s" duration="150150/30000s"/>
    <clip name="Scene B" offset="108450450/30000s" duration="150150/30000s"><video ref="r2" offset="0s" duration="150150/30000s"/></clip>
    <transition name="Cross Dissolve" offset="108570570/30000s" duration="30030/30000s"/>
   </spine>
  </sequence>
 </project></event></library>
</fcpxml>
"""

EDL_NDF = """TITLE: SHOW EDIT
FCM: NON-DROP FRAME

001  AX       V     C        00:00:00:00 00:00:10:00 01:00:00:00 01:00:10:00
* FROM CLIP NAME: Scene A
002  AX       A     C        00:00:00:00 00:00:10:00 01:00:00:00 01:00:10:00
003  AX       V     C        00:00:05:00 00:00:05:00 01:00:10:00 01:00:10:00
003  BX       V     D    030 00:00:00:00 00:00:20:15 01:00:10:00 01:00:30:15
* FROM CLIP NAME: Scene A
* TO CLIP NAME: Scene B
"""

EDL_DF = """TITLE: DF
FCM: DROP FRAME
001  AX       V     C        00:00:00;00 00:01:00;02 01:00:00;00 01:01:00;02
* FROM CLIP NAME: Long Take
"""


def test_fcpxml_and_edl_parsing():
    assert rational_seconds("1001/30000s") == Fraction(1001, 30000) and rational_seconds("10s") == 10 and rational_seconds("0s") == 0
    tl = parse_fcpxml(FCPXML)
    assert tl["fps"] == 29.97
    assert [c["name"] for c in tl["clips"]] == ["Scene A", "Scene B"]
    assert tl["clips"][0]["start_s"] == 0.0 and abs(tl["clips"][0]["end_s"] - 10.01) < 1e-9          # timeline position, not source start
    assert abs(tl["clips"][1]["start_s"] - 15.015) < 1e-9 and abs(tl["clips"][1]["end_s"] - 20.02) < 1e-9
    e = edl_clips(EDL_NDF, fps=29.97)
    rate = 30000 / 1001
    assert [c["name"] for c in e["clips"]] == ["Scene A", "Scene B"]
    assert e["clips"][0]["start_s"] == 0.0 and abs(e["clips"][0]["end_s"] - 300 / rate) < 1e-9
    assert abs(e["clips"][1]["start_s"] - 300 / rate) < 1e-9 and abs(e["clips"][1]["end_s"] - 915 / rate) < 1e-9
    df = edl_clips(EDL_DF, fps=29.97)
    assert abs(df["clips"][0]["end_s"] - 1800 / rate) < 1e-9                       # 00:01:00;02 is frame 1800 in drop-frame
    e30 = edl_clips(EDL_NDF, fps=30, start_tc="00:59:50:00")
    assert e30["clips"][0]["start_s"] == 10.0


def test_fcpxml_prefers_the_project_sequence_over_a_compound_clip():
    # a compound clip's <sequence> sits in <resources> and comes first in the document; the edit is the project's
    compound = FCPXML.replace("<resources>", """<resources>
    <media id="m1" name="Compound Clip 1"><sequence format="r1" duration="2400/24s"><spine>
      <asset-clip ref="a1" name="Inside compound" offset="0s" duration="2400/24s"/></spine></sequence></media>""", 1)
    assert compound != FCPXML
    tl = parse_fcpxml(compound)
    assert [c["name"] for c in tl["clips"]] == ["Scene A", "Scene B"]


def test_time_to_frame_survives_microsecond_storage():
    from beatrig.analysis import time_to_frame
    for f in range(1, 400):
        t = round(f / 30.0, 6)                      # what notes.json stores for a hit exactly on frame f's start
        assert int(time_to_frame(t, 30)) == f


def test_timeline_import_cli_writes_window_rigs():
    d = tmpdir()
    write_clicks(os.path.join(d, "mix.wav"), click_times(24.0), 24.0)
    with open(os.path.join(d, "show.fcpxml"), "w", encoding="utf-8") as fh:
        fh.write(FCPXML)
    out = os.path.join(d, "clips")
    assert cli(["timeline", "import", os.path.join(d, "show.fcpxml"), "--audio", os.path.join(d, "mix.wav"), "--out", out]) == 0
    idx = json.load(open(os.path.join(out, "timeline.json"), encoding="utf-8"))
    assert [c["rig"] for c in idx["clips"]] == ["01_Scene_A.rig.json", "02_Scene_B.rig.json"]
    A._MEMO.clear()
    rig = beatrig.load_rig(os.path.join(out, "02_Scene_B.rig.json"))
    assert rig.audio_in_s == 15.015 and rig.audio_out_s == 20.02 and rig.data["audio"] == "../mix.wav" and rig.fps == 29.97
    assert [i for i in beatrig.check(rig) if i["level"] == "error"] == []
    res = beatrig.bake(rig, notes={})
    assert res["frames"] == math.ceil(round(5.005 * 29.97, 6))
    assert os.path.exists(os.path.join(out, "analysis.json"))                      # one cache for every clip rig


# ---------- tempo candidates
def test_tempo_candidates_and_pinning():
    d = tmpdir()
    write_loop(os.path.join(d, "loop.wav"), 120)
    A._MEMO.clear()
    an = A.open_analysis(os.path.join(d, "loop.wav"), d)
    c = an.meta["tempo_candidates"]
    assert {x["ratio"] for x in c} == {"x1", "x2", "x1/2", "x3/2", "x2/3", "x4/3", "x3/4"}
    main = next(x for x in c if x["ratio"] == "x1")
    assert abs(main["bpm"] - 120) < 0.5 and max(x["confidence"] for x in c) == 1.0
    assert all(0 <= x["confidence"] <= 1 for x in c)
    by = {x["ratio"]: x for x in c}
    assert by["x1"]["confidence"] > by["x3/2"]["confidence"] and by["x1"]["confidence"] > by["x4/3"]["confidence"]
    assert an.summary()["tempo_candidates"] == c
    rig = beatrig.Rig.from_dict({"audio": "loop.wav", "fps": 30, "tempo": {"bpm": by["x4/3"]["bpm"]},
                                 "nodes": [{"id": "l", "op": "lfo", "rate_beats": 1}], "outputs": []}, d)
    from beatrig.bake import resolve_tempo
    assert abs(float(resolve_tempo(rig, an, {}).bpm_at(1.0)) - by["x4/3"]["bpm"]) < 1e-9     # a candidate pinned
