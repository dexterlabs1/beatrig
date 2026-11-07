"""notes.json v0.2: linked pattern loops, sections (auto from the analysis, editable) and layer keys by section."""
import importlib.util
import json
import os

import numpy as np

from helpers import ROOT, tmpdir
import beatrig
from beatrig import analysis as A
from beatrig.cli import main as cli
from beatrig.notes import expand_track, make_pattern, normalise, place_pattern, track_hits

NOTES = {"bpm_map": [{"time": 0.0, "bpm": 120.0}], "grid_offset": 0.0,
         "patterns": {"kick4": {"length_beats": 4, "hits": [{"beat": 0, "vel": 1}, {"beat": 1.5, "vel": 0.7, "len_beats": 0.25}]}},
         "sections": [{"name": "intro", "time": 0.0}, {"name": "drop", "time": 4.0}],
         "tracks": {"kick": [{"time": 0.5, "vel": 0.9}, {"pattern": "kick4", "at_beat": 4, "repeat": 2}]}}


def test_patterns_expand_save_and_follow_edits():
    d = tmpdir()
    p = os.path.join(d, "notes.json")
    beatrig.save_notes(p, NOTES)
    raw = json.load(open(p, encoding="utf-8"))
    assert raw["tracks"]["kick"] == [{"time": 0.5, "vel": 0.9}, {"pattern": "kick4", "at_beat": 4.0, "repeat": 2}]
    n = beatrig.load_notes(p)
    hits = [h for h in n["tracks"]["kick"] if "time" in h]
    assert [h["time"] for h in hits] == [0.5, 2.0, 2.75, 4.0, 4.75]                 # beat 4, 5.5, 8, 9.5 at 120 bpm
    assert all(h.get("from_pattern", {}).get("pattern") == "kick4" for h in hits[1:]) and "from_pattern" not in hits[0]
    assert hits[2]["len_beats"] == 0.25 and hits[2]["vel"] == 0.7
    b1 = open(p, "rb").read()
    beatrig.save_notes(p, n)                                                        # expanded copies are not frozen in
    assert open(p, "rb").read() == b1 and normalise(n) == normalise(NOTES)
    n["patterns"]["kick4"]["hits"][1]["beat"] = 2.0                                  # edit the pattern: every copy follows
    assert [h["time"] for h in track_hits(n, "kick")] == [0.5, 2.0, 3.0, 4.0, 5.0]  # even with stale copies in the list
    rig = beatrig.Rig.from_dict({"frames": 200, "fps": 30, "nodes": [{"id": "k", "op": "notes", "track": "kick"}], "outputs": []}, d)
    sig = beatrig.evaluate(rig, None, n)["k"]
    assert np.flatnonzero(sig.v[0]).tolist() == [15, 60, 90, 120, 150]


def test_make_and_place_patterns():
    n = {"bpm_map": [{"time": 0.0, "bpm": 120.0}], "grid_offset": 0.0,
         "tracks": {"hat": [{"time": t, "vel": 0.5} for t in np.arange(0, 4, 0.25).round(3)]}}
    p = make_pattern(n, "hat2", 2, from_track="hat", at_beat=2)
    assert [h["beat"] for h in p["hits"]] == [0.0, 0.5, 1.0, 1.5]
    place_pattern(n, "hat", "hat2", at_beat=4, repeat=2, replace=True)
    times = [h["time"] for h in n["tracks"]["hat"] if "time" in h]
    assert times == sorted(times) and len(times) == 16                               # 8 plain hits left + 8 linked
    assert sum(1 for h in n["tracks"]["hat"] if h.get("from_pattern")) == 8
    explicit = make_pattern(n, "x", 1, hits=[{"beat": 0.5, "vel": 1}])
    assert explicit == {"length_beats": 1.0, "hits": [{"beat": 0.5, "vel": 1.0}]}
    assert [h["time"] for h in expand_track({"tracks": {"t": [{"pattern": "x", "at_beat": 0, "repeat": 3}]}, "patterns": n["patterns"],
                                             "bpm_map": [], "grid_offset": 0.0}, "t")] == [0.25, 0.75, 1.25]   # no bpm map: 120


def test_layer_keys_by_section():
    d = tmpdir()
    rig = beatrig.Rig.from_dict({"frames": 300, "fps": 30, "nodes": [{"id": "one", "op": "const", "value": 1}],
                                 "layers": [{"id": "a", "intensity": [{"frame": 0, "value": 1},
                                                                      {"section": "drop", "value": 0, "fade_frames": 30}]}],
                                 "outputs": [{"id": "o", "layer": "a", "signal": "one", "target": {"kind": "mpc", "asset": "/Game/M", "param": "P"}}]}, d)
    assert [i for i in beatrig.check(rig, NOTES) if i["level"] == "error"] == []
    v = np.array(beatrig.bake(rig, notes=NOTES)["outputs"][0]["channels"][0]["values"])
    assert v[120] == 1 and 0 < v[135] < 1 and v[150] == 0                             # the fade starts at the section (4 s)
    bad = dict(NOTES, sections=[{"name": "intro", "time": 0.0}])
    assert any("section 'drop'" in i["msg"] for i in beatrig.check(rig, bad) if i["level"] == "error")
    ghost = dict(NOTES, tracks={"kick": [{"pattern": "nope", "at_beat": 0}]})
    assert any("unknown pattern" in i["msg"] for i in beatrig.check(rig, ghost))


def _demo_dir():
    spec = importlib.util.spec_from_file_location("make_demo_audio", os.path.join(ROOT, "examples", "make_demo_audio.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    d = tmpdir()
    m.make(os.path.join(d, "audio", "demo_110bpm.wav"))
    with open(os.path.join(ROOT, "examples", "ring-demo", "rig.json"), encoding="utf-8") as fh:
        rig = json.load(fh)
    with open(os.path.join(d, "rig.json"), "w", encoding="utf-8") as fh:
        json.dump(rig, fh)
    return d


def test_sections_init_and_pattern_cli_on_the_demo(capsys):
    d = _demo_dir()
    A._MEMO.clear()
    rp = os.path.join(d, "rig.json")
    assert cli(["notes", "sections", "init", rp]) == 0
    n = beatrig.load_notes(os.path.join(d, "notes.json"))
    names = [s["name"] for s in n["sections"]]
    assert names == ["intro", "breakdown", "main"], n["sections"]
    bar = 4 * 60 / 110
    t_break, t_back = n["sections"][1]["time"], n["sections"][2]["time"]
    assert 7 * bar <= t_break < 7 * bar + 0.05 and 10 * bar <= t_back < 10 * bar + 0.05   # on the bar, never early
    assert cli(["notes", "sections", "init", rp]) == 2                                     # refuses to overwrite
    assert cli(["notes", "pattern", "make", rp, "--name", "four", "--length-beats", "4", "--hits", "0:1,1:0.8,2:1,3:0.8"]) == 0
    assert cli(["notes", "pattern", "place", rp, "--name", "four", "--track", "kick", "--at-beat", "8", "--repeat", "3"]) == 0
    assert cli(["notes", "pattern", "list", rp]) == 0
    out = capsys.readouterr().out
    assert "placed on kick at beat 8.0 x3" in out
    n = beatrig.load_notes(os.path.join(d, "notes.json"))
    assert sum(1 for h in n["tracks"]["kick"] if h.get("from_pattern")) == 12
    raw = json.load(open(os.path.join(d, "notes.json"), encoding="utf-8"))
    assert raw["tracks"]["kick"] == [{"pattern": "four", "at_beat": 8.0, "repeat": 3}] and len(raw["sections"]) == 3
