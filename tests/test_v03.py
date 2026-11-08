"""v0.3 engine: output blend (replace | multiply | add), beats min_confidence + velocity, flash_limit "strongest",
rig.cache_dir, and timeline import with source in-points, transitions and handles."""
import hashlib
import json
import math
import os
from fractions import Fraction

import numpy as np

from helpers import FPS, click_times, ctx, tmpdir, write_clicks, write_loop
import beatrig
from beatrig import analysis as A
from beatrig.cli import main as cli
from beatrig.ops import SPECS, beat_times
from beatrig.safety import flash_report
from beatrig.signal import Signal, empty_aux
from beatrig.timeline import edl_clips, parse_fcpxml


def errors(rig, notes=None):
    return [i for i in beatrig.check(rig, notes or {}) if i["level"] == "error"]


def warnings(rig, notes=None):
    return [i for i in beatrig.check(rig, notes or {}) if i["level"] == "warning"]


# ---------- output blend
# Audio-free and arithmetic-only (no exp/cos/pow), so its bytes are the same on every platform. The hash is the
# v0.2.0 engine's bake.json for this rig: a rig without "blend" must keep baking byte-identically.
GOLDEN_RIG = {"fps": 30, "frames": 240, "seed": 2, "nodes": [
    {"id": "hits", "op": "notes", "track": "k"},
    {"id": "lim", "op": "flash_limit", "in": "hits", "max_flashes_per_s": 3},
    {"id": "env", "op": "envelope", "in": "lim", "shape": "saw_down", "length_ms": 300},
    {"id": "bin", "op": "envelope", "in": "hits", "shape": "binary", "length_ms": 100},
    {"id": "ring", "op": "chase", "in": "env", "channels": 4, "step_ms": 50},
    {"id": "tri", "op": "lfo", "shape": "tri", "rate_beats": 2, "channels": 4, "phase_step": 0.25},
    {"id": "mix", "op": "math", "fn": "max", "in": ["ring", "tri"]},
    {"id": "one", "op": "lfo", "shape": "tri", "rate_beats": 4},
    {"id": "col", "op": "gradient", "in": "one", "stops": [[0, [0, 0, 0.1]], [1, [1, 0.5, 0.25]]]},
    {"id": "spin", "op": "accumulate", "in": "bin", "per_s": 0.5}],
    "layers": [{"id": "a", "intensity": [{"frame": 0, "value": 1}, {"frame": 120, "value": 0.25, "fade_frames": 30}]}],
    "outputs": [
        {"id": "lights", "layer": "a", "signal": "mix", "target": {"kind": "light", "actors": ["L0", "L1", "L2", "L3"]}, "min": 0, "max": 5000},
        {"id": "glow", "signal": "env", "target": {"kind": "mpc", "asset": "/Game/M", "param": "Glow"}, "min": 0.5, "max": 1.5},
        {"id": "tint", "signal": "col", "target": {"kind": "material", "actors": ["A"], "param": "Tint", "value_type": "color"}},
        {"id": "turn", "signal": "spin", "target": {"kind": "transform", "actors": ["A"], "property": "rotation", "axis": "yaw"}, "min": 0, "max": 360}]}
GOLDEN_NOTES = {"bpm_map": [{"time": 0.0, "bpm": 120.0}], "grid_offset": 0.0,
                "tracks": {"k": [{"time": 0.25 * k + 0.01, "vel": 0.5 + 0.125 * (k % 4)} for k in range(30)]}}
GOLDEN_SHA256 = "5eb24bb9aebdc76bc184b0341f55d4cde2ed69bfb722df2ae81ba564c0315614"


def _golden(**output_extra):
    rig = json.loads(json.dumps(GOLDEN_RIG))
    for o in rig["outputs"]:
        o.update(output_extra.get(o["id"], {}))
    return rig


def test_blend_default_bakes_byte_identically_to_v02():
    d = tmpdir()
    out = os.path.join(d, "bake.json")
    res = beatrig.bake(beatrig.Rig.from_dict(_golden(), d), notes=GOLDEN_NOTES, out=out)
    data = open(out, "rb").read()
    assert hashlib.sha256(data).hexdigest() == GOLDEN_SHA256, "a rig without blend no longer bakes like v0.2.0"
    assert all("blend" not in o for o in res["outputs"])
    # an explicit "replace" is the same bake (only the rig hash, which covers the rig text, differs)
    rep = beatrig.bake(beatrig.Rig.from_dict(_golden(**{k: {"blend": "replace"} for k in ("lights", "glow", "tint", "turn")}), d),
                       notes=GOLDEN_NOTES)
    assert rep["outputs"] == res["outputs"] and rep["rig_hash"] != res["rig_hash"]


def test_blend_multiply_and_add_bake_gains_and_offsets():
    d = tmpdir()
    base = beatrig.bake(beatrig.Rig.from_dict(_golden(), d), notes=GOLDEN_NOTES)
    rig = _golden(glow={"blend": "multiply", "min": 0.6, "max": 1.55}, lights={"blend": "add", "min": -200, "max": 800},
                  tint={"blend": "multiply"})
    res = beatrig.bake(beatrig.Rig.from_dict(rig, d), notes=GOLDEN_NOTES)
    by, b0 = {o["id"]: o for o in res["outputs"]}, {o["id"]: o for o in base["outputs"]}
    assert by["glow"]["blend"] == "multiply" and by["lights"]["blend"] == "add" and by["tint"]["blend"] == "multiply"
    assert "blend" not in by["turn"]
    # values stay gains / offsets: min + v (x layer) x (max - min), exactly the replace formula on the new range
    v0 = np.array(b0["glow"]["channels"][0]["values"])                         # replace bake on 0.5..1.5
    g = np.array(by["glow"]["channels"][0]["values"])
    assert np.allclose(g, np.round(0.6 + (v0 - 0.5) * 0.95, 4), atol=1.5e-4) and g.min() >= 0.6 - 1e-9 and g.max() <= 1.55 + 1e-9
    assert by["glow"]["min"] == 0.6 and by["glow"]["max"] == 1.55
    l0, l1 = np.array(b0["lights"]["channels"][2]["values"]), np.array(by["lights"]["channels"][2]["values"])
    assert np.allclose(l1, np.round(-200 + l0 / 5000 * 1000, 4), atol=1.5e-3)
    # colour multiply: the colour values are per-component gains, baked exactly as a replace colour would be
    assert by["tint"]["channels"] == b0["tint"]["channels"] and by["tint"]["value_type"] == "color"
    assert by["glow"]["channels"][0]["keys"][0] == [0, g[0]]


def test_bake_carries_a_non_default_key_tolerance():
    d = tmpdir()
    assert "key_tolerance" not in beatrig.bake(beatrig.Rig.from_dict(_golden(), d), notes=GOLDEN_NOTES)
    same = dict(_golden(), bake={"key_tolerance": 0.002})
    assert "key_tolerance" not in beatrig.bake(beatrig.Rig.from_dict(same, d), notes=GOLDEN_NOTES)
    loose = beatrig.bake(beatrig.Rig.from_dict(dict(_golden(), bake={"key_tolerance": 0.01}), d), notes=GOLDEN_NOTES)
    assert loose["key_tolerance"] == 0.01


def test_blend_check_rejects_what_makes_no_sense():
    d = tmpdir()
    def issues(**extra):
        return beatrig.Rig.from_dict(_golden(**extra), d)
    assert errors(issues(glow={"blend": "multiply", "min": 0.6, "max": 1.5}, lights={"blend": "add", "min": 0, "max": 900})) == []
    assert any("blend 'screen'" in e["msg"] for e in errors(issues(glow={"blend": "screen"})))
    assert any("target.additive" in e["msg"] for e in errors(issues(turn={"blend": "add"})))         # transforms: additive
    assert any("target.additive" in e["msg"] for e in errors(issues(turn={"blend": "multiply"})))
    no_range = json.loads(json.dumps(GOLDEN_RIG))
    o = next(o for o in no_range["outputs"] if o["id"] == "glow")
    del o["min"], o["max"]
    o["blend"] = "multiply"
    e = errors(beatrig.Rig.from_dict(no_range, d))
    assert any("needs min and max" in x["msg"] and "gain range" in x["msg"] for x in e)
    o["blend"] = "add"
    assert any("offset range" in x["msg"] for x in errors(beatrig.Rig.from_dict(no_range, d)))
    assert errors(issues(tint={"blend": "multiply"})) == [] and errors(issues(tint={"blend": "add"})) == []   # colour: per component
    w = warnings(issues(glow={"blend": "multiply", "min": -0.5, "max": 1.5}))
    assert any("negative gain" in x["msg"] for x in w)
    nia = json.loads(json.dumps(GOLDEN_RIG))
    nia["outputs"].append({"id": "spawn", "signal": "env", "blend": "multiply", "min": 0.5, "max": 2,
                           "target": {"kind": "niagara", "actors": ["FX_0"], "param": "SpawnRate"}})
    assert errors(beatrig.Rig.from_dict(nia, d)) == []


# ---------- beats: min_confidence + velocity
class _FakeAnalysis:
    def __init__(self, beats):
        self.beats = beats


def _beat_ctx(beats, frames=300):
    c = ctx(frames=frames, bpm=120.0)
    c.analysis = _FakeAnalysis(beats)
    return c


BEATS = [{"time": 0.5 * k + 0.01, "strength": 0.8, "confidence": conf} for k, conf in
         enumerate([0.2, 0.6, 0.4, 0.9, 0.55, 0.5, 1.0, 0.1])]


def test_beats_min_confidence_and_velocity():
    c = _beat_ctx(BEATS)
    t_all, v_all = beat_times({"op": "beats"}, c)
    assert len(t_all) == 8 and np.allclose(v_all, 0.8)                            # defaults: v0.2 behaviour
    t, v = beat_times({"op": "beats", "min_confidence": 0.5}, c)
    assert np.allclose(t, [b["time"] for b in BEATS if b["confidence"] >= 0.5])   # 0.5 itself is kept
    t, v = beat_times({"op": "beats", "velocity": "confidence"}, c)
    assert np.allclose(v, [b["confidence"] for b in BEATS])
    # every/offset count the whole grid first, then weak beats drop out (bar counting never shifts)
    t, _ = beat_times({"op": "beats", "every": 2, "min_confidence": 0.5}, c)
    assert np.allclose(t, [BEATS[k]["time"] for k in (0, 2, 4, 6) if BEATS[k]["confidence"] >= 0.5])
    # subdivisions inherit their beat's confidence (and the 0.6 velocity factor)
    t, v = beat_times({"op": "beats", "subdivide": 2, "velocity": "confidence", "min_confidence": 0.85}, c)
    want = [(BEATS[3]["time"], 0.9), (BEATS[3]["time"] + 0.25, 0.9 * 0.6), (BEATS[6]["time"], 1.0), (BEATS[6]["time"] + 0.25, 0.6)]
    assert np.allclose(t, [w[0] for w in want]) and np.allclose(v, [w[1] for w in want])
    s = SPECS["beats"].fn({"id": "b", "op": "beats", "min_confidence": 0.5, "velocity": "confidence"}, [], c)
    hit = np.flatnonzero(s.v[0])
    assert hit.tolist() == [math.ceil(b["time"] * FPS) for b in BEATS if b["confidence"] >= 0.5]
    assert np.allclose(s.v[0, hit], [b["confidence"] for b in BEATS if b["confidence"] >= 0.5])


def test_beats_check_and_notes_init_min_conf():
    d = tmpdir()
    write_loop(os.path.join(d, "loop.wav"), 120)
    rig = {"audio": "loop.wav", "fps": FPS, "nodes": [{"id": "b", "op": "beats", "min_confidence": 0.95, "velocity": "confidence"}],
           "outputs": []}
    with open(os.path.join(d, "rig.json"), "w", encoding="utf-8") as fh:
        json.dump(rig, fh)
    r = beatrig.load_rig(os.path.join(d, "rig.json"))
    assert errors(r) == []
    assert errors(beatrig.Rig.from_dict(dict(rig, nodes=[{"id": "b", "op": "beats", "min_confidence": 1.5}]), d))
    assert errors(beatrig.Rig.from_dict(dict(rig, nodes=[{"id": "b", "op": "beats", "velocity": "loud"}]), d))
    A._MEMO.clear()
    an = beatrig.analyze(r)
    confs = [b["confidence"] for b in an.beats]
    keep = sum(c >= 0.95 for c in confs)
    assert 0 < keep < len(confs), confs                                           # the threshold really filters
    assert cli(["notes", "init", os.path.join(d, "rig.json"), "--track", "all", "--from", "beats"]) == 0
    assert cli(["notes", "init", os.path.join(d, "rig.json"), "--track", "sure", "--from", "beats", "--min-conf", "0.95"]) == 0
    notes = beatrig.load_notes(os.path.join(d, "notes.json"))
    assert len(notes["tracks"]["all"]) == len(confs) and len(notes["tracks"]["sure"]) == keep
    sure = {round(b["time"], 6) for b in an.beats if b["confidence"] >= 0.95}
    assert {h["time"] for h in notes["tracks"]["sure"]} == sure
    assert cli(["notes", "init", os.path.join(d, "rig.json"), "--track", "x", "--from", "beats", "--min-conf", "2"]) == 2
    sig = beatrig.evaluate(r, an, {})["b"]
    assert int(np.count_nonzero(sig.v[0])) == keep


# ---------- flash_limit strongest
def trig(spec, n, lens=None):
    """spec: {frame: velocity}"""
    v = np.zeros((1, n), np.float32)
    aux = empty_aux(v.shape)
    for f, vel in spec.items():
        v[0, f] = vel
        if lens:
            aux["len"][0, f] = lens.get(f, np.nan)
    return Signal(v, "trigger", v != 0, aux)


def lim(s, c, mode="strongest", rate=3):
    return SPECS["flash_limit"].fn({"id": "l", "op": "flash_limit", "max_flashes_per_s": rate, "mode": mode}, [s], c)


def old_pipeline_limit(spec, gap):
    """the hand-built pipeline's policy: strongest first, a hit keeps `gap` frames clear on both sides of every kept hit"""
    kept = []
    for f, vel in sorted(spec.items(), key=lambda kv: -kv[1]):
        if all(abs(f - k) >= gap for k in kept):
            kept.append(f)
    return sorted(kept)


def test_flash_limit_strongest_is_an_offline_choice_that_never_moves_a_hit():
    c = ctx(frames=120)
    s = trig({0: 0.5, 2: 0.9, 6: 0.7, 30: 0.4, 50: 0.6, 59: 0.8}, 120, lens={2: 2.0, 59: 0.25})
    out = lim(s, c)
    got = {int(f): round(float(out.v[0, f]), 4) for f in np.flatnonzero(out.v[0])}
    # 0.9 at 2 beats 0.5 at 0 and 0.7 at 6; 0.8 at 59 beats 0.6 at 50; 0.4 at 30 is clear of both: all on their own frames
    assert got == {2: 0.9, 30: 0.4, 59: 0.8}, got
    assert out.aux["len"][0, 2] == 2.0 and out.aux["len"][0, 59] == 0.25        # per-hit data stays with its hit
    assert np.isnan(out.aux["len"][0, 0]) and out.hits[0].tolist() == (out.v[0] != 0).tolist()
    first = lim(s, c, "first")
    assert np.flatnonzero(first.v[0]).tolist() == [0, 30, 50]                     # "first" keeps the first of each
    assert np.array_equal(first.v, lim(s, c, "drop").v)                           # "drop" is its v0.2 name
    tie = lim(trig({4: 0.7, 7: 0.7}, 60), ctx(frames=60))
    assert np.flatnonzero(tie.v[0]).tolist() == [4]                               # earliest on a tie
    # offline: a stronger hit after a weak one removes the weak one (the output at 0 depends on frame 5), which is why
    # this mode is exempt from the truncation rule; but the kept hit is exactly where its sound is
    assert np.flatnonzero(lim(trig({0: 0.5}, 60), ctx(frames=60)).v[0]).tolist() == [0]
    assert np.flatnonzero(lim(trig({0: 0.5, 5: 0.9}, 60), ctx(frames=60)).v[0]).tolist() == [5]


def test_flash_limit_strongest_matches_the_old_pipeline_and_is_safe():
    rng = np.random.default_rng(7)
    n = 900
    frames = sorted(set(rng.integers(0, n, 260).tolist()))
    spec = {f: round(float(rng.uniform(0.1, 1.0)), 3) for f in frames}
    src = trig(spec, n)
    for fps, rate in ((30, 3), (24, 4), (30, 2), (25, 5), (30, 2.5), (25, 3)):
        c = ctx(frames=n, fps=fps)
        out = lim(src, c, rate=rate)
        kept = np.flatnonzero(out.v[0])
        assert np.array_equal(out.v[0, kept], src.v[0, kept])                     # no hit moves or changes
        assert np.all(np.diff(kept) >= fps / rate)                                # spacing holds
        if rate == int(rate) and fps % rate == 0:                                # a whole-frame gap: the old pipeline
            assert kept.tolist() == old_pipeline_limit(spec, int(fps // rate)), (fps, rate)
        env = SPECS["envelope"].fn({"id": "e", "op": "envelope", "shape": "punch", "length_ms": 200}, [out], c)
        assert flash_report(env.v[0], fps, 0, 1, max_flashes_per_s=rate)["ok"], (fps, rate)


def test_flash_limit_fractional_rates_count_every_window():
    """2.5 per second: hits 12 frames apart would put 3 in some 1 s windows; at most floor(rate) may share one"""
    rng = np.random.default_rng(11)
    n = 900
    v = np.zeros((1, n), np.float32)
    at = rng.choice(n, 320, replace=False)
    v[0, at] = rng.uniform(0.1, 1.0, len(at))
    src = Signal(v, "trigger", v != 0, empty_aux(v.shape))
    for fps in (30, 25, 24):
        for rate in (2.5, 1.5, 3.5, 3, 4):
            k = int(math.floor(rate))
            c = ctx(frames=n, fps=fps)
            for mode in ("first", "delay", "strongest"):
                out = lim(src, c, mode, rate)
                hits = np.flatnonzero(out.v[0])
                assert len(hits) and np.all(np.diff(hits) >= fps / rate - 1e-9), (fps, rate, mode)
                assert np.all(hits[k:] - hits[:-k] >= fps - 1e-9), (fps, rate, mode)    # k + 1 hits never share 1 s
                env = SPECS["envelope"].fn({"id": "e", "op": "envelope", "shape": "punch", "length_ms": 200}, [out], c)
                assert flash_report(env.v[0], fps, 0, 1, max_flashes_per_s=rate)["ok"], (fps, rate, mode)
                if mode != "strongest":                                          # first / delay stay causal
                    for cut in (37, 250, 601):
                        part = lim(Signal(v[:, :cut], "trigger", v[:, :cut] != 0, empty_aux((1, cut))), ctx(frames=cut, fps=fps), mode, rate)
                        assert np.array_equal(part.v[0], out.v[0, :cut]), (fps, rate, mode, cut)
    # the old 12-frame spacing at 2.5 per second let 3 hits into one second; now the third waits for the window
    c = ctx(frames=120)
    reg = trig({f: 1.0 for f in range(0, 120, 12)}, 120)
    assert np.flatnonzero(lim(reg, c, "first", 2.5).v[0]).tolist() == [0, 12, 36, 48, 72, 84, 108]
    assert np.flatnonzero(lim(reg, c, "delay", 2.5).v[0]).tolist() == [0, 12, 30, 42, 60, 72, 90, 102]
    # below 1 per second no flash fits a 1 s window at all: nothing passes (and the check says so)
    assert not lim(reg, c, "first", 0.5).v.any() and not lim(reg, c, "strongest", 0.5).v.any()
    # values at a fractional rate stay safe too
    from test_safety import strobe
    x = Signal(strobe(8, 8)[None, :].astype(np.float32))
    y = SPECS["flash_limit"].fn({"id": "l", "op": "flash_limit", "max_flashes_per_s": 2.5}, [x], ctx(frames=240)).v[0]
    assert flash_report(y, FPS, 0, 1, max_flashes_per_s=2.5)["ok"]


def test_flash_limit_mode_check():
    d = tmpdir()
    base = {"fps": FPS, "frames": 90, "nodes": [{"id": "l", "op": "lfo", "rate_hz": 1},
                                               {"id": "t", "op": "threshold", "in": "l"},
                                               {"id": "f", "op": "flash_limit", "in": "t", "mode": "strongest"}], "outputs": []}
    assert errors(beatrig.Rig.from_dict(base, d)) == []
    for m in ("first", "drop", "delay"):
        base["nodes"][2]["mode"] = m
        assert errors(beatrig.Rig.from_dict(base, d)) == []
    base["nodes"][2]["mode"] = "loudest"
    assert errors(beatrig.Rig.from_dict(base, d))
    base["nodes"][2].update(mode="first", max_flashes_per_s=0.5)
    assert any("below 1" in w["msg"] for w in warnings(beatrig.Rig.from_dict(base, d)))


# ---------- rig.cache_dir
def test_cache_dir_shares_one_analysis_and_the_second_analyze_is_a_pure_read():
    root = tmpdir()
    write_clicks(os.path.join(root, "mix.wav"), click_times(12.0), 12.0)
    nodes = [{"id": "k", "op": "band", "lo": 40, "hi": 120}, {"id": "on", "op": "onsets"}]
    paths = []
    for name, (t0, t1) in (("scene_a", (0.0, 6.0)), ("scene_b", (6.0, 12.0))):
        d = os.path.join(root, "scenes", name)
        os.makedirs(d)
        p = os.path.join(d, "rig.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"audio": "../../mix.wav", "cache_dir": "../../cache", "fps": FPS, "audio_in_s": t0, "audio_out_s": t1,
                       "nodes": nodes, "outputs": []}, fh)
        paths.append(p)
    cache = os.path.join(root, "cache")
    A._MEMO.clear()
    ra = beatrig.load_rig(paths[0])
    assert ra.cache_dir == os.path.abspath(cache) and errors(ra) == []
    a = beatrig.analyze(ra)
    assert not a.loaded_from_cache and os.path.exists(os.path.join(cache, "analysis.json"))
    stamp = {f: os.stat(os.path.join(cache, f)).st_mtime_ns for f in ("analysis.json", "analysis.npz")}
    A._MEMO.clear()                                                   # a fresh process
    scans = []
    orig = A.Analysis._scan
    A.Analysis._scan = lambda self, *a_, **k: scans.append(1) or orig(self, *a_, **k)
    try:
        rb = beatrig.load_rig(paths[1])
        b = beatrig.analyze(rb)
    finally:
        A.Analysis._scan = orig
    assert b.loaded_from_cache and scans == []                        # no audio pass
    assert {f: os.stat(os.path.join(cache, f)).st_mtime_ns for f in stamp} == stamp   # and nothing written
    for p in paths:
        assert not os.path.exists(os.path.join(os.path.dirname(p), "analysis.json"))
    assert beatrig.bake(rb, b, {})["frames"] == 180
    bad = beatrig.Rig.from_dict({"audio": "x.wav", "fps": FPS, "cache_dir": 5, "nodes": [], "outputs": []}, root)
    assert any(e["where"] == "rig.cache_dir" for e in errors(bad))


# ---------- bake.start_frame
SF_NODES = [{"id": "k", "op": "band", "lo": 40, "hi": 120}, {"id": "kN", "op": "normalize", "in": "k", "range_db": 120},
            {"id": "on", "op": "onsets"}, {"id": "w", "op": "lfo", "shape": "tri", "rate_beats": 2},
            {"id": "n", "op": "notes", "track": "k"}, {"id": "pr", "op": "probability", "in": "n", "p": 0.5},
            {"id": "loud", "op": "loudness"}]
SF_NOTES = {"bpm_map": [{"time": 0.0, "bpm": 120.0}], "grid_offset": 0.0,
            "tracks": {"k": [{"time": 0.1 * k + 0.01, "vel": 0.8} for k in range(200)]}}


def _sf_rig(d, start=None, **extra):
    rig = dict({"audio": "clicks.wav", "fps": FPS, "audio_offset_frames": 120, "nodes": SF_NODES,
                "layers": [{"id": "L", "intensity": [{"frame": 0, "value": 1}, {"frame": 400, "value": 0.2, "fade_frames": 60}]}],
                "outputs": [{"id": "o", "layer": "L", "signal": "w", "target": {"kind": "light", "actors": ["L0"]}, "min": 0, "max": 10}]},
               **extra)
    if start is not None:
        rig["bake"] = {"start_frame": start}
    return beatrig.Rig.from_dict(rig, d)


def test_start_frame_bakes_only_its_range():
    d = tmpdir()
    write_clicks(os.path.join(d, "clicks.wav"), click_times(12.0), 12.0)
    A._MEMO.clear()
    full_rig, part_rig = _sf_rig(d), _sf_rig(d, 300)
    an = beatrig.analyze(full_rig)
    full, part = beatrig.evaluate(full_rig, an, SF_NOTES), beatrig.evaluate(part_rig, an, SF_NOTES)
    assert full["k"].frames == 120 + 360 and part["k"].frames == 480 - 300
    for nid in ("k", "on", "w", "n", "pr", "loud"):                  # stateless or keyed by sequence frame: identical
        assert np.array_equal(part[nid].v, full[nid].v[:, 300:]), nid
    # whole-bake statistics see only the baked range: normalize's percentiles leave out the 120 silent frames before
    # the audio and everything before frame 300
    from beatrig.analysis import norm_db
    want, _, _ = norm_db(full["k"].v[:, 300:].ravel(), 120)
    assert np.allclose(part["kN"].v.ravel(), want, atol=1e-6)
    assert not np.allclose(part["kN"].v, full["kN"].v[:, 300:], atol=1e-3)
    # layer keys are sequence frames; values / key frames index the baked range; bake.json says where it starts
    rf, rp = beatrig.bake(full_rig, an, SF_NOTES), beatrig.bake(part_rig, an, SF_NOTES)
    assert "start_frame" not in rf and rp["start_frame"] == 300 and rp["frames"] == 180
    vf, vp = rf["outputs"][0]["channels"][0]["values"], rp["outputs"][0]["channels"][0]["values"]
    assert vp == vf[300:]
    kp = rp["outputs"][0]["channels"][0]["keys"]
    assert kp[0][0] == 0 and kp[-1][0] == 179 and all(vp[f] == x for f, x in kp)
    # an explicit 0 is the default bake (only the rig hash differs)
    r0 = beatrig.bake(_sf_rig(d, 0), an, SF_NOTES)
    assert r0["outputs"] == rf["outputs"] and "start_frame" not in r0
    # explicit frames count from start_frame
    rx = beatrig.bake(_sf_rig(d, 300, frames=50), an, SF_NOTES)
    assert rx["frames"] == 50 and rx["outputs"][0]["channels"][0]["values"] == vf[300:350]


def test_start_frame_check():
    d = tmpdir()
    for bad in (-1, 1.5, "10", True):
        assert any(e["where"] == "rig.bake.start_frame" for e in errors(_sf_rig(d, bad, audio=None, frames=100)))
    assert errors(_sf_rig(d, 10, audio=None, frames=100)) == []
    write_clicks(os.path.join(d, "clicks.wav"), click_times(4.0), 4.0)
    A._MEMO.clear()
    try:
        beatrig.bake(_sf_rig(d, 5000), notes=SF_NOTES)
        raise AssertionError("a start past the end baked")
    except beatrig.RigError as e:
        assert "start_frame" in str(e)


# ---------- timeline: source in-points, transitions, handles
FCPXML_V03 = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.10">
 <resources>
  <format id="r1" name="FFVideoFormat1080p30" frameDuration="1/30s" width="1920" height="1080"/>
  <asset id="a1" name="Scene_A_render" start="3600s" duration="120s" hasVideo="1" format="r1"/>
  <asset id="a2" name="Scene_C_render" start="0s" duration="60s" hasVideo="1" format="r1"/>
  <media id="m1" name="Scene B compound"><sequence format="r1" duration="90s" tcStart="10s"><spine>
   <asset-clip ref="a2" name="Inside compound" offset="10s" start="0s" duration="90s"/></spine></sequence></media>
 </resources>
 <library><event name="Show"><project name="Show Edit">
  <sequence format="r1" duration="20s" tcStart="0s">
   <spine>
    <asset-clip ref="a1" offset="0s" name="Scene A" start="3610s" duration="4s"/>
    <transition name="Cross Dissolve" offset="7/2s" duration="1s"/>
    <ref-clip ref="m1" offset="4s" name="Scene B" start="30s" duration="4s"/>
    <transition name="Fade" offset="15/2s" duration="1s"/>
    <clip name="Scene C" offset="8s" start="5s" duration="4s"><video ref="a2" offset="2s" start="4s" duration="20s"/></clip>
    <gap name="Gap" offset="12s" duration="2s"/>
    <asset-clip ref="a2" offset="14s" name="Scene D" duration="4s"/>
    <title name="Credits" offset="18s" start="3600s" duration="2s"/>
   </spine>
  </sequence>
 </project></event></library>
</fcpxml>
"""

EDL_V03 = """TITLE: SCENES
FCM: NON-DROP FRAME

001  AX       V     C        00:00:05:00 00:00:15:00 01:00:00:00 01:00:10:00
* FROM CLIP NAME: Scene A
002  AX       V     C        00:00:15:00 00:00:15:00 01:00:10:00 01:00:10:00
002  BX       V     D    030 00:00:02:00 00:00:12:00 01:00:10:00 01:00:20:00
* FROM CLIP NAME: Scene A
* TO CLIP NAME: Scene B
003  CX       V     W001 015 00:00:00:00 00:00:05:00 01:00:20:00 01:00:25:00
* FROM CLIP NAME: Scene B
* TO CLIP NAME: Scene C
"""


def test_fcpxml_source_in_and_transitions():
    tl = parse_fcpxml(FCPXML_V03)
    by = {c["name"]: c for c in tl["clips"]}
    assert list(by) == ["Scene A", "Scene B", "Scene C", "Scene D", "Credits"]   # the compound's inner clip is not the edit
    assert by["Scene A"]["source_in"] == 10                  # asset-clip start 3610 s - asset start 3600 s
    assert by["Scene B"]["source_in"] == 20                  # ref-clip start 30 s - the compound sequence's tcStart 10 s
    assert by["Scene C"]["source_in"] == 7                   # inner video start 4 s + clip start 5 s - video offset 2 s
    assert by["Scene D"]["source_in"] == 0 and by["Credits"]["source_in"] == 0   # no start; a title has no media
    assert by["Scene A"]["transition_in"] is None and by["Scene A"]["transition_out"]["name"] == "Cross Dissolve"
    assert by["Scene B"]["transition_in"]["name"] == "Cross Dissolve" and by["Scene B"]["transition_out"]["name"] == "Fade"
    assert by["Scene C"]["transition_in"]["name"] == "Fade" and by["Scene C"]["transition_out"] is None
    assert by["Scene D"]["transition_in"] is None                                   # a gap sits between
    assert (by["Scene B"]["transition_in"]["start_s"], by["Scene B"]["transition_in"]["end_s"]) == (3.5, 4.5)
    # the v0.2 fixture: Scene A's source in is 3 frames (start 3003/30000 s), Scene B's is 0
    from test_midi_timeline import FCPXML
    old = parse_fcpxml(FCPXML)
    assert old["clips"][0]["source_in"] == Fraction(3003, 30000) and old["clips"][1]["source_in"] == 0
    assert old["clips"][1]["transition_out"]["name"] == "Cross Dissolve"


def test_edl_source_in_and_transitions():
    e = edl_clips(EDL_V03, fps=30)
    by = {c["name"]: c for c in e["clips"]}
    assert list(by) == ["Scene A", "Scene B", "Scene C"]
    assert by["Scene A"]["source_in"] == 5 and by["Scene B"]["source_in"] == 2 and by["Scene C"]["source_in"] == 0
    d = by["Scene B"]["transition_in"]
    assert d == {"name": "dissolve", "start_s": 10.0, "end_s": 11.0, "frames": 30} and by["Scene A"]["transition_out"] == d
    w = by["Scene C"]["transition_in"]
    assert w["name"] == "wipe" and w["frames"] == 15 and by["Scene B"]["transition_out"] == w
    assert by["Scene A"]["transition_in"] is None and by["Scene C"]["transition_out"] is None
    shifted = edl_clips(EDL_V03, fps=30, source_start_tc="00:00:01:00")
    assert shifted["clips"][0]["source_in"] == 4
    df = edl_clips(EDL_V03.replace("NON-DROP FRAME", "DROP FRAME"), fps=29.97)
    assert df["clips"][0]["source_in"] == Fraction(150 * 1001, 30000)            # 5 s of timecode at 29.97


def test_timeline_import_sets_offset_start_frame_and_handles():
    d = tmpdir()
    write_clicks(os.path.join(d, "mix.wav"), click_times(24.0), 24.0)
    fx = os.path.join(d, "show.fcpxml")
    with open(fx, "w", encoding="utf-8") as fh:
        fh.write(FCPXML_V03)
    plain = os.path.join(d, "plain")
    res = beatrig.import_timeline(fx, os.path.join(d, "mix.wav"), plain)
    by = {c["name"]: c for c in res["clips"]}
    assert [by[n]["audio_offset_frames"] for n in ("Scene A", "Scene B", "Scene C", "Scene D")] == [300, 600, 210, 0]
    assert [by[n]["start_frame"] for n in ("Scene A", "Scene B", "Scene C", "Scene D")] == [300, 600, 210, 0]
    assert by["Scene A"]["handles_needed"] == [0, 15] and by["Scene B"]["handles_needed"] == [15, 15]
    assert by["Scene C"]["handles_needed"] == [15, 0] and by["Scene D"]["handles_needed"] == [0, 0]
    assert any("Scene A" in w and "after" in w and "--handles 15" in w for w in res["warnings"])
    assert any("Scene C" in w and "before" in w and "--handles 15" in w for w in res["warnings"])
    rig_b = json.load(open(os.path.join(plain, by["Scene B"]["rig"]), encoding="utf-8"))
    assert rig_b["audio_in_s"] == 4.0 and rig_b["audio_out_s"] == 8.0 and rig_b["audio_offset_frames"] == 600
    assert rig_b["bake"] == {"start_frame": 600} and "bake" not in json.load(open(os.path.join(plain, by["Scene D"]["rig"]), encoding="utf-8"))
    assert rig_b["_timeline"]["source_in_s"] == 20.0 and rig_b["_timeline"]["transition_in"]["frames"] == 30
    A._MEMO.clear()
    rb = beatrig.load_rig(os.path.join(plain, by["Scene B"]["rig"]))
    res_b = beatrig.bake(rb, notes={})
    assert res_b["frames"] == 4 * FPS and res_b["start_frame"] == 600                # the clip only, from media frame 600

    out = os.path.join(d, "handles")
    assert cli(["timeline", "import", fx, "--audio", os.path.join(d, "mix.wav"), "--out", out, "--handles", "15"]) == 0
    assert beatrig.import_timeline(fx, os.path.join(d, "mix.wav"), out, handles=15)["warnings"] == []
    idx = json.load(open(os.path.join(out, "timeline.json"), encoding="utf-8"))
    assert idx["handles"] == 15
    cb = next(c for c in idx["clips"] if c["name"] == "Scene B")
    assert cb["audio_out_s"] == 8.5 and cb["start_frame"] == 585 and cb["handles"] == {"before": 15, "after": 15}
    cd = next(c for c in idx["clips"] if c["name"] == "Scene D")
    assert cd["start_frame"] == 0 and cd["handles"] == {"before": 0, "after": 15}           # clamped at the media's start
    A._MEMO.clear()
    rb = beatrig.load_rig(os.path.join(out, cb["rig"]))
    assert errors(rb) == []
    res_b = beatrig.bake(rb, notes={})
    assert res_b["frames"] == 15 + 4 * FPS + 15 and res_b["start_frame"] == 585 and res_b["audio_offset_frames"] == 600
    assert beatrig.bake(beatrig.load_rig(os.path.join(out, cd["rig"])), notes={})["frames"] == 4 * FPS + 15
    # media frame 600 + k is mix time 4 s + k / 30: identical to a plain window rig whose frame 0 is 4 s, and the 15
    # handle frames before it read the mix before the cut (what plays under the incoming dissolve)
    nodes = [{"id": "k", "op": "band", "lo": 40, "hi": 120}, {"id": "on", "op": "onsets"}]
    def sig(extra):
        r = beatrig.Rig.from_dict(dict({"audio": "../mix.wav", "fps": FPS, "nodes": nodes, "outputs": []}, **extra), out)
        return beatrig.evaluate(r, beatrig.analyze(r), {})
    media = sig({"audio_in_s": 4.0, "audio_out_s": 8.5, "audio_offset_frames": 600, "bake": {"start_frame": 585}})
    window = sig({"audio_in_s": 4.0, "audio_out_s": 8.5})
    pre = sig({"audio_in_s": 3.5, "audio_out_s": 4.0})
    assert media["k"].frames == 150
    assert np.array_equal(media["k"].v[:, 15:], window["k"].v) and np.array_equal(media["on"].v[:, 15:], window["on"].v)
    assert np.allclose(media["k"].v[:, :15], pre["k"].v[:, :15], atol=1e-7)
    assert media["on"].v[:, :15].tolist() == pre["on"].v[:, :15].tolist() and media["on"].v[:, :15].any()


def test_timeline_import_edl_offsets():
    d = tmpdir()
    write_clicks(os.path.join(d, "mix.wav"), click_times(24.0), 24.0)
    p = os.path.join(d, "show.edl")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(EDL_V03)
    res = beatrig.import_timeline(p, os.path.join(d, "mix.wav"), os.path.join(d, "clips"), fps=30, handles=30)
    by = {c["name"]: c for c in res["clips"]}
    assert [by[n]["audio_offset_frames"] for n in ("Scene A", "Scene B", "Scene C")] == [150, 60, 0]
    assert [by[n]["start_frame"] for n in ("Scene A", "Scene B", "Scene C")] == [120, 30, 0]
    assert by["Scene A"]["handles_needed"] == [0, 30] and by["Scene B"]["handles_needed"] == [0, 15]
    assert by["Scene A"]["audio_in_s"] == 0.0 and by["Scene A"]["audio_out_s"] == 11.0
    assert res["warnings"] == []
    A._MEMO.clear()
    r = beatrig.load_rig(os.path.join(d, "clips", by["Scene B"]["rig"]))
    assert errors(r) == [] and beatrig.bake(r, notes={})["frames"] == 30 + 10 * 30 + 30
