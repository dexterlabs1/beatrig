"""Determinism, key simplification, notes, check, and the ring demo end to end."""
import importlib.util
import json
import os

import numpy as np

from helpers import FPS, ROOT, rig_folder, tmpdir
import beatrig
from beatrig import analysis as A
from beatrig.bake import rdp
from beatrig.notes import normalise, parse_grid, snap
from beatrig.tempo import Tempo

LIGHTS = [f"Light_{i}" for i in range(8)]
RIG = {"fps": FPS, "seed": 4, "nodes": [
    {"id": "kick", "op": "band", "lo": 40, "hi": 120},
    {"id": "kickN", "op": "normalize", "in": "kick"},
    {"id": "kickS", "op": "smooth", "in": "kickN", "attack_ms": 15, "release_ms": 220},
    {"id": "on", "op": "onsets", "in": "kick"},
    {"id": "hits", "op": "notes", "track": "kick"},
    {"id": "pulse", "op": "envelope", "in": "hits", "shape": "punch", "length_beats": 0.5},
    {"id": "ring", "op": "chase", "in": "pulse", "channels": 8, "step_beats": 0.125},
    {"id": "cyc", "op": "cycle", "in": "on", "channels": 8, "order": "random"},
    {"id": "cycE", "op": "envelope", "in": "cyc", "shape": "gloss", "length_ms": 400},
    {"id": "mix", "op": "math", "fn": "max", "in": ["ring", "kickS", "cycE"]},
    {"id": "wave", "op": "lfo", "rate_beats": 2, "channels": 8, "phase_step": 0.125},
], "layers": [{"id": "a", "intensity": [{"frame": 0, "value": 1}, {"frame": 300, "value": 0, "fade_frames": 30}]},
              {"id": "b", "intensity": [{"frame": 0, "value": 0}, {"frame": 300, "value": 1, "fade_frames": 30}]}],
    "outputs": [
        {"id": "lights", "layer": "a", "signal": "mix", "target": {"kind": "light", "actors": LIGHTS, "property": "intensity"}, "min": 500, "max": 20000},
        {"id": "orbs", "layer": "b", "signal": "wave", "target": {"kind": "material", "actors": [f"Orb_{i}" for i in range(8)], "param": "Glow"}},
        {"id": "glow", "signal": "kickS", "target": {"kind": "mpc", "asset": "/Game/Test/MPC", "param": "Glow"}, "min": 0.2, "max": 3}]}
NOTES = {"bpm_map": [{"time": 0.0, "bpm": 120.0}], "grid_offset": 0.0,
         "tracks": {"kick": [{"time": 1.01 + 0.5 * k, "vel": 0.9, **({"len_beats": 1.0, "shape": "gloss"} if k % 4 == 0 else {})} for k in range(40)]}}


def _bake(d, name):
    A._MEMO.clear()
    rig = beatrig.load_rig(os.path.join(d, "rig.json"))
    out = os.path.join(d, name)
    return beatrig.bake(rig, out=out), out


def test_determinism_byte_identical():
    d = rig_folder(json.loads(json.dumps(RIG)), NOTES)
    r1, p1 = _bake(d, "bake1.json")              # fresh analysis
    r2, p2 = _bake(d, "bake2.json")              # from the on-disk cache
    rig = beatrig.load_rig(os.path.join(d, "rig.json"))
    A._MEMO.clear()
    an = beatrig.analyze(rig, use_cache=False)   # rebuilt from scratch again
    p3 = os.path.join(d, "bake3.json")
    beatrig.bake(rig, an, out=p3)
    b1, b2, b3 = (open(p, "rb").read() for p in (p1, p2, p3))
    assert b1 == b2 == b3, "bakes differ"
    assert len(r1["outputs"]) == 3 and r1["frames"] == 720
    # bake_from_objects gives the same structure from memory
    r4 = beatrig.bake_from_objects(json.loads(json.dumps(RIG, sort_keys=False)) | {"audio": "clicks.wav"}, NOTES, d)
    assert r4["outputs"] == r1["outputs"]


def test_bake_keys_keep_hits_and_tolerance():
    d = rig_folder(json.loads(json.dumps(RIG)), NOTES)
    res, _ = _bake(d, "bake.json")
    rig = beatrig.load_rig(os.path.join(d, "rig.json"))
    sig = beatrig.evaluate(rig, beatrig.analyze(rig), NOTES)
    for o in res["outputs"]:
        tol = 0.002 * abs(o["max"] - o["min"])
        s = sig[o["signal"]].broadcast(len(o["channels"]))
        for c, ch in enumerate(o["channels"]):
            v = np.array(ch["values"])
            k = np.array(ch["keys"])
            frames = k[:, 0].astype(int)
            assert frames[0] == 0 and frames[-1] == len(v) - 1
            assert set(np.flatnonzero(s.hits[c]).tolist()) <= set(frames.tolist()), f"{o['id']}[{c}] lost a hit frame"
            assert np.max(np.abs(np.interp(np.arange(len(v)), frames, k[:, 1]) - v)) <= tol + 1e-9
            assert np.array_equal(k[:, 1], v[frames])
    # hits from the notes reach the lights through envelope -> chase -> math
    lights = res["outputs"][0]["channels"][3]
    note_frames = [int(np.ceil((1.01 + 0.5 * k) * FPS)) for k in range(40)]
    step = 0.125 * 0.5 * FPS                                   # 0.125 beat at 120 bpm = 1.875 frames
    want = {int(np.ceil(f + 3 * step)) for f in note_frames if f + 3 * step < res["frames"]}
    assert want <= {f for f, _ in lights["keys"]}


def test_rdp_maths():
    rng = np.random.default_rng(0)
    v = np.cumsum(rng.standard_normal(5000)) * 0.01
    v[1000:1003] += 5
    forced = [17, 1000, 2500, 4321]
    for tol in (0.0, 0.002, 0.05, 1.0):
        k = rdp(v, tol, forced)
        assert k[0] == 0 and k[-1] == 4999 and set(forced) <= set(k.tolist())
        assert np.max(np.abs(np.interp(np.arange(5000), k, v[k]) - v)) <= tol + 1e-12
    assert len(rdp(v, 1.0, forced)) < len(rdp(v, 0.05, forced)) < 5000
    assert rdp(np.zeros(100), 0.01).tolist() == [0, 99]


def test_notes_round_trip_and_snap():
    d = tmpdir()
    p = os.path.join(d, "notes.json")
    n1 = beatrig.save_notes(p, NOTES)
    b1 = open(p, "rb").read()
    n2 = beatrig.load_notes(p)
    assert normalise(n2) == n1 == normalise(NOTES)
    beatrig.save_notes(p, n2)
    assert open(p, "rb").read() == b1
    beatrig.save_notes(p, {"bpm_map": [], "grid_offset": 0, "tracks": {"empty": []}})
    assert beatrig.load_notes(p)["tracks"] == {"empty": []}
    assert parse_grid("1/8") == 0.5 and parse_grid("1/4") == 1.0 and parse_grid("1/16") == 0.25 and parse_grid(2) == 2.0
    t = Tempo([(0.0, 120.0)], 0.0)
    hits = [{"time": 0.52, "vel": 1}, {"time": 0.9, "vel": 0.5}, {"time": 1.13, "vel": 1}]
    full, st = snap(hits, t, 1.0)
    assert [h["time"] for h in full] == [0.5, 1.0, 1.0] and st["moved"] == 3
    half, _ = snap(hits, t, 1.0, strength=0.5)
    assert [h["time"] for h in half] == [0.51, 0.95, 1.065]
    eighth, _ = snap(hits, Tempo([(0.0, 120.0)], 0.1), parse_grid("1/8"))    # grid offset shifts the lines
    assert [h["time"] for h in eighth] == [0.6, 0.85, 1.1]
    t2 = Tempo([(0.0, 120.0), (2.0, 60.0)], 0.0)
    assert abs(float(t2.beats_at(3.0)) - 5.0) < 1e-12 and abs(float(t2.time_of_beat(5.0)) - 3.0) < 1e-12


def _issues(rig, notes=None):
    return beatrig.check(beatrig.Rig.from_dict(rig, tmpdir()), notes)


def test_check_catches_errors():
    base = {"frames": 100, "nodes": [{"id": "a", "op": "const", "value": 1}], "outputs": []}
    assert [i for i in _issues(base) if i["level"] == "error"] == []
    cyc = dict(base, nodes=[{"id": "a", "op": "smooth", "in": "b", "attack_ms": 10, "release_ms": 10},
                            {"id": "b", "op": "smooth", "in": "a", "attack_ms": 10, "release_ms": 10}])
    assert any("cycle" in i["msg"] for i in _issues(cyc))
    unk = dict(base, nodes=[{"id": "a", "op": "wobble"}])
    assert any("unknown op" in i["msg"] for i in _issues(unk))
    par = dict(base, nodes=[{"id": "a", "op": "const", "valu": 1}])
    assert any("unknown param" in i["msg"] for i in _issues(par))
    mis = dict(base, nodes=[{"id": "w", "op": "lfo", "rate_hz": 1, "channels": 3}],
               outputs=[{"id": "o", "signal": "w", "target": {"kind": "light", "actors": LIGHTS}}])
    assert any("3 channels but the target has 8 actors" in i["msg"] for i in _issues(mis))
    mpc = dict(mis, outputs=[{"id": "o", "signal": "w", "target": {"kind": "mpc", "asset": "/x", "param": "p"}}])
    assert any("MPC param takes 1" in i["msg"] for i in _issues(mpc))
    neg = dict(base, nodes=[{"id": "a", "op": "const"}, {"id": "d", "op": "delay", "in": "a", "ms": -10}])
    assert any("rule 1" in i["msg"] for i in _issues(neg))
    off = dict(base, audio_offset_frames=-2)
    assert any("rule 1" in i["msg"] for i in _issues(off))
    sm = dict(base, nodes=[{"id": "a", "op": "const"}, {"id": "s", "op": "smooth", "in": "a", "attack_ms": 0, "release_ms": 10}])
    assert any("rule 2" in i["msg"] for i in _issues(sm))
    env = dict(base, nodes=[{"id": "a", "op": "const"}, {"id": "e", "op": "envelope", "in": "a"}])
    assert any("needs a trigger input" in i["msg"] for i in _issues(env))
    missing = dict(base, nodes=[{"id": "s", "op": "smooth", "in": "nope", "attack_ms": 1, "release_ms": 1}])
    assert any("does not exist" in i["msg"] for i in _issues(missing))
    try:
        beatrig.bake(beatrig.Rig.from_dict(unk, tmpdir()), notes={})
        raise AssertionError("bake accepted an invalid rig")
    except beatrig.RigError as e:
        assert "unknown op" in str(e)


def _demo_module():
    spec = importlib.util.spec_from_file_location("make_demo_audio", os.path.join(ROOT, "examples", "make_demo_audio.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ring_demo_end_to_end():
    """generates the synthetic demo track into a temp copy of examples/ring-demo, then analyse, notes init, bake"""
    m = _demo_module()
    d = tmpdir()
    m.make(os.path.join(d, "audio", "demo_110bpm.wav"))
    with open(os.path.join(ROOT, "examples", "ring-demo", "rig.json"), encoding="utf-8") as fh:
        rig_d = json.load(fh)
    with open(os.path.join(d, "rig.json"), "w", encoding="utf-8") as fh:
        json.dump(rig_d, fh)
    A._MEMO.clear()
    rig = beatrig.load_rig(os.path.join(d, "rig.json"))
    assert [i for i in beatrig.check(rig) if i["level"] == "error"] == []
    an = beatrig.analyze(rig)
    assert abs(an.bpm_map[0]["bpm"] - 110) < 1 and not an.meta["fallback"]
    from beatrig.bake import make_ctx
    from beatrig.ops import detect_onsets
    times, vels = detect_onsets(rig.node("kickOn"), make_ctx(rig, an, {}), rig.nodes)
    kicks = [b * 4 * 60 / 110 + k * 60 / 110 for b in range(14) if not 7 <= b <= 9 for k in range(4)]
    kicks = [t for t in kicks if t < 30]
    assert len(times) == len(kicks), (len(times), len(kicks))
    lag = times - np.array(kicks)
    assert np.all(lag >= 0) and np.all(lag < 0.035), lag     # detected after the kick, within ~1 frame
    notes = {"bpm_map": an.bpm_map, "grid_offset": an.grid_offset, "tracks": {"kick": [{"time": float(t), "vel": float(v)} for t, v in zip(times, vels)]}}
    res = beatrig.bake(rig, an, notes)
    assert res["frames"] == 900 and [o["id"] for o in res["outputs"]] == ["lights", "orbs", "glow"]


def test_rdp_simplifies_smooth_decay():
    """a smooth exponential decay keeps far fewer keys than frames, within tolerance. (A decay steep enough that its
    per-frame second difference exceeds 8 * tol / 4 legitimately needs a key on every frame: see the report.)"""
    v = 0.2 + 2.8 * np.exp(-np.arange(300) / 60.0)          # output units, range 0.2..3
    tol = 0.002 * 2.8
    k = rdp(np.round(v, 4), tol)
    assert len(k) < 300 / 4, len(k)
    assert np.max(np.abs(np.interp(np.arange(300), k, np.round(v, 4)[k]) - np.round(v, 4))) <= tol + 1e-12
    # through the real pipeline: a smooth op's output carries no forced frames
    from helpers import ctx
    from beatrig.ops import SPECS
    from beatrig.signal import Signal
    step = np.zeros(300, np.float32)
    step[:5] = 1
    s = SPECS["smooth"].fn({"id": "s", "op": "smooth", "attack_ms": 15, "release_ms": 2000}, [Signal(step)], ctx(300))
    assert not s.hits.any()
    vals = np.round(0.2 + s.v[0].astype(np.float64) * 2.8, 4)
    k2 = rdp(vals, tol, np.flatnonzero(s.hits[0]))
    assert len(k2) < 300 / 4, len(k2)


def _rdp_reference(v, tol, forced=()):
    """the plain one-span-at-a-time RDP the batched one must match frame for frame"""
    v = np.asarray(v, np.float64)
    keep = np.zeros(len(v), bool)
    keep[0] = keep[-1] = True
    keep[list(forced)] = True
    idx = np.flatnonzero(keep)
    stack = list(zip(idx[:-1].tolist(), idx[1:].tolist()))
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        t = np.arange(1, b - a, dtype=np.float64) / (b - a)
        lin = v[a] + (v[b] - v[a]) * (t[:, None] if v.ndim == 2 else t)
        err = np.abs(v[a + 1:b] - lin)
        err = err.max(axis=1) if v.ndim == 2 else err
        k = int(np.argmax(err))
        if err[k] > tol:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return np.flatnonzero(keep)


def test_batched_rdp_matches_the_reference():
    rng = np.random.default_rng(3)
    for n, dims in ((3000, None), (3000, 4), (257, None), (5, 4)):
        shape = (n,) if dims is None else (n, dims)
        v = np.round(np.cumsum(rng.standard_normal(shape) * 0.02, axis=0), 4)
        v[n // 3:n // 3 + 4] = 0.5                                       # plateaus and ties
        forced = sorted(set(rng.integers(0, n, 12).tolist()))
        for tol in (0.0, 0.003, 0.05, 2.0):
            assert np.array_equal(rdp(v, tol, forced), _rdp_reference(v, tol, forced)), (n, dims, tol)


def test_cached_analyze_is_a_pure_read():
    """editing attack/release (a new envelope) never rewrites the cache; only a new audio pass (a new band) does, and
    envelopes are never stored in it"""
    d = rig_folder({"fps": FPS, "nodes": [{"id": "k", "op": "band", "lo": 40, "hi": 120}], "outputs": []})
    A._MEMO.clear()
    npz = os.path.join(d, "analysis.npz")
    def run(nodes):
        rig = beatrig.Rig.from_dict({"audio": "clicks.wav", "fps": FPS, "nodes": nodes, "outputs": []}, d)
        an = beatrig.analyze(rig)
        beatrig.evaluate(rig, an, {})
        return os.stat(npz).st_mtime_ns
    m0 = run([{"id": "k", "op": "band", "lo": 40, "hi": 120}])
    for att in (3, 7, 3):
        assert run([{"id": "k", "op": "band", "lo": 40, "hi": 120, "attack_ms": att}]) == m0
    A._MEMO.clear()                                                   # a fresh process reads the same file
    assert run([{"id": "k", "op": "band", "lo": 40, "hi": 120, "attack_ms": 9}]) == m0
    m1 = run([{"id": "k", "op": "band", "lo": 200, "hi": 400}])       # a new band needs an audio pass: written once
    assert m1 != m0 and run([{"id": "k", "op": "band", "lo": 200, "hi": 400, "release_ms": 50}]) == m1
    with np.load(npz) as z:
        assert not [k for k in z.files if k.startswith("env:")]
