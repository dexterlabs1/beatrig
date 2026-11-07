"""Per-hit flow ops (push, shuffle_hits, every_n, probability, quantize), groups, and the spatial ops (spatial_delay,
dome_wave, ripple knn) on synthetic positions."""
import json
import os

import numpy as np

from helpers import ctx, tmpdir
import beatrig
from beatrig.ops import SPECS
from beatrig.signal import Signal, empty_aux
from beatrig.spatial import dome_angles, knn_hops, rot_axes


def run(node, ins, c):
    node = dict(node, id=node.get("id", "n"))
    return SPECS[node["op"]].fn(node, ins, c)


def trig(frames, n=300, vels=None, channels=1):
    v = np.zeros((channels, n), np.float32)
    aux = empty_aux(v.shape)
    for i, f in enumerate(frames):
        v[0, f] = 1.0 if vels is None else vels[i]
        aux["len"][0, f] = 0.25 * (i + 1)
    return Signal(v, "trigger", v != 0, aux)


def where(sig):
    return [np.flatnonzero(sig.v[c]).tolist() for c in range(sig.channels)]


def test_push_patterns():
    c = ctx(frames=100)
    hits = [10, 20, 30, 40]
    alt = run({"op": "push", "channels": 4}, [trig(hits, 100)], c)
    assert where(alt) == [[10, 30], [20, 40], [10, 30], [20, 40]]               # evens, odds, evens, odds
    assert alt.aux["len"][2, 30] == 0.75                                        # per-hit length travels with the hit
    oi = run({"op": "push", "channels": 5, "pattern": "outside_in"}, [trig(hits, 100)], c)
    assert where(oi) == [[10, 40], [20], [30], [20], [10, 40]]
    ex = run({"op": "push", "channels": 3, "pattern": [[0, 2], 1], "start": 1}, [trig(hits, 100)], c)
    assert where(ex) == [[20, 40], [10, 30], [20, 40]]


def test_shuffle_hits_bag_and_random():
    c = ctx(frames=400)
    hits = list(range(5, 395, 5))
    bag = run({"op": "shuffle_hits", "channels": 4, "seed": 9}, [trig(hits, 400)], c)
    again = run({"op": "shuffle_hits", "channels": 4, "seed": 9}, [trig(hits, 400)], c)
    assert np.array_equal(bag.v, again.v)
    order = [int(np.flatnonzero(bag.v[:, f])[0]) for f in hits]
    for r in range(len(order) // 4):
        assert sorted(order[4 * r:4 * r + 4]) == [0, 1, 2, 3]                    # each channel once per round
    assert all(a != b for a, b in zip(order, order[1:]))                        # no repeat, also across rounds
    rnd = run({"op": "shuffle_hits", "channels": 6, "mode": "random", "count": 2, "seed": 1}, [trig(hits, 400)], c)
    assert all(int((rnd.v[:, f] != 0).sum()) == 2 for f in hits)
    other = run({"op": "shuffle_hits", "channels": 4, "seed": 10}, [trig(hits, 400)], c)
    assert not np.array_equal(other.v, bag.v)


def test_every_n_and_probability():
    c = ctx(frames=1000)
    hits = list(range(3, 1000, 7))
    e = run({"op": "every_n", "n": 3, "offset": 1}, [trig(hits, 1000)], c)
    assert where(e)[0] == hits[1::3]
    p1 = run({"op": "probability", "p": 0.3, "seed": 4}, [trig(hits, 1000)], c)
    p2 = run({"op": "probability", "p": 0.3, "seed": 4}, [trig(hits, 1000)], c)
    assert np.array_equal(p1.v, p2.v)
    kept = where(p1)[0]
    assert 0.15 < len(kept) / len(hits) < 0.45 and set(kept) <= set(hits)
    fewer = run({"op": "probability", "p": 0.3, "seed": 4}, [trig(hits[::2], 1000)], c)
    assert set(where(fewer)[0]) == set(kept) & set(hits[::2])                   # each hit's draw is its own
    assert where(run({"op": "probability", "p": 1}, [trig(hits, 1000)], c))[0] == hits
    assert where(run({"op": "probability", "p": 0}, [trig(hits, 1000)], c))[0] == []


def test_quantize_only_delays_onto_the_grid():
    c = ctx(frames=300, bpm=120)                     # 1/16 = a quarter beat = 0.125 s = 3.75 frames
    hits = [10, 15, 31, 33, 60, 61, 200]
    q = run({"op": "quantize", "grid": "1/16"}, [trig(hits, 300)], c)
    got = where(q)[0]
    grid_frames = sorted({int(np.ceil(round(k * 3.75, 6))) for k in range(80)})
    assert set(got) <= set(grid_frames)
    for f in hits:                                   # each hit moves to the first grid frame at or after it
        assert min(g for g in grid_frames if g >= f) in got
    assert 15 in got and 60 in got                   # already on a grid line: stays
    half = run({"op": "quantize", "grid": "1/4", "strength": 0.5}, [trig([16], 300)], c)
    assert where(half)[0] == [23]                    # 16 -> next beat at 30, half way = 23
    # never earlier, per hit
    for f in hits:
        one = run({"op": "quantize", "grid": "1/8"}, [trig([f], 300)], c)
        assert where(one)[0][0] >= f


SCENE = {"version": 1, "units": "cm", "actors": {
    **{f"Light_{i}": {"class": "PointLight", "location": [100.0 * i, 0.0, 0.0], "rotation": [0, 0, 0],
                      "components": [{"name": "LightComponent0", "class": "PointLightComponent", "kind": "light"}]} for i in range(5)},
    "Top": {"location": [0.0, 0.0, 500.0]}, "Up": {"location": [500.0, 0.0, 100.0]}, "Right": {"location": [0.0, 500.0, 100.0]},
    "Horizon": {"location": [800.0, 0.0, 0.0]}},
    "mpcs": {"/Game/FX/MPC_Show": {"scalars": ["Glow"], "vectors": ["Tint"]}},
    "camera": {"actor": "Dome_Cam", "location": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "projection": "domemaster", "dome_angle": 180}}


def scene_ctx(frames=200, bpm=120):
    c = ctx(frames=frames, bpm=bpm)
    c.scene = json.loads(json.dumps(SCENE))
    return c


def first_hit(sig):
    return [int(np.flatnonzero(sig.v[i])[0]) if np.any(sig.v[i]) else None for i in range(sig.channels)]


def test_spatial_delay_by_distance():
    c = scene_ctx()
    lights = [f"Light_{i}" for i in range(5)]
    out = run({"op": "spatial_delay", "actors": lights, "origin": "Light_0", "speed_cm_s": 1000}, [trig([10], 200)], c)
    assert first_hit(out) == [10, 13, 16, 19, 22]                # 100 cm at 1000 cm/s = 0.1 s = 3 frames per light
    inw = run({"op": "spatial_delay", "actors": lights, "origin": "Light_0", "speed_cm_s": 1000, "direction": "in"}, [trig([10], 200)], c)
    assert first_hit(inw) == [22, 19, 16, 13, 10]
    mid = run({"op": "spatial_delay", "actors": lights, "origin": [200, 0, 0], "speed_cm_s": 1000}, [trig([10], 200)], c)
    assert first_hit(mid) == [16, 13, 10, 13, 16]
    absd = run({"op": "spatial_delay", "actors": lights, "origin": [-100, 0, 0], "speed_cm_s": 1000, "relative": False}, [trig([10], 200)], c)
    assert first_hit(absd) == [13, 16, 19, 22, 25]
    plane = run({"op": "spatial_delay", "actors": lights, "origin": [0, 0, 0], "shape": "plane", "dir": [-1, 0, 0],
                 "speed_cm_s": 1000}, [trig([10], 200)], c)
    assert first_hit(plane) == [22, 19, 16, 13, 10]              # sweep along -x: the far end goes first
    beats = run({"op": "spatial_delay", "actors": lights, "origin": "Light_0", "beats_per_m": 1}, [trig([10], 200)], c)
    assert first_hit(beats) == [10, 25, 40, 55, 70]              # 1 m = 1 beat = 15 frames at 120 bpm


def test_dome_angles_and_waves():
    fwd, right, up = rot_axes([90, 0, 0])
    assert np.allclose(fwd, [0, 0, 1]) and np.allclose(right, [0, 1, 0]) and np.allclose(up, [-1, 0, 0])
    cam = SCENE["camera"]
    pos = np.array([SCENE["actors"][a]["location"] for a in ("Top", "Horizon", "Up", "Right")])
    zen, az = dome_angles(cam, pos)
    assert abs(zen[0]) < 1e-9 and abs(zen[1] - 90) < 1e-9
    assert abs(az[2]) < 1e-6 and abs(az[3] - 90) < 1e-6          # the dome looks along the camera's UP axis; azimuth 0 = camera forward, 90 = right
    tilt = dict(cam, rotation=[-60, 0, 0])                       # pitch -60: the dome tilts toward +x, centre 30 deg above the horizon
    zt, _ = dome_angles(tilt, np.array([[np.cos(np.radians(30)), 0, np.sin(np.radians(30))], [1, 0, 0], [0, 0, 1]]) * 1000)
    assert abs(zt[0]) < 1e-6 and abs(zt[1] - 30) < 1e-6 and abs(zt[2] - 60) < 1e-6
    c = scene_ctx()
    acts = ["Top", "Horizon"]
    rings = run({"op": "dome_wave", "actors": acts, "mode": "zenith_rings", "deg_per_s": 300}, [trig([10], 200)], c)
    assert first_hit(rings) == [10, 19]                          # 90 degrees at 300 deg/s = 0.3 s = 9 frames
    rim = run({"op": "dome_wave", "actors": acts, "deg_per_s": 300, "direction": "in"}, [trig([10], 200)], c)
    assert first_hit(rim) == [19, 10]
    sweep = run({"op": "dome_wave", "actors": ["Up", "Right"], "mode": "azimuth_sweep", "deg_per_s": 300}, [trig([10], 200)], c)
    assert first_hit(sweep) == [10, 19]
    ccw = run({"op": "dome_wave", "actors": ["Up", "Right"], "mode": "azimuth_sweep", "deg_per_s": 300, "direction": "ccw"},
              [trig([10], 200)], c)
    assert first_hit(ccw) == [10, 37]                            # counter-clockwise from up, right is 270 degrees away
    beat = run({"op": "dome_wave", "actors": acts, "deg_per_beat": 45}, [trig([10], 200)], c)
    assert first_hit(beat) == [10, 40]                           # 90 deg = 2 beats = 30 frames


def test_ripple_knn_uses_real_positions():
    pos = np.array([[0, 0, 0], [100, 0, 0], [1000, 0, 0], [1100, 0, 0], [1200, 0, 0]], float)
    hop = knn_hops(pos, 1, 3)
    assert hop[0, 1] == 1 and hop[2, 3] == 1 and hop[3, 4] == 1 and hop[2, 4] == 2 and hop[0, 2] == -1
    c = scene_ctx()
    c.scene["actors"].update({f"P{i}": {"location": list(p)} for i, p in enumerate(pos)})
    v = np.zeros((5, 200), np.float32)
    v[2, 10] = 1
    r = run({"op": "ripple", "neighbours": "knn", "k": 1, "hops": 2, "actors": [f"P{i}" for i in range(5)], "step_ms": 100,
             "falloff": 0.5}, [Signal(v)], c)
    assert r.v[2, 10] == 1 and r.v[3, 13] == 0.5 and r.v[4, 16] == 0.25 and r.v[0].max() == 0 and r.v[1].max() == 0


def _errors(rig, notes=None):
    return [i for i in beatrig.check(beatrig.Rig.from_dict(rig, rig.pop("_dir", tmpdir())), notes) if i["level"] == "error"]


def test_groups_and_scene_in_check_and_bake():
    d = tmpdir()
    with open(os.path.join(d, "scene.json"), "w", encoding="utf-8") as fh:
        json.dump(SCENE, fh)
    rig = {"frames": 90, "scene": "scene.json", "tempo": {"bpm": 120}, "groups": {"row": [f"Light_{i}" for i in range(5)], "all": ["@row"]},
           "nodes": [{"id": "b", "op": "beats"}, {"id": "hits", "op": "notes", "track": "k"},
                     {"id": "wave", "op": "spatial_delay", "in": "hits", "actors": "@row", "origin": "Light_0", "speed_cm_s": 2000},
                     {"id": "env", "op": "envelope", "in": "wave", "shape": "gloss", "length_ms": 300}],
           "outputs": [{"id": "lights", "signal": "env", "target": {"kind": "light", "actors": "@all"}, "min": 0, "max": 1000},
                       {"id": "tint", "signal": "env", "target": {"kind": "mpc", "asset": "/Game/FX/MPC_Other", "param": "X"}}]}
    rig["nodes"] = [n for n in rig["nodes"] if n["id"] != "b"]
    notes = {"bpm_map": [], "grid_offset": 0, "tracks": {"k": [{"time": 0.5, "vel": 1}]}}
    r = beatrig.Rig.from_dict(json.loads(json.dumps(rig)), d)
    issues = beatrig.check(r, notes)
    errs = [i for i in issues if i["level"] == "error"]
    assert any("5 channels" in e["msg"] and "MPC" in e["msg"] for e in errs), errs      # 5-channel signal into an MPC
    assert any("MPC '/Game/FX/MPC_Other' is not in the scene scan" in i["msg"] for i in issues)
    rig["outputs"] = rig["outputs"][:1]
    r = beatrig.Rig.from_dict(json.loads(json.dumps(rig)), d)
    assert [i for i in beatrig.check(r, notes) if i["level"] == "error"] == []
    res = beatrig.bake(r, notes=notes)
    out = res["outputs"][0]
    assert out["target"]["actors"] == [f"Light_{i}" for i in range(5)] and [c["actor"] for c in out["channels"]] == out["target"]["actors"]
    firsts = [next(i for i, x in enumerate(c["values"]) if x > 0) for c in out["channels"]]
    assert firsts == [15, 17, 18, 20, 21]                      # 0.5 s, then +100 cm at 2000 cm/s = 1.5 frames each
    bad = dict(json.loads(json.dumps(rig)), _dir=d)
    bad["nodes"][1]["actors"] = "@nope"
    assert any("unknown group" in e["msg"] for e in _errors(bad, notes))
    noscene = json.loads(json.dumps(rig))
    del noscene["scene"]
    assert any("needs real positions" in e["msg"] for e in _errors(noscene, notes))
    ghost = dict(json.loads(json.dumps(rig)), _dir=d)
    ghost["groups"]["row"].append("Ghost")
    assert any("not in the scene" in e["msg"] for e in _errors(ghost, notes))


def test_new_ops_bake_byte_identical():
    """onsets -> per-hit flow (random ones included) -> spatial delay -> colour + light outputs: same bytes every time,
    from a fresh analysis and from the cache"""
    from helpers import rig_folder
    from beatrig import analysis as A
    rig = {"fps": 30, "seed": 7, "scene": "scene.json", "groups": {"row": [f"Light_{i}" for i in range(5)]}, "nodes": [
        {"id": "on", "op": "onsets"},
        {"id": "q", "op": "quantize", "in": "on", "grid": "1/8"},
        {"id": "p", "op": "probability", "in": "q", "p": 0.7},
        {"id": "sh", "op": "shuffle_hits", "in": "p", "channels": 5, "mode": "random", "count": 2},
        {"id": "pu", "op": "push", "in": "on", "channels": 5, "pattern": "outside_in"},
        {"id": "lim", "op": "flash_limit", "in": "sh", "mode": "delay"},
        {"id": "env", "op": "envelope", "in": "lim", "shape": "gloss", "length_ms": 250},
        {"id": "wave", "op": "spatial_delay", "in": "env", "actors": "@row", "origin": [0, 0, 0], "speed_cm_s": 900},
        {"id": "col", "op": "palette_cycle", "palette": "candy", "period_beats": 8, "channels": 5, "phase_step": 0.2},
        {"id": "lit", "op": "scale_color", "in": ["col", "wave"]}],
        "outputs": [{"id": "lights", "signal": "wave", "target": {"kind": "light", "actors": "@row"}, "min": 0, "max": 2000},
                    {"id": "tint", "signal": "lit", "target": {"kind": "light", "actors": "@row", "property": "color"}}]}
    d = rig_folder(rig)
    with open(os.path.join(d, "scene.json"), "w", encoding="utf-8") as fh:
        json.dump(SCENE, fh)
    paths = []
    for i, cache in enumerate((False, True, True)):
        A._MEMO.clear()
        r = beatrig.load_rig(os.path.join(d, "rig.json"))
        an = beatrig.analyze(r, use_cache=cache)
        p = os.path.join(d, f"bake{i}.json")
        beatrig.bake(r, an, {}, out=p)
        paths.append(p)
    blobs = [open(p, "rb").read() for p in paths]
    assert blobs[0] == blobs[1] == blobs[2]
    res = json.loads(blobs[0])
    assert res["outputs"][1]["value_type"] == "color" and len(res["outputs"][0]["channels"]) == 5
