"""Colour signals (gradient, palette_cycle, mix/scale), colour through the channel ops, colour outputs in check and
bake (4-vector keys), and transform outputs + accumulate."""
import json
import os

import numpy as np

from helpers import ctx, tmpdir
import beatrig
from beatrig.bake import rdp
from beatrig.color import PALETTES, parse_color, srgb_to_linear
from beatrig.ops import SPECS
from beatrig.signal import Signal, empty_aux


def run(node, ins, c):
    node = dict(node, id=node.get("id", "n"))
    return SPECS[node["op"]].fn(node, ins, c)


def ramp(n=120, channels=1):
    return Signal(np.tile(np.linspace(0, 1, n, dtype=np.float32), (channels, 1)))


def test_parse_colours_and_palettes():
    assert np.allclose(parse_color("#ffffff"), [1, 1, 1, 1]) and np.allclose(parse_color("#000000"), [0, 0, 0, 1])
    assert np.allclose(parse_color("#80808080")[:3], srgb_to_linear(128 / 255)) and abs(parse_color("#80808080")[3] - 128 / 255) < 1e-9
    assert np.allclose(parse_color([0.5, 0.25, 0]), [0.5, 0.25, 0, 1])          # numbers are linear, as given
    for name in ("ember", "ice", "aurora", "candy"):
        assert name in PALETTES and len(PALETTES[name]) >= 4
    for bad in ("#12345", "red", [1, 2], [1, "a", 3]):
        try:
            parse_color(bad)
            raise AssertionError(f"accepted {bad!r}")
        except ValueError:
            pass


def test_gradient_maps_value_to_colour():
    c = ctx(frames=120)
    g = run({"op": "gradient", "stops": [[0, [0, 0, 0, 1]], [1, [1, 0.5, 0, 1]]]}, [ramp(120, 2)], c)
    assert g.kind == "color" and g.v.shape == (2, 120, 4)
    assert np.allclose(g.v[0, 0], [0, 0, 0, 1]) and np.allclose(g.v[0, -1], [1, 0.5, 0, 1])
    assert np.allclose(g.v[1, 60], [60 / 119, 30 / 119, 0, 1], atol=1e-6)
    over = run({"op": "gradient", "palette": "ice"}, [Signal(np.array([[-1.0, 2.0]], np.float32))], ctx(frames=2))
    ice = [parse_color(h) for h in PALETTES["ice"]]
    assert np.allclose(over.v[0, 0], ice[0], atol=1e-6) and np.allclose(over.v[0, 1], ice[-1], atol=1e-6)   # clamped


def test_palette_cycle_is_eased_and_periodic():
    c = ctx(frames=600, bpm=120)                              # 15 frames per beat
    pc = run({"op": "palette_cycle", "palette": "candy", "period_beats": 8, "channels": 3, "phase_step": 1 / 3}, [], c)
    assert pc.kind == "color" and pc.v.shape == (3, 600, 4)
    cols = np.array([parse_color(h) for h in PALETTES["candy"]])
    step = 8 * 15 / len(cols)                                 # frames per palette step (20)
    assert np.allclose(pc.v[0, 0], cols[0], atol=1e-6) and np.allclose(pc.v[0, int(step)], cols[1], atol=1e-6)
    assert np.allclose(pc.v[0, 120], pc.v[0, 0], atol=1e-6)   # one period = 8 beats = 120 frames
    d = np.abs(np.diff(pc.v[0].astype(np.float64), axis=0)).max(axis=1)
    assert d.max() < 0.25                                     # no jumps
    assert d[0] < d[int(step / 2)] / 5                        # slow at a palette colour, fastest mid-step (eased)
    assert np.allclose(pc.v[1, 0], pc.v[0, 40], atol=1e-5)    # channel 1 leads by a third of the cycle
    held = run({"op": "palette_cycle", "palette": "candy", "period_s": 6, "hold": 0.5}, [], ctx(frames=180))
    assert np.allclose(held.v[0, :15], cols[0], atol=1e-6)    # rests on each colour for half of its step


def test_mix_and_scale_colour():
    c = ctx(frames=4)
    a = Signal(np.tile([1.0, 0, 0, 1], (1, 4, 1)), "color")
    b = Signal(np.tile([0, 0, 1.0, 0.5], (1, 4, 1)), "color")
    m = run({"op": "mix_color", "amount": 0.25, "in": ["a", "b"]}, [a, b], c)
    assert np.allclose(m.v[0, 0], [0.75, 0, 0.25, 0.875])
    t = Signal(np.array([[0, 0.5, 1, 1]], np.float32))
    m2 = run({"op": "mix_color", "in": ["a", "b", "t"]}, [a, b, t], c)
    assert np.allclose(m2.v[0, :, 0], [1, 0.5, 0, 0]) and np.allclose(m2.v[0, :, 2], [0, 0.5, 1, 1])
    s = run({"op": "scale_color", "in": ["a", "t"], "gain": 2}, [a, t], c)
    assert np.allclose(s.v[0, :, 0], [0, 1, 2, 2]) and np.allclose(s.v[0, :, 3], 1)       # alpha kept


def test_colour_passes_through_every_colour_op():
    """every op that declares colour support keeps kind 'color' and shape [C, N, 4]"""
    c = ctx(frames=90)
    c.scene = {"actors": {f"L{i}": {"location": [100.0 * i, 0, 0]} for i in range(4)},
               "camera": {"location": [0, 0, -100], "rotation": [90, 0, 0]}}
    one = run({"op": "gradient", "palette": "ember"}, [ramp(90)], c)
    four = run({"op": "fanout", "channels": 4}, [one], c)
    cases = {
        "delay": ({"op": "delay", "ms": 50, "step_ms": 20}, four),
        "chase": ({"op": "chase", "channels": 4, "step_ms": 40}, one),
        "fanout": ({"op": "fanout", "channels": 4}, one),
        "select": ({"op": "select", "channels": [3, 1]}, four),
        "reverse": ({"op": "reverse"}, four),
        "mirror": ({"op": "mirror"}, four),
        "shuffle": ({"op": "shuffle", "seed": 3}, four),
        "smooth": ({"op": "smooth", "attack_ms": 50, "release_ms": 200}, four),
        "spatial_delay": ({"op": "spatial_delay", "actors": ["L0", "L1", "L2", "L3"], "origin": "L0", "speed_cm_s": 1000}, one),
        "dome_wave": ({"op": "dome_wave", "actors": ["L0", "L1", "L2", "L3"], "deg_per_s": 90}, one),
    }
    colour_ops = {n for n, s in SPECS.items() if s.color and s.kind != "color"}
    assert colour_ops == set(cases), colour_ops ^ set(cases)
    for name, (node, sig) in cases.items():
        out = run(node, [sig], c)
        want_c = len(node.get("channels", [])) if name == "select" else 4
        assert out.kind == "color" and out.v.shape == (want_c, 90, 4), (name, out.kind, out.v.shape)
    d = run({"op": "delay", "ms": 100}, [one], c)                     # 3 frames, exact
    assert np.array_equal(d.v[0, 3:], one.v[0, :-3]) and np.all(d.v[0, :3] == 0)


def _issues(rig, notes=None):
    return [i for i in beatrig.check(beatrig.Rig.from_dict(rig, tmpdir()), notes) if i["level"] == "error"]


def test_check_rejects_colour_where_values_belong():
    base = {"frames": 60, "nodes": [{"id": "w", "op": "lfo", "rate_hz": 1}, {"id": "col", "op": "gradient", "in": "w", "palette": "ice"}],
            "outputs": []}
    assert _issues(base) == []
    for bad in ({"id": "x", "op": "curve", "in": "col"}, {"id": "x", "op": "math", "fn": "max", "in": ["col", "w"]},
                {"id": "x", "op": "normalize", "in": "col"}, {"id": "x", "op": "gradient", "in": "col"},
                {"id": "x", "op": "scale_color", "in": ["w", "col"]}, {"id": "x", "op": "mix_color", "in": ["col", "w"]},
                {"id": "x", "op": "flash_limit", "in": "col"}):
        errs = _issues(dict(base, nodes=base["nodes"] + [bad]))
        assert errs, f"{bad['op']} accepted a colour"
    ok = dict(base, nodes=base["nodes"] + [{"id": "x", "op": "scale_color", "in": ["col", "w"]}],
              outputs=[{"id": "o", "signal": "x", "target": {"kind": "mpc", "asset": "/Game/M", "param": "Tint"}},
                       {"id": "l", "signal": "col", "target": {"kind": "light", "actors": ["A", "B"], "property": "color"}}])
    assert _issues(ok) == [], _issues(ok)
    to_tr = dict(base, outputs=[{"id": "o", "signal": "col", "target": {"kind": "transform", "actors": ["A"], "property": "scale", "axis": "x"}}])
    assert any("colour" in e["msg"] for e in _issues(to_tr))
    val_to_col = dict(base, outputs=[{"id": "o", "signal": "w", "target": {"kind": "light", "actors": ["A"], "property": "color"}}])
    assert any("colour target needs a colour signal" in e["msg"] for e in _issues(val_to_col))
    intens = dict(base, outputs=[{"id": "o", "signal": "col", "target": {"kind": "light", "actors": ["A"]}}])
    assert any("property to \"color\"" in e["msg"] for e in _issues(intens))
    bad_pal = dict(base, nodes=[{"id": "w", "op": "lfo", "rate_hz": 1}, {"id": "col", "op": "gradient", "in": "w", "palette": "nope"}])
    assert any("unknown palette" in e["msg"] for e in _issues(bad_pal))


COLOR_RIG = {"frames": 240, "fps": 30, "tempo": {"bpm": 120}, "nodes": [
    {"id": "cyc", "op": "palette_cycle", "palette": "aurora", "period_beats": 16},
    {"id": "w", "op": "lfo", "rate_beats": 8},
    {"id": "tint", "op": "scale_color", "in": ["cyc", "w"]},
    {"id": "spin", "op": "accumulate", "in": "w", "per_s": 0.5}],
    "layers": [{"id": "a", "intensity": [{"frame": 0, "value": 1}, {"frame": 120, "value": 0.5, "fade_frames": 30}]}],
    "outputs": [
        {"id": "tint", "layer": "a", "signal": "tint", "target": {"kind": "mpc", "asset": "/Game/FX/MPC_Show", "param": "Tint"}},
        {"id": "spin", "signal": "spin", "target": {"kind": "transform", "actors": ["Disc_0", "Disc_1"], "property": "rotation",
                                                     "axis": "yaw", "additive": True}, "min": 0, "max": 360}]}


def test_colour_and_transform_bake():
    d = tmpdir()
    rig = beatrig.Rig.from_dict(json.loads(json.dumps(COLOR_RIG)), d)
    r1 = beatrig.bake(rig, notes={}, out=os.path.join(d, "b1.json"))
    r2 = beatrig.bake(rig, notes={}, out=os.path.join(d, "b2.json"))
    assert open(os.path.join(d, "b1.json"), "rb").read() == open(os.path.join(d, "b2.json"), "rb").read()
    tint = r1["outputs"][0]
    assert tint["value_type"] == "color" and tint["target"]["value_type"] == "color"
    assert "min" not in tint and "max" not in tint
    ch = tint["channels"][0]
    v = np.array(ch["values"])
    k = np.array(ch["keys"])
    assert v.shape == (240, 4) and k.shape[1] == 5 and k[0, 0] == 0 and k[-1, 0] == 239
    assert len(k) < 240 / 2                                       # the RDP actually simplifies
    rng = (v.max(axis=0) - v.min(axis=0)).max()
    rec = np.stack([np.interp(np.arange(240), k[:, 0], k[:, 1 + j]) for j in range(4)], axis=1)
    assert np.max(np.abs(rec - v)) <= 0.002 * rng + 1e-9          # every component within tolerance
    sig = beatrig.evaluate(rig, None, {})["tint"].v[0].astype(np.float64)
    assert np.allclose(v[:120, :3], np.round(sig[:120, :3], 4)) and np.allclose(v[200:, :3], np.round(sig[200:, :3] * 0.5, 4))
    assert np.allclose(v[:, 3], np.round(sig[:, 3], 4))          # the layer scales RGB, not alpha
    spin = r1["outputs"][1]
    assert spin["target"] == COLOR_RIG["outputs"][1]["target"] and [c["actor"] for c in spin["channels"]] == ["Disc_0", "Disc_1"]
    sv = np.array(spin["channels"][0]["values"])
    assert sv[0] == 0 and np.all(np.diff(sv) >= -1e-9) and sv[-1] > 0     # a continuous, monotonic spin


def test_rdp_vector_tolerance():
    rng = np.random.default_rng(1)
    v = np.cumsum(rng.standard_normal((800, 4)) * 0.01, axis=0)
    for tol in (0.0, 0.01, 0.1):
        k = rdp(v, tol, [100, 400])
        rec = np.stack([np.interp(np.arange(800), k, v[k, j]) for j in range(4)], axis=1)
        assert np.max(np.abs(rec - v)) <= tol + 1e-12 and {0, 100, 400, 799} <= set(k.tolist())


def test_transform_check():
    base = {"frames": 30, "nodes": [{"id": "a", "op": "const", "value": 1}], "outputs": []}
    def out(t):
        return dict(base, outputs=[{"id": "o", "signal": "a", "target": dict({"kind": "transform", "actors": ["X"]}, **t)}])
    assert _issues(out({"property": "rotation", "axis": "yaw"})) == []
    assert _issues(out({"property": "location", "axis": "z", "additive": False})) == []
    assert any("axis" in e["msg"] for e in _issues(out({"property": "rotation", "axis": "z"})))
    assert any("property" in e["msg"] for e in _issues(out({"property": "spin", "axis": "x"})))
    assert any("additive" in e["msg"] for e in _issues(out({"property": "scale", "axis": "x", "additive": "yes"})))


def test_accumulate_is_a_causal_running_sum():
    c = ctx(frames=90, bpm=120)
    one = Signal(np.ones((1, 90), np.float32))
    a = run({"op": "accumulate", "per_s": 0.5}, [one], c)
    assert np.allclose(a.v[0], np.arange(90) * 0.5 / 30, atol=1e-6)            # value at a frame's start
    b = run({"op": "accumulate", "per_beat": 1.0, "start": 2}, [one], c)
    assert np.allclose(b.v[0, 15], 2 + 1.0, atol=1e-5)                         # 15 frames = one beat at 120 bpm
    x = np.zeros((1, 90), np.float32)
    x[0, 30] = 1
    s = run({"op": "accumulate", "per_s": 30}, [Signal(x)], c)
    assert np.all(s.v[0, :31] == 0) and np.all(s.v[0, 31:] == 1)               # frame 30's input shows from frame 31
    w = run({"op": "accumulate", "per_s": 30, "wrap": 1.0}, [Signal(np.full((1, 90), 0.75, np.float32))], c)
    assert w.v.max() < 1.0 and w.v.min() >= 0


def test_palette_cycle_phase_step_offsets_every_channel():
    """8 channels, phase_step 0.125: channel k is channel 0 shifted by k x 0.125 of the period, so the channels never all
    sit on the same colour (darkness across every light at once comes from the brightness signal, e.g. scale_color by
    a kick chase during a breakdown, not from the palette)"""
    c = ctx(frames=900, bpm=120)                              # 16 beats at 120 bpm = 8 s = 240 frames
    for node in ({"period_beats": 16}, {"period_s": 8}):
        pc = run(dict({"op": "palette_cycle", "palette": "aurora", "channels": 8, "phase_step": 0.125}, **node), [], c)
        for k in range(1, 8):
            sh = 30 * k                                       # k x 0.125 x 240 frames
            assert np.allclose(pc.v[k, :900 - sh], pc.v[0, sh:], atol=1e-6), (node, k)
        lum = pc.v[..., :3].astype(np.float64) @ [0.2126, 0.7152, 0.0722]
        assert lum.max(axis=0).min() > 0.2                    # some channel is always lit
        assert len(set(lum.argmax(axis=0)[::30].tolist())) == 8   # and which one is brightest moves round the ring
