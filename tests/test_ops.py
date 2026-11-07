"""Op maths on synthetic signals (no audio): chase, cycle, ripple, envelope, smooth, delay, layers."""
import numpy as np

from helpers import ctx
from beatrig.ops import SPECS
from beatrig.signal import Signal, empty_aux
from beatrig.bake import layer_curve


def run(node, ins, c):
    node = dict(node, id=node.get("id", "n"))
    return SPECS[node["op"]].fn(node, ins, c)


def trig(frames, n=300, vel=1.0, channels=1):
    v = np.zeros((channels, n), np.float32)
    for f in frames:
        v[0, f] = vel
    return Signal(v, "trigger", v != 0, empty_aux(v.shape))


def test_chase_is_channel0_delayed_by_i_steps():
    c = ctx(frames=300, bpm=120)
    env = run({"op": "envelope", "shape": "gloss", "length_beats": 0.5}, [trig([10, 50, 51, 120])], c)
    for params, step in (({"step_beats": 0.2}, 3), ({"step_ms": 100}, 3), ({"step_ms": 50}, 1.5)):
        ring = run(dict({"op": "chase", "channels": 6}, **params), [env], c)
        assert ring.v.shape == (6, 300)
        for i in range(6):
            d = step * i
            if float(d).is_integer():
                d = int(d)
                want = np.concatenate((np.zeros(d), env.v[0, :300 - d]))
                assert np.array_equal(ring.v[i], want.astype(np.float32)), (params, i)
            else:                                  # fractional delay: linear interpolation between two past frames
                f = np.arange(300) - d
                want = np.interp(f, np.arange(300), env.v[0], left=0)
                assert np.allclose(ring.v[i], want, atol=1e-6), (params, i)
        # hit frames move with the chase
        assert np.flatnonzero(ring.hits[2]).tolist() == [int(np.ceil(f + 2 * step)) for f in (10, 50, 51, 120)]


def test_chase_direction_and_trigger_ceil():
    c = ctx(frames=100)
    ring = run({"op": "chase", "channels": 4, "step_ms": 50, "direction": -1}, [trig([10], n=100)], c)
    # wrap: channel 0 leads, then 3, 2, 1; 1.5-frame steps round UP for triggers (never early)
    assert [int(np.flatnonzero(ring.v[i])[0]) for i in range(4)] == [10, 15, 13, 12]
    ring = run({"op": "chase", "channels": 4, "step_ms": 50, "direction": -1, "wrap": False}, [trig([10], n=100)], c)
    assert [int(np.flatnonzero(ring.v[i])[0]) for i in range(4)] == [15, 13, 12, 10]


def test_cycle_orders():
    c = ctx(frames=100)
    hits = [10, 20, 30, 40, 50]
    seq = run({"op": "cycle", "channels": 3}, [trig(hits, n=100)], c)
    assert [np.flatnonzero(seq.v[i]).tolist() for i in range(3)] == [[10, 40], [20, 50], [30]]
    pp = run({"op": "cycle", "channels": 3, "order": "pingpong"}, [trig(hits, n=100)], c)
    assert [np.flatnonzero(pp.v[i]).tolist() for i in range(3)] == [[10, 50], [20, 40], [30]]
    r1 = run({"op": "cycle", "channels": 4, "order": "random", "seed": 7}, [trig(range(5, 95, 3), n=100)], c)
    r2 = run({"op": "cycle", "channels": 4, "order": "random", "seed": 7}, [trig(range(5, 95, 3), n=100)], c)
    assert np.array_equal(r1.v, r2.v)
    assert np.all((r1.v != 0).sum(axis=0) == np.array([1 if f in range(5, 95, 3) else 0 for f in range(100)]))
    ch = [int(np.flatnonzero(r1.v[:, f])[0]) for f in range(5, 95, 3)]
    assert all(a != b for a, b in zip(ch, ch[1:])), "random cycle repeated a channel"


def test_ripple_spreads_to_neighbours():
    c = ctx(frames=100)
    v = np.zeros((5, 100), np.float32)
    v[2, 10] = 1.0
    s = Signal(v)
    r = run({"op": "ripple", "neighbours": 2, "falloff": 0.5, "step_ms": 100, "ring": True}, [s], c)
    assert r.v[2, 10] == 1.0
    assert r.v[1, 13] == 0.5 and r.v[3, 13] == 0.5
    assert r.v[0, 16] == 0.25 and r.v[4, 16] == 0.25
    assert np.count_nonzero(r.v) == 5
    v = np.zeros((5, 100), np.float32)
    v[0, 10] = 1.0
    ring = run({"op": "ripple", "neighbours": 1, "falloff": 0.5, "step_ms": 100, "ring": True}, [Signal(v)], c)
    line = run({"op": "ripple", "neighbours": 1, "falloff": 0.5, "step_ms": 100, "ring": False}, [Signal(v)], c)
    assert ring.v[4, 13] == 0.5 and ring.v[1, 13] == 0.5
    assert line.v[4].max() == 0 and line.v[1, 13] == 0.5


def test_envelope_attack0_rises_instantly_and_overlaps_take_max():
    c = ctx(frames=200, bpm=120)
    e = run({"op": "envelope", "shape": "punch", "length_beats": 1}, [trig([10], vel=0.8)], c)
    assert e.v[0, 9] == 0 and abs(e.v[0, 10] - 0.8) < 1e-6
    assert np.all(np.diff(e.v[0, 10:26]) < 0) and e.v[0, 26:].max() == 0
    two = run({"op": "envelope", "shape": "saw_down", "length_ms": 1000}, [trig([10, 20], vel=1.0)], c)
    assert abs(two.v[0, 20] - 1.0) < 1e-6 and abs(two.v[0, 25] - (1 - 5 / 30)) < 1e-6
    att = run({"op": "envelope", "shape": "gloss", "length_ms": 500, "attack_ms": 100}, [trig([10])], c)
    assert att.v[0, 10] == 0 and np.max(np.diff(att.v[0])) < 0.9


def test_smooth_never_jumps():
    c = ctx(frames=120)
    step = np.zeros(120, np.float32)
    step[30:80] = 1.0
    for att, rel in ((100, 300), (250, 250)):
        s = run({"op": "smooth", "attack_ms": att, "release_ms": rel}, [Signal(step)], c)
        d = np.diff(s.v[0].astype(np.float64))
        lim = 1 - np.exp(-(1000 / 30) / min(att, rel))
        assert np.max(np.abs(d)) <= lim + 1e-6 and np.max(np.abs(d)) < 0.3
        assert s.v[0, 29] == 0                      # nothing before the step (causal)
    sp = run({"op": "spring", "freq_hz": 1.5, "damping": 0.7}, [Signal(step)], c)
    assert sp.v[0, 29] == 0 and np.max(np.abs(np.diff(sp.v[0]))) < 0.3


def test_delay_never_early():
    c = ctx(frames=100)
    d = run({"op": "delay", "ms": 50, "step_ms": 0}, [trig([10, 40], n=100)], c)
    assert np.flatnonzero(d.v[0]).tolist() == [12, 42]           # 1.5 frames -> next frame


def test_lfo_beat_locked():
    c = ctx(frames=120, bpm=120)                                   # 15 frames per beat
    l = run({"op": "lfo", "shape": "saw_up", "rate_beats": 1, "channels": 2, "phase_step": 0.5}, [], c)
    assert abs(l.v[0, 0]) < 1e-6 and abs(l.v[0, 15]) < 1e-6 and abs(l.v[0, 7] - 7 / 15) < 1e-6
    assert abs(l.v[1, 0] - 0.5) < 1e-6


def test_layer_crossfade_eased():
    keys = [{"frame": 0, "value": 1}, {"frame": 450, "value": 0, "ease": "in_out", "fade_frames": 30}]
    v = layer_curve(keys, 600)
    assert v[449] == 1 and v[450] == 1 and abs(v[465] - 0.5) < 1e-9 and v[480] == 0 and v[599] == 0
    d = np.diff(v)
    assert np.all(d <= 1e-12)
    assert abs(d[450]) < 1 / 30 / 5 and abs(d[479]) < 1 / 30 / 5      # eased: slow at both ends
    assert np.max(np.abs(d)) <= 1.5 / 30 + 1e-9                         # smoothstep peak slope 1.5x linear
    other = layer_curve([{"frame": 0, "value": 0}, {"frame": 450, "value": 1, "ease": "in_out", "fade_frames": 30}], 600)
    assert np.allclose(v + other, 1)
    lin = layer_curve([{"frame": 0, "value": 0}, {"frame": 10, "value": 1, "ease": "linear", "fade_frames": 10}], 30)
    assert abs(lin[15] - 0.5) < 1e-9
