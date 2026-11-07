"""Flash safety: the pure flash_report, the bake-time rule (error with rig.safety, warning without), and the causal
flash_limit op (triggers and values)."""
import numpy as np

from helpers import ctx, tmpdir
import beatrig
from beatrig.ops import SPECS
from beatrig.safety import flash_report, transitions
from beatrig.signal import Signal, empty_aux

FPS = 30


def run(node, ins, c):
    node = dict(node, id=node.get("id", "n"))
    return SPECS[node["op"]].fn(node, ins, c)


def strobe(hz, seconds=4, fps=FPS):
    """on/off square wave at hz (each cycle = one flash)"""
    t = np.arange(int(seconds * fps)) / fps
    return ((t * hz) % 1.0 < 0.5).astype(float)


def pulse(hz, seconds=4, fps=FPS):
    t = np.arange(int(seconds * fps)) / fps
    return 0.5 - 0.5 * np.cos(2 * np.pi * hz * t)


def test_transitions_zigzag():
    v = np.array([0, 0.5, 1, 0.95, 1, 0.2, 0.25, 0, 0.9, 0.9, 0.1])
    ends, dirs = transitions(v, 0.5)
    assert ends.tolist() == [2, 7, 8, 10] and dirs.tolist() == [1, -1, 1, -1]   # the 1 -> 0.95 -> 1 wobble is no transition
    assert transitions(np.zeros(10), 0.1)[0].tolist() == []


def test_flash_report_10hz_fails_2hz_passes():
    fast = flash_report(strobe(10), FPS, 0, 1)
    assert not fast["ok"] and fast["max_per_s"] >= 9 and fast["windows"]
    assert fast["windows"][0][0] < 0.2 and fast["windows"][-1][1] > 3.5
    assert len(fast["per_second"]) == 4 and min(fast["per_second"]) >= 9
    slow = flash_report(pulse(2), FPS, 0, 1)
    assert slow["ok"] and slow["max_per_s"] <= 2 and slow["windows"] == []
    assert flash_report(strobe(10) * 0.05, FPS, 0, 1)["ok"]                    # swings under 10 % are not flashes
    rgba = np.stack([strobe(10)] * 3 + [np.ones(120)], axis=1)
    assert not flash_report(rgba, FPS, 0, 1)["ok"] and flash_report(rgba * [1, 1, 1, 1], FPS, 0, 1)["max_per_s"] >= 9   # colour rows: luminance
    assert not flash_report(strobe(4), FPS, 0, 1)["ok"] and flash_report(strobe(3), FPS, 0, 1)["max_per_s"] == 3


def _rig(hz, safety=True, limit=False):
    nodes = [{"id": "w", "op": "lfo", "shape": "square", "rate_hz": hz}]
    sig = "w"
    if limit:
        nodes += [{"id": "on", "op": "threshold", "in": "w", "level": 0.5, "cooldown_ms": 0},
                  {"id": "lim", "op": "flash_limit", "in": "on"},
                  {"id": "env", "op": "envelope", "in": "lim", "shape": "gloss", "length_ms": 150}]
        sig = "env"
    rig = {"frames": 150, "fps": FPS, "nodes": nodes,
           "outputs": [{"id": "strobe", "signal": sig, "target": {"kind": "light", "actors": ["Light_0"]}, "min": 0, "max": 5000}]}
    if safety:
        rig["safety"] = {"max_flashes_per_s": 3, "min_swing": 0.1, "outputs": "auto"}
    return beatrig.Rig.from_dict(rig, tmpdir())


def test_bake_enforces_the_limit_with_a_safety_block():
    try:
        beatrig.bake(_rig(10), notes={})
        raise AssertionError("a 10 Hz strobe baked under a safety block")
    except beatrig.RigError as e:
        assert "flashes" in str(e) and "Light_0" in str(e)
    res = beatrig.bake(_rig(2), notes={})
    assert res["safety"]["violations"] == []
    lax = beatrig.bake(_rig(10), notes={}, strict=False)
    assert lax["safety"]["violations"][0]["output"] == "strobe"
    warn = beatrig.bake(_rig(10, safety=False), notes={})                       # no block: reported, not refused
    assert warn["safety"]["violations"]
    issues = beatrig.check(_rig(10, safety=False), {}, bake=warn)
    assert any(i["level"] == "warning" and "flashes" in i["msg"] for i in issues)
    issues = beatrig.check(_rig(10), {}, bake=lax)
    assert any(i["level"] == "error" and "flashes" in i["msg"] for i in issues)
    fixed = beatrig.bake(_rig(10, limit=True), notes={})                         # flash_limit makes the same strobe pass
    assert fixed["safety"]["violations"] == []


def trig(frames, n):
    v = np.zeros((1, n), np.float32)
    v[0, frames] = 1
    return Signal(v, "trigger", v != 0, empty_aux(v.shape))


def test_flash_limit_triggers_drop_and_delay():
    c = ctx(frames=300)
    hits = list(range(0, 300, 3))                                               # 10 Hz
    lim = run({"op": "flash_limit", "max_flashes_per_s": 3}, [trig(hits, 300)], c)
    got = np.flatnonzero(lim.v[0]).tolist()
    assert set(got) <= set(hits) and all(b - a >= 10 for a, b in zip(got, got[1:]))
    env = run({"op": "envelope", "shape": "punch", "length_ms": 200}, [lim], c)
    assert flash_report(env.v[0], FPS, 0, 1)["ok"]
    dl = run({"op": "flash_limit", "max_flashes_per_s": 3, "mode": "delay"}, [trig([0, 4, 5, 40], 300)], c)
    assert np.flatnonzero(dl.v[0]).tolist() == [0, 10, 40]                      # 4 and 5 merge into one hit at 10, never earlier
    ok = run({"op": "flash_limit", "max_flashes_per_s": 3}, [trig([0, 20, 40], 300)], c)
    assert np.flatnonzero(ok.v[0]).tolist() == [0, 20, 40]                       # slow hits pass untouched


def test_flash_limit_values_is_causal_and_safe():
    c = ctx(frames=240)
    x = Signal(strobe(8, 8)[None, :].astype(np.float32))
    y = run({"op": "flash_limit"}, [x], c).v[0]
    assert flash_report(y, FPS, 0, 1)["ok"], flash_report(y, FPS, 0, 1)["max_per_s"]
    for k in (37, 100, 181):                                                    # truncation: the past never changes
        c2 = ctx(frames=k)
        y2 = run({"op": "flash_limit"}, [Signal(x.v[:, :k])], c2).v[0]
        assert np.array_equal(y[:k], y2)
    calm = Signal(pulse(1, 8)[None, :].astype(np.float32))
    assert np.allclose(run({"op": "flash_limit"}, [calm], c).v, calm.v)         # a safe signal passes unchanged
