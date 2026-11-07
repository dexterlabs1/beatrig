"""ue/keys.py: the pure half of the Unreal applier (tick maths, key planning, tags, read-back compare)."""
import os
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ue"))
import keys as K  # noqa: E402


def test_rate_parsing():
    assert K.rate(30) == 30
    assert K.rate(29.97) == Fraction(30000, 1001)
    assert K.rate(23.976) == Fraction(24000, 1001)
    assert K.rate("60000/1001") == Fraction(60000, 1001)
    assert K.rate((24000, 1)) == 24000
    with pytest.raises(K.PlanError):
        K.rate(0)


def test_ticks_per_frame():
    assert K.ticks_per_frame(24000, 30) == 800
    assert K.ticks_per_frame(24000, 24) == 1000
    assert K.ticks_per_frame(24000, 60) == 400
    assert K.ticks_per_frame(24000, 29.97) == Fraction(4004, 5)        # 800.8
    assert K.ticks_per_frame(24000, 23.976) == Fraction(1001)


def test_frame_to_ticks_integer_rates():
    assert K.frame_to_ticks(0) == 0
    assert K.frame_to_ticks(1) == 800
    assert K.frame_to_ticks(450, 24000, 30) == 360000              # 15 s
    assert K.frame_to_ticks(90, 24000, 30) == 72000


def test_frame_to_ticks_ntsc_rounds_exactly():
    # 29.97: frame f is f * 800.8 ticks; rounding comes from the exact rational, never from float noise
    assert K.frame_to_ticks(1, 24000, 29.97) == 801                # 800.8
    assert K.frame_to_ticks(5, 24000, 29.97) == 4004               # exact
    assert K.frame_to_ticks(2, 24000, 29.97) == 1602               # 1601.6
    assert K.frame_to_ticks(30000, 24000, 29.97) == 24024000       # 1001 s of ticks
    assert K.frame_to_ticks(1, 24000, 23.976) == 1001


def test_ticks_round_trip():
    for fps in (24, 25, 30, 60, 29.97, 23.976):
        for f in (0, 1, 7, 1234, 107892):
            back = K.ticks_to_frame(K.frame_to_ticks(f, 24000, fps), 24000, fps)
            assert abs(back - f) < 1e-3


def test_check_rate():
    assert K.check_rate(30, 30) is None
    assert K.check_rate(29.97, "30000/1001") is None
    assert "set the rig's fps" in K.check_rate(30, 29.97)


def test_tags():
    assert K.tag("lights", "Light_0") == "beatrig:lights:Light_0"
    assert K.tag("glow") == "beatrig:glow"
    assert K.parse_tag("beatrig:lights:Light_0") == ("lights", "Light_0")
    assert K.parse_tag("beatrig:glow") == ("glow", None)
    assert K.parse_tag("Intensity") is None
    assert K.parse_tag("beatrig:") is None
    assert K.is_ours("beatrig:glow")
    assert K.is_ours("beatrig:glow", "glow")
    assert not K.is_ours("beatrig:glow", "lights")
    assert not K.is_ours("my beatrig:glow")


BAKE = {
    "version": 1, "fps": 30, "frames": 100,
    "outputs": [
        {"id": "glow", "target": {"kind": "mpc", "asset": "/Game/FX/MPC_Show", "param": "Glow"}, "min": 0, "max": 1,
         "channels": [{"values": [], "keys": [[0, 0.2], [10, 1.0], [99, 0.2]]}]},
        {"id": "lights", "target": {"kind": "light", "actors": ["L0", "L1"], "property": "intensity"},
         "min": 500, "max": 20000,
         "channels": [{"actor": "L0", "values": [], "keys": [[0, 500.0], [4, 20000.0], [99, 500.0]]},
                      {"actor": "L1", "values": [], "keys": [[0, 500.0], [8, 20000.0], [99, 500.0]]}]},
    ],
}


def test_plan_mpc_uses_ticks_and_light_uses_frames():
    plans = K.plan_bake(BAKE, display_rate=30, tick_resolution=24000)
    assert [p["tag"] for p in plans] == ["beatrig:glow", "beatrig:lights:L0", "beatrig:lights:L1"]
    mpc = plans[0]
    assert mpc["time_base"] == "ticks"
    assert mpc["keys"] == [(0, 0.2), (8000, 1.0), (79200, 0.2)]
    assert mpc["range"] == [0, 100]                      # section ranges are always display frames
    light = plans[1]
    assert light["time_base"] == "frames"
    assert light["property"] == "Intensity"
    assert light["keys"] == [(0, 500.0), (4, 20000.0), (99, 500.0)]


def test_plan_start_frame_offsets_both_bases():
    plans = K.plan_bake(BAKE, 30, 24000, start_frame=100)
    assert plans[0]["keys"][0] == (80000, 0.2)
    assert plans[1]["keys"][0] == (100, 500.0)
    assert plans[1]["range"] == [100, 200]


def test_plan_only_and_errors():
    assert [p["tag"] for p in K.plan_bake(BAKE, 30, 24000, only=["glow"])] == ["beatrig:glow"]
    with pytest.raises(K.PlanError, match="displays at"):
        K.plan_bake(BAKE, 29.97, 24000)
    bad = {"version": 1, "fps": 30, "outputs": [{"id": "x", "target": {"kind": "dmx"}, "channels": []}]}
    with pytest.raises(K.PlanError, match="unknown target kind"):
        K.plan_bake(bad, 30, 24000)
    with pytest.raises(K.PlanError, match="bake version"):
        K.plan_bake({"version": 2}, 30, 24000)


def test_compare_keys():
    exp = [(0, 1.0), (10, 2.0), (20, 3.0)]
    assert K.compare_keys(exp, exp)["ok"]
    r = K.compare_keys(exp, [(0, 1.0), (10, 2.5), (20, 3.0)])
    assert not r["ok"] and r["first_bad"] == [10.0, 2.0, 2.5] and r["max_value_err"] == 0.5
    r = K.compare_keys(exp, [(0, 1.0), (20, 3.0), (30, 4.0)])
    assert r["missing"] == 1 and r["extra"] == 1 and not r["ok"]
