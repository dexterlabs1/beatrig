"""ue/keys.py: the pure half of the Unreal applier (tick maths, key planning, tags, read-back compare)."""
import json
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


# ---------------------------------------------------------------- v0.2: colour, transform, Niagara
def _out(oid, target, keys, actor="A0"):
    return {"id": oid, "target": target, "channels": [{"actor": actor, "values": [], "keys": keys}]}


COLOR_KEYS = [[0, 1.0, 0.5, 0.0, 1.0], [10, 0.0, 0.25, 1.0, 1.0], [99, 0.2, 0.2, 0.2, 1.0]]


def test_plan_colour_per_kind():
    mpc = K.plan_output(dict(_out("tint", {"kind": "mpc", "asset": "/Game/FX/MPC_Show", "param": "Tint",
                                           "value_type": "color"}, COLOR_KEYS, None)), 30, 30, 24000)[0]
    assert mpc["value_type"] == "color" and mpc["time_base"] == "ticks"
    assert mpc["keys"][1] == (8000, (0.0, 0.25, 1.0, 1.0))
    assert K.channel_names(mpc) == ["TintR", "TintG", "TintB", "TintA"]          # parameter channels: no separator
    light = K.plan_output(_out("col", {"kind": "light", "actors": ["A0"], "property": "color", "value_type": "color"},
                               COLOR_KEYS), 30, 30, 24000)[0]
    assert light["track"] == "MovieSceneColorTrack" and light["property"] == "LightColor"
    assert light["time_base"] == "frames" and light["keys"][1] == (10, (0.0, 0.25, 1.0, 1.0))
    assert K.channel_names(light) == ["Color.R", "Color.G", "Color.B", "Color.A"]
    mat = K.plan_output(_out("mc", {"kind": "material", "actors": ["A0"], "param": "Color", "value_type": "color"},
                             COLOR_KEYS), 30, 30, 24000)[0]
    assert mat["time_base"] == "ticks" and K.channel_names(mat) == ["ColorR", "ColorG", "ColorB", "ColorA"]


def test_value_type_inferred_and_validated():
    o = _out("c", {"kind": "light", "actors": ["A0"], "property": "color"}, COLOR_KEYS)
    assert K.value_type(o) == "color"                                            # 5-wide key rows
    assert K.value_type(_out("f", {"kind": "light"}, [[0, 1.0]])) == "float"
    with pytest.raises(K.PlanError, match="value_type"):
        K.value_type(_out("x", {"kind": "mpc", "value_type": "rgb"}, []))
    with pytest.raises(K.PlanError, match="can't take colour"):
        K.plan_output(_out("t", {"kind": "transform", "actors": ["A0"], "property": "location", "axis": "x"},
                           COLOR_KEYS), 30, 30, 24000)
    with pytest.raises(K.PlanError, match="keys are"):
        K.plan_output(_out("c", {"kind": "mpc", "asset": "/x", "param": "T", "value_type": "color"},
                           [[0, 1.0]], None), 30, 30, 24000)
    with pytest.raises(K.PlanError, match="colour property"):
        K.plan_output(_out("c", {"kind": "light", "actors": ["A0"], "property": "intensity", "value_type": "color"},
                           COLOR_KEYS), 30, 30, 24000)


def test_plan_transform_contract_form():
    def tr(prop, axis, **kw):
        return K.plan_output(_out("t", dict({"kind": "transform", "actors": ["A0"], "property": prop, "axis": axis}, **kw),
                                  [[0, 0.0], [10, 5.0]]), 30, 30, 24000)[0]
    p = tr("rotation", "yaw")
    assert p["channel_name"] == "Rotation.Z" and p["channel_index"] == 5 and p["blend"] == "additive"
    assert tr("rotation", "roll")["channel_name"] == "Rotation.X" and tr("rotation", "pitch")["channel_name"] == "Rotation.Y"
    assert tr("location", "z")["channel_name"] == "Location.Z" and tr("scale", "x")["channel_index"] == 6
    assert tr("location", "x", additive=False)["blend"] == "absolute"
    assert p["time_base"] == "frames" and p["keys"] == [(0, 0.0), (10, 5.0)]
    old = K.plan_output(_out("t", {"kind": "transform", "actors": ["A0"], "channel": "location.y"}, [[0, 1.0]]),
                        30, 30, 24000)[0]                                       # the v0.1 form still plans
    assert old["channel_name"] == "Location.Y"
    with pytest.raises(K.PlanError, match="transform needs"):
        tr("location", "yaw")


def test_plan_niagara_param_prefix():
    p = K.plan_output(_out("n", {"kind": "niagara", "actors": ["Fx"], "param": "Rate"}, [[0, 1.0]], "Fx"), 30, 30, 24000)[0]
    assert p["param"] == "User.Rate" and p["track"] == "MovieSceneNiagaraFloatParameterTrack"
    assert K.niagara_param("User.Rate") == "User.Rate"
    assert K.channel_names(p) is None


def test_frame_keys_and_colour_compare():
    p = K.plan_output(_out("tint", {"kind": "mpc", "asset": "/x", "param": "Tint", "value_type": "color"},
                           COLOR_KEYS, None), 29.97, 29.97, 24000)[0]
    fk = K.frame_keys(p, K.ticks_per_frame(24000, 29.97))
    # ticks back to display frames: exact up to the whole-tick rounding (99 x 800.8 = 79279.2 -> 79279 ticks)
    assert all(abs(f - want) < 0.01 for (f, _), want in zip(fk, (0, 10, 99)))
    comps = K.split_components(fk)
    assert len(comps) == 4 and comps[2] == [(fk[0][0], 0.0), (fk[1][0], 1.0), (fk[2][0], 0.2)]
    reports = [K.compare_keys(c, c) for c in comps]
    assert K.merge_reports(reports)["ok"] and K.merge_reports(reports)["expected"] == 12
    bad = list(comps[1])
    bad[1] = (bad[1][0], 0.9)
    m = K.merge_reports([K.compare_keys(comps[0], comps[0]), K.compare_keys(comps[1], bad)])
    assert not m["ok"] and m["first_bad"][1:] == [0.25, 0.9] and m["max_value_err"] == 0.65


def test_source_hash_tracks_content_not_line_endings(tmp_path):
    (tmp_path / "a.py").write_bytes(b"x = 1\n")
    (tmp_path / "notes.txt").write_bytes(b"ignored")
    h1 = K.source_hash(str(tmp_path))
    (tmp_path / "a.py").write_bytes(b"x = 1\r\n")
    assert K.source_hash(str(tmp_path)) == h1
    (tmp_path / "notes.txt").write_bytes(b"changed")
    assert K.source_hash(str(tmp_path)) == h1
    (tmp_path / "a.py").write_bytes(b"x = 2\n")
    assert K.source_hash(str(tmp_path)) != h1


def test_fixtures_plan_cleanly():
    """the hand-made bakes in tests/fixtures plan without errors at their own rate"""
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    kinds = set()
    for name in sorted(os.listdir(here)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(here, name), encoding="utf-8") as fh:
            bake = json.load(fh)
        plans = K.plan_bake(bake, bake["fps"], 24000)
        assert plans, name
        kinds |= {(p["kind"], p["value_type"]) for p in plans}
    assert kinds >= {("light", "float"), ("light", "color"), ("material", "float"), ("material", "color"),
                     ("mpc", "float"), ("mpc", "color"), ("transform", "float"), ("niagara", "float")}


def test_engine_colour_and_transform_bakes_plan():
    """a bake from the engine's colour and transform outputs plans as the applier expects"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from beatrig.ops import SPECS
    if "gradient" not in SPECS or "accumulate" not in SPECS:
        pytest.skip("engine has no colour/transform ops yet")
    import beatrig
    rig = {"fps": 30, "frames": 90, "tempo": {"bpm": 120},
           "nodes": [{"id": "w", "op": "lfo", "rate_hz": 1}, {"id": "col", "op": "gradient", "in": "w", "palette": "ice"},
                     {"id": "spin", "op": "accumulate", "in": "w", "per_s": 0.5}],
           "outputs": [{"id": "tint", "signal": "col", "target": {"kind": "mpc", "asset": "/Game/FX/MPC_Show", "param": "Tint"}},
                       {"id": "lc", "signal": "col", "target": {"kind": "light", "actors": ["L0"], "property": "color"}},
                       {"id": "yaw", "signal": "spin", "min": 0, "max": 360,
                        "target": {"kind": "transform", "actors": ["D0"], "property": "rotation", "axis": "yaw"}}]}
    bake = beatrig.bake_from_objects(rig, {"tracks": {}}, os.path.dirname(os.path.abspath(__file__)))
    plans = {p["tag"]: p for p in K.plan_bake(bake, 30, 24000)}
    assert plans["beatrig:tint"]["value_type"] == "color" and len(plans["beatrig:tint"]["keys"][0][1]) == 4
    assert plans["beatrig:lc:L0"]["track"] == "MovieSceneColorTrack"
    assert plans["beatrig:yaw:D0"]["channel_name"] == "Rotation.Z" and plans["beatrig:yaw:D0"]["blend"] == "additive"
