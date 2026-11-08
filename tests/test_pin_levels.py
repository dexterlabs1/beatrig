"""normalize ref_db (a fixed level scale) and `beatrig pin-levels`: pinning changes nothing on the audio it was pinned
from, and a louder copy of the mix then reads higher instead of being normalised back."""
import json
import os

import numpy as np
import soundfile as sf

from helpers import click_times, tmpdir, write_clicks
import beatrig
from beatrig import analysis as A
from beatrig.cli import main as cli


def test_norm_db_ref_is_a_fixed_scale():
    x = np.abs(np.sin(np.linspace(0, 40, 4000))) * 0.1 + 1e-4
    a, lo, hi = A.norm_db(x, 24, ref_db=-10.0)
    assert (lo, hi) == (-34.0, -10.0)
    b, _, _ = A.norm_db(x * 2, 24, ref_db=-10.0)          # +6.02 dB
    peak = np.argmax(x)
    assert abs((b[peak] - a[peak]) - 6.0206 / 24) < 1e-3
    c, _, _ = A.norm_db(x * 2, 24)                         # percentile mode follows the gain: no change
    d, _, _ = A.norm_db(x, 24)
    assert np.allclose(c, d)


def test_pinned_scale_equals_the_percentile_scale_when_the_floor_binds():
    x = 0.05 + 0.01 * np.abs(np.sin(np.linspace(0, 30, 5000)))     # ~2 dB of range: the lo_pct floor binds, not range_db
    v, lo, hi = A.norm_db(x, 30)
    assert lo > hi - 30
    w, _, _ = A.norm_db(x, 30, ref_db=hi, floor_db=lo)
    assert np.allclose(v, w)


def _rig(folder, audio):
    rig = {"audio": audio, "fps": 30, "frames": None, "seed": 1, "nodes": [
        {"id": "loud_raw", "op": "loudness", "attack_ms": 15, "release_ms": 200},
        {"id": "loud", "op": "normalize", "in": "loud_raw", "range_db": 24},
        {"id": "kick_raw", "op": "band", "lo": 40, "hi": 120},
        {"id": "kick", "op": "normalize", "in": "kick_raw", "range_db": 30, "hi_pct": 99}],
        "outputs": [{"id": "glow", "signal": "loud", "target": {"kind": "mpc", "asset": "/Game/M", "param": "G"}},
                    {"id": "kk", "signal": "kick", "target": {"kind": "mpc", "asset": "/Game/M", "param": "K"}}]}
    p = os.path.join(folder, "rig.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rig, f)
    return p


def _values(rig_path, oid):
    res = beatrig.bake(beatrig.load_rig(rig_path))
    o = next(o for o in res["outputs"] if o["id"] == oid)
    return np.asarray(o["channels"][0]["values"], float)


def test_pin_levels_keeps_the_reference_and_lets_a_louder_mix_drive_harder():
    d = tmpdir()
    write_clicks(os.path.join(d, "mix.wav"), click_times(duration=8.0), duration=8.0)
    y, sr = sf.read(os.path.join(d, "mix.wav"))
    sf.write(os.path.join(d, "loud.wav"), y * 0.25, sr)     # quieter copy = the reference; mix.wav is +12 dB
    ref = _rig(d, "loud.wav")
    before = _values(ref, "glow")
    assert cli(["pin-levels", ref]) == 0
    raw = json.load(open(ref, encoding="utf-8"))
    pinned = {n["id"]: n.get("ref_db") for n in raw["nodes"] if n["op"] == "normalize"}
    assert all(isinstance(v, float) for v in pinned.values()), pinned
    after = _values(ref, "glow")
    assert np.max(np.abs(after - before)) < 2e-3            # same audio: the bake is unchanged (rounding of ref_db only)
    # the same pinned rig on the 12 dB louder mix reads higher; an unpinned rig would normalise it back
    raw["audio"] = "mix.wav"
    hot = os.path.join(d, "hot.json")
    json.dump(raw, open(hot, "w", encoding="utf-8"))
    louder = _values(hot, "glow")
    assert np.mean(louder) > np.mean(after) + 0.2
    raw2 = dict(raw); raw2["nodes"] = [{k: v for k, v in n.items() if k != "ref_db"} for n in raw["nodes"]]
    cold = os.path.join(d, "cold.json")
    json.dump(raw2, open(cold, "w", encoding="utf-8"))
    assert np.max(np.abs(_values(cold, "glow") - before)) < 0.02


def test_pin_levels_respects_existing_pins_and_dry_run():
    d = tmpdir()
    write_clicks(os.path.join(d, "mix.wav"), click_times(duration=6.0), duration=6.0)
    p = _rig(d, "mix.wav")
    assert cli(["pin-levels", p, "--dry-run"]) == 0
    assert all("ref_db" not in n for n in json.load(open(p))["nodes"])
    assert cli(["pin-levels", p, "--nodes", "kick"]) == 0
    nodes = {n["id"]: n for n in json.load(open(p))["nodes"]}
    assert "ref_db" in nodes["kick"] and "ref_db" not in nodes["loud"]
    nodes["kick"]["ref_db"] = -3.0
    raw = json.load(open(p)); raw["nodes"] = list(nodes.values()); json.dump(raw, open(p, "w"))
    assert cli(["pin-levels", p]) == 0                      # an existing pin is kept without --force
    assert {n["id"]: n for n in json.load(open(p))["nodes"]}["kick"]["ref_db"] == -3.0
