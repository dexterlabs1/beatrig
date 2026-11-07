"""Rule 1: nothing ahead of the audio. Synthetic clicks at known times; band envelopes and onset hits must not move
before the first frame at or after each click; any frame's value must not change when later audio is cut off."""
import math
import os

import numpy as np
import soundfile as sf

from helpers import FPS, click_times, rig_folder, write_clicks
import beatrig
from beatrig import analysis as A

RIG = {"fps": FPS, "nodes": [
    {"id": "low", "op": "band", "lo": 40, "hi": 120},
    {"id": "high", "op": "band", "lo": 2000, "hi": 8000, "attack_ms": 2, "release_ms": 80},
    {"id": "loud", "op": "loudness"},
    {"id": "on", "op": "onsets"},
    {"id": "onHigh", "op": "onsets", "in": "high"},
    {"id": "lowS", "op": "smooth", "in": "low", "attack_ms": 40, "release_ms": 200},
    {"id": "lowSp", "op": "spring", "in": "low", "freq_hz": 3, "damping": 0.5},
    {"id": "lowH", "op": "hold", "in": "low", "ms": 100},
    {"id": "env", "op": "envelope", "in": "on", "shape": "punch", "length_ms": 300},
    {"id": "ring", "op": "chase", "in": "env", "channels": 4, "step_ms": 100},
], "outputs": []}


def _signals(rig_dir):
    A._MEMO.clear()
    rig = beatrig.load_rig(os.path.join(rig_dir, "rig.json"))
    an = beatrig.analyze(rig)
    return beatrig.evaluate(rig, an, {}), an


def first_frames(times):
    return [math.ceil(t * FPS) for t in times]


def test_no_rise_before_click_and_rise_on_first_frame():
    times = click_times()
    sig, _ = _signals(rig_folder(dict(RIG)))
    ff = first_frames(times)
    for nid in ("low", "high", "loud", "lowS", "lowH"):
        v = sig[nid].v[0].astype(np.float64)
        assert np.all(v[:ff[0]] == 0), f"{nid}: non-zero before the first click"
        for f in ff:
            pre = v[f - 6:f]
            assert np.all(np.diff(pre) <= 1e-9), f"{nid}: rises in the 6 frames before frame {f}: {pre}"
            assert v[f] > v[f - 1], f"{nid}: no rise on frame {f}"


def test_peak_lands_on_first_frame_at_or_after_click():
    times = click_times()
    sig, _ = _signals(rig_folder(dict(RIG)))
    v = sig["high"].v[0]
    for f in first_frames(times):
        assert int(np.argmax(v[f - 3:f + 6])) + f - 3 == f, f"high band peak not on frame {f}"


def test_onset_hits_on_first_frame_at_or_after_click():
    times = click_times()
    sig, _ = _signals(rig_folder(dict(RIG)))
    want = first_frames(times)
    for nid in ("on", "onHigh"):
        s = sig[nid]
        assert s.kind == "trigger"
        got = np.flatnonzero(s.v[0]).tolist()
        assert got == want, f"{nid}: hits {got[:6]}.. want {want[:6]}.."


def test_truncation_invariance():
    """cut the audio at T: every frame whose start is <= T must be identical (no op may read later audio)"""
    times = click_times()
    d_full = rig_folder(dict(RIG), times=times)
    full, _ = _signals(d_full)
    d_cut = rig_folder(dict(RIG, frames=len(full["low"].v[0])), times=times)
    x, sr = sf.read(os.path.join(d_cut, "clicks.wav"), always_2d=True)
    T = 9.0
    x[int(T * sr):] = 0
    sf.write(os.path.join(d_cut, "clicks.wav"), x, sr, subtype="FLOAT")
    cut, _ = _signals(d_cut)
    last = int(T * FPS)                        # frame start <= T
    for nid in ("low", "high", "loud", "lowS", "lowSp", "lowH"):
        a, b = full[nid].v[:, :last + 1], cut[nid].v[:, :last + 1]
        assert np.array_equal(a, b), f"{nid}: differs before the cut (max {np.max(np.abs(a - b))})"


def test_at_frames_reads_only_completed_blocks():
    env = np.arange(1, 101, dtype=np.float32)          # value k+1 is known at (k+1) ms
    v = A.at_frames(env, 1, 4, 1000.0 / 10)            # frames every 10 ms
    assert v.tolist() == [0, 10, 20, 30]


def test_time_to_frame_is_first_frame_at_or_after():
    assert A.time_to_frame([0.0, 0.001, 1 / 30, 1.010], 30).tolist() == [0, 1, 1, 31]
    assert A.time_to_frame([1.010], 30, offset=5).tolist() == [36]


def test_regular_loop_gets_a_proven_pulse():
    """a perfectly regular 16 s loop is the easy case: proven pulse, correct tempo, a beat on (nearly) every beat"""
    from helpers import tmpdir, write_loop
    for bpm in (90, 120, 140):
        d = tmpdir()
        write_loop(os.path.join(d, "loop.wav"), bpm)
        A._MEMO.clear()
        an = A.open_analysis(os.path.join(d, "loop.wav"), d)
        got = an.bpm_map[0]["bpm"]
        n_beats = 16.0 * bpm / 60
        assert not an.meta["fallback"], (bpm, an.meta["pulse"], an.meta["rejected_passages"])
        assert abs(got - bpm) < 0.5, (bpm, got)
        assert len(an.beats) >= 0.85 * n_beats, (bpm, len(an.beats), n_beats)
        t = np.array([b["time"] for b in an.beats])
        lag = t - np.round(t / (60 / bpm)) * (60 / bpm)
        assert np.all(lag >= 0) and np.all(lag < 0.04), (bpm, lag)      # detected at or just after each beat
