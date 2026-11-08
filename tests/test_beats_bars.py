"""Neural beats and bars. The model's activations are mocked (synthetic logits), so these run without torch; one
test runs the real model when the [ml] extra is installed. What must hold whatever the model says:
- a beat is written only where a real onset supports it, at that onset's time, never at (or before) the model's frame;
- bar counting follows the model's downbeats through beats without an onset and through stretches where the model is
  silent, and does not flip on a few contrary bars;
- the tempo map's grid offset is a downbeat's phase, and the beats op can count on the bar grid (anchor "bar")."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from helpers import FPS, ROOT, tmpdir, write_clicks, write_loop
import beatrig
from beatrig import analysis as A
from beatrig import neural
from beatrig.cli import main as cli
from beatrig.ops import beat_times

STEP = 0.005


def onset_fn(times, duration, seed=11):
    """5 ms onset function with a unit peak at each time (index k is known at (k + 1) * 5 ms) over a small floor"""
    rng = np.random.default_rng(seed)
    f = 0.02 * rng.random(int(duration / STEP))
    for t in times:
        k = int(round(t / STEP)) - 1
        f[k] = 1.0
        f[k + 1] = max(f[k + 1], 0.3)
    return f


def logits(times, duration, early=0.02, height=6.0, fps=50.0):
    """a logit track with a peak 20 ms ahead of each time (a model frame is often ahead of the onset)"""
    x = np.full(int(duration * fps) + 1, -8.0)
    for t in times:
        i = int(round((t - early) * fps))
        for d in (-1, 0, 1):
            if 0 <= i + d < len(x):
                x[i + d] = max(x[i + d], height - 4 * abs(d))
    return x


def test_written_beats_sit_on_onsets_and_bars_count_through_gaps():
    T, dur = 0.5, 40.0
    model = 0.51 + T * np.arange(int((dur - 1) / T))            # the model hears a beat every 0.5 s
    hole = (model > 10.0) & (model < 12.0)                      # ... but no onset supports these four
    f = onset_fn(model[~hole], dur)                             # (only the noise floor there)
    downs = model[2::4]                                         # first downbeat = model beat 2 (a 2-beat pickup)
    g = A.neural_grid(f, model[~hole], logits(model, dur), logits(downs, dur), dur, dict(A.DEFAULT_SETTINGS))
    times = np.array([b["time"] for b in g["beats"]])
    assert len(times) == len(model) - hole.sum()
    assert not np.any((times > 10.0) & (times < 12.0))           # no onset, no beat
    on = (np.round(times / STEP).astype(int) - 1)
    assert np.allclose(times, (on + 1) * STEP) and np.all(f[on] == 1.0)    # each one on an onset peak, at (k + 1) * 5 ms
    idx = np.round((times - 0.51) / T).astype(int)
    assert np.all(times >= model[idx] - 1e-9)                    # never ahead of the model's own (earlier) frame
    pos = np.array([b["pos"] for b in g["beats"]])
    bar = np.array([b["bar"] for b in g["beats"]])
    assert np.array_equal(pos, (idx - 2) % 4) and np.array_equal(bar, (idx - 2) // 4)   # counting never slips
    assert bar[0] == -1 and g["bars"]["meter"] == 4
    assert abs(g["bpm_map"][0]["bpm"] - 120) < 0.05
    assert abs(g["grid_offset"] - 1.51) < 0.006                  # first downbeat's phase within a 2 s bar
    assert len(g["bars"]["downbeats"]) == len(downs)


def test_model_silence_keeps_the_bar_count():
    T, dur = 0.5, 60.0
    model = 0.51 + T * np.arange(int((dur - 1) / T))
    quiet = (model > 20.0) & (model < 30.0)                     # 20 beats where the model gives nothing
    f = onset_fn(model[~quiet], dur)
    downs = model[::4]
    g = A.neural_grid(f, model[~quiet], logits(model[~quiet], dur), logits(downs[(downs < 20) | (downs > 30)], dur), dur,
                      dict(A.DEFAULT_SETTINGS))
    assert len(g["passages"]) == 2                               # split at the silence
    times = np.array([b["time"] for b in g["beats"]])
    idx = np.round((times - 0.51) / T).astype(int)
    assert np.array_equal([b["pos"] for b in g["beats"]], idx % 4)
    assert np.array_equal([b["bar"] for b in g["beats"]], idx // 4)


def _down_probs(n_beats, phase_of_beat, fps=50.0, T=0.5):
    """skeleton times and a downbeat probability track with a peak on each beat where phase_of_beat(i) says so"""
    t = 0.5 + T * np.arange(n_beats)
    p = np.full(int((t[-1] + 1) * fps), 0.02)
    for i in range(n_beats):
        if phase_of_beat(i):
            p[int(round(t[i] * fps))] = 0.97
    return t, p


def test_bar_phase_survives_a_short_contrary_stretch_and_follows_a_real_shift():
    # bars 20-27: the model hears the bar line one beat early (syncopated chords) and then comes back
    t, p = _down_probs(200, lambda i: (i % 4 == 3) if 80 <= i < 112 else (i % 4 == 0))
    meter, pos = A.bar_positions(t, p, 50.0)
    assert meter == 4 and np.array_equal(pos, np.arange(200) % 4)
    # a real shift: from beat 82 on, every bar line is two beats later (a bar of 2), for good
    t, p = _down_probs(200, lambda i: (i % 4 == 2) if i >= 82 else (i % 4 == 0))
    meter, pos = A.bar_positions(t, p, 50.0)
    want = np.where(np.arange(200) >= 82, (np.arange(200) - 2) % 4, np.arange(200) % 4)
    assert np.array_equal(pos, want)
    t, p = _down_probs(120, lambda i: i % 3 == 0)
    assert A.bar_positions(t, p, 50.0)[0] == 3                   # waltz


def test_skeleton_fills_gaps_and_drops_near_duplicates():
    model = list(0.5 * np.arange(20))
    model.insert(6, 2.53)                                        # a spurious extra beat right after beat 5 (2.5 s)
    del model[12:15]                                             # beats 11, 12 and 13 missed
    probs = np.ones(len(model))
    probs[6] = 0.6
    st, real, sp = A.beat_skeleton(np.array(model), probs)
    assert len(st) == 20 and np.allclose(st, 0.5 * np.arange(20), atol=1e-9)
    assert real.sum() == 17 and not real[11:14].any() and sp[11:14].max() == 0


def test_beats_anchor_bar_counts_on_the_bar_grid():
    class FakeAnalysis:
        meta = {"bars": {"meter": 4}}
        beats = [{"time": 0.5 * k + 0.01, "strength": 1.0, "confidence": 1.0, "bar": (k + 1) // 4 - (1 if k < 3 else 0),
                  "pos": (k + 1) % 4} for k in range(16) if k != 7]   # beat 7 has no onset: not in the list
    from helpers import ctx
    c = ctx(frames=300)
    c.analysis = FakeAnalysis()
    for b in FakeAnalysis.beats:                                  # beat k is pos (k + 1) % 4: beat 3, 7, 11, 15 are downbeats
        assert b["bar"] * 4 + b["pos"] == b["bar"] * 4 + (round((b["time"] - 0.01) / 0.5) + 1) % 4
    t, _ = beat_times({"op": "beats", "every": 4, "anchor": "bar"}, c)
    assert np.allclose(t, [0.5 * k + 0.01 for k in (3, 11, 15)])                 # downbeats (7 was never written)
    t, _ = beat_times({"op": "beats", "every": 4, "offset": 2, "anchor": "bar"}, c)
    assert np.allclose(t, [0.5 * k + 0.01 for k in (1, 5, 9, 13)])               # the third beat of each bar
    t, _ = beat_times({"op": "beats", "every": 4}, c)                            # anchor first: list index, as before
    assert np.allclose(t, [FakeAnalysis.beats[i]["time"] for i in (0, 4, 8, 12)])
    t, _ = beat_times({"op": "beats", "every": 2, "offset": 1, "subdivide": 2, "anchor": "bar"}, c)
    assert len(t) and np.allclose((t - 0.01) % 0.5, 0.25)                        # the off-beat eighths only


def test_anchor_bar_without_downbeats_uses_the_tempo_map():
    class Classic:
        meta = {}
        beats = [{"time": 0.5 * k, "strength": 1.0, "confidence": 1.0} for k in range(12) if k != 5]
    from helpers import ctx
    c = ctx(frames=300)
    c.analysis = Classic()
    t, _ = beat_times({"op": "beats", "every": 4, "anchor": "bar"}, c)
    assert np.allclose(t, [0.0, 2.0, 4.0])                       # beat numbers from the grid offset, gap or not


# ---------- the analysis with mocked activations
def fake_activations(beat_times_s, down_times_s, calls=None):
    def act(audio, weights=None, checkpoint=neural.MODEL, device=None):
        if calls is not None:
            calls.append(audio)
        dur = float(beatrig.analysis.sf.info(audio).duration)
        return {"beat": logits(beat_times_s, dur).astype(np.float32), "downbeat": logits(down_times_s, dur).astype(np.float32),
                "fps": 50.0, "model": "beat_this mock", "device": "cpu", "seconds": 0.0}
    return act


@pytest.fixture
def neural_on(monkeypatch):
    monkeypatch.delenv("BEATRIG_EXTRAS", raising=False)
    monkeypatch.setattr(neural, "available", lambda: (True, ""))
    A._MEMO.clear()
    yield monkeypatch
    A._MEMO.clear()


def _click_rig(d, times, duration, **extra):
    write_clicks(os.path.join(d, "clicks.wav"), times, duration)
    rig = dict({"audio": "clicks.wav", "fps": FPS, "nodes": [
        {"id": "down", "op": "beats", "every": 4, "anchor": "bar"},
        {"id": "all", "op": "beats"}], "outputs": []}, **extra)
    with open(os.path.join(d, "rig.json"), "w", encoding="utf-8") as fh:
        json.dump(rig, fh)
    return os.path.join(d, "rig.json")


def test_analyze_with_the_model_end_to_end(neural_on):
    d = tmpdir()
    clicks = 1.010 + 0.5 * np.arange(44)                         # 23 s of clicks at 120 BPM
    calls = []
    neural_on.setattr(neural, "activations", fake_activations(clicks, clicks[1::4], calls))
    rp = _click_rig(d, clicks, 24.0)
    rig = beatrig.load_rig(rp)
    an = beatrig.analyze(rig)
    assert an.meta["beat_tracker"]["name"] == "neural" and an.tracker_label().startswith("neural (beat_this mock")
    assert len(calls) == 1 and "nn:beat" in an.arrays
    sig = beatrig.evaluate(rig, an, {})
    hits = np.flatnonzero(sig["down"].v[0])
    assert hits.tolist() == [math.ceil(t * FPS) for t in clicks[1::4]]          # bar lines, on the first frame at/after
    assert int(np.count_nonzero(sig["all"].v[0])) == len(clicks)
    assert an.bars_summary()["meter"] == 4 and 0 <= an.grid_offset - (clicks[1] % 2.0) < 0.015   # at the onset, not before
    # the second analyze is a pure read: no model run
    A._MEMO.clear()
    an2 = beatrig.analyze(beatrig.load_rig(rp))
    assert an2.loaded_from_cache and len(calls) == 1 and an2.beats == an.beats
    assert cli(["notes", "init", rp, "--track", "bars", "--from", "downbeats"]) == 0
    notes = beatrig.load_notes(os.path.join(d, "notes.json"))
    assert [h["time"] for h in notes["tracks"]["bars"]] == [b["time"] for b in an.beats if b["pos"] == 0]
    assert notes["grid_offset"] == an.grid_offset                # patterns placed at beat 0, 4, 8 ... land on bars


def test_cache_upgrades_when_the_extra_arrives_and_classic_stays_classic(neural_on):
    d = tmpdir()
    clicks = 1.010 + 0.5 * np.arange(44)
    calls = []
    neural_on.setattr(neural, "activations", fake_activations(clicks, clicks[::4], calls))
    rp = _click_rig(d, clicks, 24.0)
    neural_on.setenv("BEATRIG_EXTRAS", "off")
    an = beatrig.analyze(beatrig.load_rig(rp))
    assert an.meta["beat_tracker"] == {"name": "classic", "why": "disabled"} and not calls
    neural_on.delenv("BEATRIG_EXTRAS")
    A._MEMO.clear()
    an = beatrig.analyze(beatrig.load_rig(rp))                   # the extra is there now: rebuilt with the model
    assert not an.loaded_from_cache and an.meta["beat_tracker"]["name"] == "neural" and len(calls) == 1
    d2 = tmpdir()
    rp2 = _click_rig(d2, clicks, 24.0, analysis={"beats": "classic"})
    A._MEMO.clear()
    an = beatrig.analyze(beatrig.load_rig(rp2))
    assert an.meta["beat_tracker"] == {"name": "classic", "why": "chosen"} and len(calls) == 1


def test_a_failing_model_leaves_the_classic_grid(neural_on):
    d = tmpdir()
    clicks = 1.010 + 0.5 * np.arange(44)
    def boom(*a, **k):
        raise OSError("no network")
    neural_on.setattr(neural, "activations", boom)
    an = beatrig.analyze(beatrig.load_rig(_click_rig(d, clicks, 24.0)))
    assert an.meta["beat_tracker"]["name"] == "classic" and "no network" in an.meta["beat_tracker"]["why"]
    assert len(an.beats) > 30 and "bar" not in an.beats[0]


def test_neural_setting_without_the_extra_is_an_error(monkeypatch):
    monkeypatch.setattr(neural, "available", lambda: (False, "beat-this is not installed: pip install beatrig[ml]"))
    A._MEMO.clear()
    d = tmpdir()
    rp = _click_rig(d, 1.010 + 0.5 * np.arange(10), 6.0, analysis={"beats": "neural"})
    assert cli(["analyze", rp]) == 1
    bad = beatrig.Rig.from_dict({"frames": 30, "analysis": {"beats": "dnn", "stems": "maybe"}, "nodes": [
        {"id": "b", "op": "beats", "anchor": "bars"}], "outputs": []}, d)
    wheres = {i["where"] for i in beatrig.check(bad) if i["level"] == "error"}
    assert {"rig.analysis.beats", "rig.analysis.stems"} <= wheres and any("anchor" in i["msg"] for i in beatrig.check(bad))


def test_availability_check_imports_nothing():
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from beatrig import neural\n"
            "sys.modules['beat_this'] = None\n"
            "ok, why = neural.available()\n"
            "print(ok, 'torch' in sys.modules, why)") % ROOT
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    # a core install reports torch first; with torch present, the hidden beat-this
    assert out.stdout.startswith("False False "), out.stdout
    assert "is not installed: pip install beatrig[ml]" in out.stdout, out.stdout


# ---------- the front end
def test_logmel_frames_and_pieces_join_exactly(monkeypatch):
    d = tmpdir()
    sr = 48000
    rng = np.random.default_rng(2)
    x = 0.1 * rng.standard_normal((int(7.3 * sr), 2))
    t = np.arange(len(x)) / sr
    x[:, 0] += 0.5 * np.sin(2 * np.pi * 1000 * t)
    import soundfile as sf
    sf.write(os.path.join(d, "n.wav"), x, sr, subtype="FLOAT")
    whole = neural._mono_22k(os.path.join(d, "n.wav"))
    monkeypatch.setattr(neural, "READ_S", 2)                      # 2 s pieces with context
    pieces = neural._mono_22k(os.path.join(d, "n.wav"))
    assert len(whole) == len(pieces) == math.ceil(len(x) * 22050 / sr)
    assert np.max(np.abs(whole - pieces)) < 1e-4
    S = neural.logmel(os.path.join(d, "n.wav"))
    assert S.shape == (1 + len(whole) // 441, 128) and S.dtype == np.float32
    fb = neural.mel_basis()
    band = int(np.argmax(fb[int(round(1000 / (22050 / 1024)))]))  # the mel band that holds 1 kHz
    assert int(np.argmax(S[50:-50].mean(axis=0))) == band


def test_logmel_matches_librosa():
    librosa = pytest.importorskip("librosa")
    d = tmpdir()
    write_loop(os.path.join(d, "l.wav"), 120, duration=6.0, sr=22050)
    y = neural._mono_22k(os.path.join(d, "l.wav"))
    ref = librosa.feature.melspectrogram(y=y, sr=22050, n_fft=1024, hop_length=441, power=1.0, n_mels=128, fmin=30, fmax=11000,
                                         htk=False, norm=None, center=True, pad_mode="reflect")
    ref = np.log1p(1000 * ref.T / np.sqrt(1024))
    assert np.max(np.abs(neural.logmel(os.path.join(d, "l.wav")) - ref)) < 1e-3


def test_chunks_cover_every_frame_once():
    for n in (10, 1488, 1500, 1501, 4000, 13692):
        starts = neural.chunk_starts(n)
        covered = np.zeros(n, int)
        for s in starts:
            covered[max(0, s + neural.BORDER):min(n, s + neural.CHUNK - neural.BORDER)] += 1
        assert covered.min() >= 1, n


# ---------- the real model (only with the [ml] extra; downloads the weights on first use)
def test_real_model_on_the_demo_track(monkeypatch):
    """the synthetic 110 BPM demo: a bass root per bar, so its bars are audible (a two-beat drum loop has none)"""
    pytest.importorskip("torch")
    pytest.importorskip("beat_this")
    if not neural.available()[0]:
        pytest.skip("the [ml] extra is not complete")
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_demo_audio", os.path.join(ROOT, "examples", "make_demo_audio.py"))
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    monkeypatch.delenv("BEATRIG_EXTRAS", raising=False)
    A._MEMO.clear()
    d = tmpdir()
    demo.make(os.path.join(d, "demo.wav"))
    rig = beatrig.Rig.from_dict({"audio": "demo.wav", "fps": FPS, "analysis": {"beats": "neural", "stems": "off"},
                                 "nodes": [], "outputs": []}, d)
    an = beatrig.analyze(rig)
    assert an.meta["beat_tracker"]["name"] == "neural"
    T = 60 / 110
    bt = np.array([b["time"] for b in an.beats if b["time"] < 29.9])   # the file ends where beat 55 would fall
    truth = T * np.arange(int(30 / T) + 1)
    assert len(bt) >= 50 and np.mean(A.nearest_gap(bt, truth) <= 0.07) == 1.0
    k = np.round(bt / T).astype(int)
    assert np.all(bt >= truth[k] - 1e-9)                         # never ahead of the hit
    assert np.mean(np.array([b["pos"] for b in an.beats if b["time"] < 29.9]) == k % 4) >= 0.9
    assert abs(an.bpm_map[0]["bpm"] - 110) < 0.3
