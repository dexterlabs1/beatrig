"""Drum parts and stems by default. Synthetic drums with exact truth: a kick with a broadband click on every beat, a
clap with a bright ringing tail on 2 and 4, closed hats on the off-beats. Plain band onsets put the kick's click in
the snare band and the clap's tail in the hats band; the drum op must not. Stems are faked (wavs written where a
separation would cache them), so none of this needs torch."""
import json
import math
import os

import numpy as np
import pytest
import soundfile as sf

from helpers import FPS, tmpdir
import beatrig
from beatrig import analysis as A
from beatrig import stems as S
from beatrig.cli import main as cli
from beatrig.drums import PARTS, part_hits

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM


def _sos_noise(rng, n, lo=None, hi=None):
    import scipy.signal as sg
    x = rng.standard_normal(n)
    if lo and hi:
        return sg.sosfilt(sg.butter(4, [lo, hi], btype="band", fs=SR, output="sos"), x)
    if lo:
        return sg.sosfilt(sg.butter(4, lo, btype="high", fs=SR, output="sos"), x)
    return x


def drums(duration=16.0, seed=4, pre_echo=0.0):
    """(drums signal, truth {part: times}); pre_echo: a quiet smear of each hit 90 ms ahead of it (a separated stem's)"""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    x = np.zeros(n)
    t = np.arange(int(0.4 * SR)) / SR
    kick = np.sin(2 * np.pi * np.cumsum(45 + 90 * np.exp(-t / 0.03)) / SR) * np.exp(-t / 0.18)
    kick[:int(0.004 * SR)] += 0.6 * _sos_noise(rng, int(0.004 * SR)) * np.exp(-np.arange(int(0.004 * SR)) / SR / 0.001)
    clap = 0.6 * _sos_noise(rng, len(t), 800, 12000) * np.exp(-t / 0.06)
    hat = 0.25 * _sos_noise(rng, len(t), 8000) * np.exp(-t / 0.012)
    truth = {"kick": [], "snare": [], "hats": []}
    def add(t0, s, part):
        i = int(round(t0 * SR))
        if i >= n:
            return
        m = min(len(s), n - i)
        x[i:i + m] += s[:m]
        if pre_echo:
            j = int(round((t0 - 0.09) * SR))
            if j >= 0:
                x[j:j + m] += pre_echo * s[:m]
        truth[part].append(t0)
    for b in range(int(duration / BEAT) - 1):
        tb = 0.5 + b * BEAT
        add(tb, kick, "kick")
        if b % 2 == 1:
            add(tb, clap, "snare")
        add(tb + BEAT / 2, hat, "hats")
    return x, {k: np.array(v) for k, v in truth.items()}


def pad(duration=16.0, seed=9):
    """a sustained, swelling synth chord with a fast tremolo in 1-5 kHz: rings on like a clap and wobbles like onsets"""
    t = np.arange(int(duration * SR)) / SR
    tone = sum(np.sin(2 * np.pi * f * t + k) for k, f in enumerate((1250, 1875, 2500, 3125)))
    return 0.25 * tone * (0.55 + 0.45 * np.sin(2 * np.pi * 6.0 * t)) * (0.6 + 0.4 * np.sin(2 * np.pi * t / 4))


def write(path, x):
    sf.write(path, np.stack((x, x), axis=1) / max(1.0, np.max(np.abs(x)) / 0.9), SR, subtype="FLOAT")
    return path


def prf(det, truth, early=0.002, late=0.05):
    used = np.zeros(len(truth), bool)
    tp = 0
    for t in det:
        j = np.flatnonzero((~used) & (t >= truth - early) & (t <= truth + late))
        if len(j):
            used[j[0]] = True
            tp += 1
    return tp / max(1, len(det)), tp / max(1, len(truth))


def _analysis(path):
    A._MEMO.clear()
    return A.open_analysis(path, os.path.dirname(path), {"beats": "classic"}, None)


def test_parts_from_a_drum_mix_beat_plain_band_onsets():
    d = tmpdir()
    x, truth = drums()
    a = _analysis(write(os.path.join(d, "drums.wav"), x))
    got = {p: (part_hits(a, p)[0] + 1) * 0.005 for p in PARTS}
    for p in PARTS:
        prec, rec = prf(got[p], truth[p])
        assert prec >= 0.95 and rec >= 0.9, (p, prec, rec)
    # the plain band onsets rigs used to write (snare 180-2500, hats 6000-16000) take the kick's click for a snare and
    # the clap's tail for a hat
    plain = lambda lo, hi: A.sg.find_peaks(a.onset_fn(lo, hi).astype(np.float64), distance=16)[0]
    f = a.onset_fn(180, 2500).astype(np.float64)
    pk = plain(180, 2500)
    pk = pk[f[pk] >= 0.3 * np.percentile(f[pk], 95)]
    assert prf((pk + 1) * 0.005, truth["snare"])[0] < 0.7
    hats = got["hats"]
    assert not np.any(np.min(np.abs(hats[:, None] - truth["snare"][None, :]), axis=1) < 0.03)   # no hat on a clap


def test_drum_hits_are_never_early():
    d = tmpdir()
    x, truth = drums()
    write(os.path.join(d, "drums.wav"), x)
    rig = beatrig.Rig.from_dict({"audio": "drums.wav", "fps": FPS, "nodes": [{"id": p, "op": "drum", "part": p} for p in PARTS],
                                 "outputs": []}, d)
    A._MEMO.clear()
    sig = beatrig.evaluate(rig, beatrig.analyze(rig), {})
    for p in PARTS:
        frames = np.flatnonzero(sig[p].v[0])
        first = np.array([math.ceil(t * FPS) for t in truth[p]])
        near = first[np.argmin(np.abs(first[None, :] - frames[:, None]), axis=1)]
        assert np.all(frames >= near) and np.all(frames - near <= 2), p      # on the first frame at/after the hit


def test_stem_pre_echo_is_not_a_hit():
    d = tmpdir()
    x, truth = drums(pre_echo=0.08)
    a = _analysis(write(os.path.join(d, "drums.wav"), x))
    for p in PARTS:
        prec, rec = prf((part_hits(a, p)[0] + 1) * 0.005, truth[p])
        assert prec >= 0.95 and rec >= 0.85, (p, prec, rec)


def _fake_stems(audio, drum_signal, duration=16.0):
    """write the four stems where a separation of `audio` caches them (drums = the clean drums, the rest silent)"""
    d = S.stem_dir(audio)
    os.makedirs(d, exist_ok=True)
    silence = np.zeros(int(duration * SR))
    for s in S.STEMS:
        write(os.path.join(d, s + ".wav"), drum_signal if s == "drums" else silence)
    with open(os.path.join(d, "stems.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": "fake", "device": "none", "seconds": 0.0}, fh)
    return {s: os.path.join(d, s + ".wav") for s in S.STEMS}


def _mix_rig(d, **analysis):
    x, truth = drums()
    write(os.path.join(d, "mix.wav"), x + pad())
    rig = {"audio": "mix.wav", "fps": FPS, "nodes": [{"id": "snare", "op": "drum", "part": "snare"},
                                                     {"id": "snareMix", "op": "drum", "part": "snare", "source": "mix"}],
           "outputs": []}
    if analysis:
        rig["analysis"] = analysis
    with open(os.path.join(d, "rig.json"), "w", encoding="utf-8") as fh:
        json.dump(rig, fh)
    return os.path.join(d, "rig.json"), x, truth


def _snare_times(sig, name):
    return np.flatnonzero(sig[name].v[0]) / FPS


def test_cached_stems_are_used_by_default_and_the_summary_says_so():
    d = tmpdir()
    rp, x, truth = _mix_rig(d)
    _fake_stems(os.path.join(d, "mix.wav"), x)
    A._MEMO.clear()
    rig = beatrig.load_rig(rp)
    an = beatrig.analyze(rig)
    assert an.stems_info["used"] and set(an.stems) == set(S.STEMS)
    assert an.stems_label().startswith("separated stems (bass, drums, other, vocals)") and "fake" in an.stems_label()
    assert an.summary()["stems"] == an.stems_label()
    sig = beatrig.evaluate(rig, an, {})
    on_stem = prf(_snare_times(sig, "snare"), truth["snare"], late=0.07)
    on_mix = prf(_snare_times(sig, "snareMix"), truth["snare"], late=0.07)
    assert on_stem[0] >= 0.95 and on_stem[1] >= 0.9
    assert on_mix[0] < on_stem[0]                                 # the ringing synth fools the mix, not the stem
    assert os.path.exists(os.path.join(S.stem_cache_dir(S.cached(os.path.join(d, "mix.wav"))["drums"]), "analysis.json"))


def test_stems_off_and_no_stems_fall_back_to_the_mix():
    d = tmpdir()
    rp, x, truth = _mix_rig(d, stems="off")
    _fake_stems(os.path.join(d, "mix.wav"), x)
    A._MEMO.clear()
    rig = beatrig.load_rig(rp)
    an = beatrig.analyze(rig)
    assert not an.stems_info["used"] and an.stems == {} and "analysis.stems is off" in an.stems_label()
    sig = beatrig.evaluate(rig, an, {})
    assert np.array_equal(sig["snare"].v, sig["snareMix"].v)      # auto reads the mix
    d2 = tmpdir()
    rp2, _, _ = _mix_rig(d2)
    A._MEMO.clear()
    an = beatrig.analyze(beatrig.load_rig(rp2))                   # BEATRIG_EXTRAS=off (the test default): no separation
    assert an.stems_label() == "the mix (BEATRIG_EXTRAS=off)"
    bad = beatrig.Rig.from_dict(dict(json.load(open(rp2, encoding="utf-8")), nodes=[
        {"id": "s", "op": "drum", "part": "snare", "source": "stem"}]), d2)
    with pytest.raises(beatrig.OpError, match="needs the drums stem"):
        beatrig.evaluate(bad, an, {})


def test_auto_separates_once_on_a_cuda_gpu(monkeypatch):
    d = tmpdir()
    rp, x, truth = _mix_rig(d)
    calls = []
    def fake_separate(audio, log=print, **kw):
        calls.append(audio)
        return _fake_stems(audio, x)
    monkeypatch.delenv("BEATRIG_EXTRAS", raising=False)
    monkeypatch.setattr(S, "available", lambda: (True, ""))
    monkeypatch.setattr(S, "separate", fake_separate)
    monkeypatch.setattr(S, "cuda_available", lambda: False)
    A._MEMO.clear()
    an = beatrig.analyze(beatrig.load_rig(rp))
    assert not calls and "no CUDA GPU" in an.stems_label()        # CPU only: auto does not separate
    monkeypatch.setattr(S, "cuda_available", lambda: True)
    A._MEMO.clear()
    an = beatrig.analyze(beatrig.load_rig(rp))
    assert len(calls) == 1 and an.stems_info["used"]
    A._MEMO.clear()
    beatrig.analyze(beatrig.load_rig(rp))
    assert len(calls) == 1                                        # cached next to the audio: never twice


def test_drum_checks_and_notes_init():
    d = tmpdir()
    x, truth = drums()
    write(os.path.join(d, "drums.wav"), x)
    rig = {"audio": "drums.wav", "fps": FPS, "nodes": [], "outputs": []}
    with open(os.path.join(d, "rig.json"), "w", encoding="utf-8") as fh:
        json.dump(rig, fh)
    bad = beatrig.Rig.from_dict(dict(rig, nodes=[{"id": "a", "op": "drum", "part": "cowbell"},
                                                 {"id": "b", "op": "drum", "part": "kick", "source": "stems"},
                                                 {"id": "c", "op": "drum"}]), d)
    msgs = [i["msg"] for i in beatrig.check(bad) if i["level"] == "error"]
    assert len(msgs) >= 3 and any("cowbell" in m for m in msgs)
    A._MEMO.clear()
    assert cli(["notes", "init", os.path.join(d, "rig.json"), "--track", "kick", "--from", "drum:kick"]) == 0
    assert cli(["notes", "init", os.path.join(d, "rig.json"), "--track", "x", "--from", "drum:cowbell"]) == 2
    hits = np.array([h["time"] for h in beatrig.load_notes(os.path.join(d, "notes.json"))["tracks"]["kick"]])
    assert prf(hits, truth["kick"]) >= (0.95, 0.9)
