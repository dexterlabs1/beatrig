"""Song structure: labelled sections on bar lines. A synthetic 120 BPM song with known parts (intro: kick and hats;
groove: + bass and clap; breakdown: a pad only; build: kick, clap and a rising sweep, no bass; drop: everything and a
lead; outro: hats and pad), analysed from the mix alone and with (faked) stems."""
import json
import os

import numpy as np
import scipy.signal as sg
import soundfile as sf

from helpers import FPS, tmpdir
import beatrig
from beatrig import analysis as A
from beatrig import stems as S
from beatrig.cli import main as cli

SR = 22050
BAR = 2.0                                    # 4 beats at 120 BPM
PARTS = [("intro", 8), ("groove", 8), ("breakdown", 8), ("build", 8), ("drop", 8), ("outro", 4)]
DROPOUT_BAR = 13                             # the groove's bar 6: kick, clap and bass stop for one bar


def song(dropout=False, seed=1):
    """{"drums", "bass", "other"} signals and the start bar of each part"""
    rng = np.random.default_rng(seed)
    n = int(sum(b for _, b in PARTS) * BAR * SR)
    out = {k: np.zeros(n) for k in ("drums", "bass", "other")}
    t = np.arange(int(0.5 * SR)) / SR
    kick = np.sin(2 * np.pi * np.cumsum(45 + 80 * np.exp(-t / 0.03)) / SR) * np.exp(-t / 0.15)
    clap = 0.5 * sg.sosfilt(sg.butter(2, [900, 7000], btype="band", fs=SR, output="sos"), rng.standard_normal(len(t))) * np.exp(-t / 0.07)
    hat = 0.15 * sg.sosfilt(sg.butter(4, 7000, btype="high", fs=SR, output="sos"), rng.standard_normal(len(t))) * np.exp(-t / 0.02)
    def add(k, t0, s, g=1.0):
        i = int(round(t0 * SR))
        m = min(len(s), n - i)
        if m > 0:
            out[k][i:i + m] += g * s[:m]
    starts, bar0 = {}, 0
    for name, bars in PARTS:
        starts[name] = bar0
        for b in range(bar0, bar0 + bars):
            tb = b * BAR
            drop_out = dropout and b == DROPOUT_BAR
            kick_on = name in ("intro", "groove", "build", "drop") and not drop_out
            bass_on = name in ("groove", "drop") and not drop_out
            for q in range(4):
                if kick_on:
                    add("drums", tb + q * 0.5, kick, 0.9)
                if name in ("groove", "build", "drop") and q in (1, 3) and not drop_out:
                    add("drums", tb + q * 0.5, clap)
                if name in ("intro", "groove", "drop", "outro"):
                    add("drums", tb + q * 0.5 + 0.25, hat)
            if bass_on:
                tt = np.arange(int(BAR * SR)) / SR
                saw = 2 * ((55 * tt) % 1.0) - 1
                add("bass", tb, 0.5 * sg.sosfilt(sg.butter(2, 250, fs=SR, output="sos"), saw))
            if name in ("breakdown", "outro"):
                tt = np.arange(int(BAR * SR)) / SR
                add("other", tb, 0.12 * sum(np.sin(2 * np.pi * f * tt) for f in (220, 277, 330, 440)))
            if name == "build":
                tt = np.arange(int(BAR * SR)) / SR
                level = 0.02 * 10 ** (1.5 * (b - bar0) / bars)       # a noise riser climbing ~30 dB over the build
                add("other", tb, level * sg.sosfilt(sg.butter(2, [1000, 6000], btype="band", fs=SR, output="sos"),
                                                   rng.standard_normal(len(tt))))
            if name == "drop":
                tt = np.arange(int(BAR * SR)) / SR
                add("other", tb, 0.15 * (2 * ((440 * tt) % 1.0) - 1))
        bar0 += bars
    return out, starts


def _rig(d, stems=None, dropout=False):
    parts, starts = song(dropout)
    mix = sum(parts.values())
    peak = np.max(np.abs(mix)) / 0.9
    sf.write(os.path.join(d, "song.wav"), mix / peak, SR, subtype="FLOAT")
    if stems:
        sd = S.stem_dir(os.path.join(d, "song.wav"))
        os.makedirs(sd, exist_ok=True)
        for s in S.STEMS:
            x = parts.get(s, np.zeros_like(mix)) / peak
            sf.write(os.path.join(sd, s + ".wav"), np.stack((x, x), axis=1), SR, subtype="FLOAT")
    rig = {"audio": "song.wav", "fps": FPS, "nodes": [], "outputs": []}
    with open(os.path.join(d, "rig.json"), "w", encoding="utf-8") as fh:
        json.dump(rig, fh)
    return os.path.join(d, "rig.json"), starts


def _check(secs, starts, an):
    bars = [s["bar"] for s in secs]
    want = [starts[p] for p, _ in PARTS]
    for b in want:                                             # every part starts within a bar of a section
        assert min(abs(b - x) for x in bars) <= 1, (b, bars)
    lab = {p: [s for s in secs if s["bar"] <= starts[p] + 1][-1]["label"] for p, _ in PARTS}
    assert lab["intro"] == "intro" and lab["breakdown"] == "breakdown" and lab["build"] == "build"
    assert lab["drop"] == "drop" and lab["outro"] == "outro" and lab["groove"] in ("groove", "drop")
    edges, _ = A.bar_lines(an)
    for s in secs[1:]:                                         # on a bar line, never before it
        assert abs(s["time"] - edges[s["bar"]]) < 1e-3
        assert s["time"] >= s["bar"] * BAR - 1e-6


def test_structure_from_the_mix():
    d = tmpdir()
    rp, starts = _rig(d)
    A._MEMO.clear()
    an = beatrig.analyze(beatrig.load_rig(rp))
    assert not an.stems_info["used"]
    _check(A.song_structure(an), starts, an)


def test_structure_with_stems_and_cli():
    d = tmpdir()
    rp, starts = _rig(d, stems=True)
    A._MEMO.clear()
    an = beatrig.analyze(beatrig.load_rig(rp))
    assert an.stems_info["used"]
    secs = A.song_structure(an)
    _check(secs, starts, an)
    names = [s["name"] for s in secs]
    assert len(set(names)) == len(names)                      # repeats are numbered
    assert cli(["notes", "sections", "init", rp]) == 0
    notes = beatrig.load_notes(os.path.join(d, "notes.json"))
    assert [s["name"] for s in notes["sections"]] == names and notes["sections"][0]["time"] == 0.0


def test_a_one_bar_dropout_starts_a_section_after_it():
    d = tmpdir()
    rp, starts = _rig(d, stems=True, dropout=True)
    A._MEMO.clear()
    an = beatrig.analyze(beatrig.load_rig(rp))
    secs = A.song_structure(an, min_bars=2)
    assert any(s["bar"] == DROPOUT_BAR + 1 for s in secs), [(s["bar"], s["name"]) for s in secs]


def test_bar_lines_follow_the_downbeats():
    class Fake:
        meta = {"duration": 20.0, "bpm_map": [{"time": 0, "bpm": 120.0}], "grid_offset": 0.3,
                "bars": {"meter": 4, "downbeats": [1.1, 3.1, 5.1, 7.1]}}
    edges, bar_len = A.bar_lines(Fake())
    assert bar_len == 2.0 and edges[0] == 0.0 and np.allclose(edges[1:], 1.1 + 2.0 * np.arange(10))
    Fake.meta = dict(Fake.meta, bars=None)
    edges, _ = A.bar_lines(Fake())                            # no downbeats: bars of 4 beats from the grid offset
    assert edges[0] == 0.0 and np.allclose(edges[1:], 0.3 + 2.0 * np.arange(1, 10))
