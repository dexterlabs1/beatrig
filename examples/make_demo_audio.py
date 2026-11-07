"""Deterministic synthetic demo track for the beatrig examples and tests.

    py -3 examples/make_demo_audio.py [out.wav] [--force]

110 BPM, 30 s, 48 kHz stereo, 16-bit. Kick on every beat, snare on 2 and 4, eighth-note hats, an eighth-note bass line.
Structure (bars of 4 beats, 2.18 s each):
  bars 0-5    groove
  bar  6      groove + a snare fill over beats 3-4
  bars 7-9    breakdown: no kick or snare, sustained bass + a swelling pad, quiet hats
  bars 10-13  groove (the track ends inside bar 13)
Same output bytes on every run (fixed seed, no dither, no timestamps).
"""
import argparse
import os

import numpy as np
import scipy.signal as sg
import soundfile as sf

SR = 48000
BPM = 110.0
DURATION = 30.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ring-demo", "audio", "demo_110bpm.wav")


def section(bar):
    if 7 <= bar <= 9:
        return "breakdown"
    if bar == 6:
        return "fill"
    return "groove"


def add(buf, t0, sig, gain=(1.0, 1.0)):
    i = int(round(t0 * SR))
    if i >= buf.shape[0]:
        return
    n = min(len(sig), buf.shape[0] - i)
    buf[i:i + n, 0] += sig[:n] * gain[0]
    buf[i:i + n, 1] += sig[:n] * gain[1]


def kick():
    t = np.arange(int(0.45 * SR)) / SR
    f = 45 + 80 * np.exp(-t / 0.03)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t / 0.22) + 0.3 * np.exp(-t / 0.002)


def snare(rng):
    t = np.arange(int(0.3 * SR)) / SR
    noise = sg.sosfilt(sg.butter(2, [900, 7000], btype="band", fs=SR, output="sos"), rng.standard_normal(len(t)))
    return 0.9 * noise * np.exp(-t / 0.1) + 0.5 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.06)


def hat(rng, open_=False):
    t = np.arange(int((0.18 if open_ else 0.06) * SR)) / SR
    noise = sg.sosfilt(sg.butter(4, 7000, btype="high", fs=SR, output="sos"), rng.standard_normal(len(t)))
    return noise * np.exp(-t / (0.06 if open_ else 0.015))


def bass_note(freq, length):
    t = np.arange(int(length * SR)) / SR
    saw = 2 * ((freq * t) % 1.0) - 1
    saw = sg.sosfilt(sg.butter(2, 300, fs=SR, output="sos"), saw)
    env = np.minimum(1, t / 0.005) * np.exp(-t / max(length, 0.05) * 1.5)
    return saw * env


def pad(length):
    t = np.arange(int(length * SR)) / SR
    tone = sum(np.sin(2 * np.pi * f * t + k) for k, f in enumerate((164.8, 207.7, 246.9, 329.6)))
    env = np.sin(np.pi * np.clip(t / length, 0, 1)) ** 2
    return 0.12 * tone * env


def make(path=DEFAULT_OUT):
    rng = np.random.default_rng(110)
    n = int(DURATION * SR)
    buf = np.zeros((n, 2))
    K, S = kick(), snare(rng)
    roots = [41.2, 41.2, 49.0, 36.7]                          # E1 E1 G1 D1, one per bar
    bars = int(np.ceil(DURATION / BAR))
    for bar in range(bars):
        sec = section(bar)
        t_bar = bar * BAR
        root = roots[bar % 4]
        if sec == "breakdown":
            if bar == 7:
                add(buf, t_bar, pad(3 * BAR), (0.9, 1.0))
            add(buf, t_bar, 0.5 * bass_note(root, BAR * 0.95))
            for e in range(8):
                add(buf, t_bar + e * BEAT / 2, 0.08 * hat(rng), (0.7, 1.0))
            continue
        for b in range(4):
            tb = t_bar + b * BEAT
            add(buf, tb, 0.9 * K)
            if b in (1, 3):
                add(buf, tb, 0.55 * S)
            for e in range(2):
                add(buf, tb + e * BEAT / 2, (0.22 if e else 0.12) * hat(rng, open_=(e == 1 and b == 3)), (1.0, 0.75))
                add(buf, tb + e * BEAT / 2, 0.35 * bass_note(root * (2 if (b == 2 and e == 1) else 1), BEAT / 2 * 0.9))
        if sec == "fill":
            for s in range(8):                                # sixteenths over beats 3-4, rising
                add(buf, t_bar + 2 * BEAT + s * BEAT / 4, (0.25 + 0.05 * s) * S)
    buf *= 0.89 / np.max(np.abs(buf))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, np.round(buf * 32767) / 32768, SR, subtype="PCM_16")
    return path


def ensure(path=DEFAULT_OUT):
    """generate the demo track if it is missing; returns its path"""
    if not os.path.exists(path):
        make(path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out) and not a.force:
        print(f"exists: {a.out} (--force to rewrite)")
    else:
        print(f"wrote {make(a.out)}")
