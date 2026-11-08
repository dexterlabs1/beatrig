"""Shared test fixtures: synthetic audio, temp rig folders, an op-level context without audio."""
import atexit
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# "auto" analysis settings behave like a core install unless a test asks for the extras itself, so the suite gives the
# same results with or without torch / demucs / beat-this installed (and never separates every test clip)
os.environ.setdefault("BEATRIG_EXTRAS", "off")

from beatrig.ops import Ctx  # noqa: E402
from beatrig.tempo import Tempo  # noqa: E402

SR = 48000
FPS = 30
_TMP = []


def tmpdir():
    d = tempfile.mkdtemp(prefix="beatrig_test_")
    _TMP.append(d)
    return d


@atexit.register
def _cleanup():
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)


def click_times(duration=24.0, first=1.010, period=0.5):
    """clicks a little after a frame boundary (1.010 s is inside frame 30, so the first frame at or after it is 31)"""
    return np.arange(first, duration - 0.5, period)


def write_clicks(path, times, duration=24.0, seed=3):
    """stereo click track: a broadband noise burst plus a 60 Hz thump at each time, silence elsewhere"""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    x = np.zeros((n, 2))
    t = np.arange(int(0.12 * SR)) / SR
    burst = rng.standard_normal(len(t)) * np.exp(-t / 0.004)
    thump = np.sin(2 * np.pi * 60 * t) * np.exp(-t / 0.03)
    for tc in times:
        i = int(round(tc * SR))
        m = min(len(t), n - i)
        x[i:i + m, 0] += 0.5 * burst[:m] + 0.6 * thump[:m]
        x[i:i + m, 1] += 0.4 * burst[:m] + 0.6 * thump[:m]
    sf.write(path, x / max(1.0, np.max(np.abs(x)) / 0.9), SR, subtype="FLOAT")
    return path


def rig_folder(rig, notes=None, audio=None, times=None, duration=24.0):
    d = tmpdir()
    if audio is None:
        write_clicks(os.path.join(d, "clicks.wav"), click_times(duration) if times is None else times, duration)
        rig.setdefault("audio", "clicks.wav")
    with open(os.path.join(d, "rig.json"), "w", encoding="utf-8") as fh:
        json.dump(rig, fh, indent=1)
    if notes is not None:
        with open(os.path.join(d, "notes.json"), "w", encoding="utf-8") as fh:
            json.dump(notes, fh, indent=1)
    return d


def ctx(frames=300, bpm=120.0, fps=FPS, seed=1):
    c = Ctx(fps, frames, 0, Tempo([(0.0, bpm)], 0.0), None, {}, seed)
    c.rig_nodes = []
    return c


def write_loop(path, bpm, duration=16.0, sr=44100, seed=5):
    """perfectly regular mono drum loop: kick every beat, snare on 2 and 4, hats on the eighths"""
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    x = np.zeros(n)
    beat = 60.0 / bpm
    t = np.arange(int(0.4 * sr)) / sr
    kick = np.sin(2 * np.pi * np.cumsum(45 + 95 * np.exp(-t * 30)) / sr) * np.exp(-t * 9)
    snare = (0.5 * rng.standard_normal(len(t)) + 0.3 * np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 18)
    hat = 0.18 * np.diff(rng.standard_normal(len(t) + 1)) * np.exp(-t * 60)
    def add(t0, s):
        i = int(round(t0 * sr))
        if i < n:
            m = min(len(s), n - i)
            x[i:i + m] += s[:m]
    for b in range(int(duration / beat) + 1):
        add(b * beat, 0.9 * kick)
        if b % 4 in (1, 3):
            add(b * beat, snare)
        add(b * beat, hat)
        add(b * beat + beat / 2, hat)
    sf.write(path, x / np.max(np.abs(x)) * 0.85, sr, subtype="PCM_16")
    return path
