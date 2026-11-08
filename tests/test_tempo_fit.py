"""The classic beat grid's tempo: a line fit through the verified beats, each indexed by the periods between
neighbours. The comb's period sits on the 5 ms lag grid (0.470 s for a 0.46875 s beat); indexing every beat by
round((t - t0) / T) let that error pile up until far beats took the wrong index and the tempo map drifted."""
import numpy as np

from helpers import ctx  # noqa: F401  (puts the repo on sys.path)
from beatrig import analysis as A
from beatrig.tempo import Tempo


def _onset_function(times, duration, seed=7):
    """a 5 ms onset function with a unit peak at each time (index k is known at (k + 1) * 5 ms) over a small noise floor"""
    rng = np.random.default_rng(seed)
    f = 0.02 * rng.random(int(duration / 0.005))
    for t in times:
        k = int(round(t / 0.005)) - 1
        f[k] = 1.0
        f[k + 1] = max(f[k + 1], 0.3)
    return f


def test_long_track_tempo_does_not_drift():
    bpm, off, duration = 128.0, 0.23, 240.0
    times = off + np.arange(int((duration - 1) / (60 / bpm))) * 60 / bpm
    g = A.beat_grid(_onset_function(times, duration), times, duration, dict(A.DEFAULT_SETTINGS))
    assert not g["fallback"] and len(g["passages"]) == 1
    assert abs(g["bpm_map"][0]["bpm"] - bpm) < 0.02, g["bpm_map"]
    beats = np.array([b["time"] for b in g["beats"]])
    on_click = A.nearest_gap(beats, times) <= 0.0051
    assert np.sum(on_click) >= len(times) - 2 and np.mean(on_click) > 0.99
    # the tempo map (what lfo, patterns, quantize and beat delays count on) still lands on the last beats
    tempo = Tempo([(m["time"], m["bpm"]) for m in g["bpm_map"]], g["grid_offset"])
    n_last = np.round(tempo.beats_at(beats[-1]))
    assert abs(float(tempo.time_of_beat(n_last)) - beats[-1]) < 0.02
