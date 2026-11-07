"""Tempo: one piecewise-constant bpm map in audio seconds, plus a grid offset. Beat position = integral of bpm/60 from
grid_offset. Serves rig.tempo.map (frames), notes.json bpm_map (seconds), length_beats, step_beats and rate_beats."""
import numpy as np


class Tempo:
    def __init__(self, segments, grid_offset=0.0):
        """segments: [(time_s, bpm), ...]; the first segment also covers everything before its time"""
        segs = sorted((float(t), float(b)) for t, b in segments) or [(0.0, 120.0)]
        if any(b <= 0 for _, b in segs):
            raise ValueError("bpm must be > 0")
        self.t = np.array([s[0] for s in segs])
        self.bpm = np.array([s[1] for s in segs])
        # cumulative beats at each segment start (segment 0 starts at beat 0)
        self.B = np.concatenate(([0.0], np.cumsum(np.diff(self.t) * self.bpm[:-1] / 60.0)))
        self.grid_offset = float(grid_offset)
        self._b0 = float(self._raw(np.array([self.grid_offset]))[0])

    def _seg(self, t):
        return np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1)

    def _raw(self, t):
        t = np.asarray(t, np.float64)
        i = self._seg(t)
        return self.B[i] + (t - self.t[i]) * self.bpm[i] / 60.0

    def bpm_at(self, t):
        return self.bpm[self._seg(np.asarray(t, np.float64))]

    def beat_seconds(self, t):
        return 60.0 / self.bpm_at(t)

    def beats_at(self, t):
        """beat position at audio time t (0 at grid_offset)"""
        return self._raw(t) - self._b0

    def time_of_beat(self, b):
        """inverse of beats_at"""
        raw = np.asarray(b, np.float64) + self._b0
        i = np.clip(np.searchsorted(self.B, raw, side="right") - 1, 0, len(self.B) - 1)
        return self.t[i] + (raw - self.B[i]) * 60.0 / self.bpm[i]

    def to_map(self):
        return [{"time": round(float(t), 6), "bpm": round(float(b), 4)} for t, b in zip(self.t, self.bpm)]
