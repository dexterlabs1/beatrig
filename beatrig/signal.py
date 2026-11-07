"""Signal: float32[channels, frames] plus a hit mask (frames RDP must keep) and, for triggers, per-hit metadata."""
from typing import Optional

import numpy as np

SHAPES = ["binary", "punch", "gloss", "razor", "rush", "adsr", "saw_down", "custom"]


class Signal:
    """v: float32 [C, N] (or [C, N, 4] linear RGBA for kind "color"). kind: "value" | "trigger" | "color".
    hits: bool [C, N], frames that must survive key simplification.
    aux (triggers only, optional): "len" float32 [C, N] per-hit length in beats (NaN = op default),
    "shape" int16 [C, N] per-hit index into SHAPES (-1 = op default)."""
    __slots__ = ("v", "kind", "hits", "aux")

    def __init__(self, v, kind: str = "value", hits=None, aux: Optional[dict] = None):
        v = np.asarray(v, dtype=np.float32)
        if v.ndim == 1:
            v = v[None, :]
        if kind == "color" and v.ndim == 2:
            v = v[None, :, :]
        self.v = v
        self.kind = kind
        shape = v.shape[:2]
        if hits is None:
            hits = (v != 0) if kind == "trigger" else np.zeros(shape, bool)
        self.hits = np.broadcast_to(np.asarray(hits, bool), shape).copy() if np.ndim(hits) == 0 else np.asarray(hits, bool).reshape(shape)
        self.aux = aux or {}

    @property
    def channels(self):
        return self.v.shape[0]

    @property
    def frames(self):
        return self.v.shape[1]

    def take(self, idx):
        """channel permutation / selection (idx: list of input channel indices, one per output channel)"""
        idx = list(idx)
        return Signal(self.v[idx], self.kind, self.hits[idx], {k: a[idx] for k, a in self.aux.items()})

    def broadcast(self, n):
        if self.channels == n:
            return self
        if self.channels != 1:
            raise ValueError(f"cannot broadcast {self.channels} channels to {n}")
        return self.take([0] * n)

    def __repr__(self):
        return f"Signal({self.kind}, {self.channels}x{self.frames})"


def empty_aux(shape):
    return {"len": np.full(shape, np.nan, np.float32), "shape": np.full(shape, -1, np.int16)}


def frame_delays(d):
    """delay in frames (float array); values within 1e-6 of an integer snap to it so float noise never adds a frame"""
    d = np.asarray(d, dtype=np.float64)
    r = np.round(d)
    return np.where(np.abs(d - r) < 1e-6, r, d)


def delay(sig, d):
    """delay each channel by d frames: anything broadcastable to [C, N] (scalar, [C, 1] per channel, [N] per frame);
    must be >= 0. Values: linear interpolation between the two past frames (causal: both are at or before f).
    Triggers: each hit moves to the first frame at or after hit + d, so a delayed hit is never early. The hit mask
    moves the same way."""
    C, N = sig.v.shape[:2]
    d = frame_delays(np.broadcast_to(np.asarray(d, np.float64), (C, N)))
    if np.any(d < 0):
        raise ValueError("negative delay: visuals may not lead the audio")
    if sig.kind == "color":                                    # each RGBA component like a value signal
        parts = [delay(Signal(sig.v[..., k], "value", sig.hits), d) for k in range(sig.v.shape[2])]
        return Signal(np.stack([p.v for p in parts], axis=-1), "color", parts[0].hits)
    if sig.kind == "trigger":
        v = np.zeros_like(sig.v)
        aux = {k: np.full_like(a, np.nan if k == "len" else -1) for k, a in sig.aux.items()}
        hits = np.zeros_like(sig.hits)
        for c in range(C):
            src = np.flatnonzero(sig.hits[c] | (sig.v[c] != 0))
            dst = src + np.ceil(d[c, src] - 1e-9).astype(np.int64)
            ok = dst < N
            src, dst = src[ok], dst[ok]
            # collisions: keep the strongest (process weakest first so the strongest writes last)
            order = np.argsort(sig.v[c, src], kind="stable")
            src, dst = src[order], dst[order]
            v[c, dst] = sig.v[c, src]
            hits[c, dst] = True
            for k, a in sig.aux.items():
                aux[k][c, dst] = a[c, src]
        return Signal(v, "trigger", hits, aux)
    f = np.arange(N)[None, :] - d
    i0 = np.floor(f).astype(np.int64)
    w = (f - i0).astype(np.float32)
    rows = np.arange(C)[:, None]
    def at(i):
        return np.where(i >= 0, sig.v[rows, np.clip(i, 0, N - 1)], 0.0)
    v = at(i0) * (1 - w) + at(i0 + 1) * w
    hits = np.zeros_like(sig.hits)
    for c in range(C):
        src = np.flatnonzero(sig.hits[c])
        dst = src + np.ceil(d[c, src] - 1e-9).astype(np.int64)
        hits[c, dst[dst < N]] = True
    return Signal(v.astype(np.float32), "value", hits)
