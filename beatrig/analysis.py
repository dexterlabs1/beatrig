"""Causal audio analysis and its cache.

Timing rule (ITU-R BT.1359 audio/video timing): visuals are never ahead of the audio. Everything that reaches a curve
is causal (sosfilt and one-pole followers over past samples only), a frame reads the audio up to its own start time,
and a hit lands on the first frame at or after its onset.

The beat grid is not a plain tempo tracker (those happily lock to a double or half tempo, or to texture). A passage
only gets beats when its onset autocorrelation clearly beats a block-shuffled null (periodicity destroyed, texture
kept) and its period holds across windows. A beat is only written where a real onset peak exists, so missing notes
stay missing and tempo drift is followed rather than gridded.

Grids: band/loudness envelopes on an exact 1 ms grid, the onset function on a 5 ms grid. Block j covers samples
[floor(j*sr*step/1000), floor((j+1)*sr*step/1000)) and its value is known only at the block's END, so grid index k
is the value at (k + 1) * step ms. Audio is read in 10 s chunks (a whole number of 1 ms and 5 ms blocks at any sample
rate) with the filter state carried across chunks, so a 60-min show never sits in memory as float64.

The beats are checked against an independent reference onset list: log-mel spectral flux on a ~12 kHz mono copy
(21 ms Hann window, 5 ms hop, 40 mel bands, peak-picked), streamed through the same chunk loop. It is a different
method from the octave-band flux that places the beats, so the agreement test is not circular.
"""
import hashlib
import json
import os
from typing import Optional

import numpy as np
import soundfile as sf


class _Lazy:
    """a module imported on first attribute use: scipy.signal is most of beatrig's import time, and check, the MCP
    server and the preview server often never need it"""

    def __init__(self, name):
        self._name, self._mod = name, None

    def __getattr__(self, attr):
        if self._mod is None:
            import importlib
            self._mod = importlib.import_module(self._name)
        return getattr(self._mod, attr)


sg = _Lazy("scipy.signal")

ANALYSIS_VERSION = 5
ONSET_EDGES = [30, 60, 120, 240, 480, 960, 1920, 3840, 7680, 15360]   # octave bands for the onset function
STEP_MS = 1                 # envelope grid
HOP_MS = 5                  # onset grid
CHUNK_S = 10                # read size
REF_NFFT, REF_MELS = 256, 40
ENV_KEEP = 24               # derived envelopes kept in memory per analysis (never written to the cache)
DEFAULT_SETTINGS = dict(bpm_min=75.0, bpm_max=150.0, pulse_z=4.0, min_conf=0.15, min_agree=0.15, win_s=16.0, hop_s=4.0,
                        beats="auto", stems="auto")
BEAT_TRACKERS = ("auto", "neural", "classic")
STEM_MODES = ("auto", "on", "off")
NOT_IN_IDENT = ("stems",)   # settings that never change the mix's analysis (they choose what is opened next to it)
# tempo candidates: the metrical levels a tempo estimate is most often confused with
CANDIDATE_RATIOS = [("x1", 1.0), ("x2", 2.0), ("x1/2", 0.5), ("x3/2", 1.5), ("x2/3", 2 / 3), ("x4/3", 4 / 3), ("x3/4", 0.75)]

_MEMO = {}                  # (audio, size, mtime, settings hash) -> Analysis, so the preview server re-bakes fast


# ---------- filters / grids
def k_weight(sr):
    """ITU-R BS.1770 pre-filter (high shelf + RLB high-pass) for any sample rate; at 48 kHz this is the BS.1770 table"""
    G, f0, Q = 3.999843853973347, 1681.974450955533, 0.7071752369554196
    K = np.tan(np.pi * f0 / sr)
    Vh = 10 ** (G / 20)
    Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    s1 = [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0, 1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = np.tan(np.pi * f0 / sr)
    a0 = 1 + K / Q + K * K
    s2 = [1.0, -2.0, 1.0, 1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    return np.array([s1, s2])


def band_sos(lo, hi, sr, order=4):
    nyq = sr / 2
    lo, hi = float(lo or 0), float(hi or nyq)
    if lo <= 0 and hi >= nyq * 0.999:
        return None
    if lo <= 0:
        return sg.butter(order, hi, btype="low", fs=sr, output="sos")
    if hi >= nyq * 0.999:
        return sg.butter(order, lo, btype="high", fs=sr, output="sos")
    return sg.butter(order, [lo, hi], btype="band", fs=sr, output="sos")


def grid_power(y, sr, step_ms):
    """mean power per step_ms block along axis 0 (exact boundaries at any sample rate; trailing partial block dropped).
    A block's value is known only at its END (causal)."""
    n = y.shape[0]
    nb = (n * 1000) // (sr * step_ms)
    if (sr * step_ms) % 1000 == 0:                         # whole samples per block (48 kHz): a reshape, no cumsum
        m = sr * step_ms // 1000
        sq = np.square(y[:nb * m])
        return sq.reshape((nb, m) + y.shape[1:]).mean(axis=1)
    e = (np.arange(nb + 1) * sr * step_ms) // 1000
    if nb == 0:
        return np.zeros((0,) + y.shape[1:])
    return np.add.reduceat(np.square(y[:e[-1]]), e[:-1], axis=0) / np.diff(e).reshape((-1,) + (1,) * (y.ndim - 1))


def follow(x, attack_ms, release_ms, dt_ms=1.0):
    """one-pole up/down follower on past samples only (~0.5 s per 3.6M steps, i.e. a 60-min show on the 1 ms grid)"""
    a, r = 1 - np.exp(-dt_ms / attack_ms), 1 - np.exp(-dt_ms / release_ms)
    y = np.empty(len(x), dtype=np.float64)
    v = 0.0
    for i, s in enumerate(np.asarray(x, np.float64).tolist()):
        v += (a if s > v else r) * (s - v)
        y[i] = v
    return y


def at_frames(env, step_ms, frames, fps, offset=0, t0=0.0):
    """value for sequence frame f = env at the frame's start time (audio time t0 + (f - offset) / fps); never reads
    the future. Grid index k is known at (k + 1) * step ms, so frame f reads k = floor(t_ms / step) - 1. Frames before
    the audio read silence (0); frames past the end hold the last value. t0 = rig audio_in_s (a window into a long file)."""
    t_ms = (np.arange(frames) - offset) * 1000.0 / fps
    if t0:
        t_ms = np.round(t_ms + float(t0) * 1000.0, 6)
    k = np.floor(t_ms / step_ms).astype(np.int64) - 1
    if len(env) == 0:
        return np.zeros(frames, np.float32)
    return np.where(k >= 0, env[np.clip(k, 0, len(env) - 1)], 0.0).astype(np.float32)


def time_to_frame(t, fps, offset=0, t0=0.0):
    """first sequence frame whose start is at or after audio time t (t0 = audio time of frame `offset`)"""
    # round to 1e-4 frame before ceil: notes store times to 1 us, so a hit saved exactly on a frame start must not slip a
    # frame late; 1e-4 frame (3 us at 30 fps) is far below anything audible or visible
    return (np.ceil(np.round((np.asarray(t, np.float64) - float(t0)) * fps, 4)).astype(np.int64) + int(offset))


def norm_db(v, range_db, hi_pct=99.5, lo_pct=10):
    """0..1 on a dB scale over the whole range: hi_pct -> 1, floor = max(lo_pct, top - range_db) -> 0.
    A static gain only: timing is unaffected."""
    d = 20 * np.log10(np.maximum(np.asarray(v, np.float64), 1e-7))
    hi = np.percentile(d, hi_pct)
    lo = max(np.percentile(d, lo_pct), hi - range_db)
    if hi - lo < 1:
        lo = hi - 1
    return np.clip((d - lo) / (hi - lo), 0, 1), float(lo), float(hi)


def flux(p):
    """positive log-energy rise over 20 ms of a 5 ms power series, gated by the noise floor.
    Index k = energy known up to (k + 1) * 5 ms."""
    e = np.log(np.convolve(p, np.ones(2) / 2)[:len(p)] + 1e-10)      # 10 ms
    d = np.maximum(e[4:] - e[:-4], 0)                                 # rise over 20 ms
    d = np.concatenate((np.zeros(min(4, len(p))), d))
    if not len(e):
        return d
    floor = np.percentile(e, 20)
    return d * np.clip((e - floor) / 12, 0, 1)                         # a rise out of the noise floor counts for little


# ---------- reference onsets (independent check of the beats; never used to place them)
def _hz_to_mel(f):
    """Slaney mel scale: linear below 1 kHz, logarithmic above"""
    f = np.asarray(f, np.float64)
    sp, lin_mel, step = 200.0 / 3, 1000.0 / (200.0 / 3), np.log(6.4) / 27.0
    return np.where(f >= 1000.0, lin_mel + np.log(np.maximum(f, 1e-12) / 1000.0) / step, f / sp)


def _mel_to_hz(m):
    m = np.asarray(m, np.float64)
    sp, lin_mel, step = 200.0 / 3, 1000.0 / (200.0 / 3), np.log(6.4) / 27.0
    return np.where(m >= lin_mel, 1000.0 * np.exp(step * (m - lin_mel)), sp * m)


def mel_filters(sr, n_fft, n_mels):
    """[n_mels, n_fft/2 + 1] area-normalised triangular mel filters from 0 Hz to Nyquist"""
    fft_f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2.0), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_f[None, :]
    w = np.maximum(0, np.minimum(-ramps[:-2] / fdiff[:-1, None], ramps[2:] / fdiff[1:, None]))
    return (w * (2.0 / (mel_f[2:] - mel_f[:-2]))[:, None]).astype(np.float32)


def peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait):
    """indices n where x[n] is the max of x[n-pre_max : n+post_max], at least mean(x[n-pre_avg : n+post_avg]) + delta,
    and more than `wait` samples after the previous pick (windows truncated at the ends)"""
    n = len(x)
    if n == 0:
        return np.zeros(0, np.int64)
    pad = np.concatenate((np.full(pre_max, -np.inf, x.dtype), x, np.full(post_max, -np.inf, x.dtype)))
    mov_max = np.lib.stride_tricks.sliding_window_view(pad, pre_max + post_max)[:n].max(axis=1)
    cs = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    i = np.arange(n)
    a, b = np.maximum(0, i - pre_avg), np.minimum(n, i + post_avg)
    mov_avg = ((cs[b] - cs[a]) / (b - a)).astype(x.dtype)
    cand = np.flatnonzero((x == mov_max) & (x >= mov_avg + x.dtype.type(delta)))
    out, last = [], -np.inf
    for k in cand.tolist():
        if k > last + wait:
            out.append(k)
            last = k
    return np.asarray(out, np.int64)


class RefFlux:
    """streaming log-mel spectral flux: feed() mono chunks at rate `sr`, finish() -> onset times (s).
    Frame k is centred on sample k * hop (zero-padded by n_fft / 2 at both ends). Mel energies are kept as float32
    (40 x frames, ~115 MB at 60 min); the dB floor (80 dB under the loudest frame) needs the whole file, so the flux
    is formed after the scan."""

    def __init__(self, sr):
        self.sr, self.hop = sr, max(1, int(round(0.005 * sr)))
        self.win = sg.get_window("hann", REF_NFFT).astype(np.float32)
        self.mel = mel_filters(sr, REF_NFFT, REF_MELS).T.copy()
        self.buf = np.zeros(REF_NFFT // 2, np.float32)
        self.n_in, self.out = 0, []

    def _frames(self):
        n = (len(self.buf) - REF_NFFT) // self.hop + 1
        if n <= 0:
            return
        fr = np.lib.stride_tricks.sliding_window_view(self.buf, REF_NFFT)[::self.hop][:n]
        spec = np.abs(np.fft.rfft(fr * self.win, axis=1)) ** 2
        self.out.append((spec.astype(np.float32) @ self.mel).astype(np.float32))
        self.buf = self.buf[n * self.hop:]

    def feed(self, y):
        self.n_in += len(y)
        self.buf = np.concatenate((self.buf, np.asarray(y, np.float32)))
        self._frames()

    def finish(self):
        total = 1 + self.n_in // self.hop
        self.buf = np.concatenate((self.buf, np.zeros(REF_NFFT // 2, np.float32)))
        self._frames()
        if not self.out or self.n_in < 512:
            return np.zeros(0)
        S = np.concatenate(self.out)[:total]
        self.out = []
        top = np.float32(10 * np.log10(max(1e-10, float(S.max())))) - np.float32(80.0)
        env = np.zeros(len(S), np.float32)
        lag_pad = 1 + REF_NFFT // (2 * self.hop)
        prev = None
        for s0 in range(0, len(S), 65536):                    # row blocks keep the dB temporaries small
            blk = np.maximum(np.float32(10.0) * np.log10(np.maximum(np.float32(1e-10), S[s0:s0 + 65536])), top)
            both = blk if prev is None else np.concatenate((prev, blk))
            d = np.maximum(0, both[1:] - both[:-1]).mean(axis=1)
            j0 = s0 + lag_pad if prev is None else s0 - 1 + lag_pad
            m = min(len(d), len(env) - j0)
            if m > 0:
                env[j0:j0 + m] = d[:m]
            prev = blk[-1:]
        del S
        env = env - env.min()
        env /= env.max() + np.finfo(np.float32).tiny
        w = lambda s: int(np.ceil(s * self.sr // self.hop))
        pk = peak_pick(env, w(0.03), 1, w(0.10), w(0.10) + 1, 0.07, w(0.03))
        return pk * self.hop / self.sr


# ---------- pulse detection
def comb(ac, lags):
    """mean autocorrelation at T, 2T, 3T, 4T"""
    tot, cnt = np.zeros(len(lags)), np.zeros(len(lags))
    for k in range(1, 5):
        idx = k * lags
        ok = idx < len(ac)
        tot[ok] += ac[idx[ok]]
        cnt[ok] += 1
    return tot / np.maximum(cnt, 1)


def acf(f):
    f = f - f.mean()
    n = len(f)
    F = np.fft.rfft(f, 2 * n)
    c = np.fft.irfft(np.abs(F) ** 2)[:n]
    return c / (c[0] + 1e-12)


def comb_at(ac, lag):
    """comb() at a fractional lag (linear interpolation of the autocorrelation)"""
    idx = np.arange(1, 5) * float(lag)
    idx = idx[idx < len(ac) - 1]
    return float(np.mean(np.interp(idx, np.arange(len(ac)), ac))) if len(idx) else 0.0


def pulse_windows(f, step, win_s=16, hop_s=4, seed=1, acfs=None):
    """for each window: best period, comb score, z-score against a block-shuffled null (periodicity destroyed, texture
    kept). acfs: optional list that receives each window's autocorrelation (for the tempo candidates)"""
    rng = np.random.default_rng(seed)
    W, Hp = int(round(win_s / step)), int(round(hop_s / step))
    lags = np.arange(int(0.30 / step), int(1.25 / step))
    block = int(0.15 / step)
    out = []
    if len(f) < lags[-1] + 2:
        return out
    for s in range(0, max(1, len(f) - W + 1), Hp):
        w = f[s:s + W]
        a = acf(w)
        if acfs is not None:
            acfs.append(a.astype(np.float32))
        c = comb(a, lags)
        k = int(np.argmax(c))
        nulls = []
        m = len(w) // block
        for _ in range(10):
            p = rng.permutation(m)
            nulls.append(np.max(comb(acf(w[:m * block].reshape(m, block)[p].ravel()), lags)))
        nm, ns = np.mean(nulls), np.std(nulls) + 1e-6
        out.append(dict(t=(s + len(w) / 2) * step, T=float(lags[k] * step), score=float(c[k]), z=float((c[k] - nm) / ns)))
    return out


def fold(T, bpm_min, bpm_max):
    """one metrical level: the comb flips between T, T/2 and 2T on the same music, so fold into [bpm_min, bpm_max)"""
    lo, hi = 60.0 / bpm_max, 60.0 / bpm_min
    while T <= lo:
        T *= 2
    while T > hi:
        T /= 2
    return T


def nearest_gap(t, ons):
    """|t - nearest of the sorted times ons| for each t (O(n log m), no n x m matrix)"""
    j = np.searchsorted(ons, t)
    lo = ons[np.clip(j - 1, 0, len(ons) - 1)]
    hi = ons[np.clip(j, 0, len(ons) - 1)]
    return np.minimum(np.abs(t - lo), np.abs(hi - t))


def agreement(times, ons, T, win=0.05):
    """fraction of beats within +-win of a reference onset, and the same at other phases of the period (chance at this density)"""
    if not len(ons):
        return 0.0, 0.0
    ons = np.sort(np.asarray(ons, np.float64))
    near = lambda t: float(np.mean(nearest_gap(t, ons) <= win))
    return near(times), float(np.mean([near(times + T * p) for p in np.arange(0.2, 0.9, 0.1)]))


def passages(wins, z_min, min_windows=3, tol=0.02):
    """runs of windows with a significant, stable period (octave changes not bridged)"""
    res, run = [], []
    def close():
        if len(run) >= min_windows:
            res.append(list(run))
        run.clear()
    for w in wins:
        if w["z"] >= z_min and (not run or abs(w["T"] / np.median([r["T"] for r in run]) - 1) <= tol):
            run.append(w)
        else:
            close()
            if w["z"] >= z_min:
                run.append(w)
    close()
    return res


def track_beats(f, t0, t1, T, thr, step, tol=0.14):
    """walk the period from the strongest onset in both directions; only real onset peaks are kept"""
    i0, i1 = int(t0 / step), min(int(t1 / step), len(f) - 1)
    peaks, _ = sg.find_peaks(f[i0:i1], distance=max(1, int(0.12 / step)))
    peaks = peaks + i0
    if not len(peaks):
        return []
    a = int(peaks[np.argmax(f[peaks])])
    found = {a: f[a]}
    for sign in (1, -1):
        cur, per, miss = a * step, T, 0
        while True:
            pred = cur + sign * per
            if pred < t0 or pred > t1 or miss > 6:
                break
            near = peaks[np.abs(peaks * step - pred) <= tol * per]
            near = near[f[near] >= thr]
            if len(near):
                k = int(near[np.argmax(f[near] * (1 - 0.5 * np.abs(near * step - pred) / (tol * per)))])
                t = k * step
                per += 0.1 * ((t - cur) * sign - per)        # follows drift slowly
                per = float(np.clip(per, T * 0.9, T * 1.1))
                cur = t
                found[k] = f[k]
                miss = 0
            else:
                cur, miss = pred, miss + 1
    return sorted(found.items())


def tempo_candidates(acfs, step, bpm):
    """the main tempo and its usual confusions (x2, x1/2, x3/2, x2/3, x4/3, x3/4), each with its onset-autocorrelation
    support (median over the analysis windows of the comb score at that period) and a confidence relative to the best
    candidate (1.0 = best supported). A human or an agent picks one; rig.tempo.bpm pins it."""
    if not acfs or not bpm:
        return []
    rows = []
    for label, r in CANDIDATE_RATIOS:
        b = bpm * r
        lag = 60.0 / b / step
        sup = float(np.median([comb_at(a, lag) for a in acfs]))
        rows.append({"ratio": label, "bpm": round(b, 2), "support": round(sup, 4)})
    top = max(max(r["support"] for r in rows), 1e-9)
    for r in rows:
        r["confidence"] = round(max(0.0, r["support"]) / top, 3)
    return sorted(rows, key=lambda r: (-r["confidence"], r["ratio"] != "x1"))


def beat_grid(f, ref_ons, duration, s, acfs_out=None):
    """verified beats: [{time, strength, confidence}], accepted/rejected passages, bpm map, grid offset, pulse summary.
    time = (k + 1) * 5 ms, when the onset is known. With no proven pulse, the strongest window's folded period is used
    as a low-confidence tempo (fallback=True) and no beats are written. acfs_out: a list that receives each window's
    onset autocorrelation (tempo candidates around another tempo)."""
    step = HOP_MS / 1000
    # short clips: shrink the window so at least ~5 windows (3 are needed for a passage) fit
    win = min(s["win_s"], max(4.0, duration / 2))
    hop = min(s["hop_s"], win / 4)
    acfs = [] if acfs_out is None else acfs_out
    wins = pulse_windows(f, step, win, hop, acfs=acfs)
    for w in wins:
        w["T"] = fold(w["T"], s["bpm_min"], s["bpm_max"])
    fp = f[f > 0]
    thr = float(np.percentile(fp, 70)) if len(fp) else 0.0
    runs = passages(wins, s["pulse_z"])
    half = win / 2
    spans = [[0.0 if r[0] is wins[0] else max(0.0, r[0]["t"] - half),
              duration if r[-1] is wins[-1] else min(duration, r[-1]["t"] + half)] for r in runs]
    for s0, s1 in zip(spans[:-1], spans[1:]):      # neighbouring windows overlap: split at the midpoint
        if s0[1] > s1[0]:
            s0[1] = s1[0] = (s0[1] + s1[0]) / 2
    pas, rejected, beats = [], [], []
    for run, (t0, t1) in zip(runs, spans):
        T = float(np.median([r["T"] for r in run]))
        bb = track_beats(f, t0, t1, T, thr, step)
        times = np.array([k * step for k, _ in bb])
        if len(times) < 8:
            rejected.append(dict(start=round(t0, 3), end=round(t1, 3), bpm=round(60 / T, 2), reason="fewer than 8 beats"))
            continue
        ioi = np.diff(times)
        good = ioi[np.abs(ioi - T) < 0.25 * T]
        hit_rate = len(times) / max(1, (times[-1] - times[0]) / T + 1)
        jitter = float(np.std(good) / T) if len(good) > 2 else 1.0
        agree, chance = agreement(times, ref_ons, T)
        zm = float(np.median([r["z"] for r in run]))
        # tempo from a line fit through the beat times: the comb's period is quantised to the 5 ms lag grid. Each beat's
        # index counts the periods between neighbours (round(interval / T), so a missed beat counts 2): rounding
        # (t - t0) / T instead lets the quantised period's error pile up until a far beat gets the wrong index and
        # bends the slope (a 128 BPM track with a 0.470 s comb period read as 127.68 and drifted 0.7 s by 4 minutes)
        n_idx = np.concatenate(([0.0], np.cumsum(np.maximum(1.0, np.round(np.diff(times) / T)))))
        T_fit = float(np.polyfit(n_idx, times, 1)[0]) if n_idx[-1] > n_idx[0] else T
        conf = float(np.clip((zm - s["pulse_z"]) / 8 + 0.3, 0, 1) * np.clip(hit_rate, 0, 1) * np.clip(1.2 - jitter / 0.12, 0, 1)
                     * np.clip((agree - chance) / 0.4, 0, 1))
        row = dict(start=round(t0, 3), end=round(t1, 3), bpm=round(60 / T_fit, 2), hit_rate=round(float(hit_rate), 2), jitter=round(jitter, 3),
                   z=round(zm, 1), agree=round(agree, 2), chance=round(chance, 2), confidence=round(conf, 2), beats=len(times))
        if conf < s["min_conf"] or agree - chance < s["min_agree"]:
            rejected.append(row)
            continue
        pas.append(row)
        fs = np.array([v for _, v in bb])
        ref = np.percentile(fs, 90) + 1e-9
        for k, v in bb:
            beats.append(dict(time=round((k + 1) * step, 6), strength=round(float(min(1.0, v / ref)), 3),
                              confidence=round(float(conf * min(1.0, 0.4 + 0.6 * v / ref)), 3)))
    beats.sort(key=lambda b: b["time"])
    fallback = False
    if pas:
        bpm_map = []
        for p in pas:
            t = 0.0 if not bpm_map else p["start"]
            if not bpm_map or abs(bpm_map[-1]["bpm"] - p["bpm"]) > 0.05:
                bpm_map.append(dict(time=t, bpm=p["bpm"]))
        T0 = 60 / pas[0]["bpm"]
        b0 = next(b["time"] for b in beats if b["time"] >= pas[0]["start"])
        grid_offset = b0 - np.floor(b0 / T0) * T0
        cov = sum(p["end"] - p["start"] for p in pas) / max(duration, 1e-9)
        c = float(np.average([p["confidence"] for p in pas], weights=[max(1e-9, p["end"] - p["start"]) for p in pas]))
        conf = c * min(1.0, cov / 0.5)
        pulse = dict(confidence=round(conf, 2), covered=round(cov, 2), label="pulse" if conf >= 0.5 else "partial pulse")
    else:
        fallback = True
        if wins:
            T0 = max(wins, key=lambda w: w["z"])["T"]
        else:
            T0 = 0.5
        bpm_map = [dict(time=0.0, bpm=round(60 / T0, 2))]
        pk = int(np.argmax(f)) if len(f) else 0
        tpk = (pk + 1) * step
        grid_offset = tpk - np.floor(tpk / T0) * T0
        pulse = dict(confidence=0.0, covered=0.0, label="no proven pulse: tempo is a low-confidence estimate, no beats")
    pulse["windows_z_median"] = round(float(np.median([w["z"] for w in wins])), 1) if wins else None
    main = max(pas, key=lambda p: p["end"] - p["start"])["bpm"] if pas else bpm_map[0]["bpm"]
    return dict(beats=beats, passages=pas, rejected_passages=rejected, bpm_map=bpm_map, grid_offset=round(float(grid_offset), 6),
                fallback=fallback, pulse=pulse, tempo_candidates=tempo_candidates(acfs, step, main))


# ---------- neural beats and bars (activations from beatrig.neural; everything here is plain numpy)
def nn_peaks(logits):
    """(frame positions, peak logits) where a logit is > 0 and the largest within +-3 frames (+-60 ms at 50 fps); a
    run of adjacent equal maxima counts once, at its middle"""
    x = np.asarray(logits, np.float64)
    if not len(x):
        return np.zeros(0), np.zeros(0)
    pad = np.concatenate((np.full(3, -np.inf), x, np.full(3, -np.inf)))
    mx = np.lib.stride_tricks.sliding_window_view(pad, 7).max(axis=1)
    cand = np.flatnonzero((x > 0) & (x >= mx))
    if not len(cand):
        return np.zeros(0), np.zeros(0)
    runs = np.split(cand, np.flatnonzero(np.diff(cand) > 1) + 1)
    pos, val = [], []
    for r in runs:
        k = int(r[0]) if len(r) == 1 else None
        if k is not None and 0 < k < len(x) - 1:              # sub-frame position: the parabola through the peak
            den = x[k - 1] - 2 * x[k] + x[k + 1]
            pos.append(k + (0.5 * (x[k - 1] - x[k + 1]) / den if den < 0 else 0.0))
        else:
            pos.append(float(r.mean()))
        val.append(float(x[r].max()))
    return np.array(pos), np.array(val)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float64), -60, 60)))


def _local_period(times, half=8):
    """inter-beat interval around each interval: the mean of the neighbouring intervals within 25 % of their median
    (a missing or extra beat is left out; the mean undoes the 20 ms frame grid that a median would keep)"""
    ioi = np.diff(times)
    out = np.empty(len(ioi))
    for i in range(len(ioi)):
        w = ioi[max(0, i - half):i + half + 1]
        med = np.median(w)
        sel = w[np.abs(w - med) <= 0.25 * med]
        out[i] = sel.mean() if len(sel) else med
    return out


def beat_skeleton(times, probs):
    """the model's beats as a continuous count: spurious near-duplicates (closer than 0.6 of the local period) drop the
    weaker one, and every gap of k periods gets k - 1 virtual beats, so counting does not slip where the model is
    silent (a drum-less breakdown). Returns (times, real mask, probs)."""
    t, p = np.asarray(times, np.float64), np.asarray(probs, np.float64)
    while len(t) > 2:
        T = _local_period(t)
        bad = np.flatnonzero(np.diff(t) < 0.6 * T)
        if not len(bad):
            break
        i = int(bad[0])
        j = i if p[i] < p[i + 1] else i + 1
        t, p = np.delete(t, j), np.delete(p, j)
    if len(t) < 3:
        return t, np.ones(len(t), bool), p
    T = _local_period(t)
    k = np.maximum(1, np.round(np.diff(t) / T)).astype(int)
    st, real, sp = [t[0]], [True], [p[0]]
    for i in range(len(k)):
        for j in range(1, k[i]):
            st.append(t[i] + (t[i + 1] - t[i]) * j / k[i])
            real.append(False)
            sp.append(0.0)
        st.append(t[i + 1])
        real.append(True)
        sp.append(p[i + 1])
    return np.array(st), np.array(real), np.array(sp)


def bar_positions(skel_t, down_prob, fps, meter=None, switch=1e-12):
    """(meter, position in bar per skeleton beat; 0 = downbeat). The meter is the most common spacing of the model's
    downbeat peaks (3 or 4 and so on; 4 unless one spacing is clearly the most common). The phase is a Viterbi path:
    in phase p, beat i is a downbeat when (i - p) % meter == 0; each beat scores its downbeat activation (the largest
    within +-2 frames, held to 0.1..0.9 so no single beat is decisive) and a phase change costs log(switch). A real
    shift (a bar of 2 between sections) is taken where it happens once about 7 bars after it agree; a stretch where
    the model hears the bar line elsewhere and then comes back (syncopated chords in a drum-less breakdown) needs
    about 13 contrary bars to break the count."""
    n = len(skel_t)
    fr = np.clip(np.round(skel_t * fps).astype(int), 0, len(down_prob) - 1)
    pad = np.concatenate((np.zeros(2), down_prob, np.zeros(2)))
    a = np.lib.stride_tricks.sliding_window_view(pad, 5).max(axis=1)[fr]
    if meter is None:                                   # 3 when the downbeat spacings are clearly multiples of 3, else 4
        dpk, _ = nn_peaks(np.log(np.maximum(down_prob, 1e-9) / np.maximum(1 - down_prob, 1e-9)))
        idx = np.unique(np.searchsorted(skel_t, dpk / fps - 0.5 * np.median(np.diff(skel_t)) if n > 1 else dpk / fps))
        gaps = np.diff(idx)
        gaps = gaps[(gaps >= 2) & (gaps <= 16)]
        meter = 4
        if len(gaps) >= 3 and np.mean(gaps % 3 == 0) >= 0.5 and np.mean(gaps % 3 == 0) > np.mean(gaps % 4 == 0) + 0.2:
            meter = 3
    a = np.clip(a, 0.1, 0.9)
    ld, ln = np.log(a), np.log(1 - a)
    ph = np.arange(meter)
    score = np.where((0 - ph) % meter == 0, ld[0], ln[0]) if n else np.zeros(meter)
    back = np.zeros((n, meter), int)
    pen = np.log(switch)
    for i in range(1, n):
        best = int(np.argmax(score))
        stay = score
        move = score[best] + pen
        back[i] = np.where(stay >= move, ph, best)
        score = np.maximum(stay, move) + np.where((i - ph) % meter == 0, ld[i], ln[i])
    path = np.zeros(n, int)
    if n:
        path[-1] = int(np.argmax(score))
        for i in range(n - 1, 0, -1):
            path[i - 1] = back[i, path[i]]
    return meter, (np.arange(n) - path) % meter


def neural_grid(f, ref_ons, beat_logits, down_logits, duration, s, fps=50.0):
    """verified beats and bars from the model's activations, in beat_grid()'s format plus "bars", or None when the
    model found fewer than 8 beats (the classic grid then stands).
    - Every model beat needs a real onset: the strongest octave-flux peak (above the 70th percentile, as for the
      classic grid, and 5 % of a strong hit) within 0.14 periods of it, weighted toward the nearest. The beat is
      written at that onset's time ((k + 1) * 5 ms, when the rise is known), never at the model's frame, so it is
      never ahead of the sound. A beat with no onset is not written.
    - Beats the model did not give (gaps) and beats without an onset stay in the skeleton that counts bars, so a
      breakdown without drums does not shift the bar count. Each written beat carries "bar" (0 = the first bar that
      starts with a downbeat; a pickup is bar -1) and "pos" (0 = downbeat).
    - Passages split where the model is silent for 8 beats or more, or the tempo moves by more than 3 %. Each is
      checked like a classic passage: tempo from a line fit, agreement with the independent reference onsets above
      chance, and confidence (mean model probability x hit rate x jitter x agreement); a passage under min_conf or
      min_agree writes no beats.
    - grid_offset is the first downbeat's phase within a bar, so beats 0, meter, 2 meter ... of the tempo map are
      downbeats (patterns and bar windows lock to real bars)."""
    step = HOP_MS / 1000
    bt, bl = nn_peaks(beat_logits)
    if len(bt) < 8:
        return None
    skel_t, real, sp = beat_skeleton(bt / fps, _sigmoid(bl))
    meter, pos = bar_positions(skel_t, _sigmoid(down_logits), fps)
    downs = np.flatnonzero(pos == 0)
    first = int(downs[0]) if len(downs) else 0
    bar = np.cumsum(pos == 0) - 1                       # beats before the first downbeat: bar -1
    # onset support: a peak above the 70th percentile of the onset function (as for the classic grid) and at least
    # 5 % of a strong hit, so a flicker in the noise floor never passes for a beat
    pk, _ = sg.find_peaks(f)
    fp = f[f > 0]
    thr = float(np.percentile(fp, 70)) if len(fp) else 0.0
    if len(pk):
        thr = max(thr, 0.05 * float(np.percentile(f[pk], 99)))
    pk = pk[f[pk] >= thr]
    pt = (pk + 1) * step
    Tl = np.concatenate((_local_period(skel_t), [0.0]))
    Tl[-1] = Tl[-2] if len(Tl) > 1 else 0.5
    snap = np.full(len(skel_t), -1)
    for i in np.flatnonzero(real):
        tol = 0.14 * Tl[i]
        lo, hi = np.searchsorted(pt, skel_t[i] - tol), np.searchsorted(pt, skel_t[i] + tol, side="right")
        if hi > lo:
            c = pk[lo:hi]
            w = f[c] * (1 - 0.5 * np.abs(pt[lo:hi] - skel_t[i]) / tol)
            snap[i] = int(c[np.argmax(w)])
    # passages: break where the model is silent for 8 beats or more (those virtual beats belong to no passage) and
    # where the local period moves more than 3 % from the passage's recent beats
    segs, cur, gap = [], [], []
    for i in range(len(skel_t)):
        if not real[i]:
            gap.append(i)
            continue
        if len(gap) >= 8 or (cur and abs(Tl[i] / float(np.median(Tl[cur[-32:]])) - 1) > 0.03):
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.extend(gap)
        gap = []
        cur.append(i)
    if cur:
        segs.append(cur)
    pas, rejected, beats = [], [], []
    for seg in segs:
        idx = np.asarray(seg)
        ok = idx[snap[idx] >= 0]
        r = idx[real[idx]]
        t0 = max(0.0, float(skel_t[r[0]]) - 0.5 * float(Tl[r[0]]))
        t1 = min(duration, float(skel_t[r[-1]]) + 0.5 * float(Tl[r[-1]]))
        if len(ok) < 8:
            rejected.append(dict(start=round(t0, 3), end=round(t1, 3), bpm=round(60 / float(np.median(Tl[r])), 2),
                                 reason="fewer than 8 beats with an onset"))
            continue
        times = (snap[ok] + 1) * step
        T_fit, b0 = np.polyfit(ok.astype(np.float64), times, 1)
        T_fit = float(T_fit)
        jitter = float(np.std(times - (b0 + T_fit * ok)) / T_fit)
        hit_rate = len(ok) / len(idx)
        agree, chance = agreement(times, ref_ons, T_fit)
        mp = float(np.mean(sp[r]))
        conf = float(mp * np.clip(hit_rate, 0, 1) * np.clip(1.2 - jitter / 0.12, 0, 1) * np.clip((agree - chance) / 0.4, 0, 1))
        row = dict(start=round(t0, 3), end=round(t1, 3), bpm=round(60 / T_fit, 2), hit_rate=round(float(hit_rate), 2),
                   jitter=round(jitter, 3), agree=round(agree, 2), chance=round(chance, 2), confidence=round(conf, 2),
                   beats=int(len(ok)), model_prob=round(mp, 2))
        if conf < s["min_conf"] or agree - chance < s["min_agree"]:
            rejected.append(row)
            continue
        pas.append(row)
        fs = f[snap[ok]]
        ref = np.percentile(fs, 90) + 1e-9
        for i in ok:
            v = f[snap[i]]
            beats.append(dict(time=round(float((snap[i] + 1) * step), 6), strength=round(float(min(1.0, v / ref)), 3),
                              confidence=round(float(sp[i] * min(1.0, 0.4 + 0.6 * v / ref)), 3),
                              bar=int(bar[i]), pos=int(pos[i])))
    if not pas:
        return dict(beats=[], passages=[], rejected_passages=rejected, bars=None)
    beats.sort(key=lambda b: b["time"])
    bpm_map = []
    for p in pas:
        t = 0.0 if not bpm_map else p["start"]
        if not bpm_map or abs(bpm_map[-1]["bpm"] - p["bpm"]) > 0.05:
            bpm_map.append(dict(time=t, bpm=p["bpm"]))
    # downbeat times for bar lines: the onset when the downbeat has one, else the skeleton's (model or virtual) time
    dtimes = [float((snap[i] + 1) * step) if snap[i] >= 0 else float(skel_t[i]) for i in downs]
    bar_len = meter * 60.0 / pas[0]["bpm"]
    d0 = next((d for d in dtimes if d >= pas[0]["start"] - 1e-9), dtimes[0] if dtimes else beats[0]["time"])
    grid_offset = d0 - np.floor(d0 / bar_len) * bar_len
    cov = sum(p["end"] - p["start"] for p in pas) / max(duration, 1e-9)
    c = float(np.average([p["confidence"] for p in pas], weights=[max(1e-9, p["end"] - p["start"]) for p in pas]))
    conf = c * min(1.0, cov / 0.5)
    pulse = dict(confidence=round(conf, 2), covered=round(cov, 2), label="pulse" if conf >= 0.5 else "partial pulse")
    dconf = float(np.mean(_sigmoid(down_logits)[np.clip(np.round(skel_t[downs] * fps).astype(int), 0, len(down_logits) - 1)])) if len(downs) else 0.0
    bars = dict(meter=int(meter), downbeats=[round(d, 6) for d in dtimes], first_downbeat=round(float(skel_t[first]), 6),
                confidence=round(dconf, 2), skeleton_beats=int(len(skel_t)), model_beats=int(real.sum()))
    return dict(beats=beats, passages=pas, rejected_passages=rejected, bpm_map=bpm_map, grid_offset=round(float(grid_offset), 6),
                fallback=False, pulse=pulse, bars=bars)


# ---------- specs
def weights_for(weights, channels):
    if weights is None:
        return np.full(channels, 1.0 / channels)
    w = np.asarray(weights, np.float64)
    if w.shape != (channels,):
        raise ValueError(f"weights {list(w)} do not match the audio's {channels} channels")
    return w


def spec_key(kind, lo, hi, w):
    wk = ",".join(f"{x:g}" for x in w)
    return f"loud:{wk}" if kind == "loud" else f"band:{float(lo):g}:{float(hi):g}:{wk}"


def _f32(a):
    return np.asarray(a, np.float32)


class Analysis:
    """Cached analysis of one audio file. Built by open_analysis(); arrays are always float32 in memory and on disk, so a
    fresh analysis and a cached one give identical bakes."""

    def __init__(self, audio, cache_dir, settings, default_weights=None):
        self.audio = os.path.abspath(audio)
        self.cache_dir = cache_dir
        self.settings = dict(DEFAULT_SETTINGS, **(settings or {}))
        info = sf.info(self.audio)
        self.sr, self.channels, self.n_samples = int(info.samplerate), int(info.channels), int(info.frames)
        self.duration = self.n_samples / self.sr
        self.default_weights = default_weights
        st = os.stat(self.audio)
        self.ident = dict(audio=self.audio.replace("\\", "/"), size=st.st_size, mtime_ns=st.st_mtime_ns, version=ANALYSIS_VERSION,
                          settings={k: v for k, v in self.settings.items() if k not in NOT_IN_IDENT},
                          weights=None if default_weights is None else [float(x) for x in default_weights])
        self.ident_hash = hashlib.sha256(json.dumps(self.ident, sort_keys=True).encode()).hexdigest()
        self.arrays = {}
        self.meta = {}
        self.dirty = False
        self.loaded_from_cache = False
        self.stems, self.stems_info = {}, {"used": False, "why": "not requested"}   # set by bake.analyze()

    def content_hash(self):
        """sha256 of the audio file (it names the stem folder), kept in the cache so a cached analyze never re-reads a
        long show to find its stems"""
        if not self.meta.get("content_sha256"):
            from .stems import content_hash
            self.meta["content_sha256"] = content_hash(self.audio)
            self.dirty = True
        return self.meta["content_sha256"]

    def tracker(self):
        """(name, why) the beat grid comes from with these settings on this install: "neural" or "classic". "auto" is
        neural when the [ml] extra is installed (and BEATRIG_EXTRAS is not "off")."""
        want = self.settings.get("beats", "auto")
        if want == "classic":
            return "classic", "chosen"
        from . import neural
        ok, why = neural.available()
        if want == "auto" and neural.disabled():
            return "classic", "disabled"
        if ok:
            return "neural", ""
        if want == "neural":
            raise ValueError(f"analysis.beats is \"neural\" but {why}")
        return "classic", "unavailable"

    # -- paths
    @property
    def json_path(self):
        return os.path.join(self.cache_dir, "analysis.json")

    @property
    def npz_path(self):
        return os.path.join(self.cache_dir, "analysis.npz")

    def _load(self):
        try:
            with open(self.json_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("ident_hash") != self.ident_hash:
                return False
            with np.load(self.npz_path) as z:
                arrays = {k: _f32(z[k]) for k in z.files}
            if "onset" not in arrays or "peaks" not in arrays:
                return False
        except (OSError, ValueError, KeyError):
            return False
        bt = meta.get("beat_tracker") or {}
        if (self.settings.get("beats", "auto") == "auto" and bt.get("name") == "classic"
                and bt.get("why") in ("unavailable", "disabled") and self.tracker()[0] == "neural"):
            return False                                      # the [ml] extra arrived since: upgrade to neural beats
        self.meta, self.arrays, self.loaded_from_cache = meta, arrays, True
        return True

    def save(self):
        """write the cache, only when an audio pass produced something new. Envelopes are cheap derivations of the
        cached band power and stay in memory only, so editing attack/release never rewrites (or grows) the cache:
        a cached analyze is a pure read."""
        if not self.dirty:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        stored = {k: a for k, a in self.arrays.items() if not k.startswith("env:")}
        meta = dict(self.meta, ident=self.ident, ident_hash=self.ident_hash, arrays=sorted(stored))
        tmp = self.npz_path + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, **stored)
        os.replace(tmp, self.npz_path)
        with open(self.json_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(meta, fh, indent=1, sort_keys=True)
            fh.write("\n")
        self.dirty = False

    # -- the causal pass
    def _scan(self, specs, base, ref=True):
        """one pass over the file. specs: {key: (sos | None, weights)} -> 1 ms weighted power (sum over channels of
        w_c * power of the filtered channel, so phase differences between channels cannot cancel). base: onset octave powers, 5 ms peaks, reference mono
        (ref False: no reference onsets, for a stem whose beats are never read)."""
        sr, C = self.sr, self.channels
        st = {k: dict(sos=sos, w=w, zi=None if sos is None else np.zeros((sos.shape[0], 2, C)), out=[]) for k, (sos, w) in specs.items()}
        if base:
            octs = [dict(sos=band_sos(lo, hi, sr, 2), out=[]) for lo, hi in zip(ONSET_EDGES[:-1], ONSET_EDGES[1:]) if lo < sr / 2]
            for o in octs:
                o["zi"] = np.zeros((o["sos"].shape[0], 2))
            peaks = []
            q = max(1, int(round(sr / 12000)))
            lp = sg.butter(6, 0.8 * sr / q / 2, fs=sr, output="sos")
            lpzi, pos = np.zeros((lp.shape[0], 2)), 0
            self._ref = RefFlux(sr / q) if ref else None
            mw = None                                         # onset mono: equal downmix unless channel_weights says otherwise
            if self.default_weights is not None:
                mw = weights_for(self.default_weights, C)
                mw = mw / (mw.sum() or 1.0)
        for blk in sf.blocks(self.audio, blocksize=sr * CHUNK_S, dtype="float32", always_2d=True):
            x = blk.astype(np.float64)
            for s in st.values():
                y = x
                if s["sos"] is not None:
                    y, s["zi"] = sg.sosfilt(s["sos"], x, axis=0, zi=s["zi"])
                s["out"].append(grid_power(y, sr, STEP_MS) @ s["w"])
            if base:
                mono = x.mean(axis=1) if mw is None else x @ mw
                for o in octs:
                    yb, o["zi"] = sg.sosfilt(o["sos"], mono, zi=o["zi"])
                    o["out"].append(grid_power(yb, sr, HOP_MS))
                nb = (len(mono) * 1000) // (sr * HOP_MS)
                if nb:
                    e = (np.arange(nb + 1) * sr * HOP_MS) // 1000
                    peaks.append(np.maximum.reduceat(np.abs(mono[:e[-1]]), e[:-1]))
                if ref:
                    yl, lpzi = sg.sosfilt(lp, mono, zi=lpzi)
                    self._ref.feed(yl[(-pos) % q::q].astype(np.float32))
                pos += len(mono)
        for k, s in st.items():
            self.arrays["pow:" + k] = _f32(np.concatenate(s["out"]) if s["out"] else np.zeros(0))
        if base:
            f = np.zeros(0)
            for o in octs:
                p = np.concatenate(o["out"]) if o["out"] else np.zeros(0)
                f = flux(p) if not len(f) else f + flux(p)
            self.arrays["onset"] = _f32(f)
            self.arrays["peaks"] = _f32(np.concatenate(peaks) if peaks else np.zeros(0))
        self.dirty = True

    def build_base(self):
        if self.settings.get("beats") == "none":             # a stem: its bands and onsets are read, never its beats
            self._scan({}, base=True, ref=False)
            self.meta = dict(sr=self.sr, channels=self.channels, n_samples=self.n_samples, duration=round(self.duration, 6),
                             beats=[], passages=[], rejected_passages=[], bpm_map=[], grid_offset=0.0, fallback=True,
                             pulse={"confidence": 0.0, "covered": 0.0, "label": "not analysed (a stem)"}, tempo_candidates=[],
                             beat_tracker={"name": "none"})
            return
        self._scan({}, base=True)
        f = self.arrays["onset"].astype(np.float64)
        ref = self._ref.finish()
        del self._ref
        acfs = []
        g = beat_grid(f, ref, self.duration, self.settings, acfs_out=acfs)
        name, why = self.tracker()
        info = {"name": "classic", "why": why}
        if name == "neural":
            info = self._neural(f, ref, g, acfs)
        g["beat_tracker"] = info
        self.meta = dict(sr=self.sr, channels=self.channels, n_samples=self.n_samples, duration=round(self.duration, 6), **g)

    def _neural(self, f, ref, g, acfs):
        """replace the classic grid in g with the neural one (beats verified against the onsets, bars, tempo map);
        keep the classic grid, with the reason, when the model fails or finds no verified passage"""
        from . import neural
        try:
            act = neural.activations(self.audio, self.default_weights)
        except Exception as e:                                # a failed download, out of memory: the classic grid stands
            return {"name": "classic", "why": f"failed: {type(e).__name__}: {e}"}
        self.arrays["nn:beat"], self.arrays["nn:down"] = _f32(act["beat"]), _f32(act["downbeat"])
        info = {"name": "neural", "model": act["model"], "device": act["device"], "seconds": act["seconds"]}
        ng = neural_grid(f, ref, act["beat"], act["downbeat"], self.duration, self.settings, act["fps"])
        if ng is None or not ng["passages"]:
            return dict(info, name="classic", why="the model found no verified passage")
        info["classic"] = {"bpm_map": g["bpm_map"], "beats": len(g["beats"]), "pulse": g["pulse"]["label"]}
        main = max(ng["passages"], key=lambda p: p["end"] - p["start"])["bpm"]
        g.update(ng)
        g["tempo_candidates"] = tempo_candidates(acfs, HOP_MS / 1000, main)
        return info

    # -- accessors
    def _spec(self, kind, lo, hi, weights):
        w = weights_for(weights if weights is not None else self.default_weights, self.channels)
        key = spec_key(kind, lo, hi, w)
        sos = k_weight(self.sr) if kind == "loud" else band_sos(lo, hi, self.sr)
        return key, sos, w

    def ensure(self, reqs):
        """reqs: [(kind 'band'|'loud', lo, hi, weights)]; one pass computes every missing power series"""
        todo = {}
        for kind, lo, hi, weights in reqs:
            key, sos, w = self._spec(kind, lo, hi, weights)
            if "pow:" + key not in self.arrays:
                todo[key] = (sos, w)
        if todo:
            self._scan(todo, base=False)

    def envelope(self, kind, lo, hi, attack_ms, release_ms, weights=None):
        """1 ms amplitude envelope (raw linear), cached per (band, weights, attack, release)"""
        key, sos, w = self._spec(kind, lo, hi, weights)
        ek = f"env:{key}:{float(attack_ms):g}:{float(release_ms):g}"
        if ek not in self.arrays:
            self.ensure([(kind, lo, hi, weights)])
            env_keys = [k for k in self.arrays if k.startswith("env:")]
            for k in env_keys[:max(0, len(env_keys) - ENV_KEEP + 1)]:      # oldest first: bounded memory in a server
                del self.arrays[k]
            self.arrays[ek] = _f32(follow(np.sqrt(self.arrays["pow:" + key].astype(np.float64)), attack_ms, release_ms, STEP_MS))
        return self.arrays[ek]

    def onset_fn(self, lo=None, hi=None, weights=None):
        """5 ms onset function: the octave-band flux (full range) or, for a band, that band's flux"""
        if lo is None and hi is None:
            return self.arrays["onset"]
        key, sos, w = self._spec("band", lo, hi, weights)
        self.ensure([("band", lo, hi, weights)])
        p1 = self.arrays["pow:" + key].astype(np.float64)
        n5 = len(p1) // 5
        return _f32(flux(p1[:n5 * 5].reshape(n5, 5).mean(axis=1)))

    @property
    def beats(self):
        return self.meta["beats"]

    @property
    def bpm_map(self):
        return self.meta["bpm_map"]

    @property
    def grid_offset(self):
        return self.meta["grid_offset"]

    def summary(self, px_per_s: float = 200) -> dict:
        """for the preview page: waveform peaks (max |mono| per 1/px_per_s s; 200 = the native 5 ms grid), beats, bpm map"""
        pk = self.arrays["peaks"].astype(np.float64)
        n = max(1, int(round(200 / px_per_s)))
        if n > 1:
            m = len(pk) // n
            pk = pk[:m * n].reshape(m, n).max(axis=1)
        m = self.meta
        return dict(audio=self.ident["audio"], sr=self.sr, channels=self.channels, duration=m["duration"],
                    bpm=m["bpm_map"][0]["bpm"] if m["bpm_map"] else None, bpm_map=m["bpm_map"], grid_offset=m["grid_offset"],
                    fallback=m["fallback"], pulse=m["pulse"], passages=m["passages"], beats=m["beats"],
                    tempo_candidates=m.get("tempo_candidates", []), beat_tracker=self.tracker_label(),
                    bars=self.bars_summary(), stems=self.stems_label(),
                    peaks_per_s=200 / n, peaks=np.round(pk, 4).tolist())

    def tracker_label(self):
        """one line: where the beat grid came from"""
        bt = self.meta.get("beat_tracker") or {"name": "classic", "why": "cache from an older version"}
        if bt.get("name") == "neural":
            return f"neural ({bt.get('model')}, {bt.get('device')}, {bt.get('seconds')} s), every beat on a verified onset"
        why = {"unavailable": "the [ml] extra is not installed", "disabled": "BEATRIG_EXTRAS=off", "chosen": "analysis.beats = classic"}
        return f"classic onset-autocorrelation grid ({why.get(bt.get('why'), bt.get('why'))})"

    def bars_summary(self):
        """{meter, bars, confidence, first_downbeat} when the analysis knows the downbeats, else None"""
        b = self.meta.get("bars")
        if not b:
            return None
        return {"meter": b["meter"], "bars": len(b["downbeats"]), "confidence": b["confidence"], "first_downbeat": b["first_downbeat"]}

    def stems_label(self):
        """one line: whether stem features (drum parts, structure) read separated stems or the mix, and why"""
        s = self.stems_info or {}
        if s.get("used"):
            how = ", ".join(str(x) for x in (s.get("model"), s.get("device"), f"{s['seconds']} s" if "seconds" in s else None) if x)
            return f"separated stems ({', '.join(sorted(self.stems))}) from {s.get('dir_name', 'the stem cache')}" + (f" ({how})" if how else "")
        return f"the mix ({s.get('why', 'stems not requested')})"


# ---------- song sections (energy changes between bars)
SECTION_BANDS = [("sub", 30, 60), ("kick", 40, 120), ("bass", 80, 250), ("lowmid", 250, 800), ("mid", 800, 2500),
                 ("presence", 2500, 6000), ("air", 6000, 14000)]


def band_levels(a, edges, t1, bands=SECTION_BANDS):
    """[bands, windows] RMS level of each band per window [edges[i], edges[i+1] or t1), 0..1 on a dB scale over the
    whole song (10th..99.5th percentile)"""
    a.ensure([("band", lo, hi, None) for _, lo, hi in bands])
    L = []
    for _, lo, hi in bands:
        env = a.envelope("band", lo, hi, 10, 150).astype(np.float64)
        db = 20 * np.log10(env + 1e-9)
        lo_db, hi_db = np.percentile(db, 10), np.percentile(db, 99.5)
        ends = list(edges[1:]) + [t1]
        row = []
        for e0, e1 in zip(edges, ends):
            seg = env[int(e0 * 1000):int(e1 * 1000)]
            v = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) if len(seg) else lo_db
            row.append(float(np.clip((v - lo_db) / max(hi_db - lo_db, 1e-6), 0, 1)))
        L.append(row)
    return np.array(L)


def section_boundaries(a, window_s=None, min_windows=2):
    """Candidate section boundaries: bar lines (windows of one bar from the grid offset, or window_s) where the band
    levels change sharply (summed change >= 0.9 or one band >= 0.35, the rule describe_audio uses). Boundaries closer
    than min_windows keep the stronger change. Returns (sections, levels): sections = [{"time", "level", "rose",
    "fell"}] starting with time 0; level = the mean band level over the section. Energy changes, not labels."""
    m = a.meta
    dur = float(m["duration"])
    bpm = m["bpm_map"][0]["bpm"] if m.get("bpm_map") else None
    w = float(window_s) if window_s else (4 * 60.0 / bpm if bpm else 2.0)
    g = float(m.get("grid_offset") or 0.0) % w
    edges = np.arange(g, dur, w)
    if g >= 0.25 * w:
        edges = np.concatenate(([0.0], edges))               # a real pickup before the first bar line
    elif len(edges):
        edges[0] = 0.0                                       # a sliver before the first bar joins the first bar
    if len(edges) < 2:
        return [{"time": 0.0, "level": None, "rose": [], "fell": []}], None
    L = band_levels(a, edges, dur)
    names = [b[0] for b in SECTION_BANDS]
    cands = []
    for i in range(1, L.shape[1]):
        d = L[:, i] - L[:, i - 1]
        score = float(np.abs(d).sum())
        if score >= 0.9 or np.abs(d).max() >= 0.35:
            cands.append((i, score, d))
    keep = []
    for c in sorted(cands, key=lambda c: -c[1]):                 # strongest first; drop near neighbours
        if all(abs(c[0] - k[0]) >= min_windows for k in keep):
            keep.append(c)
    keep.sort(key=lambda c: c[0])
    starts = [0] + [k[0] for k in keep]
    out = []
    for j, s0 in enumerate(starts):
        s1 = starts[j + 1] if j + 1 < len(starts) else L.shape[1]
        d = keep[j - 1][2] if j else np.zeros(len(names))
        out.append({"time": round(float(edges[s0]) if s0 else 0.0, 3), "level": round(float(L[:, s0:s1].mean()), 3),
                    "rose": [names[k] for k in np.flatnonzero(d >= 0.2)], "fell": [names[k] for k in np.flatnonzero(d <= -0.2)]})
    return out, L


def name_sections(secs):
    """heuristic, editable names from each section's energy relative to the song: the first is "intro", then
    "breakdown" (clearly quieter than the median section), "peak" (clearly louder) or "main"; repeats get _2, _3"""
    levels = [s["level"] for s in secs if s["level"] is not None]
    med = float(np.median(levels)) if levels else 0.0
    out, used = [], {}
    for i, s in enumerate(secs):
        lv = s["level"] if s["level"] is not None else med
        base = "intro" if i == 0 else "breakdown" if lv < med - 0.12 else "peak" if lv > med + 0.12 else "main"
        used[base] = used.get(base, 0) + 1
        out.append({"name": base if used[base] == 1 else f"{base}_{used[base]}", "time": s["time"]})
    return out


# ---------- song structure (labelled sections on bar lines)
STRUCT_WIDTHS = (2, 4, 8)          # bars each side of a candidate boundary in the contrast novelty
STRUCT_FLOOR_DB = 24.0             # levels more than this under the song's loud bars count as "absent", all alike
STRUCT_Z = 1.5                     # novelty peak (robust z) that makes a boundary
PHRASE_BONUS = 0.25                # novelty bonus on 8-bar phrase lines (half on 4-bar lines)
BASS_ON_DB = -15.0                 # a bar "has bass" when the bass stem is within this of its loud bars
BASS_FLOOR_ON_DB = -6.0            # ... or, from the mix alone, when the 50-250 Hz floor between hits is
RISE_DB = 6.0                      # a full section whose rising part (other stem / mid bands) climbs this much is a build


def bar_lines(a):
    """bar line times over the whole song: the analysis' downbeats (the model's bars, extended by whole bars before
    the first and after the last), else bars of 4 beats from the grid offset. A bar line starting in the last quarter
    bar of the file is dropped."""
    m = a.meta
    dur = float(m["duration"])
    bpm = m["bpm_map"][0]["bpm"] if m.get("bpm_map") else 120.0
    bars = m.get("bars") or {}
    bar_len = int(bars.get("meter") or 4) * 60.0 / bpm
    downs = np.asarray(bars.get("downbeats") or [], np.float64)
    if len(downs):
        pre = np.arange(downs[0] - bar_len, 0.25 * bar_len - 1e-9, -bar_len)[::-1]
        post = np.arange(downs[-1] + bar_len, dur, bar_len)
        edges = np.concatenate((pre, downs, post))
    else:
        edges = np.arange(float(m.get("grid_offset") or 0.0) % bar_len, dur, bar_len)
    edges = edges[edges < dur - 0.25 * bar_len]
    if not len(edges) or edges[0] > 0.25 * bar_len:
        edges = np.concatenate(([0.0], edges))                # a pickup before the first bar line
    else:
        edges[0] = 0.0                                        # a sliver before the first bar joins the first bar
    return edges, bar_len


def _bar_db(env, edges, dur):
    """per bar RMS level in dB relative to the song's loud bars (95th percentile), floored at -STRUCT_FLOOR_DB"""
    out = []
    for e0, e1 in zip(edges, list(edges[1:]) + [dur]):
        seg = env[int(e0 * 1000):int(e1 * 1000)].astype(np.float64)
        out.append(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) if len(seg) else -200.0)
    out = np.array(out)
    return np.clip(out - np.percentile(out, 95), -STRUCT_FLOOR_DB, 6.0)


def _bar_floor_db(env, edges, dur, pct=20):
    """per bar, the pct-th percentile of the envelope in dB re the song's loud bars: a sustained bass keeps the low
    band up between kicks, a kick alone lets it fall"""
    out = []
    for e0, e1 in zip(edges, list(edges[1:]) + [dur]):
        seg = env[int(e0 * 1000):int(e1 * 1000)].astype(np.float64)
        out.append(20 * np.log10(np.percentile(seg, pct) + 1e-9) if len(seg) else -200.0)
    out = np.array(out)
    return np.clip(out - np.percentile(out, 95), -STRUCT_FLOOR_DB, 6.0)


def bar_features(a, edges):
    """per bar: {"mix", stem names (dB re the loud bars), "kick"/"snare"/"hats" (hits per bar, from the drums stem
    when there is one), "bands" ([7, bars] SECTION_BANDS levels 0..1), "bass" (the bass stem, else the floor of the
    50-250 Hz envelope between hits), "rise" (dB of what builds up: the other stem, else the mid bands)}"""
    from .drums import part_hits
    dur = float(a.meta["duration"])
    a.ensure([("loud", None, None, None)])
    f = {"mix": _bar_db(a.envelope("loud", None, None, 10, 150), edges, dur)}
    stems = getattr(a, "stems", None) or {}
    for s in ("drums", "bass", "other", "vocals"):
        if s in stems:
            stems[s].ensure([("loud", None, None, None)])
            f[s] = _bar_db(stems[s].envelope("loud", None, None, 10, 150), edges, dur)
    src = stems.get("drums", a)
    bins = list(edges) + [dur]
    for part in ("kick", "snare", "hats"):
        f[part] = np.histogram((part_hits(src, part)[0] + 1) * HOP_MS / 1000.0, bins=bins)[0].astype(np.float64)
    f["bands"] = band_levels(a, edges, dur)
    if "bass" not in f:                                       # no stems: the floor of the 50-250 Hz band between hits
        f["bass"] = _bar_floor_db(a.envelope("band", 50, 250, 10, 150), edges, dur)
    if "other" in f:
        f["rise"] = f["other"]
    else:
        mid = f["bands"][3:6].mean(axis=0)
        f["rise"] = (mid - 1.0) * STRUCT_FLOOR_DB
    return f


def structure_novelty(F, widths=STRUCT_WIDTHS):
    """contrast novelty: for the bar line before bar i, the distance between the mean feature vector of the w bars
    before and the w bars after, averaged over the widths (a multi-scale checkerboard on the bar self-similarity)"""
    n = len(F)
    nov = np.zeros(n)
    for i in range(1, n):
        nov[i] = np.mean([np.linalg.norm(F[max(0, i - w):i].mean(axis=0) - F[i:min(n, i + w)].mean(axis=0)) for w in widths])
    return nov


def _phrase_offset(nov):
    """the bar (mod 8) where 8-bar phrases start: the residue with the most novelty"""
    return int(np.argmax([nov[o::8].sum() for o in range(8)])) if len(nov) else 0


def song_structure(a, min_bars=None):
    """Labelled song sections on bar lines: [{"name", "time", "bar", "label", "bars", "kick", "bass_db", "energy_db",
    "novelty"}], the first at time 0.
    - Features per bar: the mix level, each stem's level (all levels in dB under the song's loud bars, floored at
      -24 dB so "absent" is one value), kick / snare / hats hits per bar (drum parts, from the drums stem when there
      is one) and the 7 SECTION_BANDS levels.
    - Boundaries: peaks of a multi-scale contrast novelty over those features (2, 4 and 8 bars each side), with a
      25 % bonus on the song's 8-bar phrase lines, kept where the peak's robust z-score is at least 1.5, strongest
      first and at least min_bars apart (default 8 bars in a song of 64 bars or more, 4 from 24 bars, else 2). Drums
      (a kick on half the beats) or bass (the stem within 15 dB of its loud bars; from the mix, the 50-250 Hz floor
      between hits within 6 dB) coming in or going out for 2 bars or more also makes a boundary, and so does a bar
      with no kick between kick bars (a one-bar drop-out), on the bar after it. Boundaries are bar lines, so a
      section never starts before its music changes.
    - Labels from what plays: no kick = "breakdown" (shorter than 3 bars: "break"); kick without bass = "build" (after
      a breakdown or a break) or "intro" (first); kick and bass = "intro" (first), "drop" (after a build, breakdown or
      break, or straight after a drop and at least as loud), "build" (what rises climbs 6 dB or more and a breakdown
      or drop follows), "dip" (2 dB or more under the song's other full sections) or "groove". The last section is
      "outro" when it has no kick or is 3 dB or more under the full sections. Repeats are numbered (drop_2, ...).
    Offline: the novelty compares bars before and after each line; the times are bar lines, never early."""
    edges, bar_len = bar_lines(a)
    nb = len(edges)
    dur = float(a.meta["duration"])
    if nb < 2:
        return [{"name": "intro", "time": 0.0, "bar": 0, "label": "intro", "bars": nb, "kick": None, "bass_db": None,
                 "energy_db": None, "novelty": None}]
    if min_bars is None:                                      # phrases of 8 bars in a song; a short clip has short parts
        min_bars = 8 if nb >= 64 else 4 if nb >= 24 else 2
    f = bar_features(a, edges)
    meter = int((a.meta.get("bars") or {}).get("meter") or 4)
    stems = getattr(a, "stems", None) or {}
    cols = [f["mix"] / 6.0] + [f[s] / 6.0 for s in ("drums", "bass", "other", "vocals") if s in stems]
    if "bass" not in stems:
        cols.append(f["bass"] / 6.0)                          # the mix's bass bands stand in
    cols += [f["kick"] / (meter / 2), f["snare"] / (meter / 2), f["hats"] / meter]
    cols += list(f["bands"] * 4.0)
    F = np.stack(cols, axis=1)
    raw = structure_novelty(F)
    off = _phrase_offset(raw)
    bonus = np.array([1 + PHRASE_BONUS if (i - off) % 8 == 0 else 1 + PHRASE_BONUS / 2 if (i - off) % 4 == 0 else 1.0 for i in range(nb)])
    nov = raw * bonus
    med = float(np.median(nov[1:]))
    mad = float(np.median(np.abs(nov[1:] - med))) or 1e-9
    z = (nov - med) / mad
    padded = np.concatenate((np.zeros(2), nov, np.zeros(2)))
    cands = [(float(z[i]), i) for i in range(1, nb) if nov[i] == padded[i:i + 5].max() and z[i] >= STRUCT_Z]
    kick_on = f["kick"] >= meter / 2
    bass_on = f["bass"] >= (BASS_ON_DB if "bass" in stems else BASS_FLOOR_ON_DB)
    for i in range(1, nb - 1):                                # one-bar drop-outs start a section on the bar after
        if not kick_on[i] and kick_on[i - 1] and kick_on[i + 1] and (i < 2 or kick_on[i - 2]):
            cands.append((STRUCT_Z + 0.5, i + 1))
    for i in range(2, nb - 1):                                # drums or bass come in or go out (and stay for 2 bars)
        for on in (kick_on, bass_on):
            if on[i] == on[i + 1] and on[i - 1] == on[i - 2] and on[i] != on[i - 1]:
                cands.append((STRUCT_Z + 1.0, i))
    keep = []
    for score, i in sorted(cands, key=lambda c: (-c[0], c[1])):
        if all(abs(i - k) >= min_bars for k in keep) and i >= min(min_bars, 4) and nb - i >= 1:
            keep.append(i)
    starts = [0] + sorted(keep)
    secs = []
    for j, s0 in enumerate(starts):
        s1 = starts[j + 1] if j + 1 < len(starts) else nb
        sl = slice(s0, s1)
        q = max(1, (s1 - s0) // 4)
        secs.append({"bar": s0, "bars": s1 - s0, "time": round(float(edges[s0]), 3) if s0 else 0.0,
                     "kick": round(float(np.median(f["kick"][sl])), 2), "kick_on": bool(np.median(kick_on[sl].astype(float)) >= 0.5),
                     "bass_db": round(float(np.median(f["bass"][sl])), 1), "bass_on": bool(np.median(bass_on[sl].astype(float)) >= 0.5),
                     "energy_db": round(float(np.mean(f["mix"][sl])), 1),
                     "rise_db": float(np.mean(f["rise"][s1 - q:s1]) - np.mean(f["rise"][s0:s0 + q])),
                     "novelty": round(float(z[s0]), 1) if s0 else None})
    full = [s for s in secs if s["kick_on"] and s["bass_on"]]
    full_db = float(np.median([s["energy_db"] for s in full])) if full else 0.0
    labels = []
    for j, s in enumerate(secs):
        prev = labels[-1] if labels else None
        nxt_kick = secs[j + 1]["kick_on"] if j + 1 < len(secs) else None
        if not s["kick_on"]:
            lab = "intro" if j == 0 else "breakdown" if s["bars"] >= 3 else "break"
        elif not s["bass_on"]:
            lab = "intro" if j == 0 else "build"
        elif j == 0:
            lab = "intro"
        elif prev in ("build", "breakdown", "break"):
            lab = "drop"
        elif prev == "drop" and s["energy_db"] >= secs[j - 1]["energy_db"] - 1.0:
            lab = "drop"
        elif s["rise_db"] >= RISE_DB and j + 1 < len(secs) and (nxt_kick is False or not secs[j + 1]["bass_on"]):
            lab = "build"
        elif s["energy_db"] <= full_db - 2.0:
            lab = "dip"
        else:
            lab = "groove"
        if j == len(secs) - 1 and j > 0 and (not s["kick_on"] or s["energy_db"] <= full_db - 3.0):
            lab = "outro"
        labels.append(lab)
    used, out = {}, []
    for s, lab in zip(secs, labels):
        used[lab] = used.get(lab, 0) + 1
        out.append({"name": lab if used[lab] == 1 else f"{lab}_{used[lab]}", "time": s["time"], "bar": s["bar"], "label": lab,
                    "bars": s["bars"], "kick": s["kick"], "bass_db": s["bass_db"], "energy_db": s["energy_db"],
                    "novelty": s["novelty"]})
    return out


def open_analysis(audio: str, cache_dir: str, settings: Optional[dict] = None, default_weights=None, use_cache: bool = True) -> "Analysis":
    """load the cache if the audio's size/mtime and the settings match, else run the base analysis"""
    a = Analysis(audio, cache_dir, settings, default_weights)
    memo_key = (a.ident_hash, os.path.abspath(cache_dir), json.dumps(default_weights))
    if use_cache and memo_key in _MEMO:
        return _MEMO[memo_key]
    if not (use_cache and a._load()):
        a.build_base()
    _MEMO[memo_key] = a
    return a
