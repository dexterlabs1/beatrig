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
"""
import hashlib
import json
import os
from typing import Optional

import numpy as np
import scipy.signal as sg
import soundfile as sf

ANALYSIS_VERSION = 3
ONSET_EDGES = [30, 60, 120, 240, 480, 960, 1920, 3840, 7680, 15360]   # octave bands for the onset function
STEP_MS = 1                 # envelope grid
HOP_MS = 5                  # onset grid
CHUNK_S = 10                # read size
DEFAULT_SETTINGS = dict(bpm_min=75.0, bpm_max=150.0, pulse_z=4.0, min_conf=0.15, min_agree=0.15, win_s=16.0, hop_s=4.0)

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
    e = (np.arange(nb + 1) * sr * step_ms) // 1000
    cs = np.concatenate((np.zeros((1,) + y.shape[1:]), np.cumsum(np.square(y), axis=0)))
    return (cs[e[1:]] - cs[e[:-1]]) / np.diff(e).reshape((-1,) + (1,) * (y.ndim - 1))


def follow(x, attack_ms, release_ms, dt_ms=1.0):
    """one-pole up/down follower on past samples only (~0.5 s per 3.6M steps, i.e. a 60-min show on the 1 ms grid)"""
    a, r = 1 - np.exp(-dt_ms / attack_ms), 1 - np.exp(-dt_ms / release_ms)
    y = np.empty(len(x), dtype=np.float64)
    v = 0.0
    for i, s in enumerate(np.asarray(x, np.float64).tolist()):
        v += (a if s > v else r) * (s - v)
        y[i] = v
    return y


def at_frames(env, step_ms, frames, fps, offset=0):
    """value for sequence frame f = env at the frame's start time (audio time (f - offset) / fps); never reads the
    future. Grid index k is known at (k + 1) * step ms, so frame f reads k = floor(t_ms / step) - 1. Frames before the
    audio read silence (0); frames past the end hold the last value."""
    t_ms = (np.arange(frames) - offset) * 1000.0 / fps
    k = np.floor(t_ms / step_ms).astype(np.int64) - 1
    if len(env) == 0:
        return np.zeros(frames, np.float32)
    return np.where(k >= 0, env[np.clip(k, 0, len(env) - 1)], 0.0).astype(np.float32)


def time_to_frame(t, fps, offset=0):
    """first sequence frame whose start is at or after audio time t"""
    return (np.ceil(np.round(np.asarray(t, np.float64) * fps, 6)).astype(np.int64) + int(offset))


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


def pulse_windows(f, step, win_s=16, hop_s=4, seed=1):
    """for each window: best period, comb score, z-score against a block-shuffled null (periodicity destroyed, texture kept)"""
    rng = np.random.default_rng(seed)
    W, Hp = int(round(win_s / step)), int(round(hop_s / step))
    lags = np.arange(int(0.30 / step), int(1.25 / step))
    block = int(0.15 / step)
    out = []
    if len(f) < lags[-1] + 2:
        return out
    for s in range(0, max(1, len(f) - W + 1), Hp):
        w = f[s:s + W]
        c = comb(acf(w), lags)
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


def agreement(times, ons, T, win=0.05):
    """fraction of beats within +-win of a reference onset, and the same at other phases of the period (chance at this density)"""
    if not len(ons):
        return 0.0, 0.0
    near = lambda t: float(np.mean(np.min(np.abs(t[:, None] - ons[None, :]), axis=1) <= win))
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


def beat_grid(f, ref_ons, duration, s):
    """verified beats: [{time, strength, confidence}], accepted/rejected passages, bpm map, grid offset, pulse summary.
    time = (k + 1) * 5 ms, when the onset is known. With no proven pulse, the strongest window's folded period is used
    as a low-confidence tempo (fallback=True) and no beats are written."""
    step = HOP_MS / 1000
    # short clips: shrink the window so at least ~5 windows (3 are needed for a passage) fit
    win = min(s["win_s"], max(4.0, duration / 2))
    hop = min(s["hop_s"], win / 4)
    wins = pulse_windows(f, step, win, hop)
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
        # tempo from a line fit through the beat times: the comb's period is quantised to the 5 ms lag grid
        n_idx = np.round((times - times[0]) / T)
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
    return dict(beats=beats, passages=pas, rejected_passages=rejected, bpm_map=bpm_map, grid_offset=round(float(grid_offset), 6),
                fallback=fallback, pulse=pulse)


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
                          settings=self.settings)
        self.ident_hash = hashlib.sha256(json.dumps(self.ident, sort_keys=True).encode()).hexdigest()
        self.arrays = {}
        self.meta = {}
        self.dirty = False
        self.loaded_from_cache = False

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
        self.meta, self.arrays, self.loaded_from_cache = meta, arrays, True
        return True

    def save(self):
        if not self.dirty:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        meta = dict(self.meta, ident=self.ident, ident_hash=self.ident_hash, arrays=sorted(self.arrays))
        tmp = self.npz_path + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, **self.arrays)
        os.replace(tmp, self.npz_path)
        with open(self.json_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(meta, fh, indent=1, sort_keys=True)
            fh.write("\n")
        self.dirty = False

    # -- the causal pass
    def _scan(self, specs, base):
        """one pass over the file. specs: {key: (sos | None, weights)} -> 1 ms weighted power (sum over channels of
        w_c * power of the filtered channel, so phase differences between channels cannot cancel). base: onset octave powers, 5 ms peaks, reference mono."""
        sr, C = self.sr, self.channels
        st = {k: dict(sos=sos, w=w, zi=None if sos is None else np.zeros((sos.shape[0], 2, C)), out=[]) for k, (sos, w) in specs.items()}
        if base:
            octs = [dict(sos=band_sos(lo, hi, sr, 2), out=[]) for lo, hi in zip(ONSET_EDGES[:-1], ONSET_EDGES[1:]) if lo < sr / 2]
            for o in octs:
                o["zi"] = np.zeros((o["sos"].shape[0], 2))
            peaks = []
            q = max(1, int(round(sr / 12000)))
            lp = sg.butter(6, 0.8 * sr / q / 2, fs=sr, output="sos")
            lpzi, ref, pos = np.zeros((lp.shape[0], 2)), [], 0
        for blk in sf.blocks(self.audio, blocksize=sr * CHUNK_S, dtype="float32", always_2d=True):
            x = blk.astype(np.float64)
            for s in st.values():
                y = x
                if s["sos"] is not None:
                    y, s["zi"] = sg.sosfilt(s["sos"], x, axis=0, zi=s["zi"])
                s["out"].append(grid_power(y, sr, STEP_MS) @ s["w"])
            if base:
                mono = x.mean(axis=1)
                for o in octs:
                    yb, o["zi"] = sg.sosfilt(o["sos"], mono, zi=o["zi"])
                    o["out"].append(grid_power(yb, sr, HOP_MS))
                nb = (len(mono) * 1000) // (sr * HOP_MS)
                if nb:
                    e = (np.arange(nb + 1) * sr * HOP_MS) // 1000
                    peaks.append(np.maximum.reduceat(np.abs(mono[:e[-1]]), e[:-1]))
                yl, lpzi = sg.sosfilt(lp, mono, zi=lpzi)
                ref.append(yl[(-pos) % q::q])
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
            self._ref = (np.concatenate(ref).astype(np.float32) if ref else np.zeros(0, np.float32), sr / q)
        self.dirty = True

    def _reference_onsets(self):
        """independent onset list (librosa flux, ~21 ms window), used only to check our beats, never to place them"""
        y, rsr = self._ref
        if len(y) < 512:
            return np.zeros(0)
        import warnings
        import librosa
        hop = max(1, int(round(0.005 * rsr)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)        # librosa: empty mel filters at this small n_fft
            env = librosa.onset.onset_strength(y=y, sr=rsr, n_fft=256, hop_length=hop, n_mels=40)
            return librosa.onset.onset_detect(onset_envelope=env, sr=rsr, hop_length=hop, units="time", backtrack=False)

    def build_base(self):
        self._scan({}, base=True)
        g = beat_grid(self.arrays["onset"].astype(np.float64), self._reference_onsets(), self.duration, self.settings)
        self.meta = dict(sr=self.sr, channels=self.channels, n_samples=self.n_samples, duration=round(self.duration, 6), **g)
        del self._ref

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
            self.arrays[ek] = _f32(follow(np.sqrt(self.arrays["pow:" + key].astype(np.float64)), attack_ms, release_ms, STEP_MS))
            self.dirty = True
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
                    peaks_per_s=200 / n, peaks=np.round(pk, 4).tolist())


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
