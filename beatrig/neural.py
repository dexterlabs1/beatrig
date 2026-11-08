"""Neural beat and downbeat activations (optional extra: pip install beatrig[ml]).

The model is "Beat This!" (Foscarin, Schlueter and Widmer, ISMIR 2024; code and weights MIT, github.com/CPJKU/beat_this):
a transformer over a log-mel spectrogram that gives a beat and a downbeat logit per 20 ms frame, with no tempo prior
and no DBN. The weights (about 77 MB) download once into the torch hub cache.

This module only computes the activations. It never decides where a beat goes: analysis.neural_grid() keeps a beat
only where a real onset supports it and places it at that onset, so the timing rule (never ahead of the audio) and the
"beats only where an onset supports them" rule hold as for the classic grid. The model looks at 30 s of audio around a
frame, past and future, which is fine for an offline analysis that is cached.

The log-mel front end is computed here in numpy (22.05 kHz mono, 1024-point Hann STFT, hop 441, 128 Slaney mel bands
30 Hz..11 kHz, magnitude scaled by 1/sqrt(1024), log(1 + 1000 x)): the spectrogram the model was trained on, without
importing torchaudio. The core install never imports torch.
"""
import importlib.util
import inspect
import math
import os
import time

import numpy as np
import soundfile as sf

MODEL = "final0"
CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"
SR = 22050
HOP = 441                   # 50 frames per second
FPS = SR / HOP
N_FFT = 1024
N_MELS = 128
F_MIN, F_MAX = 30.0, 11000.0
CHUNK = 1500                # model frames per inference chunk (30 s, the training length)
BORDER = 6                  # frames dropped at each chunk edge (the model was not trained on them)
READ_S = 60                 # audio read per front-end chunk
_MODELS = {}                # (checkpoint, device) -> model, so a server analyses several files with one load


def available():
    """(ok, reason): can this install run the neural tracker? Looks the packages up without importing them (importing
    torch takes seconds, and a core install must never import it)."""
    for mod, pip in (("torch", "torch"), ("beat_this", "beat-this"), ("einops", "einops"),
                     ("rotary_embedding_torch", "rotary-embedding-torch")):
        try:
            if importlib.util.find_spec(mod) is None:
                return False, f"{pip} is not installed: pip install beatrig[ml]"
        except (ImportError, ValueError):
            return False, f"{pip} is not importable: pip install beatrig[ml]"
    return True, ""


def disabled():
    """BEATRIG_EXTRAS=off makes "auto" analysis settings behave like a core install (CI, or reproducing one)"""
    return os.environ.get("BEATRIG_EXTRAS", "").strip().lower() in ("off", "0", "false", "no")


# ---------- front end
def _hz_to_mel(f):
    f = np.asarray(f, np.float64)
    return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / (np.log(6.4) / 27.0), f / (200.0 / 3.0))


def _mel_to_hz(m):
    m = np.asarray(m, np.float64)
    return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), m * 200.0 / 3.0)


def mel_basis():
    """[N_FFT/2 + 1, N_MELS] triangular Slaney-scale filters with unit peaks (no area normalisation)"""
    freqs = np.linspace(0.0, SR // 2, N_FFT // 2 + 1)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2))
    fd = np.diff(pts)
    slopes = pts[None, :] - freqs[:, None]
    return np.maximum(0.0, np.minimum(-slopes[:, :-2] / fd[:-1], slopes[:, 2:] / fd[1:])).astype(np.float32)


def _mono_22k(audio, weights=None):
    """the file as 22.05 kHz mono float32, resampled in READ_S pieces with 0.5 s of context on both sides (chunk
    boundaries fall on whole periods of the resampling ratio, so the pieces join sample-exactly)"""
    from math import gcd
    import scipy.signal as sg
    info = sf.info(audio)
    sr, n = int(info.samplerate), int(info.frames)
    g = gcd(SR, sr)
    up, down = SR // g, sr // g
    w = None
    if weights is not None:
        w = np.asarray(weights, np.float64)
        w = w / (w.sum() or 1.0)
    step = max(down, (READ_S * sr) // down * down)
    ctx = (sr // 2) // down * down + down
    out = []
    for a in range(0, n, step):
        b = min(n, a + step)
        a0, b0 = max(0, a - ctx), min(n, b + ctx)
        x, _ = sf.read(audio, start=a0, stop=b0, dtype="float64", always_2d=True)
        m = x.mean(axis=1) if w is None or len(w) != x.shape[1] else x @ w
        y = sg.resample_poly(m, up, down) if (up, down) != (1, 1) else m
        i0 = (a - a0) * up // down
        out.append(y[i0:i0 + (b - a) * up // down + (1 if b == n else 0)].astype(np.float32))
    y = np.concatenate(out) if out else np.zeros(0, np.float32)
    return y[:int(math.ceil(n * up / down))]


def logmel(audio, weights=None):
    """[frames, 128] float32 log-mel spectrogram at 50 frames per second; frame k is centred on k / 50 s"""
    y = _mono_22k(audio, weights)
    if len(y) <= N_FFT // 2:
        y = np.pad(y, (0, N_FFT // 2 + 1 - len(y)))
    y = np.pad(y, (N_FFT // 2, N_FFT // 2), mode="reflect")
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)   # periodic Hann
    fb = mel_basis()
    n_frames = 1 + (len(y) - N_FFT) // HOP
    out = np.empty((n_frames, N_MELS), np.float32)
    for f0 in range(0, n_frames, 8192):
        f1 = min(n_frames, f0 + 8192)
        fr = np.lib.stride_tricks.sliding_window_view(y[f0 * HOP:(f1 - 1) * HOP + N_FFT], N_FFT)[::HOP][:f1 - f0]
        mag = np.abs(np.fft.rfft(fr * win, axis=1)).astype(np.float32) / np.float32(np.sqrt(N_FFT))
        out[f0:f1] = np.log1p(np.float32(1000.0) * (mag @ fb))
    return out


# ---------- model
def load_model(checkpoint=MODEL, device="cpu"):
    """the Beat This! model in eval mode on `device` (weights from the torch hub cache, downloaded on first use)"""
    key = (checkpoint, str(device))
    if key in _MODELS:
        return _MODELS[key]
    import torch
    from beat_this.model.beat_tracker import BeatThis
    from beat_this.utils import replace_state_dict_key
    if os.path.exists(str(checkpoint)):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    else:
        ck = torch.hub.load_state_dict_from_url(f"{CHECKPOINT_URL}/{checkpoint}.ckpt", file_name=f"beat_this-{checkpoint}.ckpt",
                                                map_location="cpu", progress=False)
    params = set(inspect.signature(BeatThis).parameters)
    net = BeatThis(**{k: v for k, v in ck["hyper_parameters"].items() if k in params})
    net.load_state_dict(replace_state_dict_key(ck["state_dict"], "model.", ""))
    net = net.to(device).eval()
    _MODELS[key] = net
    return net


def chunk_starts(n, size=CHUNK, border=BORDER):
    """chunk start frames: chunks overlap by 2 * border, the first starts at -border (zero padded) and the last ends
    at the end of the piece"""
    starts = np.arange(-border, n - border, size - 2 * border)
    if n > size - 2 * border:
        starts[-1] = n - (size - border)
    return [int(s) for s in starts]


def frame_logits(spect, checkpoint=MODEL, device=None, batch=8):
    """(beat logits, downbeat logits), float32 [frames]. Each 30 s chunk's border frames are dropped; where chunks
    overlap the earlier chunk wins."""
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(checkpoint, dev)
    n = len(spect)
    beat = np.full(n, -1000.0, np.float32)
    down = np.full(n, -1000.0, np.float32)
    starts = chunk_starts(n)
    x = torch.from_numpy(np.ascontiguousarray(spect, np.float32))
    pieces = []
    for s in starts:
        c = x[max(s, 0):min(s + CHUNK, n)]
        pieces.append(torch.nn.functional.pad(c, (0, 0, max(0, -s), max(0, min(BORDER, s + CHUNK - n)))))
    res = {}
    with torch.inference_mode():
        for b0 in range(0, len(pieces), batch):
            grp = pieces[b0:b0 + batch]
            L = max(len(p) for p in grp)
            if any(len(p) != L for p in grp):                 # only a lone short piece differs; run it on its own
                outs = [net(p[None].to(dev)) for p in grp]
                outs = [{k: o[k][0] for k in ("beat", "downbeat")} for o in outs]
            else:
                o = net(torch.stack(grp).to(dev))
                outs = [{k: o[k][i] for k in ("beat", "downbeat")} for i in range(len(grp))]
            for i, o in enumerate(outs):
                res[starts[b0 + i]] = (o["beat"].float().cpu().numpy(), o["downbeat"].float().cpu().numpy())
    for s in reversed(starts):                                  # earlier chunks overwrite later ones
        b, d = res[s]
        b, d = b[BORDER:len(b) - BORDER], d[BORDER:len(d) - BORDER]
        lo, hi = s + BORDER, min(n, s + BORDER + len(b))
        beat[lo:hi], down[lo:hi] = b[:hi - lo], d[:hi - lo]
    return beat, down


def activations(audio, weights=None, checkpoint=MODEL, device=None):
    """run the tracker on a file: {"beat", "downbeat": float32 logits per frame, "fps": 50, "model", "device",
    "seconds"}"""
    import torch
    t0 = time.perf_counter()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    spect = logmel(audio, weights)
    beat, down = frame_logits(spect, checkpoint, dev)
    return {"beat": beat, "downbeat": down, "fps": FPS, "model": f"beat_this {checkpoint}", "device": dev,
            "seconds": round(time.perf_counter() - t0, 2)}
