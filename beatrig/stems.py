"""Source separation (optional extra: pip install beatrig[stems], which brings torch + demucs).

`beatrig stems <rig>` runs htdemucs once per audio file and caches drums/bass/vocals/other as wav next to the audio,
in <audio name>.stems/<content hash>/, so moving or re-baking never separates twice and an edited file never reuses
stale stems. band/loudness/onsets nodes then read a stem with "stem": "drums". Each stem gets its own analysis cache
inside its folder. The core install never imports torch.

Separation is offline and looks at whole segments of audio, not only the past. The analysis of a stem is causal as
usual; the stem itself can carry a little pre-echo, so check onset timing on your material (see docs/ops.md).
"""
import hashlib
import json
import os
import time

import numpy as np
import soundfile as sf

STEMS = ["drums", "bass", "vocals", "other"]
MODEL = "htdemucs"


def content_hash(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stem_dir(audio, digest=None, model=MODEL):
    base, _ = os.path.splitext(os.path.abspath(audio))
    digest = digest or content_hash(audio)
    return os.path.join(base + ".stems", f"{model}-{digest[:16]}")


def cached(audio, model=MODEL):
    """{stem: wav path} when this audio's stems exist, else None. Hashes the file (about 1 s per GB)."""
    d = stem_dir(audio, model=model)
    paths = {s: os.path.join(d, s + ".wav") for s in STEMS}
    return paths if all(os.path.exists(p) for p in paths.values()) else None


def available():
    """(ok, reason): can this install separate stems?"""
    try:
        import torch  # noqa: F401
        import demucs.pretrained  # noqa: F401
    except ImportError as e:
        return False, f"{e.name or e} is not installed: pip install beatrig[stems]"
    return True, ""


def _stereo(x):
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x[:, :2]                                            # the model is stereo: a multichannel mix uses its front pair


def separate(audio, model=MODEL, device=None, chunk_s=120.0, context_s=4.0, log=print):
    """run the separation (if not cached) and return {stem: wav path}. device: "cuda", "cpu" or None (cuda if available).
    A long file is processed in chunk_s pieces, each with context_s of audio on both sides and a linear crossfade over
    the shared context, written straight to disk: memory stays flat for a 60-min show (one chunk, not 4 full stems)."""
    d = stem_dir(audio, model=model)
    paths = {s: os.path.join(d, s + ".wav") for s in STEMS}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    net = get_model(model)
    net.eval()
    info = sf.info(audio)
    sr, n_in = int(info.samplerate), int(info.frames)
    msr = int(net.samplerate)
    s1 = s2 = 0.0                                              # whole-file mono mean/std, as the reference CLI normalises
    for blk in sf.blocks(audio, blocksize=sr * 30, dtype="float64", always_2d=True):
        m = _stereo(blk).mean(axis=1)
        s1, s2 = s1 + m.sum(), s2 + np.square(m).sum()
    mu = s1 / max(1, n_in)
    sd = float(np.sqrt(max(s2 / max(1, n_in) - mu * mu, 0.0))) + 1e-8
    n_out = int(round(n_in * msr / sr))
    C, X = int(round(chunk_s * msr)), int(round(context_s * msr))
    tmp = {s: os.path.join(d, s + ".part.wav") for s in STEMS}
    os.makedirs(d, exist_ok=True)
    writers = {s: sf.SoundFile(tmp[s], "w", msr, 2, "FLOAT") for s in STEMS}
    prev_extra = None
    try:
        for b0 in range(0, n_out, C):
            b1 = min(n_out, b0 + C)
            w0, w1 = max(0, b0 - X), min(n_out, b1 + X)        # window, in model-rate samples
            i0, i1 = int(np.floor(w0 * sr / msr)), min(n_in, int(np.ceil(w1 * sr / msr)) + 1)
            x, _ = sf.read(audio, start=i0, stop=i1, dtype="float32", always_2d=True)
            wav = torch.from_numpy(_stereo(x).T.copy())
            if sr != msr:
                import julius
                wav = julius.resample_frac(wav, sr, msr)
            off = int(round(i0 * msr / sr))
            need = w1 - off
            if wav.shape[1] < need:
                wav = torch.nn.functional.pad(wav, (0, need - wav.shape[1]))
            wav = wav[:, w0 - off:w1 - off]
            with torch.no_grad():
                out = apply_model(net, ((wav - mu) / sd)[None].float(), device=dev, shifts=1, split=True, overlap=0.25,
                                  progress=False)[0]
            out = (out * sd + mu).cpu().numpy()                # [sources, 2, w1 - w0]
            body = out[:, :, b0 - w0:b1 - w0].copy()
            if prev_extra is not None:                         # crossfade the shared context with the previous chunk
                k = min(prev_extra.shape[2], body.shape[2])
                ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
                body[:, :, :k] = prev_extra[:, :, :k] * (1 - ramp) + body[:, :, :k] * ramp
            prev_extra = out[:, :, b1 - w0:].copy()
            for name, src in zip(net.sources, body):
                if name in writers:
                    writers[name].write(src.T)
    finally:
        for w in writers.values():
            w.close()
    for s in STEMS:
        os.replace(tmp[s], paths[s])
    with open(os.path.join(d, "stems.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"model": model, "audio": os.path.basename(audio), "samplerate": msr, "device": dev, "chunk_s": chunk_s,
                   "context_s": context_s, "seconds": round(time.perf_counter() - t0, 1)}, fh, indent=1)
    log(f"stems: {model} on {dev}, {time.perf_counter() - t0:.1f} s -> {d}")
    return paths


def used_stems(nodes):
    """the stems a rig's nodes read (a node's own "stem", or an onsets node's referenced band's)"""
    by_id = {n.get("id"): n for n in nodes}
    out = set()
    for n in nodes:
        st = n.get("stem")
        if not st and n.get("op") == "onsets":
            ref = n.get("in")
            src = ref if isinstance(ref, dict) else by_id.get(ref) or {}
            st = src.get("stem")
        if st:
            out.add(st)
    return sorted(out)
