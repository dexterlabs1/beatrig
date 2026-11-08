"""The op set. Every op is causal: a frame's value depends only on audio up to the frame's start and on earlier frames.
Each op: fn(node, inputs: [Signal], ctx) -> Signal. SPECS holds the params (defaults; REQ = required), the input
arity, static channel/kind inference used by check(), and whether the op accepts colour signals.
The colour, per-hit flow, spatial and safety families live in their own modules and register here on import."""
import numpy as np

from .analysis import at_frames, follow, norm_db, sg, time_to_frame, HOP_MS, STEP_MS
from .signal import Signal, SHAPES, delay as delay_sig, empty_aux

REQ = object()
COMMON = {"id", "op", "in", "label", "comment"}


class OpError(Exception):
    def __init__(self, node_id, msg):
        super().__init__(f"{node_id}: {msg}")
        self.node_id, self.msg = node_id, msg


class Ctx:
    """per-bake constants: frame grid, tempo, analysis, notes"""

    def __init__(self, fps, frames, offset, tempo, analysis=None, notes=None, seed=1, default_weights=None, t0=0.0,
                 scene=None, safety=None, stems=None, start=0):
        # frame i of every signal is sequence frame start + i; offset is audio_offset_frames - start (may be < 0)
        self.fps, self.frames, self.offset = float(fps), int(frames), int(offset)
        self.start = int(start)                                         # rig bake.start_frame
        self.tempo, self.analysis, self.notes, self.seed = tempo, analysis, notes or {}, int(seed)
        self.default_weights = default_weights
        self.t0 = float(t0 or 0.0)                                      # rig audio_in_s: audio time of frame `offset`
        self.t = (np.arange(self.frames) - self.offset) / self.fps + self.t0   # audio seconds at each frame's start
        self.scene = scene or {}                                        # scene.json dict (positions, camera)
        self.safety = safety or {"max_flashes_per_s": 3.0, "min_swing": 0.1}
        self.stems = stems or {}                                        # stem name -> Analysis of that stem
        self.rig_nodes = []

    def frame_of(self, times):
        """first frame at or after each audio time"""
        return time_to_frame(times, self.fps, self.offset, self.t0)

    def sample(self, env, step_ms):
        return at_frames(env, step_ms, self.frames, self.fps, self.offset, self.t0)

    def ms_frames(self, ms):
        return float(ms) * self.fps / 1000.0

    def beats_frames(self, beats):
        """[N] frames for `beats` beats at each frame's tempo"""
        return float(beats) * self.tempo.beat_seconds(self.t) * self.fps

    def length_frames(self, node, ms_key, beats_key, default_beats=None):
        """[N] frames from <x>_ms or <x>_beats (tempo-following); None if neither is set and no default"""
        if node.get(ms_key) is not None:
            return np.full(self.frames, self.ms_frames(node[ms_key]))
        b = node.get(beats_key, default_beats)
        return None if b is None else self.beats_frames(b)

    def need_analysis(self, node):
        """the analysis a source node reads: the mix, or with "stem": "drums" that stem's analysis"""
        stem = node.get("stem")
        if stem:
            if stem not in self.stems:
                raise OpError(node.get("id"), f"stem {stem!r} is not separated yet: run 'beatrig stems <rig>' (needs the [stems] extra)")
            return self.stems[stem]
        if self.analysis is None:
            raise OpError(node.get("id"), "needs audio analysis (rig has no audio)")
        return self.analysis


class Spec:
    def __init__(self, fn, params, inputs, kind, chans, one_of=(), color=False):
        self.fn, self.params, self.inputs, self.kind, self.chans, self.one_of = fn, params, inputs, kind, chans, one_of
        self.color = color                  # True: accepts colour inputs (passes them through unless its kind says otherwise)


SPECS = {}
STEMS = ["drums", "bass", "vocals", "other"]


def op(name, params=None, inputs="one", kind="value", chans=None, one_of=(), color=False):
    """inputs: None | 'one' | 'many' | 'ref' (onsets: optional band node id or spec, not a signal input).
    color: the op accepts colour signals (ops whose kind is "color" always do)."""
    def deco(fn):
        SPECS[name] = Spec(fn, params or {}, inputs, kind, chans or (lambda n, c: c[0] if c else 1), one_of,
                           color or kind == "color")
        return fn
    return deco


def P(node, key):
    """param with its default"""
    v = node.get(key)
    if v is None:
        d = SPECS[node["op"]].params.get(key)
        return None if d is REQ else d
    return v


def hits_signal(frames, vels, ctx, lens=None, shapes=None):
    """1-channel trigger from hit frames; collisions keep the strongest; frames outside the bake are dropped"""
    N = ctx.frames
    v = np.zeros((1, N), np.float32)
    aux = empty_aux((1, N))
    frames = np.asarray(frames, np.int64)
    vels = np.clip(np.asarray(vels, np.float64), 0, 1)
    order = np.argsort(vels, kind="stable")                 # strongest written last
    for i in order:
        f = int(frames[i])
        if 0 <= f < N and vels[i] > 0:
            v[0, f] = vels[i]
            aux["len"][0, f] = np.nan if lens is None or lens[i] is None else lens[i]
            aux["shape"][0, f] = -1 if shapes is None or shapes[i] is None else SHAPES.index(shapes[i])
    return Signal(v, "trigger", v != 0, aux)


# ---------- sources
def band_weights(node, ctx):
    return node.get("weights")


@op("band", {"lo": REQ, "hi": REQ, "attack_ms": 10, "release_ms": 150, "weights": None, "stem": None}, inputs=None,
    chans=lambda n, c: 1)
def op_band(node, ins, ctx):
    a = ctx.need_analysis(node)
    env = a.envelope("band", P(node, "lo"), P(node, "hi"), P(node, "attack_ms"), P(node, "release_ms"), node.get("weights"))
    return Signal(ctx.sample(env, STEP_MS))


@op("loudness", {"attack_ms": 10, "release_ms": 150, "weights": None, "stem": None}, inputs=None, chans=lambda n, c: 1)
def op_loudness(node, ins, ctx):
    a = ctx.need_analysis(node)
    env = a.envelope("loud", None, None, P(node, "attack_ms"), P(node, "release_ms"), node.get("weights"))
    return Signal(ctx.sample(env, STEP_MS))


def onset_band(node, rig_nodes):
    """(lo, hi, weights, stem) for an onsets node: `in` is a band node id, an inline {lo, hi} spec, or None (full
    range). The stem is the node's own "stem", else the referenced band's."""
    ref = node.get("in")
    if ref is None:
        return None, None, None, node.get("stem")
    if isinstance(ref, dict):
        return ref.get("lo"), ref.get("hi"), ref.get("weights"), node.get("stem") or ref.get("stem")
    src = next((n for n in rig_nodes if n.get("id") == ref), None)
    if src is None or src.get("op") != "band":
        raise OpError(node.get("id"), f"in={ref!r} must be a band node or a {{lo, hi}} spec")
    return src.get("lo"), src.get("hi"), src.get("weights"), node.get("stem") or src.get("stem")


def detect_onsets(node, ctx, rig_nodes):
    """causal spectral-flux peaks -> (times s, velocities). time = (k + 1) * 5 ms, when the rise is known"""
    lo, hi, w, stem = onset_band(node, rig_nodes)
    a = ctx.need_analysis(dict(node, stem=stem))
    f = a.onset_fn(lo, hi, w).astype(np.float64)
    gap = max(1, int(round(float(P(node, "min_gap_ms")) / HOP_MS)))
    pk, _ = sg.find_peaks(f, distance=gap)
    pk = pk[f[pk] > 0]
    if not len(pk):
        return np.zeros(0), np.zeros(0)
    ref = np.percentile(f[pk], 95) + 1e-12
    rel = f[pk] / ref
    keep = rel >= float(P(node, "threshold"))
    return (pk[keep] + 1) * HOP_MS / 1000.0, np.clip(rel[keep], 0.05, 1.0)


@op("onsets", {"threshold": 0.3, "min_gap_ms": 80, "stem": None}, inputs="ref", kind="trigger", chans=lambda n, c: 1)
def op_onsets(node, ins, ctx):
    times, vels = detect_onsets(node, ctx, ctx.rig_nodes)
    return hits_signal(ctx.frame_of(times), vels, ctx)


def beat_numbers(a, ctx, bt):
    """(beat number per beat, meter) on the bar grid: n = bar * meter + pos from the analysis' downbeats when it has
    them (a beat without an onset keeps its number, so counting never slips), else the tempo map's beat count from
    the grid offset (bars of 4)"""
    bars = (getattr(a, "meta", None) or {}).get("bars") or {}
    meter = int(bars.get("meter") or 4)
    if len(a.beats) and all("bar" in b and "pos" in b for b in a.beats):
        return np.array([b["bar"] * meter + b["pos"] for b in a.beats], np.int64), meter
    return np.round(ctx.tempo.beats_at(np.asarray(bt, np.float64))).astype(np.int64), meter


def beat_times(node, ctx):
    """verified beats (analysis), every/offset picking, causal subdivision (spacing from the PREVIOUS interval, only
    up to the next real beat). v0.3: min_confidence drops beats (and their subdivisions) whose analysis confidence is
    below it, after every/offset have counted the full grid; velocity "confidence" uses that confidence as the hit
    velocity instead of the onset strength (subdivisions keep the 0.6 factor).
    anchor "first" (default) counts every/offset along the beat list from its first beat. anchor "bar" counts on the
    bar grid: beat number n (n % meter == 0 on a downbeat; subdivision j of beat n is n * subdivide + j) is kept when
    (n - offset) % every == 0, so every 4 = the downbeats and offset is a phase within the bar, not a skip."""
    a = ctx.need_analysis(node)
    bt = np.array([b["time"] for b in a.beats])
    vel_key = "confidence" if P(node, "velocity") == "confidence" else "strength"
    bv = np.array([b.get(vel_key, 1.0) for b in a.beats])
    bc = np.array([b.get("confidence", 1.0) for b in a.beats])
    every, off, sub = int(P(node, "every")), int(P(node, "offset")), int(P(node, "subdivide"))
    by_bar = P(node, "anchor") == "bar"
    num = beat_numbers(a, ctx, bt)[0] * max(1, sub) if by_bar and len(bt) else np.zeros(len(bt), np.int64)
    times, vels, confs, nums = list(bt), list(bv), list(bc), list(num)
    if sub > 1 and len(bt) > 2:
        T = 60.0 / np.array([ctx.tempo.bpm_at(t) for t in bt])
        for i in range(1, len(bt)):
            prev = bt[i] - bt[i - 1]
            if abs(prev - T[i]) > 0.25 * T[i]:
                continue                                  # gap in the grid: no interval to subdivide from
            nxt = bt[i + 1] if i + 1 < len(bt) else np.inf
            for j in range(1, sub):
                t = bt[i] + j * prev / sub
                if t < nxt - 0.02:
                    times.append(t)
                    vels.append(bv[i] * 0.6)
                    confs.append(bc[i])
                    nums.append(num[i] + j)
        o = np.argsort(times, kind="stable")
        times, vels, confs, nums = np.array(times)[o], np.array(vels)[o], np.array(confs)[o], np.array(nums)[o]
    times, vels, confs, nums = np.asarray(times), np.asarray(vels), np.asarray(confs), np.asarray(nums, np.int64)
    if by_bar and (every > 1 or off):
        sel = (nums - off) % max(1, every) == 0
        times, vels, confs = times[sel], vels[sel], confs[sel]
    elif every > 1 or off:
        sel = np.arange(len(times))
        sel = sel[(sel - off) % every == 0] if every > 1 else sel[off:]
        sel = sel[sel >= off]
        times, vels, confs = times[sel], vels[sel], confs[sel]
    mc = float(P(node, "min_confidence") or 0.0)
    if mc > 0:
        keep = confs >= mc - 1e-9
        times, vels = times[keep], vels[keep]
    return times, vels


@op("beats", {"every": 1, "offset": 0, "subdivide": 1, "min_confidence": 0.0, "velocity": "strength", "anchor": "first"},
    inputs=None, kind="trigger", chans=lambda n, c: 1)
def op_beats(node, ins, ctx):
    times, vels = beat_times(node, ctx)
    return hits_signal(ctx.frame_of(times), vels, ctx)


@op("notes", {"track": REQ}, inputs=None, kind="trigger", chans=lambda n, c: 1)
def op_notes(node, ins, ctx):
    """hits from notes.json; pattern placements are expanded (on the notes' bpm map) so every copy follows its pattern"""
    from .notes import track_hits
    hits = track_hits(ctx.notes, node["track"])
    times = [h["time"] for h in hits]
    return hits_signal(ctx.frame_of(times), [h.get("vel", 1.0) for h in hits], ctx,
                       [h.get("len_beats") for h in hits], [h.get("shape") for h in hits])


def lfo_wave(shape, p):
    p = p - np.floor(p)
    if shape == "sine":
        return 0.5 - 0.5 * np.cos(2 * np.pi * p)
    if shape == "tri":
        return 1 - np.abs(2 * p - 1)
    if shape == "saw_up":
        return p
    if shape == "saw_down":
        return 1 - p
    if shape == "square":
        return (p < 0.5).astype(np.float64)
    if shape == "semi_sine":
        return np.sin(np.pi * p)
    raise ValueError(shape)


LFO_SHAPES = ["sine", "tri", "saw_up", "saw_down", "square", "semi_sine"]


@op("lfo", {"shape": "sine", "rate_beats": None, "rate_hz": None, "phase": 0.0, "channels": 1, "phase_step": 0.0}, inputs=None,
    chans=lambda n, c: int(n.get("channels", 1)))
def op_lfo(node, ins, ctx):
    """rate_beats = beats per cycle, locked to the tempo map and grid offset; rate_hz = cycles per second"""
    if node.get("rate_hz") is not None:
        cyc = float(node["rate_hz"]) * ctx.t
    else:
        cyc = ctx.tempo.beats_at(ctx.t) / float(node.get("rate_beats") or 1.0)
    C = int(P(node, "channels"))
    ph = float(P(node, "phase")) + float(P(node, "phase_step")) * np.arange(C)[:, None]
    return Signal(lfo_wave(P(node, "shape"), cyc[None, :] + ph))


@op("const", {"value": 0.0, "channels": 1}, inputs=None, chans=lambda n, c: int(n.get("channels", 1)))
def op_const(node, ins, ctx):
    return Signal(np.full((int(P(node, "channels")), ctx.frames), float(P(node, "value"))))


# ---------- shaping
@op("normalize", {"range_db": 24, "hi_pct": 99.5, "lo_pct": 10})
def op_normalize(node, ins, ctx):
    s = ins[0]
    v, _, _ = norm_db(s.v.ravel(), float(P(node, "range_db")), float(P(node, "hi_pct")), float(P(node, "lo_pct")))
    return Signal(v.reshape(s.v.shape), "value", s.hits)


def follow_rows(v, attack_ms, release_ms, dt_ms):
    return np.stack([follow(row, attack_ms, release_ms, dt_ms) for row in v])


@op("smooth", {"attack_ms": REQ, "release_ms": REQ}, kind=lambda n, k: "color" if k and k[0] == "color" else "value", color=True)
def op_smooth(node, ins, ctx):
    """one-pole follower per channel; a colour signal is smoothed per RGBA component (eased colour changes)"""
    s = ins[0]
    a, r, dt = float(node["attack_ms"]), float(node["release_ms"]), 1000.0 / ctx.fps
    if s.kind == "color":
        C, N, K = s.v.shape
        rows = s.v.transpose(0, 2, 1).reshape(C * K, N)
        return Signal(follow_rows(rows, a, r, dt).reshape(C, K, N).transpose(0, 2, 1), "color", s.hits)
    return Signal(follow_rows(s.v, a, r, dt), "value", s.hits)


@op("spring", {"freq_hz": 2.0, "damping": 0.6})
def op_spring(node, ins, ctx):
    """2nd-order follower (w^2 / (s^2 + 2 zeta w s + w^2)), bilinear-discretised, lfilter: causal, starts at rest"""
    s = ins[0]
    w = 2 * np.pi * float(P(node, "freq_hz"))
    b, a = sg.bilinear([w * w], [1.0, 2 * float(P(node, "damping")) * w, w * w], fs=ctx.fps)
    return Signal(sg.lfilter(b, a, s.v.astype(np.float64), axis=1), "value", s.hits)


CURVE_PRESETS = {"s_curve": lambda x: x * x * (3 - 2 * x), "ease_in": lambda x: x * x, "ease_out": lambda x: 1 - (1 - x) ** 2}


@op("curve", {"gamma": 1.0, "preset": None})
def op_curve(node, ins, ctx):
    s = ins[0]
    x = np.clip(s.v.astype(np.float64), 0, None)
    if node.get("preset"):
        y = CURVE_PRESETS[node["preset"]](np.clip(x, 0, 1))
    else:
        y = x ** float(P(node, "gamma"))
    return Signal(y, "value", s.hits)


@op("remap", {"in_lo": REQ, "in_hi": REQ, "out_lo": REQ, "out_hi": REQ, "clamp": True})
def op_remap(node, ins, ctx):
    s = ins[0]
    a, b, c, d = (float(node[k]) for k in ("in_lo", "in_hi", "out_lo", "out_hi"))
    y = (s.v.astype(np.float64) - a) / ((b - a) or 1e-12) * (d - c) + c
    if P(node, "clamp"):
        y = np.clip(y, min(c, d), max(c, d))
    return Signal(y, "value", s.hits)


@op("threshold", {"level": 0.5, "cooldown_ms": 100}, kind="trigger")
def op_threshold(node, ins, ctx):
    """rising edge through `level` (reads this frame and the previous one only); velocity 1"""
    s = ins[0]
    lvl, cool = float(P(node, "level")), ctx.ms_frames(P(node, "cooldown_ms"))
    v = np.zeros_like(s.v)
    for c in range(s.channels):
        x = s.v[c]
        prev = np.concatenate(([-np.inf], x[:-1]))
        cand = np.flatnonzero((x >= lvl) & (prev < lvl))
        last = -np.inf
        for f in cand.tolist():
            if f - last >= cool:
                v[c, f] = 1.0
                last = f
    return Signal(v, "trigger", v != 0, empty_aux(v.shape))


def env_shape(name, k, L, node, fps, points=None):
    """shape value at k frames after the hit for a hit of length L frames (velocity 1)"""
    t = k / max(L, 1e-9)
    if name == "binary":
        y = (t < 1).astype(np.float64)
    elif name == "punch":
        y = np.where(t < 1, (np.exp(-5 * t) - np.exp(-5)) / (1 - np.exp(-5)), 0.0)
    elif name == "gloss":
        y = np.where(t < 1, 0.5 * (1 + np.cos(np.pi * np.clip(t, 0, 1))), 0.0)
    elif name == "razor":
        y = np.where(t < 1, (1 - np.clip(t, 0, 1)) ** 4, 0.0)
    elif name == "rush":
        y = np.where(t < 0.8, (t / 0.8) ** 2, np.where(t < 1, 0.5 * (1 + np.cos(np.pi * (t - 0.8) / 0.2)), 0.0))
    elif name == "saw_down":
        y = np.where(t < 1, 1 - t, 0.0)
    elif name == "custom":
        pts = np.asarray(points or [[0, 1], [1, 0]], np.float64)
        y = np.where(t <= 1, np.interp(t, pts[:, 0], pts[:, 1]), 0.0)
    elif name == "adsr":
        ms = k * 1000.0 / fps
        gate = L * 1000.0 / fps
        A = float(node.get("attack_ms", 0) or 0)
        D = float(node.get("decay_ms", 100) if node.get("decay_ms") is not None else 100)
        S = float(node.get("sustain", 0.7) if node.get("sustain") is not None else 0.7)
        R = float(node.get("release_ms", 200) if node.get("release_ms") is not None else 200)
        def ads(m):
            m = np.asarray(m, np.float64)
            att = np.where(A > 0, np.clip(m / (A or 1), 0, 1), 1.0)
            dec = np.where(m < A, 1.0, 1 - (1 - S) * np.clip((m - A) / (D or 1e-9), 0, 1))
            return np.where(m < A, att, dec)
        g = float(ads(gate))
        y = np.where(ms < gate, ads(ms), np.where(ms < gate + R, g * (1 - (ms - gate) / (R or 1e-9)), 0.0))
        return y
    else:
        raise ValueError(name)
    A = float(node.get("attack_ms", 0) or 0)
    if A > 0:                                           # optional eased attack for any shape
        r = np.clip(k * 1000.0 / fps / A, 0, 1)
        y = y * r * r * (3 - 2 * r)
    return y


@op("envelope", {"shape": "punch", "length_beats": None, "length_ms": None, "attack_ms": 0, "decay_ms": None, "sustain": None,
                 "release_ms": None, "points": None})
def op_envelope(node, ins, ctx):
    """trigger -> shaped hits, velocity-scaled, overlapping hits take the max. Per-hit len_beats/shape (notes) win."""
    s = ins[0]
    N = ctx.frames
    base_len = ctx.length_frames(node, "length_ms", "length_beats", 0.5)
    out = np.zeros(s.v.shape, np.float64)
    lens = s.aux.get("len")
    shapes = s.aux.get("shape")
    tail = 0.0
    if (node.get("shape") or "punch") == "adsr" or (shapes is not None and np.any(shapes == SHAPES.index("adsr"))):
        tail = ctx.ms_frames(node.get("release_ms") if node.get("release_ms") is not None else 200)
    for c in range(s.channels):
        for h in np.flatnonzero(s.v[c] != 0).tolist():
            vel = float(s.v[c, h])
            L = float(base_len[h])
            if lens is not None and not np.isnan(lens[c, h]):
                L = float(lens[c, h]) * ctx.tempo.beat_seconds(ctx.t[h]) * ctx.fps
            name = P(node, "shape")
            if shapes is not None and shapes[c, h] >= 0:
                name = SHAPES[int(shapes[c, h])]
            n = min(N - h, int(np.ceil(L + tail)) + 1)
            k = np.arange(n, dtype=np.float64)
            y = vel * env_shape(name, k, L, node, ctx.fps, node.get("points"))
            np.maximum(out[c, h:h + n], y, out=out[c, h:h + n])
    return Signal(out, "value", s.hits)


def per_channel_delay(node, ctx, C, base_ms="ms", base_beats="beats", step_ms="step_ms", step_beats="step_beats", idx=None):
    base = ctx.length_frames(node, base_ms, base_beats)
    step = ctx.length_frames(node, step_ms, step_beats)
    d = np.zeros((C, ctx.frames))
    if base is not None:
        d += base[None, :]
    if step is not None:
        idx = np.arange(C) if idx is None else np.asarray(idx)
        d += idx[:, None] * step[None, :]
    return d


@op("delay", {"ms": None, "beats": None, "step_ms": None, "step_beats": None}, one_of=(("ms", "beats"),),
    kind=lambda n, k: k[0] if k else "value", color=True)
def op_delay(node, ins, ctx):
    s = ins[0]
    return delay_sig(s, per_channel_delay(node, ctx, s.channels))


MATH = ["add", "mul", "max", "min", "sub", "mix"]


def math_chans(node, c):
    n = max(c) if c else 1
    return n


@op("math", {"fn": REQ, "amount": 0.5}, inputs="many", chans=math_chans,
    kind=lambda n, k: "trigger" if k and all(x == "trigger" for x in k) and n.get("fn") in ("max", "min", "add", "mul") else "value")
def op_math(node, ins, ctx):
    fn = node["fn"]
    n = max(s.channels for s in ins)
    try:
        sigs = [s.broadcast(n) for s in ins]
    except ValueError as e:
        raise OpError(node.get("id"), str(e)) from None
    vs = [s.v.astype(np.float64) for s in sigs]
    if fn in ("sub", "mix") and len(vs) != 2:
        raise OpError(node.get("id"), f"{fn} takes exactly 2 inputs")
    if fn == "add":
        y = np.sum(vs, axis=0)
    elif fn == "mul":
        y = np.prod(vs, axis=0)
    elif fn == "max":
        y = np.max(vs, axis=0)
    elif fn == "min":
        y = np.min(vs, axis=0)
    elif fn == "sub":
        y = vs[0] - vs[1]
    elif fn == "mix":
        a = float(P(node, "amount"))
        y = vs[0] * (1 - a) + vs[1] * a
    else:
        raise OpError(node.get("id"), f"unknown fn {fn!r}")
    hits = np.any([s.hits for s in sigs], axis=0)
    kind = SPECS["math"].kind(node, [s.kind for s in ins])
    return Signal(y, kind, hits, empty_aux(y.shape) if kind == "trigger" else None)


@op("hold", {"ms": REQ})
def op_hold(node, ins, ctx):
    """peak hold: max over the last `ms` (this frame and earlier ones only)"""
    s = ins[0]
    h = int(np.ceil(ctx.ms_frames(node["ms"]) - 1e-9))
    y = s.v.astype(np.float64).copy()
    for k in range(1, h + 1):
        np.maximum(y[:, k:], s.v[:, :-k], out=y[:, k:])
    return Signal(y, "value", s.hits)


# ---------- channels
def same_kind(n, k):
    return k[0] if k else "value"


@op("fanout", {"channels": REQ}, kind=same_kind, chans=lambda n, c: int(n["channels"]), color=True)
def op_fanout(node, ins, ctx):
    s = ins[0]
    if s.channels != 1:
        raise OpError(node.get("id"), f"fanout needs a 1-channel input, got {s.channels}")
    return s.broadcast(int(node["channels"]))


def chase_index(C, direction, wrap):
    """delay multiplier per channel: direction 1 -> 0, 1, .., C-1. direction -1 -> wrap: 0, C-1, .., 1 (channel 0
    still leads, the chase runs the other way round the ring); no wrap: C-1, .., 0 (starts at the last channel)"""
    i = np.arange(C)
    if int(direction) >= 0:
        return i
    return (C - i) % C if wrap else C - 1 - i


@op("chase", {"channels": REQ, "step_ms": None, "step_beats": None, "direction": 1, "wrap": True}, kind=same_kind,
    chans=lambda n, c: int(n["channels"]), one_of=(("step_ms", "step_beats"),), color=True)
def op_chase(node, ins, ctx):
    """channel i = in delayed by i * step"""
    s = ins[0]
    C = int(node["channels"])
    if s.channels not in (1, C):
        raise OpError(node.get("id"), f"chase input has {s.channels} channels, expected 1 or {C}")
    idx = chase_index(C, P(node, "direction"), P(node, "wrap"))
    return delay_sig(s.broadcast(C), per_channel_delay(node, ctx, C, idx=idx))


CYCLE_ORDERS = ["sequential", "pingpong", "random"]


@op("cycle", {"channels": REQ, "order": "sequential", "seed": None}, kind="trigger", chans=lambda n, c: int(n["channels"]))
def op_cycle(node, ins, ctx):
    """each hit goes to the next channel. random: seeded (node seed, else rig seed), never the same channel twice running"""
    s = ins[0]
    C = int(node["channels"])
    src = s if s.channels == 1 else Signal(s.v.max(axis=0), s.kind, s.hits.any(axis=0))
    frames = np.flatnonzero(src.v[0] != 0)
    order = P(node, "order")
    if order == "sequential":
        ch = np.arange(len(frames)) % C
    elif order == "pingpong":
        period = max(1, 2 * C - 2)
        k = np.arange(len(frames)) % period
        ch = np.where(k < C, k, period - k)
    elif order == "random":
        rng = np.random.default_rng(int(node["seed"]) if node.get("seed") is not None else ctx.seed)
        ch, last = [], -1
        for _ in range(len(frames)):
            c = int(rng.integers(C - 1 if C > 1 and last >= 0 else C))
            if C > 1 and last >= 0 and c >= last:
                c += 1
            ch.append(c)
            last = c
        ch = np.array(ch, np.int64)
    else:
        raise OpError(node.get("id"), f"unknown order {order!r}")
    v = np.zeros((C, ctx.frames), np.float32)
    aux = empty_aux(v.shape)
    for f, c in zip(frames.tolist(), np.asarray(ch).tolist()):
        v[c, f] = src.v[0, f]
        if s.channels == 1:
            for k in aux:
                if k in s.aux:
                    aux[k][c, f] = s.aux[k][0, f]
    return Signal(v, "trigger", v != 0, aux)


@op("ripple", {"neighbours": 2, "falloff": 0.5, "step_ms": 40, "ring": True, "k": 3, "hops": 2, "actors": None}, kind=same_kind)
def op_ripple(node, ins, ctx):
    """each channel's signal spreads to neighbours at distance d, delayed d*step and scaled falloff^d; max-combined.
    neighbours = N: channel index distance on a ring/line. neighbours = "knn": graph distance (hops) in the graph
    joining each actor to its k nearest actors by real position (scene.json); channel i = actors[i]."""
    s = ins[0]
    C = s.channels
    fo, step = float(P(node, "falloff")), ctx.ms_frames(P(node, "step_ms"))
    if P(node, "neighbours") == "knn":
        from .spatial import actor_positions, knn_hops
        pos = actor_positions(ctx, node)
        if len(pos) != C:
            raise OpError(node.get("id"), f"ripple knn: input has {C} channels but {len(pos)} actors")
        hop = knn_hops(pos, int(P(node, "k")), int(P(node, "hops")))
        out = s.v.astype(np.float64).copy()
        hits = s.hits.copy()
        for d in range(1, int(P(node, "hops")) + 1):
            mask = hop == d
            if not mask.any():
                continue
            dl = delay_sig(s, d * step)
            for j in range(C):
                src = np.flatnonzero(mask[j])
                if len(src):
                    np.maximum(out[j], dl.v[src].max(axis=0) * fo ** d, out=out[j])
                    hits[j] |= dl.hits[src].any(axis=0)
        return Signal(out, s.kind, hits, empty_aux(out.shape) if s.kind == "trigger" else None)
    nb, ring = int(P(node, "neighbours")), bool(P(node, "ring"))
    out = s.v.astype(np.float64).copy()
    hits = s.hits.copy()
    for d in range(1, nb + 1):
        dl = delay_sig(s, d * step)
        for sign in (1, -1):
            for j in range(C):
                i = j - sign * d
                if ring:
                    i %= C
                elif not 0 <= i < C:
                    continue
                if ring and 2 * d == C and sign == -1:
                    continue                                  # opposite channel on an even ring: count once
                np.maximum(out[j], dl.v[i] * fo ** d, out=out[j])
                hits[j] |= dl.hits[i]
    return Signal(out, s.kind, hits, empty_aux(out.shape) if s.kind == "trigger" else None)


@op("mirror", {}, kind=same_kind, color=True)
def op_mirror(node, ins, ctx):
    """symmetric pattern: out[i] = in[min(i, C-1-i)]"""
    C = ins[0].channels
    return ins[0].take([min(i, C - 1 - i) for i in range(C)])


@op("reverse", {}, kind=same_kind, color=True)
def op_reverse(node, ins, ctx):
    return ins[0].take(list(range(ins[0].channels))[::-1])


@op("select", {"channels": REQ}, kind=same_kind, chans=lambda n, c: len(n.get("channels") or []), color=True)
def op_select(node, ins, ctx):
    idx = [int(i) for i in node["channels"]]
    if any(not 0 <= i < ins[0].channels for i in idx):
        raise OpError(node.get("id"), f"select {idx} out of range for {ins[0].channels} channels")
    return ins[0].take(idx)


@op("merge", {"fn": "max"}, kind=lambda n, k: k[0] if k and n.get("fn", "max") == "max" else "value", chans=lambda n, c: 1)
def op_merge(node, ins, ctx):
    s = ins[0]
    fn = P(node, "fn")
    y = s.v.max(axis=0) if fn == "max" else s.v.mean(axis=0)
    kind = s.kind if fn == "max" else "value"
    return Signal(y, kind, s.hits.any(axis=0)[None, :], empty_aux((1, ctx.frames)) if kind == "trigger" else None)


@op("shuffle", {"seed": None}, kind=same_kind, color=True)
def op_shuffle(node, ins, ctx):
    """fixed channel permutation (node seed, else rig seed)"""
    rng = np.random.default_rng(int(node["seed"]) if node.get("seed") is not None else ctx.seed)
    return ins[0].take(rng.permutation(ins[0].channels).tolist())


@op("accumulate", {"per_s": None, "per_beat": None, "start": 0.0, "wrap": None}, one_of=(("per_s", "per_beat"),))
def op_accumulate(node, ins, ctx):
    """running sum (an integral): out[f] = start + the sum over frames before f of in x rate x frame time. A constant
    1 with per_s 0.5 turns half a unit per second; per_beat follows the tempo map. Causal by construction; the output
    is continuous even when the input jumps, so a trigger envelope becomes an eased step (spin bursts, phase pushes).
    wrap: fold into [0, wrap) (an instant jump at each wrap; leave it off for transform outputs)."""
    s = ins[0]
    if node.get("per_beat") is not None:
        inc = float(node["per_beat"]) * ctx.tempo.bpm_at(ctx.t) / 60.0 / ctx.fps
    else:
        inc = np.full(ctx.frames, float(node["per_s"] if node.get("per_s") is not None else 1.0) / ctx.fps)
    x = s.v.astype(np.float64) * inc[None, :]
    y = float(P(node, "start")) + np.concatenate((np.zeros((s.channels, 1)), np.cumsum(x, axis=1)[:, :-1]), axis=1)
    if node.get("wrap"):
        y = np.mod(y, float(node["wrap"]))
    return Signal(y, "value", s.hits)


def input_ids(node):
    """signal inputs of a node (onsets' `in` is a band reference, not a signal)"""
    spec = SPECS.get(node.get("op"))
    ref = node.get("in")
    if spec is None or spec.inputs is None or ref is None:
        return []
    if spec.inputs == "ref":
        return [ref] if isinstance(ref, str) else []
    if isinstance(ref, list):
        return list(ref)
    return [ref]


# the colour, per-hit flow, spatial and safety families register their ops on import
from . import color, flow, spatial, safety  # noqa: E402,F401
