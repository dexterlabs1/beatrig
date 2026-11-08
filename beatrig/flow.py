"""Per-hit flow ops: triggers in, triggers out, deterministic. They decide, hit by hit, where (which channel) and
whether (every Nth, by chance) a hit goes, or move it later onto the beat grid. Nothing is ever moved earlier.
Per-hit metadata (len_beats/shape from notes) travels with each hit."""
import zlib

import numpy as np

from .notes import parse_grid
from .ops import OpError, P, REQ, op
from .signal import Signal, empty_aux

PUSH_PATTERNS = ["alternate", "sequential", "pingpong", "outside_in", "inside_out", "halves"]
SHUFFLE_MODES = ["bag", "random"]


def _merged(s):
    """1-channel view of a trigger (several channels merge, strongest wins; per-hit aux only from a 1-channel input)"""
    if s.channels == 1:
        return s.v[0], s.aux, True
    return s.v.max(axis=0), {}, False


def _route(s, frames, chans_per_hit, C, frames_total):
    """hits (frames from the merged input) -> a C-channel trigger; chans_per_hit: list of channel lists"""
    src, aux_in, one = _merged(s)
    v = np.zeros((C, frames_total), np.float32)
    aux = empty_aux(v.shape)
    for f, chs in zip(frames, chans_per_hit):
        for c in chs:
            v[c, f] = src[f]
            if one:
                for k in aux:
                    if k in aux_in:
                        aux[k][c, f] = aux_in[k][0, f]
    return Signal(v, "trigger", v != 0, aux)


def push_steps(pattern, C):
    """a pattern name or an explicit list of steps (each an int or a list of ints) -> list of channel lists"""
    if isinstance(pattern, list):
        steps = [[int(x)] if isinstance(x, (int, float)) else [int(y) for y in x] for x in pattern]
        if not steps or any(not st or any(not 0 <= c < C for c in st) for st in steps):
            raise ValueError(f"pattern {pattern} needs non-empty steps of channels 0..{C - 1}")
        return steps
    i = list(range(C))
    if pattern == "alternate":
        return [i[0::2], i[1::2]] if C > 1 else [i]
    if pattern == "sequential":
        return [[c] for c in i]
    if pattern == "pingpong":
        return [[c] for c in i + i[-2:0:-1]]
    if pattern in ("outside_in", "inside_out"):
        steps = [sorted({c, C - 1 - c}) for c in range((C + 1) // 2)]
        return steps if pattern == "outside_in" else steps[::-1]
    if pattern == "halves":
        return [i[:(C + 1) // 2], i[(C + 1) // 2:]] if C > 1 else [i]
    raise ValueError(f"unknown pattern {pattern!r} (one of {PUSH_PATTERNS} or a list of steps)")


@op("push", {"channels": REQ, "pattern": "alternate", "start": 0}, kind="trigger", chans=lambda n, c: int(n["channels"]))
def op_push(node, ins, ctx):
    """each hit lights the next step of a pattern: "alternate" (even channels, then odd), "sequential", "pingpong",
    "outside_in", "inside_out", "halves", or an explicit list of steps like [[0, 4], [2, 6], [1, 5], [3, 7]]"""
    C = int(node["channels"])
    steps = push_steps(P(node, "pattern"), C)
    src, _, _ = _merged(ins[0])
    frames = np.flatnonzero(src != 0).tolist()
    k0 = int(P(node, "start"))
    return _route(ins[0], frames, [steps[(k0 + k) % len(steps)] for k in range(len(frames))], C, ctx.frames)


def node_seed(node, ctx):
    return int(node["seed"]) if node.get("seed") is not None else ctx.seed


@op("shuffle_hits", {"channels": REQ, "mode": "bag", "count": 1, "seed": None}, kind="trigger", chans=lambda n, c: int(n["channels"]))
def op_shuffle_hits(node, ins, ctx):
    """each hit goes to random channel(s). "bag": every channel once per round in a fresh random order (no channel
    twice in a row across rounds); "random": `count` distinct channels drawn independently per hit. Seeded."""
    C = int(node["channels"])
    count = max(1, min(C, int(P(node, "count"))))
    rng = np.random.default_rng(node_seed(node, ctx))
    src, _, _ = _merged(ins[0])
    frames = np.flatnonzero(src != 0).tolist()
    mode = P(node, "mode")
    picks = []
    if mode == "bag":
        bag, last = [], -1
        for _ in frames:
            chs = []
            while len(chs) < count:
                if not bag:
                    bag = rng.permutation(C).tolist()
                    if C > 1 and bag[0] == last:
                        bag[0], bag[1] = bag[1], bag[0]
                c = bag.pop(0)
                if c not in chs:
                    chs.append(c)
            last = chs[-1]
            picks.append(sorted(chs))
    elif mode == "random":
        picks = [sorted(rng.choice(C, count, replace=False).tolist()) for _ in frames]
    else:
        raise OpError(node.get("id"), f"unknown mode {mode!r} (one of {SHUFFLE_MODES})")
    return _route(ins[0], frames, picks, C, ctx.frames)


@op("every_n", {"n": REQ, "offset": 0}, kind="trigger")
def op_every_n(node, ins, ctx):
    """per channel: pass hit number offset, offset + n, offset + 2n, ... (hits counted from 0); drop the rest"""
    s = ins[0]
    n, off = int(node["n"]), int(P(node, "offset"))
    if n < 1 or off < 0:
        raise OpError(node.get("id"), "n must be >= 1 and offset >= 0")
    keep = np.zeros(s.v.shape, bool)
    for c in range(s.channels):
        idx = np.flatnonzero(s.v[c] != 0)
        i = np.arange(len(idx))
        keep[c, idx[(i >= off) & ((i - off) % n == 0)]] = True
    return _mask(s, keep)


def _mask(s, keep):
    v = np.where(keep, s.v, 0).astype(np.float32)
    aux = {k: np.where(keep, a, np.nan if k == "len" else -1).astype(a.dtype) for k, a in s.aux.items()}
    return Signal(v, "trigger", v != 0, aux or empty_aux(v.shape))


def _mix64(x):
    x = x.astype(np.uint64)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def unit_random(key, channel, frames):
    """deterministic uniform [0, 1) per (key, channel, frame): a splitmix64 hash, so one hit's draw never depends on
    the other hits (editing a note re-rolls nothing else)"""
    with np.errstate(over="ignore"):
        h = _mix64(np.full(len(frames), key, np.uint64) + np.uint64(0x9E3779B97F4A7C15))
        h = _mix64(h ^ np.uint64(channel))
        h = _mix64(h ^ np.asarray(frames, np.uint64))
    return (h >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


@op("probability", {"p": REQ, "seed": None}, kind="trigger")
def op_probability(node, ins, ctx):
    """pass each hit with probability p. The draw is keyed by (seed, channel, frame), or with no node seed by
    (rig seed, node id, channel, frame), so it is stable when other hits change"""
    s = ins[0]
    p = float(node["p"])
    if not 0 <= p <= 1:
        raise OpError(node.get("id"), "p must be within 0..1")
    key = int(node["seed"]) if node.get("seed") is not None else (ctx.seed * 0x100000001 + zlib.crc32(str(node.get("id")).encode())) % 2**63
    keep = np.zeros(s.v.shape, bool)
    for c in range(s.channels):
        idx = np.flatnonzero(s.v[c] != 0)
        if len(idx):
            keep[c, idx[unit_random(key, c, idx + ctx.start) < p]] = True      # keyed by sequence frame
    return _mask(s, keep)


@op("quantize", {"grid": "1/16", "strength": 1.0}, kind="trigger")
def op_quantize(node, ins, ctx):
    """delay each hit to the next grid line at or after it (grid "1/16" etc., on the beat grid of the tempo map;
    strength < 1 moves part of the way). A hit only ever moves later, so this is causal. A hit already on a grid line
    (inside the same frame) stays put. Colliding hits keep the strongest."""
    s = ins[0]
    g = parse_grid(P(node, "grid"))
    strength = float(P(node, "strength"))
    if g <= 0 or not 0 <= strength <= 1:
        raise OpError(node.get("id"), "grid must be > 0 and strength within 0..1")
    C, N = s.v.shape
    v = np.zeros_like(s.v)
    aux = {k: np.full_like(a, np.nan if k == "len" else -1) for k, a in s.aux.items()} or empty_aux(v.shape)
    hits = np.zeros(s.v.shape, bool)
    for c in range(C):
        src = np.flatnonzero(s.v[c] != 0)
        if not len(src):
            continue
        t_prev = ctx.t[src] - 1.0 / ctx.fps                  # the hit's onset is after the previous frame's start
        k = np.floor(ctx.tempo.beats_at(t_prev) / g + 1e-9) + 1
        fq = np.maximum(ctx.frame_of(ctx.tempo.time_of_beat(k * g)), src)
        dst = src + np.ceil(strength * (fq - src) - 1e-9).astype(np.int64)
        ok = dst < N
        src, dst = src[ok], dst[ok]
        order = np.argsort(s.v[c, src], kind="stable")       # strongest written last
        src, dst = src[order], dst[order]
        v[c, dst] = s.v[c, src]
        hits[c, dst] = True
        for key, a in s.aux.items():
            aux[key][c, dst] = a[c, src]
    return Signal(v, "trigger", hits, aux)
