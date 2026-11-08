"""Drum parts: kick, snare (or clap) and hats hits, from the separated drums stem when there is one, else from the mix.

Plain band onsets on a drum mix confuse the parts: a kick's click lands in the snare and hats bands on every beat, a
clap's bright tail lands in the hats band, and a separated stem smears a little of each hit up to ~100 ms ahead of
it. Each part starts from its band's causal onset peaks (the onsets op's onset function) and then:
- every part drops pre-echo: a peak with 10x its power in the same band within the next 120 ms;
- snare (1-5 kHz) keeps a hit only if the band still holds 2 % of the hit's peak power 40-80 ms later (a snare or
  clap rings on; a kick's click is gone in a few ms);
- hats (6-16 kHz) keeps a hit only if 1 % of its power is still there 10-20 ms later (not a click), and drops one on
  a snare hit (within 25 ms) unless its rise is twice what the song's snares put in the band. Hats that always land
  on the snare are therefore left to the snare track;
- then the strongest hits min_gap_ms apart, and the threshold relative to the part's own 95th percentile.
These tests look up to 80 ms past the onset: an offline choice, like flash_limit "strongest". A kept hit still lands
on its own onset frame (the first frame at or after the rise is known), never earlier.
"""
import numpy as np

from .analysis import HOP_MS, sg
from .ops import OpError, P, REQ, hits_signal, op

PARTS = {"kick": {"lo": 35.0, "hi": 120.0, "min_gap_ms": 110},
         "snare": {"lo": 1000.0, "hi": 5000.0, "min_gap_ms": 90},
         "hats": {"lo": 6000.0, "hi": 16000.0, "min_gap_ms": 55}}
SOURCES = ["auto", "stem", "mix"]
PRE_ECHO_POWER, PRE_ECHO_MS = 10.0, 120  # a peak with 10x its power within 120 ms after it is that hit's smear
SUSTAIN_MS, SUSTAIN_MIN = (40, 80), 0.02  # snare: 2 % of the peak power still there 40-80 ms on
CLICK_MS, CLICK_MIN = (10, 20), 0.01      # hats: 1 % still there 10-20 ms on (a kick's click is gone)
CLAP_LEAK_MS, CLAP_LEAK_MULT = 25, 2.0


def part_bands():
    """{part: (lo, hi)}: the bands a drum op needs (analysis computes them in one pass)"""
    return {k: (v["lo"], v["hi"]) for k, v in PARTS.items()}


def drum_analysis(node, ctx):
    """(the analysis a drum node reads, "stem" | "mix"): source "auto" reads the drums stem when the analysis has it"""
    src = P(node, "source")
    if src == "mix":
        return ctx.need_analysis({"id": node.get("id")}), "mix"
    if "drums" in ctx.stems:
        return ctx.stems["drums"], "stem"
    if src == "stem":
        raise OpError(node.get("id"), "source \"stem\" needs the drums stem: run 'beatrig stems <rig>' (needs the [stems] extra)")
    return ctx.need_analysis({"id": node.get("id")}), "mix"


def _power5(a, lo, hi):
    """a band's power on the 5 ms grid (index k = the block ending at (k + 1) * 5 ms)"""
    key, _, _ = a._spec("band", lo, hi, None)
    a.ensure([("band", lo, hi, None)])
    p1 = a.arrays["pow:" + key].astype(np.float64)
    n5 = len(p1) // 5
    return p1[:n5 * 5].reshape(n5, 5).mean(axis=1)


def _rise5(p):
    """linear power rise over 20 ms (10 ms smoothing), on the same grid as the onset function"""
    q = np.convolve(p, np.ones(2) / 2)[:len(p)]
    return np.concatenate((np.zeros(min(4, len(q))), np.maximum(q[4:] - q[:-4], 0)))


def _wmax(r, h=2):
    """r's max over each index's +-h neighbours (an index's peak power, robust to the peak's exact block)"""
    pad = np.concatenate((np.full(h, -np.inf), r, np.full(h, -np.inf)))
    return np.lib.stride_tricks.sliding_window_view(pad, 2 * h + 1).max(axis=1)


def _peaks(f, p):
    """onset peaks of a band without pre-echo. The test runs on power, not on the onset function: the onset
    function measures a rise in dB, and a quiet smear rising out of silence can out-rise the hit that follows it."""
    pk, _ = sg.find_peaks(f)
    pk = pk[f[pk] > 0]
    if not len(pk):
        return pk
    pw = _wmax(p)[pk]
    reach = PRE_ECHO_MS // HOP_MS
    q = np.zeros(len(f) + reach + 1)
    q[pk] = pw
    later = np.lib.stride_tricks.sliding_window_view(q[1:], reach).max(axis=1)   # max over the next `reach` blocks
    return pk[later[pk] < PRE_ECHO_POWER * pw]


def _apart(pk, f, gap_ms):
    """the strongest peaks at least gap_ms apart (ties: the earlier), as the onsets op's min_gap_ms"""
    gap = max(1, int(round(float(gap_ms) / HOP_MS)))
    if not len(pk):
        return np.asarray(pk, np.int64)
    taken = np.zeros(len(f) + 2 * gap, bool)                 # index k + gap = frame k
    kept = []
    for k in pk[np.lexsort((pk, -f[pk]))].tolist():         # strongest first; ties: the earlier
        if not taken[k + 1:k + 2 * gap].any():               # no kept peak closer than gap
            taken[k + gap] = True
            kept.append(k)
    return np.sort(np.asarray(kept, np.int64))


def _ratio_after(p, pk, a_ms, b_ms):
    """mean power a_ms..b_ms after each peak / the peak's power (0 where that span runs past the end)"""
    a0, a1 = a_ms // HOP_MS, b_ms // HOP_MS
    cs = np.concatenate(([0.0], np.cumsum(p)))
    ok = pk + a1 <= len(p)
    out = np.zeros(len(pk))
    k = pk[ok]
    out[ok] = (cs[k + a1] - cs[k + a0]) / (a1 - a0) / (_wmax(p)[k] + 1e-20)
    return out


def _off_snares(a, pk, p, sn):
    """hats peaks that are not a snare's bright tail: one within 25 ms of a snare hit stays only when its 6-16 kHz
    rise is twice the song's median for snare hits"""
    if not len(sn) or not len(pk):
        return pk
    rh = _wmax(_rise5(p))
    rs = _wmax(_rise5(_power5(a, PARTS["snare"]["lo"], PARTS["snare"]["hi"])))
    med = float(np.median(rh[sn] / (rs[sn] + 1e-20)))
    j = np.clip(np.searchsorted(sn, pk), 1, len(sn) - 1) if len(sn) > 1 else np.zeros(len(pk), int)
    dist = np.minimum(np.abs(sn[j] - pk), np.abs(sn[np.maximum(j - 1, 0)] - pk))
    on_snare = dist <= CLAP_LEAK_MS // HOP_MS
    return pk[~on_snare | (rh[pk] / (rs[pk] + 1e-20) >= CLAP_LEAK_MULT * med)]


def part_hits(a, part, threshold=0.3, min_gap_ms=None):
    """(peak indices on the 5 ms grid, velocities) of one drum part in analysis a. Order: peaks without pre-echo, the
    part's own test (snare rings on, hats are not a click), the strongest min_gap_ms apart, hats off the snares, and
    last the threshold, relative to the 95th percentile of the part's own peaks (so a loud kick in the hats band does
    not set the hats' scale), with a floor at a third of it relative to every peak in the band."""
    spec = PARTS[part]
    gap = spec["min_gap_ms"] if min_gap_ms is None else min_gap_ms
    memo = a.__dict__.setdefault("_drum_parts", {})          # hats reuse the snare hits; the structure reads all three
    key = (part, float(threshold), float(gap))
    if key not in memo:
        memo[key] = _part_hits(a, part, spec, float(threshold), float(gap))
    return memo[key]


def _part_hits(a, part, spec, threshold, gap):
    p = _power5(a, spec["lo"], spec["hi"])
    f = a.onset_fn(spec["lo"], spec["hi"]).astype(np.float64)
    pk = _peaks(f, p)
    if not len(pk):
        return pk, np.zeros(0)
    band95 = float(np.percentile(f[_apart(pk, f, gap)], 95))
    if part == "snare":
        pk = pk[_ratio_after(p, pk, *SUSTAIN_MS) >= SUSTAIN_MIN]
    elif part == "hats":
        pk = pk[_ratio_after(p, pk, *CLICK_MS) >= CLICK_MIN]
    pk = _apart(pk, f, gap)
    if part == "hats":
        pk = _off_snares(a, pk, p, np.asarray(part_hits(a, "snare", threshold)[0], np.int64))
    if not len(pk):
        return pk, np.zeros(0)
    rel = f[pk] / (np.percentile(f[pk], 95) + 1e-12)
    keep = (rel >= float(threshold)) & (f[pk] >= float(threshold) / 3 * band95)
    return pk[keep], rel[keep]


def drum_times(node, ctx):
    """(times s, velocities, "stem" | "mix") of a drum node's hits; time = (k + 1) * 5 ms, when the rise is known"""
    part = node.get("part")
    if part not in PARTS:
        raise OpError(node.get("id"), f"part must be one of {list(PARTS)}, got {part!r}")
    a, src = drum_analysis(node, ctx)
    gap = node.get("min_gap_ms")
    pk, vel = part_hits(a, part, float(P(node, "threshold")), None if gap is None else float(gap))
    return (pk + 1) * HOP_MS / 1000.0, np.clip(vel, 0.05, 1.0), src


@op("drum", {"part": REQ, "threshold": 0.3, "min_gap_ms": None, "source": "auto"}, inputs=None, kind="trigger",
    chans=lambda n, c: 1)
def op_drum(node, ins, ctx):
    times, vels, _ = drum_times(node, ctx)
    return hits_signal(ctx.frame_of(times), vels, ctx)
