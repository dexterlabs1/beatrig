"""Photosensitive-flash safety. A dome fills the whole field of view, so a fast strobe is a real risk.

A transition is a monotonic swing between two turning points of at least `min_swing` x the output range (a zigzag with
that threshold). One flash = a pair of opposite transitions (up then down, or down then up), as in the WCAG / Harding
general-flash rule. A channel fails when any 1 s window holds more than `max_flashes_per_s` flashes. A transition is
placed at the frame where it ends (its turning point).

flash_report() is pure (values in, report out), so a plot can overlay it. check_bake() turns a bake result into check
issues. The `flash_limit` op (registered here) enforces the limit causally: it only drops, delays or holds.
"""
from collections import deque

import numpy as np

from .ops import OpError, op
from .signal import Signal, empty_aux

DEFAULTS = {"max_flashes_per_s": 3.0, "min_swing": 0.1, "outputs": "auto"}
LUMA = np.array([0.2126, 0.7152, 0.0722])                       # Rec. 709 luminance of linear RGB
EMISSIVE_HINTS = ("emiss", "glow", "emit", "bright", "intens", "light", "lum", "flash", "strobe")


def turning_points(v):
    """candidate pivots: the first and last frames plus every frame where the slope changes sign (plateaus skipped)"""
    v = np.asarray(v, np.float64)
    n = len(v)
    if n < 3:
        return np.arange(n)
    d = np.diff(v)
    nz = np.flatnonzero(d)
    if not len(nz):
        return np.array([0, n - 1])
    s = np.sign(d[nz])
    turn = nz[:-1][s[1:] != s[:-1]] + 1                            # first frame of the extreme (start of any plateau)
    return np.unique(np.concatenate(([0], turn, [n - 1])))


def transitions(v, swing):
    """zigzag legs of size >= swing -> (end frames int64, directions +1/-1). Pure; O(turning points)."""
    v = np.asarray(v, np.float64)
    if len(v) == 0 or swing <= 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    idx = turning_points(v)
    vals = v[idx].tolist()
    idx = idx.tolist()
    ends, dirs = [], []
    lo_v, lo_i, hi_v, hi_i = vals[0], idx[0], vals[0], idx[0]
    d = 0
    piv = ext = ext_i = None
    for i, x in zip(idx, vals):
        if d == 0:                                                 # no leg yet: wait for the first swing either way
            if x < lo_v:
                lo_v, lo_i = x, i
            if x > hi_v:
                hi_v, hi_i = x, i
            if x - lo_v >= swing and i == hi_i:
                d, piv, ext, ext_i = 1, lo_v, x, i
            elif hi_v - x >= swing and i == lo_i:
                d, piv, ext, ext_i = -1, hi_v, x, i
        elif d == 1:
            if x > ext:
                ext, ext_i = x, i
            elif ext - x >= swing:
                ends.append(ext_i)
                dirs.append(1)
                d, piv, ext, ext_i = -1, ext, x, i
        else:
            if x < ext:
                ext, ext_i = x, i
            elif x - ext >= swing:
                ends.append(ext_i)
                dirs.append(-1)
                d, piv, ext, ext_i = 1, ext, x, i
    if d != 0 and abs(ext - piv) >= swing:                         # the last, still open leg
        ends.append(ext_i)
        dirs.append(d)
    return np.asarray(ends, np.int64), np.asarray(dirs, np.int64)


def flash_report(values, fps, lo=None, hi=None, max_flashes_per_s=3.0, min_swing=0.1) -> dict:
    """Flash counts for one channel of per-frame values (or colour rows [[r, g, b, a], ...], judged on Rec. 709 luminance).
    lo/hi: the output range (default: the values' own min/max). A transition needs min_swing x (hi - lo).
    Returns {"flashes", "max_per_s", "limit", "ok", "per_second": [flashes in each whole second],
             "windows": [[t0, t1], ...] (merged 1 s windows, in seconds, holding more than the limit)}."""
    v = np.asarray(values, np.float64)
    if v.ndim == 2 and v.shape[1] in (3, 4):                     # colour rows [[r, g, b(, a)], ...]: judge luminance
        v = v[:, :3] @ LUMA
    v = v.ravel()
    fps = float(fps)
    if lo is None or hi is None:
        lo, hi = (float(v.min()), float(v.max())) if len(v) else (0.0, 0.0)
    rng = abs(float(hi) - float(lo))
    limit = float(max_flashes_per_s)
    n_sec = int(np.ceil(len(v) / fps)) if len(v) else 0
    empty = {"flashes": 0, "max_per_s": 0, "limit": limit, "ok": True, "per_second": [0] * n_sec, "windows": []}
    if rng <= 0 or len(v) < 2:
        return empty
    ends, _ = transitions(v, float(min_swing) * rng)
    if not len(ends):
        return empty
    W = fps                                                        # 1 s in frames; window = [e, e + W)
    upto = np.searchsorted(ends, ends + W - 1e-9, side="left")
    counts = (upto - np.arange(len(ends))) // 2                    # flashes in the window that starts at each leg
    per_second = np.bincount((ends / fps).astype(np.int64), minlength=n_sec)[:n_sec] // 2
    windows = []
    for j in np.flatnonzero(counts > limit).tolist():
        t0, t1 = ends[j] / fps, (ends[j] + W) / fps
        if windows and t0 <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], t1)
        else:
            windows.append([t0, t1])
    return {"flashes": int(len(ends) // 2), "max_per_s": int(counts.max()), "limit": limit, "ok": not windows,
            "per_second": per_second.astype(int).tolist(), "windows": [[round(a, 3), round(b, 3)] for a, b in windows]}


def settings(rig_data):
    s = rig_data.get("safety")
    return dict(DEFAULTS, **(s if isinstance(s, dict) else {})), isinstance(s, dict)


def monitored(output, chosen):
    """is this output checked? chosen: "auto" | "all" | "none" | [output ids]"""
    if isinstance(chosen, list):
        return output.get("id") in chosen
    if chosen == "none":
        return False
    t = output.get("target") or {}
    kind = t.get("kind")
    if chosen == "all":
        return kind != "transform"
    if kind in ("light", "mpc"):
        return True
    if kind == "material":
        p = str(t.get("param", "")).lower()
        return output.get("value_type") == "color" or t.get("value_type") == "color" or any(h in p for h in EMISSIVE_HINTS)
    return False


def channel_series(output, arrays=None):
    """[(label, values 1-D, lo, hi)] for a bake output: each channel, plus the mean of all channels (the field's overall
    brightness) when there are several. Colour channels use Rec. 709 luminance, range 0..max luminance of the output.
    arrays: the channels' values as numpy arrays (bake passes them, saving a list -> array conversion)"""
    chans = output["channels"]
    if not chans:
        return []
    vals = arrays if arrays is not None else [np.asarray(c["values"], np.float64) for c in chans]
    color = vals[0].ndim == 2 if len(vals) else False
    rows = []
    if color:
        lum = [v[:, :3] @ LUMA for v in vals]
        lo, hi = 0.0, max(float(x.max()) for x in lum) if lum else 0.0
        series = lum
    else:
        lo, hi = float(output.get("min", 0.0)), float(output.get("max", 1.0))
        series = list(vals)
    for c, s in zip(chans, series):
        rows.append((c.get("actor") or output["id"], s, lo, hi))
    if len(series) > 1:
        rows.append(("(mean of all channels)", np.mean(series, axis=0), lo, hi))
    return rows


def check_bake(rig_data, result, arrays=None) -> tuple:
    """(issues, summary) for a bake result. Issues are errors when the rig has a `safety` block, else warnings.
    summary = {"limit", "min_swing", "violations": [{"output", "channel", "max_per_s", "windows"}]}"""
    s, configured = settings(rig_data)
    level = "error" if configured else "warning"
    by_id = {o.get("id"): o for o in rig_data.get("outputs", [])}
    issues, viol = [], []
    for o in result["outputs"]:
        if not monitored(dict(by_id.get(o["id"], {}), **{"target": o["target"]}), s["outputs"]):
            continue
        for label, vals, lo, hi in channel_series(o, (arrays or {}).get(o["id"])):
            r = flash_report(vals, result["fps"], lo, hi, s["max_flashes_per_s"], s["min_swing"])
            if r["ok"]:
                continue
            viol.append({"output": o["id"], "channel": label, "max_per_s": r["max_per_s"], "windows": r["windows"][:20]})
            w = ", ".join(f"{a:.2f}-{b:.2f}s" for a, b in r["windows"][:4]) + (" ..." if len(r["windows"]) > 4 else "")
            issues.append({"level": level, "where": f"output {o['id']}",
                           "msg": f"{label} flashes {r['max_per_s']} times in 1 s (limit {s['max_flashes_per_s']:g}) at {w}: "
                                  "photosensitivity risk; add a flash_limit op or slow the envelope"})
    return issues, {"limit": float(s["max_flashes_per_s"]), "min_swing": float(s["min_swing"]), "violations": viol}


# ---------- the op
def _limit_triggers(s, interval, mode):
    """per channel: hits closer than `interval` frames to the last emitted hit are dropped, or (mode delay) held until
    the interval has passed (several held hits merge, keeping the strongest). Only drops or delays: never early."""
    C, N = s.v.shape
    v = np.zeros_like(s.v)
    aux = empty_aux(v.shape)
    for c in range(C):
        last, pend = -np.inf, None                                 # pend = (frame, vel, src frame)
        def emit(f, vel, src):
            v[c, f] = vel
            for k in aux:
                if k in s.aux:
                    aux[k][c, f] = s.aux[k][c, src]
        for f in np.flatnonzero(s.v[c] != 0).tolist():
            if pend is not None and pend[0] <= f:
                emit(*pend)
                last, pend = pend[0], None
            vel = float(s.v[c, f])
            if f - last >= interval:
                emit(f, vel, f)
                last = f
            elif mode == "delay":
                at = int(np.ceil(last + interval - 1e-9))
                if pend is None or vel > pend[1]:
                    pend = (at, vel, f) if at < N else None
        if pend is not None and pend[0] < N:
            emit(*pend)
    return Signal(v, "trigger", v != 0, aux)


def _limit_values(x, fps, max_per_s, swing):
    """causal clamp on one channel: track zigzag transitions as they are confirmed; when the trailing 1 s already holds
    2 * max_per_s - 1 of them, hold the output short of completing another until one leaves the window"""
    budget = max(1, int(2 * max_per_s) - 1)
    W = fps
    out = np.empty(len(x), np.float64)
    conf = deque()
    xs = np.asarray(x, np.float64).tolist()
    d = 0
    lo = hi = ext = xs[0] if xs else 0.0
    eps = 1e-6 * max(swing, 1e-12)
    for i, xi in enumerate(xs):
        while conf and conf[0] <= i - W:
            conf.popleft()
        full = len(conf) >= budget
        y = xi
        if d == 0:
            if full:
                y = min(max(y, hi - swing + eps), lo + swing - eps)
            if y - lo >= swing:
                d, ext = 1, y
                conf.append(i)
            elif hi - y >= swing:
                d, ext = -1, y
                conf.append(i)
            lo, hi = min(lo, y), max(hi, y)
        elif d == 1:
            if y > ext:
                ext = y
            elif ext - y >= swing:
                if full:
                    y = ext - swing + eps
                else:
                    d, ext = -1, y
                    conf.append(i)
        else:
            if y < ext:
                ext = y
            elif y - ext >= swing:
                if full:
                    y = ext + swing - eps
                else:
                    d, ext = 1, y
                    conf.append(i)
        out[i] = y
    return out


@op("flash_limit", {"max_flashes_per_s": None, "min_swing": None, "mode": "drop"}, kind=lambda n, k: k[0] if k else "value")
def op_flash_limit(node, ins, ctx):
    """Causal flash limiter. Triggers: at most max_flashes_per_s hits per second per channel (later hits dropped, or
    with mode "delay" held until allowed). Values: the output holds short of a new transition (>= min_swing) while the
    trailing second is already at the limit. Defaults come from rig.safety, else 3 per second and 0.1."""
    s = ins[0]
    if s.kind == "color":
        raise OpError(node.get("id"), "flash_limit takes a value or trigger signal (limit the value before gradient)")
    rate = float(node["max_flashes_per_s"] if node.get("max_flashes_per_s") is not None else ctx.safety["max_flashes_per_s"])
    swing = float(node["min_swing"] if node.get("min_swing") is not None else ctx.safety["min_swing"])
    if rate <= 0:
        raise OpError(node.get("id"), "max_flashes_per_s must be > 0")
    if s.kind == "trigger":
        mode = node.get("mode") or "drop"
        if mode not in ("drop", "delay"):
            raise OpError(node.get("id"), f"mode {mode!r} not in ['drop', 'delay']")
        return _limit_triggers(s, ctx.fps / rate, mode)
    v = np.stack([_limit_values(row, ctx.fps, rate, swing) for row in s.v])
    return Signal(v, "value", s.hits)


