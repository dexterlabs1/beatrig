"""Colour signals: kind "color", values float32 [channels, frames, 4] in linear RGBA.

Colours in rig.json are written either as "#rrggbb" / "#rrggbbaa" hex (sRGB, decoded to linear) or as [r, g, b] /
[r, g, b, a] numbers (already linear). Gradients interpolate in linear RGB. Named palettes are sRGB hex lists.
"""
import numpy as np

from .ops import OpError, P, op
from .signal import Signal

PALETTES = {
    "ember":  ["#120400", "#5c1200", "#c43a00", "#ff8a1f", "#ffd27a", "#fff4dc"],      # dark to bright, warm
    "ice":    ["#02060f", "#0a2147", "#1c5fa8", "#4fb3e8", "#bfeaff", "#f4fcff"],      # dark to bright, cold
    "aurora": ["#07203a", "#0c7a5a", "#36e39a", "#5a7bff", "#b45cff", "#ff6fb3"],      # greens to violets
    "candy":  ["#ff4f9a", "#ffb03b", "#ffe14d", "#45e0c6", "#5a8cff", "#b76cff"],      # saturated brights
    "sunset": ["#1b0b2e", "#5a1a5e", "#b8325a", "#f0603c", "#ffb04a", "#ffe6a0"],
    "ocean":  ["#001018", "#003a4d", "#00708a", "#1fb0b8", "#8de8dc", "#e8fffa"],
    "mono":   ["#000000", "#ffffff"],
}


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, np.float64), 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def parse_color(c):
    """'#rrggbb' / '#rrggbbaa' (sRGB) or [r, g, b(, a)] (linear) -> np.array([r, g, b, a]) linear"""
    if isinstance(c, str):
        h = c.strip().lstrip("#")
        if len(h) not in (6, 8) or any(ch not in "0123456789abcdefABCDEF" for ch in h):
            raise ValueError(f"bad colour {c!r} (want #rrggbb or #rrggbbaa)")
        v = [int(h[i:i + 2], 16) / 255.0 for i in range(0, len(h), 2)]
        rgb = srgb_to_linear(v[:3])
        return np.array([*rgb, v[3] if len(v) == 4 else 1.0])
    if isinstance(c, (list, tuple)) and len(c) in (3, 4) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in c):
        return np.array([float(x) for x in c] + ([1.0] if len(c) == 3 else []))
    raise ValueError(f"bad colour {c!r} (want '#rrggbb' or [r, g, b(, a)] linear)")


def palette_colors(name):
    if name not in PALETTES:
        raise ValueError(f"unknown palette {name!r} (one of {sorted(PALETTES)})")
    return [parse_color(c) for c in PALETTES[name]]


def stops_of(node):
    """(t [K], colours [K, 4]) from a node's `stops` [[t, colour], ...] or `palette` (evenly spaced)"""
    if node.get("stops") is not None:
        st = sorted(((float(t), parse_color(c)) for t, c in node["stops"]), key=lambda s: s[0])
        if not st:
            raise ValueError("stops is empty")
        return np.array([s[0] for s in st]), np.array([s[1] for s in st])
    cols = palette_colors(node.get("palette") or "ember")
    return np.linspace(0.0, 1.0, len(cols)), np.array(cols)


def sample(ts, cols, x):
    """x (any shape) -> colours [..., 4], linear interpolation between stops, clamped at the ends"""
    x = np.asarray(x, np.float64)
    return np.stack([np.interp(x, ts, cols[:, k]) for k in range(4)], axis=-1)


def check_color_node(n):
    """static errors for a colour op's own params (bad colours, unknown palette); [] when fine"""
    errs = []
    try:
        if n.get("op") in ("gradient", "palette_cycle"):
            if n.get("stops") is not None and n.get("palette") is not None:
                errs.append("give stops or palette, not both")
            elif n.get("stops") is not None:
                if not isinstance(n["stops"], list) or not all(isinstance(s, (list, tuple)) and len(s) == 2 for s in n["stops"]):
                    errs.append("stops must be [[t, colour], ...]")
                else:
                    stops_of(n)
            else:
                stops_of(n)
    except ValueError as e:
        errs.append(str(e))
    return errs


@op("gradient", {"stops": None, "palette": None}, kind="color")
def op_gradient(node, ins, ctx):
    """value -> colour: each channel's value (clamped 0..1) picks a colour along the stops or palette"""
    s = ins[0]
    if s.kind == "color":
        raise OpError(node.get("id"), "gradient takes a value signal, not a colour")
    ts, cols = stops_of(node)
    return Signal(sample(ts, cols, np.clip(s.v, 0, 1)), "color", s.hits)


def _ease(x):
    return x * x * (3 - 2 * x)


@op("palette_cycle", {"palette": None, "stops": None, "period_beats": None, "period_s": None, "hold": 0.0, "channels": 1,
                      "phase_step": 0.0, "phase": 0.0}, inputs=None, kind="color", chans=lambda n, c: int(n.get("channels", 1)),
    one_of=(("period_beats", "period_s"),))
def op_palette_cycle(node, ins, ctx):
    """colour drifts through the palette and back to its first colour once per period (beats follow the tempo map).
    Each step eases in and out (zero velocity on every palette colour); `hold` (0..1) = share of each step spent
    resting on a colour. Channel c is offset by phase + c * phase_step (in cycles)."""
    ts, cols = stops_of(node)
    K = len(cols)
    if node.get("period_s") is not None:
        cyc = ctx.t / float(node["period_s"])
    else:
        cyc = ctx.tempo.beats_at(ctx.t) / float(node.get("period_beats") or 4.0)
    C = int(P(node, "channels"))
    ph = cyc[None, :] + float(P(node, "phase")) + float(P(node, "phase_step")) * np.arange(C)[:, None]
    pos = (ph - np.floor(ph)) * K                                  # 0..K: colour index along the loop
    i0 = np.floor(pos).astype(np.int64) % K
    hold = float(np.clip(P(node, "hold"), 0, 0.99))
    f = np.clip((pos - np.floor(pos) - hold) / (1 - hold), 0, 1)
    w = _ease(f)[..., None]
    out = cols[i0] * (1 - w) + cols[(i0 + 1) % K] * w
    return Signal(out, "color")


@op("mix_color", {"amount": 0.5}, inputs="many", kind="color", chans=lambda n, c: max(c) if c else 1)
def op_mix_color(node, ins, ctx):
    """in = [a, b] colours, mixed a*(1-amount) + b*amount; or in = [a, b, t] with t a value signal (0..1) per frame"""
    if len(ins) not in (2, 3):
        raise OpError(node.get("id"), "mix_color takes in = [a, b] or [a, b, t]")
    if ins[0].kind != "color" or ins[1].kind != "color" or (len(ins) == 3 and ins[2].kind == "color"):
        raise OpError(node.get("id"), "mix_color needs two colour inputs (and an optional value mix signal)")
    n = max(s.channels for s in ins)
    a, b = ins[0].broadcast(n), ins[1].broadcast(n)
    t = np.clip(ins[2].broadcast(n).v, 0, 1)[..., None] if len(ins) == 3 else float(P(node, "amount"))
    hits = a.hits | b.hits | (ins[2].broadcast(n).hits if len(ins) == 3 else False)
    return Signal(a.v * (1 - t) + b.v * t, "color", hits)


@op("scale_color", {"gain": 1.0, "alpha": False}, inputs="many", kind="color", chans=lambda n, c: max(c) if c else 1)
def op_scale_color(node, ins, ctx):
    """in = [colour, value]: RGB x value x gain (brightness follows the value); alpha too when alpha is true"""
    if len(ins) != 2 or ins[0].kind != "color" or ins[1].kind == "color":
        raise OpError(node.get("id"), "scale_color takes in = [colour, value]")
    n = max(s.channels for s in ins)
    c, v = ins[0].broadcast(n), ins[1].broadcast(n)
    g = v.v.astype(np.float64)[..., None] * float(P(node, "gain"))
    out = c.v.astype(np.float64).copy()
    out[..., :3] *= g
    if P(node, "alpha"):
        out[..., 3:] *= g
    return Signal(out, "color", c.hits | v.hits)


