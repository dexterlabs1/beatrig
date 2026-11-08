"""Pure helpers for the Unreal applier: frame-rate and tick maths, key planning, tagging and read-back comparison.

No `unreal` import here, so everything in this file runs (and is tested) outside the editor. beatrig_apply.py does the
editor calls; this file decides what to write.

Two time bases, and the reason this file exists:
  - Scripting float/double channels (MovieSceneScriptingFloatChannel.add_key, transform channels, section ranges) take
    DISPLAY frames by default (time_unit=DISPLAY_RATE).
  - MovieSceneParameterSection.add_scalar_parameter_key / add_color_parameter_key (MPC tracks, component material
    tracks) take TICK time: display frame x tick_resolution / display_rate. At 30 fps with the default 24000 ticks/s
    that is 800 ticks per frame. Passing display frames there silently keys everything into the first second.
  - Reading back is the other way round: channel.get_keys() reports display frames for both.

Colour keys ([frame, r, g, b, a], linear RGBA) become four channels per track. Their names differ by track type, and
channel_names() says which to read back.

Output blend (v0.3): a "multiply" or "add" output carries gains or offsets. Over a base section (the show's own keys,
left active and untouched) the applier writes an ADDITIVE section: offset_frame_keys() gives base x (gain - 1) or the
offset, from the base evaluated at every frame. With no base it writes an ABSOLUTE section of static x gain
(blend_frame_keys()). Both use the engine's relative-tolerance RDP (rdp() here is a pure-Python twin of
beatrig.bake.rdp).
"""
import hashlib
import json
import os
import re
from fractions import Fraction

TAG_PREFIX = "beatrig:"

# Common NTSC rates written as decimals in rig.json/bake.json -> the exact rational Unreal uses.
_NTSC = {23.976: Fraction(24000, 1001), 29.97: Fraction(30000, 1001), 47.952: Fraction(48000, 1001),
         59.94: Fraction(60000, 1001), 119.88: Fraction(120000, 1001)}

# bake target kind -> which time base its keys take, and which track the applier creates
TRACKS = {
    "mpc": ("ticks", "MovieSceneMaterialParameterCollectionTrack"),
    "material": ("ticks", "MovieSceneComponentMaterialTrack"),
    "light": ("frames", "MovieSceneFloatTrack"),
    "niagara": ("frames", "MovieSceneNiagaraFloatParameterTrack"),
    "transform": ("frames", "MovieScene3DTransformTrack"),
}
# the same for colour outputs (value_type "color"); niagara and transform have no colour form
COLOR_TRACKS = {
    "mpc": ("ticks", "MovieSceneMaterialParameterCollectionTrack"),
    "material": ("ticks", "MovieSceneComponentMaterialTrack"),
    "light": ("frames", "MovieSceneColorTrack"),
}

# light target "property" -> the light component's property name/path for a float track
LIGHT_PROPERTIES = {"intensity": "Intensity", "attenuation_radius": "AttenuationRadius",
                    "source_radius": "SourceRadius", "indirect_lighting_intensity": "IndirectLightingIntensity",
                    "volumetric_scattering_intensity": "VolumetricScatteringIntensity", "temperature": "Temperature"}

# light colour "property" -> the light component property (an FColor; Sequencer converts the linear keys)
LIGHT_COLOR_PROPERTIES = {"color": "LightColor", "light_color": "LightColor"}

# transform "property.axis" -> index into MovieScene3DTransformSection.get_all_channels(), and that channel's name.
# UE 5.8 order: Location.X/Y/Z, Rotation.X/Y/Z, Scale.X/Y/Z (no weight channel). Rotation X = roll, Y = pitch, Z = yaw.
TRANSFORM_CHANNELS = {"location.x": 0, "location.y": 1, "location.z": 2, "rotation.x": 3, "rotation.y": 4,
                      "rotation.z": 5, "scale.x": 6, "scale.y": 7, "scale.z": 8}
TRANSFORM_CHANNEL_NAMES = {k: k.split(".")[0].capitalize() + "." + k.split(".")[1].upper() for k in TRANSFORM_CHANNELS}
_AXES = {"x": "x", "y": "y", "z": "z", "roll": "x", "pitch": "y", "yaw": "z"}


class PlanError(ValueError):
    """a bake that can't be applied as asked (bad rate, unknown target kind/property)"""


def rate(value):
    """fps as an exact Fraction. Accepts 30, 29.97, "30000/1001", (30000, 1001) or a Fraction."""
    if isinstance(value, Fraction):
        r = value
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        r = Fraction(int(value[0]), int(value[1]))
    elif isinstance(value, str) and "/" in value:
        a, b = value.split("/")
        r = Fraction(int(a), int(b))
    else:
        f = float(value)
        r = next((exact for approx, exact in _NTSC.items() if abs(f - approx) < 5e-4), None)
        if r is None:
            r = Fraction(f).limit_denominator(1001)
    if r <= 0:
        raise PlanError(f"frame rate must be > 0, got {value!r}")
    return r


def ticks_per_frame(tick_resolution, display_rate):
    """exact ticks per display frame (800 at 24000/30; 800.8 at 24000/29.97)"""
    return rate(tick_resolution) / rate(display_rate)


def frame_to_ticks(frame, tick_resolution=24000, display_rate=30):
    """display frame -> integer tick, rounded half away from zero from the exact rational"""
    t = Fraction(frame) * ticks_per_frame(tick_resolution, display_rate)
    n = t.numerator // t.denominator
    rem = t - n
    return int(n + (1 if rem >= Fraction(1, 2) else 0))


def ticks_to_frame(ticks, tick_resolution=24000, display_rate=30):
    """tick -> display frame as a float (for read-back comparison)"""
    return float(Fraction(int(ticks)) / ticks_per_frame(tick_resolution, display_rate))


def check_rate(bake_fps, display_rate):
    """the applier never resamples: the bake's fps must match the sequence's display rate. Returns None or an error."""
    if rate(bake_fps) != rate(display_rate):
        return (f"bake is {float(rate(bake_fps)):g} fps but the sequence displays at {float(rate(display_rate)):g} fps; "
                f"set the rig's fps to the sequence's display rate and bake again")
    return None


def tag(output_id, channel=None):
    """the display name every created track carries; undo removes exactly the tracks with this prefix"""
    return f"{TAG_PREFIX}{output_id}" + (f":{channel}" if channel is not None else "")


def parse_tag(name):
    """'beatrig:lights:Light_0' -> ('lights', 'Light_0'); anything else -> None"""
    name = str(name or "")
    if not name.startswith(TAG_PREFIX):
        return None
    rest = name[len(TAG_PREFIX):]
    oid, _, ch = rest.partition(":")
    return (oid, ch or None) if oid else None


def is_ours(name, output_id=None):
    p = parse_tag(name)
    return p is not None and (output_id is None or p[0] == output_id)


def to_time_base(keys, base, start_frame=0, tick_resolution=24000, display_rate=30):
    """[[frame, value], ...] -> [(time, value), ...] in the given base ('frames' or 'ticks'), shifted by start_frame.
    Colour rows [frame, r, g, b, a] -> [(time, (r, g, b, a)), ...]."""
    if base not in ("frames", "ticks"):
        raise PlanError(f"unknown time base {base!r}")
    out = []
    for k in keys:
        f = int(k[0]) + int(start_frame)
        t = f if base == "frames" else frame_to_ticks(f, tick_resolution, display_rate)
        out.append((t, float(k[1]) if len(k) == 2 else tuple(float(x) for x in k[1:5])))
    return out


def value_type(output):
    """'color' or 'float': the target's (or output's) value_type, else inferred from the key rows ([f, r, g, b, a])"""
    t = output.get("target") or {}
    vt = t.get("value_type") or output.get("value_type")
    if vt:
        vt = str(vt).lower()
        if vt in ("color", "colour"):
            return "color"
        if vt in ("float", "value", "scalar"):
            return "float"
        raise PlanError(f"{output.get('id')}: value_type {vt!r} unknown (float or color)")
    for ch in output.get("channels", []):
        for k in ch.get("keys") or []:
            return "color" if len(k) == 5 else "float"
    return "float"


def transform_channel(output):
    """a transform target -> 'location.x' ... 'scale.z'. Accepts the contract form {property, axis} (rotation axes may be
    roll/pitch/yaw) and the older {channel: "location.x"}."""
    t = output.get("target") or {}
    if t.get("channel"):
        prop, _, axis = str(t["channel"]).lower().partition(".")
    else:
        prop, axis = str(t.get("property", "")).lower(), str(t.get("axis", "")).lower()
    c = f"{prop}.{_AXES.get(axis, axis) if prop == 'rotation' else axis}"
    if c not in TRANSFORM_CHANNELS:
        raise PlanError(f"{output.get('id')}: transform needs property location|rotation|scale and axis x|y|z "
                        f"(rotation also roll|pitch|yaw), got {prop!r}/{axis!r}")
    return c


def niagara_param(name):
    """'Rate' or 'User.Rate' -> 'User.Rate' (Sequencer parameter tracks drive user parameters)"""
    name = str(name)
    return name if name.startswith("User.") else "User." + name


def channel_names(plan):
    """the channel names to read back for a planned track, in key order. Scalar parameter channels carry the parameter
    name; colour parameter channels append R/G/B/A with no separator (Tint -> TintR); a colour property track's
    channels are Color.R..A; transform channels are Location.X etc. None means 'the first channel'."""
    kind, vt, p = plan["kind"], plan["value_type"], plan.get("param")
    if kind in ("mpc", "material"):
        return [p + c for c in "RGBA"] if vt == "color" else [p]
    if kind == "light" and vt == "color":
        return ["Color." + c for c in "RGBA"]
    if kind == "transform":
        return [plan["channel_name"]]
    return None


def source_hash(folder):
    """hash of the in-editor modules (every .py in folder), so a long-lived editor session can tell a stale import"""
    h = hashlib.sha1()
    for name in sorted(n for n in os.listdir(folder) if n.endswith(".py")):
        h.update(name.encode())
        with open(os.path.join(folder, name), "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def plan_output(output, bake_fps, display_rate, tick_resolution, start_frame=0):
    """One bake output -> a list of track writes. Each item:
        {"tag", "output", "kind", "value_type" (float|color), "track", "time_base", "actor" | None, "asset" | None,
         "param" | None, "property" | None, "channel_index" | None, "channel_name" | None, "blend" (absolute|additive),
         "output_blend" (replace|multiply|add), "material_index", "range": [first, last+1] display frames,
         "keys": [(time, value | (r, g, b, a)), ...], and for multiply/add "src_keys": the gains/offsets in display
         frames (the applier reads the base and replaces "keys", "range" and "blend", see blended_plan)}
    Raises PlanError for anything that can't be written."""
    err = check_rate(bake_fps, display_rate)
    if err:
        raise PlanError(err)
    oid = output.get("id")
    t = output.get("target") or {}
    kind = t.get("kind")
    if kind not in TRACKS:
        raise PlanError(f"{oid}: unknown target kind {kind!r} (expected one of {sorted(TRACKS)})")
    vt = value_type(output)
    if vt == "color" and kind not in COLOR_TRACKS:
        raise PlanError(f"{oid}: a {kind} target can't take colour (colour goes to mpc, light or material)")
    base, track = (COLOR_TRACKS if vt == "color" else TRACKS)[kind]
    prop = chan_idx = chan_name = None
    blend = "absolute"
    if kind == "light":
        p = str(t.get("property", "color" if vt == "color" else "intensity")).lower()
        table = LIGHT_COLOR_PROPERTIES if vt == "color" else LIGHT_PROPERTIES
        if p not in table:
            raise PlanError(f"{oid}: light {'colour ' if vt == 'color' else ''}property {p!r} is not supported "
                            f"(one of {sorted(table)})")
        prop = table[p]
    if kind == "transform":
        prop = transform_channel(output)
        chan_idx, chan_name = TRANSFORM_CHANNELS[prop], TRANSFORM_CHANNEL_NAMES[prop]
        blend = "additive" if t.get("additive", True) else "absolute"
    if kind in ("mpc", "material", "niagara") and not t.get("param"):
        raise PlanError(f"{oid}: {kind} target needs a param")
    param = niagara_param(t["param"]) if kind == "niagara" else t.get("param")
    mode = output_blend(output)
    if mode != "replace" and kind == "transform":
        raise PlanError(f"{oid}: blend {mode!r} is not for transforms; they already layer on the actor's own animation, "
                        f"use target.additive for transforms")
    width = 5 if vt == "color" else 2
    plans = []
    for ch in output.get("channels", []):
        keys = ch.get("keys") or []
        if not keys:
            continue
        actor = ch.get("actor")
        if kind != "mpc" and not actor:
            raise PlanError(f"{oid}: {kind} channel has no actor")
        bad = next((k for k in keys if len(k) != width), None)
        if bad is not None:
            raise PlanError(f"{oid}: {vt} keys are [frame{', r, g, b, a' if width == 5 else ', value'}], got {bad!r}")
        first, last = int(keys[0][0]) + int(start_frame), int(keys[-1][0]) + int(start_frame)
        p = {
            "tag": tag(oid, actor), "output": oid, "kind": kind, "value_type": vt, "track": track, "time_base": base,
            "actor": actor, "asset": t.get("asset"), "param": param, "property": prop,
            "channel_index": chan_idx, "channel_name": chan_name, "blend": blend, "output_blend": mode,
            "material_index": int(t.get("material_index", 0)),
            "range": [first, last + 1],
            "keys": to_time_base(keys, base, start_frame, tick_resolution, display_rate),
        }
        if mode != "replace":
            # gains / offsets in display frames; the applier turns them into final keys once it has read the base
            p["src_keys"] = to_time_base(keys, "frames", start_frame)
        plans.append(p)
    return plans


def bake_start(bake, start_frame=0):
    """the sequence frame of the bake's key frame 0: the bake's own "start_frame" (a bake whose values start at that
    sequence frame; default 0) plus the applier's start_frame (--start), which shifts it further"""
    try:
        return int(bake.get("start_frame") or 0) + int(start_frame)
    except (TypeError, ValueError):
        raise PlanError(f"bake start_frame must be a whole frame, got {bake.get('start_frame')!r}") from None


def plan_bake(bake, display_rate, tick_resolution, start_frame=0, only=None):
    """every output in a bake.json dict (or just the ids in `only`) -> track writes. A key at frame f lands on sequence
    frame bake["start_frame"] + start_frame + f."""
    if int(bake.get("version", 0)) != 1:
        raise PlanError(f"unsupported bake version {bake.get('version')!r}")
    start = bake_start(bake, start_frame)
    out = []
    for o in bake.get("outputs", []):
        if only and o.get("id") not in only:
            continue
        out.extend(plan_output(o, bake.get("fps", 30), display_rate, tick_resolution, start))
    return out


def frame_keys(plan, tpf):
    """a plan's keys as [(display frame, value or (r, g, b, a))] (read-back always reports display frames).
    tpf = ticks per display frame."""
    if plan["time_base"] == "frames":
        return list(plan["keys"])
    tpf = Fraction(tpf)
    return [(float(Fraction(int(t)) / tpf), v) for t, v in plan["keys"]]


def split_components(keys):
    """[(frame, (r, g, b, a))] -> four [(frame, value)] lists, one per component"""
    return [[(f, v[i]) for f, v in keys] for i in range(4)]


def merge_reports(reports):
    """several compare_keys reports (one per colour component) -> one report of the same shape"""
    bad = next((r["first_bad"] for r in reports if r["first_bad"] is not None), None)
    return {"ok": all(r["ok"] for r in reports), "expected": sum(r["expected"] for r in reports),
            "actual": sum(r["actual"] for r in reports), "missing": sum(r["missing"] for r in reports),
            "extra": sum(r["extra"] for r in reports),
            "max_value_err": max((r["max_value_err"] for r in reports), default=0.0), "first_bad": bad}


# ------------------------------------------------------------------ output blend (v0.3): multiply / add onto a base
BLENDS = ("replace", "multiply", "add")
NEUTRAL = {"multiply": 1.0, "add": 0.0}     # the gain / offset that leaves the base unchanged
DEFAULT_KEY_TOLERANCE = 0.002                # the engine's default bake.key_tolerance (relative)


def output_blend(output):
    """'replace' (default), 'multiply' or 'add' from an output's "blend" """
    b = str(output.get("blend") or "replace").lower()
    if b not in BLENDS:
        raise PlanError(f"{output.get('id')}: blend {b!r} unknown (one of {', '.join(BLENDS)})")
    return b


def rdp(values, tol, forced=()):
    """Indices of linear keys whose reconstruction stays within tol of values everywhere (vertical error). The first and
    last indices and every forced index are kept. values may hold floats or equal-length tuples (colour): the error is
    then the largest component error. Pure Python; keeps exactly the indices beatrig.bake.rdp keeps (a span's split is
    its largest error, the first one on ties)."""
    n = len(values)
    if n == 0:
        return []
    keep = [False] * n
    keep[0] = keep[-1] = True
    for i in forced:
        if 0 <= i < n:
            keep[i] = True
    vec = isinstance(values[0], (tuple, list))
    idx = [i for i in range(n) if keep[i]]
    stack = [(a, b) for a, b in zip(idx, idx[1:]) if b - a >= 2]
    while stack:
        a, b = stack.pop()
        span = b - a
        va, vb = values[a], values[b]
        best, at = -1.0, -1
        if vec:
            d = [y - x for x, y in zip(va, vb)]
            for i in range(1, span):
                t = i / span
                vj = values[a + i]
                e = max(abs(vj[k] - (va[k] + d[k] * t)) for k in range(len(va)))
                if e > best:
                    best, at = e, a + i
        else:
            d = vb - va
            for i in range(1, span):
                e = abs(values[a + i] - (va + d * (i / span)))
                if e > best:
                    best, at = e, a + i
        if best > tol:
            keep[at] = True
            if at - a >= 2:
                stack.append((a, at))
            if b - at >= 2:
                stack.append((at, b))
    return [i for i in range(n) if keep[i]]


def sample_keys(keys, frames, neutral):
    """[(frame, value or tuple)] linear keys -> their value at each frame in `frames` (ascending). Frames before the
    first key or after the last get `neutral` (a gain of 1 / an offset of 0 outside the bake range)."""
    out = []
    keys = sorted((float(f), v) for f, v in keys)
    if not keys:
        return [neutral for _ in frames]
    vec = isinstance(keys[0][1], (tuple, list))
    j, last = 0, len(keys) - 1
    for f in frames:
        if f < keys[0][0] - 1e-9 or f > keys[-1][0] + 1e-9:
            out.append(neutral)
            continue
        while j < last and keys[j + 1][0] <= f:
            j += 1
        f0, v0 = keys[j]
        if j == last or abs(f - f0) < 1e-9:
            out.append(v0)
            continue
        f1, v1 = keys[j + 1]
        t = (f - f0) / (f1 - f0)
        out.append(tuple(a + (b - a) * t for a, b in zip(v0, v1)) if vec else v0 + (v1 - v0) * t)
    return out


def combine(base, gain, mode):
    """final = base x gain (multiply) or base + gain (add); tuples combine per component (colour: RGBA)"""
    if mode not in NEUTRAL:
        raise PlanError(f"combine: blend {mode!r} is not multiply or add")
    mul = mode == "multiply"
    if isinstance(base, (tuple, list)):
        g = gain if isinstance(gain, (tuple, list)) else (gain,) * len(base)
        return tuple(b * x if mul else b + x for b, x in zip(base, g))
    return base * gain if mul else base + gain


def value_range(values):
    """max - min over floats, or the largest per-component range over tuples"""
    if not values:
        return 0.0
    if isinstance(values[0], (tuple, list)):
        return max(max(v[k] for v in values) - min(v[k] for v in values) for k in range(len(values[0])))
    return max(values) - min(values)


def offset_value(base, gain, mode):
    """the additive offset that turns base into the blended value: base x (gain - 1) for multiply, the offset itself for
    add (tuples per component)"""
    if mode == "add":
        return gain
    if mode != "multiply":
        raise PlanError(f"offset_value: blend {mode!r} is not multiply or add")
    if isinstance(base, (tuple, list)):
        g = gain if isinstance(gain, (tuple, list)) else (gain,) * len(base)
        return tuple(b * (x - 1.0) for b, x in zip(base, g))
    return base * (gain - 1.0)


def offset_frame_keys(src_keys, mode, span_first, base_values, tol_rel=DEFAULT_KEY_TOLERANCE):
    """The keys of the ADDITIVE section that sits on top of an untouched base, as [(display frame, offset)].
    src_keys: the output's gain/offset keys [(display frame, value)] (already shifted by the start frame).
    base_values: the base evaluated at display frames span_first, span_first + 1, ... The span is normally the bake
    range plus one frame each side (first - 1 .. last + 1): outside the gain keys the gain is neutral, so the offset is
    exactly 0 there and those two keys bring the section back to nothing (frames outside play the base alone).
    multiply: offset = base x (gain - 1); add: offset = the offset keys (the base doesn't matter, pass zeros).
    Simplified with the engine's RDP; the tolerance is tol_rel x the range of the final curve (base + offset), since an
    error in the offset is the same error in the final value. Every gain key frame is kept (they include the hits)."""
    n = len(base_values)
    if n == 0:
        return []
    vec = isinstance(base_values[0], (tuple, list))
    neutral = NEUTRAL[mode]
    if vec:
        neutral = (neutral,) * len(base_values[0])
    frames = range(int(span_first), int(span_first) + n)
    offs = [offset_value(b, g, mode) for b, g in zip(base_values, sample_keys(src_keys, frames, neutral))]
    final = [combine(b, o, "add") for b, o in zip(base_values, offs)]
    rng = value_range(final)
    tol = tol_rel * rng if rng > 0 else tol_rel
    forced = sorted({int(round(float(f))) - int(span_first) for f, _ in src_keys})
    idx = rdp(offs, tol, [i for i in forced if 0 <= i < n])
    return [(int(span_first) + i, offs[i]) for i in idx]


def target_key(plan):
    """one string per animated target (the same MPC parameter, the same light property of one actor, ...), so two beatrig
    outputs blending the same target can be told apart from two blending neighbours in one shared section"""
    return "|".join(str(plan.get(k)) for k in ("kind", "asset", "actor", "param", "property", "material_index",
                                                "channel_name", "value_type"))


def blend_frame_keys(src_keys, mode, span_first, base_values, tol_rel=DEFAULT_KEY_TOLERANCE):
    """The keys of final = base (x|+) gain, as [(display frame, value or (r, g, b, a))] (the no-base case: written as an
    ABSOLUTE section over the bake range, with base_values the recorded static value at every frame).
    src_keys: the output's gain/offset keys [(display frame, value)] (already shifted by the start frame).
    base_values: the base evaluated at display frames span_first, span_first + 1, ... (one per frame of the span).
    The gain between its keys is their linear interpolation (what Sequencer would play for the gain curve), neutral
    outside them. The result is simplified with the engine's relative-tolerance RDP (tolerance x the final values'
    range); every gain key frame is kept (they include the trigger hits), and so are the frames just outside the gain
    keys when the span reaches them."""
    n = len(base_values)
    if n == 0:
        return []
    vec = isinstance(base_values[0], (tuple, list))
    neutral = NEUTRAL[mode]
    if vec:
        neutral = (neutral,) * len(base_values[0])
    frames = range(int(span_first), int(span_first) + n)
    gains = sample_keys(src_keys, frames, neutral)
    final = [combine(b, g, mode) for b, g in zip(base_values, gains)]
    rng = value_range(final)
    tol = tol_rel * rng if rng > 0 else tol_rel
    forced = set()
    for f, _ in src_keys:
        i = int(round(float(f))) - int(span_first)
        forced.add(i)
    if src_keys:
        forced.add(int(round(float(src_keys[0][0]))) - int(span_first) - 1)
        forced.add(int(round(float(src_keys[-1][0]))) - int(span_first) + 1)
    idx = rdp(final, tol, sorted(i for i in forced if 0 <= i < n))
    return [(int(span_first) + i, final[i]) for i in idx]


def blended_plan(plan, frame_keys, rng, section_blend, tick_resolution=24000, display_rate=30):
    """a copy of a multiply/add plan ready to write: keys in the plan's time base, section range rng = [start, end)
    display frames, and the section's blend type ("additive": an offset over a base section; "absolute": final values
    over a static base)"""
    if section_blend not in ("additive", "absolute"):
        raise PlanError(f"section blend {section_blend!r} is not additive or absolute")
    out = dict(plan)
    out["keys"] = frames_to_time_base(frame_keys, plan["time_base"], tick_resolution, display_rate)
    out["range"] = list(rng)
    out["blend"] = section_blend
    return out


# What a multiply/add track was applied over is recorded in the sequence asset's metadata, one tag per track:
#   "beatrig.base:<track tag>" -> JSON {"v": 1, "output", "mode", "target": target_key(plan), "kind": "section" | "static",
#                                       "section": <base section path inside the sequence>, "track": <base track name>,
#                                       "channels": [...], "fp": <fingerprint of the base keys> (multiply only),
#                                       "static": value | [r, g, b, a]}
BASE_META_PREFIX = "beatrig.base:"


def base_meta_key(track_tag):
    return BASE_META_PREFIX + str(track_tag)


def base_meta_tag(key):
    """'beatrig.base:beatrig:glow' -> 'beatrig:glow'; anything else -> None"""
    key = str(key)
    return key[len(BASE_META_PREFIX):] if key.startswith(BASE_META_PREFIX) else None


def encode_record(rec):
    return json.dumps(rec, sort_keys=True, separators=(",", ":"))


def decode_record(text):
    try:
        rec = json.loads(text)
    except (TypeError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def relative_path(obj_path, seq_path):
    """an object path inside a sequence, relative to it ('MovieScene_0.Track_1.Section_0'), so a renamed or moved
    sequence still resolves its records; None when obj_path is not inside seq_path"""
    prefix = str(seq_path) + ":"
    return obj_path[len(prefix):] if str(obj_path).startswith(prefix) else None


def fingerprint(rows):
    """short stable hash of anything JSON-able (the base channels' keys), to tell when a base was edited"""
    return hashlib.sha1(json.dumps(rows, separators=(",", ":"), default=str).encode()).hexdigest()[:16]


def srgb8_to_linear(c):
    """an 8-bit sRGB colour (r, g, b, a 0..255; an FColor such as LightColor) -> linear RGBA floats, the way
    FLinearColor(FColor) converts (sRGB curve on RGB, alpha linear)"""
    def ch(x):
        x = x / 255.0
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
    r, g, b, a = (list(c) + [255])[:4]
    return (ch(r), ch(g), ch(b), a / 255.0)


def editor_property_name(name):
    """an Unreal property name -> its Python editor-property name ('AttenuationRadius' -> 'attenuation_radius')"""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name)).lower()


def frames_to_time_base(frame_keys, base, tick_resolution=24000, display_rate=30):
    """[(display frame, value)] -> [(time, value)] in 'frames' or 'ticks' (no start shift: the frames already carry it)"""
    if base == "frames":
        return [(int(f), v) for f, v in frame_keys]
    if base == "ticks":
        return [(frame_to_ticks(int(f), tick_resolution, display_rate), v) for f, v in frame_keys]
    raise PlanError(f"unknown time base {base!r}")


def compare_keys(expected, actual, value_tol=1e-3, frame_tol=0.01):
    """expected/actual: [(display frame, value), ...]. Returns a small report:
    {"ok", "expected", "actual", "missing", "extra", "max_value_err", "first_bad": [frame, want, got] | None}"""
    exp = sorted((float(f), float(v)) for f, v in expected)
    act = sorted((float(f), float(v)) for f, v in actual)
    j, missing, max_err, first_bad, matched = 0, 0, 0.0, None, 0
    for f, v in exp:
        while j < len(act) and act[j][0] < f - frame_tol:
            j += 1
        if j < len(act) and abs(act[j][0] - f) <= frame_tol:
            e = abs(act[j][1] - v)
            max_err = max(max_err, e)
            if e > value_tol * max(1.0, abs(v)) and first_bad is None:
                first_bad = [f, v, act[j][1]]
            matched += 1
            j += 1
        else:
            missing += 1
            if first_bad is None:
                first_bad = [f, v, None]
    extra = len(act) - matched
    return {"ok": missing == 0 and extra == 0 and first_bad is None, "expected": len(exp), "actual": len(act),
            "missing": missing, "extra": extra, "max_value_err": round(max_err, 6), "first_bad": first_bad}
