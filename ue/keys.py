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
"""
import hashlib
import os
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
         "material_index", "range": [first, last+1] display frames, "keys": [(time, value | (r, g, b, a)), ...]}
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
        plans.append({
            "tag": tag(oid, actor), "output": oid, "kind": kind, "value_type": vt, "track": track, "time_base": base,
            "actor": actor, "asset": t.get("asset"), "param": param, "property": prop,
            "channel_index": chan_idx, "channel_name": chan_name, "blend": blend,
            "material_index": int(t.get("material_index", 0)),
            "range": [first, last + 1],
            "keys": to_time_base(keys, base, start_frame, tick_resolution, display_rate),
        })
    return plans


def plan_bake(bake, display_rate, tick_resolution, start_frame=0, only=None):
    """every output in a bake.json dict (or just the ids in `only`) -> track writes"""
    if int(bake.get("version", 0)) != 1:
        raise PlanError(f"unsupported bake version {bake.get('version')!r}")
    out = []
    for o in bake.get("outputs", []):
        if only and o.get("id") not in only:
            continue
        out.extend(plan_output(o, bake.get("fps", 30), display_rate, tick_resolution, start_frame))
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
