"""Pure helpers for the Unreal applier: frame-rate and tick maths, key planning, tagging and read-back comparison.

No `unreal` import here, so everything in this file runs (and is tested) outside the editor. beatrig_apply.py does the
editor calls; this file decides what to write.

Two time bases, and the reason this file exists:
  - Scripting float/double channels (MovieSceneScriptingFloatChannel.add_key, transform channels, section ranges) take
    DISPLAY frames by default (time_unit=DISPLAY_RATE).
  - MovieSceneParameterSection.add_scalar_parameter_key (MPC tracks, component material tracks) takes TICK time:
    display frame x tick_resolution / display_rate. At 30 fps with the default 24000 ticks/s that is 800 ticks per
    frame. Passing display frames there silently keys everything into the first second of the sequence.
  - Reading back is the other way round: channel.get_keys() reports display frames for both.
"""
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

# light target "property" -> the light component's property name/path for a float track
LIGHT_PROPERTIES = {"intensity": "Intensity", "attenuation_radius": "AttenuationRadius",
                    "source_radius": "SourceRadius", "indirect_lighting_intensity": "IndirectLightingIntensity",
                    "volumetric_scattering_intensity": "VolumetricScatteringIntensity", "temperature": "Temperature"}

# transform target "channel" -> index into MovieScene3DTransformSection.get_all_channels()
TRANSFORM_CHANNELS = {"location.x": 0, "location.y": 1, "location.z": 2, "rotation.x": 3, "rotation.y": 4,
                      "rotation.z": 5, "scale.x": 6, "scale.y": 7, "scale.z": 8}


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
    """[[frame, value], ...] -> [(time, value), ...] in the given base ('frames' or 'ticks'), shifted by start_frame"""
    if base == "frames":
        return [(int(f) + int(start_frame), float(v)) for f, v in keys]
    if base == "ticks":
        return [(frame_to_ticks(int(f) + int(start_frame), tick_resolution, display_rate), float(v)) for f, v in keys]
    raise PlanError(f"unknown time base {base!r}")


def plan_output(output, bake_fps, display_rate, tick_resolution, start_frame=0):
    """One bake output -> a list of track writes. Each item:
        {"tag", "kind", "track", "time_base", "actor" | None, "asset" | None, "param" | None, "property" | None,
         "channel_index" | None, "range": [first, last+1] display frames, "keys": [(time, value), ...]}
    Raises PlanError for anything that can't be written."""
    err = check_rate(bake_fps, display_rate)
    if err:
        raise PlanError(err)
    t = output.get("target") or {}
    kind = t.get("kind")
    if kind not in TRACKS:
        raise PlanError(f"{output.get('id')}: unknown target kind {kind!r} (expected one of {sorted(TRACKS)})")
    base, track = TRACKS[kind]
    prop = chan_idx = None
    if kind == "light":
        p = str(t.get("property", "intensity")).lower()
        if p not in LIGHT_PROPERTIES:
            raise PlanError(f"{output.get('id')}: light property {p!r} is not supported (one of {sorted(LIGHT_PROPERTIES)})")
        prop = LIGHT_PROPERTIES[p]
    if kind == "transform":
        c = str(t.get("channel", "")).lower()
        if c not in TRANSFORM_CHANNELS:
            raise PlanError(f"{output.get('id')}: transform channel {c!r} unknown (one of {sorted(TRANSFORM_CHANNELS)})")
        prop, chan_idx = c, TRANSFORM_CHANNELS[c]
    if kind in ("mpc", "material", "niagara") and not t.get("param"):
        raise PlanError(f"{output.get('id')}: {kind} target needs a param")
    plans = []
    for ch in output.get("channels", []):
        keys = ch.get("keys") or []
        if not keys:
            continue
        actor = ch.get("actor")
        if kind != "mpc" and not actor:
            raise PlanError(f"{output.get('id')}: {kind} channel has no actor")
        first, last = int(keys[0][0]) + int(start_frame), int(keys[-1][0]) + int(start_frame)
        plans.append({
            "tag": tag(output["id"], actor), "kind": kind, "track": track, "time_base": base,
            "actor": actor, "asset": t.get("asset"), "param": t.get("param"), "property": prop,
            "channel_index": chan_idx, "material_index": int(t.get("material_index", 0)),
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
