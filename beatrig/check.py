"""Static rig validation: unknown ops/params, missing refs, cycles, channel counts vs targets, colour vs value signals,
groups and scene references, and the timing/easing rules (rule 1: nothing ahead of the audio; rule 2: only trigger
envelopes may move instantly). Needs no audio. Given a bake result it also reports flash-safety violations."""
from typing import Optional

from .ops import SPECS, REQ, COMMON, LFO_SHAPES, MATH, CYCLE_ORDERS, CURVE_PRESETS, input_ids
from .signal import SHAPES

TOP_KEYS = {"audio", "fps", "frames", "audio_offset_frames", "tempo", "seed", "nodes", "layers", "outputs", "bake",
            "channel_weights", "notes", "analysis", "name", "description", "comment",
            "scene", "groups", "safety", "audio_in_s", "audio_out_s", "cache_dir"}
TARGETS = {"light": ("actors",), "material": ("actors", "param"), "niagara": ("actors", "param"), "mpc": ("asset", "param"),
           "transform": ("actors", "property")}
TRANSFORM_AXES = {"location": ["x", "y", "z"], "rotation": ["roll", "pitch", "yaw"], "scale": ["x", "y", "z"]}
COLOR_TARGETS = ("mpc", "light", "material")
EASES = {"in_out", "in", "out", "linear"}
ENUMS = {("lfo", "shape"): LFO_SHAPES, ("envelope", "shape"): SHAPES, ("math", "fn"): MATH, ("cycle", "order"): CYCLE_ORDERS,
         ("merge", "fn"): ["max", "mean"], ("curve", "preset"): list(CURVE_PRESETS),
         ("shuffle_hits", "mode"): ["bag", "random"], ("spatial_delay", "shape"): ["point", "axis", "plane"],
         ("spatial_delay", "direction"): ["out", "in"], ("dome_wave", "mode"): ["zenith_rings", "azimuth_sweep"],
         ("flash_limit", "mode"): ["first", "drop", "delay", "strongest"], ("beats", "velocity"): ["strength", "confidence"],
         ("beats", "anchor"): ["first", "bar"], ("drum", "part"): ["kick", "snare", "hats"],
         ("drum", "source"): ["auto", "stem", "mix"]}
BLENDS = ("replace", "multiply", "add")
TRIGGER_IN = ("envelope", "cycle", "push", "shuffle_hits", "every_n", "probability", "quantize")
SPATIAL = ("spatial_delay", "dome_wave")
SAFETY_KEYS = {"max_flashes_per_s", "min_swing", "outputs"}


class CycleError(Exception):
    pass


def order_nodes(nodes):
    """dependency order (Kahn), ties broken by declaration order so evaluation is deterministic"""
    ids = [n.get("id") for n in nodes]
    deps = {n.get("id"): [d for d in input_ids(n) if d in ids] for n in nodes}
    done, out = set(), []
    while len(out) < len(nodes):
        ready = [n for n in nodes if n.get("id") not in done and all(d in done for d in deps[n.get("id")])]
        if not ready:
            left = [i for i in ids if i not in done]
            raise CycleError(f"dependency cycle among {left}")
        n = ready[0]
        done.add(n.get("id"))
        out.append(n)
    return out


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check(rig, notes: Optional[dict] = None, bake: Optional[dict] = None) -> list:
    """Validate a Rig. Returns [{"level": "error"|"warning", "where": str, "msg": str}]; bake refuses on any error.
    notes: also check note tracks, patterns and section names. bake: a bake result whose values get the flash-safety
    check (errors with a rig "safety" block, warnings without)."""
    d = rig.data
    issues = []
    def err(where, msg):
        issues.append({"level": "error", "where": where, "msg": msg})
    def warn(where, msg):
        issues.append({"level": "warning", "where": where, "msg": msg})

    for k in d:
        if k not in TOP_KEYS and not k.startswith("_"):
            err("rig", f"unknown key {k!r}")
    if not _num(d.get("fps", 30)) or d.get("fps", 30) <= 0:
        err("rig.fps", "fps must be > 0")
    if not isinstance(d.get("audio_offset_frames", 0), int) or d.get("audio_offset_frames", 0) < 0:
        err("rig.audio_offset_frames", "must be an integer >= 0: visuals may not lead the audio (rule 1)")
    if d.get("frames") is not None and (not isinstance(d["frames"], int) or d["frames"] <= 0):
        err("rig.frames", "must be a positive integer or null")
    if d.get("audio") is None and d.get("frames") is None:
        err("rig", "no audio: set frames")
    for k in ("audio_in_s", "audio_out_s"):
        if d.get(k) is not None and (not _num(d[k]) or d[k] < 0):
            err(f"rig.{k}", "must be a number of seconds >= 0")
    if _num(d.get("audio_in_s")) and _num(d.get("audio_out_s")) and d["audio_out_s"] <= d["audio_in_s"]:
        err("rig.audio_out_s", "must be after audio_in_s")
    cw = d.get("channel_weights")
    if cw is not None and (not isinstance(cw, list) or not cw or not all(_num(x) and x >= 0 for x in cw) or not any(cw)):
        err("rig.channel_weights", "must be a list of weights >= 0, one per audio channel (e.g. 5.1: [1, 1, 1, 0.5, 0.7, 0.7])")
    if d.get("cache_dir") is not None and (not isinstance(d["cache_dir"], str) or not d["cache_dir"].strip()):
        err("rig.cache_dir", "must be a folder path (relative to the rig) for the analysis cache")
    an = d.get("analysis")
    if an is not None and not isinstance(an, dict):
        err("rig.analysis", "must be an object of analysis settings")
    elif an:
        if an.get("beats", "auto") not in ("auto", "neural", "classic"):
            err("rig.analysis.beats", "must be \"auto\" (neural when the [ml] extra is installed), \"neural\" or \"classic\"")
        if an.get("stems", "auto") not in ("auto", "on", "off"):
            err("rig.analysis.stems", "must be \"auto\" (separate on a CUDA GPU when the [stems] extra is installed), \"on\" or \"off\"")
    tempo = d.get("tempo") or {}
    if tempo.get("bpm") is not None and (not _num(tempo["bpm"]) or tempo["bpm"] <= 0):
        err("rig.tempo.bpm", "bpm must be > 0")
    for i, m in enumerate(tempo.get("map") or []):
        if not _num(m.get("frame")) or m["frame"] < 0 or not _num(m.get("bpm")) or m["bpm"] <= 0:
            err(f"rig.tempo.map[{i}]", "needs frame >= 0 and bpm > 0")
    tol = (d.get("bake") or {}).get("key_tolerance", 0.002)
    if not _num(tol) or tol < 0:
        err("rig.bake.key_tolerance", "must be >= 0")
    sf0 = (d.get("bake") or {}).get("start_frame")
    if sf0 is not None and (not isinstance(sf0, int) or isinstance(sf0, bool) or sf0 < 0):
        err("rig.bake.start_frame", "must be an integer >= 0 (the first sequence frame baked)")
    sf = d.get("safety")
    if sf is not None:
        if not isinstance(sf, dict):
            err("rig.safety", "must be an object {max_flashes_per_s, min_swing, outputs}")
        else:
            for k in sf:
                if k not in SAFETY_KEYS and not k.startswith("_"):
                    err("rig.safety", f"unknown key {k!r}")
            for k in ("max_flashes_per_s", "min_swing"):
                if k in sf and (not _num(sf[k]) or sf[k] <= 0):
                    err(f"rig.safety.{k}", "must be > 0")
            o = sf.get("outputs", "auto")
            if not (o in ("auto", "all", "none") or (isinstance(o, list) and all(isinstance(x, str) for x in o))):
                err("rig.safety.outputs", "must be \"auto\", \"all\", \"none\" or a list of output ids")

    groups = d.get("groups")
    if groups is not None and (not isinstance(groups, dict)
                               or not all(isinstance(v, list) and all(isinstance(a, str) for a in v) for v in groups.values())):
        err("rig.groups", "must be {name: [actor, ...]}")
        groups = {}
    groups = groups or {}

    def unknown_groups(actors):
        items = [actors] if isinstance(actors, str) else (actors or [])
        return [a for a in items if isinstance(a, str) and a.startswith("@") and a[1:] not in groups]

    scene = None
    try:
        scene = rig.load_scene()
    except Exception as e:                                         # RigError: unreadable or missing scene.json
        err("rig.scene", str(e))
    scene_actors = (scene or {}).get("actors") or {}

    raw_nodes = d.get("nodes", [])
    nodes = rig.resolved_nodes() if hasattr(rig, "resolved_nodes") else raw_nodes
    ids = [n.get("id") for n in raw_nodes]
    by_id = {}
    for i, n in enumerate(nodes):
        nid = n.get("id")
        where = f"node {nid or '#' + str(i)}"
        if not isinstance(nid, str) or not nid:
            err(where, "missing id")
            continue
        if nid in by_id:
            err(where, "duplicate id")
        by_id[nid] = n
        spec = SPECS.get(n.get("op"))
        if spec is None:
            err(where, f"unknown op {n.get('op')!r}")
            continue
        for k in n:
            if k not in COMMON and k not in spec.params and not k.startswith("_"):
                err(where, f"unknown param {k!r} for op {n['op']}")
        for k, dv in spec.params.items():
            if dv is REQ and n.get(k) is None:
                err(where, f"missing required param {k!r}")
        for group in spec.one_of:
            got = [g for g in group if n.get(g) is not None]
            if n["op"] == "delay" and group == ("ms", "beats") and not got:
                err(where, "needs ms or beats")
            if n["op"] in ("chase", "spatial_delay", "dome_wave") and not got:
                err(where, f"needs {group[0]} or {group[1]}")
            if len(got) > 1:
                err(where, f"give one of {list(group)}, not both")
        for (o, k), allowed in ENUMS.items():
            if n["op"] == o and n.get(k) is not None and n[k] not in allowed:
                err(where, f"{k}={n[k]!r} not in {allowed}")
        if spec.inputs in ("one", "many") and n.get("in") is None:
            err(where, "missing input 'in'")
        if spec.inputs == "one" and isinstance(n.get("in"), list):
            err(where, "takes a single input, not a list")
        if spec.inputs == "many" and (not isinstance(n.get("in"), list) or len(n["in"]) < 2):
            err(where, "'in' must be a list of at least 2 node ids")
        for ref in input_ids(n):
            if ref not in ids:
                err(where, f"input {ref!r} does not exist")
        raw = next((r for r in raw_nodes if r.get("id") == nid), n)
        bad = unknown_groups(raw.get("actors"))
        if bad:
            err(where, f"unknown group(s) {bad} (define them in rig.groups)")
        # rule 1: no negative delays
        for k in ("ms", "beats", "step_ms", "step_beats", "speed_cm_s", "beats_per_m", "deg_per_s", "deg_per_beat"):
            if n["op"] in ("delay", "chase", "ripple") + SPATIAL and n.get(k) is not None and (not _num(n[k]) or n[k] < 0):
                err(where, f"{k} must be >= 0 (rule 1: nothing ahead of the audio; use direction for reverse chases)")
        for k in ("speed_cm_s", "deg_per_s"):
            if n.get(k) == 0 and n["op"] in SPATIAL:
                err(where, f"{k} must be > 0")
        # rule 2: eased motion
        if n["op"] == "smooth":
            for k in ("attack_ms", "release_ms"):
                if n.get(k) is not None and (not _num(n[k]) or n[k] <= 0):
                    err(where, f"{k} must be > 0: only trigger envelopes may move instantly (rule 2)")
        if n["op"] == "spring" and (P_ok(n, "freq_hz") is False or P_ok(n, "damping") is False):
            err(where, "freq_hz and damping must be > 0")
        if n["op"] == "lfo":
            if n.get("rate_beats") is not None and n.get("rate_hz") is not None:
                err(where, "give rate_beats or rate_hz, not both")
            for k in ("rate_beats", "rate_hz"):
                if n.get(k) is not None and (not _num(n[k]) or n[k] <= 0):
                    err(where, f"{k} must be > 0")
            if n.get("shape") == "square":
                warn(where, "square LFO switches instantly; smooth it or use tri/sine (rule 2)")
        if n["op"] == "curve" and n.get("gamma") is not None and (not _num(n["gamma"]) or n["gamma"] <= 0):
            err(where, "gamma must be > 0")
        if n["op"] == "envelope" and n.get("shape") == "custom" and not n.get("points"):
            err(where, "custom shape needs points")
        if n["op"] in ("lfo", "const", "fanout", "chase", "cycle", "push", "shuffle_hits", "palette_cycle") and n.get("channels") is not None \
                and (not isinstance(n["channels"], int) or n["channels"] < 1):
            err(where, "channels must be an integer >= 1")
        if n["op"] == "notes" and notes is not None and n.get("track") not in (notes.get("tracks") or {}):
            warn(where, f"track {n.get('track')!r} not in notes.json (run notes init)")
        if n["op"] == "onsets":
            ref = n.get("in")
            if isinstance(ref, str) and ref in by_id and by_id[ref].get("op") != "band":
                err(where, f"in={ref!r} must be a band node or a {{lo, hi}} spec")
            elif isinstance(ref, str) and ref not in ids:
                err(where, f"input {ref!r} does not exist")
        if n.get("stem") is not None and n.get("stem") not in ("drums", "bass", "vocals", "other"):
            err(where, f"stem {n['stem']!r} not in ['drums', 'bass', 'vocals', 'other']")
        _check_v02_node(n, where, err, warn, scene, scene_actors)

    # cycles + static channel / kind inference
    chans, kinds = {}, {}
    try:
        ordered = order_nodes([n for n in nodes if n.get("id")])
    except CycleError as e:
        err("nodes", str(e))
        ordered = []
    for n in ordered:
        spec = SPECS.get(n.get("op"))
        if spec is None:
            continue
        ins = [i for i in input_ids(n) if i in chans]
        if spec.inputs == "ref":
            ins = []
        if len(ins) != len(input_ids(n)) and spec.inputs != "ref":
            continue
        c_in, k_in = [chans[i] for i in ins], [kinds[i] for i in ins]
        where = f"node {n['id']}"
        try:
            chans[n["id"]] = int(spec.chans(n, c_in))
            kinds[n["id"]] = spec.kind(n, k_in) if callable(spec.kind) else spec.kind
        except (TypeError, ValueError, KeyError):
            continue
        op = n["op"]
        if "color" in k_in and not spec.color:
            err(where, f"{op} does not take colour signals ({[i for i, k in zip(ins, k_in) if k == 'color']} are colour); "
                       "use scale_color / mix_color, or gradient after the value ops")
        if op in ("mix_color", "scale_color") and k_in:
            want = ["color", "color"] + (["value"] if len(k_in) == 3 else []) if op == "mix_color" else ["color", "value"]
            if op == "mix_color" and len(k_in) not in (2, 3):
                err(where, "mix_color takes in = [a, b] or [a, b, t]")
            elif op == "scale_color" and len(k_in) != 2:
                err(where, "scale_color takes in = [colour, value]")
            elif [k == "color" for k in k_in] != [w == "color" for w in want]:
                err(where, f"{op} inputs must be {want} signals, got {k_in}")
        if op == "gradient" and k_in and k_in[0] == "color":
            err(where, "gradient takes a value signal, not a colour")
        if op in TRIGGER_IN and k_in and k_in[0] != "trigger":
            err(where, f"{op} needs a trigger input; put a threshold op in front of {ins[0]!r}")
        if op == "math" and c_in and any(c not in (1, max(c_in)) for c in c_in):
            err(where, f"channel counts {c_in} don't broadcast (each must be 1 or {max(c_in)})")
        if op in ("mix_color", "scale_color") and c_in and any(c not in (1, max(c_in)) for c in c_in):
            err(where, f"channel counts {c_in} don't broadcast (each must be 1 or {max(c_in)})")
        if op == "chase" and c_in and c_in[0] not in (1, chans[n["id"]]):
            err(where, f"input has {c_in[0]} channels, expected 1 or {chans[n['id']]}")
        if op in SPATIAL and c_in and c_in[0] not in (1, chans[n["id"]]):
            err(where, f"input has {c_in[0]} channels, expected 1 or {chans[n['id']]} (one per actor)")
        if op == "ripple" and n.get("neighbours") == "knn" and c_in and n.get("actors") and c_in[0] != len(n["actors"]):
            err(where, f"ripple knn: input has {c_in[0]} channels but {len(n['actors'])} actors")
        if op == "fanout" and c_in and c_in[0] != 1:
            err(where, f"fanout needs a 1-channel input, got {c_in[0]}")
        if op == "select" and c_in and any(not isinstance(i, int) or not 0 <= i < c_in[0] for i in n.get("channels") or []):
            err(where, f"select {n.get('channels')} out of range for {c_in[0]} channels")
        if op == "push" and isinstance(n.get("pattern"), list) and isinstance(n.get("channels"), int):
            C = n["channels"]
            flat = [c for st in n["pattern"] for c in ([st] if isinstance(st, int) else (st if isinstance(st, list) else [None]))]
            if not n["pattern"] or any(not isinstance(c, int) or not 0 <= c < C for c in flat):
                err(where, f"pattern steps must be channels 0..{C - 1} (ints or lists of ints)")

    sec_names = {str(x.get("name")) for x in (notes or {}).get("sections") or []} if notes is not None else None
    layers = d.get("layers", [])
    lids = set()
    for i, L in enumerate(layers):
        lid = L.get("id")
        where = f"layer {lid or '#' + str(i)}"
        if not lid or lid in lids:
            err(where, "missing or duplicate id")
        lids.add(lid)
        keys = L.get("intensity", [])
        if not keys:
            warn(where, "no intensity keys (treated as 1)")
        for k in keys:
            if k.get("section") is not None and k.get("frame") is None:
                if sec_names is not None and str(k["section"]) not in sec_names:
                    err(where, f"key names section {k['section']!r}, which is not in notes.json sections")
            elif not _num(k.get("frame")):
                err(where, "every intensity key needs a numeric frame or a section name")
        frames = [k.get("frame") for k in keys if k.get("section") is None or k.get("frame") is not None]
        if all(_num(f) for f in frames) and frames != sorted(frames):
            err(where, "intensity keys need frames in ascending order")
        for j, k in enumerate(keys):
            if k.get("ease") is not None and k["ease"] not in EASES:
                err(where, f"ease {k['ease']!r} not in {sorted(EASES)}")
            if j > 0 and not k.get("fade_frames"):
                warn(where, f"key at {k.get('section') or k.get('frame')} has no fade_frames: an instant switch (rule 2)")

    seen_targets = {}
    oids = set()
    for i, o in enumerate(d.get("outputs", [])):
        oid = o.get("id")
        where = f"output {oid or '#' + str(i)}"
        if not oid or oid in oids:
            err(where, "missing or duplicate id")
        oids.add(oid)
        sig = o.get("signal")
        if sig not in ids:
            err(where, f"signal {sig!r} does not exist")
        if o.get("layer") is not None and o["layer"] not in lids:
            err(where, f"layer {o['layer']!r} does not exist")
        t = o.get("target") or {}
        kind = t.get("kind")
        if kind not in TARGETS:
            err(where, f"target kind {kind!r} not in {sorted(TARGETS)}")
            continue
        for k in TARGETS[kind]:
            if not t.get(k):
                err(where, f"target needs {k!r}")
        for k in ("min", "max"):
            if o.get(k) is not None and not _num(o[k]):
                err(where, f"{k} must be a number")
        bad = unknown_groups(t.get("actors"))
        if bad:
            err(where, f"unknown group(s) {bad} (define them in rig.groups)")
        actors = rig.expand_actors(t.get("actors")) if hasattr(rig, "expand_actors") else list(t.get("actors") or [])
        c, sk = chans.get(sig), kinds.get(sig)
        wants_color = o.get("value_type") == "color" or t.get("value_type") == "color" or \
            (kind == "light" and str(t.get("property", "")).lower() == "color")
        for vt in (o.get("value_type"), t.get("value_type")):
            if vt is not None and vt not in ("float", "color"):
                err(where, f"value_type {vt!r} not in ['float', 'color']")
        if sk == "color":
            if kind not in COLOR_TARGETS:
                err(where, f"signal {sig!r} is a colour; {kind} targets take values (colour goes to mpc, light or material)")
            elif kind == "light" and str(t.get("property", "intensity")).lower() != "color":
                err(where, f"signal {sig!r} is a colour: set the light target's property to \"color\"")
            if o.get("min") is not None or o.get("max") is not None:
                warn(where, "min/max are ignored for colour outputs (scale brightness in the graph: scale_color)")
        elif wants_color and sk is not None:
            err(where, f"a colour target needs a colour signal; {sig!r} is a {sk} (use gradient or palette_cycle)")
        if kind == "transform":
            prop, axis = t.get("property"), t.get("axis")
            if prop not in TRANSFORM_AXES:
                err(where, f"transform property {prop!r} not in {sorted(TRANSFORM_AXES)}")
            elif axis not in TRANSFORM_AXES[prop]:
                err(where, f"transform axis {axis!r} for {prop} not in {TRANSFORM_AXES[prop]}")
            if t.get("additive") is not None and not isinstance(t["additive"], bool):
                err(where, "additive must be true or false")
        _check_blend(o, where, kind, sk == "color" or wants_color, err, warn)
        if kind == "mpc":
            if c is not None and c != 1:
                err(where, f"signal {sig!r} has {c} channels; an MPC param takes 1 (merge or select first)")
            tk = [("mpc", t.get("asset"), t.get("param"))]
        else:
            if c is not None and actors and c not in (1, len(actors)):
                err(where, f"signal {sig!r} has {c} channels but the target has {len(actors)} actors (need 1 or {len(actors)})")
            prop = t.get("param") or t.get("property", "intensity")
            if kind == "transform":
                prop = f"{t.get('property')}.{t.get('axis')}"
            tk = [(kind, a, prop) for a in actors]
        for key in tk:
            if key in seen_targets:
                warn(where, f"{key[1]}.{key[2]} is also driven by output {seen_targets[key]!r}: the applier would write two tracks")
            seen_targets.setdefault(key, oid)
        if scene is not None:
            _check_scene_target(where, kind, t, actors, sk == "color" or wants_color, scene, warn)

    if notes is not None:
        pats = notes.get("patterns") or {}
        for name, items in (notes.get("tracks") or {}).items():
            for h in items:
                if isinstance(h, dict) and "pattern" in h and "time" not in h and h["pattern"] not in pats:
                    warn(f"notes track {name}", f"placement of unknown pattern {h['pattern']!r}")
    if bake is not None:
        from .safety import check_bake
        issues.extend(check_bake(d, bake)[0])
    return issues


def _check_blend(o, where, kind, color, err, warn):
    """output.blend: replace (default) | multiply (values are gains on the target's own animation) | add (offsets)"""
    blend = o.get("blend", "replace")
    if blend is None or blend == "replace":
        return
    if blend not in BLENDS:
        err(where, f"blend {blend!r} not in {list(BLENDS)}")
        return
    if kind == "transform":
        err(where, f"blend {blend!r} does not apply to transform outputs: they blend through target.additive "
                   "(true, the default, = an offset on the actor's own animation)")
        return
    if color:                                       # colour gains / offsets are per RGBA component; no min/max
        return
    what = ("gain range (1 = the base unchanged), e.g. 0.6 and 1.5" if blend == "multiply"
            else "offset range in the target's units (0 = the base unchanged), e.g. -0.2 and 0.5")
    missing = [k for k in ("min", "max") if o.get(k) is None]
    if missing:
        err(where, f"blend {blend!r} needs min and max: the {what} (the 0..1 default would be taken literally)")
    elif blend == "multiply" and _num(o["min"]) and _num(o["max"]) and min(o["min"], o["max"]) < 0:
        warn(where, f"blend multiply with a negative gain ({o['min']}..{o['max']}) flips the base's sign")


def _check_v02_node(n, where, err, warn, scene, scene_actors):
    """params of the colour / flow / spatial / safety ops that need more than an enum"""
    op = n["op"]
    if op in ("gradient", "palette_cycle"):
        from .color import check_color_node
        for e in check_color_node(n):
            err(where, e)
    if op == "palette_cycle" and n.get("period_beats") is None and n.get("period_s") is None:
        err(where, "needs period_beats or period_s")
    for k in ("period_beats", "period_s"):
        if op == "palette_cycle" and n.get(k) is not None and (not _num(n[k]) or n[k] <= 0):
            err(where, f"{k} must be > 0")
    if op == "push" and isinstance(n.get("pattern"), str):
        from .flow import PUSH_PATTERNS
        if n["pattern"] not in PUSH_PATTERNS:
            err(where, f"pattern {n['pattern']!r} not in {PUSH_PATTERNS} (or give a list of steps)")
    if op == "every_n" and n.get("n") is not None and (not isinstance(n["n"], int) or n["n"] < 1):
        err(where, "n must be an integer >= 1")
    if op == "probability" and n.get("p") is not None and (not _num(n["p"]) or not 0 <= n["p"] <= 1):
        err(where, "p must be within 0..1")
    if op == "quantize":
        from .notes import parse_grid
        try:
            if parse_grid(n.get("grid", "1/16")) <= 0:
                raise ValueError
        except (ValueError, ZeroDivisionError):
            err(where, f"grid {n.get('grid')!r} is not a grid like '1/16' or a number of beats")
    if op == "accumulate" and n.get("wrap"):
        warn(where, "wrap jumps back instantly at every wrap (rule 2); fine for a shader phase, not for a transform")
    if op == "beats" and n.get("min_confidence") is not None and (not _num(n["min_confidence"]) or not 0 <= n["min_confidence"] <= 1):
        err(where, "min_confidence must be within 0..1")
    if op == "flash_limit":
        for k in ("max_flashes_per_s", "min_swing"):
            if n.get(k) is not None and (not _num(n[k]) or n[k] <= 0):
                err(where, f"{k} must be > 0")
        if _num(n.get("max_flashes_per_s")) and 0 < n["max_flashes_per_s"] < 1:
            warn(where, "max_flashes_per_s below 1: no flash fits the 1 s window rule, so triggers pass nothing")
    if op == "ripple":
        nb = n.get("neighbours", 2)
        if nb == "knn":
            if not n.get("actors"):
                err(where, "neighbours \"knn\" needs actors (their scene positions define the neighbours)")
            for k in ("k", "hops"):
                if n.get(k) is not None and (not isinstance(n[k], int) or n[k] < 1):
                    err(where, f"{k} must be an integer >= 1")
        elif not isinstance(nb, int) or nb < 0:
            err(where, "neighbours must be an integer >= 0 or \"knn\"")
    if op in SPATIAL or (op == "ripple" and n.get("neighbours") == "knn"):
        if scene is None:
            err(where, f"{op} needs real positions: set the rig's \"scene\" to a scene.json")
        else:
            missing = [a for a in n.get("actors") or [] if a not in scene_actors or not scene_actors[a].get("location")]
            if missing:
                err(where, f"actors not in the scene (or without a location): {missing[:6]}")
            if op == "spatial_delay" and isinstance(n.get("origin"), str) and n["origin"] not in scene_actors:
                err(where, f"origin actor {n['origin']!r} is not in the scene")
    if op == "spatial_delay" and n.get("origin") is not None and not isinstance(n["origin"], str) \
            and not (isinstance(n["origin"], list) and len(n["origin"]) == 3 and all(_num(x) for x in n["origin"])):
        err(where, "origin must be an actor name or [x, y, z] (cm)")
    if op == "dome_wave":
        cam = n.get("camera") or (scene or {}).get("camera")
        if n.get("direction") is not None:
            ok = ["out", "in"] if n.get("mode", "zenith_rings") == "zenith_rings" else ["cw", "ccw"]
            if n["direction"] not in ok:
                err(where, f"direction {n['direction']!r} not in {ok} for mode {n.get('mode', 'zenith_rings')}")
        if not cam:
            err(where, "needs a dome camera: scene.json \"camera\" or the node's \"camera\" {location, rotation}")
        elif scene_actors:
            from .spatial import outside_dome
            out = outside_dome(cam, scene_actors, n.get("actors") or [])
            if out:
                warn(where, f"{len(out)} actor(s) are outside the dome's field of view, e.g. {out[:3]} (zenith deg)")


def _check_scene_target(where, kind, t, actors, color, scene, warn):
    """warnings for names the scene scan doesn't know: actors, light components, material params, MPCs"""
    sa = scene.get("actors") or {}
    if kind == "mpc":
        mpcs = scene.get("mpcs") or {}
        asset = t.get("asset")
        if mpcs and asset not in mpcs:
            warn(where, f"MPC {asset!r} is not in the scene scan")
        elif asset in mpcs:
            names = mpcs[asset].get("vectors" if color else "scalars") or []
            if t.get("param") not in names:
                warn(where, f"MPC {asset} has no {'vector' if color else 'scalar'} param {t.get('param')!r} (scan lists {names[:8]})")
        return
    missing = [a for a in actors if a not in sa]
    if missing:
        warn(where, f"{len(missing)} actor(s) not in the scene scan, e.g. {missing[:4]}")
    for a in actors:
        comps = (sa.get(a) or {}).get("components") or []
        if a not in sa or not comps:
            continue
        if kind == "light" and not any(c.get("kind") == "light" for c in comps):
            warn(where, f"{a} has no light component in the scan")
            break
        if kind == "material":
            names = {p for c in comps for m in c.get("materials") or [] for p in (m.get("vectors" if color else "scalars") or [])}
            if names and t.get("param") not in names:
                warn(where, f"{a} has no material {'vector' if color else 'scalar'} param {t.get('param')!r} in the scan")
                break


def P_ok(n, k):
    v = n.get(k)
    if v is None:
        return None
    return _num(v) and v > 0
