"""Static rig validation: unknown ops/params, missing refs, cycles, channel counts vs targets, and the timing/easing rules
(rule 1: nothing ahead of the audio; rule 2: only trigger envelopes may move instantly). Needs no audio."""
from typing import Optional

from .ops import SPECS, REQ, COMMON, LFO_SHAPES, MATH, CYCLE_ORDERS, CURVE_PRESETS, input_ids
from .signal import SHAPES

TOP_KEYS = {"audio", "fps", "frames", "audio_offset_frames", "tempo", "seed", "nodes", "layers", "outputs", "bake",
            "channel_weights", "notes", "analysis", "name", "description", "comment"}
TARGETS = {"light": ("actors",), "material": ("actors", "param"), "niagara": ("actors", "param"), "mpc": ("asset", "param")}
EASES = {"in_out", "in", "out", "linear"}
ENUMS = {("lfo", "shape"): LFO_SHAPES, ("envelope", "shape"): SHAPES, ("math", "fn"): MATH, ("cycle", "order"): CYCLE_ORDERS,
         ("merge", "fn"): ["max", "mean"], ("curve", "preset"): list(CURVE_PRESETS)}


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


def check(rig, notes: Optional[dict] = None) -> list:
    """Validate a Rig. Returns [{"level": "error"|"warning", "where": str, "msg": str}]; bake refuses on any error."""
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
    tempo = d.get("tempo") or {}
    if tempo.get("bpm") is not None and (not _num(tempo["bpm"]) or tempo["bpm"] <= 0):
        err("rig.tempo.bpm", "bpm must be > 0")
    for i, m in enumerate(tempo.get("map") or []):
        if not _num(m.get("frame")) or m["frame"] < 0 or not _num(m.get("bpm")) or m["bpm"] <= 0:
            err(f"rig.tempo.map[{i}]", "needs frame >= 0 and bpm > 0")
    tol = (d.get("bake") or {}).get("key_tolerance", 0.002)
    if not _num(tol) or tol < 0:
        err("rig.bake.key_tolerance", "must be >= 0")

    nodes = d.get("nodes", [])
    ids = [n.get("id") for n in nodes]
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
            if n["op"] == "chase" and not got:
                err(where, "needs step_ms or step_beats")
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
        # rule 1: no negative delays
        for k in ("ms", "beats", "step_ms", "step_beats"):
            if n["op"] in ("delay", "chase", "ripple") and n.get(k) is not None and (not _num(n[k]) or n[k] < 0):
                err(where, f"{k} must be >= 0 (rule 1: nothing ahead of the audio; use direction for reverse chases)")
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
        for k in ("channels",):
            if n["op"] in ("lfo", "const", "fanout", "chase", "cycle") and n.get(k) is not None and (not isinstance(n[k], int) or n[k] < 1):
                err(where, "channels must be an integer >= 1")
        if n["op"] == "notes" and notes is not None and n.get("track") not in (notes.get("tracks") or {}):
            warn(where, f"track {n.get('track')!r} not in notes.json (run notes init)")
        if n["op"] == "onsets":
            ref = n.get("in")
            if isinstance(ref, str) and ref in by_id and by_id[ref].get("op") != "band":
                err(where, f"in={ref!r} must be a band node or a {{lo, hi}} spec")
            elif isinstance(ref, str) and ref not in ids:
                err(where, f"input {ref!r} does not exist")

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
        if n["op"] in ("envelope", "cycle") and k_in and k_in[0] != "trigger":
            err(where, f"{n['op']} needs a trigger input; put a threshold op in front of {ins[0]!r}")
        if n["op"] == "math" and c_in and any(c not in (1, max(c_in)) for c in c_in):
            err(where, f"channel counts {c_in} don't broadcast (each must be 1 or {max(c_in)})")
        if n["op"] == "chase" and c_in and c_in[0] not in (1, chans[n["id"]]):
            err(where, f"input has {c_in[0]} channels, expected 1 or {chans[n['id']]}")
        if n["op"] == "fanout" and c_in and c_in[0] != 1:
            err(where, f"fanout needs a 1-channel input, got {c_in[0]}")
        if n["op"] == "select" and c_in and any(not isinstance(i, int) or not 0 <= i < c_in[0] for i in n.get("channels") or []):
            err(where, f"select {n.get('channels')} out of range for {c_in[0]} channels")

    layers = d.get("layers", [])
    lids = set()
    for i, L in enumerate(layers):
        lid = L.get("id")
        where = f"layer {lid or '#' + str(i)}"
        if not lid or lid in lids:
            err(where, "missing or duplicate id")
        lids.add(lid)
        keys = L.get("intensity", [])
        frames = [k.get("frame") for k in keys]
        if not keys:
            warn(where, "no intensity keys (treated as 1)")
        if any(not _num(f) for f in frames) or frames != sorted(frames):
            err(where, "intensity keys need numeric frames in ascending order")
        for j, k in enumerate(keys):
            if k.get("ease") is not None and k["ease"] not in EASES:
                err(where, f"ease {k['ease']!r} not in {sorted(EASES)}")
            if j > 0 and not k.get("fade_frames"):
                warn(where, f"key at frame {k.get('frame')} has no fade_frames: an instant switch (rule 2)")

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
        c = chans.get(sig)
        if kind == "mpc":
            if c is not None and c != 1:
                err(where, f"signal {sig!r} has {c} channels; an MPC param takes 1 (merge or select first)")
            tk = [("mpc", t.get("asset"), t.get("param"))]
        else:
            actors = t.get("actors") or []
            if c is not None and actors and c not in (1, len(actors)):
                err(where, f"signal {sig!r} has {c} channels but the target has {len(actors)} actors (need 1 or {len(actors)})")
            prop = t.get("param") or t.get("property", "intensity")
            tk = [(kind, a, prop) for a in actors]
        for key in tk:
            if key in seen_targets:
                warn(where, f"{key[1]}.{key[2]} is also driven by output {seen_targets[key]!r}: the applier would write two tracks")
            seen_targets.setdefault(key, oid)
    return issues


def P_ok(n, k):
    v = n.get(k)
    if v is None:
        return None
    return _num(v) and v > 0
