"""beatrig Unreal applier: writes a bake.json into a Level Sequence as plain Sequencer keys. Runs inside the UE 5.x editor.

Tested in UE 5.8.3 (see docs/unreal.md for what was run). The pure half (tick maths, key planning, tags) lives in keys.py
and is unit tested; this file only makes the editor calls.

Two ways in:
  1. As a module, kept loaded by beatrig.live.Link: `rpc(payload_path)` reads {"op", "args"} from a JSON file, runs
     apply / check / undo / push / scan / playhead_get / playhead_set / ensure_sequence / throttle / hello, and writes
     the result to <payload_path>.out. Python callers can use apply(), check(), undo() etc. directly.
  2. As a script (editor console, or Remote Control ExecuteFile):
        beatrig_apply.py apply <sequence asset path> <bake.json> [--start N] [--only id,id] [--save]
        beatrig_apply.py undo  <sequence asset path> [output id] [--save]
        beatrig_apply.py check <sequence asset path> <bake.json> [--start N] [--only id,id]
     The last printed line is BEATRIG_RESULT {json}.

What it writes, per bake output target kind (value_type float unless noted):
    mpc        sequence-level MovieSceneMaterialParameterCollectionTrack; scalar param, or vector param for colour (TICKS)
    light      on the light component: MovieSceneFloatTrack (Intensity...) or, for colour, MovieSceneColorTrack
               (LightColor), keys in display frames
    material   MovieSceneComponentMaterialTrack on the actor's first primitive component; scalar or vector param (TICKS)
    niagara    MovieSceneNiagaraFloatParameterTrack (User.<param>) on the Niagara component. Python can't set the track's
               parameter, so the track is created through LevelSequenceEditorSubsystem.paste_tracks (no clipboard)
    transform  MovieScene3DTransformTrack on the actor binding, one channel, additive blend by default (display frames)
Every key is linear, so curves can't overshoot. Every track is named "beatrig:<output id>[:<actor>]"; `undo` removes
exactly those tracks and nothing else. Bindings the applier creates carry the binding tag "beatrig"; once their tracks
are gone they are removed too, and bindings it found already in the sequence are never removed. `apply` first removes
the tracks it is about to replace, so it is idempotent. Each call is one editor transaction (Ctrl+Z undoes it).

Output blend (v0.3, output "blend": "multiply" | "add"; the bake's values are gains or offsets). The base is the first
active non-beatrig section that drives the same target; it is only read, never changed, so a section shared with other
parameters is fine. Over it the beatrig track is an ADDITIVE section (blend type set before any key) holding the offset:
base x (gain - 1) for multiply, from the base evaluated at every frame, or the offsets for add, with 0 one frame each
side of the bake range so the base plays alone outside it. With no base section, the track is an ABSOLUTE section of
static x gain (static + offset) over the bake range, the static value (MPC default, property value) read at the first
apply. What each track was applied over is recorded in the sequence asset's metadata ("beatrig.base:<track tag>"):
`check` recomputes from the current base, `push` rewrites outputs whose base was edited, `undo` deletes the track and
the record. One beatrig blend per target.
"""
import importlib.util
import json
import os
import re
import sys
import time
import traceback
import uuid

import unreal

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(mod_name, file_name):
    """load a module next to this file under a beatrig-specific name (a bare `keys` could collide in a shared editor)"""
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(HERE, file_name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


K = _load_sibling("beatrig_keys", "keys.py")
SCAN = _load_sibling("beatrig_scan", "beatrig_scan.py")
SRC_HASH = K.source_hash(HERE)
RESULT_PREFIX = "BEATRIG_RESULT "
OWNER_TAG = "beatrig"                       # binding tag on every binding the applier created
_PUSHED = {}                                # (sequence package, output id) -> content hash of what push last wrote
_CACHE = {"actors_key": None, "actors": {}, "niagara_float": None}
_LIN = unreal.MovieSceneKeyInterpolation.LINEAR

# Sequencer's own copy format for one track. Only the parameter is set here; sections and keys are added afterwards
# through the normal scripting API. RegisteredTypeIndex is the editor session's index for the Niagara float type.
_NIAGARA_TRACK_T3D = """Begin Object Class=/Script/Sequencer.MovieSceneCopyableTrack Name="{copy}"
   Begin Object Class=/Script/Niagara.MovieSceneNiagaraFloatParameterTrack Name="{track}"
   End Object
   Begin Object Name="{track}"
      Parameter=(Name="{param}",TypeDefHandle=(RegisteredTypeIndex={index}))
   End Object
   Track="{track}"
   bIsRootTrack=False
End Object
"""


# ------------------------------------------------------------------ editor helpers (API differences across 5.x)
def _lib():
    return unreal.LevelSequenceEditorBlueprintLibrary


def _rate(fr):
    """unreal.FrameRate -> Fraction"""
    return K.rate((fr.numerator, fr.denominator))


def _pkg(obj):
    """asset object -> package path (/Game/Show/LS_Main)"""
    return obj.get_path_name().split(".")[0]


def _seq_tracks(seq):
    fn = getattr(seq, "get_tracks", None) or getattr(seq, "get_master_tracks")
    return list(fn())


def _seq_remove_track(seq, track):
    fn = getattr(seq, "remove_track", None) or getattr(seq, "remove_master_track")
    return fn(track)


def _binding_key(b):
    """a binding's guid as text. GOTCHA: str(Guid) is the wrapper's repr with a memory address, different on every
    call, so it can't be used to de-duplicate (component bindings come back from get_bindings() and as children)."""
    g = b.get_id()
    return g.to_string() if hasattr(g, "to_string") else unreal.GuidLibrary.conv_guid_to_string(g)


def _all_bindings(seq):
    """every binding once, including component bindings under actors"""
    out, seen = [], set()
    stack = list(seq.get_bindings())
    while stack:
        b = stack.pop(0)
        key = _binding_key(b)
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
        kids = getattr(b, "get_child_possessables", None)
        if kids:
            stack.extend(kids())
    return out


def _all_tracks(seq):
    """[(track, binding or None)] for root tracks and every binding's tracks"""
    out = [(t, None) for t in _seq_tracks(seq)]
    for b in _all_bindings(seq):
        out.extend((t, b) for t in b.get_tracks())
    return out


def _track_name(track):
    try:
        return str(track.get_display_name())
    except Exception:
        return ""


def _set_name(track, name):
    if not hasattr(track, "set_display_name"):
        raise RuntimeError(f"{type(track).__name__} cannot be named; beatrig cannot tag it for undo")
    track.set_display_name(name)


def _alive(actor, name):
    try:
        return actor is not None and name in (actor.get_actor_label(), actor.get_name())
    except Exception:
        return False


def _find_actor(name):
    """actor in the open level by label, then by object name. The label map is cached per world + actor count and
    rebuilt when the name is missing or a cached actor has gone stale (deleted, relabelled)."""
    acts = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    key = (world.get_path_name() if world else "", len(acts))
    a = _CACHE["actors"].get(name) if _CACHE["actors_key"] == key else None
    if not _alive(a, name):
        m = {}
        for x in acts:
            m.setdefault(x.get_actor_label(), x)
            m.setdefault(x.get_name(), x)
        _CACHE["actors_key"], _CACHE["actors"] = key, m
        a = m.get(name)
        if not _alive(a, name):
            raise RuntimeError(f"actor not found in the open level: {name}")
    return a


def _owned(binding):
    try:
        return OWNER_TAG in [str(t) for t in binding.get_binding_tags()]
    except Exception:
        return False


def _claim(binding, ctx):
    binding.tag_binding(OWNER_TAG)
    ctx["created"] += 1
    return binding


def _actor_binding(seq, actor, ctx):
    label = actor.get_actor_label()
    for b in seq.get_bindings():
        if str(b.get_name()) == label and not (hasattr(b, "get_parent") and b.get_parent().is_valid()):
            return b
    return _claim(seq.add_possessable(actor), ctx)


def _component_binding(seq, actor, component, ctx):
    """component bindings need their actor bound first (seen in practice on 5.x)"""
    parent = _actor_binding(seq, actor, ctx)
    kids = getattr(parent, "get_child_possessables", None)
    if kids:
        for b in kids():
            if str(b.get_name()) == component.get_name():
                return b
    b = seq.add_possessable(component)
    if hasattr(b, "set_parent"):
        try:
            if not b.get_parent().is_valid():
                b.set_parent(parent)
        except Exception:
            pass
    return _claim(b, ctx)


def _prune_owned(seq):
    """remove bindings the applier created that hold no tracks and no child bindings (components before actors).
    Bindings without the tag are never touched. Returns the removed binding names."""
    removed = []
    while True:
        victim = None
        for b in _all_bindings(seq):
            if _owned(b) and not list(b.get_tracks()) and not list(b.get_child_possessables()):
                victim = b
                break
        if victim is None:
            return removed
        removed.append(str(victim.get_name()))
        try:
            victim.untag_binding(OWNER_TAG)
        except Exception:
            pass
        victim.remove()


def _component(actor, cls, what):
    c = actor.get_component_by_class(cls)
    if c is None:
        raise RuntimeError(f"{actor.get_actor_label()}: no {what} component")
    return c


def _linearise(section):
    """parameter-section keys are created with auto (cubic) tangents; beatrig keys are always linear"""
    for ch in section.get_all_channels():
        for k in ch.get_keys():
            if hasattr(k, "set_interpolation_mode"):
                k.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_LINEAR)


def _add_frame_keys(channel, keys):
    for f, v in keys:
        channel.add_key(unreal.FrameNumber(int(f)), float(v), 0.0, unreal.MovieSceneTimeUnit.DISPLAY_RATE, _LIN)


def _add_param_keys(section, plan):
    """GOTCHA: add_scalar/color_parameter_key take TICK time, not display frames (keys.py does the conversion)"""
    name = unreal.Name(plan["param"])
    color = plan["value_type"] == "color"
    if isinstance(section, getattr(unreal, "MovieSceneComponentMaterialParameterSection", ())):
        # UE 5.4+: component material sections key by MaterialParameterInfo (+ material layer / asset names)
        info = unreal.MaterialParameterInfo()
        info.set_editor_property("name", name)
        for t, v in plan["keys"]:
            if color:
                section.add_color_parameter_key(info, unreal.FrameNumber(int(t)), unreal.LinearColor(*v), "", "",
                                                default_interpolation=_LIN)
            else:
                section.add_scalar_parameter_key(info, unreal.FrameNumber(int(t)), float(v), "", "", _LIN)
        return
    for t, v in plan["keys"]:
        if color:
            section.add_color_parameter_key(name, unreal.FrameNumber(int(t)), unreal.LinearColor(*v), _LIN)
        else:
            section.add_scalar_parameter_key(name, unreal.FrameNumber(int(t)), float(v), _LIN)


def _new_section(track, rng):
    """a section over rng = [start, end) display frames; a None end is left unbounded (a blended track mirrors an
    infinite base section, so the parameter keeps the base's value outside the keys instead of dropping to default)"""
    s = track.add_section()
    lo, hi = rng
    if lo is not None and hi is not None:
        s.set_range(int(lo), int(hi))
        return s
    if lo is None:
        s.set_start_frame_bounded(False)
    else:
        s.set_start_frame(int(lo))
    if hi is None:
        s.set_end_frame_bounded(False)
    else:
        s.set_end_frame(int(hi))
    return s


def _section(track, plan):
    """the plan's section, its blend type set BEFORE any key: set afterwards, the stored type changes but Sequencer
    keeps evaluating the section the old way (seen on 5.8.3: an additive MPC section set after its keys still averaged)"""
    sec = _new_section(track, plan["range"])
    if plan["kind"] == "transform" or plan.get("blend") == "additive":
        sec.set_blend_type(unreal.MovieSceneBlendType.ADDITIVE if plan["blend"] == "additive"
                           else unreal.MovieSceneBlendType.ABSOLUTE)
    return sec


def _channels_by_name(section):
    return {str(c.channel_name): c for c in section.get_all_channels()}


def _niagara_float_index():
    """this session's registered index of the Niagara float type, read from a float binding Niagara itself defines"""
    if _CACHE["niagara_float"] is None:
        cdo = unreal.get_default_object(unreal.NiagaraSpriteRendererProperties)
        txt = cdo.get_editor_property("sprite_rotation_binding").export_text()
        m = re.search(r'Name="Particles\.SpriteRotation",TypeDefHandle=\(RegisteredTypeIndex=(\d+)\)', txt)
        if not m:
            raise RuntimeError("could not find the Niagara float type in this editor; drive the system from an MPC target")
        _CACHE["niagara_float"] = int(m.group(1))
    return _CACHE["niagara_float"]


def _niagara_float_track(seq, binding, param):
    """a MovieSceneNiagaraFloatParameterTrack for `param` on a Niagara component binding. The track's Parameter
    property is private to Python, so the track is pasted from Sequencer's copy format (text in, no clipboard)."""
    if not re.fullmatch(r"[A-Za-z0-9_. ]+", param):
        raise RuntimeError(f"Niagara parameter name {param!r} has characters beatrig can't write")
    uid = uuid.uuid4().hex[:12]
    text = _NIAGARA_TRACK_T3D.format(copy=f"BeatrigCopy_{uid}", track=f"BeatrigNiagaraTrack_{uid}", param=param,
                                     index=_niagara_float_index())
    params = unreal.MovieScenePasteTracksParams()
    params.set_editor_property("bindings", [binding])
    params.set_editor_property("sequence", seq)
    out = unreal.get_editor_subsystem(unreal.LevelSequenceEditorSubsystem).paste_tracks(text, params) or []
    tracks = [t for t in out if isinstance(t, unreal.MovieSceneNiagaraFloatParameterTrack)]
    if not tracks:
        raise RuntimeError("Sequencer refused the Niagara parameter track (paste_tracks returned nothing); "
                           "drive the system from an MPC target instead")
    return tracks[0]


def _niagara_track_param(track):
    """the parameter name a Niagara parameter track drives, read from a T3D export of the track to a temp file"""
    import tempfile
    path = os.path.join(tempfile.gettempdir(), f"beatrig-niagara-{uuid.uuid4().hex[:8]}.t3d")
    task = unreal.AssetExportTask()
    for k, v in (("object", track), ("filename", path), ("exporter", unreal.ObjectExporterT3D()), ("automated", True),
                 ("prompt", False), ("replace_identical", True)):
        task.set_editor_property(k, v)
    try:
        unreal.Exporter.run_asset_export_task(task)
        with open(path, encoding="utf-8", errors="replace") as fh:
            m = re.search(r'Parameter=\(Name="([^"]*)"', fh.read())
        return m.group(1) if m else None
    except OSError:
        return None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


_MPCS = {}


def _mpc(path):
    m = _MPCS.get(path)
    if m is None or not unreal.SystemLibrary.is_valid(m):
        m = unreal.load_asset(path)
        if m is None:
            raise RuntimeError(f"MPC asset not found: {path}")
        _MPCS[path] = m
    return m


# ------------------------------------------------------------------ writers, one per target kind
def _write(seq, plan, ctx):
    kind, color = plan["kind"], plan["value_type"] == "color"
    if kind == "mpc":
        track = seq.add_track(unreal.MovieSceneMaterialParameterCollectionTrack)
        track.set_editor_property("mpc", _mpc(plan["asset"]))
        _set_name(track, plan["tag"])
        sec = _section(track, plan)
        _add_param_keys(sec, plan)
        _linearise(sec)
        return track
    actor = _find_actor(plan["actor"])
    if kind == "light":
        comp = _component(actor, unreal.LightComponentBase, "light")
        b = _component_binding(seq, actor, comp, ctx)
        track = b.add_track(unreal.MovieSceneColorTrack if color else unreal.MovieSceneFloatTrack)
        track.set_property_name_and_path(plan["property"], plan["property"])
        _set_name(track, plan["tag"])
        sec = _section(track, plan)
        chans = list(sec.get_all_channels())
        if color:
            for ch, comp_keys in zip(chans, K.split_components(plan["keys"])):
                _add_frame_keys(ch, comp_keys)
        else:
            _add_frame_keys(chans[0], plan["keys"])
        return track
    if kind == "material":
        comp = _component(actor, unreal.PrimitiveComponent, "primitive (mesh)")
        b = _component_binding(seq, actor, comp, ctx)
        track = b.add_track(unreal.MovieSceneComponentMaterialTrack)
        idx = plan["material_index"]
        if hasattr(track, "set_material_info"):                             # UE 5.3+: ComponentMaterialInfo
            info = unreal.ComponentMaterialInfo()
            info.set_editor_property("material_slot_index", idx)
            info.set_editor_property("material_type", unreal.ComponentMaterialType.INDEXED_MATERIAL)
            track.set_material_info(info)
        else:                                                               # older engines
            track.set_material_index(idx)
        _set_name(track, plan["tag"])
        sec = _section(track, plan)
        _add_param_keys(sec, plan)
        _linearise(sec)
        return track
    if kind == "niagara":
        comp = _component(actor, unreal.NiagaraComponent, "Niagara")
        b = _component_binding(seq, actor, comp, ctx)
        track = _niagara_float_track(seq, b, plan["param"])
        _set_name(track, plan["tag"])
        sec = _section(track, plan)
        _add_frame_keys(sec.get_all_channels()[0], plan["keys"])
        return track
    if kind == "transform":
        b = _actor_binding(seq, actor, ctx)
        track = b.add_track(unreal.MovieScene3DTransformTrack)
        _set_name(track, plan["tag"])
        sec = _section(track, plan)
        ch = _channels_by_name(sec).get(plan["channel_name"]) or sec.get_all_channels()[plan["channel_index"]]
        _add_frame_keys(ch, plan["keys"])
        return track
    raise RuntimeError(f"unknown kind {kind}")


# ------------------------------------------------------------------ output blend: the base, its evaluation, records
def _records(seq):
    """{track tag: record} for every base a multiply/add output was applied over (asset metadata, saved with it)"""
    out = {}
    try:
        tags = dict(unreal.EditorAssetLibrary.get_metadata_tag_values(seq))
    except Exception:
        return out
    for k, v in tags.items():
        t = K.base_meta_tag(str(k))
        rec = K.decode_record(str(v)) if t else None
        if rec:
            out[t] = rec
    return out


def _save_record(seq, tag, rec):
    unreal.EditorAssetLibrary.set_metadata_tag(seq, K.base_meta_key(tag), K.encode_record(rec))


def _drop_record(seq, tag):
    unreal.EditorAssetLibrary.remove_metadata_tag(seq, K.base_meta_key(tag))


def _resolve_section(seq, rel):
    """a recorded section, if it still sits in a track of this sequence (a deleted track's section can linger in
    memory, so it is looked up through the sequence's tracks rather than trusted by path)"""
    if not rel:
        return None
    sec = unreal.find_object(None, seq.get_path_name() + ":" + rel)
    if sec is None:
        return None
    track = sec.get_outer()
    for t, _ in _all_tracks(seq):
        if t == track:
            return sec if any(s == sec for s in t.get_sections()) else None
    return None


def _find_binding(seq, actor, component=None):
    """the actor's (or its component's) existing binding, matched the way the writers match; never creates one"""
    label = actor.get_actor_label()
    for b in seq.get_bindings():
        if str(b.get_name()) == label and not (hasattr(b, "get_parent") and b.get_parent().is_valid()):
            if component is None:
                return b
            for c in b.get_child_possessables():
                if str(c.get_name()) == component.get_name():
                    return c
            return None
    return None


def _material_slot(track):
    if hasattr(track, "get_material_info"):
        return int(track.get_material_info().get_editor_property("material_slot_index"))
    return int(track.get_material_index())


def _param_names(sec):
    try:
        return [str(n) for n in sec.get_parameter_names()]
    except Exception:
        return [str(c.channel_name) for c in sec.get_all_channels()]


_COMPONENTS = {"light": (unreal.LightComponentBase, "light"), "material": (unreal.PrimitiveComponent, "primitive (mesh)"),
               "niagara": (unreal.NiagaraComponent, "Niagara")}


def _base_candidates(seq, plan):
    """(track, section) pairs not written by beatrig that drive the plan's target, in sequence order"""
    kind = plan["kind"]
    out = []
    if kind == "mpc":
        want = _pkg(_mpc(plan["asset"]))
        for t in _seq_tracks(seq):
            if not isinstance(t, unreal.MovieSceneMaterialParameterCollectionTrack) or K.is_ours(_track_name(t)):
                continue
            m = t.get_editor_property("mpc")
            if m is not None and _pkg(m) == want:
                out += [(t, s) for s in t.get_sections() if plan["param"] in _param_names(s)]
        return out
    actor = _find_actor(plan["actor"])
    if kind == "transform":
        b = _find_binding(seq, actor)
        tracks = [t for t in b.get_tracks() if isinstance(t, unreal.MovieScene3DTransformTrack)] if b else []
    else:
        comp = _component(actor, *_COMPONENTS[kind])
        b = _find_binding(seq, actor, comp)
        tracks = list(b.get_tracks()) if b else []
        if kind == "light":
            cls = unreal.MovieSceneColorTrack if plan["value_type"] == "color" else unreal.MovieSceneFloatTrack
            tracks = [t for t in tracks if type(t) is cls and str(t.get_property_name()) == plan["property"]]
        elif kind == "material":
            tracks = [t for t in tracks if isinstance(t, unreal.MovieSceneComponentMaterialTrack)
                      and _material_slot(t) == plan["material_index"]]
        else:
            tracks = [t for t in tracks if isinstance(t, unreal.MovieSceneNiagaraFloatParameterTrack)
                      and not K.is_ours(_track_name(t)) and _niagara_track_param(t) == plan["param"]]
    for t in tracks:
        if K.is_ours(_track_name(t)):
            continue
        for s in t.get_sections():
            if kind != "material" or plan["param"] in _param_names(s):
                out.append((t, s))
    return out


def _target_channels(sec, plan):
    """the section's channels for the plan's target, in key order (None when it lacks one)"""
    names = K.channel_names(plan)
    chans = list(sec.get_all_channels())
    if names is None:
        return chans[:1] or None
    by = {str(c.channel_name): c for c in chans}
    sel = [by.get(n) for n in names]
    return sel if all(c is not None for c in sel) else None


def _bounds(sec):
    """(start, end) display frames, end exclusive; None for an unbounded side"""
    return (sec.get_start_frame() if sec.has_start_frame() else None,
            sec.get_end_frame() if sec.has_end_frame() else None)


def _key_time(k):
    ft = k.get_time()
    return ft.frame_number.value + ft.sub_frame


def _evaluate(seq, chans, lo, hi):
    """the channels' values at display frames lo..hi (inclusive), one per frame; tuples for several channels.
    GOTCHA: SequencerScriptingRange() defaults to an internal rate of 60000/s, so the range is built with the
    sequence's display rate spelled out (evaluate_keys then samples once per display frame)."""
    fr = seq.get_display_rate()
    r = unreal.SequencerScriptingRange()
    r.import_text(f"(bHasStart=True,bHasEnd=True,InclusiveStart={int(lo)},ExclusiveEnd={int(hi) + 1},"
                  f"InternalRate=(Numerator={fr.numerator},Denominator={fr.denominator}))")
    cols = [list(c.evaluate_keys(r, fr)) for c in chans]
    n = int(hi) - int(lo) + 1
    if any(len(col) != n for col in cols):
        raise RuntimeError(f"evaluate_keys returned {[len(col) for col in cols]} values for {n} frames")
    return cols[0] if len(cols) == 1 else [tuple(v) for v in zip(*cols)]


def _base_fp(sec, chans):
    """fingerprint of what the base plays (keys, tangents, extrapolation, default, section bounds), not its active
    state; a live push compares it to tell that the base was edited"""
    rows = [list(_bounds(sec))]
    for c in chans:
        keys = []
        for k in c.get_keys():
            keys.append([_key_time(k), k.get_value()] + [str(getattr(k, g)()) for g in
                        ("get_interpolation_mode", "get_tangent_mode", "get_arrive_tangent", "get_leave_tangent")
                        if hasattr(k, g)])
        rows.append([str(c.channel_name), str(c.get_pre_infinity_extrapolation()),
                     str(c.get_post_infinity_extrapolation()), c.get_default() if c.has_default() else None, keys])
    return K.fingerprint(rows)


def _static_value(plan):
    """the target's own value with no section driving it: the MPC default, the light or material property, the
    Niagara user parameter, the actor's transform channel. Read once, at the first apply, and recorded."""
    kind, color = plan["kind"], plan["value_type"] == "color"
    if kind == "mpc":
        m = _mpc(plan["asset"])
        v, found = (m.get_vector_parameter_default_value(plan["param"]) if color
                    else m.get_scalar_parameter_default_value(plan["param"]))
        if not found:
            raise RuntimeError(f"{plan['asset']} has no {'vector' if color else 'scalar'} parameter {plan['param']!r}")
        return (v.r, v.g, v.b, v.a) if color else float(v)
    actor = _find_actor(plan["actor"])
    if kind == "light":
        v = _component(actor, *_COMPONENTS["light"]).get_editor_property(K.editor_property_name(plan["property"]))
        return K.srgb8_to_linear((v.r, v.g, v.b, v.a)) if color else float(v)
    if kind == "material":
        mat = _component(actor, *_COMPONENTS["material"]).get_material(plan["material_index"])
        if mat is None:
            raise RuntimeError(f"{plan['actor']}: no material in slot {plan['material_index']}")
        lib, p = unreal.MaterialEditingLibrary, plan["param"]
        if isinstance(mat, unreal.MaterialInstanceConstant):
            v = (lib.get_material_instance_vector_parameter_value(mat, p) if color
                 else lib.get_material_instance_scalar_parameter_value(mat, p))
        else:
            base = mat.get_base_material()
            v = (lib.get_material_default_vector_parameter_value(base, p) if color
                 else lib.get_material_default_scalar_parameter_value(base, p))
        return (v.r, v.g, v.b, v.a) if color else float(v)
    if kind == "niagara":
        comp = _component(actor, *_COMPONENTS["niagara"])
        for name in (plan["param"][len("User."):], plan["param"]):
            r = comp.get_variable_float(name)
            v, ok = r if isinstance(r, tuple) else (r, True)
            if ok:
                return float(v)
        raise RuntimeError(f"{plan['actor']}: Niagara user parameter {plan['param']!r} not readable; key a base track")
    if kind == "transform":
        prop, axis = plan["property"].split(".")
        if prop == "rotation":
            rot = actor.get_actor_rotation()
            return float({"x": rot.roll, "y": rot.pitch, "z": rot.yaw}[axis])
        v = actor.get_actor_location() if prop == "location" else actor.get_actor_scale3d()
        return float(getattr(v, axis))
    raise RuntimeError(f"unknown kind {kind}")


def _blend_from(seq, plan, sec, static, tol):
    """the plan, ready to write, for a multiply/add output over a base section (sec) or a static value.
    Over a section: an unbounded ADDITIVE section holding the offset on frames first - 1 .. last + 1 (multiply:
    base x (gain - 1), from the base evaluated at every frame; add: the offsets), 0 at both ends, so the base plays alone
    outside the bake range. Over a static value: an ABSOLUTE section of static x gain (or + offset) on the bake range."""
    first, last = int(plan["src_keys"][0][0]), int(plan["src_keys"][-1][0])
    mode = plan["output_blend"]
    if sec is None:
        fk = K.blend_frame_keys(plan["src_keys"], mode, first, [static] * (last - first + 1), tol)
        rng, how = [first, last + 1], "absolute"
    else:
        lo, hi = first - 1, last + 1
        if mode == "multiply":
            values = _evaluate(seq, _target_channels(sec, plan), lo, hi)
        else:                                               # an offset doesn't depend on the base
            width = 4 if plan["value_type"] == "color" else 0
            values = [(0.0,) * width if width else 0.0] * (hi - lo + 1)
        fk = K.offset_frame_keys(plan["src_keys"], mode, lo, values, tol)
        # unbounded: the 0 keys at both ends (constant extrapolation) keep the offset at 0 outside the bake range, and a
        # section that never ends never hands the target back. GOTCHA (5.8.3): when a bounded beatrig section on a
        # component material slot ended, Sequencer restored the material's own value at the frames after it, over the
        # base track on the same slot, until the sequence was next re-evaluated.
        rng, how = [None, None], "additive"
    return K.blended_plan(plan, fk, rng, how, _rate(seq.get_tick_resolution()), _rate(seq.get_display_rate()))


def _channel_names_of(sec, plan):
    return [str(c.channel_name) for c in (_target_channels(sec, plan) or [])]


def _find_base(seq, plan):
    """the first active section, on a track beatrig didn't write, that drives the plan's target (None if there is none).
    Shared sections are fine: the base is only read, never changed."""
    for _, s in _base_candidates(seq, plan):
        if s.is_active() and _target_channels(s, plan) is not None:
            return s
    return None


def _uncovered(sec, plan):
    """bake frames a bounded base section doesn't cover (the offset there adds onto whatever the target holds)"""
    s, e = _bounds(sec)
    first, last = int(plan["src_keys"][0][0]), int(plan["src_keys"][-1][0])
    if (s is None or s <= first) and (e is None or e > last):
        return None
    return [first if s is None else max(first, s), last if e is None else min(last, e - 1)]


def _blend(seq, plan, rec, held, tol):
    """resolve a multiply/add plan's base and compute what to write. rec: this track's record from an earlier apply;
    held: {target key: tag} of targets other tracks already blend. Returns (plan to write, record, warning or None)."""
    tkey = K.target_key(plan)
    if tkey in held:
        raise RuntimeError(f"{held[tkey]} already blends this target; two additive layers on one parameter would be "
                           f"ambiguous, so blend one beatrig output per target")
    mode = plan["output_blend"]
    sec = _find_base(seq, plan)
    rec_new = {"v": 1, "output": plan["output"], "mode": mode, "target": tkey}
    if sec is None:
        static = rec.get("static") if rec and rec.get("kind") == "static" else None
        if static is None:
            static = _static_value(plan)
        static = tuple(static) if isinstance(static, (list, tuple)) else float(static)
        rec_new.update({"kind": "static", "track": None, "static": list(static) if isinstance(static, tuple) else static})
        return _blend_from(seq, plan, None, static, tol), rec_new, None
    chans = _target_channels(sec, plan)
    rec_new.update({"kind": "section", "section": K.relative_path(sec.get_path_name(), seq.get_path_name()),
                    "track": _track_name(sec.get_outer()), "channels": _channel_names_of(sec, plan),
                    "fp": _base_fp(sec, chans) if mode == "multiply" else None})
    gap = _uncovered(sec, plan)
    warn = (f"the base section doesn't cover bake frames {gap[0]}-{gap[1]}; there the offset adds onto the target's "
            f"current value") if gap else None
    return _blend_from(seq, plan, sec, None, tol), rec_new, warn


def _forget(seq, records, output_id=None, keep=()):
    """drop the records of every track of output_id (all outputs when None) except the tags in keep. The base was never
    changed, so there is nothing to give back."""
    out = []
    for tag in sorted(records):
        if K.is_ours(tag, output_id) and tag not in keep:
            records.pop(tag)
            _drop_record(seq, tag)
            out.append(tag)
    return out


def _remove_tag(seq, tag):
    for t, b in _all_tracks(seq):
        if _track_name(t) == tag:
            (b.remove_track(t) if b is not None else _seq_remove_track(seq, t))


def _bases_changed(seq, output_ids):
    """output ids (of those given) whose recorded base section was edited since the apply (multiply: its keys'
    fingerprint changed), or is gone or inactive (the offset would then add onto nothing)"""
    out = set()
    for tag, rec in _records(seq).items():
        p = K.parse_tag(tag)
        if not p or p[0] not in output_ids or rec.get("kind") != "section" or p[0] in out:
            continue
        sec = _resolve_section(seq, rec.get("section"))
        if sec is None or not sec.is_active():
            out.add(p[0])
            continue
        if rec.get("fp") is None:
            continue
        by = {str(c.channel_name): c for c in sec.get_all_channels()}
        chans = [by.get(n) for n in rec.get("channels") or []]
        if not chans or None in chans or _base_fp(sec, chans) != rec.get("fp"):
            out.add(p[0])
    return out


# ------------------------------------------------------------------ commands
def _load_seq(sequence):
    if isinstance(sequence, unreal.LevelSequence):
        return sequence
    seq = unreal.load_asset(sequence)
    if seq is None or not isinstance(seq, unreal.LevelSequence):
        raise RuntimeError(f"not a Level Sequence: {sequence}")
    return seq


def _load_bake(bake):
    if isinstance(bake, dict):
        return bake
    with open(bake, encoding="utf-8") as fh:
        return json.load(fh)


def _plans(seq, bake, start, only):
    return K.plan_bake(bake, _rate(seq.get_display_rate()), _rate(seq.get_tick_resolution()), start, only)


def _is_open(seq):
    cur = _lib().get_current_level_sequence()
    return cur is not None and cur.get_path_name() == seq.get_path_name()


def _refresh(seq):
    """re-evaluate now if this sequence is the one open in Sequencer, so the viewport shows the new keys"""
    if not _is_open(seq):
        return False
    _lib().force_update()
    return True


def remove_ours(seq, output_id=None):
    """remove every beatrig-tagged track (or only one output's). Returns the removed tags."""
    removed = []
    for t in _seq_tracks(seq):
        n = _track_name(t)
        if K.is_ours(n, output_id):
            _seq_remove_track(seq, t)
            removed.append(n)
    for b in _all_bindings(seq):
        for t in list(b.get_tracks()):
            n = _track_name(t)
            if K.is_ours(n, output_id):
                b.remove_track(t)
                removed.append(n)
    pkg = _pkg(seq)
    for key in [k for k in _PUSHED if k[0] == pkg and (output_id is None or k[1] == output_id)]:
        del _PUSHED[key]
    return sorted(removed)


def _tag_counts(seq):
    counts = {}
    for t, _ in _all_tracks(seq):
        p = K.parse_tag(_track_name(t))
        if p:
            counts[p[0]] = counts.get(p[0], 0) + 1
    return counts


def _tolerance(bake):
    return float(bake.get("key_tolerance") or K.DEFAULT_KEY_TOLERANCE)


def _apply(seq, bake, start=0, only=None, remove_ids=()):
    """one transaction: drop the tracks being replaced (plus remove_ids), write, prune emptied bindings.
    multiply/add plans: read the base (never changed), write the offset as an additive section over it (or static x
    gain as an absolute one) and record what it was applied over. Records of tracks that stop being blended go."""
    plans = _plans(seq, bake, start, only) if bake.get("outputs") else []
    tol = _tolerance(bake)
    ctx = {"created": 0}
    written, errors, replaced, dropped = [], [], [], []
    n0 = len(_all_bindings(seq))
    t0 = time.perf_counter()
    with unreal.ScopedEditorTransaction("beatrig apply"):
        ids = sorted({p["output"] for p in plans})
        for oid in ids:
            replaced += remove_ours(seq, oid)
        for oid in remove_ids:
            dropped += remove_ours(seq, oid)
        records = _records(seq)
        keep = {p["tag"] for p in plans if p["output_blend"] != "replace"}
        for oid in sorted(set(ids) | set(remove_ids)):
            _forget(seq, records, oid, keep)
        for p in plans:
            try:
                if p["output_blend"] == "replace":
                    _write(seq, p, ctx)
                    written.append({"tag": p["tag"], "kind": p["kind"], "value_type": p["value_type"],
                                    "keys": len(p["keys"]), "time_base": p["time_base"]})
                    continue
                held = {r["target"]: t for t, r in records.items() if r.get("target") and t != p["tag"]}
                final, rec, warn = _blend(seq, p, records.get(p["tag"]), held, tol)
                try:
                    _write(seq, final, ctx)
                except Exception:
                    _remove_tag(seq, p["tag"])
                    raise
                _save_record(seq, p["tag"], rec)
                records[p["tag"]] = rec
                w = {"tag": p["tag"], "kind": p["kind"], "value_type": p["value_type"], "keys": len(final["keys"]),
                     "time_base": p["time_base"], "blend": p["output_blend"], "section": final["blend"],
                     "base": rec["track"] if rec["kind"] == "section" else "static", "range": final["range"]}
                if warn:
                    w["warning"] = warn
                written.append(w)
            except Exception as e:
                if p["output_blend"] != "replace" and records.pop(p["tag"], None) is not None:
                    _drop_record(seq, p["tag"])                    # nothing written for this track: forget it
                errors.append({"tag": p["tag"], "output": p["output"], "error": f"{type(e).__name__}: {e}"})
        pruned = _prune_owned(seq)
    return {"ok": not errors, "sequence": _pkg(seq), "written": written, "replaced": len(replaced),
            "removed": sorted(dropped), "errors": errors, "keys": sum(w["keys"] for w in written),
            "bindings": {"before": n0, "after": len(_all_bindings(seq)), "created": ctx["created"], "pruned": pruned},
            "write_ms": round((time.perf_counter() - t0) * 1000, 2)}


def _record(seq, hashes, res):
    """remember what was written (output id -> content hash) so a later push can skip unchanged outputs"""
    pkg, failed = _pkg(seq), {e["output"] for e in res["errors"]}
    for oid, h in (hashes or {}).items():
        if oid in failed:
            _PUSHED.pop((pkg, oid), None)
        else:
            _PUSHED[(pkg, oid)] = h


def apply(sequence, bake, only=None, start=0, save=False, refresh=True, hashes=None):
    """write a bake (path or dict) into a Level Sequence. only = output ids to write (default all). hashes: optional
    {output id: content hash} from beatrig.live, recorded so the next push can skip these outputs."""
    seq = _load_seq(sequence)
    res = _apply(seq, _load_bake(bake), start, only)
    _record(seq, hashes, res)
    res["refreshed"] = _refresh(seq) if refresh else False
    if save and res["ok"]:
        unreal.EditorAssetLibrary.save_loaded_asset(seq)
    res["saved"] = bool(save and res["ok"])
    return res


def push(sequence, bake, hashes=None, expect=None, remove=None, start=0, refresh=True, focus=False):
    """live push from beatrig.live.Link: write the (changed) outputs in `bake`, drop the outputs in `remove`, and report
    which `expect` outputs ({id: [hash, track count]}) are no longer in the sequence as last pushed, so the client can
    resend them. focus=True opens the sequence in Sequencer first (it replaces whatever is open there)."""
    seq = _load_seq(sequence)
    if focus and not _is_open(seq):
        _lib().open_level_sequence(seq)
    if bake.get("outputs") or remove:
        res = _apply(seq, bake, start, None, remove or ())
    else:                                          # nothing changed: no transaction, nothing to walk
        res = {"ok": True, "sequence": _pkg(seq), "written": [], "replaced": 0, "removed": [], "errors": [], "keys": 0,
               "bindings": None, "write_ms": 0.0}
    pkg = _pkg(seq)
    _record(seq, hashes, res)
    counts = _tag_counts(seq) if expect else {}
    moved = _bases_changed(seq, set(expect)) if expect else set()      # a blended output whose base changed
    res["stale"] = sorted(oid for oid, (h, n) in (expect or {}).items()
                          if _PUSHED.get((pkg, oid)) != h or counts.get(oid, 0) != int(n) or oid in moved)
    res["base_edited"] = sorted(moved)
    t0 = time.perf_counter()
    res["refreshed"] = _refresh(seq) if refresh else False
    res["refresh_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return res


def undo(sequence, output_id=None, save=False, refresh=True):
    """remove beatrig's tracks (all, or one output's), the bindings it created that are now empty, and the records of
    multiply/add tracks. Base sections were never changed, so they play exactly as before."""
    seq = _load_seq(sequence)
    n0 = len(_all_bindings(seq))
    with unreal.ScopedEditorTransaction("beatrig undo"):
        removed = remove_ours(seq, output_id)
        _forget(seq, _records(seq), output_id)
        pruned = _prune_owned(seq)
    refreshed = _refresh(seq) if refresh else False
    if save:
        unreal.EditorAssetLibrary.save_loaded_asset(seq)
    return {"ok": True, "sequence": _pkg(seq), "removed": removed, "refreshed": refreshed, "saved": bool(save),
            "bindings": {"before": n0, "after": len(_all_bindings(seq)), "pruned": pruned}}


def _read_track(track, plan):
    """a written track read back as one [(display frame, value)] list per channel (4 for colour), in key order"""
    names = K.channel_names(plan)
    width = len(names) if names else 1
    out = [[] for _ in range(width)]
    for sec in track.get_sections():
        chans = list(sec.get_all_channels())
        if names is None:
            sel = chans[:1]
        else:
            by = {str(c.channel_name): c for c in chans}
            sel = [by.get(n) for n in names]
        for i, c in enumerate(sel):
            if c is None:
                continue
            for k in c.get_keys():
                ft = k.get_time()
                out[i].append((ft.frame_number.value + ft.sub_frame, k.get_value()))
    return out


def _check_blend(seq, plan, rec, tol):
    """a multiply/add plan's expected track, recomputed from the target's current base exactly as apply computes it.
    Returns (plan, problem or None)."""
    if rec is None:
        return None, "no base record (apply this output with its blend first)"
    if rec.get("mode") != plan["output_blend"]:
        return None, f"applied as {rec.get('mode')}, the bake says {plan['output_blend']}; apply again"
    sec = _find_base(seq, plan)
    if rec.get("kind") == "static":
        if sec is not None:
            return None, (f"a base section ({_track_name(sec.get_outer())}) now drives this target, but the track was "
                          f"applied over a static value (Sequencer averages the two); apply again")
        st = rec.get("static")
        return _blend_from(seq, plan, None, tuple(st) if isinstance(st, list) else float(st), tol), None
    if sec is None:
        return None, (f"the base section ({rec.get('track')}) is gone or inactive, so the offset adds onto nothing; "
                      f"apply again")
    return _blend_from(seq, plan, sec, None, tol), None


def check(sequence, bake, only=None, start=0):
    """read every planned track back and compare it key for key with the bake (multiply/add outputs: with the offset,
    or static x gain, recomputed from the target's current base, so a base edited since the apply fails)"""
    seq = _load_seq(sequence)
    bake = _load_bake(bake)
    plans = _plans(seq, bake, start, only)
    tol = _tolerance(bake)
    records = _records(seq) if any(p["output_blend"] != "replace" for p in plans) else {}
    tracks = {}
    for t, _ in _all_tracks(seq):
        tracks.setdefault(_track_name(t), t)
    tpf = K.ticks_per_frame(_rate(seq.get_tick_resolution()), _rate(seq.get_display_rate()))
    rows = []
    for p in plans:
        t = tracks.get(p["tag"])
        if t is None:
            rows.append({"tag": p["tag"], "ok": False, "error": "track missing"})
            continue
        problem = None
        if p["output_blend"] != "replace":
            try:
                final, problem = _check_blend(seq, p, records.get(p["tag"]), tol)
            except Exception as e:
                final, problem = None, f"{type(e).__name__}: {e}"
            if final is None:
                rows.append({"tag": p["tag"], "ok": False, "error": problem})
                continue
            p = final
        want, got = K.frame_keys(p, tpf), _read_track(t, p)
        if p["value_type"] == "color":
            rep = K.merge_reports([K.compare_keys(w, g) for w, g in zip(K.split_components(want), got)])
        else:
            rep = K.compare_keys(want, got[0])
        if p["kind"] == "niagara":                     # the parameter is invisible to Python; read it from an export
            rep["param"] = _niagara_track_param(t)
            if rep["param"] != p["param"]:
                rep["ok"] = False
                rep["error"] = f"track drives {rep['param']!r}, expected {p['param']!r}"
        if p["output_blend"] != "replace":
            rep["blend"] = p["output_blend"]
            if problem:
                rep["ok"] = False
                rep["error"] = problem
        rep["tag"] = p["tag"]
        rows.append(rep)
    return {"ok": all(r["ok"] for r in rows), "sequence": _pkg(seq), "tracks": rows,
            "keys": sum(r.get("expected", 0) for r in rows)}


def _position():
    L = _lib()
    p = L.get_global_position(unreal.MovieSceneTimeUnit.DISPLAY_RATE)
    ft = p.get_editor_property("frame")
    return ft.frame_number.value + ft.sub_frame


def playhead_get():
    """the sequence open in Sequencer, its playhead (display frames) and whether it is playing"""
    L = _lib()
    seq = L.get_current_level_sequence()
    if seq is None:
        return {"ok": True, "sequence": None, "frame": None, "playing": False}
    r = seq.get_display_rate()
    return {"ok": True, "sequence": _pkg(seq), "frame": _position(), "playing": bool(L.is_playing()),
            "fps": r.numerator / r.denominator, "range": [seq.get_playback_start(), seq.get_playback_end()],
            "audio": _has_audio(seq)}


def _has_audio(seq):
    """True when the sequence plays sound itself (an audio track with an active section), so a linked page should
    stay quiet instead of doubling the music"""
    for t in seq.get_tracks():
        if isinstance(t, unreal.MovieSceneAudioTrack):
            if any(s.is_active() for s in t.get_sections()):
                return True
    return False


def playhead_set(frame=None, playing=None, sequence=None):
    """move the Sequencer playhead (display frames) and/or play/pause. sequence: open this one first if it isn't open
    (that replaces whatever Sequencer was showing)."""
    L = _lib()
    if sequence:
        seq = _load_seq(sequence)
        if not _is_open(seq):
            L.open_level_sequence(seq)
    elif L.get_current_level_sequence() is None:
        return {"ok": False, "error": "no sequence is open in Sequencer; pass sequence= to open one"}
    if frame is not None:
        p = unreal.MovieSceneSequencePlaybackParams()
        f = float(frame)
        p.set_editor_property("frame", unreal.FrameTime(unreal.FrameNumber(int(f // 1)), f - f // 1))
        p.set_editor_property("position_type", unreal.MovieScenePositionType.FRAME)
        p.set_editor_property("update_method", unreal.UpdatePositionMethod.JUMP)
        L.set_global_position(p, unreal.MovieSceneTimeUnit.DISPLAY_RATE)
    if playing is True and not L.is_playing():
        L.play()
    elif playing is False and L.is_playing():
        L.pause()
    return playhead_get()


def ensure_sequence(path, fps=30, frames=None):
    """create a Level Sequence at `path` if there is none (display rate fps, playback range 0..frames). An existing
    sequence is returned untouched."""
    created = False
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        seq = _load_seq(path)
    else:
        folder, name = path.rsplit("/", 1)
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        seq = tools.create_asset(name, folder, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
        if seq is None:
            raise RuntimeError(f"could not create {path}")
        r = K.rate(fps)
        seq.set_display_rate(unreal.FrameRate(r.numerator, r.denominator))
        seq.set_playback_start(0)
        if frames:
            seq.set_playback_end(int(frames))
        created = True
    r = seq.get_display_rate()
    return {"ok": True, "sequence": _pkg(seq), "created": created, "fps": r.numerator / r.denominator,
            "range": [seq.get_playback_start(), seq.get_playback_end()], "bindings": len(_all_bindings(seq))}


def _perf_settings():
    return unreal.find_object(None, "/Script/UnrealEd.Default__EditorPerformanceSettings")


def throttle(enabled=None):
    """read (and with enabled=True/False set) the editor's 'Use Less CPU when in Background'. While it is on, an editor
    in the background ticks a few times a second and every Remote Control call waits for a tick (~330 ms here).
    The change is in memory only (never written to the ini)."""
    o = _perf_settings()
    name = "bThrottleCPUWhenNotForeground"
    was = bool(o.get_editor_property(name))
    if enabled is not None and bool(enabled) != was:
        o.set_editor_property(name, bool(enabled), unreal.PropertyAccessChangeNotifyMode.NEVER)
    return {"ok": True, "was": was, "now": bool(o.get_editor_property(name))}


def hello():
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    cur = _lib().get_current_level_sequence()
    return {"ok": True, "src_hash": SRC_HASH, "engine": unreal.SystemLibrary.get_engine_version(),
            "level": world.get_outermost().get_path_name() if world else None,
            "open_sequence": _pkg(cur) if cur else None, "throttled": throttle()["now"]}


def scan(paths=None):
    return {"ok": True, "scene": SCAN.scan(paths or ["/Game"])}


_OPS = {"hello": hello, "apply": apply, "check": check, "undo": undo, "push": push, "scan": scan,
        "playhead_get": playhead_get, "playhead_set": playhead_set, "ensure_sequence": ensure_sequence,
        "throttle": throttle}


def rpc(payload_path):
    """run {"op", "args", "src"} from a JSON file and write the result to <payload_path>.out (nothing is printed, so
    a live session doesn't flood the Output Log). A "src" hash that differs from this module's asks the caller to
    reload it."""
    t0 = time.perf_counter()
    try:
        with open(payload_path, encoding="utf-8") as fh:
            req = json.load(fh)
        if req.get("src") and req["src"] != SRC_HASH:
            res = {"ok": False, "stale_module": True, "src_hash": SRC_HASH}
        elif req.get("op") not in _OPS:
            res = {"ok": False, "error": f"unknown op {req.get('op')!r} (one of {sorted(_OPS)})"}
        else:
            res = _OPS[req["op"]](**(req.get("args") or {}))
    except Exception as e:
        res = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=8)[-2000:]}
    res["editor_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    out = payload_path + ".out"
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, separators=(",", ":"))
    os.replace(out + ".tmp", out)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="beatrig_apply")
    ap.add_argument("cmd", choices=["apply", "undo", "check"])
    ap.add_argument("sequence")
    ap.add_argument("arg", nargs="?", help="bake.json (apply/check) or output id (undo)")
    ap.add_argument("--start", type=int, default=0, help="sequence frame of bake frame 0")
    ap.add_argument("--only", default="", help="comma-separated output ids")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args(argv)
    only = [s for s in a.only.split(",") if s] or None
    try:
        if a.cmd == "apply":
            res = apply(a.sequence, a.arg, only, a.start, a.save)
        elif a.cmd == "undo":
            res = undo(a.sequence, a.arg, a.save)
        else:
            res = check(a.sequence, a.arg, only, a.start)
    except Exception as e:
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(RESULT_PREFIX + json.dumps(res, separators=(",", ":")))
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
