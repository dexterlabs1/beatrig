"""beatrig Unreal applier: writes a bake.json into a Level Sequence as plain Sequencer keys. Runs inside the UE 5.x editor.

STATUS: untested. Written against the UE 5.8 editor Python API but not yet run in an editor. The pure half (tick maths,
key planning, tags) lives in keys.py and is unit tested; this file only makes the editor calls.

Usage (editor Python console, or remotely through Remote Control, which is what `beatrig-mcp`'s ue_apply does):
    beatrig_apply.py apply <sequence asset path> <bake.json> [--start N] [--only id,id] [--save]
    beatrig_apply.py undo  <sequence asset path> [output id] [--save]
    beatrig_apply.py check <sequence asset path> <bake.json> [--start N] [--only id,id]

What it writes, per bake output target kind:
    mpc        sequence-level MovieSceneMaterialParameterCollectionTrack, one scalar param (keys in TICKS)
    light      MovieSceneFloatTrack on the actor's light component binding, e.g. Intensity (keys in display frames)
    material   MovieSceneComponentMaterialTrack on the actor's first primitive component, scalar param (TICKS)
    niagara    MovieSceneNiagaraFloatParameterTrack on the Niagara component (needs an editor that lets Python set
               the track's parameter; otherwise it fails with a clear error: drive Niagara from an MPC instead)
    transform  MovieScene3DTransformTrack on the actor binding, additive blend, one channel (display frames)
Every key is linear, so curves can't overshoot. Every track is named "beatrig:<output id>[:<actor>]"; `undo` removes
exactly those tracks and nothing else. `apply` first removes the tracks it is about to replace, so it is idempotent.
The whole apply is one editor transaction (Ctrl+Z undoes it too). The last printed line is
    BEATRIG_RESULT {json}
which is what the MCP tool parses.
"""
import json
import os
import sys

import unreal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keys as K  # noqa: E402

RESULT_PREFIX = "BEATRIG_RESULT "


# ------------------------------------------------------------------ editor helpers (API differences across 5.x)
def _rate(fr):
    """unreal.FrameRate -> Fraction"""
    return K.rate((fr.numerator, fr.denominator))


def _seq_tracks(seq):
    fn = getattr(seq, "get_tracks", None) or getattr(seq, "get_master_tracks")
    return list(fn())


def _seq_remove_track(seq, track):
    fn = getattr(seq, "remove_track", None) or getattr(seq, "remove_master_track")
    return fn(track)


def _all_bindings(seq):
    """every binding, including component bindings under actors"""
    out, seen = [], set()
    stack = list(seq.get_bindings())
    while stack:
        b = stack.pop(0)
        key = str(b.get_id())
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
        kids = getattr(b, "get_child_possessables", None)
        if kids:
            stack.extend(kids())
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


def _actors_by_label():
    sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    out = {}
    for a in sub.get_all_level_actors():
        out.setdefault(a.get_actor_label(), a)
        out.setdefault(a.get_name(), a)
    return out


def _actor_binding(seq, actor):
    label = actor.get_actor_label()
    for b in seq.get_bindings():
        if str(b.get_name()) == label and not (hasattr(b, "get_parent") and b.get_parent().is_valid()):
            return b
    return seq.add_possessable(actor)


def _component_binding(seq, actor, component):
    """component bindings need their actor bound first (seen in practice on 5.x)"""
    parent = _actor_binding(seq, actor)
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
    return b


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
        channel.add_key(unreal.FrameNumber(int(f)), float(v), 0.0, unreal.SequenceTimeUnit.DISPLAY_RATE,
                        unreal.MovieSceneKeyInterpolation.LINEAR)


def _add_tick_scalar_keys(section, param, keys):
    """GOTCHA: add_scalar_parameter_key takes TICK time, not display frames (keys.py does the conversion)"""
    name = unreal.Name(param)
    for t, v in keys:
        section.add_scalar_parameter_key(name, unreal.FrameNumber(int(t)), float(v))


def _new_section(track, rng):
    s = track.add_section()
    s.set_range(int(rng[0]), int(rng[1]))
    return s


# ------------------------------------------------------------------ writers, one per target kind
def _write(seq, plan, actors):
    kind = plan["kind"]
    if kind == "mpc":
        mpc = unreal.load_asset(plan["asset"])
        if mpc is None:
            raise RuntimeError(f"MPC asset not found: {plan['asset']}")
        track = seq.add_track(unreal.MovieSceneMaterialParameterCollectionTrack)
        track.set_editor_property("mpc", mpc)
        _set_name(track, plan["tag"])
        sec = _new_section(track, plan["range"])
        _add_tick_scalar_keys(sec, plan["param"], plan["keys"])
        _linearise(sec)
        return track
    actor = actors.get(plan["actor"])
    if actor is None:
        raise RuntimeError(f"actor not found in the open level: {plan['actor']}")
    if kind == "light":
        comp = _component(actor, unreal.LightComponent, "light")
        b = _component_binding(seq, actor, comp)
        track = b.add_track(unreal.MovieSceneFloatTrack)
        track.set_property_name_and_path(plan["property"], plan["property"])
        _set_name(track, plan["tag"])
        sec = _new_section(track, plan["range"])
        _add_frame_keys(sec.get_all_channels()[0], plan["keys"])
        return track
    if kind == "material":
        comp = _component(actor, unreal.PrimitiveComponent, "primitive (mesh)")
        b = _component_binding(seq, actor, comp)
        track = b.add_track(unreal.MovieSceneComponentMaterialTrack)
        idx = plan["material_index"]
        try:
            track.set_editor_property("material_index", idx)
        except Exception:
            info = track.get_editor_property("material_info")          # 5.3+: ComponentMaterialInfo
            info.set_editor_property("material_slot_index", idx)
            track.set_editor_property("material_info", info)
        _set_name(track, plan["tag"])
        sec = _new_section(track, plan["range"])
        _add_tick_scalar_keys(sec, plan["param"], plan["keys"])
        _linearise(sec)
        return track
    if kind == "niagara":
        comp = _component(actor, unreal.NiagaraComponent, "Niagara")
        b = _component_binding(seq, actor, comp)
        track = b.add_track(unreal.MovieSceneNiagaraFloatParameterTrack)
        try:
            var = unreal.NiagaraVariable()
            var.set_editor_property("name", unreal.Name("User." + plan["param"].removeprefix("User.")))
            track.set_editor_property("parameter", var)
        except Exception as e:
            b.remove_track(track)
            raise RuntimeError(f"this editor's Python cannot set a Niagara parameter track ({e}); "
                               f"drive the Niagara system from an MPC target instead") from None
        _set_name(track, plan["tag"])
        sec = _new_section(track, plan["range"])
        _add_frame_keys(sec.get_all_channels()[0], plan["keys"])
        return track
    if kind == "transform":
        b = _actor_binding(seq, actor)
        track = b.add_track(unreal.MovieScene3DTransformTrack)
        _set_name(track, plan["tag"])
        sec = _new_section(track, plan["range"])
        sec.set_blend_type(unreal.MovieSceneBlendType.ADDITIVE)
        _add_frame_keys(sec.get_all_channels()[plan["channel_index"]], plan["keys"])
        return track
    raise RuntimeError(f"unknown kind {kind}")


# ------------------------------------------------------------------ commands
def _load_seq(path):
    seq = unreal.load_asset(path)
    if seq is None or not isinstance(seq, unreal.LevelSequence):
        raise RuntimeError(f"not a Level Sequence: {path}")
    return seq


def _load_bake(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _plans(seq, bake, start, only):
    return K.plan_bake(bake, _rate(seq.get_display_rate()), _rate(seq.get_tick_resolution()), start, only)


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
    return sorted(removed)


def cmd_apply(seq_path, bake_path, start=0, only=None, save=False):
    seq = _load_seq(seq_path)
    bake = _load_bake(bake_path)
    plans = _plans(seq, bake, start, only)
    actors = _actors_by_label() if any(p["kind"] != "mpc" for p in plans) else {}
    written, errors = [], []
    with unreal.ScopedEditorTransaction("beatrig apply"):
        replaced = []
        for oid in sorted({K.parse_tag(p["tag"])[0] for p in plans}):
            replaced += remove_ours(seq, oid)
        for p in plans:
            try:
                _write(seq, p, actors)
                written.append({"tag": p["tag"], "kind": p["kind"], "keys": len(p["keys"]), "time_base": p["time_base"]})
            except Exception as e:
                errors.append({"tag": p["tag"], "error": str(e)})
    if save and not errors:
        unreal.EditorAssetLibrary.save_loaded_asset(seq)
    return {"ok": not errors, "sequence": seq_path, "written": written, "replaced": len(replaced), "errors": errors,
            "saved": bool(save and not errors), "keys": sum(w["keys"] for w in written)}


def cmd_undo(seq_path, output_id=None, save=False):
    seq = _load_seq(seq_path)
    with unreal.ScopedEditorTransaction("beatrig undo"):
        removed = remove_ours(seq, output_id)
    if save:
        unreal.EditorAssetLibrary.save_loaded_asset(seq)
    return {"ok": True, "sequence": seq_path, "removed": removed, "saved": bool(save)}


def _read_keys(track, plan):
    """read a written track back as [(display frame, value)]; get_keys() reports display frames for every channel"""
    out = []
    for sec in track.get_sections():
        chans = list(sec.get_all_channels())
        if plan["kind"] in ("mpc", "material"):
            chans = [c for c in chans if str(c.channel_name) == plan["param"]]      # channel_name is a property
        elif plan["kind"] == "transform":
            chans = [chans[plan["channel_index"]]]
        else:
            chans = chans[:1]
        for c in chans:
            for k in c.get_keys():
                ft = k.get_time()
                out.append((ft.frame_number.value + ft.sub_frame, k.get_value()))
    return out


def cmd_check(seq_path, bake_path, start=0, only=None):
    seq = _load_seq(seq_path)
    bake = _load_bake(bake_path)
    plans = _plans(seq, bake, start, only)
    tracks = {}
    for t in _seq_tracks(seq):
        tracks.setdefault(_track_name(t), t)
    for b in _all_bindings(seq):
        for t in b.get_tracks():
            tracks.setdefault(_track_name(t), t)
    tpf = K.ticks_per_frame(_rate(seq.get_tick_resolution()), _rate(seq.get_display_rate()))
    rows = []
    for p in plans:
        t = tracks.get(p["tag"])
        if t is None:
            rows.append({"tag": p["tag"], "ok": False, "error": "track missing"})
            continue
        want = p["keys"] if p["time_base"] == "frames" else [(float(tk / tpf), v) for tk, v in p["keys"]]
        rep = K.compare_keys(want, _read_keys(t, p))
        rep["tag"] = p["tag"]
        rows.append(rep)
    return {"ok": all(r["ok"] for r in rows), "sequence": seq_path, "tracks": rows}


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
            res = cmd_apply(a.sequence, a.arg, a.start, only, a.save)
        elif a.cmd == "undo":
            res = cmd_undo(a.sequence, a.arg, a.save)
        else:
            res = cmd_check(a.sequence, a.arg, a.start, only)
    except Exception as e:
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(RESULT_PREFIX + json.dumps(res, separators=(",", ":")))
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
