"""Scene scan for beatrig: what a rig can target in the open level. Runs inside the Unreal editor; beatrig.live calls
scan() through beatrig_apply.rpc. The result is the scene.json structure in docs/design.md ("v0.2 contract additions").

Kept cheap: one pass over the level's actors, material parameter names read once per base material, MPC assets loaded
(they are tiny), Level Sequences only listed from the asset registry (never loaded). In a World Partition level only
the loaded actors are seen.
"""
import time

import unreal

_NIAGARA = getattr(unreal, "NiagaraComponent", None)


def _vec(v):
    return [round(v.x, 4), round(v.y, 4), round(v.z, 4)]


def _rot(r):
    return [round(r.pitch, 4), round(r.yaw, 4), round(r.roll, 4)]


def _enum(e):
    return str(getattr(e, "name", str(e).split(".")[-1]))


def _kind(c):
    if isinstance(c, unreal.LightComponentBase):
        return "light"
    if _NIAGARA is not None and isinstance(c, _NIAGARA):
        return "niagara"
    if isinstance(c, unreal.CameraComponent):
        return "camera"
    if isinstance(c, unreal.MeshComponent):
        return "mesh"
    return None


def _editor_only(c):
    try:
        return bool(c.get_editor_property("is_editor_only"))
    except Exception:
        return False


def _asset_path(obj):
    return obj.get_path_name().split(".")[0] if obj is not None else None


def _material_entry(mat, cache):
    """(material path, scalars, vectors); parameter names are cached by base material (instances share them)"""
    if mat is None:
        return None, [], []
    shown = mat
    if isinstance(mat, unreal.MaterialInstanceDynamic):                # runtime instance: report the asset behind it
        try:
            shown = mat.get_editor_property("parent") or mat
        except Exception:
            pass
    try:
        key = mat.get_base_material().get_path_name()
    except Exception:
        key = mat.get_path_name()
    if key not in cache:
        lib = unreal.MaterialEditingLibrary
        try:
            cache[key] = (sorted(str(n) for n in lib.get_scalar_parameter_names(mat)),
                          sorted(str(n) for n in lib.get_vector_parameter_names(mat)))
        except Exception:
            cache[key] = ([], [])
    sc, ve = cache[key]
    return _asset_path(shown), list(sc), list(ve)


def _component_entry(c, kind, mat_cache):
    d = {"name": c.get_name(), "class": c.get_class().get_name(), "kind": kind}
    if kind == "mesh":
        try:
            slots = [str(s) for s in c.get_material_slot_names()]
        except Exception:
            slots = []
        mats = []
        for i in range(c.get_num_materials()):
            path, sc, ve = _material_entry(c.get_material(i), mat_cache)
            m = {"slot": i, "scalars": sc, "vectors": ve}
            if i < len(slots):
                m["name"] = slots[i]
            if path:
                m["material"] = path
            mats.append(m)
        d["materials"] = mats
    elif kind == "niagara":
        try:
            d["system"] = _asset_path(c.get_asset())
        except Exception:
            pass
    return d


def _assets(module, cls, paths):
    ar = unreal.AssetRegistryHelpers.get_asset_registry()
    flt = unreal.ARFilter(class_paths=[unreal.TopLevelAssetPath(module, cls)], package_paths=list(paths),
                          recursive_paths=True)
    return sorted(str(a.package_name) for a in ar.get_assets(flt))


def _mpcs(paths):
    out = {}
    for pkg in _assets("/Script/Engine", "MaterialParameterCollection", paths):
        m = unreal.load_asset(pkg)
        if m is None:
            continue
        out[pkg] = {"scalars": [str(p.get_editor_property("parameter_name")) for p in m.get_editor_property("scalar_parameters")],
                    "vectors": [str(p.get_editor_property("parameter_name")) for p in m.get_editor_property("vector_parameters")]}
    return out


def _camera(actors, labels):
    """the dome camera (an OWL 360 capture actor) if there is one, else the first camera actor, else None"""
    for a in actors:
        if "OWL360" not in a.get_class().get_name():
            continue
        cam = {"actor": labels[a], "class": a.get_class().get_name(), "location": _vec(a.get_actor_location()),
               "rotation": _rot(a.get_actor_rotation()), "projection": None, "dome_angle": None}
        for c in a.get_components_by_class(unreal.ActorComponent):
            if not c.get_class().get_name().endswith("CaptureComponent"):
                continue
            try:
                p = _enum(c.get_editor_property("projection_type"))
                cam["projection"] = p[3:].replace("_", "").lower() if p.startswith("CT_") else p.lower()
            except Exception:
                pass
            try:
                cam["dome_angle"] = float(c.get_editor_property("dome_angle"))
            except Exception:
                pass
            break
        return cam
    for a in actors:
        if isinstance(a, unreal.CameraActor):
            cc = a.get_component_by_class(unreal.CameraComponent)
            cam = {"actor": labels[a], "class": a.get_class().get_name(), "location": _vec(a.get_actor_location()),
                   "rotation": _rot(a.get_actor_rotation()), "projection": "perspective", "dome_angle": None}
            if cc is not None:
                cam["fov"] = round(float(cc.get_editor_property("field_of_view")), 4)
            return cam
    return None


def scan(paths=("/Game",)):
    t0 = time.perf_counter()
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    actors = list(unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors())
    labels, out, mat_cache = {}, {}, {}
    for a in actors:
        label = a.get_actor_label()
        key = label if label not in out else f"{label} ({a.get_name()})"   # duplicate labels: keep both
        labels[a] = key
        entry = {"class": a.get_class().get_name(), "location": _vec(a.get_actor_location()),
                 "rotation": _rot(a.get_actor_rotation())}
        root = a.root_component
        if root is not None:
            try:
                entry["mobility"] = _enum(root.get_editor_property("mobility")).lower()
            except Exception:
                pass
        folder = str(a.get_folder_path())
        if folder and folder != "None":
            entry["folder"] = folder
        comps = []
        for c in a.get_components_by_class(unreal.ActorComponent):
            kind = _kind(c)
            if kind is None or _editor_only(c):
                continue
            comps.append(_component_entry(c, kind, mat_cache))
        entry["components"] = comps
        out[key] = entry
    t_actors = time.perf_counter()
    scene = {"version": 1, "source": "unreal", "level": world.get_outermost().get_path_name() if world else None,
             "units": "cm", "engine": unreal.SystemLibrary.get_engine_version(), "actors": out,
             "mpcs": _mpcs(paths), "sequences": _assets("/Script/LevelSequence", "LevelSequence", paths),
             "camera": _camera(actors, labels)}
    t1 = time.perf_counter()
    scene["scan_ms"] = {"actors": round((t_actors - t0) * 1000, 2), "total": round((t1 - t0) * 1000, 2)}
    return scene
