"""Live check of output blend (multiply / add) against a running editor (not collected by pytest; needs Remote Control).

Builds bases the way a show animator would key them, in a scratch Level Sequence, then for each mode applies the blend
fixture and reads the evaluated values back off the world at frames inside and outside the bake range:
  - MPC Glow (scalar) and Tint (colour) keyed in ONE MPC section (Sequencer's default layout), cubic keys out to
    frame 150 (the bake covers 0-119)
  - a point light's Intensity on a float track (section 0-300, keys out to frame 200)
  - with --material: a cube's material scalar Glow on a component material track
Each mode: apply -> base sections still active with unchanged keys -> evaluated = base x gain (base + offset) within the
RDP tolerance, = base outside the range -> check -> edit a base key -> (multiply) push reports base_edited and rewrites,
(add) values follow with no push -> check -> undo (base keys exactly as edited, records and tracks gone).
Then: the static fallback (no base section: an absolute section of static x gain), a base section appearing after a
static apply (check fails), two outputs on one target (refused), and push timings.

    py -3 tests/ue_live_blend.py --sequence /Game/BeatrigTest/LS_Blend [--mpc /Game/BeatrigTest/MPC_Blend] [--material]

The sequence is WIPED (every track and binding): point it at a scratch sequence. One that already has content is refused
unless you pass --wipe. The MPC (scalar Glow, vector Tint) and, with --material, a material with a scalar Glow are created
in the sequence's folder if missing. Test actors Lamp_0, Lamp_1 (and Orb_0) are spawned in the open level and removed at
the end unless --keep. Nothing is saved. Exit code 0 when every step passed.
"""
import argparse
import copy
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beatrig.live import Link, _keys  # noqa: E402
from ue_live_smoke import remap  # noqa: E402

K = _keys()
FIX = os.path.join(ROOT, "tests", "fixtures")
GLOW = [(0, 1.0), (30, 3.0), (60, 2.0), (90, 1.0), (119, 1.5), (150, 0.5)]
TINT = [(0, (1.0, 0.0, 0.0, 1.0)), (60, (0.0, 1.0, 0.0, 1.0)), (150, (0.0, 0.0, 1.0, 1.0))]
LAMP = [(0, 1000.0), (45, 4000.0), (119, 2000.0), (200, 3000.0)]
ORB = [(0, 2.0), (50, 6.0), (119, 3.0), (180, 1.0)]
FRAMES = [0, 17, 30, 40, 77, 119, 120, 121, 150, 200]           # bake range 0-119; 120+ is outside
CHANNELS = {"glow": ["Glow"], "tint": ["TintR", "TintG", "TintB", "TintA"], "lamp": None, "orb": ["Glow"]}
FAIL = []


def expect(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail not in ("", None) else ""))
    if not ok:
        FAIL.append(name)


class Editor:
    """runs snippets of editor Python through the Link's connection; a snippet reports with out(<json-able>)"""

    def __init__(self, link, seq, mpc, material):
        self.link, self.seq, self.mpc, self.material = link, seq, mpc, material

    def __call__(self, body):
        head = (f"import unreal, json\nL = unreal.LevelSequenceEditorBlueprintLibrary\n"
                f"seq = unreal.load_asset({self.seq!r})\nmpc = unreal.load_asset({self.mpc!r})\n"
                f"MAT = {self.material!r}\n"
                "eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)\n"
                "world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()\n"
                "def actor(name):\n    return next((a for a in eas.get_all_level_actors() if a.get_actor_label() == name), None)\n"
                "def out(x):\n    print('RESULT ' + json.dumps(x))\n")
        # the snippet's globals are cleared at the end: its helper functions hold that dict in a reference cycle, so it
        # would otherwise live until a later GC pass with wrappers of objects Sequencer has since destroyed in it (the
        # dynamic material instance it creates). Freeing such a wrapper then crashed the 5.8.3 editor inside python311.
        reply = self.link._exec(f"(lambda g: (exec({(head + body)!r}, g), g.clear()))({{}})")
        lines = [str(e.get("Output", "")) for e in reply.get("LogOutput") or []]
        for s in lines:
            if s.startswith("RESULT "):
                return json.loads(s[len("RESULT "):])
        raise RuntimeError(f"editor snippet gave no RESULT: {reply.get('CommandResult') or lines[-5:]}")


ASSETS = """
tools = unreal.AssetToolsHelpers.get_asset_tools()
made = []
if mpc is None:
    folder, name = MPC_PATH.rsplit("/", 1)
    mpc = tools.create_asset(name, folder, unreal.MaterialParameterCollection, unreal.MaterialParameterCollectionFactoryNew())
    s = unreal.CollectionScalarParameter(); s.set_editor_property("parameter_name", "Glow"); s.set_editor_property("default_value", 0.0)
    v = unreal.CollectionVectorParameter(); v.set_editor_property("parameter_name", "Tint")
    v.set_editor_property("default_value", unreal.LinearColor(0, 0, 0, 1))
    mpc.set_editor_property("scalar_parameters", [s]); mpc.set_editor_property("vector_parameters", [v])
    made.append(MPC_PATH)
if MAT and not unreal.EditorAssetLibrary.does_asset_exist(MAT):
    folder, name = MAT.rsplit("/", 1)
    m = tools.create_asset(name, folder, unreal.Material, unreal.MaterialFactoryNew())
    MEL = unreal.MaterialEditingLibrary
    p = MEL.create_material_expression(m, unreal.MaterialExpressionScalarParameter, -300, 0)
    p.set_editor_property("parameter_name", "Glow"); p.set_editor_property("default_value", 1.0)
    MEL.connect_material_property(p, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    MEL.recompile_material(m)
    unreal.EditorAssetLibrary.save_asset(MAT, False)     # a saved asset, so a crash or restart doesn't drop it
    made.append(MAT)
out({"made": made, "content": len(seq.get_tracks()) + len(seq.get_bindings())})
"""

SETUP = """
for t in list(seq.get_tracks()):
    seq.remove_track(t)
for b in list(seq.get_bindings()):
    for c in list(b.get_child_possessables()):
        c.remove()
    b.remove()
for k in list(dict(unreal.EditorAssetLibrary.get_metadata_tag_values(seq))):
    if str(k).startswith("beatrig."):
        unreal.EditorAssetLibrary.remove_metadata_tag(seq, k)
if L.get_current_level_sequence() is not None:             # let Sequencer restore what the old tracks animated
    L.refresh_current_level_sequence()
    L.force_update()
seq.set_playback_start(0)
seq.set_playback_end(300)
for name, x in (("Lamp_0", 0), ("Lamp_1", 400)):
    a = actor(name)
    if a is None:
        a = eas.spawn_actor_from_class(unreal.PointLight, unreal.Vector(x, 0, 300))
        a.set_actor_label(name)
    a.get_component_by_class(unreal.LightComponentBase).set_editor_property("intensity", 5000.0)
paths = {}
if WITH_MPC:
    t = seq.add_track(unreal.MovieSceneMaterialParameterCollectionTrack); t.set_editor_property("mpc", mpc)
    s = t.add_section()                                   # one section for both parameters: Sequencer's default
    for f, v in GLOW:
        s.add_scalar_parameter_key("Glow", unreal.FrameNumber(f * 800), v)
    for f, c in TINT:
        s.add_color_parameter_key("Tint", unreal.FrameNumber(f * 800), unreal.LinearColor(*c))
    paths["glow"] = paths["tint"] = s.get_path_name()
def bind(a, comp):
    b = seq.add_possessable(a)
    cb = seq.add_possessable(comp)
    if not cb.get_parent().is_valid():
        cb.set_parent(b)
    return cb
l0 = actor("Lamp_0")
cb = bind(l0, l0.get_component_by_class(unreal.LightComponentBase))
lt = cb.add_track(unreal.MovieSceneFloatTrack)
lt.set_property_name_and_path("Intensity", "Intensity")
ls = lt.add_section()
ls.set_range(0, 300)
for f, v in LAMP:
    ls.get_all_channels()[0].add_key(unreal.FrameNumber(f), v)
paths["lamp"] = ls.get_path_name()
if MAT:
    o = actor("Orb_0")
    if o is None:
        # GOTCHA: set the material only here, once. Setting it again while Sequencer holds its own dynamic instance on
        # the slot, then removing that track, crashed 5.8.3 (access violation) three times in this runner's setup.
        o = eas.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 400, 100))
        o.set_actor_label("Orb_0")
        o.static_mesh_component.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/Cube.Cube"))
        o.static_mesh_component.set_material(0, unreal.load_asset(MAT))
    comp = o.static_mesh_component
    cb = bind(o, comp)
    mt = cb.add_track(unreal.MovieSceneComponentMaterialTrack)
    info = unreal.ComponentMaterialInfo()
    info.set_editor_property("material_slot_index", 0)
    info.set_editor_property("material_type", unreal.ComponentMaterialType.INDEXED_MATERIAL)
    mt.set_material_info(info)
    ms = mt.add_section()
    ms.set_range(0, 300)
    pi = unreal.MaterialParameterInfo(); pi.set_editor_property("name", "Glow")
    for f, v in ORB:
        ms.add_scalar_parameter_key(pi, unreal.FrameNumber(f * 800), v, "", "")
    paths["orb"] = ms.get_path_name()
cur = L.get_current_level_sequence()
if cur is None or cur.get_path_name() != seq.get_path_name():
    L.open_level_sequence(seq)
L.refresh_current_level_sequence()
out({"paths": paths})
"""

STATE = """
res = {"world": {}, "base": {}, "active": {}, "keys": {}, "meta": {}}
fr = seq.get_display_rate()
def rng(lo, hi):
    r = unreal.SequencerScriptingRange()
    r.import_text(f"(bHasStart=True,bHasEnd=True,InclusiveStart={lo},ExclusiveEnd={hi + 1},InternalRate=(Numerator={fr.numerator},Denominator={fr.denominator}))")
    return r
for name, p in PATHS.items():
    s = unreal.find_object(None, p)
    res["active"][name] = s.is_active()
    chans = {str(c.channel_name): c for c in s.get_all_channels()}
    want = CHANNELS[name] or [list(chans)[0]]
    res["base"][name] = [list(chans[n].evaluate_keys(rng(-1, 300), fr)) for n in want]      # frames -1..300
    res["keys"][name] = [[[k.get_time().frame_number.value, k.get_value(), str(k.get_interpolation_mode())]
                          for k in chans[n].get_keys()] for n in want]
comps = {n: actor(a).get_component_by_class(unreal.LightComponentBase) for n, a in (("lamp", "Lamp_0"), ("lamp1", "Lamp_1"))}
def mid_value(comp):
    # Glow on the dynamic instance Sequencer drives (read and let go: the instance is transient)
    m = comp.get_material(0)
    if not isinstance(m, unreal.MaterialInstanceDynamic):
        return None
    get = getattr(m, "k2_get_scalar_parameter_value", None) or getattr(m, "get_scalar_parameter_value", None)
    return get("Glow") if get else None
orb = actor("Orb_0") if MAT else None
for f in FRAMES:
    p = unreal.MovieSceneSequencePlaybackParams()
    p.set_editor_property("frame", unreal.FrameTime(unreal.FrameNumber(f), 0.0))
    p.set_editor_property("position_type", unreal.MovieScenePositionType.FRAME)
    p.set_editor_property("update_method", unreal.UpdatePositionMethod.JUMP)
    L.set_global_position(p, unreal.MovieSceneTimeUnit.DISPLAY_RATE)
    L.force_update()
    t = unreal.MaterialLibrary.get_vector_parameter_value(world, mpc, "Tint")
    w = {"glow": unreal.MaterialLibrary.get_scalar_parameter_value(world, mpc, "Glow"), "tint": [t.r, t.g, t.b, t.a],
         "lamp": comps["lamp"].get_editor_property("intensity"), "lamp1": comps["lamp1"].get_editor_property("intensity")}
    if orb is not None:
        w["orb"] = mid_value(orb.static_mesh_component)
    res["world"][f] = w
res["meta"] = {str(k): str(v) for k, v in dict(unreal.EditorAssetLibrary.get_metadata_tag_values(seq)).items()
               if str(k).startswith("beatrig.")}
res["tracks"] = [str(t.get_display_name()) for t in seq.get_tracks()] + [str(t.get_display_name()) for b in seq.get_bindings() for c in b.get_child_possessables() for t in c.get_tracks()]
out(res)
"""


class Run:
    def __init__(self, link, ed, args):
        self.link, self.ed, self.a = link, ed, args
        self.paths = {}

    def setup(self, with_mpc=True):
        body = (SETUP.replace("WITH_MPC", repr(with_mpc)).replace("GLOW", repr(GLOW)).replace("TINT", repr(TINT))
                .replace("LAMP", repr(LAMP)).replace("ORB", repr(ORB)))
        self.paths = self.ed(body)["paths"]
        return self.paths

    def state(self, paths=None):
        r = self.ed(STATE.replace("PATHS", repr(paths or self.paths)).replace("FRAMES", repr(FRAMES))
                    .replace("CHANNELS", repr(CHANNELS)))
        r["world"] = {int(f): v for f, v in r["world"].items()}
        return r

    def bake(self, name, keep=None, material=False):
        b = remap(json.load(open(os.path.join(FIX, name), encoding="utf-8")),
                  [("/Game/FX/MPC_Show", self.a.mpc), ("Light_", "Lamp_")])
        if material:                                       # a material scalar output, from the Glow output's keys
            g = next(o for o in b["outputs"] if o["target"].get("param") == "Glow")
            o = copy.deepcopy(g)
            o["id"] = g["id"].replace("glow", "orb")
            o["target"] = {"kind": "material", "actors": ["Orb_0"], "param": "Glow"}
            o["channels"][0]["actor"] = "Orb_0"
            b["outputs"].append(o)
        if keep is not None:
            b["outputs"] = [o for o in b["outputs"] if o["id"] in keep]
        return b


def gain_keys(bake, oid):
    o = next(o for o in bake["outputs"] if o["id"] == oid)
    return [(k[0], tuple(k[1:]) if len(k) == 5 else k[1]) for k in o["channels"][0]["keys"]]


def compare(st, bake, mode, ids, label, verbose):
    """evaluated world value vs base(f) (x|+) gain(f) at FRAMES. Returns {target: worst error / RDP tolerance}; the
    tolerance is key_tolerance x the final curve's range over the additive span (frames -1..120)."""
    worst = {}
    if verbose:
        print(f"  {label} (target, frame, base, gain, want, got):")
    for name, oid in ids.items():
        cols = st["base"][name]
        base = [tuple(v) for v in zip(*cols)] if len(cols) == 4 else cols[0]          # index = frame + 1
        nt = (K.NEUTRAL[mode],) * 4 if len(cols) == 4 else K.NEUTRAL[mode]
        gk = gain_keys(bake, oid)
        span = range(-1, 121)
        want_span = [K.combine(base[f + 1], g, mode) for f, g in zip(span, K.sample_keys(gk, span, nt))]
        tol = K.DEFAULT_KEY_TOLERANCE * K.value_range(want_span)
        for f in FRAMES:
            g = K.sample_keys(gk, [f], nt)[0]
            want = K.combine(base[f + 1], g, mode)
            got = st["world"][f][name]
            got = tuple(got) if isinstance(got, list) else got
            if got is None:
                worst[name] = float("inf")
                continue
            err = max(abs(x - y) for x, y in zip(want, got)) if isinstance(want, tuple) else abs(want - got)
            slack = 1e-6 * (1.0 if isinstance(want, tuple) else max(1.0, abs(want)))     # float32 storage
            worst[name] = max(worst.get(name, 0.0), round(err / (tol + slack), 3))
            if verbose:
                def fmt(x):
                    return "(" + ", ".join(f"{v:.4f}" for v in x) + ")" if isinstance(x, tuple) else f"{x:.4f}"
                print(f"    {name:5} f{f:<4} base {fmt(base[f + 1]):34} gain {fmt(g):34} want {fmt(want):34} got {fmt(got)}")
    return worst


def run_mode(r, mode, fixture, ids):
    print(f"\n=== {mode}: {fixture} (Glow + Tint share one MPC section) ===")
    link, seq = r.link, r.a.sequence
    r.setup()
    bake = r.bake(fixture, material=r.a.material)
    st0 = r.state()
    a = link.apply(bake, seq)
    for w in a.get("written", []):
        print(f"    wrote {w['tag']}: {w['keys']} keys, {w.get('section')} section {w.get('range')}, base {w.get('base')}"
              + (f", warning: {w['warning']}" if w.get("warning") else ""))
    expect("apply ok", a["ok"], a.get("errors") or "")
    expect("additive over every base section", all(w.get("section") == "additive" for w in a.get("written", [])))
    st = r.state()
    expect("base sections still active, keys untouched", all(st["active"].values()) and st["keys"] == st0["keys"],
           st["active"])
    expect("one record per track", sorted(K.base_meta_tag(k) for k in st["meta"]) == sorted(w["tag"] for w in a["written"]))
    worst = compare(st, bake, mode, ids, "evaluated after apply", r.a.verbose)
    expect("evaluated = base (x|+) gain within the RDP tolerance; = base outside the range",
           all(v <= 1.0 for v in worst.values()), worst)
    outside = all(abs((st["world"][f]["glow"]) - st["base"]["glow"][0][f + 1]) < 1e-6 for f in (120, 121, 150, 200))
    expect("frames 120-200 play the base exactly (Glow)", outside)
    c = link.check(bake, seq)
    expect("check ok", c["ok"], [(t["tag"], t["ok"], t.get("expected"), t.get("max_value_err"), t.get("error"))
                                 for t in c["tracks"]])
    # edit the base while the blend sits on it: Glow key 30 -> 5.0, light key 45 -> 6000
    r.ed(f"""
s = unreal.find_object(None, {r.paths['glow']!r})
ch = {{str(c.channel_name): c for c in s.get_all_channels()}}["Glow"]
[k for k in ch.get_keys() if k.get_time().frame_number.value == 30][0].set_value(5.0)
s2 = unreal.find_object(None, {r.paths['lamp']!r})
[k for k in s2.get_all_channels()[0].get_keys() if k.get_time().frame_number.value == 45][0].set_value(6000.0)
out(True)""")
    if mode == "multiply":
        c2 = link.check(bake, seq)
        expect("check fails for the outputs whose base was edited (offset stale)",
               sorted(t["tag"] for t in c2["tracks"] if not t["ok"]) == sorted([f"beatrig:{ids['glow']}",
                                                                                 f"beatrig:{ids['lamp']}:Lamp_0"]))
        t0 = time.perf_counter()
        p = link.push(bake, seq)
        print(f"    push: pushed {p['pushed']} base_edited {p['base_edited']} calls {p['calls']} "
              f"{(time.perf_counter() - t0) * 1000:.1f} ms")
        expect("push reports base_edited and rewrites exactly those", sorted(p["base_edited"]) == sorted(p["pushed"])
               == sorted([ids["glow"], ids["lamp"]]) and p["calls"] == 2 and p["ok"])
    else:
        p = link.push(bake, seq)
        expect("add: a base edit needs no rewrite (the offset doesn't depend on the base)",
               p["pushed"] == [] and p["base_edited"] == [], p)
    st2 = r.state()
    worst2 = compare(st2, bake, mode, ids, "evaluated after the base edit", r.a.verbose)
    expect("evaluated follows the edited base", all(v <= 1.0 for v in worst2.values()), worst2)
    c3 = link.check(bake, seq)
    expect("check ok after the edit", c3["ok"], [(t["tag"], t.get("error")) for t in c3["tracks"] if not t["ok"]])
    p2 = link.push(bake, seq)
    expect("push with nothing changed: 1 call, nothing written", p2["pushed"] == [] and p2["calls"] == 1)
    u = link.undo(seq)
    st3 = r.state()
    expect("undo: tracks and records gone, bases active with their keys exactly as edited",
           not st3["meta"] and not any(t.startswith("beatrig") for t in st3["tracks"]) and all(st3["active"].values())
           and st3["keys"] == st2["keys"], u.get("removed"))
    expect("after undo the world plays the base",
           all(abs(st3["world"][f]["glow"] - st3["base"]["glow"][0][f + 1]) < 1e-5 and
               abs(st3["world"][f]["lamp"] - st3["base"]["lamp"][0][f + 1]) < 1e-2 for f in FRAMES))
    return a.get("rt_ms")


def run_fallback(r):
    print("\n=== no base section: Lamp_1 multiply, MPC Glow add with no MPC track ===")
    link, seq = r.link, r.a.sequence
    r.setup(with_mpc=False)
    paths = {"lamp": r.paths["lamp"]}
    b = r.bake("bake_blend_multiply.json", keep=["lamp_gain"])
    b["outputs"][0]["target"]["actors"] = ["Lamp_1"]
    b["outputs"][0]["channels"][0]["actor"] = "Lamp_1"
    b["outputs"] += r.bake("bake_blend_add.json", keep=["glow_add"])["outputs"]
    a = link.apply(b, seq)
    for w in a.get("written", []):
        print(f"    wrote {w['tag']}: {w['keys']} keys, {w.get('section')} section {w.get('range')}, base {w.get('base')}")
    expect("static base: absolute sections", a["ok"] and all(w["section"] == "absolute" and w["base"] == "static"
                                                             for w in a["written"]), a.get("errors") or "")
    st = r.state(paths)
    recs = {K.base_meta_tag(k): K.decode_record(v) for k, v in st["meta"].items()}
    expect("static values recorded (light 5000, MPC default 0)",
           recs["beatrig:lamp_gain:Lamp_1"]["static"] == 5000.0 and recs["beatrig:glow_add"]["static"] == 0.0)
    ok = True
    for f in FRAMES:
        gl = K.sample_keys(gain_keys(b, "lamp_gain"), [f], None)[0]
        go = K.sample_keys(gain_keys(b, "glow_add"), [f], None)[0]
        wl = 5000.0 * gl if gl is not None else 5000.0
        wg = go if go is not None else 0.0
        gotl, gotg = st["world"][f]["lamp1"], st["world"][f]["glow"]
        ok = ok and abs(gotl - wl) <= 0.002 * 5000 * 0.95 + 1e-3 and abs(gotg - wg) <= 0.002 + 1e-6
        if r.a.verbose:
            print(f"    f{f:<4} lamp1 want {wl:.3f} got {gotl:.3f} | glow want {wg:.4f} got {gotg:.4f}")
    expect("evaluated = static x gain / static + offset; outside the range = the static value", ok)
    expect("check ok", link.check(b, seq)["ok"])
    r.ed(f"""
t = seq.add_track(unreal.MovieSceneMaterialParameterCollectionTrack); t.set_editor_property("mpc", mpc)
s = t.add_section(); s.add_scalar_parameter_key("Glow", unreal.FrameNumber(0), 2.0)
out(True)""")
    c = link.check(b, seq)
    bad = [t for t in c["tracks"] if not t["ok"]]
    expect("a base section added after a static apply makes check fail (Sequencer would average)",
           [t["tag"] for t in bad] == ["beatrig:glow_add"] and "now drives this target" in bad[0]["error"])
    a2 = link.apply(b, seq)
    expect("apply again puts the add on the new base as an additive section",
           a2["ok"] and {w["tag"]: w["section"] for w in a2["written"]}["beatrig:glow_add"] == "additive"
           and link.check(b, seq)["ok"])
    link.undo(seq)


def run_conflict(r):
    print("\n=== two outputs on one target ===")
    r.setup()
    b = r.bake("bake_blend_multiply.json", keep=["glow_gain"])
    o = copy.deepcopy(b["outputs"][0])
    o["id"] = "glow_twice"
    b["outputs"].append(o)
    a = r.link.apply(b, r.a.sequence)
    errs = a.get("errors") or []
    expect("the second blend on the same parameter is refused",
           not a["ok"] and len(errs) == 1 and errs[0]["output"] == "glow_twice" and "already blends" in errs[0]["error"],
           errs)
    r.link.undo(r.a.sequence)


def run_timing(r):
    print("\n=== push timings (multiply fixture) ===")
    link, seq = r.link, r.a.sequence
    r.setup()
    bm = r.bake("bake_blend_multiply.json")
    link.forget()
    rows = []
    p = link.push(bm, seq)
    rows.append(("push, all blended outputs", p))
    for _ in range(5):
        rows.append(("push, nothing changed", link.push(bm, seq)))
    for i in range(6):
        b2 = copy.deepcopy(bm)
        k = b2["outputs"][0]["channels"][0]["keys"]
        k[5] = [k[5][0], round(k[5][1] + 0.01 * (i + 1), 6)]
        rows.append(("push, one output changed", link.push(b2, seq)))
    for i in range(3):
        r.ed(f"""
s = unreal.find_object(None, {r.paths['lamp']!r})
[k for k in s.get_all_channels()[0].get_keys() if k.get_time().frame_number.value == 45][0].set_value({4000.0 + 100 * (i + 1)})
out(True)""")
        rows.append(("push after a base key edit", link.push(b2, seq)))
    for name, p in rows:
        print(f"    {name:28} {p['ms']:7.1f} ms  editor {p['editor_ms']:6.1f}  keys {p['keys']:4}  calls {p['calls']}"
              f"  pushed {p['pushed']}")
    link.undo(seq)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sequence", required=True, help="a scratch Level Sequence (it is wiped)")
    ap.add_argument("--mpc", default=None, help="MPC with scalar Glow and vector Tint (default: <sequence folder>/MPC_BR_Test)")
    ap.add_argument("--material", action="store_true", help="also blend a material scalar (creates <folder>/M_BR_Glow)")
    ap.add_argument("--host", default=None)
    ap.add_argument("--wipe", action="store_true", help="allow wiping a sequence that already has tracks or bindings")
    ap.add_argument("--keep", action="store_true", help="leave the test actors in the level")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every value read back")
    a = ap.parse_args(argv)
    folder = a.sequence.rsplit("/", 1)[0]
    a.mpc = a.mpc or f"{folder}/MPC_BR_Test"
    mat = f"{folder}/M_BR_Glow" if a.material else None
    with Link(a.host) as link:
        h = link.hello()
        print(f"editor {h['engine']}  level {h['level']}  open in Sequencer: {h['open_sequence']}")
        link.ensure_sequence(a.sequence, 30, 300)
        ed = Editor(link, a.sequence, a.mpc, mat)
        made = ed(ASSETS.replace("MPC_PATH", repr(a.mpc)))
        if made["made"]:
            print("created", made["made"])
            ed.mpc = a.mpc
        if made["content"] and not a.wipe:
            print(f"{a.sequence} already has tracks or bindings; this runner wipes it. Pass --wipe if it is scratch.")
            return 2
        r = Run(link, ed, a)
        ids_m = {"glow": "glow_gain", "tint": "tint_gain", "lamp": "lamp_gain"}
        ids_a = {"glow": "glow_add", "tint": "tint_add", "lamp": "lamp_add"}
        if a.material:
            ids_m["orb"], ids_a["orb"] = "orb_gain", "orb_add"
        run_mode(r, "multiply", "bake_blend_multiply.json", ids_m)
        run_mode(r, "add", "bake_blend_add.json", ids_a)
        run_fallback(r)
        run_conflict(r)
        run_timing(r)
        if not a.keep:
            ed("""
for n in ("Lamp_0", "Lamp_1", "Orb_0"):
    x = actor(n)
    if x is not None:
        eas.destroy_actor(x)
out(True)""")
    print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nPASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
