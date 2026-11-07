"""Writes the hand-made bake fixtures in this folder (deterministic; run it again after editing): one file per target
family, neutral actor and asset names. tests/ue_live_smoke.py applies them to a running editor, with --map to point
the names at a real scene.

    py -3 tests/fixtures/make_fixtures.py
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FPS, FRAMES = 30, 120
STEP = 6                                     # a key every 6 frames, plus the last frame


def _key_frames():
    return sorted(set(range(0, FRAMES, STEP)) | {FRAMES - 1})


def _scalar(fn, lo, hi):
    """per-frame values lo..hi from fn(t) in 0..1, and keys sampled from them"""
    vals = [round(lo + (hi - lo) * fn(f / FPS), 6) for f in range(FRAMES)]
    return {"values": vals, "keys": [[f, vals[f]] for f in _key_frames()]}


def _color(fn):
    """per-frame linear RGBA from fn(t) -> (r, g, b), alpha 1"""
    vals = [[round(c, 6) for c in fn(f / FPS)] + [1.0] for f in range(FRAMES)]
    return {"values": vals, "keys": [[f] + vals[f] for f in _key_frames()]}


def pulse(phase=0.0):
    return lambda t: 0.5 - 0.5 * math.cos(2 * math.pi * (t * 1.5 + phase))


def hue(phase=0.0):
    def fn(t):
        a = 2 * math.pi * (t / 4.0 + phase)
        return (0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.cos(a - 2.094), 0.5 + 0.5 * math.cos(a + 2.094))
    return fn


def output(oid, target, chans, lo=None, hi=None):
    o = {"id": oid, "target": target, "channels": chans}
    if lo is not None:
        o["min"], o["max"] = lo, hi
    return o


def bake(outputs):
    return {"version": 1, "fps": FPS, "frames": FRAMES, "audio": None, "rig_hash": "fixture", "outputs": outputs}


def ch(actor, data):
    return dict({"actor": actor}, **data)


FIXTURES = {
    "bake_scalar.json": bake([
        output("intensity", {"kind": "light", "actors": ["Light_0", "Light_1"], "property": "intensity"},
               [ch("Light_0", _scalar(pulse(0.0), 500, 20000)), ch("Light_1", _scalar(pulse(0.25), 500, 20000))],
               500, 20000),
        output("glow", {"kind": "material", "actors": ["Orb_0", "Orb_1"], "param": "Glow"},
               [ch("Orb_0", _scalar(pulse(0.1), 0.3, 6)), ch("Orb_1", _scalar(pulse(0.35), 0.3, 6))], 0.3, 6),
        output("mpc_glow", {"kind": "mpc", "asset": "/Game/FX/MPC_Show", "param": "Glow"},
               [dict({"actor": None}, **_scalar(pulse(0.5), 0.2, 3))], 0.2, 3),
    ]),
    "bake_color.json": bake([
        output("tint", {"kind": "mpc", "asset": "/Game/FX/MPC_Show", "param": "Tint", "value_type": "color"},
               [dict({"actor": None}, **_color(hue(0.0)))]),
        output("light_color", {"kind": "light", "actors": ["Light_0", "Light_1"], "property": "color",
                               "value_type": "color"},
               [ch("Light_0", _color(hue(0.1))), ch("Light_1", _color(hue(0.4)))]),
        output("orb_color", {"kind": "material", "actors": ["Orb_0", "Orb_1"], "param": "Color", "value_type": "color"},
               [ch("Orb_0", _color(hue(0.2))), ch("Orb_1", _color(hue(0.6)))]),
    ]),
    "bake_transform.json": bake([
        output("bob", {"kind": "transform", "actors": ["Mover_0"], "property": "location", "axis": "z", "additive": True},
               [ch("Mover_0", _scalar(pulse(0.0), -50, 50))], -50, 50),
        output("spin", {"kind": "transform", "actors": ["Mover_0"], "property": "rotation", "axis": "yaw",
                        "additive": True},
               [ch("Mover_0", _scalar(lambda t: t / 4.0, 0, 360))], 0, 360),
        output("swell", {"kind": "transform", "actors": ["Mover_0"], "property": "scale", "axis": "x", "additive": True},
               [ch("Mover_0", _scalar(pulse(0.2), 0, 0.5))], 0, 0.5),
    ]),
    "bake_niagara.json": bake([
        output("rate", {"kind": "niagara", "actors": ["Fx_0"], "param": "Rate"},
               [ch("Fx_0", _scalar(pulse(0.0), 10, 200))], 10, 200),
    ]),
}


def main():
    for name, data in FIXTURES.items():
        with open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(data, separators=(",", ":")))
            fh.write("\n")
        n = sum(len(c["keys"]) for o in data["outputs"] for c in o["channels"])
        print(f"{name}: {len(data['outputs'])} outputs, {n} keys")


if __name__ == "__main__":
    main()
