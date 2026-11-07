"""Live check of the Unreal applier against a running editor (not collected by pytest; needs Remote Control).

For each bake fixture: apply -> check (key for key) -> apply again (replaces in place) -> check -> undo each output,
counting the sequence's bindings before, after apply and after undo (undo must remove exactly the bindings beatrig
created). The fixtures use neutral names; --map OLD=NEW rewrites actor-name and asset-path prefixes to a real scene.

    py -3 tests/ue_live_smoke.py --sequence /Game/BeatrigLive/LS_Test \\
        --map Light_=MyLight_ --map /Game/FX/MPC_Show=/Game/MyFX/MPC_Show tests/fixtures/bake_color.json

Exit code 0 when every step passed. The sequence is created if missing and is never saved.
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from beatrig.live import Link  # noqa: E402


def remap(bake, pairs):
    """prefix-rewrite actor names (target.actors, channel actor) and target.asset"""
    def r(s):
        if not isinstance(s, str):
            return s
        for old, new in pairs:
            if s.startswith(old):
                return new + s[len(old):]
        return s
    b = copy.deepcopy(bake)
    for o in b["outputs"]:
        t = o.get("target", {})
        if "actors" in t:
            t["actors"] = [r(a) for a in t["actors"]]
        if "asset" in t:
            t["asset"] = r(t["asset"])
        for c in o["channels"]:
            c["actor"] = r(c.get("actor"))
    return b


def run_fixture(link, seq, path, pairs, verbose=False):
    with open(path, encoding="utf-8") as fh:
        bake = remap(json.load(fh), pairs)
    ids = [o["id"] for o in bake["outputs"]]
    tracks = sum(1 for o in bake["outputs"] for c in o["channels"] if c.get("keys"))
    keys = sum(len(c["keys"]) for o in bake["outputs"] for c in o["channels"])
    row = {"fixture": os.path.basename(path), "outputs": len(ids), "tracks": tracks, "keys": keys}
    b0 = link.ensure_sequence(seq, bake["fps"], bake.get("frames"))["bindings"]
    a1 = link.apply(bake, seq)
    c1 = link.check(bake, seq)
    a2 = link.apply(bake, seq)
    c2 = link.check(bake, seq)
    undone = [link.undo(seq, oid) for oid in ids]
    c3 = link.check(bake, seq)
    row.update({
        "apply": a1["ok"], "apply_ms": a1.get("rt_ms"), "check": c1["ok"], "checked_keys": c1.get("keys"),
        "reapply": a2["ok"] and a2["replaced"] == tracks, "replaced": a2["replaced"], "check2": c2["ok"],
        "bindings": [b0, a2["bindings"]["after"], undone[-1]["bindings"]["after"]],
        "created": a1["bindings"]["created"],
        "undo_removed": sum(len(u["removed"]) for u in undone),
        "after_undo_missing": sum(1 for t in c3["tracks"] if t.get("error") == "track missing"),
    })
    row["pass"] = bool(row["apply"] and row["check"] and row["reapply"] and row["check2"]
                       and row["bindings"][2] == b0 and row["undo_removed"] == tracks
                       and row["after_undo_missing"] == tracks)
    if verbose or not row["pass"]:
        for name, r in (("apply", a1), ("check", c1), ("reapply", a2), ("check2", c2)):
            if not r.get("ok"):
                print(f"  {name}: {json.dumps(r)[:1500]}")
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("fixtures", nargs="+")
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--host", default=None)
    ap.add_argument("--map", action="append", default=[], help="OLD=NEW prefix rewrite for actor names and asset paths")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    pairs = [tuple(m.split("=", 1)) for m in a.map]
    with Link(a.host) as link:
        h = link.hello()
        print(f"editor {h['engine']}  level {h['level']}  open in Sequencer: {h['open_sequence']}")
        rows = [run_fixture(link, a.sequence, f, pairs, a.verbose) for f in a.fixtures]
    cols = ["fixture", "outputs", "tracks", "keys", "apply", "apply_ms", "check", "checked_keys", "reapply", "replaced",
            "check2", "bindings", "created", "undo_removed", "after_undo_missing", "pass"]
    for r in rows:
        print("  ".join(f"{c}={r[c]}" for c in cols))
    ok = all(r["pass"] for r in rows)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
