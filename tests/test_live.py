"""beatrig.live.Link against a fake Remote Control server that speaks the same protocol as the editor (HTTP/1.1 keep-alive,
ExecutePythonCommandEx, the exec() bootstrap, rpc via a JSON payload file and a .out reply). No Unreal needed; the
in-editor side is covered by tests/ue_live_smoke.py against a real editor."""
import copy
import http.server
import importlib.util
import json
import os
import re
import socket
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from beatrig import live  # noqa: E402
from beatrig.live import Link, LiveError, output_hash, slim_bake  # noqa: E402

_spec = importlib.util.spec_from_file_location("beatrig_keys_test", os.path.join(ROOT, "ue", "keys.py"))
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)


class FakeEditor:
    """the editor side of the protocol, with push semantics like beatrig_apply.push"""

    def __init__(self):
        self.boots = 0
        self.ops = []
        self.tracks = {}                  # sequence -> {output id: hash}
        self.throttled = True
        self.stale_once = False
        self.src = K.source_hash(os.path.join(ROOT, "ue"))

    def handle(self, req):
        if self.stale_once:
            self.stale_once = False
            return {"ok": False, "stale_module": True}
        assert req["src"] == self.src
        op, a = req["op"], req["args"]
        self.ops.append((op, copy.deepcopy(a)))
        if op == "hello":
            return {"ok": True, "src_hash": self.src, "throttled": self.throttled}
        if op == "throttle":
            was = self.throttled
            if a.get("enabled") is not None:
                self.throttled = bool(a["enabled"])
            return {"ok": True, "was": was, "now": self.throttled}
        if op in ("push", "apply"):
            seq = self.tracks.setdefault(a["sequence"], {})
            outs = a["bake"]["outputs"]
            for o in outs:
                assert all("values" not in c for c in o["channels"])        # keys only reach the editor
                seq[o["id"]] = (a.get("hashes") or {}).get(o["id"], "applied")
            for oid in a.get("remove") or []:
                seq.pop(oid, None)
            stale = sorted(i for i, (h, n) in (a.get("expect") or {}).items() if seq.get(i) != h)
            return {"ok": True, "written": [{"tag": f"beatrig:{o['id']}"} for o in outs], "errors": [],
                    "keys": sum(len(c["keys"]) for o in outs for c in o["channels"]), "stale": stale,
                    "refreshed": False, "write_ms": 1.0, "replaced": 0, "bindings": {"before": 0, "after": 0}}
        if op == "undo":
            seq = self.tracks.setdefault(a["sequence"], {})
            gone = [i for i in list(seq) if a.get("output_id") in (None, i)]
            for i in gone:
                del seq[i]
            return {"ok": True, "removed": gone}
        if op == "scan":
            return {"ok": True, "scene": {"version": 1, "source": "unreal", "level": "/Game/Maps/L_Test", "units": "cm",
                                          "actors": {"Light_0": {"class": "PointLight", "location": [0, 0, 0],
                                                                 "rotation": [0, 0, 0], "components": [
                                                                     {"name": "LightComponent0", "class": "PointLightComponent",
                                                                      "kind": "light"}]}},
                                          "mpcs": {"/Game/FX/MPC_Show": {"scalars": ["Glow"], "vectors": ["Tint"]}},
                                          "sequences": ["/Game/Show/LS_Main"], "camera": None, "scan_ms": {"total": 1.0}}}
        if op == "playhead_get":
            return {"ok": True, "sequence": "/Game/Show/LS_Main", "frame": 12.0, "playing": False}
        if op == "playhead_set":
            return {"ok": True, "sequence": a.get("sequence") or "/Game/Show/LS_Main", "frame": a.get("frame"),
                    "playing": bool(a.get("playing"))}
        return {"ok": False, "error": f"unknown op {op}"}


def make_server(editor):
    state = {"connections": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"                                     # keep-alive, like the editor

        def setup(self):
            super().setup()
            state["connections"] += 1

        def log_message(self, *a):
            pass

        def do_PUT(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["functionName"] == "ExecutePythonCommandEx"
            cmd = body["parameters"]["PythonCommand"]
            log = []
            if cmd.startswith("exec("):
                editor.boots += 1
                log.append({"Type": "Info", "Output": f"BEATRIG_BOOT {editor.src}\r\n"})
            else:
                m = re.match(r"__import__\('beatrig_apply'\)\.rpc\('(.+)'\)$", cmd)
                assert m, cmd
                path = m.group(1)
                with open(path, encoding="utf-8") as fh:
                    res = editor.handle(json.load(fh))
                res["editor_ms"] = 0.5
                with open(path + ".out", "w", encoding="utf-8") as fh:
                    json.dump(res, fh)
            data = json.dumps({"ReturnValue": True, "LogOutput": log}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


@pytest.fixture
def rc():
    editor = FakeEditor()
    srv, state = make_server(editor)
    yield editor, f"127.0.0.1:{srv.server_address[1]}", state
    srv.shutdown()
    srv.server_close()


def bake3():
    def out(oid, kind, actors, keys):
        t = {"kind": kind, "actors": actors, "param": "Glow"} if kind != "mpc" else {"kind": "mpc", "asset": "/Game/FX/M", "param": "Glow"}
        return {"id": oid, "target": t, "min": 0, "max": 1,
                "channels": [{"actor": a, "values": [0.0] * 10, "keys": keys} for a in (actors or [None])]}
    return {"version": 1, "fps": 30, "frames": 10, "outputs": [
        out("lights", "light", ["L0", "L1"], [[0, 0.0], [9, 1.0]]),
        out("orbs", "material", ["O0"], [[0, 0.5], [9, 0.5]]),
        out("glow", "mpc", None, [[0, 0.2], [5, 1.0], [9, 0.2]])]}


def test_output_hash_follows_what_the_applier_writes():
    b = bake3()
    o = b["outputs"][0]
    h = output_hash(o, 30)
    o2 = copy.deepcopy(o)
    o2["channels"][0]["values"] = [1.0] * 10                 # per-frame values are not sent, so they don't count
    o2["min"] = 5
    assert output_hash(o2, 30) == h
    o2["channels"][0]["keys"][1] = [9, 0.9]
    assert output_hash(o2, 30) != h
    assert output_hash(o, 30, start=10) != h and output_hash(o, 25) != h
    o3 = copy.deepcopy(o)
    o3["target"]["property"] = "color"
    assert output_hash(o3, 30) != h
    s = slim_bake(b, {"glow"})
    assert [x["id"] for x in s["outputs"]] == ["glow"] and "values" not in s["outputs"][0]["channels"][0]


def test_boots_once_and_keeps_one_connection(rc):
    editor, host, state = rc
    link = Link(host, keep_awake=False)
    for _ in range(5):
        assert link.hello()["ok"]
    assert editor.boots == 1 and link.connects == 1 and state["connections"] == 1
    assert link.status()["calls"] == 6                       # boot + 5 calls
    link.close()


def test_push_writes_only_changed_outputs(rc):
    editor, host, _ = rc
    link = Link(host, keep_awake=False)
    b = bake3()
    r = link.push(b, "/Game/Show/LS")
    assert r["pushed"] == ["lights", "orbs", "glow"] and r["skipped"] == [] and r["calls"] == 1 and r["ok"]
    r = link.push(b, "/Game/Show/LS")
    assert r["pushed"] == [] and r["skipped"] == ["lights", "orbs", "glow"] and r["keys"] == 0
    assert editor.ops[-1][1]["bake"]["outputs"] == []        # nothing but hashes travelled
    b2 = copy.deepcopy(b)
    b2["outputs"][2]["channels"][0]["keys"][1] = [5, 0.7]    # one node changed -> one output
    r = link.push(b2, "/Game/Show/LS")
    assert r["pushed"] == ["glow"] and r["skipped"] == ["lights", "orbs"] and r["keys"] == 3
    assert [o["id"] for o in editor.ops[-1][1]["bake"]["outputs"]] == ["glow"]
    b3 = copy.deepcopy(b2)
    b3["outputs"] = [o for o in b3["outputs"] if o["id"] != "orbs"]
    r = link.push(b3, "/Game/Show/LS")
    assert r["removed"] == ["orbs"] and "orbs" not in editor.tracks["/Game/Show/LS"]
    r = link.push(b3, "/Game/Show/LS", only_changed=False)
    assert r["pushed"] == ["lights", "glow"]
    link.close()


def test_push_resends_outputs_the_editor_lost(rc):
    editor, host, _ = rc
    link = Link(host, keep_awake=False)
    b = bake3()
    link.push(b, "/Game/Show/LS")
    del editor.tracks["/Game/Show/LS"]["orbs"]               # e.g. an undo from another client
    r = link.push(b, "/Game/Show/LS")
    assert r["calls"] == 2 and r["pushed"] == ["orbs"] and r["skipped"] == ["lights", "glow"]
    link.undo("/Game/Show/LS", "glow")                       # undo through the link clears its cache entry
    assert link.push(b, "/Game/Show/LS")["pushed"] == ["glow"]
    link.close()


def test_apply_then_push_skips_in_one_call(rc):
    editor, host, _ = rc
    with Link(host, keep_awake=False) as link:
        assert link.apply(bake3(), "/Game/Show/LS")["ok"]
        r = link.push(bake3(), "/Game/Show/LS")
        assert r["pushed"] == [] and r["calls"] == 1 and len(r["skipped"]) == 3


def test_keep_awake_turns_throttle_off_and_restores_it(rc):
    editor, host, _ = rc
    link = Link(host, keep_awake=True)
    link.hello()
    assert editor.throttled is False
    link.close()
    assert editor.throttled is True
    editor.throttled = False                                 # someone turned it off before: leave it off
    link2 = Link(host, keep_awake=True)
    link2.hello()
    link2.close()
    assert editor.throttled is False


def test_stale_module_reboots_once(rc):
    editor, host, _ = rc
    link = Link(host, keep_awake=False)
    link.hello()
    editor.stale_once = True
    assert link.hello()["ok"] and editor.boots == 2
    link.close()


def test_scan_and_cli(rc, tmp_path):
    editor, host, _ = rc
    with Link(host, keep_awake=False) as link:
        scene = link.scan()
    assert scene["actors"]["Light_0"]["components"][0]["kind"] == "light" and "rt" in scene["scan_ms"]
    out = tmp_path / "scene.json"
    assert live.cli_scan(["--host", host, "--out", str(out)]) == 0
    assert json.loads(out.read_text())["mpcs"]["/Game/FX/MPC_Show"]["vectors"] == ["Tint"]


def test_cli_hook_registers_scan():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    live.add_cli(sub)
    a = ap.parse_args(["scan", "--out", "x.json", "--host", "127.0.0.1:1"])
    assert a.fn is live.cli_scan and a.out == "x.json"


def test_playhead_round_trip(rc):
    editor, host, _ = rc
    with Link(host, keep_awake=False) as link:
        assert link.playhead_get()["frame"] == 12.0
        r = link.playhead_set(30, True, "/Game/Show/LS")
        assert r["frame"] == 30 and r["playing"] is True
    assert editor.ops[-1] == ("playhead_set", {"frame": 30, "playing": True, "sequence": "/Game/Show/LS"})


def test_no_editor_is_a_clear_error():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                                # a port nobody listens on
    link = Link(f"127.0.0.1:{port}", keep_awake=False, timeout=2)
    with pytest.raises(LiveError, match="no Unreal editor answering"):
        link.hello()
    link.close()


def test_mcp_tools_go_through_the_link(rc, tmp_path):
    editor, host, _ = rc
    from beatrig import mcp_server as M
    bp = tmp_path / "bake.json"
    bp.write_text(json.dumps(bake3()))
    r = json.loads(M.live_push(str(bp), "/Game/Show/LS", host=host))
    assert r["pushed"] == ["lights", "orbs", "glow"]
    r = json.loads(M.live_push(str(bp), "/Game/Show/LS", host=host))     # same server-wide link: nothing to do
    assert r["pushed"] == [] and len(r["skipped"]) == 3
    r = json.loads(M.ue_scan(out=str(tmp_path / "scene.json"), host=host))
    assert r["by_kind"]["light"]["names"] == ["Light_0"] and os.path.exists(r["scene_path"])
    assert json.loads(M.playhead_get(host=host))["frame"] == 12.0
    r = json.loads(M.ue_undo("/Game/Show/LS", host=host))
    assert r["ok"] and sorted(r["removed"]) == ["glow", "lights", "orbs"]
    M._LINKS.pop(host).close()
