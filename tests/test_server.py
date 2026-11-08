"""Preview server: lean /api/state, binary curves and peaks (matching the bake), server-sent events, /api/edit, and
pattern placements that bake exactly like the plain hits they replace."""
import http.client
import json
import math
import os
import struct
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from helpers import FPS, rig_folder
import beatrig
from beatrig import server as sv

RIG = {"fps": FPS, "seed": 1, "nodes": [
    {"id": "kick", "op": "band", "lo": 40, "hi": 120},
    {"id": "kickN", "op": "normalize", "in": "kick"},
    {"id": "hits", "op": "notes", "track": "kick"},
    {"id": "pulse", "op": "envelope", "in": "hits", "shape": "punch", "length_beats": 0.5},
    {"id": "ring", "op": "chase", "in": "pulse", "channels": 4, "step_beats": 0.25},
], "layers": [{"id": "main", "intensity": [{"frame": 0, "value": 1},
                                           {"section": "break", "value": 0, "ease": "in_out", "fade_frames": 15}]}],
    "outputs": [
    {"id": "lights", "layer": "main", "signal": "ring", "target": {"kind": "light", "actors": [f"L{i}" for i in range(4)]},
     "min": 0, "max": 100},
    {"id": "glow", "signal": "kickN", "target": {"kind": "mpc", "asset": "/Game/T/MPC_T", "param": "Glow"}, "min": 0, "max": 1}]}
BPM = 120.0
HITS = [{"time": round(0.02 + b * 0.5, 6), "vel": 0.5 + 0.1 * (b % 4)} for b in range(8)]   # beats 0..7 at 120 bpm
NOTES = {"bpm_map": [{"time": 0.0, "bpm": BPM}], "grid_offset": 0.02, "sections": [{"name": "break", "time": 6.0}],
         "tracks": {"kick": HITS}}


class Server:
    def __init__(self, rig=RIG, notes=NOTES):
        d = rig_folder(json.loads(json.dumps(rig)), json.loads(json.dumps(notes)), duration=10.0)
        self.dir = d
        self.events = sv.Events()
        sv.Handler.state = sv.State(Path(d) / "rig.json", sv.Engine(), self.events)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), sv.Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def req(self, method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request(method, path, None if body is None else json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    def json(self, method, path, body=None):
        st, data = self.req(method, path, body)
        return st, json.loads(data)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def unpack(blob, dtype="<f4"):
    n = struct.unpack("<I", blob[:4])[0]
    assert (4 + n) % 4 == 0
    return json.loads(blob[4:4 + n]), np.frombuffer(blob[4 + n:], dtype)


def test_state_is_metadata_and_curves_match_the_bake():
    s = Server()
    try:
        st, d = s.json("GET", "/api/state")
        assert st == 200 and d["ok"]
        assert "peaks" not in d["analysis"]
        o = {x["id"]: x for x in d["bake"]["outputs"]}
        ch = o["lights"]["channels"][0]
        assert "values" not in ch and "keys" not in ch and ch["key_count"] > 0 and ch["actor"] == "L0"
        assert o["lights"]["hash"] and o["lights"]["components"] == 1
        # lod 0 = the bake's own per-frame values
        ref = beatrig.bake(beatrig.load_rig(os.path.join(s.dir, "rig.json")), None, json.loads(json.dumps(NOTES)))
        want = np.array([c["values"] for c in ref["outputs"][0]["channels"]], np.float32)
        st, blob = s.req("GET", "/api/curves?output=lights&lod=0")
        h, a = unpack(blob)
        assert st == 200 and h["layout"] == "values" and (h["channels"], h["frames"]) == want.shape
        assert np.array_equal(a.reshape(want.shape), want)
        # lod 2 = min/max per 16 frames
        st, blob = s.req("GET", "/api/curves?output=lights&lod=2")
        h, a = unpack(blob)
        m = a.reshape(h["channels"], h["count"], 2)
        assert h["bucket"] == 16 and h["count"] == math.ceil(want.shape[1] / 16)
        assert np.allclose(m[:, 0, 0], want[:, :16].min(axis=1)) and np.allclose(m[:, 0, 1], want[:, :16].max(axis=1))
        assert np.all(m[..., 0] <= m[..., 1])
        st, blob = s.req("GET", "/api/curves?band=kick&lod=0")
        assert st == 200 and unpack(blob)[0]["channels"] == 1
        st, blob = s.req("GET", "/api/peaks?lod=1")
        h, p = unpack(blob, np.uint8)
        assert st == 200 and h["rate"] == 50 and h["count"] == len(p) > 0
        assert s.req("GET", "/api/curves?output=nope")[0] == 404
        ops = s.json("GET", "/api/ops")[1]["ops"]
        assert "envelope" in ops and "fn" in ops["math"]["required"]
    finally:
        s.close()


def read_event(resp):
    """next server-sent event as (name, data); skips comments and retry lines"""
    name, data = None, None
    while True:
        line = resp.fp.readline().decode("utf-8").rstrip("\n")
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            data = json.loads(line[6:])
        elif line == "" and name:
            return name, data


def test_events_announce_changes():
    s = Server()
    try:
        c = http.client.HTTPConnection("127.0.0.1", s.port, timeout=10)
        c.request("GET", "/api/events")
        r = c.getresponse()
        assert r.status == 200 and r.getheader("Content-Type").startswith("text/event-stream")
        name, d = read_event(r)
        assert name == "rev" and d["rev"] == 1 and d["server"] == sv.SERVER_ID
        assert read_event(r)[0] == "live"
        s.json("POST", "/api/notify", {"source": "mcp", "action": "bake", "detail": "test"})
        name, d = read_event(r)
        assert name == "rev" and d["rev"] == 2 and d["activity"][-1]["action"] == "bake"
        c.close()
    finally:
        s.close()


def test_edit_renames_a_section_and_its_layer_key_together():
    s = Server()
    try:
        _, d = s.json("GET", "/api/state")
        rig, notes = d["rig"], d["notes"]
        notes["sections"][0]["name"] = "drop"
        st, err = s.json("POST", "/api/notes", notes)                    # alone it is invalid: the layer names "break"
        assert st == 400 and "break" in err["error"]
        rig["layers"][0]["intensity"][1]["section"] = "drop"
        st, ok = s.json("POST", "/api/edit", {"rig": rig, "notes": notes})
        assert st == 200 and ok["ok"] and ok["dirty"]
    finally:
        s.close()


def test_pattern_placements_bake_like_their_hits():
    """the page keeps placements as written; a pattern built from hits must bake to the same frames"""
    s = Server()
    try:
        _, d = s.json("GET", "/api/state")
        before = {o["id"]: o["hash"] for o in d["bake"]["outputs"]}
        notes = d["notes"]
        tempo = beatrig.Tempo([(0.0, BPM)], notes["grid_offset"])
        beats = [float(tempo.beats_at(h["time"])) for h in HITS[:4]]
        notes["patterns"] = {"k4": {"length_beats": 4, "hits": [{"beat": round(b, 9), "vel": h["vel"]} for b, h in zip(beats, HITS[:4])]}}
        notes["tracks"]["kick"] = [{"pattern": "k4", "at_beat": 0, "repeat": 1}] + HITS[4:]
        st, r = s.json("POST", "/api/notes", notes)
        assert st == 200, r
        assert {o["id"]: o["hash"] for o in r["bake"]["outputs"]} == before
        _, d2 = s.json("GET", "/api/state")
        assert d2["notes"]["tracks"]["kick"][0] == {"pattern": "k4", "at_beat": 0, "repeat": 1}   # kept as a placement
    finally:
        s.close()


def get_full(port, path):
    """(status, content type, body) for a GET"""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", path)
    r = c.getresponse()
    out = (r.status, r.getheader("Content-Type") or "", r.read())
    c.close()
    return out


def test_page_files_are_served_and_packaged():
    """every script and stylesheet the page loads is served with its type and is in the wheel (pyproject force-include)"""
    import re
    s = Server()
    try:
        st, ctype, html = get_full(s.port, "/")
        assert st == 200 and ctype.startswith("text/html")
        refs = re.findall(r'(?:src|href)="([\w.-]+\.(?:js|css))"', html.decode("utf-8"))
        assert {"app.js", "app.css", "preview3d.js"} <= set(refs)
        for ref in refs:
            st, ctype, body = get_full(s.port, "/" + ref)
            assert st == 200 and body
            assert ctype.startswith("text/javascript" if ref.endswith(".js") else "text/css"), (ref, ctype)
        st, ctype, popout = get_full(s.port, "/preview")                  # the pop-out window is the same page
        assert st == 200 and ctype.startswith("text/html") and popout == html
    finally:
        s.close()
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    for ref in refs + ["index.html"]:
        assert f'"web/{ref}" = "beatrig/_web/{ref}"' in pyproject, f"{ref} is not packaged in the wheel"
