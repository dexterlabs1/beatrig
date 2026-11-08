"""Stems: the core install never imports torch; stem nodes explain what to run when stems are missing; with the
[stems] extra installed, a separation is cached by content hash and band/onsets read the drums stem."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from helpers import ROOT, tmpdir, write_loop
import beatrig
from beatrig import analysis as A
from beatrig import stems as S


def test_core_never_imports_torch():
    code = ("import sys, json, os, tempfile; sys.path.insert(0, %r); sys.path.insert(0, os.path.join(%r, 'tests'))\n"
            "import beatrig; from helpers import write_loop\n"
            "d = tempfile.mkdtemp(); write_loop(os.path.join(d, 'l.wav'), 120)\n"
            "r = beatrig.Rig.from_dict({'audio': 'l.wav', 'fps': 30, 'nodes': [{'id': 'b', 'op': 'band', 'lo': 40, 'hi': 120}], 'outputs': []}, d)\n"
            "beatrig.bake(r, notes={})\n"
            "print(json.dumps([m for m in ('torch', 'demucs', 'librosa', 'numba') if m in sys.modules]))") % (ROOT, ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def test_missing_stems_say_what_to_run():
    d = tmpdir()
    write_loop(os.path.join(d, "loop.wav"), 120, duration=6.0)
    rig = beatrig.Rig.from_dict({"audio": "loop.wav", "fps": 30, "nodes": [
        {"id": "k", "op": "band", "lo": 40, "hi": 120, "stem": "drums"},
        {"id": "on", "op": "onsets", "in": "k"}], "outputs": []}, d)
    assert S.used_stems(rig.nodes) == ["drums"]
    assert S.cached(os.path.join(d, "loop.wav")) is None
    try:
        beatrig.analyze(rig)
        raise AssertionError("analyzed stems that do not exist")
    except beatrig.RigError as e:
        assert "beatrig stems" in str(e)
    bad = beatrig.Rig.from_dict({"frames": 30, "nodes": [{"id": "k", "op": "band", "lo": 40, "hi": 120, "stem": "piano"}], "outputs": []}, d)
    assert any("stem 'piano'" in i["msg"] for i in beatrig.check(bad))


def test_separation_and_stem_nodes():
    pytest.importorskip("demucs")
    pytest.importorskip("torch")
    d = tmpdir()
    write_loop(os.path.join(d, "loop.wav"), 120, duration=8.0)
    paths = S.separate(os.path.join(d, "loop.wav"), log=lambda *a: None)
    assert set(paths) == set(S.STEMS) and all(os.path.exists(p) for p in paths.values())
    assert S.cached(os.path.join(d, "loop.wav")) == paths and S.separate(os.path.join(d, "loop.wav")) == paths   # cached
    meta = json.load(open(os.path.join(os.path.dirname(paths["drums"]), "stems.json"), encoding="utf-8"))
    assert meta["model"] == "htdemucs"
    A._MEMO.clear()
    rig = beatrig.Rig.from_dict({"audio": "loop.wav", "fps": 30, "nodes": [
        {"id": "k", "op": "band", "lo": 40, "hi": 120, "stem": "drums"},
        {"id": "kmix", "op": "band", "lo": 40, "hi": 120},
        {"id": "on", "op": "onsets", "in": "k", "threshold": 0.3, "min_gap_ms": 150}], "outputs": []}, d)
    an = beatrig.analyze(rig)
    assert set(an.stems) == set(S.STEMS) and an.stems_info["used"]           # cached stems are always used (auto)
    sig = beatrig.evaluate(rig, an, {})
    assert sig["k"].v.max() > 0 and not np.array_equal(sig["k"].v, sig["kmix"].v)
    hits = np.flatnonzero(sig["on"].v[0])
    beats = np.arange(16) * 15                                       # a kick on every beat at 120 bpm, 30 fps
    assert len(hits) >= 14 and all(0 <= h - beats[np.argmin(np.abs(beats - h))] <= 2 for h in hits)   # on time, never early


def test_chunked_separation_matches_one_pass():
    """3 s chunks with 1 s context and a crossfade give (nearly) the same stems as one pass over the whole file"""
    pytest.importorskip("demucs")
    import soundfile as sf
    d1, d2 = tmpdir(), tmpdir()
    write_loop(os.path.join(d1, "loop.wav"), 100, duration=9.0)
    write_loop(os.path.join(d2, "loop.wav"), 100, duration=9.0)
    one = S.separate(os.path.join(d1, "loop.wav"), log=lambda *a: None)
    chunked = S.separate(os.path.join(d2, "loop.wav"), chunk_s=3.0, context_s=1.0, log=lambda *a: None)
    mix, _ = sf.read(os.path.join(d1, "loop.wav"))
    ref = np.sqrt(np.mean(mix ** 2))
    for s in S.STEMS:
        a, _ = sf.read(one[s])
        b, sr = sf.read(chunked[s])
        assert a.shape == b.shape and sr == 44100
        err = np.sqrt(np.mean((a - b) ** 2)) / ref                   # difference relative to the mix level
        assert err < 0.05, (s, err)
