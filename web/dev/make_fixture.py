"""Writes the preview page's dev fixture: a synthetic drum loop (loop.wav) plus rig.json and notes.json.

Only for developing the page without the engine (BEATRIG_FAKE=1). Stdlib only, deterministic (fixed seed).
    py -3 web/dev/make_fixture.py
"""
import json
import math
import random
import struct
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR, BPM, BARS = 44100, 120.0, 8
BEAT = 60.0 / BPM


def drum_loop():
    rnd = random.Random(7)
    n = int(SR * BEAT * 4 * BARS)
    out = [0.0] * n

    def add(start_s, length_s, fn):
        s0 = int(start_s * SR)
        for i in range(int(length_s * SR)):
            if s0 + i < n:
                out[s0 + i] += fn(i / SR)

    for b in range(BARS * 4):
        t = b * BEAT
        # kick on every beat: pitch sweep 140 -> 45 Hz, fast decay
        ph = [0.0]

        def kick(x, ph=ph):
            f = 45 + 95 * math.exp(-x * 30)
            ph[0] += 2 * math.pi * f / SR
            return 0.9 * math.sin(ph[0]) * math.exp(-x * 9)
        add(t, 0.45, kick)
        if b % 4 in (1, 3):  # snare on 2 and 4: noise + 190 Hz body
            add(t, 0.25, lambda x: (0.5 * (rnd.random() * 2 - 1) + 0.3 * math.sin(2 * math.pi * 190 * x)) * math.exp(-x * 18))
        for k in (0, 1):     # closed hats on the eighths (differenced noise = bright)
            prev = [0.0]

            def hat(x, prev=prev):
                v = rnd.random() * 2 - 1
                d, prev[0] = v - prev[0], v
                return 0.18 * d * math.exp(-x * 60)
            add(t + k * BEAT / 2 + 0.0, 0.08, hat)
    peak = max(abs(v) for v in out) or 1.0
    return [v / peak * 0.85 for v in out]


def main():
    samples = drum_loop()
    with wave.open(str(HERE / "loop.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(struct.pack("<h", int(v * 32767)) for v in samples))
    beats = int(BARS * 4)
    kick = [{"time": round(i * BEAT, 4), "vel": 0.95 if i % 4 == 0 else 0.75, "len_beats": 0.5, "shape": "punch"}
            for i in range(beats)]
    hat = [{"time": round(i * BEAT + BEAT / 2, 4), "vel": 0.5, "len_beats": 0.25, "shape": "razor"}
           for i in range(8, beats)]
    notes = {"bpm_map": [{"time": 0.0, "bpm": BPM}], "grid_offset": 0.0, "tracks": {"kick": kick, "hat": hat}}
    rig = {
        "audio": "loop.wav", "fps": 30, "frames": None, "audio_offset_frames": 0,
        "tempo": {"bpm": None, "map": []}, "seed": 1, "notes": "notes.json",
        "nodes": [
            {"id": "kick", "op": "band", "lo": 40, "hi": 120},
            {"id": "kickN", "op": "normalize", "in": "kick", "range_db": 24},
            {"id": "kickS", "op": "smooth", "in": "kickN", "attack_ms": 15, "release_ms": 220},
            {"id": "hits", "op": "notes", "track": "kick"},
            {"id": "pulse", "op": "envelope", "in": "hits", "shape": "punch", "length_beats": 0.5},
            {"id": "ring", "op": "chase", "in": "pulse", "channels": 8, "step_beats": 0.125},
            {"id": "hat", "op": "notes", "track": "hat"},
            {"id": "hatE", "op": "envelope", "in": "hat", "shape": "razor", "length_beats": 0.25},
        ],
        "layers": [{"id": "main", "intensity": [{"frame": 0, "value": 1}]}],
        "outputs": [
            {"id": "lights", "layer": "main", "signal": "ring",
             "target": {"kind": "light", "actors": [f"Light_{i}" for i in range(8)], "property": "intensity"},
             "min": 500, "max": 20000},
            {"id": "glow", "signal": "kickS", "target": {"kind": "mpc", "asset": "/Game/Demo/MPC_Demo", "param": "Glow"},
             "min": 0.2, "max": 3},
            {"id": "strobe", "signal": "hatE", "target": {"kind": "mpc", "asset": "/Game/Demo/MPC_Demo", "param": "Strobe"},
             "min": 0, "max": 1},
        ],
        "bake": {"key_tolerance": 0.002},
    }
    (HERE / "notes.json").write_text(json.dumps(notes, indent=1) + "\n", encoding="utf-8")
    (HERE / "rig.json").write_text(json.dumps(rig, indent=2) + "\n", encoding="utf-8")
    print("wrote", HERE / "loop.wav", len(samples) / SR, "s")


if __name__ == "__main__":
    main()
