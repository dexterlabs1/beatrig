"""Standard MIDI File (SMF) import, pure Python (no dependencies).

read_smf(bytes) parses format 0/1/2 files: delta times, running status, note on/off (note on with velocity 0 is a
note off), sysex and meta events, the tempo map (FF 51) and SMPTE time division. import_midi() turns mapped notes into
notes.json tracks: time in seconds of audio (MIDI time 0 + offset_s), vel = velocity / 127, optional len_beats from the
note length, and a bpm_map from the file's tempo map.
"""
import struct


class MidiError(ValueError):
    pass


def _vlq(data, i):
    """variable-length quantity at data[i] -> (value, next index)"""
    v = 0
    for _ in range(4):
        if i >= len(data):
            raise MidiError("truncated variable-length value")
        b = data[i]
        i += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80:
            return v, i
    raise MidiError("variable-length value longer than 4 bytes")


def _chunks(data):
    i = 0
    while i + 8 <= len(data):
        tag, n = data[i:i + 4], struct.unpack(">I", data[i + 4:i + 8])[0]
        yield tag, data[i + 8:i + 8 + n]
        i += 8 + n


def _track(data):
    """one MTrk -> [(abs_tick, event)], event = ("note", ch, note, vel) | ("tempo", uspq) | ("other",)"""
    out, i, tick, status = [], 0, 0, None
    while i < len(data):
        delta, i = _vlq(data, i)
        tick += delta
        if i >= len(data):
            raise MidiError("truncated event")
        b = data[i]
        if b == 0xFF:                                        # meta
            if i + 1 >= len(data):
                raise MidiError("truncated meta event")
            typ = data[i + 1]
            n, i = _vlq(data, i + 2)
            payload = data[i:i + n]
            i += n
            if typ == 0x51 and n == 3:
                out.append((tick, ("tempo", (payload[0] << 16) | (payload[1] << 8) | payload[2])))
            elif typ == 0x2F:
                break
            continue
        if b in (0xF0, 0xF7):                                # sysex
            n, i = _vlq(data, i + 1)
            i += n
            continue
        if b & 0x80:
            status = b
            i += 1
        elif status is None:
            raise MidiError("running status with no previous status byte")
        hi = status & 0xF0
        n = 1 if hi in (0xC0, 0xD0) else 2
        if i + n > len(data):
            raise MidiError("truncated channel message")
        d = data[i:i + n]
        i += n
        if hi == 0x90:
            out.append((tick, ("note", status & 0x0F, d[0], d[1])))
        elif hi == 0x80:
            out.append((tick, ("note", status & 0x0F, d[0], 0)))
    return out


def read_smf(data: bytes) -> dict:
    """parse an SMF -> {"format", "division", "smpte": (fps, ticks_per_frame) | None, "tempos": [(tick, uspq)],
    "notes": [(on_tick, off_tick | None, channel 0-15, note, velocity 1-127)]} (notes sorted by on_tick)"""
    chunks = list(_chunks(bytes(data)))
    if not chunks or chunks[0][0] != b"MThd" or len(chunks[0][1]) < 6:
        raise MidiError("not a standard MIDI file (no MThd header)")
    fmt, ntrk, div = struct.unpack(">HHH", chunks[0][1][:6])
    smpte = None
    if div & 0x8000:
        fps = 256 - (div >> 8)
        smpte = (29.97 if fps == 29 else float(fps), div & 0xFF)
    elif div == 0:
        raise MidiError("division is 0")
    tempos, notes = [], []
    for ti, (tag, body) in enumerate(c for c in chunks[1:] if c[0] == b"MTrk"):
        open_ = {}
        for tick, ev in _track(body):
            if ev[0] == "tempo":
                if ev[1] > 0:
                    tempos.append((tick, ev[1]))
            elif ev[0] == "note":
                _, ch, note, vel = ev
                key = (ti, ch, note)
                if vel > 0:
                    open_.setdefault(key, []).append(len(notes))
                    notes.append([tick, None, ch, note, vel])
                elif open_.get(key):
                    notes[open_[key].pop(0)][1] = tick       # first in, first out per (track, channel, note)
    tempos.sort(key=lambda t: t[0])
    notes.sort(key=lambda n: (n[0], n[2], n[3]))
    return {"format": fmt, "division": div, "smpte": smpte, "tempos": tempos, "notes": [tuple(n) for n in notes]}


class TickClock:
    """MIDI ticks -> seconds and beats through the tempo map (default 120 bpm until the first tempo event)"""

    def __init__(self, smf):
        self.smpte = smf["smpte"]
        self.ppq = None if self.smpte else smf["division"]
        segs = [(0, 500000)]
        for tick, us in smf["tempos"]:
            if tick == segs[-1][0]:
                segs[-1] = (tick, us)
            else:
                segs.append((tick, us))
        self.segs = segs
        self.sec = [0.0]
        for (t0, us), (t1, _) in zip(segs[:-1], segs[1:]):
            self.sec.append(self.sec[-1] + (t1 - t0) * us / 1e6 / self.ppq if self.ppq else 0.0)

    def seconds(self, tick):
        if self.smpte:
            fps, tpf = self.smpte
            return tick / (fps * tpf)
        k = max(i for i, (t, _) in enumerate(self.segs) if t <= tick)
        t0, us = self.segs[k]
        return self.sec[k] + (tick - t0) * us / 1e6 / self.ppq

    def beats(self, tick):
        if self.smpte:
            return self.seconds(tick) * 2.0                  # SMPTE files carry no tempo: 120 bpm
        return tick / self.ppq

    def bpm_map(self, offset_s=0.0):
        if self.smpte:
            return [{"time": round(offset_s, 6), "bpm": 120.0}]
        return [{"time": round(self.sec[k] + offset_s, 6), "bpm": round(60e6 / us, 4)} for k, (_, us) in enumerate(self.segs)]


def parse_map(spec) -> dict:
    """'36:kick,38:snare' (or a dict) -> {36: 'kick', 38: 'snare'}; a note may be a number or a name like C1/F#2
    (C-1 = 0, so C1 = 24 and C2 = 36, the drum-map convention)"""
    if isinstance(spec, dict):
        return {note_number(k): str(v) for k, v in spec.items()}
    out = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition(":")
        if not v:
            raise MidiError(f"bad --map entry {part!r} (want NOTE:track)")
        out[note_number(k)] = v.strip()
    return out


NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_number(k):
    s = str(k).strip()
    if s.lstrip("-").isdigit():
        n = int(s)
    else:
        name = s[0].upper()
        if name not in NAMES:
            raise MidiError(f"bad note {s!r}")
        acc = s[1:].count("#") - s[1:].lower().count("b")
        octave = int(s[1:].replace("#", "").replace("b", "").replace("B", "") or 0)
        n = NAMES[name] + acc + 12 * (octave + 1)
    if not 0 <= n <= 127:
        raise MidiError(f"note {s!r} out of range 0-127")
    return n


def import_midi(path, mapping, channel=None, offset_s=0.0, lengths=False) -> dict:
    """Read a .mid and return {"bpm_map", "grid_offset", "tracks": {name: [hits]}, "stats"}.
    mapping: {note: track} or '36:kick,38:snare'. channel: 1-16 to keep one MIDI channel (None = all).
    offset_s: audio time of MIDI time 0. lengths: also write len_beats from each note's length."""
    with open(path, "rb") as fh:
        smf = read_smf(fh.read())
    m = parse_map(mapping)
    if channel is not None and not 1 <= int(channel) <= 16:
        raise MidiError("channel must be 1-16")
    clock = TickClock(smf)
    tracks = {name: [] for name in dict.fromkeys(m.values())}
    skipped = 0
    for on, off, ch, note, vel in smf["notes"]:
        if note not in m or (channel is not None and ch != int(channel) - 1):
            skipped += 1
            continue
        h = {"time": round(clock.seconds(on) + float(offset_s), 6), "vel": round(vel / 127.0, 4)}
        if lengths and off is not None and off > on:
            h["len_beats"] = round(clock.beats(off) - clock.beats(on), 4)
        tracks[m[note]].append(h)
    return {"bpm_map": clock.bpm_map(float(offset_s)), "grid_offset": round(float(offset_s), 6), "tracks": tracks,
            "stats": {"format": smf["format"], "notes": len(smf["notes"]), "skipped": skipped,
                      "tempo_changes": len(clock.segs), "smpte": smf["smpte"] is not None}}
