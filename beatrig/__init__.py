"""beatrig: an offline, causal music -> animation engine. Analyse audio, evaluate a JSON motion graph (rig.json) and
bake deterministic per-frame values plus simplified linear keys (bake.json) for Sequencer-style timelines.

Stable public API:
    load_rig(path) -> Rig
    analyze(rig) -> Analysis | None                                 (meta["tempo_candidates"]: x2, x1/2, x3/2, ... with confidence)
    evaluate(rig, analysis=None, notes=None) -> dict[node_id, Signal]
    bake(rig, analysis=None, notes=None, out=None, strict=True) -> dict   (the bake.json structure, with a "safety" summary)
    bake_from_objects(rig_dict, notes_dict, base_dir) -> dict       (unsaved rig/notes, nothing written)
    check(rig, notes=None, bake=None) -> list[issue]                (bake: also the flash-safety rule on its values)
    load_notes(path) -> dict, save_notes(path, notes) -> dict       (pattern placements expand at load, save keeps them linked)

v0.2 additions:
    flash_report(values, fps, lo, hi, max_flashes_per_s, min_swing) -> dict   photosensitivity: per-second flash counts
    expand_patterns(notes), track_hits(notes, track), make_pattern(...), place_pattern(...)   linked note loops
    section_boundaries(analysis) -> (sections, levels), name_sections(sections)                song sections from energy
    import_midi(path, mapping, channel=None, offset_s=0) -> dict    Standard MIDI File -> notes tracks + bpm map
    import_timeline(path, audio, out_dir) -> dict                   FCPXML / EDL -> one window rig per clip
    PALETTES, parse_color(c)                                        named palettes, '#rrggbb' -> linear RGBA
    dome_angles(camera, positions) -> (zenith, azimuth)             fulldome geometry used by dome_wave

v0.3 additions (formats in docs/ops.md):
    output "blend": replace | multiply | add                       gains / offsets on the target's own animation
    rig "cache_dir"                                                 scene rigs share one show-mix analysis
    rig "bake": {"start_frame": S}                                  bake only sequence frames S.. (bake.json "start_frame")
    import_timeline(path, audio, out_dir, handles=N) -> dict        source in-point -> audio_offset_frames, transitions

Unreleased additions:
    rig "analysis": {"beats": auto|neural|classic, "stems": auto|on|off}   neural beats/bars ([ml]), stems by default
    song_structure(analysis) -> [section]                           labelled sections on bar lines
    ops: drum (part kick|snare|hats), beats "anchor": "bar"
"""
from .analysis import Analysis, name_sections, section_boundaries, song_structure
from .bake import analyze, bake, bake_from_objects, evaluate, layer_curve, rdp, resolve_layer_keys, write_bake
from .check import check
from .color import PALETTES, parse_color
from .midi import import_midi
from .notes import expand_patterns, load_notes, make_pattern, place_pattern, save_notes, track_hits
from .ops import OpError
from .rig import Rig, RigError, load_rig
from .safety import flash_report
from .signal import Signal
from .spatial import dome_angles
from .tempo import Tempo
from .timeline import import_timeline

__version__ = "0.3.0"
__all__ = ["Analysis", "Rig", "RigError", "OpError", "Signal", "Tempo", "load_rig", "analyze", "evaluate", "bake",
           "bake_from_objects", "write_bake", "check", "load_notes", "save_notes", "layer_curve", "rdp", "__version__",
           "flash_report", "expand_patterns", "track_hits", "make_pattern", "place_pattern", "section_boundaries",
           "name_sections", "song_structure", "import_midi", "import_timeline", "PALETTES", "parse_color", "dome_angles", "resolve_layer_keys"]
