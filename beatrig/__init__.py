"""beatrig: an offline, causal music -> animation engine. Analyse audio, evaluate a JSON motion graph (rig.json) and
bake deterministic per-frame values plus simplified linear keys (bake.json) for Sequencer-style timelines.

Stable public API:
    load_rig(path) -> Rig
    analyze(rig) -> Analysis | None
    evaluate(rig, analysis=None, notes=None) -> dict[node_id, Signal]
    bake(rig, analysis=None, notes=None, out=None) -> dict          (the bake.json structure)
    bake_from_objects(rig_dict, notes_dict, base_dir) -> dict       (unsaved rig/notes, nothing written)
    check(rig, notes=None) -> list[issue]
    load_notes(path) -> dict, save_notes(path, notes) -> dict
"""
from .analysis import Analysis
from .bake import analyze, bake, bake_from_objects, evaluate, layer_curve, rdp, write_bake
from .check import check
from .notes import load_notes, save_notes
from .ops import OpError
from .rig import Rig, RigError, load_rig
from .signal import Signal
from .tempo import Tempo

__version__ = "0.1.0"
__all__ = ["Analysis", "Rig", "RigError", "OpError", "Signal", "Tempo", "load_rig", "analyze", "evaluate", "bake",
           "bake_from_objects", "write_bake", "check", "load_notes", "save_notes", "layer_curve", "rdp", "__version__"]
