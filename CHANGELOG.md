# Changelog

All notable changes to beatrig. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [semantic versioning](https://semver.org/).

## [0.1.0] - unreleased

First public version.

### Added
- Offline audio analysis with a cache: causal band envelopes on a 1 ms grid, octave-band spectral-flux onsets, and a
  verified beat grid with a tempo map.
- The motion graph (`rig.json`): sources, shaping and channel ops (chase, cycle, ripple and more), eased layers,
  and outputs for lights, material parameters, material parameter collections and Niagara parameters.
- Editable hit lanes (`notes.json`), filled from onsets or the beat grid, with snapping.
- Deterministic bake (`bake.json`): per-frame values plus simplified linear keys that keep every hit.
- CLI: `beatrig analyze | notes | bake | check | serve`.
- Preview page on localhost: waveform, beat grid, note lanes, curves, a light-ring schematic and a param panel.
- MCP server (`beatrig-mcp`, optional `[mcp]` extra) covering analysis, rig and notes editing, bake, a `plot` image
  tool, the preview page, and Unreal apply/check/undo over Remote Control.
- Unreal applier (`ue/beatrig_apply.py`), untested in an editor so far; its key and tick maths (`ue/keys.py`) are
  unit tested.
