# Changelog

All notable changes to beatrig. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [semantic versioning](https://semver.org/).

## [0.2.0] - 2026-11-07

Live editing in Unreal, colour, spatial waves across real scene layouts, flash safety, and a much lighter core.

### Added
- **Live link** (`beatrig.live.Link`): a persistent Remote Control session that pushes only the outputs whose content
  changed. Measured on UE 5.8.3: slider release to keys evaluated in the editor in 104 ms median (re-bake 53 ms, push 49 ms).
  Two-way playhead sync (`playhead_get` / `playhead_set`, about 25 / 45 ms).
- **Scene scan** (`beatrig scan`, MCP `ue_scan`): actors, positions, light/mesh/Niagara/camera components, material slot
  parameters, MPCs and their parameters, sequences, and the dome camera (OWL 360 Domemaster detected). Writes `scene.json`.
- **Groups**: `rig.groups` with `@name` expansion anywhere an actor list is accepted.
- **Colour signals**: `gradient`, `palette_cycle`, `mix_color`, `scale_color`, seven built-in palettes, and colour outputs
  for MPC vector params, light colour and material vector params (4-vector key simplification).
- **Transform outputs** (location / rotation / scale per axis, additive by default) and the `accumulate` op for
  continuous spin and phase.
- **Per-hit flow ops**: `push`, `shuffle_hits`, `every_n`, `probability` (hash-seeded, so editing one hit re-rolls
  nothing else), `quantize` (delay-only, so still causal).
- **Spatial ops** driven by scanned positions: `spatial_delay` (from a point, axis or plane, in cm/s or beats per metre),
  `dome_wave` (`zenith_rings` / `azimuth_sweep` across a tilted fulldome) and k-nearest-neighbour `ripple`.
- **Flash safety**: `rig.safety` makes `check` and `bake` fail any light or emissive output that flashes more than
  `max_flashes_per_s` (default 3) in any one-second window; `flash_limit` enforces it causally; `plot` overlays it.
- **Sections and patterns** in `notes.json`: auto-detected song sections that layers can key on, and linked loops
  (edit the pattern, every copy follows). CLI `notes sections init`, `notes pattern make|place|list`.
- **MIDI file import**: `notes import-midi rig.json song.mid --map 36:kick,38:snare`.
- **Tempo candidates**: `analyze` reports the main tempo plus x2, x1/2, x3/2, x2/3, x4/3 and x3/4 with a confidence
  each; `rig.tempo.bpm` pins one.
- **Long files**: `audio_in_s` / `audio_out_s` windows into a long mix (window values equal the full file's exactly),
  `channel_weights` for multichannel (e.g. 5.1) mixes, and `beatrig timeline import` for FCPXML and EDL slot lists.
- **Stems** (optional `[stems]` extra, Demucs htdemucs, CUDA or CPU, streamed in chunks): `band`, `onsets` and
  `loudness` take `"stem": "drums"` etc. With a sensitive kick setting, kick precision went from 0.71 on the mix to
  1.0 on the drums stem (recall 1.0 both).
- Unreal applier: colour, transform and Niagara writers, all tested live (apply, key-for-key check, re-apply, undo).
- Preview page: live toggle, playhead link, metronome, tap-record, patterns, section lane, colour strips, plan and
  dome views from the scene scan, add/remove nodes and outputs, flash-safety bands, binary curve transport and a
  server-sent event stream.
- MCP: `ue_scan`, `live_push`, `playhead_get`, `playhead_set`; flash overlay in `plot`.

### Changed
- **Core dependencies are now numpy, scipy and soundfile only** (librosa, and with it numba/llvmlite, removed).
  Fresh install 439 MB to 227 MB; `import beatrig` 0.57 s to 0.21 s.
- 60-minute show, measured: fresh analysis 28 s (was 43 s), peak memory 356 MB (was 4.4 GB), bake 4.1 s.
- `check` now bakes in memory to run the flash rule (`--static` skips it). The analysis cache version is now 4, so
  existing caches rebuild once.
- `delay` passes its input kind through, so `notes -> delay -> envelope` is valid.

### Fixed
- Unreal applier on UE 5.8: `MovieSceneTimeUnit`, `ComponentMaterialInfo`, and material parameter keys (which take
  a `MaterialParameterInfo`). Light, material and MPC targets now apply and check in a real editor.
- `undo` now removes the bindings beatrig created (and only those).
- Binding de-duplication: UE Python's `str(Guid)` includes a memory address; beatrig uses `Guid.to_string()`.
- MCP `list_ops` failed to serialise ops whose kind follows their input.

## [0.1.0] - 2026-11-07

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
