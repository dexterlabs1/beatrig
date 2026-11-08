# Changelog

All notable changes to beatrig. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [semantic versioning](https://semver.org/).

## [Unreleased]

Better ears: stems by default, neural beats and bars, drum parts with no band to tune, labelled song sections.

### Added
- **Neural beats and bars** (optional extra `[ml]`: the Beat This! tracker, MIT). `analysis.beats` `auto` (default)
  uses it when installed. Every beat is checked against a real onset and written at that onset's time, never at the
  model's frame, so beats stay "only where an onset supports them, never ahead of it". Each beat carries `bar` and
  `pos`; analysis.json carries `bars` (meter, downbeats, confidence); the grid offset becomes a downbeat's phase, so
  patterns and bar windows lock to real bars. A failed model run (no network for the first download) leaves the
  classic grid, and the analysis says why. beatrig computes the model's spectrogram itself (no torchaudio import).
- `beats` op `anchor: "bar"`: `every`/`offset` count on the bar grid (`every: 4` = downbeats), unshifted by beats
  without an onset. `notes init --from downbeats`, MCP `notes_init(source="downbeats")`.
- **Stems by default**: `analysis.stems` `auto` (default) uses cached stems and separates once when the `[stems]` extra
  and a CUDA GPU are there; `on` separates on any device; `off` reads only stems that nodes name. The analysis
  summary (CLI, MCP `analyze`, `Analysis.stems_label()`) says whether stem features read stems or the mix, and why.
- **`drum` op** (`part` kick | snare | hats, `source` auto | stem | mix): drum-part hits from the drums stem (else the
  mix) with built-in bands, stem pre-echo removed, snares told from kick clicks and hats from clicks and clap tails.
  `notes init --from drum:kick`, MCP `notes_init(source="kick")`.
- **Song structure**: `song_structure(analysis)`; `notes sections init` labels sections on bar lines (`intro`,
  `groove`, `dip`, `build`, `breakdown`, `break`, `drop`, `outro`, numbered repeats) from a multi-scale novelty over
  stem levels, drum-part densities and band levels, plus drums/bass presence changes. `--method energy` keeps the
  v0.2 rule. MCP `describe_audio` lists the sections.
- `BEATRIG_EXTRAS=off` makes `auto` analysis settings behave like a core install (the test suite sets it).

### Measured (one 4.5-minute 128 BPM house track, RTX 5090; the synthetic demo has exact truth)
| | before | after |
|---|---|---|
| Tempo map | 127.68 BPM (0.2 s off the beats by the end) | 128.00 BPM |
| Beat F-measure vs the true grid (±70 ms) | 0.987 | 0.992 |
| Downbeat F-measure (`beats every 4`) | 0.10 (demo: 0.0) | 0.99 (demo: 0.96) |
| Section boundaries within ±1 bar of a hand annotation | 7/9, precision 7/11 | 9/9, precision 9/10 |
| Section labels matching the hand annotation | 2/10 | 10/10 (8/9 boundaries and weaker labels without stems) |
| Snare precision / recall, demo drums stem (`onsets` 180-2500 Hz vs `drum`) | 0.29 / 0.86 | 0.96 / 0.79 |
| Hats precision / recall, demo drums stem (`onsets` 6-16 kHz vs `drum`) | 1.00 / 0.97 | 1.00 / 0.77 (the demo's hats on every snare are left to the snare track) |
| Real track, drums stem: snare hits on beats 2 and 4 / hats hits on beats 2 and 4 (clap tails) | 40 % / 27 % | 66 % / 1 % |
| Cold analysis (stems cached) / separation / model | 1.7 s / - / - | 9.0 s / 5.1 s / 1.4 s |

### Changed
- Each stem has its own analysis cache folder (they used to share one file and overwrite each other).
- `stems.available()` looks the packages up without importing torch.

### Fixed
- Preview page: with the playhead linked to a sequence that has its own audio track, the page mutes itself, so the
  music isn't heard twice. `playhead_get` reports `audio`.
- Analysis: the tempo of a long passage drifted. The line fit through the verified beats indexed each beat by
  `round((t - t0) / T)` with the comb's period, which sits on the 5 ms lag grid; the error piled up until far beats
  took the wrong index. A 128 BPM track read as 127.68 and its tempo map was 0.7 s late by the fourth minute (so
  `lfo`, patterns, `quantize`, beat delays and `subdivide` drifted). Beats are now indexed by the periods between
  neighbours. The beat times themselves were always right. Analysis caches rebuild once (analysis version 5).

## [0.3.0] - 2026-11-07

Audio-reactive gains on top of a show's existing keys, without touching them.

### Added
- **Output blend** `multiply` / `add`. Baked values are gains or offsets, applied over the target's existing animation
  ("base") in the sequence. The base section stays active and untouched: beatrig writes an additive layer holding
  `base·(gain−1)` (or the offset), so a section shared with other parameters (Sequencer's default for an MPC or a
  material slot) works. With no base, an absolute section of static × gain is written. Edits to the base are detected
  on the next push and the layer is rewritten. `undo` removes only the layer. Tested live in UE 5.8.3 on MPC scalar
  (shared Glow + Tint section), MPC colour, light intensity and a material scalar, in both modes.
- `beats`: `min_confidence` and `velocity: confidence`; `notes init --from beats --min-conf`.
- `flash_limit` mode `strongest`: an offline choice that keeps the strongest of close hits at its own frame.
- `rig.cache_dir`, so scene rigs cut from one show mix share one analysis.
- `bake.start_frame`: bake only part of the sequence; statistics use only that range.
- Timeline import records source in-points and transitions, and `--handles N` bakes each clip plus its dissolve
  handles only.

### Changed
- `flash_limit` mode `first` is the documented default (`drop` stays as an alias). Every mode caps hits per one-second
  window, so fractional rates are safe.
- bake.json carries `"blend"` (non-replace outputs), `"start_frame"` (when not 0) and `"key_tolerance"` (when not the
  default). Rigs that use none of these bake byte-identically to 0.2.0.

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
- FCPXML import read a compound clip's `<sequence>` (in `<resources>`) instead of the project's edit when the compound
  came first in the file; it now prefers `<project>/<sequence>`.
- A hit saved exactly on a frame start could land one frame late after a round trip through `notes.json` (times are
  stored to 1 microsecond); frame placement now rounds to 1e-4 of a frame first.

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
