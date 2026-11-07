# beatrig ops reference (v0)

The file formats are specified in [the design doc](design.md). This page is the op-by-op reference, plus the semantics
the engine settles where the design doc leaves room. Params in brackets are optional (shown with their defaults).
Every node also accepts `id`, `op`, `in`, `label`, `comment`, and any key starting with `_`.

A **signal** is `float32[channels, frames]`. A **trigger** is a signal that is non-zero only on hit frames, where the
value is the hit velocity (0..1]. Every signal also carries a **hit mask**: the frames that key simplification must
keep. Sources set it, delays and chases move it, `math` unions it, and shaping ops pass it through.

## Rules every op follows
1. **Never ahead of the audio** (ITU-R BT.1359 audio/video timing). A frame reads the audio only up to its own
   start time. A hit lands on the first frame at or after its onset (`frame = ceil(t * fps) + audio_offset_frames`).
   All delays are >= 0. Fractional delays interpolate between the two past frames (values) or round **up** (triggers).
2. **Eased motion.** Continuous signals are smoothed. Only trigger envelopes rise instantly. Keys are linear.
3. **Deterministic.** Seeds are explicit (`seed` on the node, else the rig's `seed`). There is no wall-clock input.

## Sources
| op | params | output | notes |
|---|---|---|---|
| `band` | `lo`, `hi` (Hz), [`attack_ms` 10, `release_ms` 150, `weights` null] | 1 ch, raw linear amplitude | Butterworth band (order 4, low/high-pass at the edges). Per-channel power is weighted and summed, so channels can't phase-cancel. Then a one-pole follower runs on an exact 1 ms grid, sampled at each frame's start. `weights` (one per audio channel) defaults to the rig's `channel_weights`, else equal. |
| `loudness` | [`attack_ms` 10, `release_ms` 150, `weights`] | 1 ch | Same as `band`, with the ITU-R BS.1770 K-weighting filter (computed for any sample rate). |
| `onsets` | [`in`, `threshold` 0.3, `min_gap_ms` 80] | trigger, 1 ch | Causal spectral-flux peaks on a 5 ms grid. `in` is a `band` node id, an inline `{lo, hi}`, or omitted for the full-range octave-band flux. Velocity is the peak relative to the 95th percentile of all peaks. Peaks below `threshold` are dropped. The onset time is when the rise is known (`(k+1)·5 ms`). |
| `beats` | [`every` 1, `offset` 0, `subdivide` 1] | trigger, 1 ch | The **verified** beat grid from the analysis. Beats exist only where a real onset supports them. `offset` is an index offset into the beat list for `every`. `subdivide` inserts evenly spaced hits sized from the **previous** interval (causal), never past the next real beat. Subdivision velocity is 0.6 × the beat's. |
| `notes` | `track` | trigger, 1 ch | Hits from `notes.json`. Per-hit `len_beats` and `shape` travel with the trigger to `envelope`. |
| `lfo` | [`shape` sine, `rate_beats` \| `rate_hz`, `phase` 0, `channels` 1, `phase_step` 0] | N ch, 0..1 | `rate_beats` = beats per cycle, locked to the tempo map and grid offset. `rate_hz` = cycles per second of audio time. Shapes: `sine` (starts at 0), `tri`, `saw_up`, `saw_down`, `square` (check warns: instant edges), `semi_sine`. Channel c gets `phase + c·phase_step`. Default rate: 1 beat. |
| `const` | [`value` 0, `channels` 1] | N ch | |

## Shaping
| op | params | notes |
|---|---|---|
| `normalize` | [`range_db` 24, `hi_pct` 99.5, `lo_pct` 10] | dB scale to 0..1 over the whole bake (all channels together): `hi_pct` maps to 1, and the floor is max(`lo_pct`, top − `range_db`). This is a static gain, so timing is unaffected. |
| `smooth` | `attack_ms`, `release_ms` (both > 0) | One-pole up/down follower per channel at frame rate. |
| `spring` | [`freq_hz` 2, `damping` 0.6] | 2nd-order follower ω²/(s²+2ζωs+ω²), bilinear-discretised, starts at rest. Damping below 1 overshoots. |
| `curve` | [`gamma` 1] or `preset` (`s_curve`, `ease_in`, `ease_out`) | Negative values clip to 0. Presets clip to 0..1. |
| `remap` | `in_lo`, `in_hi`, `out_lo`, `out_hi`, [`clamp` true] | |
| `threshold` | [`level` 0.5, `cooldown_ms` 100] | Fires on a rising edge through `level` (reads this frame and the previous one only), velocity 1. |
| `envelope` | `in`=trigger, [`shape` punch, `length_beats` 0.5 \| `length_ms`, `attack_ms` 0, `decay_ms` 100, `sustain` 0.7, `release_ms` 200, `points`] | Shapes: `binary`, `punch` (exponential), `gloss` (cosine), `razor` ((1−t)⁴), `rush` (eased rise to 80 %, then a fast fall), `saw_down`, `adsr` (gate = length, then release), `custom` (`points` = [[t 0..1, v], ...]). Velocity-scaled. Overlapping hits take the max. Per-hit `len_beats`/`shape` from notes override. `attack_ms` > 0 adds an eased ramp to any shape. |
| `delay` | `ms` \| `beats`, [`step_ms` \| `step_beats`] | Channel c is delayed by base + c·step. Beat delays follow the tempo map. |
| `math` | `fn` (`add`, `mul`, `max`, `min`, `sub`, `mix`), `in`=[a, b, ...], [`amount` 0.5] | 1-channel inputs broadcast to N. `sub` and `mix` take exactly 2 inputs (`mix` = a·(1−amount) + b·amount). The result is a trigger only if every input is one and `fn` is add/mul/max/min. |
| `hold` | `ms` | Peak hold: max over the last `ms` (this frame and earlier). |

## Channels (the flow family)
| op | params | notes |
|---|---|---|
| `fanout` | `channels` | 1 → N copies. |
| `chase` | `channels`, `step_ms` \| `step_beats`, [`direction` 1, `wrap` true] | Channel i = input delayed by i·step. Input is 1 channel or already N. With `direction` −1 and `wrap`, channel 0 still leads and the chase runs the other way round the ring (0, N−1, N−2, ...). Without `wrap`, it starts at the last channel. `wrap` has no effect on direction 1. |
| `cycle` | `in`=trigger, `channels`, [`order` sequential, `seed`] | Each hit goes to the next channel. `pingpong` = 0,1,..,N−1,N−2,..,1,0... `random` is seeded and never picks the same channel twice in a row. |
| `ripple` | [`neighbours` 2, `falloff` 0.5, `step_ms` 40, `ring` true] | Each channel spreads to its neighbours at distance d, delayed d·step and scaled by falloff^d. Overlaps combine with max. On an even ring the opposite channel counts once. |
| `mirror` | | out[i] = in[min(i, N−1−i)] (a symmetric pattern). |
| `reverse` | | out[i] = in[N−1−i]. |
| `select` | `channels` (list of indices) | |
| `merge` | [`fn` max \| mean] | N → 1. `max` keeps a trigger a trigger. |
| `shuffle` | [`seed`] | A fixed permutation. |

## Layers
`layers[].intensity` = keys `{frame, value, ease, fade_frames}`. **A key's fade starts at `frame`** and reaches `value` at
`frame + fade_frames` (0 = a cut, and check warns about it). The value before the first key is the first key's value.
Eases: `in_out` (default, smoothstep: zero velocity at both ends), `in`, `out`, `linear`. An output in a layer bakes
`min + (value · intensity) · (max − min)`. To switch looks, give two layers opposite keys at the same frame.

## Tempo resolution
Order of precedence: `rig.tempo.map` (frames), then `rig.tempo.bpm`, then the `notes.json` `bpm_map`, then the
analysis bpm map, then 120. The grid offset comes from `rig.tempo.grid_offset`, then from notes (when notes carry a
bpm map), then from the analysis. Beat positions are the integral of bpm/60 from the grid offset.

If the analysis finds **no proven pulse**, it reports the strongest window's folded period as a low-confidence tempo
(`fallback: true`, flagged by `analyze`) and writes **no beats**. A `beats` op is then empty, while `lfo`/`*_beats`
still run on that tempo. Set `rig.tempo.bpm` to override it.

## Additions to the design-doc formats (all optional)
- rig: `notes` (path to the notes file, default `notes.json`), `analysis` (settings: `bpm_min` 75, `bpm_max` 150 (tempo is folded into this range),
  `pulse_z` 4, `min_conf` 0.15, `min_agree` 0.15, `win_s` 16, `hop_s` 4; clips under 32 s use `win_s` = duration/2 (min 4 s)), `tempo.grid_offset`, `bake.decimals` (4).
- bake.json: `audio_offset_frames`, `notes_hash`, and per output `layer` and `signal`. MPC channels carry no `actor`.
  Values are rounded to `bake.decimals`. Keys are `[frame, value]` taken from those values, and their
  linear-interpolation error is at most `key_tolerance · |max − min|`.
- Target kinds checked: `light` (actors, [property]), `material` (actors, param), `niagara` (actors, param), `mpc`
  (asset, param; 1 channel). Signal channels must be 1 (same value to every actor) or equal the actor count.

## Python API
```python
import beatrig
rig = beatrig.load_rig("rig.json")              # Rig; Rig.from_dict(d, base_dir) for in-memory rigs
an = beatrig.analyze(rig)                        # Analysis (cached next to the rig) or None without audio
sig = beatrig.evaluate(rig, an, notes)           # {node_id: Signal}; Signal.v is float32 [channels, frames]
res = beatrig.bake(rig, an, notes, out="bake.json")
res = beatrig.bake_from_objects(rig_dict, notes_dict, base_dir)   # unsaved rig/notes; notes_dict None = read the file
issues = beatrig.check(rig, notes)               # [{"level", "where", "msg"}]
notes = beatrig.load_notes(path); beatrig.save_notes(path, notes)
an.summary(px_per_s=200)                         # waveform peaks, beats, bpm map, passages for a preview page
```
