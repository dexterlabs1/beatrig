# beatrig ops reference (v0.3, with the unreleased additions)

The file formats are specified in [the design doc](design.md). This page is the op-by-op reference, plus the semantics
the engine settles where the design doc leaves room. Params in brackets are optional (shown with their defaults).
Every node also accepts `id`, `op`, `in`, `label`, `comment`, and any key starting with `_`.

A **signal** is `float32[channels, frames]`. A **trigger** is a signal that is non-zero only on hit frames, where the
value is the hit velocity (0..1]. A **colour** signal is `float32[channels, frames, 4]`, linear RGBA. Every signal
also carries a **hit mask**: the frames that key simplification must
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
| `band` | `lo`, `hi` (Hz), [`attack_ms` 10, `release_ms` 150, `weights` null, `stem` null] | 1 ch, raw linear amplitude | Butterworth band (order 4, low/high-pass at the edges). Per-channel power is weighted and summed, so channels can't phase-cancel. Then a one-pole follower runs on an exact 1 ms grid, sampled at each frame's start. `weights` (one per audio channel) defaults to the rig's `channel_weights`, else equal. `stem` (`drums`, `bass`, `vocals`, `other`) reads a separated stem instead of the mix (see Stems). |
| `loudness` | [`attack_ms` 10, `release_ms` 150, `weights`, `stem`] | 1 ch | Same as `band`, with the ITU-R BS.1770 K-weighting filter (computed for any sample rate). |
| `onsets` | [`in`, `threshold` 0.3, `min_gap_ms` 80, `stem`] | trigger, 1 ch | Causal spectral-flux peaks on a 5 ms grid. `in` is a `band` node id, an inline `{lo, hi}`, or omitted for the full-range octave-band flux. Velocity is the peak relative to the 95th percentile of all peaks. Peaks below `threshold` are dropped. The onset time is when the rise is known (`(k+1)·5 ms`). The stem is the node's own `stem`, else the referenced band's. |
| `beats` | [`every` 1, `offset` 0, `subdivide` 1, `min_confidence` 0, `velocity` strength] | trigger, 1 ch | The **verified** beat grid from the analysis. Beats exist only where a real onset supports them. `offset` is an index offset into the beat list for `every`. `subdivide` inserts evenly spaced hits sized from the **previous** interval (causal), never past the next real beat. Subdivision velocity is 0.6 × the beat's. `min_confidence` (0..1) drops beats whose analysis confidence is below it, together with their subdivisions; it applies after `every`/`offset`, so bar counting stays on the full grid. `velocity` `confidence` uses each beat's confidence as the hit velocity instead of its onset strength. `notes init --from beats --min-conf X` fills a track the same way. `anchor` (`first` \| `bar`, default `first`): `first` counts `every`/`offset` along the beat list from its first beat (as before); `bar` counts on the bar grid, so `every: 4` = the downbeats, `every: 4, offset: 2` = the third beat of each bar, and `offset` is a phase in the bar, not a skip. The beat number is the analysis' bar position (`bar` x meter + `pos`, see Neural beats and bars), else the tempo map's beat count from the grid offset; a beat without an onset keeps its number, so counting never slips. With `subdivide` S, subdivision j of beat n counts as n·S + j. `notes init --from downbeats` = `every` meter, `anchor` `bar`. |
| `drum` | `part` (`kick` \| `snare` \| `hats`), [`threshold` 0.3, `min_gap_ms` (kick 110, snare 90, hats 55), `source` auto] | trigger, 1 ch | One drum part's hits, with no band to tune: kick 35-120 Hz, snare (or clap) 1-5 kHz, hats 6-16 kHz, read from the separated drums stem when the analysis has it (`source` `auto`), else the mix; `stem` requires the stem, `mix` never reads it. On top of the band's onsets: a peak with 10x its power in the band within the next 120 ms is a stem's pre-echo and is dropped; a snare must still hold 2 % of its peak power 40-80 ms later (a kick's click is gone by then); a hat must hold 1 % 10-20 ms later (not a click), and one on a snare hit (within 25 ms) stays only when its rise is twice what the song's snares put in that band (hats that always land on the snare are left to the snare track). Then the strongest hits `min_gap_ms` apart, and `threshold` relative to the part's own 95th percentile. These tests look up to 80 ms past the onset (an offline choice, like `flash_limit` `strongest`); a hit still lands on the first frame at or after its onset. `notes init --from drum:kick`. |
| `notes` | `track` | trigger, 1 ch | Hits from `notes.json`. Per-hit `len_beats` and `shape` travel with the trigger to `envelope`. |
| `lfo` | [`shape` sine, `rate_beats` \| `rate_hz`, `phase` 0, `channels` 1, `phase_step` 0] | N ch, 0..1 | `rate_beats` = beats per cycle, locked to the tempo map and grid offset. `rate_hz` = cycles per second of audio time. Shapes: `sine` (starts at 0), `tri`, `saw_up`, `saw_down`, `square` (check warns: instant edges), `semi_sine`. Channel c gets `phase + c·phase_step`. Default rate: 1 beat. |
| `const` | [`value` 0, `channels` 1] | N ch | |

## Shaping
| op | params | notes |
|---|---|---|
| `normalize` | [`range_db` 24, `hi_pct` 99.5, `lo_pct` 10] | dB scale to 0..1 over the whole bake (all channels together): `hi_pct` maps to 1, and the floor is max(`lo_pct`, top − `range_db`). This is a static gain, so timing is unaffected. |
| `smooth` | `attack_ms`, `release_ms` (both > 0) | One-pole up/down follower per channel at frame rate. Takes colour too (each RGBA component), for eased colour changes. |
| `spring` | [`freq_hz` 2, `damping` 0.6] | 2nd-order follower ω²/(s²+2ζωs+ω²), bilinear-discretised, starts at rest. Damping below 1 overshoots. |
| `curve` | [`gamma` 1] or `preset` (`s_curve`, `ease_in`, `ease_out`) | Negative values clip to 0. Presets clip to 0..1. |
| `remap` | `in_lo`, `in_hi`, `out_lo`, `out_hi`, [`clamp` true] | |
| `threshold` | [`level` 0.5, `cooldown_ms` 100] | Fires on a rising edge through `level` (reads this frame and the previous one only), velocity 1. |
| `envelope` | `in`=trigger, [`shape` punch, `length_beats` 0.5 \| `length_ms`, `attack_ms` 0, `decay_ms` 100, `sustain` 0.7, `release_ms` 200, `points`] | Shapes: `binary`, `punch` (exponential), `gloss` (cosine), `razor` ((1−t)⁴), `rush` (eased rise to 80 %, then a fast fall), `saw_down`, `adsr` (gate = length, then release), `custom` (`points` = [[t 0..1, v], ...]). Velocity-scaled. Overlapping hits take the max. Per-hit `len_beats`/`shape` from notes override. `attack_ms` > 0 adds an eased ramp to any shape. |
| `delay` | `ms` \| `beats`, [`step_ms` \| `step_beats`] | Channel c is delayed by base + c·step. Beat delays follow the tempo map. Takes colour. |
| `accumulate` | [`per_s` 1 \| `per_beat`, `start` 0, `wrap` null] | Running sum (an integral): out[f] = start + the sum over frames **before** f of in × rate × frame time, i.e. the phase at the frame's start. A constant 1 with `per_s` 0.5 turns half a unit per second; `per_beat` follows the tempo map. The output is continuous even when the input jumps (a trigger envelope becomes an eased push), so it drives spin and phase. `wrap` folds into [0, wrap): an instant jump at each wrap (check warns), fine for a shader phase, not for a transform. |
| `math` | `fn` (`add`, `mul`, `max`, `min`, `sub`, `mix`), `in`=[a, b, ...], [`amount` 0.5] | 1-channel inputs broadcast to N. `sub` and `mix` take exactly 2 inputs (`mix` = a·(1−amount) + b·amount). The result is a trigger only if every input is one and `fn` is add/mul/max/min. |
| `hold` | `ms` | Peak hold: max over the last `ms` (this frame and earlier). |

## Channels (the flow family)
| op | params | notes |
|---|---|---|
| `fanout` | `channels` | 1 → N copies. Takes colour (as do `chase`, `mirror`, `reverse`, `select`, `shuffle`). |
| `chase` | `channels`, `step_ms` \| `step_beats`, [`direction` 1, `wrap` true] | Channel i = input delayed by i·step. Input is 1 channel or already N. With `direction` −1 and `wrap`, channel 0 still leads and the chase runs the other way round the ring (0, N−1, N−2, ...). Without `wrap`, it starts at the last channel. `wrap` has no effect on direction 1. |
| `cycle` | `in`=trigger, `channels`, [`order` sequential, `seed`] | Each hit goes to the next channel. `pingpong` = 0,1,..,N−1,N−2,..,1,0... `random` is seeded and never picks the same channel twice in a row. |
| `ripple` | [`neighbours` 2, `falloff` 0.5, `step_ms` 40, `ring` true; knn: `k` 3, `hops` 2, `actors`] | Each channel spreads to its neighbours at distance d, delayed d·step and scaled by falloff^d. Overlaps combine with max. On an even ring the opposite channel counts once. `neighbours: "knn"` uses real positions instead of channel order: each actor (channel i = `actors[i]`, needs the rig's scene) joins its `k` nearest actors (ties broken by channel order), and d is the hop count in that graph, up to `hops`. |
| `mirror` | | out[i] = in[min(i, N−1−i)] (a symmetric pattern). |
| `reverse` | | out[i] = in[N−1−i]. |
| `select` | `channels` (list of indices) | |
| `merge` | [`fn` max \| mean] | N → 1. `max` keeps a trigger a trigger. |
| `shuffle` | [`seed`] | A fixed permutation. |

## Per-hit flow (triggers in, triggers out)
Each decides, hit by hit, where a hit goes or whether it passes, or moves it later onto the grid. Nothing ever moves
earlier. Per-hit `len_beats`/`shape` from notes travel with each hit. Several input channels are merged first (the
strongest hit wins) for `push` and `shuffle_hits`; `every_n`, `probability` and `quantize` work per channel.

| op | params | notes |
|---|---|---|
| `push` | `channels`, [`pattern` alternate, `start` 0] | Each hit lights the next step of a pattern. `alternate` = even channels, then odd. `sequential`, `pingpong`, `outside_in` (0 and N−1, then 1 and N−2, ...), `inside_out`, `halves`, or an explicit list of steps, each a channel or a list of channels: `[[0, 4], [2, 6], [1, 5], [3, 7]]`. `start` = the step the first hit takes. |
| `shuffle_hits` | `channels`, [`mode` bag, `count` 1, `seed`] | Random channel(s) per hit, seeded. `bag`: every channel once per round in a fresh random order, never the same channel twice in a row (also across rounds). `random`: `count` distinct channels drawn independently per hit. |
| `every_n` | `n`, [`offset` 0] | Pass hit number offset, offset + n, offset + 2n, ... (counted from 0, per channel). |
| `probability` | `p`, [`seed`] | Pass each hit with probability p. The draw is a hash of (seed, channel, frame), or with no node `seed` of (rig seed, node id, channel, frame), so one hit's fate never depends on the other hits: editing a note re-rolls nothing else. |
| `quantize` | [`grid` "1/16", `strength` 1] | Delay each hit to the next grid line at or after it (beat grid from the tempo map; `grid` as in `notes snap`). Since the onset lies after the previous frame's start, a grid line inside the hit's own frame keeps it where it is. `strength` < 1 moves part of the way (rounded up). Colliding hits keep the strongest. |

## Colour
Colours in a rig are `"#rrggbb"` / `"#rrggbbaa"` (sRGB, decoded to linear) or `[r, g, b]` / `[r, g, b, a]` numbers
(linear, as given; alpha defaults to 1). Gradients interpolate in linear RGB. Named palettes (sRGB hex, in
`beatrig.PALETTES`): `ember`, `ice` and `ocean` (dark to bright, good for `gradient`), `aurora`, `candy`, `sunset`
(good for `palette_cycle`), `mono`.

| op | params | output | notes |
|---|---|---|---|
| `gradient` | `in` (value), `stops` [[t, colour], ...] \| `palette` (ember) | colour, same channels | Each channel's value, clamped to 0..1, picks a colour along the stops (a palette = evenly spaced stops). |
| `palette_cycle` | `period_beats` \| `period_s`, [`palette` \| `stops`, `hold` 0, `channels` 1, `phase_step` 0, `phase` 0] | colour | Drifts through the palette and back to its first colour once per period (beats follow the tempo map). Every step eases in and out (zero velocity on each palette colour). `hold` (0..1) = share of each step resting on a colour. Channel c is offset by phase + c·phase_step cycles. |
| `mix_color` | `in` = [a, b] or [a, b, t], [`amount` 0.5] | colour | a·(1−amount) + b·amount, or with a value signal t (0..1) as the per-frame amount. 1-channel inputs broadcast. |
| `scale_color` | `in` = [colour, value], [`gain` 1, `alpha` false] | colour | RGB × value × gain: brightness follows the value. `alpha` true scales alpha too. |

A colour can only go into ops that declare colour support: the colour ops above, `smooth`, `delay`, `chase`, `fanout`,
`select`, `reverse`, `mirror`, `shuffle`, `spatial_delay` and `dome_wave`. `check` errors on any other (do the value
maths first, then `gradient`, or use `scale_color`/`mix_color`).

**Colour outputs**: an `mpc` (vector param), a `light` with `"property": "color"`, or a `material` (vector param).
`"value_type": "color"` is accepted on the output or on its target; a light target with property `color`, or a
value_type colour, needs a colour signal. No `min`/`max` (check warns they are ignored): scale brightness in the graph.
A layer scales RGB by its intensity and keeps alpha. In bake.json the output carries `"value_type": "color"` (on the
output entry and in its target), each channel has `values: [[r, g, b, a], ...]` and `keys: [[frame, r, g, b, a], ...]`.
The RDP runs on the 4-vector: every component stays within `key_tolerance` × the channel's largest component range.

## Transform outputs
`target = {"kind": "transform", "actors": [...], "property": "location" | "rotation" | "scale", "axis": "x" | "y" | "z"
(rotation: "roll" | "pitch" | "yaw"), "additive": true}`. Values are `min + value × (max − min)` in cm, degrees or
scale units; with `additive` (the default) the applier writes them as an offset on top of the actor's own animation.
Pair with `accumulate` for continuous spin (`min` 0, `max` 360 → one turn per unit).

## Output blend (v0.3)
`output.blend` = `replace` (the default; v0.2 behaviour), `multiply` or `add`. With `multiply` the baked values are
**gains** and with `add` **offsets**, applied in the editor on top of the target's own animation (its base: the first
non-beatrig section already driving that target, else its static value; see the design doc for the applier side).
The engine never sees the base, so a blended output bakes exactly like a replace one: `min + (value · intensity) ·
(max − min)`, e.g. a gain from 0.6 to 1.55. Two consequences of that formula: signal 0 gives `min` (not the neutral
1 or 0), and so does a layer faded to 0. Put the neutral value inside the range where the music is quiet.

- `check`: value outputs with `multiply` or `add` need explicit `min` and `max` (the gain range, where 1 = the base
  unchanged, or the offset range in the target's units, where 0 = unchanged). The 0..1 default would be taken
  literally, and a gain of 0 blacks out the base. A negative gain warns (it flips the base's sign). Colour outputs
  blend per RGBA component (`multiply`: per-component gains such as [1.2, 1, 0.8, 1]; `add`: offsets) and take no
  min/max, as before. Transform outputs reject `blend`: they already blend through `target.additive`.
- bake.json: the output entry carries `"blend": "multiply" | "add"`. A replace output carries no `blend` key, so a
  v0.2 rig bakes byte-identically (read a missing key as `replace`).
- Flash safety judges the baked gain or offset curve against its own `min`..`max`, as for any output.

## Spatial (real positions from scene.json)
A rig points at a scene scan with `"scene": "scene.json"` (format in the design doc). Units are Unreal's: cm, rotations
`[pitch, yaw, roll]` in degrees. Channel i = `actors[i]` after `@group` expansion. The input is 1 channel (broadcast)
or one per actor. Delays are relative by default, so the first actor reached fires on time; all are ≥ 0.

| op | params | notes |
|---|---|---|
| `spatial_delay` | `actors`, `origin` (actor name or [x, y, z] cm), `speed_cm_s` \| `beats_per_m`, [`shape` point, `dir`, `direction` out, `relative` true] | Delay = distance / speed. `point`: distance from the origin. `axis`: distance from the line through the origin along `dir`. `plane`: signed distance along `dir` (a sweep across the plane through the origin). `direction` out = near first, in = far first. `relative` false (point/axis, out) measures from the origin itself, so a wave can start at an empty point. `beats_per_m` locks the travel time to the tempo. |
| `dome_wave` | `actors`, `deg_per_s` \| `deg_per_beat`, [`mode` zenith_rings, `direction`, `start_deg` 0, `relative` true, `camera`] | A wave over a (tilted) fulldome from each actor's angle seen from the dome camera (`scene.camera`, or the node's `camera` {location, rotation}). `zenith_rings`: delay by the angle from the dome centre (direction `out` = centre first, `in` = rim first). `azimuth_sweep`: delay by the angle around the dome from `start_deg` (direction `cw` or `ccw`). `deg_per_beat` = tempo-locked (360 / deg_per_beat beats per turn). `check` warns about actors past the dome's edge (`camera.dome_angle` / 2, default 90°). |

**Dome convention** (verified 2026-11-07 against a live scan and an OWL 360 Domemaster render in UE 5.8): UE `FRotator`
[pitch, yaw, roll], applied roll, then pitch, then yaw. A domemaster camera looks along the camera actor's **up** axis
(+Z after rotation): with rotation (0, 0, 0) the zenith is straight up and actors on the horizon land on the rim. Pitch the
actor to tilt the dome toward its forward axis: pitch -45 puts the dome centre 45 deg above the horizon, facing forward.
Zenith = angle between an actor and that up axis (0 at the centre, 90 on the rim of a 180 deg dome). Azimuth = angle
around it, 0 toward the camera's forward (+X) and 90 toward its right (+Y).

**Groups**: `rig.groups = {"ring": ["Light_0", ...]}`. Anywhere an actor list is accepted (output targets, spatial ops,
ripple knn) `"@ring"`, or a list containing `"@ring"`, expands to the group (groups may include groups). bake.json
writes targets with the actors expanded. `check` errors on unknown groups.

**Scene warnings**: with a scene, `check` warns about actors the scan doesn't know, light outputs on actors without a
light component, material params the actor's materials don't list (vector params for colour), and MPC assets/params
missing from the scan. Spatial ops error when the scene or an actor position is missing.

## Flash safety
`rig.safety = {"max_flashes_per_s": 3, "min_swing": 0.1, "outputs": "auto"}`. A **transition** is a monotonic swing
between two turning points of at least `min_swing` × the output range (a zigzag at that threshold); one **flash** is a
pair of opposite transitions; a channel fails when any 1 s window holds more than `max_flashes_per_s` flashes (the
WCAG / Harding general-flash rule; a dome fills the whole field of view). A transition is placed at the frame where it
ends (the first frame of its extreme).

- Checked outputs (`outputs`): `auto` = light and MPC outputs, plus material outputs whose param looks emissive
  (`emiss`, `glow`, `emit`, `bright`, `intens`, `light`, `lum`, `flash`, `strobe`) or that are colour; `all` = every
  output but transforms; `none`; or a list of output ids. Each channel is checked, plus the mean of all channels (the
  overall brightness of the field). Value outputs use their `min`..`max` as the range; colour outputs use Rec. 709
  luminance with range 0..the output's brightest luminance.
- It runs on the baked values. With a `safety` block a violation is an **error**: `bake` raises (pass `strict=False`
  to get the result anyway) and `beatrig check` fails. Without one it is a warning, and bake.json still carries the
  report: `"safety": {"limit", "min_swing", "violations": [{"output", "channel", "max_per_s", "windows": [[t0, t1], ...]}]}`.
- `beatrig.flash_report(values, fps, lo, hi, max_flashes_per_s, min_swing)` is the pure function behind it:
  `{"flashes", "max_per_s", "ok", "per_second": [...], "windows": [[t0, t1], ...]}` (seconds), for plot overlays.

| op | params | notes |
|---|---|---|
| `flash_limit` | [`max_flashes_per_s`, `min_swing` (default: rig.safety, else 3 and 0.1), `mode` first] | **Triggers** (put it before the envelope): kept hits are at least 1 / max_flashes_per_s s apart, and no 1 s window holds more than floor(max_flashes_per_s) of them, so `flash_report` passes on the result at any rate. At a whole-number rate the spacing alone ensures the window rule; at 2.5 per second hits may be 12 frames apart (at 30 fps) but the third waits until the first has left the window. Below 1 per second no flash fits the 1 s rule, so nothing passes (check warns). Modes: `first` (the default; `drop` is its v0.2 name) passes a hit when the rules allow it and drops the rest; causal. `delay` holds a refused hit until the rules allow it (several held hits merge into one, the strongest); causal. `strongest` is an **offline choice**: hits are taken strongest first (the earliest on a tie), and a hit is kept if it is clear of every kept hit, else dropped. That is the hand-built pipelines' "keep the strongest of close hits". A kept hit stays on its own frame with its own velocity and per-hit `len_beats`/`shape`, so it is never early or late. The output at a frame may depend on later hits (a stronger hit just after removes a weaker one), so this mode is exempt from the truncation rule, but nothing is ever placed ahead of its own sound. **Values** (causal; 0..1 assumed, so `min_swing` is in signal units): while the trailing second already holds 2·max − 1 transitions, the output holds just short of completing another, then follows again. A safe signal passes unchanged. |

## Layers
`layers[].intensity` = keys `{frame, value, ease, fade_frames}`. **A key's fade starts at `frame`** and reaches `value` at
`frame + fade_frames` (0 = a cut, and check warns about it). The value before the first key is the first key's value.
Eases: `in_out` (default, smoothstep: zero velocity at both ends), `in`, `out`, `linear`. An output in a layer bakes
`min + (value · intensity) · (max − min)`. To switch looks, give two layers opposite keys at the same frame.
A key may name a section instead of a frame: `{"section": "breakdown", "value": 0, "fade_frames": 30}` starts its fade
at the first frame at or after that section's time in notes.json (never before the music changes).

## Tempo resolution
Order of precedence: `rig.tempo.map` (frames), then `rig.tempo.bpm`, then the `notes.json` `bpm_map`, then the
analysis bpm map, then 120. The grid offset comes from `rig.tempo.grid_offset`, then from notes (when notes carry a
bpm map), then from the analysis. Beat positions are the integral of bpm/60 from the grid offset.

With the neural tracker the analysis grid offset is the first downbeat's phase within a bar (in [0, bar)), so beats
0, meter, 2·meter ... of the tempo map fall on downbeats: `lfo` `rate_beats` 4, patterns placed at multiples of 4 and
bar windows line up with real bars. The classic grid's offset is a beat's phase (in [0, beat)).

If the analysis finds **no proven pulse**, it reports the strongest window's folded period as a low-confidence tempo
(`fallback: true`, flagged by `analyze`) and writes **no beats**. A `beats` op is then empty, while `lfo`/`*_beats`
still run on that tempo. Set `rig.tempo.bpm` to override it.

## Tempo candidates
`analyze` also reports `tempo_candidates`: the main tempo and the levels it is most often confused with (×2, ×1/2,
×3/2, ×2/3, ×4/3, ×3/4). Each has a `support` (median over the analysis windows of the onset-autocorrelation comb at
that period) and a `confidence` relative to the best candidate (1.0 = best supported). Nothing is chosen
automatically: a human or an agent picks one and pins it with `rig.tempo.bpm`.

## notes.json v0.2: sections and linked patterns
- `sections: [{"name", "time"}]` (ordered). `beatrig notes sections init <rig>` fills them from the analysis with
  `song_structure` (see Song structure below): labelled sections on bar lines (`intro`, `groove`, `dip`, `build`,
  `breakdown`, `break`, `drop`, `outro`, numbered on repeats: `drop_2`). `--method energy` keeps the v0.2 rule (bar
  lines where the band levels change sharply, names guessed from energy: `intro`, `breakdown`, `main`, `peak`).
  `--min-bars` sets the shortest section. Edit the names freely.
- `patterns: {"kick4": {"length_beats": 4, "hits": [{"beat": 0, "vel": 1}, ...]}}`, placed on a track with
  `{"pattern": "kick4", "at_beat": 16, "repeat": 4}`. `at_beat` counts on the notes' bpm map (120 bpm from
  `grid_offset` when the notes have none). Placements expand when notes are loaded and again whenever a rig is
  evaluated; expanded hits carry `from_pattern: {"pattern", "at_beat", "copy"}`, and saving drops them and keeps the
  placement, so editing the pattern moves every copy.
- CLI: `notes pattern make <rig> --name N --length-beats 4 (--hits "0:1,1.5:0.7" | --from-track T --at-beat B)`,
  `notes pattern place <rig> --name N --track T --at-beat B [--repeat R] [--replace]`, `notes pattern list <rig>`.

## MIDI import
`beatrig notes import-midi <rig> file.mid --map 36:kick,38:snare [--channel 10] [--offset-s 0] [--lengths]
[--keep-tempo] [--replace]`. A pure-Python Standard MIDI File reader (format 0/1/2, running status, note-on velocity 0
as note-off, sysex/meta skipped, SMPTE time division). Notes may be numbers or names (`C2` = 36, the drum-map
convention). Velocity / 127 becomes `vel`; the file's tempo map becomes the notes' `bpm_map` (MIDI time 0 =
`--offset-s` in the audio, which is also the grid offset) unless `--keep-tempo`. `--lengths` writes `len_beats` from
each note's length. Imported hits are added to existing tracks unless `--replace`.

## Long files, 5.1 and timelines
- `audio_in_s` / `audio_out_s`: a window into a long file (one scene's slot in a full show mix). Frame
  `audio_offset_frames` = `audio_in_s`; frames = the window (or the rest of the file) plus the offset. The analysis
  always covers the whole file, so every slot rig shares one cache, and frame 0 sees the filters already settled on the
  audio before the window (which is the past, so still causal). Notes stay in seconds of the whole file. bake.json
  records `audio_in_s`/`audio_out_s` when set.
- `channel_weights`: one weight per audio channel, e.g. 5.1 `[1, 1, 1, 0.5, 0.7, 0.7]` (L R C LFE Ls Rs). They weight
  every band/loudness envelope and the mono mix the beat grid is found on; they are part of the analysis cache key.
- `beatrig timeline import <file.fcpxml|file.edl> --audio mix.wav --out dir/ [--fps] [--start-tc] [--track V]
  [--handles N] [--source-start-tc TC]` writes one stub rig per clip (`NN_<name>.rig.json`, each with its own notes
  file) plus `timeline.json`. The mix is assumed to start at the timeline's start, so `audio_in_s` is the clip's
  **timeline** position: FCPXML `offset` − the sequence's `tcStart` (rational times like `1001/30000s`), or EDL
  record-in − the start timecode (default: the hour of the first event, e.g. 01:00:00:00). EDL timecode is counted at
  the nominal rate and converted to real seconds (29.97 → 30000/1001), with drop-frame (`;` or `FCM: DROP FRAME`)
  handled. FCPXML reads the edit's sequence (a project's, not a compound clip's in `<resources>`), primary storyline,
  gaps skipped; EDL takes one entry per event number (the incoming clip of a dissolve).
- **Source in-point (v0.3).** Each clip's in-point inside its media becomes the rig's `audio_offset_frames`, so the
  rig's frames are **media** frames: frame `source_in` plays the clip's first timeline frame, and keys land where the
  scene's own sequence needs them. FCPXML: `asset-clip` `start` − the asset's `start`; `ref-clip` (a compound clip)
  `start` − that compound sequence's `tcStart`; `clip` / `sync-clip` through its first media child (child `start` +
  clip `start` − child `offset`); no `start` = 0; a `title` = 0. EDL: source-in − `--source-start-tc` (default
  00:00:00:00). An in-point off the frame grid is rounded to the nearest frame, with a warning.
- **Transitions and handles.** Each clip records the transition right before it (`transition_in`) and right after it
  (`transition_out`) as `{name, start_s, end_s, frames}` in timeline seconds, and `handles_needed` = [frames before
  the in point, frames after the out point] that those transitions use. FCPXML transitions are the `<transition>`
  elements next to the clip in the spine (a gap in between means none); an EDL dissolve or wipe (`D 030`, `W001 015`)
  starts at the incoming event's record-in, so the outgoing clip needs `frames` after its out point and the incoming
  one none before. Each rig bakes only its clip plus `--handles N` frames on either side: `bake.start_frame` =
  source in − N (clamped at the media's frame 0) and `audio_out_s` = the out point + N frames, so a clip whose source
  in is frame 600 bakes 15 + its length + 15 frames with N = 15, not 600 frames of other clips' music. Handle frames
  read the mix beyond the cut, which is what plays under a dissolve. The importer warns when a clip's transitions
  need more than N (or than the media has before the in point).
- Each rig carries `"_timeline": {clip, timeline_in_s, timeline_out_s, source_in_s, source_in_frames, transition_in,
  transition_out, handles: {before, after}, handles_needed}`, and `timeline.json` repeats it per clip with
  `audio_offset_frames` and `start_frame`.

## Baked range: `bake.start_frame` (v0.3)
`rig.bake.start_frame` (default 0) is the first sequence frame baked. Only sequence frames `start_frame` ..
`start_frame + frames − 1` are evaluated: `frames` is `rig.frames` when set, else the rest of the audio window
(window end − start_frame). Sequence frames still mean what they did: `audio_offset_frames`, layer key frames and
`tempo.map` frames are sequence frames, and a frame reads the same audio whatever the start. What changes:
- Whole-bake statistics see only the baked range (`normalize`'s percentiles, the flash-safety check).
- Anything carried from before the start begins fresh at it: `smooth` / `spring` / `hold` / `accumulate` state, an
  envelope tail from a hit before the start, `cycle` / `every_n` counting, `flash_limit` state. `band`, `loudness`,
  `onsets`, `beats`, `notes`, `lfo` and `probability` (whose draw is keyed by sequence frame) give exactly the values
  they give in a bake from frame 0.
- bake.json: `values[0]` and key frame 0 are sequence frame `start_frame`, and the result carries `"start_frame"` (left
  out when 0, so other rigs bake byte-identically). Flash-safety windows are in seconds from the start of the baked
  range. The applier lands key frame f on sequence frame `start_frame` + f.

## Stems (optional extra: `pip install beatrig[stems]`)
Stems are used by default. `rig.analysis.stems`:
- `auto` (the default): stems already separated are always used. A missing separation runs once, during `analyze`,
  when the [stems] extra is installed and torch sees a CUDA GPU (about 5 s for a 4.5-minute track on an RTX 5090).
  Without the extra, or on a CPU-only machine, stem features read the mix instead, and the analysis summary says so
  and why (`beatrig analyze`, the MCP `analyze` tool, `Analysis.stems_label()`).
- `on`: separate even on the CPU (slow: roughly the track's length or more).
- `off`: never separate; drum parts and the song structure read the mix; nodes that name a `stem` still read a cached
  one (and fail with what to run when it is missing).

`beatrig stems <rig> [--device cuda|cpu]` separates by hand. htdemucs writes drums/bass/vocals/other as wav next to the
audio, in `<audio name>.stems/htdemucs-<content hash>/` (an edited file never reuses stale stems). `band`/`loudness`/
`onsets` read a stem with `"stem": "drums"`; `drum` reads the drums stem by itself. With stems in use, `analyze` opens
all four (each with its own analysis cache, `<stem>.analysis/` in the stem folder) and computes the drum-part bands
and each stem's level once, so drum parts and the song structure are cheap afterwards. A stem's analysis has no beat
grid (its beats are never read). Separation is offline and looks at whole segments of audio; the analysis of a stem is
causal as usual, but a stem carries a little pre-echo (about 90 ms ahead of a hit), which `drum` removes and plain
`onsets` on a stem do not. The core install never imports torch.

## Neural beats and bars (optional extra: `pip install beatrig[ml]`)
`rig.analysis.beats`: `auto` (the default) uses the neural tracker when the [ml] extra is installed, else the classic
grid, unchanged; `neural` requires it (an error otherwise); `classic` never uses it. The tracker is Beat This!
(Foscarin, Schlueter and Widmer, ISMIR 2024; code and weights MIT): a transformer over a log-mel spectrogram that
gives a beat and a downbeat probability every 20 ms. Its weights (77 MB) download once into the torch hub cache. It
runs on the GPU when torch sees one, else on the CPU (measured: 1.4 s and 3 s for a 4.5-minute track, plus model
loading). beatrig computes the spectrogram itself in numpy, so torchaudio is never imported.

The model never places a beat. From its activations:
1. **Beats.** Its beat peaks (logit > 0, the largest within ±60 ms, sub-frame by a parabola). Each needs a real onset:
   the strongest octave-flux peak above the 70th percentile of the onset function and 5 % of a strong hit, within
   0.14 periods of it. The beat is written at that onset's time ((k + 1) × 5 ms), never at the model's frame; a model
   beat with no onset is not written. So beats exist only where an onset supports them and are never ahead of it.
2. **Bars.** A skeleton of the model's beats (near-duplicates dropped, gaps of k periods filled with k − 1 virtual
   beats, unsupported beats kept) carries the count. The meter is 3 when the model's downbeat spacings are clearly
   multiples of 3, else 4. The phase is a Viterbi path over the downbeat activations (each beat's held to 0.1..0.9, a
   phase change costing log 1e-12): a real shift is followed after about 7 bars of agreement, while a stretch where
   the model hears the bar line elsewhere and comes back (syncopated chords in a drum-less breakdown) needs about 13
   to break the count. Each written beat carries `bar` (0 = the first bar that starts with a downbeat; a pickup is −1)
   and `pos` (0 = downbeat).
3. **Passages and tempo.** Passages break where the model is silent for 8 beats or more or the local period moves
   more than 3 %. Each is checked like a classic passage: tempo from a line fit through its beats, agreement with the
   independent reference onsets above chance, confidence = mean model probability × hit rate × jitter × agreement;
   one under `min_conf` or `min_agree` writes no beats. Beat confidence = the model's probability × an onset-strength
   factor. If no passage passes (or the model fails, e.g. no network for the first download), the classic grid
   stands and the analysis says why.

The analysis keeps the activations (`nn:beat`, `nn:down` in analysis.npz) and records `beat_tracker` (`name`, `model`,
`device`, `seconds`, or the classic grid's `why`), `bars` (`meter`, `downbeats` (times; an unsupported downbeat keeps
the skeleton's time), `first_downbeat`, `confidence`) and, per beat, `bar` and `pos`. Tempo candidates are scored
around the neural tempo. The tracker keeps its own metrical level: `bpm_min`/`bpm_max` fold only the classic grid.

A cache built with the neural tracker is reused on a machine without the extra, so both bake alike. A classic cache
built because the extra was missing (or disabled) is rebuilt once the extra is there. GPU inference is not
bit-reproducible, but beats sit on the 5 ms onset grid, so a re-analysis moves a beat only when a model peak sits
right at the 0.5 threshold; the cache is what keeps bakes byte-identical.

`BEATRIG_EXTRAS=off` (environment) makes `auto` behave like a core install: no neural tracker and no automatic
separation (cached stems are still read). The test suite sets it; it also reproduces a core install's analysis.

## Song structure
`beatrig.song_structure(analysis, min_bars=None)` (used by `notes sections init` and the MCP `describe_audio`) returns
labelled sections on bar lines: `[{"name", "time", "bar", "label", "bars", "kick", "bass_db", "energy_db", "novelty"}]`.
- **Bar lines**: the analysis' downbeats (extended by whole bars before and after), else bars of 4 beats from the grid
  offset. A section time is a bar line, so a layer key naming it never fades before the music changes.
- **Features per bar**: the mix level and each stem's level (dB under the song's loud bars, floored at −24 dB so that
  "absent" is one value), kick / snare / hats hits per bar (`drum` parts, from the drums stem when there is one) and
  the 7 band levels. Without stems, bass presence is the floor of the 50-250 Hz envelope between hits.
- **Boundaries**: peaks of a contrast novelty (the distance between the mean features of the w bars before and after
  a bar line, averaged over w = 2, 4, 8), with a 25 % bonus on the song's 8-bar phrase lines (the residue mod 8 with
  the most novelty), kept where the robust z-score is at least 1.5, strongest first, at least `min_bars` apart
  (default 8 in a song of 64 bars or more, 4 from 24, else 2). Kick or bass coming in or going out (for 2 bars or
  more) is also a boundary, and so is the bar after a one-bar kick drop-out.
- **Labels** from what plays: no kick = `breakdown` (`break` under 3 bars); kick without bass = `build` (`intro` when
  first); kick and bass = `intro` (first), `drop` (after a build, breakdown or break, or right after a drop and at
  least as loud), `build` (what rises, the other stem or else the mid bands, climbs 6 dB and a breakdown or a
  bass-less part follows), `dip` (2 dB under the song's other full sections) or `groove`. The last section is `outro`
  when it has no kick or is 3 dB under the full sections. Repeats are numbered.
- Stems matter here: from the mix alone a pad or a sub-bass in a breakdown can pass for kick and bass.

## Additions to the design-doc formats (all optional)
- rig: `notes` (path to the notes file, default `notes.json`), `analysis` (settings: `bpm_min` 75, `bpm_max` 150 (the classic
  tempo is folded into this range), `pulse_z` 4, `min_conf` 0.15, `min_agree` 0.15, `win_s` 16, `hop_s` 4 (clips under
  32 s use `win_s` = duration/2, min 4 s), `beats` auto|neural|classic, `stems` auto|on|off; all but `stems` are part
  of the cache key), `tempo.grid_offset`, `bake.decimals` (4).
- analysis.json (version 5): `beat_tracker`, and with the neural tracker `bars` and per beat `bar`/`pos` (above).
- bake.json: `audio_offset_frames`, `notes_hash`, and per output `layer` and `signal`. MPC channels carry no `actor`.
  Values are rounded to `bake.decimals`. Keys are `[frame, value]` taken from those values, and their
  linear-interpolation error is at most `key_tolerance · |max − min|`.
- Target kinds checked: `light` (actors, [property]), `material` (actors, param), `niagara` (actors, param), `mpc`
  (asset, param; 1 channel), `transform` (actors, property, axis). Signal channels must be 1 (same value to every
  actor) or equal the actor count.
- The analysis cache (`analysis.json` + `analysis.npz`) holds only what needs a pass over the audio: the onset
  function, beat grid, waveform peaks and each band's 1 ms power. Envelopes (a band with its `attack_ms`/`release_ms`)
  are derived from that power in memory, so editing them never rewrites or grows the cache: a cached `analyze` is a
  pure read, and only a new band (or a new file / new settings) writes it.
- rig (v0.2): `scene`, `groups`, `safety`, `audio_in_s`, `audio_out_s` (above). bake.json (v0.2): `safety` summary,
  `audio_in_s`/`audio_out_s` when set, colour channels, targets with groups expanded.
- rig (v0.3): `cache_dir`, the folder of the analysis cache (relative to the rig; default the rig's own folder).
  Scene rigs in their own folders that point at one `cache_dir` and the same mix share one analysis: the first
  `analyze` builds it and the rest only read it. They must agree on `analysis` settings and `channel_weights`, which
  are part of the cache key (otherwise each rebuilds and overwrites it). Per output `blend` (above).
  `bake.start_frame` (above). bake.json (v0.3): `blend` on multiply/add outputs; `start_frame` when it isn't 0;
  `key_tolerance` (top level) when the rig sets a `bake.key_tolerance` other than the default 0.002, so the applier
  re-simplifies blended keys with the same tolerance.

## Python API
```python
import beatrig
rig = beatrig.load_rig("rig.json")              # Rig; Rig.from_dict(d, base_dir) for in-memory rigs
an = beatrig.analyze(rig)                        # Analysis (cached in rig.cache_dir, default next to the rig) or None
sig = beatrig.evaluate(rig, an, notes)           # {node_id: Signal}; Signal.v is float32 [channels, frames]
res = beatrig.bake(rig, an, notes, out="bake.json")
res = beatrig.bake_from_objects(rig_dict, notes_dict, base_dir)   # unsaved rig/notes; notes_dict None = read the file
issues = beatrig.check(rig, notes)               # [{"level", "where", "msg"}]
issues = beatrig.check(rig, notes, bake=res)     # + the flash-safety rule on the baked values
notes = beatrig.load_notes(path); beatrig.save_notes(path, notes)
an.summary(px_per_s=200)                         # waveform peaks, beats, bpm map, passages, tempo_candidates
an.meta["tempo_candidates"]                      # [{"ratio", "bpm", "support", "confidence"}]
beatrig.flash_report(values, fps, lo, hi, 3, 0.1)              # per-second flash counts + offending windows
secs, levels = beatrig.section_boundaries(an); beatrig.name_sections(secs)   # the v0.2 energy rule
beatrig.song_structure(an)                       # labelled sections on bar lines
an.tracker_label(); an.bars_summary(); an.stems_label(); an.stems["drums"]   # where beats, bars and stem features came from
beatrig.make_pattern(notes, "kick4", 4, hits=[{"beat": 0, "vel": 1}]); beatrig.place_pattern(notes, "kick", "kick4", 16, 4)
beatrig.track_hits(notes, "kick")                # plain + freshly expanded pattern hits
beatrig.import_midi("drums.mid", "36:kick,38:snare", channel=10)    # {"bpm_map", "grid_offset", "tracks", "stats"}
beatrig.import_timeline("edit.fcpxml", "mix.wav", "clips/", handles=15)   # one window rig per clip (+ "warnings")
beatrig.parse_color("#ff8800"); beatrig.PALETTES["ember"]; beatrig.dome_angles(camera, positions)
```
