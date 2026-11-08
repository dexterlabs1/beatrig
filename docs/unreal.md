# Unreal applier and live link

`ue/beatrig_apply.py` runs inside the Unreal editor and writes a bake into a Level Sequence as plain linear Sequencer
keys. `beatrig.live.Link` keeps it loaded in a running editor and drives it over Remote Control, so a re-bake shows up
in the viewport in about a tenth of a second. `ue/keys.py` is the pure half (tick maths, key planning, tags,
read-back comparison, the multiply/add maths and RDP); it has no `unreal` import and is unit tested
(`tests/test_ue_keys.py`).

## What is tested, and where

Everything below was run against **UE 5.8.3** (Windows, editor Python 3.11, Remote Control on `127.0.0.1:30010`) on
2026-11-07. Each target kind was put through `tests/ue_live_smoke.py`: apply, check key for key, apply again (replaces
in place), check, then undo output by output, counting the sequence's bindings throughout. The evaluated values were
then read off the actors with the sequence open in Sequencer.

| Target | Fixture | apply / check / re-apply / check / undo | Evaluated in the editor |
|---|---|---|---|
| `light` intensity | `bake_scalar.json` | pass, 5 tracks | intensity = the key interpolation at the playhead |
| `material` scalar | `bake_scalar.json` | pass | |
| `mpc` scalar | `bake_scalar.json` | pass | the world's MPC value = the key interpolation |
| `mpc` colour (vector param) | `bake_color.json` | pass, 4 channels per track | RGBA = the key interpolation |
| `light` colour | `bake_color.json` | pass | `LightColor` is an 8-bit sRGB `FColor`: linear (0.577, 0.884, 0.039) shows as (200, 242, 56) |
| `material` colour (vector param) | `bake_color.json` | pass | |
| `transform` location / rotation / scale, additive | `bake_transform.json` | pass, 3 tracks on one actor | offsets add to the actor's own transform; the unkeyed channels stay untouched |
| `niagara` user float | `bake_niagara.json` | pass; `check` also confirms the track drives `User.<param>` | not checked (no engine template system declares a user parameter) |

Bindings: every fixture leaves the sequence with exactly the bindings it had before (0 → 8 → 0 for lights plus meshes;
0 → 1 → 0 for the transform actor; 0 → 2 → 0 for the Niagara actor and its component).

**Output blend** (`multiply` / `add`, v0.3) was run on UE 5.8.3 on 2026-11-07 with `tests/ue_live_blend.py`
(fixtures `bake_blend_multiply.json` and `bake_blend_add.json`). The bases were keyed as an animator would:
- MPC `Glow` (scalar) and `Tint` (colour) in **one** MPC section, Sequencer's default layout, with cubic auto keys out
  to frame 150 (the bake covers 0-119).
- A point light's `Intensity` on a float track (section 0-300, keys out to frame 200).
- A cube's material scalar `Glow` on a component material track.

The evaluated values were read off the world's MPC instance, the light component and the cube's dynamic material
instance, with the sequence open in Sequencer, at frames 0, 17, 30, 40, 77 and 119 (in the range) and 120, 121, 150 and
200 (outside it).

| Step | multiply | add |
|---|---|---|
| apply: one additive section per track over its base | pass (217 keys, 4 tracks) | pass (92 keys, 4 tracks) |
| base sections after apply | active, keys identical | same |
| evaluated vs base × gain / base + offset (gain = the bake keys' interpolation) | every frame within the RDP tolerance (worst 0.88 of it; material 0.85); frames 120-200 are exactly the base | exact |
| `check` | pass | pass |
| edit a base key (Glow at 30, the light at 45) | `check` fails for those two outputs and passes for the others; `push` reports both as `base_edited` and rewrites them (2 calls, 28-35 ms); evaluated values follow the new base | the offset doesn't depend on the base: `push` writes nothing and the values follow the edit at once |
| `undo` | tracks and records gone; bases active, keys exactly as edited; the world plays the base | same |

Also run:
- **No base section.** A second light with no track gave an absolute section of static 5000 × gain; MPC `Glow` with
  no MPC track gave the collection default (0) + offset. Both static values were recorded, and outside the range the
  target holds its static value.
- **A base section added after a static apply.** `check` fails ("now drives this target"), and applying again puts the
  output on the new base as an additive section.
- **Two outputs on one parameter.** The second is refused.
- **Repeated material cycles.** Setup, apply, read back, check and undo ran five times for multiply and add on the
  material alone, and the whole runner passed twice in a row with `--material`. See the two GOTCHAs on component
  material tracks below.

Not run live: blended `niagara` and light-colour targets (they go through the same path). Transforms take no blend:
they already layer through `target.additive`, and a transform output with `blend` is refused.

The runner wipes the sequence you give it, and refuses one that has content unless you pass `--wipe`:

```
py -3 tests/ue_live_blend.py --sequence /Game/BeatrigTest/LS_Blend --material
```

Renders: Movie Render Queue in the editor (PIE executor), rendering through an OWL 360 dome camera, gave pixel-identical
frames on two runs and matched `bake.json` frame for frame (cross-correlation lag 0).

To repeat the live checks against your own scene, map the fixtures' neutral names onto it:

```
py -3 tests/ue_live_smoke.py --sequence /Game/BeatrigTest/LS_Test \
    --map Light_=MyLight_ --map Orb_=MySphere_ --map /Game/FX/MPC_Show=/Game/MyFX/MPC_Show tests/fixtures/bake_color.json
```

## Setup

Enable the **Remote Control API** and **Python Editor Script** plugins and allow
`PythonScriptLibrary.ExecutePythonCommandEx`, restricted to this machine (the exact ini lines are in the
[README](../README.md#unreal-integration)). The editor reads the payload files beatrig writes to the system temp folder,
so it has to run on the same machine.

## Three ways to run it

**Editor console** (Output Log, Cmd mode):

```
py "<repo>/ue/beatrig_apply.py" apply <sequence> <bake.json> [--start N] [--only id,id] [--save]
py "<repo>/ue/beatrig_apply.py" check <sequence> <bake.json> [--start N] [--only id,id]
py "<repo>/ue/beatrig_apply.py" undo  <sequence> [output id] [--save]
```

`<sequence>` is an asset path such as `/Game/Show/LS_Main`. A key at bake frame f lands on sequence frame
`start_frame` + `--start` + f: `start_frame` comes from bake.json (default 0), and `--start` shifts it further.
`--save` saves the sequence afterwards; without it the sequence is only marked dirty. The last line printed is
`BEATRIG_RESULT {json}`.

**Python** (`beatrig.live`):

```python
from beatrig.live import Link
with Link("127.0.0.1:30010") as link:
    link.push("bake.json", "/Game/Show/LS_Main")     # only the outputs that changed since the last push
    link.check("bake.json", "/Game/Show/LS_Main")    # key-for-key read-back
    scene = link.scan()                              # the scene.json dict
    link.playhead_set(120, playing=True)
```

`apply`, `check`, `undo`, `push`, `scan`, `playhead_get`, `playhead_set`, `ensure_sequence` (create a sequence if it is
missing) and `throttle` each return the editor's result dict, with `rt_ms` (round trip) and `editor_ms` (time spent in
the editor). A failure to reach or use the editor raises `LiveError` with a message that says what to do.

**MCP**: `ue_apply`, `ue_check`, `ue_undo`, `live_push`, `ue_scan`, `playhead_get`, `playhead_set` (see
[mcp.md](mcp.md)). The MCP server keeps one `Link` per editor host for its whole life.

**CLI**: `python -m beatrig.live scan --out scene.json [--host H] [--paths /Game ...]` writes `scene.json` and prints a
summary (`beatrig scan` takes the same arguments).

## How the live link talks to the editor

1. The first call bootstraps once: one `ExecuteStatement` appends the repo's `ue/` folder to the editor's `sys.path`
   and imports `ue/beatrig_apply.py` from exactly that file as the module `beatrig_apply`. It is reloaded only when the
   source hash of `ue/*.py` changes, and every request carries the client's hash, so an edited applier is picked up
   by the next call. `keys.py` and `beatrig_scan.py` load under the names `beatrig_keys` and `beatrig_scan`, because a
   bare `keys` module could collide with another plugin's in a shared editor.
2. Every later call is one short statement, `beatrig_apply.rpc("<temp>/beatrig-live-....json")`. The request (op and
   arguments; bakes travel as keys only, never per-frame values) is in that JSON file, and the result comes back in
   `<file>.out`. The Output Log stays quiet.
3. The HTTP connection is kept alive (HTTP/1.1) and reused. If the editor drops it, the Link reconnects once. A lock
   serialises calls from several threads (the editor runs every call on its game thread anyway).

### Changed-outputs push

`push(bake, sequence)` hashes what the applier writes for each output (target, value type, every channel's keys, the
bake fps and the start frame; not the per-frame values or min/max). It sends only outputs whose hash differs from the
last push of this Link to this sequence, plus the hashes of the rest. The editor replies with any of those it no longer
holds as pushed (removed by `undo` from another client, or the module reloaded), and the Link resends them in a second
call. Outputs that were pushed before but are gone from the bake are removed from the sequence. When the sequence is the
one open in Sequencer, the editor re-evaluates it at once (`LevelSequenceEditorBlueprintLibrary.force_update`), so the
viewport shows the new keys without waiting for a tick. `push(..., focus=True)` opens the sequence in Sequencer first;
the default never changes what Sequencer is showing.

A Ctrl+Z in the editor that restores an older version of a pushed track is not detected (the track is still there, with
the old keys). Push once with `only_changed=False` after undoing in the editor.

### Timings (UE 5.8.3, 2026-11-07)

A 30 s rig at 30 fps with three outputs: 8 point lights (intensity), 8 meshes (material scalar), one MPC scalar;
6,143 keys in all. Rows marked "median" are medians of 10 runs (range in brackets); the others are 1 to 3 runs.

| Step | Time |
|---|---|
| Remote Control call, editor in the background, "Use Less CPU when in Background" on | **~330 ms per call** (8 calls) |
| the same with the setting off (or the editor in the foreground), keep-alive connection | ~11 ms per call (8 calls) |
| the same with a new connection per call | 21-30 ms (5 calls) |
| bootstrap (first call of a Link) | 75 ms |
| `scan` of a 22-actor level | 1.3-1.8 ms in the editor, 15-21 ms round trip |
| `push`, all three outputs (6,143 keys, 32 bindings created) | 155-203 ms (editor 123-157, of which writing keys 97-108) |
| `push`, nothing changed | 35-60 ms (one round trip; the editor only compares hashes and track counts) |
| `push`, one node changed (one output, ~350 keys), sequence open, re-evaluated | **49 ms median** (41-63; 25 in the editor, re-evaluation 2 ms) |
| re-bake of that rig after a parameter change | 53 ms median |
| parameter change to keys in the editor and evaluated (re-bake + push) | **104 ms median** (91-148) |
| `playhead_get` / `playhead_set` | 24 / 43 ms median of 6 |
| opening a sequence in Sequencer | 160-230 ms |

Writing keys costs the editor about 16 µs per key, and hashing costs the client about 0.4 µs per key. A live push of
one output of a 60-minute show (8 channels with tens of thousands of keys each) is therefore several seconds in the
editor. That is fine for a one-off apply, but live editing of a long show should work on a shorter rig window
(`audio_in_s` / `audio_out_s`).

After every one-node push in those runs, the evaluated MPC value at the playhead read back exactly equal to the new
keys' interpolation, so the viewport has new values from the next frame on.

Blended outputs (UE 5.8.3, 2026-11-07, the three-output multiply fixture over its bases, while the machine ran a video
encode):

| Step | Time |
|---|---|
| `push`, all three blended outputs (168 keys; three bases read and evaluated) | 29 ms (editor 10) |
| `push`, nothing changed (the editor also fingerprints the recorded bases) | 11-19 ms (editor 2), 5 runs |
| `push`, one blended output changed | 18-23 ms (editor 6.4), 6 runs |
| `push` after a base key edit in the editor (detect, then rewrite: 2 calls) | 34-35 ms (editor 8-11), 3 runs |
| `evaluate_keys` over 108,001 frames (60 minutes at 30 fps) | 12 ms |

A blended track only spans the bake range (plus one frame each side), so its cost is the bake's, however long the
base is.

## GOTCHA: the editor sleeps in the background

**Symptom:** every Remote Control call takes ~330 ms while the editor window is not in front (for example while you
work in the preview page), even a `print(1)`.
**Cause:** Editor Preferences > General > Performance > **Use Less CPU when in Background**
(`EditorPerformanceSettings.bThrottleCPUWhenNotForeground`, on by default) drops a background editor to a few ticks a
second, and Remote Control answers only on a tick.
**Fix:** turn it off, or let the Link do it. `Link(keep_awake=True)`, the default, switches it off in the editor's
memory when it connects and restores the old value on `close()` or at interpreter exit. The change is never written to
the ini, but if you save any Performance preference while a Link is connected, the editor writes the current
(off) value. The MCP server's Link and `beatrig scan` leave the setting alone (`keep_awake=False`).

## What it writes

| Target kind | Track | Bound to | Key time base |
|---|---|---|---|
| `mpc` | `MovieSceneMaterialParameterCollectionTrack`: a scalar param, or for colour a vector param | the sequence (no binding) | **ticks** |
| `light` | `MovieSceneFloatTrack` (`target.property`: `intensity`, `attenuation_radius`, `source_radius`, `indirect_lighting_intensity`, `volumetric_scattering_intensity`, `temperature`), or for colour `MovieSceneColorTrack` on `LightColor` (`property: "color"`) | the actor's light component (any `LightComponentBase`) | display frames |
| `material` | `MovieSceneComponentMaterialTrack` (`target.material_index`, default 0): a scalar or vector param | the actor's first primitive component | **ticks** |
| `niagara` | `MovieSceneNiagaraFloatParameterTrack` on `User.<param>` | the actor's Niagara component | display frames |
| `transform` | `MovieScene3DTransformTrack`, one channel (`property` location/rotation/scale, `axis` x/y/z, rotation also roll/pitch/yaw), additive unless `additive: false` | the actor | display frames |

Colour outputs carry `"value_type": "color"` (or 5-wide keys `[frame, r, g, b, a]`, linear RGBA) and are written as
four channels. Actors are found in the open level by label, then by object name. Every key is linear; parameter-section
keys come out of the engine as auto (cubic) tangents and are switched to linear afterwards.

**Transform.** Each transform output gets its own track, so one actor can carry several (location.z, rotation.yaw and
scale.x on one actor all evaluate together). An additive section adds its keyed channel to whatever the actor already
has: with keys giving +40.45 on location.z, 36 on yaw and +0.1727 on scale.x, a cube at (0, 3000, 200) with scale 1
evaluated to z = 240.45, yaw 36, scale x 1.1727, every other channel unchanged. Actors with Static mobility don't move
under Sequencer; make transform targets Movable.

**Niagara.** A Niagara parameter track's `Parameter` is private to Python (`set_editor_property` refuses:
"protected and cannot be read"). The applier creates the track with
`LevelSequenceEditorSubsystem.paste_tracks(text, params)`: the text is Sequencer's own copy format for one track,
carrying `Parameter=(Name="User.<param>",TypeDefHandle=(RegisteredTypeIndex=N))`. `N` is the editor session's index of
the Niagara float type, read from a float binding Niagara defines itself (the sprite renderer's
`Particles.SpriteRotation`). Pasting text never touches the clipboard (only `copy_tracks` and an empty paste do).
Sections and keys are then added through the normal API, and `check` reads the parameter back from a T3D export of
the track. The parameter must exist as a user parameter in the system for the value to reach it.

## Output blend: multiply and add

An output with `"blend": "multiply"` or `"add"` carries gains or offsets (design contract, "Output blend"). The applier
puts them on top of the target's own animation in the sequence, its **base**, without ever touching the base.

**Why additive.** Two absolute sections on one parameter are averaged, not stacked (`Glow` 3 and 10 gave 6.5). MPC
sections refuse `OVERRIDE`. Muting the base would stop every other parameter that shares its section, and Sequencer
puts all of a collection's parameters in one section by default. An `ADDITIVE` section stacks (3 + 10 gave 13) and
leaves the base playing. See the GOTCHAs below.

1. **Base.** The first active section, on a track beatrig didn't write, that drives the same target. A section shared
   with other parameters is fine.

   | Target | Base track |
   |---|---|
   | `mpc` | an MPC track on the same collection whose section has the parameter |
   | `light` | the light component's float track on the same property, or its colour track |
   | `material` | the component material track on the same slot whose section has the parameter |
   | `niagara` | the Niagara parameter track on the same `User.` parameter |

   Transforms take no blend: they already layer on the actor's own animation through `target.additive`, so a transform
   output with `blend` is refused.
2. **Over a base section: an additive offset.** The beatrig track gets an `ADDITIVE` section, its blend type set
   **before** any key. It holds:
   - for multiply, `base(t) × (gain(t) − 1)`, with the base evaluated at every display frame of the bake range by
     `evaluate_keys` (the engine's own interpolation, so cubic and weighted tangents are exact);
   - for add, the offsets themselves (they don't depend on the base).

   A 0 key goes one frame before and one after the bake range. The section is unbounded with constant extrapolation,
   so the offset is exactly 0 outside the range and those frames play the base alone.
3. **No base section: an absolute section** of `static × gain` (or `static + offset`) over the bake range. The static
   value is the MPC parameter's default, the light or material property, or the Niagara user parameter. It is read at
   the first apply and recorded, and re-applying reuses it (so a value Sequencer is driving at that moment can't creep
   in; `undo` and apply to re-read it). This isn't additive because an additive section with nothing under it adds onto
   whatever value the world last held.
4. **Keys.** The gain between its keys is their linear interpolation, which is what Sequencer would play for a replace
   output. Offsets and static finals are simplified with the engine's RDP (`keys.rdp`, which keeps the same frames as
   `beatrig.bake.rdp`). The tolerance is `key_tolerance` (0.002, or the bake's top-level `key_tolerance` if it has
   one) × the range of the final curve, since an error in the offset is the same error in the final value. Every gain
   key frame is kept, hits included.
5. **Record.** The sequence asset's metadata gets one tag per track, `beatrig.base:<track tag>`, holding JSON:
   - the target, used to refuse a second blend on it;
   - the blend mode;
   - the base section's path inside the sequence and its track name, or the static value;
   - for multiply, a fingerprint of the base keys.

   It is saved with the asset, and the path is relative to the sequence so it survives a rename.

Afterwards:
- `check` recomputes the track from the target's **current** base and compares key for key. A multiply output fails
  once its base keys were edited. Any blend fails if the base section is gone or inactive (the offset would add onto
  nothing), or if a base section appeared after a static apply (Sequencer would average the two). Apply again in
  every case.
- `push` with nothing changed fingerprints each recorded base. A multiply output whose base keys were edited, or whose
  base section went away, is reported as `base_edited` and rewritten from the current base in a second call. An add
  output needs nothing: base edits show through at once.
- `undo` (all, or one output) deletes the beatrig track and its record. The base was never changed, so there is
  nothing to give back.
- Two beatrig blends on one target are refused ("already blends this target"), since two additive layers on one
  parameter would be ambiguous.
- If a bounded base section doesn't cover the whole bake range, the apply result carries a `warning` naming the
  frames, because there the offset adds onto whatever the target holds.

## Tags, bindings and undo

Each track is named `beatrig:<output id>` (MPC) or `beatrig:<output id>:<actor>`. Bindings the applier creates (an
actor binding, a component binding under it) carry the binding tag `beatrig`; bindings it finds already in the sequence
are reused and never tagged. `undo` removes the tracks whose name has the prefix (one output's or all), then removes
tagged bindings that are left with no tracks and no child bindings (components first, then their actor). A binding
someone else created is never removed, and a tagged binding that someone added their own track to stays. `apply`
removes the tracks it is about to replace and prunes the same way, so applying twice gives the same sequence. Each
call is one editor transaction, so Ctrl+Z undoes it too. For multiply/add outputs `undo` also drops the track's
base record; the base itself was never changed (see Output blend above).

## Playhead

`playhead_get()` returns the sequence open in Sequencer, the playhead in display frames, whether it is playing, its fps,
playback range and `audio` (the sequence has an active audio track; `sequence: null` when nothing is open). A linked
preview page mutes its own audio while `audio` is true, so the music isn't heard twice. `playhead_set(frame, playing, sequence)` jumps the playhead
(no events fire in between) and plays or pauses; with `sequence` it first opens that sequence in Sequencer, which
replaces whatever Sequencer was showing.

## Scene scan

`scan()` returns the `scene.json` structure from the [design contract](design.md) for the level open in the editor:
every actor (class, location in cm, rotation as pitch/yaw/roll, plus `mobility` and the outliner `folder` when set), its
components of kind `light`, `mesh`, `niagara` or `camera` (editor-only helpers such as camera proxy meshes are
skipped), each mesh's material slots (`slot`, `name`, `material`, and the material's scalar and vector parameter
names), every Material Parameter Collection under `/Game` with its parameters, the Level Sequence asset paths, and the
camera: an OWL 360 capture actor if there is one (`projection` such as `domemaster`, `dome_angle`), else the first
camera actor (`projection: "perspective"`, `fov`). Material parameter names are read once per base material, MPCs are
loaded (they are small), sequences are only listed. A World Partition level shows only its loaded actors.

## Time bases: the tick trap

Sequencer stores time in ticks (the sequence's tick resolution, 24000 per second by default) and shows it in display
frames. The scripting API is not consistent about which one it takes:

| Call | Takes |
|---|---|
| `MovieSceneScriptingFloatChannel.add_key(time, ...)` (light, colour property, Niagara, transform channels) | display frames (`time_unit=DISPLAY_RATE`, the default) |
| `MovieSceneSection.set_range(start, end)` | display frames |
| `MovieSceneParameterSection.add_scalar_parameter_key` / `add_color_parameter_key` (MPC and material tracks) | **ticks** |
| `channel.get_keys()[i].get_time()` when reading back | display frames, for every channel |

At 30 fps and 24000 ticks/s a frame is 800 ticks. Passing frames to `add_scalar_parameter_key` squeezes the whole
animation into the first 1/800 of its length. `keys.frame_to_ticks` does the conversion from the exact rational rate,
so NTSC rates come out right too (29.97 fps is 800.8 ticks per frame; frame 1 rounds to tick 801).

The applier never resamples: if the bake's fps differs from the sequence's display rate, `apply` and `check` refuse
and say so. Set the rig's `fps` to the sequence's display rate and bake again.

## Other gotchas found on 5.8.3

- **Colour channel names.** Parameter sections name colour channels by appending the component with no separator
  (`Tint` → `TintR`, `TintG`, `TintB`, `TintA`); a colour property track's channels are `Color.R` ... `Color.A`.
  Read-back selects channels by these names (`keys.channel_names`).
- **`str(Guid)` is not an id.** It is the Python wrapper's repr, with a memory address that changes on every call, so
  de-duplicating bindings by it double-counts component bindings (they come back both from `get_bindings()` and as
  children). Use `Guid.to_string()`.
- **A refused connection is slow on Windows.** Connecting to a localhost port nobody listens on takes about 2 s to fail,
  so the Link reports any connect failure as "no editor answering" rather than as a timeout.

## GOTCHA: `SequencerScriptingRange()` is not in ticks or frames

**Symptom:** `channel.evaluate_keys(range, display_rate)` returns far fewer values than expected (a range of
`0..8000` gave 4).
**Cause:** a `SequencerScriptingRange` built in Python carries a hidden `InternalRate` of 60000/s, and
`inclusive_start` / `exclusive_end` are in that rate. `compute_effective_range()` returns one in the tick rate (24000).
**Fix:** spell the rate out:
`r.import_text("(bHasStart=True,bHasEnd=True,InclusiveStart=0,ExclusiveEnd=121,InternalRate=(Numerator=30,Denominator=1))")`.
`evaluate_keys` then returns one value per display frame from the start (frames 0..120 here). It evaluates the channel
whether or not its section is active.

## GOTCHA: two sections on one parameter are averaged, not stacked

**Symptom:** a second track on a parameter the show already animates gives a value halfway between the two (MPC
`Glow` 3.0 and 10.0 → 6.5; light intensity 2500 and 9000 → 5750).
**Cause:** Sequencer weights overlapping absolute sections equally. MPC parameter sections refuse `OVERRIDE` and
`RELATIVE`: `set_blend_type` silently stores `Absolute` and `Additive` instead.
**Fix:** blend additively, which is what `multiply`/`add` do. `ADDITIVE` stacks on MPC, float property and component
material sections (3 + 10 gave 13, 2500 + 500 gave 3000), but only if `set_blend_type` is called **before** the keys
are added. Called afterwards, the stored type changes and the evaluation still averages. Two gotchas:
- An additive section with no absolute section under it adds onto the world's current value of the parameter, left
  over from the last evaluation (2.625 + 2 gave 4.625), not onto the collection default. That is why the no-base case
  writes an absolute section.
- Muting the base instead (`set_is_active(False)`) also works, but by default Sequencer puts every parameter of a
  collection in one section. Muting `Glow`'s section sent `Tint` to the collection default.

## GOTCHA: component material slot with two tracks: a bounded section hands the slot back

**Symptom:** right after a bounded beatrig section was added on a material slot that a base track also animates, the
frames after the section's end showed the material's own value (1.0), not the base's (2.92). This lasted until the
sequence was next re-evaluated after an edit, and only on component material tracks; MPC and light tracks were fine.
**Cause:** when a section on the slot ends, Sequencer seems to restore the slot's pre-animated value over the other
track.
**Fix:** the applier writes additive sections unbounded. The 0 keys one frame each side of the bake range keep the
offset at 0 outside it, and a section that never ends never hands the slot back.

## GOTCHA: a Python wrapper of Sequencer's dynamic material instance can crash the editor later

**Symptom:** the editor died with `EXCEPTION_ACCESS_VIOLATION` inside `python311.dll`, with no engine frames, during a
later, unrelated Remote Control call. It happened three times in the live runner, always in runs that read a material
parameter off the cube.
**Cause (likely):** the runner's snippet read `component.get_material(0)`, the `MaterialInstanceDynamic` Sequencer
creates for a component material track. The snippet's globals dict sits in a reference cycle with the helper functions
it defines, so the wrapper lived until a later garbage-collection pass, after Sequencer had replaced that instance.
**Fix:** don't keep wrappers of transient objects. The runner clears each snippet's globals at the end
(`(lambda g: (exec(code, g), g.clear()))({})`) and reads the value without holding the instance. After that, the full
runner passed twice with the material. The applier itself never touches those instances.

## Other things seen on 5.8.3 while building blend

- `add_section()` on an MPC track gives an infinite section (`has_start_frame()` is False, and `get_start_frame()`
  raises); on a float property track it gives a bounded one.
- Section object paths are stable (`/Game/X/LS.LS:MovieScene_0.MovieSceneFloatTrack_7.MovieSceneFloatSection_0`) and
  `unreal.find_object` resolves them. A deleted track's section can linger in memory, so the applier checks that a
  recorded section still sits in one of the sequence's tracks.
- Asset metadata (`EditorAssetLibrary.set_metadata_tag`) is saved in the `.uasset`. It is not part of the editor
  transaction, so after a Ctrl+Z of an apply the record outlives the track. The next apply rewrites it, and `undo`
  clears it.
- Component material sections (`MovieSceneComponentMaterialParameterSection`) have no `get_parameter_names()`; their
  channels are named after the parameter (`Glow`).
- A `MaterialInstanceDynamic` in 5.8 Python has no `k2_get_scalar_parameter_value`; use `get_scalar_parameter_value`.
- `MaterialParameterCollection.get_scalar_parameter_default_value(name)` returns `(value, found)`.
