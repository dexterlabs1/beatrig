# Unreal applier

> **Status: untested.** `ue/beatrig_apply.py` is written against the UE 5.8 editor Python API but has not been run in
> an editor yet. The pure half, `ue/keys.py`, is unit tested (`tests/test_ue_keys.py`). Run `check` after every
> `apply` until this note goes away.

## Running it

In the editor's console (Output Log, Cmd mode; `py` runs a Python file with arguments):

```
py "<repo>/ue/beatrig_apply.py" apply <sequence> <bake.json> [--start N] [--only id,id] [--save]
py "<repo>/ue/beatrig_apply.py" check <sequence> <bake.json> [--start N] [--only id,id]
py "<repo>/ue/beatrig_apply.py" undo  <sequence> [output id] [--save]
```

`<sequence>` is an asset path such as `/Game/Show/LS_Main`. `--start` places bake frame 0 at that sequence frame.
`--save` saves the sequence asset afterwards; without it the sequence is only marked dirty. The last line printed is
`BEATRIG_RESULT {json}`, which is what the MCP tools read.

From an agent, `ue_apply` / `ue_check` / `ue_undo` send the same command through Remote Control (setup in the
[README](../README.md#unreal-integration)): they write a two-line launcher to a temp file and call
`PythonScriptLibrary.ExecutePythonCommandEx` with `ExecutionMode: ExecuteFile`.

## What it writes

| Target kind | Track | Bound to | Key time base |
|---|---|---|---|
| `mpc` | `MovieSceneMaterialParameterCollectionTrack` | the sequence (no binding) | **ticks** |
| `light` | `MovieSceneFloatTrack` (property from `target.property`: `intensity`, `attenuation_radius`, `source_radius`, `indirect_lighting_intensity`, `volumetric_scattering_intensity`, `temperature`) | the actor's light component | display frames |
| `material` | `MovieSceneComponentMaterialTrack` (`target.material_index`, default 0) | the actor's first primitive component | **ticks** |
| `niagara` | `MovieSceneNiagaraFloatParameterTrack` (`User.<param>`) | the actor's Niagara component | display frames |

Actors are found in the open level by label (then by object name). Component bindings are added under their actor's
binding. Every key is written with linear interpolation; parameter-section keys are created as auto (cubic) by the
engine and switched to linear afterwards.

`ue/keys.py` and the applier also handle a `transform` kind (`MovieScene3DTransformTrack`, additive, one channel
such as `location.z`) for when the engine gains transform outputs; the engine's `check` doesn't accept it yet.

## Tags and undo

Each track is named `beatrig:<output id>` (MPC) or `beatrig:<output id>:<actor>`. `undo` removes tracks whose name
has that prefix, for one output or all of them, and nothing else. `apply` first removes the tracks it is about to
write, so applying twice gives the same sequence. The whole apply is one editor transaction, so Ctrl+Z also undoes it.
Bindings that beatrig added are left in place (empty) by `undo`.

## Time bases: the tick trap

Sequencer stores time in ticks (the sequence's tick resolution, 24000 per second by default) and shows it in display
frames. The scripting API is not consistent about which one it takes:

| Call | Takes |
|---|---|
| `MovieSceneScriptingFloatChannel.add_key(time, ...)` (light, Niagara, transform channels) | display frames (`time_unit=DISPLAY_RATE`, the default) |
| `MovieSceneSection.set_range(start, end)` | display frames |
| `MovieSceneParameterSection.add_scalar_parameter_key(name, time, value)` (MPC and material tracks) | **ticks** |
| `channel.get_keys()[i].get_time()` when reading back | display frames, for every channel |

At 30 fps and 24000 ticks/s a frame is 800 ticks. Passing frames to `add_scalar_parameter_key` squeezes the whole
animation into the first 1/800 of its length. `keys.frame_to_ticks` does the conversion from the exact rational rate,
so NTSC rates come out right too (29.97 fps is 800.8 ticks per frame; frame 1 rounds to tick 801).

The applier never resamples: if the bake's fps differs from the sequence's display rate, `apply` and `check` refuse
and say so. Set the rig's `fps` to the sequence's display rate and bake again.
