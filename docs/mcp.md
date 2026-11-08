# MCP server reference

`beatrig-mcp` is a stdio MCP server (install with `pip install -e ".[mcp]"`, register with
`claude mcp add beatrig -- beatrig-mcp`). It works on the same files as the CLI and the preview page.

Conventions:
- **Paths.** `rig_path` is a `rig.json` (or a folder that holds one), absolute or relative to the server's working
  directory. Notes, audio and the bake resolve from the rig's folder.
- **Results** are compact JSON text. Per-frame arrays stay in `bake.json`; tools return stats and paths.
- **Errors** are MCP tool errors whose message says what to do next.
- **Edits** (`set_node`, `add_node`, `remove_node`, `set_output`, `remove_output`, `set_layer`) validate the whole
  edited rig with `check` before writing. An invalid edit is refused and the file is untouched. The previous file is
  kept as `rig.json.bak` (`notes.json.bak` for notes tools). The result has a `diff`: `{"key": [old, new]}`.
- **Ids, not indexes.** Nodes, outputs and layers by `id`; notes tracks by name; hits by time in seconds of audio
  (matched within `tolerance_ms`, 10 ms by default).
- **Preview.** After a change the server POSTs `{"source": "mcp", "action", "detail"}` to
  `BEATRIG_PREVIEW_URL/api/notify` (default `http://127.0.0.1:4700`). The page reloads from disk and logs the action
  in its agent strip. Results carry `"preview": "notified" | "not running" | "http <code>"`.

## Hearing the song

| Tool | Args | Returns |
|---|---|---|
| `analyze` | `rig_path` | duration, sample rate, `bpm`, `bpm_map`, beat count, pulse confidence, tempo passages, and `bands`: for each candidate range (sub 30-60, kick 40-120, bass 80-250, lowmid 250-800, mid 800-2500, presence 2500-6000, air 6000-14000 Hz) the hit count, `hits_per_beat`, `on_beat` (share of hits within 60 ms of a beat), `contrast_db` (how far hits stand above the background) and `level_db`. `suggested_bands` lists up to three with the clearest hits. These are frequency ranges, not detected instruments. |
| `describe_audio` | `rig_path`, `start_s`, `end_s`, `window_s` (default one bar, at most 48 rows) | per window: each band's level (0..1, dB-scaled over the whole song) and hit count; `candidate_boundaries` where the band levels change sharply, with which bands rose and fell. |

## The rig

| Tool | Args | Notes |
|---|---|---|
| `list_ops` | | every op, its params with defaults (`REQUIRED` marks required ones), input arity and output kind |
| `get_rig` | `rig_path` | the rig JSON, the node graph (`id -> inputs`), notes track sizes, check errors and warnings |
| `set_node` | `rig_path`, `node_id`, `params` | merge; `null` removes a param (back to its default). Can't rename. |
| `add_node` | `rig_path`, `node`, `after` | `node` needs `id` and `op`; inserted after node `after`, or at the end |
| `remove_node` | `rig_path`, `node_id` | refused while other nodes or outputs use it; the error names them |
| `set_output` | `rig_path`, `output_id`, `fields`, `create` | merge (`target` merges one level down); `create=true` adds a new output (needs `signal`, `target`) |
| `remove_output` | `rig_path`, `output_id` | |
| `set_layer` | `rig_path`, `layer_id`, `intensity` | replaces (or creates) a layer's intensity keys `[{"frame", "value", "ease", "fade_frames"}]` |
| `check` | `rig_path` | `ok`, errors, warnings |

## Notes lanes

| Tool | Args | Notes |
|---|---|---|
| `get_notes` | `rig_path`, `track`, `start_s`, `end_s`, `limit` | track summaries; hits of one track in a range |
| `notes_init` | `rig_path`, `track`, `source`, `threshold`, `replace` | `source`: `band:LO-HI`, `onsets:<node>`, `beats`, `beats:<node>`. Won't overwrite a non-empty track without `replace=true`. Seeds the notes tempo map from the analysis. |
| `notes_edit` | `rig_path`, `track`, `edits`, `tolerance_ms` | all-or-nothing list of `{"action": "add", "time", "vel", "len_beats"?, "shape"?}`, `{"action": "move", "time", "to"}`, `{"action": "delete", "time"}`, `{"action": "velocity", "time", "vel"}` |
| `notes_snap` | `rig_path`, `track`, `grid`, `strength` | quantise to `1/4`, `1/8`, `1/16` (or a number of beats). Reports `max_earlier_ms`: snapping a hit earlier can put the visual ahead of its sound. |

## Bake and look

| Tool | Args | Returns |
|---|---|---|
| `bake` | `rig_path`, `out` | writes `bake.json` (default next to the rig); per output: target kind, channels, min, max, mean, key count, hit count |
| `plot` | `rig_path`, `output_id`, `start_s`, `end_s`, `channels`, `width` | a PNG image plus a JSON caption. Top lane: the waveform. One lane per channel (up to 16; pick with actor names or indexes): the value curve as Sequencer will interpolate it (lane floor = output `min`, top = `max`), orange ticks at trigger hits, dark dots at baked keys. Grey lines are beats, darker every fourth. A red bar along a lane's top marks where that channel flashes faster than the rig's flash-safety limit; the caption's `flash_safety` block gives the limit and, per channel, `ok`, `max_per_s` and the spans in seconds. Colour outputs are drawn as luminance (Rec. 709 of linear RGB), which is also what the safety check judges. |

`plot` draws with numpy and zlib only (no plotting library), so the image is the same on every machine. The flash
overlay comes from `beatrig.safety.flash_report`; without that module the plot is drawn without it.

## Preview page

| Tool | Returns |
|---|---|
| `preview_url` | whether the page is running, and its URL |
| `preview_start(rig_path, port=4700)` | starts `beatrig serve` in the background and waits for it |

## Unreal

| Tool | Args | Notes |
|---|---|---|
| `ue_scan` | `out`, `rig_path`, `host`, `paths` | scans the level open in the editor and writes `scene.json` to `out`, or next to `rig_path` (the rig's `"scene"` file if it names one). Returns a summary: actor names by component kind (light, mesh, niagara, camera), material parameter names with how many meshes have them, MPCs with their scalar/vector parameters, Level Sequences, and the camera (an OWL 360 dome camera's `projection` and `dome_angle`). Use these names in outputs. |
| `ue_apply` | `bake_path`, `sequence_path`, `host`, `start_frame`, `only`, `save` | writes the bake into a Level Sequence (asset path like `/Game/Show/LS_Main`). `start_frame` shifts the bake: a key at frame f lands on sequence frame `bake.start_frame` + `start_frame` + f. Outputs with `blend` `multiply`/`add` go on top of the target's own animation: an additive section holding base × (gain − 1) or the offset, with the base section left active and untouched (shared sections are fine); with no base section, an absolute section of static value × gain. `blended` lists each track's base (a track name, or `static`). One blend per target. Returns track and key counts, binding counts and timings. |
| `ue_check` | `bake_path`, `sequence_path`, `host`, `start_frame`, `only` | reads the keys back and compares them with the bake, key for key (colour tracks: all four channels; Niagara tracks: the parameter name too; multiply/add tracks: against the offset, or static × gain, recomputed from the target's current base, so a base edited, deactivated or added since the apply fails). Returns the failing tracks only. |
| `ue_undo` | `sequence_path`, `output_id`, `host`, `save` | removes beatrig's tracks (all, or one output's) and the bindings beatrig created that are now empty; base sections under multiply/add outputs were never changed |
| `live_push` | `bake_path`, `sequence_path`, `only_changed`, `start_frame`, `host` | rewrites only the outputs whose keys changed since this server's last push to that sequence; outputs gone from the bake are removed. A multiply/add output whose base was edited in the editor counts as changed (`base_edited`). Re-evaluates the sequence if it is open in Sequencer. Returns `pushed`, `skipped`, `removed`, `base_edited` (output ids) and `ms`. |
| `playhead_get` | `host` | the sequence open in Sequencer, playhead frame, playing, fps, range |
| `playhead_set` | `frame`, `playing`, `sequence_path`, `host` | jump the playhead and/or play/pause; `sequence_path` opens that sequence in Sequencer first (replacing what it showed) |

`host` defaults to `BEATRIG_UE_HOST` or `127.0.0.1:30010`. All seven go through one persistent `beatrig.live.Link` per
host (applier imported once, keep-alive connection), so they cost tens of milliseconds each. The MCP server leaves the
editor's "Use Less CPU when in Background" setting alone; with it on and the editor in the background each call takes
about 330 ms. The editor reads beatrig's temp files, so it has to run on the same machine. Tested against UE 5.8.3;
setup, what is written for each target kind, timings and gotchas: [unreal.md](unreal.md).
