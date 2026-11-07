# Contributing to beatrig

Thanks for helping. A few things keep the project what it is.

## The rules the engine never breaks
1. **Nothing ahead of the audio.** Every op is causal: a frame reads audio only up to its own start time, a hit lands
   on the first frame at or after its onset, and delays are >= 0. No look-ahead anywhere, including smoothing.
2. **Eased motion.** Continuous signals are smoothed; only trigger envelopes may rise instantly. Keys are linear.
3. **Deterministic.** The same inputs give a byte-identical bake. Seeds are explicit; nothing reads the clock.
4. **Light.** CPU only, vectorised numpy. A 60-minute song should bake in about a minute.

A change that bends one of these needs a very good reason in the pull request. The ops reference
([docs/ops.md](docs/ops.md)) describes how each op keeps them.

## Set up
```bash
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[mcp,dev]"
python examples/make_demo_audio.py
python -m pytest -q
```

## Pull requests
- Add or update tests with the change. New ops need a causality test (no output peak before its input onset) and a
  determinism test.
- Keep `rig.json`, `notes.json` and `bake.json` backward compatible, or bump the format version and say why.
- New MCP tools: small JSON results, clear errors that say what to do next, validate before writing, return a diff,
  address things by id. Never print to stdout inside a tool (stdout is the MCP transport).
- Anything that touches `ue/beatrig_apply.py` should say which engine version it was run against. Pure logic belongs
  in `ue/keys.py`, where it can be tested without the editor.
- Match the surrounding style. No new heavy dependencies (no torch, no GUI toolkits) in the core install.

## Bugs
Open an issue with the rig (or a minimal one), the command or tool call, what you expected and what happened. If it's
about timing, a `plot` image of the output helps a lot.
