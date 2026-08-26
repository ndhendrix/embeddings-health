# olmoearth-activations

Per-patch embeddings and intermediate-layer activations from OlmoEarth
pretrained models, plus the analysis needed to ask which image patches drive
which activation dimensions.

The prototypes this replaces, `extract_activations.py` and `sweep_patch.py`, are
still in this directory and are untouched. Delete them once the ports have been
run against a real checkpoint.

## Install

```bash
uv pip install -e '.[dev]'

# olmoearth_pretrain is not on PyPI. PIN THE COMMIT -- the published CONUS
# embeddings were produced with an unpinned install, and that version is now
# unrecoverable. Do not repeat it.
uv pip install --no-deps 'git+https://github.com/allenai/olmoearth_pretrain@<COMMIT_SHA>'
uv pip install 'einops>=0.7' huggingface_hub 'universal-pathlib>=0.2.5'
```

Record the commit in `model.package_commit` in your config so it lands in every
run manifest.

## Imagery

Scenes live in the repo, at `data/scenes`, which is where `--scene-dir`
defaults to. Unzip Sentinel-2 **L2A** products there so the layout is
`data/scenes/*.SAFE/GRANULE/*/IMG_DATA/{R10m,R20m,R60m}/*.jp2` -- L1C will not
match, because its bands are not in resolution subfolders. `data/` is gitignored,
which matters at roughly a gigabyte per scene.

Keep one scene per directory unless you mean otherwise: each band is globbed
across every `*.SAFE` present and the first sorted match wins, with only a
logged warning to tell you.

## Use

```bash
# One location -> embeddings + activations + manifest
python scripts/encode_location.py --lat 38.90324 --lon -77.036964 \
    --config configs/default.yaml \
    --out artifacts/dc --name dc_downtown

# Figures from a saved result (no model, no GPU)
python scripts/inspect_dimensions.py --npz artifacts/dc/dc_downtown.npz \
    --chip-npz artifacts/dc/dc_downtown_chip.npz --out artifacts/dc/figures

# The urban/rural first look
python scripts/compare_locations.py --locations configs/locations.example.yaml \
    --config configs/default.yaml --out artifacts/compare

# Does chip position matter? (The prototype's answer was: barely.)
python scripts/sweep_position.py --lat 38.90324 --lon -77.036964 \
    --out artifacts/sweep
```

Library use:

```python
from olmoearth_activations import (
    Encoder, OlmoEarthModel, RunConfig, SafeSceneSource, analysis, default_scene_dir,
    load_chip,
)

cfg = RunConfig.from_yaml("configs/default.yaml")
model = OlmoEarthModel.load(cfg.model)
source = SafeSceneSource(scene_dir=default_scene_dir())
chip = load_chip(source, 38.90324, -77.036964, cfg.tile)

result = Encoder(model, cfg.tile, cfg.tap).encode(chip, latlon=(38.90324, -77.036964))
result.grid("blk2").shape      # (H', W', D) -- one vector per 40 m patch
analysis.most_variable_dims(result, k=10)
analysis.top_patches(result, dim=17, k=10)
```

## Design notes worth knowing before you trust a number

**One pass, two outputs.** `encode()` returns embeddings *and* activations,
because they are the same forward pass. There is no second "get activations"
call.

**The token axis is never averaged in `encode`.** Pooling lives in `analysis`
as an explicit call. Keeping every patch tied to its location is the point.

**`token_exit` vs `hooks` are not interchangeable.** `token_exit` (the default)
runs one forward pass per depth and returns representations that have all passed
the encoder's final LayerNorm, so they are on a common scale. `hooks` gets every
depth from one pass but captures the residual stream *before* that LayerNorm --
so the deepest hooked tap is not the encoder's returned tokens, the taps are not
mutually comparable in scale, and depth 0 is unavailable. An R²-versus-depth
curve built from hooks is partly measuring token norms.
`tests/test_encode_integration.py::test_hooks_and_token_exit_relationship`
measures the actual relationship and prints it rather than assuming one.

**Normalization is stateless, and production gets it wrong.** Both `Normalizer`
strategies use fixed per-band constants, so normalizing a region then slicing
equals slicing then normalizing -- which is why the position sweep normalizes
once. Separately: the repo's production embedding pipeline
(`embedding_generation/embed.py`) applies *no* normalization and feeds raw
digital numbers to a model trained on normalized input. `normalize: "none"`
exists here so that path can be reproduced deliberately; `"computed"` is the
default because it is what the pretraining pipeline applies.

**Chip size is a modelling choice.** Tokens near a chip edge see less context.
The default here is 64 px for speed; production uses 128 px. A 64 px chip will
not reproduce production embeddings numerically.

**Nothing is hardcoded.** Depth, embedding dimension, grid shape, band-set count
and band order are all read off the loaded model or its output. The grid shape in
particular comes from the encoder's own returned tensor, not from
`chip_px // patch_px`, so anything the model adds to the token sequence shows up
as a loud error rather than a silent misalignment of every patch.

**The `ModelID` enum depends on your install, not on HuggingFace.** Older
`olmoearth_pretrain` versions carry only the v1 members and cannot build a v1.1
or v1.2 config. `OlmoEarthModel.load` raises with the installed member list and
the fix; `allow_path_fallback=True` is a documented, off-by-default escape hatch.

## Tests

```bash
pytest                  # synthetic fixtures; no checkpoint, no network
pytest -m integration   # needs a checkpoint download
```

The unit suite deliberately avoids torch: top-level imports are lazy (PEP 562)
so `olmoearth_activations.config` works on a machine with nothing but PyYAML, and
saved `.npz` results can be analysed without torch or a checkpoint.
