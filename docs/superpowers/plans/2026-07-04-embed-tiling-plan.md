# Embed Pipeline Sub-Tiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split each state's OlmoEarth-Nano/Clay/Base embedding work into independently-schedulable, resumable row-band tiles read directly from `$SCRATCH` (no staging copy), so every state finishes within Sherlock's fixed walltime limits.

**Architecture:** `embed.py` gains `--tile-index`/`--num-tiles`/`--merge-only`; a new shared `utils/tile_merge.py` mosaics tile outputs (extracted from `composite.py`'s existing `_merge_tiles`); each pipeline's sbatch/submit scripts switch from one-task-per-state to a flat one-task-per-(state,tile) array plus a dependent merge array, mirroring `composite.py`'s already-proven tile/merge pattern.

**Tech Stack:** Python 3.11, rasterio, numpy, bash/Slurm sbatch, uv.

## Global Constraints

- No pytest/test framework exists in this repo — tests are plain assert-based Python scripts run directly (`uv run --python 3.11 python <script>`), matching existing conventions (`validate_embedding.py`).
- Per `[[testing-before-submitting]]` memory: test new code paths interactively via `srun` before submitting any batch array. Do not submit a real production batch array as part of this plan without explicit user go-ahead — Task 10 is a manual checkpoint, not an automated step.
- Sherlock's node-local `$L_SCRATCH` variable, not `$SLURM_TMPDIR` (already fixed in the staging-leak work; not touched further here since tiling removes staging from these three pipelines entirely).
- Composites are already COG-structured (tiled, block-organized GeoTIFFs) — direct windowed reads from `$SCRATCH` are expected to be efficient without local staging (per approved spec `docs/superpowers/specs/2026-07-04-embed-tiling-design.md`).
- Tiling is row-band only (split the chip grid by rows, full width per tile) — simpler than `composite.py`'s 2-D geographic grid, and equally correct here since embed.py does windowed reads against an already-materialized raster (no STAC-locality benefit to square tiles). This is a plan-level refinement of the spec's "roughly-square" language; the spec's actual requirement (bounded, roughly-equal per-tile chip counts) is preserved.
- Tile-count planning (the spec's `plan_tiles()`) is implemented as bash arithmetic inside each submit script (`get_chips()`/`get_h_chips()` + a division by `TARGET_CHIPS_PER_TILE`), not a new Python helper — it reuses the `gdalinfo`-based `get_chips()` bash function each submit script already has today, rather than adding a second, redundant way to read a composite's dimensions. This is a plan-level implementation simplification of the spec's Python-helper suggestion; the requirement itself (submit-time tile-count planning from composite dimensions) is preserved.
- Scope: OlmoEarth Base (`oe-embed`), OlmoEarth Nano (`oe-nano-embed`), Clay (`clay-embed`). Prithvi and the stuck TX/CA composite merge are explicitly out of scope.
- Implementation order: Nano → Clay → Base (fastest to validate first, per approved testing plan).

---

## Task 1: Extract shared `merge_tiles()` helper

**Files:**
- Create: `code/embedding_generation/utils/tile_merge.py`
- Modify: `code/embedding_generation/composite.py:302-363` (remove `_merge_tiles`, import shared version)
- Test: `code/embedding_generation/tests/test_tile_merge.py`

**Interfaces:**
- Produces: `merge_tiles(tile_paths: list[Path], band_names: list[str], out_path: Path) -> None` — mosaics tile GeoTIFFs into `out_path` via windowed writes, resumable via a `.merge_ckpt` sidecar, deletes `tile_paths` on success. Used by both `composite.py` (Task 1) and `embed.py` (Task 2).

- [ ] **Step 1: Write the failing test**

Create `code/embedding_generation/tests/test_tile_merge.py`:

```python
"""Plain assert-based test for utils.tile_merge.merge_tiles (no pytest in this repo).

Run: uv run --python 3.11 python tests/test_tile_merge.py
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from utils.tile_merge import merge_tiles


def make_tile(path: Path, value: float, top: float, left: float,
              width: int, height: int, n_bands: int = 2) -> None:
    transform = from_origin(left, top, 10, 10)
    profile = dict(
        driver="GTiff", dtype="float32", count=n_bands,
        height=height, width=width, crs="EPSG:32610", transform=transform,
        nodata=np.nan,
    )
    with rasterio.open(path, "w", **profile) as dst:
        for b in range(1, n_bands + 1):
            dst.write(np.full((height, width), value * b, dtype="float32"), indexes=b)


def test_merge_reproduces_expected_mosaic():
    tmp = Path(tempfile.mkdtemp())
    try:
        # Two vertically-stacked tiles: tile0 on top (higher y), tile1 below.
        tile0 = tmp / "emb_tile000.tif"
        tile1 = tmp / "emb_tile001.tif"
        make_tile(tile0, value=1.0, top=100.0, left=0.0, width=4, height=4)
        make_tile(tile1, value=2.0, top=60.0,  left=0.0, width=4, height=4)

        out_path = tmp / "emb.tif"
        merge_tiles([tile0, tile1], ["B01", "B02"], out_path)

        assert out_path.exists(), "merged output was not written"
        assert not tile0.exists(), "tile0 should be deleted after merge"
        assert not tile1.exists(), "tile1 should be deleted after merge"

        with rasterio.open(out_path) as merged:
            assert merged.height == 8, f"expected height 8, got {merged.height}"
            assert merged.width == 4, f"expected width 4, got {merged.width}"
            band1 = merged.read(1)
            # Top 4 rows come from tile0 (value 1.0 * band1 = 1.0)
            assert np.allclose(band1[:4, :], 1.0), "top rows should be tile0's values"
            # Bottom 4 rows come from tile1 (value 2.0 * band1 = 2.0)
            assert np.allclose(band1[4:, :], 2.0), "bottom rows should be tile1's values"
            assert merged.descriptions == ("B01", "B02")

        print("test_merge_reproduces_expected_mosaic: PASS")
    finally:
        shutil.rmtree(tmp)


def test_merge_resumes_from_checkpoint():
    tmp = Path(tempfile.mkdtemp())
    try:
        tile0 = tmp / "emb_tile000.tif"
        tile1 = tmp / "emb_tile001.tif"
        make_tile(tile0, value=1.0, top=100.0, left=0.0, width=4, height=4)
        make_tile(tile1, value=2.0, top=60.0,  left=0.0, width=4, height=4)
        out_path = tmp / "emb.tif"

        # Simulate a partial merge: tmp file + checkpoint recording tile0 done.
        tmp_path = out_path.with_suffix(".tmp.tif")
        ckpt_path = out_path.with_suffix(".merge_ckpt")
        transform = from_origin(0.0, 100.0, 10, 10)
        profile = dict(
            driver="GTiff", dtype="float32", count=2, height=8, width=4,
            crs="EPSG:32610", transform=transform, nodata=np.nan,
            compress="deflate", tiled=True, blockxsize=512, blockysize=512,
            BIGTIFF="YES",
        )
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(np.full((4, 4), 1.0, dtype="float32"), indexes=1, window=rasterio.windows.Window(0, 0, 4, 4))
            dst.write(np.full((4, 4), 2.0, dtype="float32"), indexes=2, window=rasterio.windows.Window(0, 0, 4, 4))
        ckpt_path.write_text(str(tile0))

        merge_tiles([tile0, tile1], ["B01", "B02"], out_path)

        assert out_path.exists()
        assert not tile1.exists(), "tile1 should be merged and deleted on resume"
        with rasterio.open(out_path) as merged:
            band1 = merged.read(1)
            assert np.allclose(band1[4:, :], 2.0), "resumed merge should have written tile1's rows"

        print("test_merge_resumes_from_checkpoint: PASS")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_merge_reproduces_expected_mosaic()
    test_merge_resumes_from_checkpoint()
    print("ALL PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run (from a compute node — this repo's policy is no direct python on the login node):

```bash
srun -p normal -c 2 --mem=8G --time=00:10:00 bash -c \
  "cd $HOME/embeddings-health/code/embedding_generation && uv run --python 3.11 python tests/test_tile_merge.py"
```

Expected: `FAIL` with `ModuleNotFoundError: No module named 'utils.tile_merge'`.

- [ ] **Step 3: Create `utils/tile_merge.py`**

```python
"""Shared tile-mosaic helper.

Merges per-tile GeoTIFFs into one output file via windowed writes (no
full-scene RAM allocation), resumable via a `.merge_ckpt` sidecar that
records which tiles are already written — an interrupted merge (e.g. a
Slurm timeout on a huge state) resumes instead of restarting.

Used by composite.py (composite tiles) and embed.py (embedding tiles).
"""
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows


def merge_tiles(tile_paths: list[Path], band_names: list[str], out_path: Path) -> None:
    """Mosaic tile TIFs via windowed writes — no full-scene RAM allocation."""
    datasets = [rasterio.open(p) for p in tile_paths]

    out_left   = min(ds.bounds.left   for ds in datasets)
    out_bottom = min(ds.bounds.bottom for ds in datasets)
    out_right  = max(ds.bounds.right  for ds in datasets)
    out_top    = max(ds.bounds.top    for ds in datasets)

    res_x, res_y = datasets[0].res
    out_transform = rasterio.transform.from_origin(out_left, out_top, res_x, res_y)
    out_width  = round((out_right  - out_left)   / res_x)
    out_height = round((out_top    - out_bottom) / res_y)

    profile = datasets[0].profile.copy()
    profile.update(
        height=out_height, width=out_width, transform=out_transform,
        compress="deflate", tiled=True, blockxsize=512, blockysize=512,
        nodata=np.nan, BIGTIFF="YES",
    )

    tmp_path = out_path.with_suffix(".tmp.tif")
    ckpt_path = out_path.with_suffix(".merge_ckpt")

    # Resume an interrupted merge if both tmp file and checkpoint exist.
    resuming = tmp_path.exists() and ckpt_path.exists()
    already_merged: set[str] = set()
    if resuming:
        already_merged = set(ckpt_path.read_text().splitlines())
        print(f"      Resuming merge: {len(already_merged)}/{len(tile_paths)} tiles already written")
        dst_ctx = rasterio.open(tmp_path, "r+")
    else:
        ckpt_path.unlink(missing_ok=True)
        dst_ctx = rasterio.open(tmp_path, "w", **profile)

    with dst_ctx as dst:
        if not resuming:
            for i, band_name in enumerate(band_names, 1):
                dst.set_band_description(i, band_name)
        for idx, ds in enumerate(datasets):
            tile_key = str(tile_paths[idx])
            if tile_key in already_merged:
                ds.close()
                continue
            window = rasterio.windows.from_bounds(
                ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top,
                transform=out_transform,
            ).round_offsets().round_lengths()
            for band_idx in range(1, ds.count + 1):
                dst.write(ds.read(band_idx), indexes=band_idx, window=window)
            ds.close()
            already_merged.add(tile_key)
            ckpt_path.write_text("\n".join(sorted(already_merged)))

    tmp_path.rename(out_path)
    ckpt_path.unlink(missing_ok=True)
    for p in tile_paths:
        p.unlink()
        print(f"      Removed tile: {p.name}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
srun -p normal -c 2 --mem=8G --time=00:10:00 bash -c \
  "cd $HOME/embeddings-health/code/embedding_generation && uv run --python 3.11 python tests/test_tile_merge.py"
```

Expected: `test_merge_reproduces_expected_mosaic: PASS`, `test_merge_resumes_from_checkpoint: PASS`, `ALL PASSED`.

- [ ] **Step 5: Update `composite.py` to use the shared helper**

In `code/embedding_generation/composite.py`, add the import near the top (after the existing `from utils.cloud_mask import mask_s2_l2a` line):

```python
from utils.cloud_mask import mask_s2_l2a
from utils.tile_merge import merge_tiles
```

Delete the `_merge_tiles` function body (lines 302-363, the entire `def _merge_tiles(...): ...` block) — it is now `merge_tiles` in `utils/tile_merge.py`.

Find the one call site (originally around line 409):
```python
        _merge_tiles(tile_paths, bands, out_path)
```
Replace with:
```python
        merge_tiles(tile_paths, bands, out_path)
```

- [ ] **Step 6: Verify composite.py still imports cleanly**

```bash
srun -p normal -c 2 --mem=8G --time=00:10:00 bash -c \
  "cd $HOME/embeddings-health/code/embedding_generation && uv run --python 3.11 python -c 'import composite; print(\"composite.py imports OK\")'"
```

Expected: `composite.py imports OK` (no `NameError: _merge_tiles` or import errors).

- [ ] **Step 7: Commit**

```bash
cd "$HOME/embeddings-health"
git add code/embedding_generation/utils/tile_merge.py \
        code/embedding_generation/composite.py \
        code/embedding_generation/tests/test_tile_merge.py
git commit -m "$(cat <<'EOF'
refactor: extract shared merge_tiles() into utils/tile_merge.py

composite.py and the upcoming embed.py tiling both need the same
windowed tile-mosaic logic. Pulling it into utils/ avoids duplicating
~60 lines of merge/resume logic across two CLI tools.
EOF
)"
```

---

## Task 2: Add row-band tiling core to `embed.py`

**Files:**
- Modify: `code/embedding_generation/embed.py`
- Test: `code/embedding_generation/tests/test_embed_tiling.py`

**Interfaces:**
- Consumes: `merge_tiles(tile_paths, band_names, out_path)` from Task 1.
- Produces:
  - `tile_row_bounds(n_row_chips: int, tile_index: int, num_tiles: int) -> tuple[int, int]` — pure function, chip-row indices, `row_end` exclusive.
  - `iter_chips(src, chip_px, row_px_bounds: tuple[int,int] | None = None)` — now accepts optional pixel-row bounds.
  - `embed_olmoearth(..., row_px_bounds: tuple[int,int] | None = None)` and `embed_clay(..., row_px_bounds: tuple[int,int] | None = None)` — new optional kwarg, output array sized/offset to the tile.
  - CLI: `--tile-index`, `--num-tiles`, `--merge-only` on `embed.py`.

- [ ] **Step 1: Write the failing test**

Create `code/embedding_generation/tests/test_embed_tiling.py`:

```python
"""Plain assert-based tests for embed.py's tiling helpers (no pytest in this repo).

Run: uv run --python 3.11 python tests/test_embed_tiling.py
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin

from embed import tile_row_bounds, iter_chips


def test_tile_row_bounds_partitions_evenly():
    # 12 rows split into 3 tiles -> 4 rows each, no gaps/overlaps.
    bounds = [tile_row_bounds(12, i, 3) for i in range(3)]
    assert bounds == [(0, 4), (4, 8), (8, 12)], bounds
    print("test_tile_row_bounds_partitions_evenly: PASS")


def test_tile_row_bounds_distributes_remainder():
    # 10 rows split into 3 tiles -> remainder rows go to the first tiles.
    bounds = [tile_row_bounds(10, i, 3) for i in range(3)]
    assert bounds == [(0, 4), (4, 7), (7, 10)], bounds
    total_rows = sum(end - start for start, end in bounds)
    assert total_rows == 10
    # No gaps or overlaps: each tile's end == next tile's start.
    for i in range(len(bounds) - 1):
        assert bounds[i][1] == bounds[i + 1][0]
    print("test_tile_row_bounds_distributes_remainder: PASS")


def test_tile_row_bounds_single_tile_covers_everything():
    assert tile_row_bounds(37, 0, 1) == (0, 37)
    print("test_tile_row_bounds_single_tile_covers_everything: PASS")


def test_tile_row_bounds_rejects_out_of_range():
    try:
        tile_row_bounds(10, 3, 3)
        raise AssertionError("expected ValueError for tile_index >= num_tiles")
    except ValueError:
        pass
    try:
        tile_row_bounds(10, 0, 20)
        raise AssertionError("expected ValueError for num_tiles > n_row_chips")
    except ValueError:
        pass
    print("test_tile_row_bounds_rejects_out_of_range: PASS")


def test_iter_chips_row_bounds_restricts_iteration():
    # 4x4 chip grid of 2x2-px chips = 8x8 px raster.
    arr = np.arange(64, dtype="float32").reshape(1, 8, 8)
    transform = from_origin(0.0, 8.0, 1, 1)
    profile = dict(driver="GTiff", dtype="float32", count=1, height=8, width=8,
                    crs="EPSG:32610", transform=transform)
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mktemp(suffix=".tif"))
    try:
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(arr)
        with rasterio.open(tmp) as src:
            # Full raster: 4x4 = 16 chips.
            all_chips = list(iter_chips(src, chip_px=2))
            assert len(all_chips) == 16, len(all_chips)

            # Restrict to rows [2, 4) (pixel rows, i.e. chip-rows 1..1 of the 4-row grid)
            # -> should yield exactly the 2 chips in that row (row_off == 2), covering
            # both columns.
            restricted = list(iter_chips(src, chip_px=2, row_px_bounds=(2, 4)))
            assert len(restricted) == 2, len(restricted)
            assert all(row_off == 2 for row_off, col_off, win, data in restricted)
            cols = sorted(col_off for row_off, col_off, win, data in restricted)
            assert cols == [0, 2], cols
    finally:
        tmp.unlink(missing_ok=True)
    print("test_iter_chips_row_bounds_restricts_iteration: PASS")


if __name__ == "__main__":
    test_tile_row_bounds_partitions_evenly()
    test_tile_row_bounds_distributes_remainder()
    test_tile_row_bounds_single_tile_covers_everything()
    test_tile_row_bounds_rejects_out_of_range()
    test_iter_chips_row_bounds_restricts_iteration()
    print("ALL PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
srun -p normal -c 2 --mem=8G --time=00:10:00 bash -c \
  "cd $HOME/embeddings-health/code/embedding_generation && uv run --python 3.11 python tests/test_embed_tiling.py"
```

Expected: `FAIL` with `ImportError: cannot import name 'tile_row_bounds' from 'embed'`.

- [ ] **Step 3: Add `tile_row_bounds()` to `embed.py`**

In `code/embedding_generation/embed.py`, add this function immediately before `def iter_chips(...)` (around line 534):

```python
def tile_row_bounds(n_row_chips: int, tile_index: int, num_tiles: int) -> tuple[int, int]:
    """Split n_row_chips chip-rows into num_tiles contiguous, roughly-equal bands.

    Returns (row_start, row_end) as chip-row indices (row_end exclusive) for
    tile_index. Splitting by row only (not a 2-D grid) is sufficient here:
    unlike composite.py's per-tile STAC searches, embed.py does windowed reads
    against an already-materialized raster, so there's no locality benefit to
    square tiles — only the total per-tile chip count matters.
    """
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if tile_index < 0 or tile_index >= num_tiles:
        raise ValueError(f"tile_index {tile_index} out of range [0, {num_tiles})")
    if num_tiles > n_row_chips:
        raise ValueError(
            f"num_tiles ({num_tiles}) exceeds n_row_chips ({n_row_chips}); reduce num_tiles."
        )
    base, rem = divmod(n_row_chips, num_tiles)
    if tile_index < rem:
        row_start = tile_index * (base + 1)
        row_end = row_start + (base + 1)
    else:
        row_start = rem * (base + 1) + (tile_index - rem) * base
        row_end = row_start + base
    return row_start, row_end
```

- [ ] **Step 4: Add `row_px_bounds` to `iter_chips()`**

Replace the existing `iter_chips` function:

```python
def iter_chips(src: rasterio.DatasetReader, chip_px: int):
    """Yield (row_off, col_off, win, chip_data) for non-overlapping chips.

    Edge chips are zero-padded to chip_px × chip_px.
    chip_data: (C, chip_px, chip_px) float32.
    """
    h, w = src.height, src.width
    for row_off in range(0, h, chip_px):
        for col_off in range(0, w, chip_px):
            read_h = min(chip_px, h - row_off)
            read_w = min(chip_px, w - col_off)
            win = Window(col_off, row_off, read_w, read_h)
            data = src.read(window=win).astype("float32")
            if read_h < chip_px or read_w < chip_px:
                pad = np.zeros((data.shape[0], chip_px, chip_px), dtype="float32")
                pad[:, :read_h, :read_w] = data
                data = pad
            yield row_off, col_off, win, data
```

with:

```python
def iter_chips(src: rasterio.DatasetReader, chip_px: int,
               row_px_bounds: tuple[int, int] | None = None):
    """Yield (row_off, col_off, win, chip_data) for non-overlapping chips.

    Edge chips are zero-padded to chip_px × chip_px.
    chip_data: (C, chip_px, chip_px) float32.
    row_px_bounds, if given, restricts iteration to pixel rows
    [row_px_bounds[0], row_px_bounds[1]) of the full raster — used for tiling.
    Columns are never restricted (tiling is row-band only).
    """
    h, w = src.height, src.width
    row_start, row_end = row_px_bounds if row_px_bounds is not None else (0, h)
    for row_off in range(row_start, row_end, chip_px):
        for col_off in range(0, w, chip_px):
            read_h = min(chip_px, h - row_off)
            read_w = min(chip_px, w - col_off)
            win = Window(col_off, row_off, read_w, read_h)
            data = src.read(window=win).astype("float32")
            if read_h < chip_px or read_w < chip_px:
                pad = np.zeros((data.shape[0], chip_px, chip_px), dtype="float32")
                pad[:, :read_h, :read_w] = data
                data = pad
            yield row_off, col_off, win, data
```

- [ ] **Step 5: Run test to verify it passes**

```bash
srun -p normal -c 2 --mem=8G --time=00:10:00 bash -c \
  "cd $HOME/embeddings-health/code/embedding_generation && uv run --python 3.11 python tests/test_embed_tiling.py"
```

Expected: all five `PASS` lines, then `ALL PASSED`.

- [ ] **Step 6: Commit**

```bash
cd "$HOME/embeddings-health"
git add code/embedding_generation/embed.py code/embedding_generation/tests/test_embed_tiling.py
git commit -m "$(cat <<'EOF'
feat: add tile_row_bounds() and row-bounded iter_chips() to embed.py

Pure chip-grid-row splitting, unit tested. No CLI wiring yet — this is
the foundation the tile-index/num-tiles CLI args (next commit) build on.
EOF
)"
```

- [ ] **Step 7: Wire `row_px_bounds` through `embed_olmoearth()`**

In `embed_olmoearth` (around line 650), change the signature:

```python
def embed_olmoearth(
    src: rasterio.DatasetReader,
    model,
    device: torch.device,
    batch_size: int,
    test_chips: int | None,
    year: int,
    ckpt_path: Path | None = None,
    checkpoint_every: int = 500,
    embed_dim: int = OLMOEARTH_EMBED_DIM,
    row_px_bounds: tuple[int, int] | None = None,
) -> np.ndarray:
```

Update the docstring's return-shape note to mention tiling:

```python
    """Return (embed_dim, H_out, W_out) embedding map at 40m effective resolution.

    Output shape: (embed_dim, ceil(H/4), ceil(W/4)) for the full raster, or
    (embed_dim, ceil(tile_H/4), ceil(W/4)) when row_px_bounds restricts to a
    row-band tile — H_out is always relative to the tile's own top row.

    Periodically checkpoints to ckpt_path so interrupted jobs can resume.

    When the output array would exceed _MEMMAP_THRESHOLD_BYTES (8 GB by default),
    a disk-backed numpy memmap is used automatically so large states (TX, CA, MT…)
    don't OOM. The memmap file lives next to the checkpoint as <ckpt>.mmap and is
    removed by checkpoint_delete() after the COG is written.
    """
```

Change the shape computation. Replace:

```python
    h, w = src.height, src.width
    stride = OLMOEARTH_STRIDE_PX
    out_h = (h + stride - 1) // stride
    out_w = (w + stride - 1) // stride
    shape = (embed_dim, out_h, out_w)
```

with:

```python
    h, w = src.height, src.width
    stride = OLMOEARTH_STRIDE_PX
    row_start_px, row_end_px = row_px_bounds if row_px_bounds is not None else (0, h)
    tile_h = row_end_px - row_start_px
    out_h = (tile_h + stride - 1) // stride
    out_w = (w + stride - 1) // stride
    shape = (embed_dim, out_h, out_w)
```

Change the chip-iteration call. Replace:

```python
    for batch in tqdm(chips_to_batches(iter_chips(src, OLMOEARTH_CHIP_PX), batch_size),
                      desc=f"OlmoEarth chips (dim={embed_dim})", initial=n_skip // batch_size):
```

with:

```python
    for batch in tqdm(chips_to_batches(iter_chips(src, OLMOEARTH_CHIP_PX, row_px_bounds), batch_size),
                      desc=f"OlmoEarth chips (dim={embed_dim})", initial=n_skip // batch_size):
```

Change the output-offset computation inside the loop. Replace:

```python
        for i, (row_off, col_off, win) in enumerate(zip(rows, cols, wins)):
            out_r = row_off // stride
            out_c = col_off // stride
```

with:

```python
        for i, (row_off, col_off, win) in enumerate(zip(rows, cols, wins)):
            out_r = (row_off - row_start_px) // stride
            out_c = col_off // stride
```

- [ ] **Step 8: Wire `row_px_bounds` through `embed_clay()`**

Apply the identical pattern to `embed_clay` (around line 824). Change the signature:

```python
def embed_clay(
    src: rasterio.DatasetReader,
    model,
    device: torch.device,
    batch_size: int,
    test_chips: int | None,
    year: int,
    ckpt_path: Path | None = None,
    checkpoint_every: int = 500,
    row_px_bounds: tuple[int, int] | None = None,
) -> np.ndarray:
```

Replace:

```python
    h, w = src.height, src.width
    stride = CLAY_STRIDE_PX
    out_h = (h + stride - 1) // stride
    out_w = (w + stride - 1) // stride
    shape = (CLAY_EMBED_DIM, out_h, out_w)
```

with:

```python
    h, w = src.height, src.width
    stride = CLAY_STRIDE_PX
    row_start_px, row_end_px = row_px_bounds if row_px_bounds is not None else (0, h)
    tile_h = row_end_px - row_start_px
    out_h = (tile_h + stride - 1) // stride
    out_w = (w + stride - 1) // stride
    shape = (CLAY_EMBED_DIM, out_h, out_w)
```

Replace:

```python
    for batch in tqdm(chips_to_batches(iter_chips(src, CLAY_CHIP_PX), batch_size),
                      desc="Clay chips", initial=n_skip // batch_size):
```

with:

```python
    for batch in tqdm(chips_to_batches(iter_chips(src, CLAY_CHIP_PX, row_px_bounds), batch_size),
                      desc="Clay chips", initial=n_skip // batch_size):
```

Replace:

```python
        for i, (row_off, col_off, win) in enumerate(zip(rows, cols, wins)):
            out_r = row_off // stride
            out_c = col_off // stride
```

with:

```python
        for i, (row_off, col_off, win) in enumerate(zip(rows, cols, wins)):
            out_r = (row_off - row_start_px) // stride
            out_c = col_off // stride
```

- [ ] **Step 9: Add CLI args and tile-path resolution to `main()`**

In `main()`, change `--input` to not be unconditionally required, and add the new tile args. Replace:

```python
    parser.add_argument("--model", choices=["olmoearth", "prithvi", "clay"], required=True)
    parser.add_argument("--input", nargs="+", required=True,
                        help="Composite TIF(s): one for OlmoEarth/Clay; one-to-four for Prithvi.")
    parser.add_argument("--output", type=Path, required=True)
```

with:

```python
    parser.add_argument("--model", choices=["olmoearth", "prithvi", "clay"], required=True)
    parser.add_argument("--input", nargs="+", default=None,
                        help="Composite TIF(s): one for OlmoEarth/Clay; one-to-four for Prithvi. "
                             "Required unless --merge-only is set.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-index", type=int, default=0,
                        help="Which row-band tile to process (0-indexed). Only meaningful "
                             "with --num-tiles > 1.")
    parser.add_argument("--num-tiles", type=int, default=1,
                        help="Split the input into this many row-band tiles; each run with "
                             "a given --tile-index processes one band and writes a standalone "
                             "<output.stem>_tile###<output.suffix> file. Default 1 processes "
                             "the whole raster and writes directly to --output (today's "
                             "behavior, unchanged). OlmoEarth/Clay only — not supported for "
                             "--model prithvi.")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip inference; mosaic existing <output.stem>_tile###<output.suffix> "
                             "files into --output. Exits with a message (not an error) if any "
                             "expected tiles are still missing, or if --output already exists.")
```

After `args = parser.parse_args()`, add validation and the tile-suffixed path resolution:

```python
    args = parser.parse_args()

    if not args.merge_only and not args.input:
        parser.error("--input is required unless --merge-only is set")
    if args.num_tiles > 1 and args.model == "prithvi":
        parser.error("--num-tiles > 1 is not supported for --model prithvi")
    if args.pca and args.num_tiles > 1:
        parser.error(
            "--pca with --num-tiles > 1 is not supported: PCA would be fit "
            "independently per tile, producing incompatible bases across tiles. "
            "Apply --pca only after merging (num_tiles=1)."
        )
    if not (0 <= args.tile_index < max(args.num_tiles, 1)):
        parser.error(f"--tile-index {args.tile_index} out of range [0, {args.num_tiles})")
```

- [ ] **Step 10: Add `--merge-only` early-exit branch**

Immediately after the validation block from Step 9 (still before the `device = torch.device(...)` line), add:

```python
    if args.merge_only:
        if args.output.exists() and not args.force:
            print(f"Already exists, skipping merge: {args.output}")
            return
        tile_paths = sorted(
            args.output.parent.glob(f"{args.output.stem}_tile*{args.output.suffix}")
        )
        if len(tile_paths) != args.num_tiles:
            print(f"SKIP merge: expected {args.num_tiles} tiles, found {len(tile_paths)} "
                  f"({args.num_tiles - len(tile_paths)} still missing)")
            return
        if args.model == "olmoearth":
            variant = args.variant or "Base"
            embed_dim = OLMOEARTH_EMBED_DIMS.get(variant, OLMOEARTH_EMBED_DIM)
            band_names = [f"OE{i:04d}" for i in range(embed_dim)]
        elif args.model == "clay":
            band_names = [f"CL{i:04d}" for i in range(CLAY_EMBED_DIM)]
        else:
            raise SystemExit(f"--merge-only is not supported for --model {args.model}")
        print(f"Merging {len(tile_paths)}/{args.num_tiles} tiles → {args.output}…")
        merge_tiles(tile_paths, band_names, args.output)
        print("Done.")
        return
```

Add the import at the top of the file (alongside the existing `from utils.cog_writer import write_cog`):

```python
from utils.cog_writer import write_cog
from utils.tile_merge import merge_tiles
```

- [ ] **Step 11: Resolve the tile-suffixed working output path**

After the `--force` cleanup block (which currently reads `ckpt_path = args.output.with_suffix(".ckpt.npy")` then the `if args.force:` block), the rest of `main()` uses `args.output` directly in several places. Replace:

```python
    # Resolve checkpoint path early so --force can clean it up before model loading.
    ckpt_path = args.output.with_suffix(".ckpt.npy")

    if args.force:
        for p in [args.output, ckpt_path, ckpt_path.with_suffix(".n"),
                  ckpt_path.with_suffix(".mmap")]:
            if p and p.exists():
                p.unlink()
        if args.raw_output and args.raw_output.exists():
            args.raw_output.unlink()
        print("--force: cleared existing outputs and checkpoints.")
```

with:

```python
    # When tiling, this task's true working output is <output>_tile###<suffix>,
    # not the final --output path (that name is reserved for the merged file).
    # num_tiles=1 (the default) writes directly to --output — today's behavior.
    if args.num_tiles > 1:
        output_path = args.output.with_name(
            f"{args.output.stem}_tile{args.tile_index:03d}{args.output.suffix}"
        )
        raw_output_path = (
            args.raw_output.with_name(
                f"{args.raw_output.stem}_tile{args.tile_index:03d}{args.raw_output.suffix}"
            )
            if args.raw_output else None
        )
    else:
        output_path = args.output
        raw_output_path = args.raw_output

    # Resolve checkpoint path early so --force can clean it up before model loading.
    ckpt_path = output_path.with_suffix(".ckpt.npy")

    if args.force:
        for p in [output_path, ckpt_path, ckpt_path.with_suffix(".n"),
                  ckpt_path.with_suffix(".mmap")]:
            if p and p.exists():
                p.unlink()
        if raw_output_path and raw_output_path.exists():
            raw_output_path.unlink()
        print("--force: cleared existing outputs and checkpoints.")
```

Now update the three remaining places that reference `args.output`/`args.raw_output` after this point. In the `olmoearth` branch, replace:

```python
        embed_dim = OLMOEARTH_EMBED_DIMS.get(variant, OLMOEARTH_EMBED_DIM)
        with rasterio.open(args.input[0]) as src:
            print(f"Input: {args.input[0]}  shape={src.count}×{src.height}×{src.width}  CRS={src.crs}")
            raw = embed_olmoearth(src, model, device, args.batch_size,
                                  args.test_chips, args.year,
                                  ckpt_path=ckpt_path,
                                  checkpoint_every=args.checkpoint_every,
                                  embed_dim=embed_dim)
            transform_in = src.transform
            crs_in = src.crs
```

with:

```python
        embed_dim = OLMOEARTH_EMBED_DIMS.get(variant, OLMOEARTH_EMBED_DIM)
        with rasterio.open(args.input[0]) as src:
            print(f"Input: {args.input[0]}  shape={src.count}×{src.height}×{src.width}  CRS={src.crs}")
            n_row_chips = (src.height + OLMOEARTH_CHIP_PX - 1) // OLMOEARTH_CHIP_PX
            row_px_bounds = None
            if args.num_tiles > 1:
                row_start, row_end = tile_row_bounds(n_row_chips, args.tile_index, args.num_tiles)
                row_px_bounds = (row_start * OLMOEARTH_CHIP_PX,
                                  min(row_end * OLMOEARTH_CHIP_PX, src.height))
                print(f"Tile {args.tile_index}/{args.num_tiles}: rows "
                      f"[{row_px_bounds[0]}, {row_px_bounds[1]}) of {src.height}")
            raw = embed_olmoearth(src, model, device, args.batch_size,
                                  args.test_chips, args.year,
                                  ckpt_path=ckpt_path,
                                  checkpoint_every=args.checkpoint_every,
                                  embed_dim=embed_dim,
                                  row_px_bounds=row_px_bounds)
            transform_in = (
                rasterio.windows.transform(
                    Window(0, row_px_bounds[0], src.width, row_px_bounds[1] - row_px_bounds[0]),
                    src.transform,
                )
                if row_px_bounds else src.transform
            )
            crs_in = src.crs
```

Apply the same pattern to the `clay` branch. Replace:

```python
            print(f"Input: {args.input[0]}  shape={src.count}×{src.height}×{src.width}  CRS={src.crs}")
            raw = embed_clay(src, model, device, args.batch_size,
                             args.test_chips, args.year,
                             ckpt_path=ckpt_path,
                             checkpoint_every=args.checkpoint_every)
            transform_in = src.transform
            crs_in = src.crs
```

with:

```python
            print(f"Input: {args.input[0]}  shape={src.count}×{src.height}×{src.width}  CRS={src.crs}")
            n_row_chips = (src.height + CLAY_CHIP_PX - 1) // CLAY_CHIP_PX
            row_px_bounds = None
            if args.num_tiles > 1:
                row_start, row_end = tile_row_bounds(n_row_chips, args.tile_index, args.num_tiles)
                row_px_bounds = (row_start * CLAY_CHIP_PX,
                                  min(row_end * CLAY_CHIP_PX, src.height))
                print(f"Tile {args.tile_index}/{args.num_tiles}: rows "
                      f"[{row_px_bounds[0]}, {row_px_bounds[1]}) of {src.height}")
            raw = embed_clay(src, model, device, args.batch_size,
                             args.test_chips, args.year,
                             ckpt_path=ckpt_path,
                             checkpoint_every=args.checkpoint_every,
                             row_px_bounds=row_px_bounds)
            transform_in = (
                rasterio.windows.transform(
                    Window(0, row_px_bounds[0], src.width, row_px_bounds[1] - row_px_bounds[0]),
                    src.transform,
                )
                if row_px_bounds else src.transform
            )
            crs_in = src.crs
```

Add the import at the top of the file (with the other `rasterio` imports):

```python
import rasterio
import rasterio.windows
from rasterio.transform import Affine
from rasterio.windows import Window
```

Finally, replace the two remaining `args.output`/`args.raw_output` references at the bottom of `main()`. Replace:

```python
    if args.raw_output and args.pca:
        print(f"Writing raw embedding COG: {args.raw_output}  {raw.shape}")
        write_cog(raw, out_transform, crs_in, args.raw_output, band_names=raw_band_names, overviews=False)
```

with:

```python
    if raw_output_path and args.pca:
        print(f"Writing raw embedding COG: {raw_output_path}  {raw.shape}")
        write_cog(raw, out_transform, crs_in, raw_output_path, band_names=raw_band_names, overviews=False)
```

Replace:

```python
        if args.pca_model and args.pca_model.exists():
            with open(args.pca_model, "rb") as f:
                pca = pickle.load(f)
            print(f"Loaded PCA from {args.pca_model}")
        else:
            print(f"Fitting PCA ({args.pca_dims} components)…")
            pca = fit_pca(raw, n_components=args.pca_dims)
            pca_path = args.output.with_suffix(".pca.pkl")
```

with:

```python
        if args.pca_model and args.pca_model.exists():
            with open(args.pca_model, "rb") as f:
                pca = pickle.load(f)
            print(f"Loaded PCA from {args.pca_model}")
        else:
            print(f"Fitting PCA ({args.pca_dims} components)…")
            pca = fit_pca(raw, n_components=args.pca_dims)
            pca_path = output_path.with_suffix(".pca.pkl")
```

Replace:

```python
    print(f"Writing COG: {args.output}  {final.shape}")
    write_cog(final, out_transform, crs_in, args.output, band_names=band_names, overviews=False)
    checkpoint_delete(ckpt_path)
    print("Done.")
```

with:

```python
    print(f"Writing COG: {output_path}  {final.shape}")
    write_cog(final, out_transform, crs_in, output_path, band_names=band_names, overviews=False)
    checkpoint_delete(ckpt_path)
    print("Done.")
```

- [ ] **Step 12: Verify embed.py still imports cleanly and --help works**

```bash
srun -p normal -c 2 --mem=8G --time=00:10:00 bash -c \
  "cd $HOME/embeddings-health/code/embedding_generation && uv run --python 3.11 python embed.py --help"
```

Expected: argparse help text prints, including `--tile-index`, `--num-tiles`, `--merge-only`, with no `ImportError`/`SyntaxError`.

- [ ] **Step 13: Re-run the tile_row_bounds/iter_chips unit tests (regression check)**

```bash
srun -p normal -c 2 --mem=8G --time=00:10:00 bash -c \
  "cd $HOME/embeddings-health/code/embedding_generation && uv run --python 3.11 python tests/test_embed_tiling.py && uv run --python 3.11 python tests/test_tile_merge.py"
```

Expected: `ALL PASSED` from both.

- [ ] **Step 14: Commit**

```bash
cd "$HOME/embeddings-health"
git add code/embedding_generation/embed.py
git commit -m "$(cat <<'EOF'
feat: wire --tile-index/--num-tiles/--merge-only into embed.py

OlmoEarth and Clay can now process a row-band slice of a composite
(bounded chip count per invocation) and later merge per-tile outputs
into the final file, mirroring composite.py's tile/merge pattern.
Prithvi and --pca are explicitly unsupported in combination with tiling.
EOF
)"
```

---

## Task 3: srun validation — tiled+merged output matches today's single-shot output

**Files:** none (validation only, no code changes)

**Interfaces:**
- Consumes: `embed.py --tile-index/--num-tiles/--merge-only` from Task 2.

- [ ] **Step 1: Run DC (a small, already-complete state) as a single shot for comparison**

```bash
srun -p gpu -G 1 -c 8 --mem=64G --time=00:30:00 bash -c '
  cd $HOME/embeddings-health/code/embedding_generation
  uv run --python 3.11 python embed.py \
    --model olmoearth --variant Nano \
    --input '"$SCRATCH"'/embeddings-health/olmoearth_composites/s2_annual_DC_2022_olmoearth.tif \
    --output /tmp/dc_single.tif \
    --batch-size 32 --checkpoint-every 500 --force
'
```

Expected: completes in well under 30 minutes (DC is tiny); prints `Done.`; `/tmp/dc_single.tif` exists.

- [ ] **Step 2: Run the same state forced into 3 tiles**

```bash
srun -p gpu -G 1 -c 8 --mem=64G --time=00:30:00 bash -c '
  cd $HOME/embeddings-health/code/embedding_generation
  for i in 0 1 2; do
    uv run --python 3.11 python embed.py \
      --model olmoearth --variant Nano \
      --input '"$SCRATCH"'/embeddings-health/olmoearth_composites/s2_annual_DC_2022_olmoearth.tif \
      --output /tmp/dc_tiled.tif \
      --tile-index $i --num-tiles 3 \
      --batch-size 32 --checkpoint-every 500 --force
  done
  ls -la /tmp/dc_tiled_tile*.tif
'
```

Expected: three files `dc_tiled_tile000.tif`, `dc_tiled_tile001.tif`, `dc_tiled_tile002.tif` exist, each with `--force` re-run producing no leftover stale checkpoint files.

- [ ] **Step 3: Merge the 3 tiles**

```bash
srun -p normal -c 2 --mem=8G --time=00:10:00 bash -c '
  cd $HOME/embeddings-health/code/embedding_generation
  uv run --python 3.11 python embed.py \
    --model olmoearth --variant Nano \
    --output /tmp/dc_tiled.tif \
    --num-tiles 3 --merge-only
  ls -la /tmp/dc_tiled.tif
'
```

Expected: prints `Merging 3/3 tiles → /tmp/dc_tiled.tif…`, `/tmp/dc_tiled.tif` exists, the three `_tile###.tif` files are gone (deleted by `merge_tiles`).

- [ ] **Step 4: Compare the merged output against the single-shot output**

```bash
srun -p normal -c 2 --mem=8G --time=00:10:00 bash -c '
  cd $HOME/embeddings-health/code/embedding_generation
  uv run --python 3.11 python -c "
import numpy as np
import rasterio

with rasterio.open(\"/tmp/dc_single.tif\") as a, rasterio.open(\"/tmp/dc_tiled.tif\") as b:
    assert a.height == b.height and a.width == b.width, (a.shape, b.shape)
    assert a.transform == b.transform, (a.transform, b.transform)
    assert a.crs == b.crs
    arr_a, arr_b = a.read(), b.read()
    nan_a, nan_b = np.isnan(arr_a), np.isnan(arr_b)
    assert np.array_equal(nan_a, nan_b), \"NaN masks differ between single-shot and tiled+merged\"
    valid = ~nan_a
    assert np.allclose(arr_a[valid], arr_b[valid], atol=1e-4), \"pixel values differ beyond tolerance\"
    print(\"Tiled+merged output matches single-shot output.\")
"
'
```

Expected: `Tiled+merged output matches single-shot output.` with no `AssertionError`.

- [ ] **Step 5: Clean up scratch test files**

```bash
rm -f /tmp/dc_single.tif /tmp/dc_tiled.tif /tmp/dc_tiled.tif.ckpt.npy /tmp/dc_tiled.tif.ckpt.n \
      /tmp/dc_single.tif.ckpt.npy /tmp/dc_single.tif.ckpt.n
```

No commit for this task — it's a validation checkpoint. If Step 4 fails, return to Task 2 and re-check the `out_transform`/`row_px_bounds` offset math before proceeding.

---

## Task 4: Update `run_olmoearth_nano_embed_state_array.sbatch` for tiling

**Files:**
- Modify: `code/embedding_generation/slurm/run_olmoearth_nano_embed_state_array.sbatch`

**Interfaces:**
- Consumes: `TILE_TASK_FILE` env var, a file with lines `STATE TILE_IDX NUM_TILES` (produced by Task 6's submit script).
- Produces: for `NUM_TILES == 1`, moves the completed embedding straight to `$FINAL_DIR` (today's behavior). For `NUM_TILES > 1`, leaves the tile output in `$CKPT_DIR` for the merge job (Task 5) to pick up.

- [ ] **Step 1: Replace the whole file**

Replace the entire contents of `code/embedding_generation/slurm/run_olmoearth_nano_embed_state_array.sbatch` with:

```bash
#!/bin/bash
#SBATCH --job-name=oe-nano-embed
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=code/embedding_generation/slurm/logs/oe_nano_embed_%A_%a.out
#SBATCH --error=code/embedding_generation/slurm/logs/oe_nano_embed_%A_%a.err

# Per-tile OlmoEarth Nano embedding job. SLURM_ARRAY_TASK_ID indexes into
# TILE_TASK_FILE, a newline-delimited file with lines of the form:
#   STATE TILE_IDX NUM_TILES
# created by submit_olmoearth_nano_embed_all_states.sh.
#
# Reads the composite directly from $SCRATCH with windowed reads (no local
# staging) — each tile only touches its own row-band, so there's no
# multi-hundred-GB copy to make or leak. Checkpoints every CHECKPOINT_EVERY
# chips so a timed-out tile resumes mid-tile rather than from scratch.
#
# NUM_TILES=1 states (the common case for small/medium states) behave exactly
# as before: embed.py writes directly to the final name, no tile suffix, and
# this script moves it straight to FINAL_DIR. NUM_TILES>1 states leave their
# tile output in CKPT_DIR for run_olmoearth_nano_embed_merge.sbatch to mosaic.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"
VARIANT="${VARIANT:-Nano}"

: "${SCRATCH:?SCRATCH is not set}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$SCRATCH/embeddings-health/olmoearth_nano_embeddings}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
: "${TILE_TASK_FILE:?TILE_TASK_FILE must be set to the task list file path}"

LINE=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "$TILE_TASK_FILE")
STATE=$(echo "$LINE" | awk '{print $1}')
TILE_IDX=$(echo "$LINE" | awk '{print $2}')
NUM_TILES=$(echo "$LINE" | awk '{print $3}')

if [[ -z "$STATE" || -z "$TILE_IDX" || -z "$NUM_TILES" ]]; then
  echo "ERROR: could not parse line $(( SLURM_ARRAY_TASK_ID + 1 )) of $TILE_TASK_FILE: '$LINE'" >&2
  exit 1
fi

COMPOSITE="$COMPOSITE_DIR/s2_annual_${STATE}_${YEAR}_olmoearth.tif"

if [[ ! -s "$COMPOSITE" ]]; then
  echo "ERROR: composite not found or empty: $COMPOSITE" >&2
  exit 1
fi

CKPT_DIR="$SCRATCH/embeddings-health/checkpoints/olmoearth_nano/$STATE"
FINAL_DIR="$FINAL_OUT_DIR/$STATE"

PREEMPTED=0
CHILD_PID=""

# Slurm's --signal=B:TERM@600 delivers this 10 minutes before the hard time
# limit. Forward it to the embed.py child and give it up to 30s to exit
# gracefully before force-killing, leaving the remaining buffer for a clean
# exit — SIGKILL after the grace period can't be caught by any trap.
handle_preempt() {
  PREEMPTED=1
  echo "Job preempted or timed out — checkpoint preserved at: $CKPT_DIR" >&2
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$CHILD_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$CHILD_PID" 2>/dev/null || true
  fi
}

preserve_failure() {
  status=$?
  if [[ "$PREEMPTED" == 1 ]]; then
    exit 0
  fi
  echo "ERROR: task failed with status $status — partial outputs in $CKPT_DIR" >&2
  exit "$status"
}

trap handle_preempt TERM
trap preserve_failure ERR

mkdir -p "$CKPT_DIR" "$FINAL_DIR" "$CACHE_ROOT"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-${SLURMD_NODENAME:-$(hostname -s)}"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if command -v module >/dev/null 2>&1; then
  module load devel
  module load gcc/14.2.0
  module load rust/1.90.0
  module load cuda/12.6.1 || true
fi
export CC="$(command -v gcc)" CXX="$(command -v g++)"

if ! command -v uv >/dev/null 2>&1; then
  UV_INSTALL_DIR="${UV_INSTALL_DIR:-$CACHE_ROOT/uv-bin}"
  mkdir -p "$UV_INSTALL_DIR"
  if [[ -x "$UV_INSTALL_DIR/uv" ]]; then
    export PATH="$UV_INSTALL_DIR:$PATH"
  else
    echo "Installing uv into $UV_INSTALL_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
    export PATH="$UV_INSTALL_DIR:$PATH"
  fi
fi

echo "uv: $(command -v uv)"
uv --version

(cd "$EMBED_DIR" && uv sync --python 3.11 2>&1 | tail -5)

if ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import olmoearth_pretrain" 2>/dev/null; then
  echo "Installing olmoearth_pretrain (--no-deps to skip torch>=2.7 metadata conflict)..."
  VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install --no-deps \
    git+https://github.com/allenai/olmoearth_pretrain
  VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install \
    "einops>=0.7.0" "huggingface_hub" "numpy>=1.26.4" "universal-pathlib>=0.2.5"
  echo "olmoearth_pretrain installed."
else
  echo "olmoearth_pretrain already present — skipping install."
fi

SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"
OUTPUT_BASENAME="olmoearth_${SAFE_VARIANT}_${STATE}_${YEAR}"
RAW_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.tif"

BATCH_SIZE="${BATCH_SIZE:-32}"

TEST_CHIPS_ARGS=""
if [[ -n "${TEST_CHIPS:-}" ]]; then
  TEST_CHIPS_ARGS="--test-chips $TEST_CHIPS"
fi

FORCE_FLAG=""
if [[ -n "${FORCE_RESTART:-}" ]]; then
  FORCE_FLAG="--force"
fi

echo "=== OlmoEarth Nano embedding task ==="
echo "State:      $STATE"
echo "Tile:       $TILE_IDX / $NUM_TILES"
echo "Year:       $YEAR"
echo "Variant:    $VARIANT"
echo "Composite:  $COMPOSITE"
echo "Ckpt dir:   $CKPT_DIR"
echo "Final dir:  $FINAL_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Test chips: ${TEST_CHIPS:-none}"
nvidia-smi || true

cd "$EMBED_DIR"

# Run in background and wait: bash's `wait` builtin returns as soon as a
# trapped signal arrives, whereas a foreground command can block trap
# execution until the child exits on its own.
uv run --python 3.11 python embed.py \
  --model        olmoearth \
  --variant      "$VARIANT" \
  --input        "$COMPOSITE" \
  --output       "$RAW_LOCAL" \
  --tile-index   "$TILE_IDX" \
  --num-tiles    "$NUM_TILES" \
  --year         "$YEAR" \
  --batch-size   "$BATCH_SIZE" \
  --checkpoint-every "${CHECKPOINT_EVERY:-25}" \
  $FORCE_FLAG \
  $TEST_CHIPS_ARGS &
CHILD_PID=$!
wait "$CHILD_PID"

if (( NUM_TILES > 1 )); then
  TILE_OUTPUT="$CKPT_DIR/${OUTPUT_BASENAME}_tile$(printf '%03d' "$TILE_IDX").tif"
  if [[ ! -s "$TILE_OUTPUT" ]]; then
    echo "ERROR: expected tile output missing or empty: $TILE_OUTPUT" >&2
    exit 1
  fi
  trap - ERR TERM
  echo "=== Done: $STATE tile $TILE_IDX/$NUM_TILES ==="
  ls -lh "$TILE_OUTPUT"
else
  if [[ ! -s "$RAW_LOCAL" ]]; then
    echo "ERROR: expected output missing or empty: $RAW_LOCAL" >&2
    exit 1
  fi
  mv -f "$RAW_LOCAL" "$FINAL_DIR/"
  MMAP_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.ckpt.mmap"
  [[ -f "$MMAP_LOCAL" ]] && rm -f "$MMAP_LOCAL" || true
  trap - ERR TERM
  echo "=== Done ==="
  echo "Output: $FINAL_DIR/${OUTPUT_BASENAME}.tif"
fi
```

- [ ] **Step 2: Verify bash syntax**

```bash
bash -n "$HOME/embeddings-health/code/embedding_generation/slurm/run_olmoearth_nano_embed_state_array.sbatch" && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 3: Commit**

```bash
cd "$HOME/embeddings-health"
git add code/embedding_generation/slurm/run_olmoearth_nano_embed_state_array.sbatch
git commit -m "$(cat <<'EOF'
feat: switch oe-nano-embed to per-tile task-file-driven array

Removes the staging step entirely (embed.py now reads the composite
directly from $SCRATCH per tile) and resolves (STATE, TILE_IDX,
NUM_TILES) from a task file line instead of a plain state list, so
each array task can be one state's tile instead of one whole state.
EOF
)"
```

---

## Task 5: Create `run_olmoearth_nano_embed_merge.sbatch`

**Files:**
- Create: `code/embedding_generation/slurm/run_olmoearth_nano_embed_merge.sbatch`

**Interfaces:**
- Consumes: `STATE_LIST` (colon-separated) and `NUM_TILES_LIST` (colon-separated, same order/length) env vars from Task 6's submit script.

- [ ] **Step 1: Create the file**

```bash
#!/bin/bash
#SBATCH --job-name=oe-nano-merge
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=code/embedding_generation/slurm/logs/oe_nano_merge_%A_%a.out
#SBATCH --error=code/embedding_generation/slurm/logs/oe_nano_merge_%A_%a.err

# Merge tile embeddings into the final Nano embedding for one state.
# STATE_LIST and NUM_TILES_LIST (both colon-separated, same order/length)
# are set by submit_olmoearth_nano_embed_all_states.sh; SLURM_ARRAY_TASK_ID
# indexes into both. Only states with NUM_TILES > 1 are included in this
# array — single-tile states are already handled by the tile task itself.
#
# This job is submitted with --dependency=afterany:<tile-array-id> so it
# runs after all tile jobs complete. embed.py --merge-only exits cleanly
# (a SKIP message, not an error) if any tiles are still missing — the
# resubmit chain will retry those tiles and re-run this merge.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"
VARIANT="${VARIANT:-Nano}"

: "${SCRATCH:?SCRATCH is not set}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$SCRATCH/embeddings-health/olmoearth_nano_embeddings}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
: "${STATE_LIST:?STATE_LIST must be set to a colon-separated ordered state list}"
: "${NUM_TILES_LIST:?NUM_TILES_LIST must be set to a colon-separated list matching STATE_LIST}"

IFS=: read -ra STATES <<< "$STATE_LIST"
IFS=: read -ra NUM_TILES_ARR <<< "$NUM_TILES_LIST"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
STATE="${STATES[$TASK_ID]}"
NUM_TILES="${NUM_TILES_ARR[$TASK_ID]}"

CKPT_DIR="$SCRATCH/embeddings-health/checkpoints/olmoearth_nano/$STATE"
FINAL_DIR="$FINAL_OUT_DIR/$STATE"
mkdir -p "$FINAL_DIR" "$CACHE_ROOT"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-${SLURMD_NODENAME:-$(hostname -s)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if command -v module >/dev/null 2>&1; then
  module load devel
  module load gcc/14.2.0
fi
export CC="$(command -v gcc)" CXX="$(command -v g++)"

if ! command -v uv >/dev/null 2>&1; then
  UV_INSTALL_DIR="${UV_INSTALL_DIR:-$CACHE_ROOT/uv-bin}"
  [[ -x "$UV_INSTALL_DIR/uv" ]] && export PATH="$UV_INSTALL_DIR:$PATH" || { echo "ERROR: uv not found." >&2; exit 1; }
fi

SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"
OUTPUT_BASENAME="olmoearth_${SAFE_VARIANT}_${STATE}_${YEAR}"

echo "Node:  $(hostname -s)"
echo "State: $STATE  (merge-only, $NUM_TILES tiles)"
echo ""

cd "$EMBED_DIR"
uv run --python 3.11 python embed.py \
  --model      olmoearth \
  --variant    "$VARIANT" \
  --output     "$CKPT_DIR/${OUTPUT_BASENAME}.tif" \
  --num-tiles  "$NUM_TILES" \
  --year       "$YEAR" \
  --merge-only

MERGED="$CKPT_DIR/${OUTPUT_BASENAME}.tif"
if [[ -s "$MERGED" ]]; then
  mv -f "$MERGED" "$FINAL_DIR/"
  MMAP_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.ckpt.mmap"
  [[ -f "$MMAP_LOCAL" ]] && rm -f "$MMAP_LOCAL" || true
  echo "=== Merge complete: $STATE ==="
  ls -lh "$FINAL_DIR/${OUTPUT_BASENAME}.tif"
else
  echo "=== Merge skipped or failed for $STATE — tiles not yet complete ==="
fi
```

- [ ] **Step 2: Verify bash syntax**

```bash
bash -n "$HOME/embeddings-health/code/embedding_generation/slurm/run_olmoearth_nano_embed_merge.sbatch" && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 3: Commit**

```bash
cd "$HOME/embeddings-health"
git add code/embedding_generation/slurm/run_olmoearth_nano_embed_merge.sbatch
git commit -m "feat: add oe-nano-embed merge job for multi-tile states"
```

---

## Task 6: Update `submit_olmoearth_nano_embed_all_states.sh` for tile planning

**Files:**
- Modify: `code/embedding_generation/slurm/submit_olmoearth_nano_embed_all_states.sh`

**Interfaces:**
- Produces: submits the tile array (`run_olmoearth_nano_embed_state_array.sbatch`) with `TILE_TASK_FILE` set, then the dependent merge array (`run_olmoearth_nano_embed_merge.sbatch`) with `STATE_LIST`/`NUM_TILES_LIST` set, then a resubmit chain — mirroring `submit_olmoearth_composite_parallel.sh`'s existing structure.

- [ ] **Step 1: Replace the whole file**

Replace the entire contents of `code/embedding_generation/slurm/submit_olmoearth_nano_embed_all_states.sh` with:

```bash
#!/bin/bash
# Submit OlmoEarth Nano embedding inference for all states with a complete
# composite, split into per-state tiles sized to a target chip count so
# every tile finishes within the sbatch script's fixed walltime — no more
# picking a bigger walltime tier for huge states (Sherlock's ceiling is
# fixed regardless). Safe to re-run — states/tiles with existing output are
# skipped.
#
# Usage:
#   bash submit_olmoearth_nano_embed_all_states.sh
#   DRY_RUN=1 bash submit_olmoearth_nano_embed_all_states.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
YEAR="${YEAR:-2022}"
VARIANT="${VARIANT:-Nano}"

: "${SCRATCH:?Set SCRATCH before submitting.}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$SCRATCH/embeddings-health/olmoearth_nano_embeddings}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
LOG_DIR="${LOG_DIR:-$SCRATCH/embeddings-health/logs}"
CKPT_ROOT="$SCRATCH/embeddings-health/checkpoints/olmoearth_nano"
TILE_SCRIPT="$SCRIPT_DIR/run_olmoearth_nano_embed_state_array.sbatch"
MERGE_SCRIPT="$SCRIPT_DIR/run_olmoearth_nano_embed_merge.sbatch"

# OlmoEarth chip size (same composites as Base/Clay).
CHIP_SIZE=128

# Target chip count per tile — chosen so a tile's inference time comfortably
# fits the sbatch script's fixed --time (02:00:00). Starting point from the
# design spec (docs/superpowers/specs/2026-07-04-embed-tiling-design.md);
# tune down if real-world tiles still run close to the walltime.
TARGET_CHIPS_PER_TILE="${TARGET_CHIPS_PER_TILE:-150000}"

TASK_FILE="$SCRATCH/embeddings-health/cache/oe_nano_tile_tasks_${YEAR}.txt"

LOADED_GDAL=0
if command -v module >/dev/null 2>&1 && ! command -v gdalinfo >/dev/null 2>&1; then
  module load devel 2>/dev/null || true
  module load physics gdal/3.10.2 2>/dev/null && LOADED_GDAL=1 || true
fi

get_chips() {
  local tif="$1"
  if ! command -v gdalinfo >/dev/null 2>&1; then
    echo "0"; return
  fi
  local size_line width height w_chips h_chips
  size_line=$(gdalinfo "$tif" 2>/dev/null | grep "^Size is" || true)
  if [[ -z "$size_line" ]]; then echo "0"; return; fi
  width=$(echo  "$size_line" | sed 's/Size is //' | cut -d',' -f1 | tr -d ' ')
  height=$(echo "$size_line" | sed 's/Size is //' | cut -d',' -f2 | tr -d ' ')
  w_chips=$(( (width  + CHIP_SIZE - 1) / CHIP_SIZE ))
  h_chips=$(( (height + CHIP_SIZE - 1) / CHIP_SIZE ))
  echo $(( w_chips * h_chips ))
}

# num_tiles is capped at h_chips (tile_row_bounds requires num_tiles <=
# n_row_chips) — recompute h_chips here since get_chips only returns the
# product.
get_h_chips() {
  local tif="$1"
  if ! command -v gdalinfo >/dev/null 2>&1; then
    echo "1"; return
  fi
  local size_line height
  size_line=$(gdalinfo "$tif" 2>/dev/null | grep "^Size is" || true)
  if [[ -z "$size_line" ]]; then echo "1"; return; fi
  height=$(echo "$size_line" | sed 's/Size is //' | cut -d',' -f2 | tr -d ' ')
  echo $(( (height + CHIP_SIZE - 1) / CHIP_SIZE ))
}

SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"

# Discover states with a composite.
STATES=()
while IFS= read -r state; do
  STATES+=("$state")
done < <(
  find "$COMPOSITE_DIR" -maxdepth 1 -type f \
    -name "s2_annual_*_${YEAR}_olmoearth.tif" \
    -exec basename {} \; \
    | sed -E "s/^s2_annual_(.*)_${YEAR}_olmoearth\.tif$/\1/" \
    | sort
)

if (( ${#STATES[@]} == 0 )); then
  echo "ERROR: no OlmoEarth composites found in $COMPOSITE_DIR for year $YEAR" >&2
  exit 1
fi

# ------------------------------------------------------------------
# Build the flat tile task list: one line per remaining (state, tile).
# ------------------------------------------------------------------
MERGE_STATES=()
MERGE_NUM_TILES=()
> "$TASK_FILE"

total_tiles=0
skipped_states=0

for STATE in "${STATES[@]}"; do
  FINAL_TIF="$FINAL_OUT_DIR/$STATE/olmoearth_${SAFE_VARIANT}_${STATE}_${YEAR}.tif"
  if [[ -s "$FINAL_TIF" ]]; then
    (( skipped_states++ )) || true
    continue
  fi

  COMPOSITE="$COMPOSITE_DIR/s2_annual_${STATE}_${YEAR}_olmoearth.tif"
  CHIPS=$(get_chips "$COMPOSITE")
  H_CHIPS=$(get_h_chips "$COMPOSITE")

  if (( CHIPS == 0 )); then
    NUM_TILES=1
  else
    NUM_TILES=$(( (CHIPS + TARGET_CHIPS_PER_TILE - 1) / TARGET_CHIPS_PER_TILE ))
    (( NUM_TILES < 1 )) && NUM_TILES=1
    (( NUM_TILES > H_CHIPS )) && NUM_TILES=$H_CHIPS
  fi

  if (( NUM_TILES > 1 )); then
    MERGE_STATES+=("$STATE")
    MERGE_NUM_TILES+=("$NUM_TILES")
  fi

  CKPT_DIR="$CKPT_ROOT/$STATE"
  OUTPUT_BASENAME="olmoearth_${SAFE_VARIANT}_${STATE}_${YEAR}"

  for (( idx=0; idx<NUM_TILES; idx++ )); do
    if (( NUM_TILES > 1 )); then
      tile_path="$CKPT_DIR/${OUTPUT_BASENAME}_tile$(printf '%03d' "$idx").tif"
      [[ -s "$tile_path" ]] && continue
    fi
    echo "$STATE $idx $NUM_TILES" >> "$TASK_FILE"
    (( total_tiles++ )) || true
  done
done

if (( LOADED_GDAL )); then
  module unload gdal/3.10.2 2>/dev/null || true
fi

echo "Repo:          $REPO_DIR"
echo "Composites:    $COMPOSITE_DIR"
echo "Outputs:       $FINAL_OUT_DIR"
echo "Variant:       $VARIANT"
echo "Year:          $YEAR"
echo "Target chips/tile: $TARGET_CHIPS_PER_TILE"
echo ""
echo "States skipped (embedding exists): $skipped_states / ${#STATES[@]}"
echo "States needing a merge job:        ${#MERGE_STATES[@]}"
echo "Tile tasks to submit:              $total_tiles"
echo ""

if (( total_tiles == 0 && ${#MERGE_STATES[@]} == 0 )); then
  echo "All OlmoEarth Nano embeddings complete. Nothing to submit."
  exit 0
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; not submitting."
  echo ""
  echo "First 10 tasks in $TASK_FILE:"
  head -10 "$TASK_FILE"
  exit 0
fi

mkdir -p "$LOG_DIR" "$FINAL_OUT_DIR"
export REPO_DIR COMPOSITE_DIR FINAL_OUT_DIR CACHE_ROOT YEAR VARIANT

# ------------------------------------------------------------------
# Submit tile arrays in batches of 1000 (Sherlock max_array_tasks=1000).
# ------------------------------------------------------------------
MAX_ARRAY=1000
TILE_JOB_IDS=()
if (( total_tiles > 0 )); then
  batch=0
  offset=0
  while (( offset < total_tiles )); do
    end=$(( offset + MAX_ARRAY - 1 ))
    (( end >= total_tiles )) && end=$(( total_tiles - 1 ))
    count=$(( end - offset + 1 ))

    batch_file="${TASK_FILE%.txt}_batch${batch}.txt"
    sed -n "$((offset + 1)),$((end + 1))p" "$TASK_FILE" > "$batch_file"

    export TILE_TASK_FILE="$batch_file"
    JOB_ID=$(cd "$REPO_DIR" && sbatch \
      --export=ALL \
      --array="0-$(( count - 1 ))%200" \
      --output="$LOG_DIR/oe_nano_embed_%A_%a.out" \
      --error="$LOG_DIR/oe_nano_embed_%A_%a.err" \
      --parsable \
      "$TILE_SCRIPT" | cut -d';' -f1)
    echo "Submitted tile batch $batch: job $JOB_ID  ($count tasks, ≤200 concurrent)"
    TILE_JOB_IDS+=("$JOB_ID")
    (( batch++ )) || true
    (( offset += MAX_ARRAY )) || true
  done
fi
TILE_JOB_ID=$(IFS=:; echo "${TILE_JOB_IDS[*]}")

# ------------------------------------------------------------------
# Submit merge array — one task per multi-tile state, after tile jobs finish.
# ------------------------------------------------------------------
if (( ${#MERGE_STATES[@]} > 0 )); then
  MERGE_STATE_LIST=$(IFS=:; echo "${MERGE_STATES[*]}")
  MERGE_NUM_TILES_LIST=$(IFS=:; echo "${MERGE_NUM_TILES[*]}")
  MERGE_LAST_IDX=$(( ${#MERGE_STATES[@]} - 1 ))

  MERGE_DEP=""
  [[ -n "$TILE_JOB_ID" ]] && MERGE_DEP="--dependency=afterany:${TILE_JOB_ID}"

  export STATE_LIST="$MERGE_STATE_LIST"
  export NUM_TILES_LIST="$MERGE_NUM_TILES_LIST"
  MERGE_JOB_ID=$(cd "$REPO_DIR" && sbatch \
    --export=ALL \
    $MERGE_DEP \
    --array="0-${MERGE_LAST_IDX}" \
    --output="$LOG_DIR/oe_nano_merge_%A_%a.out" \
    --error="$LOG_DIR/oe_nano_merge_%A_%a.err" \
    --parsable \
    "$MERGE_SCRIPT" | cut -d';' -f1)
  echo "Submitted merge array job  $MERGE_JOB_ID  (${#MERGE_STATES[@]} states)"
  [[ -n "$TILE_JOB_ID" ]] && echo "  → depends on tile job $TILE_JOB_ID"
fi

# ------------------------------------------------------------------
# Resubmit chain: after tiles+merge complete, re-run this script to pick up
# any tiles that failed/timed out and need a retry.
# ------------------------------------------------------------------
ALL_DEPS="${TILE_JOB_ID}"
[[ -n "${MERGE_JOB_ID:-}" ]] && ALL_DEPS="${ALL_DEPS}:${MERGE_JOB_ID}"
SELF="$(realpath "${BASH_SOURCE[0]}")"
RESUBMIT_ID=$(sbatch \
  --dependency="afterany:${ALL_DEPS}" \
  --job-name=oe-nano-embed-resubmit \
  --partition=normal \
  --time=00:10:00 \
  --mem=4G \
  --cpus-per-task=1 \
  --output="$LOG_DIR/oe_nano_embed_resubmit_%j.out" \
  --error="$LOG_DIR/oe_nano_embed_resubmit_%j.err" \
  --export=ALL \
  --parsable \
  --wrap="bash '$SELF'" | cut -d';' -f1)
echo "Resubmit job $RESUBMIT_ID scheduled after tiles+merge (cancel with: scancel $RESUBMIT_ID)"
```

- [ ] **Step 2: Verify bash syntax**

```bash
bash -n "$HOME/embeddings-health/code/embedding_generation/slurm/submit_olmoearth_nano_embed_all_states.sh" && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 3: Dry-run against real composites to sanity-check the tile plan**

```bash
DRY_RUN=1 bash "$HOME/embeddings-health/code/embedding_generation/slurm/submit_olmoearth_nano_embed_all_states.sh"
```

Expected: prints a summary (skipped/merge/tile-task counts) and the first 10 lines of the generated task file, with no errors. Spot check: a small state (DC/RI) should get `NUM_TILES=1`; a large stuck state (NV/MT) should get `NUM_TILES > 1`.

- [ ] **Step 4: Commit**

```bash
cd "$HOME/embeddings-health"
git add code/embedding_generation/slurm/submit_olmoearth_nano_embed_all_states.sh
git commit -m "$(cat <<'EOF'
feat: plan per-state tile counts in submit_olmoearth_nano_embed_all_states.sh

Replaces the old walltime-tiering loop (which could never give the
largest states enough time) with a flat (state, tile_index, num_tiles)
task list sized to a target chip count per tile, plus a dependent merge
array — mirroring submit_olmoearth_composite_parallel.sh's proven
tile/merge structure.
EOF
)"
```

---

## Task 7: srun end-to-end validation on a real stuck Nano state

**Files:** none (validation only)

- [ ] **Step 1: Pick a currently-stuck state and check its composite size**

```bash
STATE=NV  # or whichever Nano state is still stuck per squeue/sacct at execution time
ls -lh "$SCRATCH/embeddings-health/olmoearth_composites/s2_annual_${STATE}_2022_olmoearth.tif"
```

- [ ] **Step 2: Dry-run the submit script scoped to just that state's tile count**

```bash
DRY_RUN=1 bash "$HOME/embeddings-health/code/embedding_generation/slurm/submit_olmoearth_nano_embed_all_states.sh" 2>&1 | grep -A2 "^$STATE " || true
grep "^$STATE " "$SCRATCH/embeddings-health/cache/oe_nano_tile_tasks_2022.txt"
```

Expected: several lines `NV <idx> <NUM_TILES>` — confirms this state is being split, and note `NUM_TILES`.

- [ ] **Step 3: Run tile 0 interactively via srun and confirm it finishes well within the walltime**

```bash
TILE_TASK_FILE=/tmp/one_task.txt
echo "$STATE 0 <NUM_TILES from step 2>" > "$TILE_TASK_FILE"
export TILE_TASK_FILE REPO_DIR="$HOME/embeddings-health" YEAR=2022 VARIANT=Nano \
       COMPOSITE_DIR="$SCRATCH/embeddings-health/olmoearth_composites" \
       FINAL_OUT_DIR="$SCRATCH/embeddings-health/olmoearth_nano_embeddings" \
       CACHE_ROOT="$SCRATCH/embeddings-health/cache"
srun -p gpu -G 1 -c 8 --mem=64G --time=02:00:00 \
  bash "$HOME/embeddings-health/code/embedding_generation/slurm/run_olmoearth_nano_embed_state_array.sbatch"
```

Expected: completes with `=== Done: NV tile 0/<N> ===` well before the 2-hour limit (compare wall-clock against the walltime — this is the key regression check for the setup-bound bottleneck from the original diagnosis).

- [ ] **Step 4: Confirm kill-and-resume works mid-tile**

Re-run the same command, but in a second terminal send `scancel <jobid>` (found via `squeue --me`) partway through (after `nvidia-smi` prints but before it finishes), then re-run the identical `srun` command again.

Expected: second run's log shows `Resuming from memmap: N chips already processed` (or the non-memmap `Resuming from checkpoint:` message) with `N > 0`, not starting over from chip 0.

- [ ] **Step 5: Clean up the test tile output so it doesn't interfere with a real submission later**

```bash
CKPT_DIR="$SCRATCH/embeddings-health/checkpoints/olmoearth_nano/$STATE"
rm -f "$CKPT_DIR"/olmoearth_Nano_${STATE}_2022_tile000*
rm -f /tmp/one_task.txt
```

No commit — validation only. If Step 3 shows a tile still running close to the walltime limit, lower `TARGET_CHIPS_PER_TILE` in Task 6 and re-plan before proceeding to Task 8.

---

## Task 8: Replicate the tile/merge pattern for Clay (`clay-embed`)

**Files:**
- Modify: `code/embedding_generation/slurm/run_clay_embed_state_array.sbatch`
- Create: `code/embedding_generation/slurm/run_clay_embed_merge.sbatch`
- Modify: `code/embedding_generation/slurm/submit_clay_embed_all_states.sh`

Same pattern as Tasks 4–6, applied to Clay. Differences from Nano: chip size 256 (not 128), no `olmoearth_pretrain` install step, output basename `clay_v1.5_<STATE>_<YEAR>`, default `--batch-size 8`, `--model clay` (no `--variant`), starting `TARGET_CHIPS_PER_TILE=20000` (placeholder — Clay's real throughput hasn't been measured yet; validate and adjust in Step 7 below before this pipeline's real batch submission).

- [ ] **Step 1: Replace the whole file**

Replace the entire contents of `code/embedding_generation/slurm/run_clay_embed_state_array.sbatch` with:

```bash
#!/bin/bash
#SBATCH --job-name=clay-embed
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=code/embedding_generation/slurm/logs/clay_embed_%A_%a.out
#SBATCH --error=code/embedding_generation/slurm/logs/clay_embed_%A_%a.err

# Per-tile Clay v1.5 embedding job. SLURM_ARRAY_TASK_ID indexes into
# TILE_TASK_FILE, a newline-delimited file with lines of the form:
#   STATE TILE_IDX NUM_TILES
# created by submit_clay_embed_all_states.sh.
#
# Reads the composite directly from $SCRATCH with windowed reads (no local
# staging) — each tile only touches its own row-band, so there's no
# multi-hundred-GB copy to make or leak. Checkpoints every CHECKPOINT_EVERY
# chips so a timed-out tile resumes mid-tile rather than from scratch.
#
# NUM_TILES=1 states (the common case for small/medium states) behave exactly
# as before: embed.py writes directly to the final name, no tile suffix, and
# this script moves it straight to FINAL_DIR. NUM_TILES>1 states leave their
# tile output in CKPT_DIR for run_clay_embed_merge.sbatch to mosaic.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"

: "${SCRATCH:?SCRATCH is not set}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$SCRATCH/embeddings-health/clay_embeddings}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
: "${TILE_TASK_FILE:?TILE_TASK_FILE must be set to the task list file path}"

LINE=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "$TILE_TASK_FILE")
STATE=$(echo "$LINE" | awk '{print $1}')
TILE_IDX=$(echo "$LINE" | awk '{print $2}')
NUM_TILES=$(echo "$LINE" | awk '{print $3}')

if [[ -z "$STATE" || -z "$TILE_IDX" || -z "$NUM_TILES" ]]; then
  echo "ERROR: could not parse line $(( SLURM_ARRAY_TASK_ID + 1 )) of $TILE_TASK_FILE: '$LINE'" >&2
  exit 1
fi

# Clay reuses the OlmoEarth 12-band composites; 10 bands are selected in embed.py.
COMPOSITE="$COMPOSITE_DIR/s2_annual_${STATE}_${YEAR}_olmoearth.tif"

if [[ ! -s "$COMPOSITE" ]]; then
  echo "ERROR: composite not found or empty: $COMPOSITE" >&2
  exit 1
fi

CKPT_DIR="$SCRATCH/embeddings-health/checkpoints/clay/$STATE"
FINAL_DIR="$FINAL_OUT_DIR/$STATE"

PREEMPTED=0
CHILD_PID=""

handle_preempt() {
  PREEMPTED=1
  echo "Job preempted or timed out — checkpoint preserved at: $CKPT_DIR" >&2
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$CHILD_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$CHILD_PID" 2>/dev/null || true
  fi
}

preserve_failure() {
  status=$?
  if [[ "$PREEMPTED" == 1 ]]; then
    exit 0
  fi
  echo "ERROR: task failed with status $status — partial outputs in $CKPT_DIR" >&2
  exit "$status"
}

trap handle_preempt TERM
trap preserve_failure ERR

mkdir -p "$CKPT_DIR" "$FINAL_DIR" "$CACHE_ROOT"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-${SLURMD_NODENAME:-$(hostname -s)}"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if command -v module >/dev/null 2>&1; then
  module load devel
  module load gcc/14.2.0
  module load rust/1.90.0
  module load cuda/12.6.1 || true
fi
export CC="$(command -v gcc)" CXX="$(command -v g++)"

if ! command -v uv >/dev/null 2>&1; then
  UV_INSTALL_DIR="${UV_INSTALL_DIR:-$CACHE_ROOT/uv-bin}"
  mkdir -p "$UV_INSTALL_DIR"
  if [[ -x "$UV_INSTALL_DIR/uv" ]]; then
    export PATH="$UV_INSTALL_DIR:$PATH"
  else
    echo "Installing uv into $UV_INSTALL_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
    export PATH="$UV_INSTALL_DIR:$PATH"
  fi
fi

echo "uv: $(command -v uv)"
uv --version

# Clay needs no special extra installs: clay_encoder.py is bundled in the repo
# and only requires torch + einops, both already in pyproject.toml. The 5 GB
# Clay checkpoint is downloaded from HuggingFace on first run and cached in
# $HF_HOME.
(cd "$EMBED_DIR" && uv sync --python 3.11 2>&1 | tail -5)

OUTPUT_BASENAME="clay_v1.5_${STATE}_${YEAR}"
RAW_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.tif"

BATCH_SIZE="${BATCH_SIZE:-8}"

TEST_CHIPS_ARGS=""
if [[ -n "${TEST_CHIPS:-}" ]]; then
  TEST_CHIPS_ARGS="--test-chips $TEST_CHIPS"
fi

FORCE_FLAG=""
if [[ -n "${FORCE_RESTART:-}" ]]; then
  FORCE_FLAG="--force"
fi

echo "=== Clay v1.5 embedding task ==="
echo "State:      $STATE"
echo "Tile:       $TILE_IDX / $NUM_TILES"
echo "Year:       $YEAR"
echo "Composite:  $COMPOSITE"
echo "Ckpt dir:   $CKPT_DIR"
echo "Final dir:  $FINAL_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Test chips: ${TEST_CHIPS:-none}"
nvidia-smi || true

cd "$EMBED_DIR"

uv run --python 3.11 python embed.py \
  --model        clay \
  --input        "$COMPOSITE" \
  --output       "$RAW_LOCAL" \
  --tile-index   "$TILE_IDX" \
  --num-tiles    "$NUM_TILES" \
  --year         "$YEAR" \
  --batch-size   "$BATCH_SIZE" \
  --checkpoint-every "${CHECKPOINT_EVERY:-25}" \
  $FORCE_FLAG \
  $TEST_CHIPS_ARGS &
CHILD_PID=$!
wait "$CHILD_PID"

if (( NUM_TILES > 1 )); then
  TILE_OUTPUT="$CKPT_DIR/${OUTPUT_BASENAME}_tile$(printf '%03d' "$TILE_IDX").tif"
  if [[ ! -s "$TILE_OUTPUT" ]]; then
    echo "ERROR: expected tile output missing or empty: $TILE_OUTPUT" >&2
    exit 1
  fi
  trap - ERR TERM
  echo "=== Done: $STATE tile $TILE_IDX/$NUM_TILES ==="
  ls -lh "$TILE_OUTPUT"
else
  if [[ ! -s "$RAW_LOCAL" ]]; then
    echo "ERROR: expected output missing or empty: $RAW_LOCAL" >&2
    exit 1
  fi
  mv -f "$RAW_LOCAL" "$FINAL_DIR/"
  MMAP_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.ckpt.mmap"
  [[ -f "$MMAP_LOCAL" ]] && rm -f "$MMAP_LOCAL" || true
  trap - ERR TERM
  echo "=== Done ==="
  echo "Output: $FINAL_DIR/${OUTPUT_BASENAME}.tif"
fi
```

- [ ] **Step 2: Verify bash syntax**

```bash
bash -n "$HOME/embeddings-health/code/embedding_generation/slurm/run_clay_embed_state_array.sbatch" && echo "syntax OK"
```

- [ ] **Step 3: Create `run_clay_embed_merge.sbatch`**

```bash
#!/bin/bash
#SBATCH --job-name=clay-merge
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=code/embedding_generation/slurm/logs/clay_merge_%A_%a.out
#SBATCH --error=code/embedding_generation/slurm/logs/clay_merge_%A_%a.err

# Merge tile embeddings into the final Clay embedding for one state.
# STATE_LIST and NUM_TILES_LIST (both colon-separated, same order/length)
# are set by submit_clay_embed_all_states.sh; SLURM_ARRAY_TASK_ID indexes
# into both. Only states with NUM_TILES > 1 are included in this array —
# single-tile states are already handled by the tile task itself.
#
# This job is submitted with --dependency=afterany:<tile-array-id> so it
# runs after all tile jobs complete. embed.py --merge-only exits cleanly
# (a SKIP message, not an error) if any tiles are still missing — the
# resubmit chain will retry those tiles and re-run this merge.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"

: "${SCRATCH:?SCRATCH is not set}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$SCRATCH/embeddings-health/clay_embeddings}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
: "${STATE_LIST:?STATE_LIST must be set to a colon-separated ordered state list}"
: "${NUM_TILES_LIST:?NUM_TILES_LIST must be set to a colon-separated list matching STATE_LIST}"

IFS=: read -ra STATES <<< "$STATE_LIST"
IFS=: read -ra NUM_TILES_ARR <<< "$NUM_TILES_LIST"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
STATE="${STATES[$TASK_ID]}"
NUM_TILES="${NUM_TILES_ARR[$TASK_ID]}"

CKPT_DIR="$SCRATCH/embeddings-health/checkpoints/clay/$STATE"
FINAL_DIR="$FINAL_OUT_DIR/$STATE"
mkdir -p "$FINAL_DIR" "$CACHE_ROOT"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-${SLURMD_NODENAME:-$(hostname -s)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if command -v module >/dev/null 2>&1; then
  module load devel
  module load gcc/14.2.0
fi
export CC="$(command -v gcc)" CXX="$(command -v g++)"

if ! command -v uv >/dev/null 2>&1; then
  UV_INSTALL_DIR="${UV_INSTALL_DIR:-$CACHE_ROOT/uv-bin}"
  [[ -x "$UV_INSTALL_DIR/uv" ]] && export PATH="$UV_INSTALL_DIR:$PATH" || { echo "ERROR: uv not found." >&2; exit 1; }
fi

OUTPUT_BASENAME="clay_v1.5_${STATE}_${YEAR}"

echo "Node:  $(hostname -s)"
echo "State: $STATE  (merge-only, $NUM_TILES tiles)"
echo ""

cd "$EMBED_DIR"
uv run --python 3.11 python embed.py \
  --model      clay \
  --output     "$CKPT_DIR/${OUTPUT_BASENAME}.tif" \
  --num-tiles  "$NUM_TILES" \
  --year       "$YEAR" \
  --merge-only

MERGED="$CKPT_DIR/${OUTPUT_BASENAME}.tif"
if [[ -s "$MERGED" ]]; then
  mv -f "$MERGED" "$FINAL_DIR/"
  MMAP_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.ckpt.mmap"
  [[ -f "$MMAP_LOCAL" ]] && rm -f "$MMAP_LOCAL" || true
  echo "=== Merge complete: $STATE ==="
  ls -lh "$FINAL_DIR/${OUTPUT_BASENAME}.tif"
else
  echo "=== Merge skipped or failed for $STATE — tiles not yet complete ==="
fi
```

- [ ] **Step 4: Verify bash syntax**

```bash
bash -n "$HOME/embeddings-health/code/embedding_generation/slurm/run_clay_embed_merge.sbatch" && echo "syntax OK"
```

- [ ] **Step 5: Replace the whole file `submit_clay_embed_all_states.sh`**

Replace the entire contents of `code/embedding_generation/slurm/submit_clay_embed_all_states.sh` with:

```bash
#!/bin/bash
# Submit Clay v1.5 embedding inference for all states with a complete
# composite, split into per-state tiles sized to a target chip count so
# every tile finishes within the sbatch script's fixed walltime. Safe to
# re-run — states/tiles with existing output are skipped.
#
# Usage:
#   bash submit_clay_embed_all_states.sh
#   DRY_RUN=1 bash submit_clay_embed_all_states.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
YEAR="${YEAR:-2022}"

: "${SCRATCH:?Set SCRATCH before submitting.}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$SCRATCH/embeddings-health/clay_embeddings}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
LOG_DIR="${LOG_DIR:-$SCRATCH/embeddings-health/logs}"
CKPT_ROOT="$SCRATCH/embeddings-health/checkpoints/clay"
TILE_SCRIPT="$SCRIPT_DIR/run_clay_embed_state_array.sbatch"
MERGE_SCRIPT="$SCRIPT_DIR/run_clay_embed_merge.sbatch"

# Clay chip size: 256×256 px.
CHIP_SIZE=256

# Target chip count per tile — placeholder pending real measurement (Clay's
# throughput hasn't been benchmarked the way Base/Nano have; validate via
# srun per Task 8 Step 7 of the embed-tiling plan and adjust if a tile runs
# close to the 4-hour walltime).
TARGET_CHIPS_PER_TILE="${TARGET_CHIPS_PER_TILE:-20000}"

TASK_FILE="$SCRATCH/embeddings-health/cache/clay_tile_tasks_${YEAR}.txt"

LOADED_GDAL=0
if command -v module >/dev/null 2>&1 && ! command -v gdalinfo >/dev/null 2>&1; then
  module load devel 2>/dev/null || true
  module load physics gdal/3.10.2 2>/dev/null && LOADED_GDAL=1 || true
fi

get_chips() {
  local tif="$1"
  if ! command -v gdalinfo >/dev/null 2>&1; then
    echo "0"; return
  fi
  local size_line width height w_chips h_chips
  size_line=$(gdalinfo "$tif" 2>/dev/null | grep "^Size is" || true)
  if [[ -z "$size_line" ]]; then echo "0"; return; fi
  width=$(echo  "$size_line" | sed 's/Size is //' | cut -d',' -f1 | tr -d ' ')
  height=$(echo "$size_line" | sed 's/Size is //' | cut -d',' -f2 | tr -d ' ')
  w_chips=$(( (width  + CHIP_SIZE - 1) / CHIP_SIZE ))
  h_chips=$(( (height + CHIP_SIZE - 1) / CHIP_SIZE ))
  echo $(( w_chips * h_chips ))
}

get_h_chips() {
  local tif="$1"
  if ! command -v gdalinfo >/dev/null 2>&1; then
    echo "1"; return
  fi
  local size_line height
  size_line=$(gdalinfo "$tif" 2>/dev/null | grep "^Size is" || true)
  if [[ -z "$size_line" ]]; then echo "1"; return; fi
  height=$(echo "$size_line" | sed 's/Size is //' | cut -d',' -f2 | tr -d ' ')
  echo $(( (height + CHIP_SIZE - 1) / CHIP_SIZE ))
}

STATES=()
while IFS= read -r state; do
  STATES+=("$state")
done < <(
  find "$COMPOSITE_DIR" -maxdepth 1 -type f \
    -name "s2_annual_*_${YEAR}_olmoearth.tif" \
    -exec basename {} \; \
    | sed -E "s/^s2_annual_(.*)_${YEAR}_olmoearth\.tif$/\1/" \
    | sort
)

if (( ${#STATES[@]} == 0 )); then
  echo "ERROR: no OlmoEarth composites found in $COMPOSITE_DIR for year $YEAR" >&2
  exit 1
fi

MERGE_STATES=()
MERGE_NUM_TILES=()
> "$TASK_FILE"

total_tiles=0
skipped_states=0

for STATE in "${STATES[@]}"; do
  FINAL_TIF="$FINAL_OUT_DIR/$STATE/clay_v1.5_${STATE}_${YEAR}.tif"
  if [[ -s "$FINAL_TIF" ]]; then
    (( skipped_states++ )) || true
    continue
  fi

  COMPOSITE="$COMPOSITE_DIR/s2_annual_${STATE}_${YEAR}_olmoearth.tif"
  CHIPS=$(get_chips "$COMPOSITE")
  H_CHIPS=$(get_h_chips "$COMPOSITE")

  if (( CHIPS == 0 )); then
    NUM_TILES=1
  else
    NUM_TILES=$(( (CHIPS + TARGET_CHIPS_PER_TILE - 1) / TARGET_CHIPS_PER_TILE ))
    (( NUM_TILES < 1 )) && NUM_TILES=1
    (( NUM_TILES > H_CHIPS )) && NUM_TILES=$H_CHIPS
  fi

  if (( NUM_TILES > 1 )); then
    MERGE_STATES+=("$STATE")
    MERGE_NUM_TILES+=("$NUM_TILES")
  fi

  CKPT_DIR="$CKPT_ROOT/$STATE"
  OUTPUT_BASENAME="clay_v1.5_${STATE}_${YEAR}"

  for (( idx=0; idx<NUM_TILES; idx++ )); do
    if (( NUM_TILES > 1 )); then
      tile_path="$CKPT_DIR/${OUTPUT_BASENAME}_tile$(printf '%03d' "$idx").tif"
      [[ -s "$tile_path" ]] && continue
    fi
    echo "$STATE $idx $NUM_TILES" >> "$TASK_FILE"
    (( total_tiles++ )) || true
  done
done

if (( LOADED_GDAL )); then
  module unload gdal/3.10.2 2>/dev/null || true
fi

echo "Repo:          $REPO_DIR"
echo "Composites:    $COMPOSITE_DIR"
echo "Outputs:       $FINAL_OUT_DIR"
echo "Year:          $YEAR"
echo "Target chips/tile: $TARGET_CHIPS_PER_TILE"
echo ""
echo "States skipped (embedding exists): $skipped_states / ${#STATES[@]}"
echo "States needing a merge job:        ${#MERGE_STATES[@]}"
echo "Tile tasks to submit:              $total_tiles"
echo ""

if (( total_tiles == 0 && ${#MERGE_STATES[@]} == 0 )); then
  echo "All Clay v1.5 embeddings complete. Nothing to submit."
  exit 0
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; not submitting."
  echo ""
  echo "First 10 tasks in $TASK_FILE:"
  head -10 "$TASK_FILE"
  exit 0
fi

mkdir -p "$LOG_DIR" "$FINAL_OUT_DIR"
export REPO_DIR COMPOSITE_DIR FINAL_OUT_DIR CACHE_ROOT YEAR

MAX_ARRAY=1000
TILE_JOB_IDS=()
if (( total_tiles > 0 )); then
  batch=0
  offset=0
  while (( offset < total_tiles )); do
    end=$(( offset + MAX_ARRAY - 1 ))
    (( end >= total_tiles )) && end=$(( total_tiles - 1 ))
    count=$(( end - offset + 1 ))

    batch_file="${TASK_FILE%.txt}_batch${batch}.txt"
    sed -n "$((offset + 1)),$((end + 1))p" "$TASK_FILE" > "$batch_file"

    export TILE_TASK_FILE="$batch_file"
    JOB_ID=$(cd "$REPO_DIR" && sbatch \
      --export=ALL \
      --array="0-$(( count - 1 ))%200" \
      --output="$LOG_DIR/clay_embed_%A_%a.out" \
      --error="$LOG_DIR/clay_embed_%A_%a.err" \
      --parsable \
      "$TILE_SCRIPT" | cut -d';' -f1)
    echo "Submitted tile batch $batch: job $JOB_ID  ($count tasks, ≤200 concurrent)"
    TILE_JOB_IDS+=("$JOB_ID")
    (( batch++ )) || true
    (( offset += MAX_ARRAY )) || true
  done
fi
TILE_JOB_ID=$(IFS=:; echo "${TILE_JOB_IDS[*]}")

if (( ${#MERGE_STATES[@]} > 0 )); then
  MERGE_STATE_LIST=$(IFS=:; echo "${MERGE_STATES[*]}")
  MERGE_NUM_TILES_LIST=$(IFS=:; echo "${MERGE_NUM_TILES[*]}")
  MERGE_LAST_IDX=$(( ${#MERGE_STATES[@]} - 1 ))

  MERGE_DEP=""
  [[ -n "$TILE_JOB_ID" ]] && MERGE_DEP="--dependency=afterany:${TILE_JOB_ID}"

  export STATE_LIST="$MERGE_STATE_LIST"
  export NUM_TILES_LIST="$MERGE_NUM_TILES_LIST"
  MERGE_JOB_ID=$(cd "$REPO_DIR" && sbatch \
    --export=ALL \
    $MERGE_DEP \
    --array="0-${MERGE_LAST_IDX}" \
    --output="$LOG_DIR/clay_merge_%A_%a.out" \
    --error="$LOG_DIR/clay_merge_%A_%a.err" \
    --parsable \
    "$MERGE_SCRIPT" | cut -d';' -f1)
  echo "Submitted merge array job  $MERGE_JOB_ID  (${#MERGE_STATES[@]} states)"
  [[ -n "$TILE_JOB_ID" ]] && echo "  → depends on tile job $TILE_JOB_ID"
fi

ALL_DEPS="${TILE_JOB_ID}"
[[ -n "${MERGE_JOB_ID:-}" ]] && ALL_DEPS="${ALL_DEPS}:${MERGE_JOB_ID}"
SELF="$(realpath "${BASH_SOURCE[0]}")"
RESUBMIT_ID=$(sbatch \
  --dependency="afterany:${ALL_DEPS}" \
  --job-name=clay-embed-resubmit \
  --partition=normal \
  --time=00:10:00 \
  --mem=4G \
  --cpus-per-task=1 \
  --output="$LOG_DIR/clay_embed_resubmit_%j.out" \
  --error="$LOG_DIR/clay_embed_resubmit_%j.err" \
  --export=ALL \
  --parsable \
  --wrap="bash '$SELF'" | cut -d';' -f1)
echo "Resubmit job $RESUBMIT_ID scheduled after tiles+merge (cancel with: scancel $RESUBMIT_ID)"
```

- [ ] **Step 6: Verify bash syntax and dry-run**

```bash
bash -n "$HOME/embeddings-health/code/embedding_generation/slurm/submit_clay_embed_all_states.sh" && echo "syntax OK"
DRY_RUN=1 bash "$HOME/embeddings-health/code/embedding_generation/slurm/submit_clay_embed_all_states.sh"
```

Expected: `syntax OK`, then a tile-plan summary with no errors.

- [ ] **Step 7: srun end-to-end validation on a real stuck Clay state**

Pick a currently-stuck Clay state and check its composite size:

```bash
STATE=NV  # or whichever Clay state is still stuck per squeue/sacct at execution time
ls -lh "$SCRATCH/embeddings-health/olmoearth_composites/s2_annual_${STATE}_2022_olmoearth.tif"
```

Dry-run the submit script scoped to just that state's tile count:

```bash
DRY_RUN=1 bash "$HOME/embeddings-health/code/embedding_generation/slurm/submit_clay_embed_all_states.sh" 2>&1 | grep -A2 "^$STATE " || true
grep "^$STATE " "$SCRATCH/embeddings-health/cache/clay_tile_tasks_2022.txt"
```

Expected: several lines `$STATE <idx> <NUM_TILES>` — note `NUM_TILES`.

Run tile 0 interactively via srun and confirm it finishes well within the walltime — since `TARGET_CHIPS_PER_TILE=20000` is an unvalidated placeholder for Clay, this is the key check:

```bash
TILE_TASK_FILE=/tmp/one_task.txt
echo "$STATE 0 <NUM_TILES from above>" > "$TILE_TASK_FILE"
export TILE_TASK_FILE REPO_DIR="$HOME/embeddings-health" YEAR=2022 \
       COMPOSITE_DIR="$SCRATCH/embeddings-health/olmoearth_composites" \
       FINAL_OUT_DIR="$SCRATCH/embeddings-health/clay_embeddings" \
       CACHE_ROOT="$SCRATCH/embeddings-health/cache"
srun -p gpu -G 1 -c 8 --mem=128G --time=04:00:00 \
  bash "$HOME/embeddings-health/code/embedding_generation/slurm/run_clay_embed_state_array.sbatch"
```

Expected: completes with `=== Done: $STATE tile 0/<N> ===` well before the 4-hour limit. If it runs close to the limit, lower `TARGET_CHIPS_PER_TILE` in `submit_clay_embed_all_states.sh` and re-dry-run before proceeding.

Confirm kill-and-resume works mid-tile: re-run the same `srun` command, but in a second terminal send `scancel <jobid>` (found via `squeue --me`) partway through, then re-run the identical `srun` command again. Expected: the second run's log shows `Resuming from memmap: N chips already processed` (or `Resuming from checkpoint:`) with `N > 0`.

Clean up the test tile output:

```bash
CKPT_DIR="$SCRATCH/embeddings-health/checkpoints/clay/$STATE"
rm -f "$CKPT_DIR"/clay_v1.5_${STATE}_2022_tile000*
rm -f /tmp/one_task.txt
```

- [ ] **Step 8: Commit**

```bash
cd "$HOME/embeddings-health"
git add code/embedding_generation/slurm/run_clay_embed_state_array.sbatch \
        code/embedding_generation/slurm/run_clay_embed_merge.sbatch \
        code/embedding_generation/slurm/submit_clay_embed_all_states.sh
git commit -m "feat: apply tile/merge pattern to clay-embed"
```

---

## Task 9: Replicate the tile/merge pattern for Base (`oe-embed`)

**Files:**
- Modify: `code/embedding_generation/slurm/run_olmoearth_embed_state_array.sbatch`
- Create: `code/embedding_generation/slurm/run_olmoearth_embed_merge.sbatch`
- Modify: `code/embedding_generation/slurm/submit_olmoearth_embed_all_states.sh`

Same pattern as Tasks 4–6 and Task 8, applied to Base. Differences from Nano: `VARIANT` defaults to `v1_1-Base` (not `Nano`); `FINAL_OUT_DIR` default `.../olmoearth_embeddings`; `CKPT_DIR=".../checkpoints/olmoearth/$STATE"`; `--mem=128G`/`--time=04:00:00`; job names `oe-embed`/`oe-merge`/`oe-embed-resubmit`; log prefixes `oe_embed`/`oe_merge`; starting `TARGET_CHIPS_PER_TILE=7000` (from the design spec's measured ~0.75s/chip rate on NY — real data, unlike Clay's placeholder, but still validate per Step 7 below since it came from one state's observation).

- [ ] **Step 1: Replace the whole file**

Replace the entire contents of `code/embedding_generation/slurm/run_olmoearth_embed_state_array.sbatch` with:

```bash
#!/bin/bash
#SBATCH --job-name=oe-embed
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=code/embedding_generation/slurm/logs/oe_embed_%A_%a.out
#SBATCH --error=code/embedding_generation/slurm/logs/oe_embed_%A_%a.err

# Per-tile OlmoEarth Base embedding job. SLURM_ARRAY_TASK_ID indexes into
# TILE_TASK_FILE, a newline-delimited file with lines of the form:
#   STATE TILE_IDX NUM_TILES
# created by submit_olmoearth_embed_all_states.sh.
#
# Reads the composite directly from $SCRATCH with windowed reads (no local
# staging) — each tile only touches its own row-band, so there's no
# multi-hundred-GB copy to make or leak. Checkpoints every CHECKPOINT_EVERY
# chips so a timed-out tile resumes mid-tile rather than from scratch. Base's
# per-chip cost (~0.75s/chip measured on NY) is the main reason tiling is
# needed here at all — large states need tens of hours of total inference,
# far more than any single job's walltime, so bounding chip count per task
# via tiling (rather than trying to fit a whole state in one job) is what
# makes forward progress possible.
#
# NUM_TILES=1 states (the common case for small/medium states) behave exactly
# as before: embed.py writes directly to the final name, no tile suffix, and
# this script moves it straight to FINAL_DIR. NUM_TILES>1 states leave their
# tile output in CKPT_DIR for run_olmoearth_embed_merge.sbatch to mosaic.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"
VARIANT="${VARIANT:-v1_1-Base}"

: "${SCRATCH:?SCRATCH is not set}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$SCRATCH/embeddings-health/olmoearth_embeddings}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
: "${TILE_TASK_FILE:?TILE_TASK_FILE must be set to the task list file path}"

LINE=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "$TILE_TASK_FILE")
STATE=$(echo "$LINE" | awk '{print $1}')
TILE_IDX=$(echo "$LINE" | awk '{print $2}')
NUM_TILES=$(echo "$LINE" | awk '{print $3}')

if [[ -z "$STATE" || -z "$TILE_IDX" || -z "$NUM_TILES" ]]; then
  echo "ERROR: could not parse line $(( SLURM_ARRAY_TASK_ID + 1 )) of $TILE_TASK_FILE: '$LINE'" >&2
  exit 1
fi

COMPOSITE="$COMPOSITE_DIR/s2_annual_${STATE}_${YEAR}_olmoearth.tif"

if [[ ! -s "$COMPOSITE" ]]; then
  echo "ERROR: composite not found or empty: $COMPOSITE" >&2
  exit 1
fi

CKPT_DIR="$SCRATCH/embeddings-health/checkpoints/olmoearth/$STATE"
FINAL_DIR="$FINAL_OUT_DIR/$STATE"

PREEMPTED=0
CHILD_PID=""

handle_preempt() {
  PREEMPTED=1
  echo "Job preempted or timed out — checkpoint preserved at: $CKPT_DIR" >&2
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$CHILD_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$CHILD_PID" 2>/dev/null || true
  fi
}

preserve_failure() {
  status=$?
  if [[ "$PREEMPTED" == 1 ]]; then
    exit 0
  fi
  echo "ERROR: task failed with status $status — partial outputs in $CKPT_DIR" >&2
  exit "$status"
}

trap handle_preempt TERM
trap preserve_failure ERR

mkdir -p "$CKPT_DIR" "$FINAL_DIR" "$CACHE_ROOT"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-${SLURMD_NODENAME:-$(hostname -s)}"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if command -v module >/dev/null 2>&1; then
  module load devel
  module load gcc/14.2.0
  module load rust/1.90.0
  module load cuda/12.6.1 || true
fi
export CC="$(command -v gcc)" CXX="$(command -v g++)"

if ! command -v uv >/dev/null 2>&1; then
  UV_INSTALL_DIR="${UV_INSTALL_DIR:-$CACHE_ROOT/uv-bin}"
  mkdir -p "$UV_INSTALL_DIR"
  if [[ -x "$UV_INSTALL_DIR/uv" ]]; then
    export PATH="$UV_INSTALL_DIR:$PATH"
  else
    echo "Installing uv into $UV_INSTALL_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
    export PATH="$UV_INSTALL_DIR:$PATH"
  fi
fi

echo "uv: $(command -v uv)"
uv --version

(cd "$EMBED_DIR" && uv sync --python 3.11 2>&1 | tail -5)

# olmoearth_pretrain is excluded from pyproject.toml because its metadata pins
# torch>=2.7, which conflicts with Sherlock's manylinux baseline. --no-deps
# bypasses that metadata check; the library works fine with torch<2.7 at
# runtime.
if ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import olmoearth_pretrain" 2>/dev/null; then
  echo "Installing olmoearth_pretrain (--no-deps to skip torch>=2.7 metadata conflict)..."
  VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install --no-deps \
    git+https://github.com/allenai/olmoearth_pretrain
  VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install \
    "einops>=0.7.0" "huggingface_hub" "numpy>=1.26.4" "universal-pathlib>=0.2.5"
  echo "olmoearth_pretrain installed."
else
  echo "olmoearth_pretrain already present — skipping install."
fi

SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"
OUTPUT_BASENAME="olmoearth_${SAFE_VARIANT}_${STATE}_${YEAR}"
RAW_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.tif"

BATCH_SIZE="${BATCH_SIZE:-32}"

TEST_CHIPS_ARGS=""
if [[ -n "${TEST_CHIPS:-}" ]]; then
  TEST_CHIPS_ARGS="--test-chips $TEST_CHIPS"
fi

FORCE_FLAG=""
if [[ -n "${FORCE_RESTART:-}" ]]; then
  FORCE_FLAG="--force"
fi

echo "=== OlmoEarth Base embedding task ==="
echo "State:      $STATE"
echo "Tile:       $TILE_IDX / $NUM_TILES"
echo "Year:       $YEAR"
echo "Variant:    $VARIANT"
echo "Composite:  $COMPOSITE"
echo "Ckpt dir:   $CKPT_DIR"
echo "Final dir:  $FINAL_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Test chips: ${TEST_CHIPS:-none}"
nvidia-smi || true

cd "$EMBED_DIR"

uv run --python 3.11 python embed.py \
  --model        olmoearth \
  --variant      "$VARIANT" \
  --input        "$COMPOSITE" \
  --output       "$RAW_LOCAL" \
  --tile-index   "$TILE_IDX" \
  --num-tiles    "$NUM_TILES" \
  --year         "$YEAR" \
  --batch-size   "$BATCH_SIZE" \
  --checkpoint-every "${CHECKPOINT_EVERY:-25}" \
  $FORCE_FLAG \
  $TEST_CHIPS_ARGS &
CHILD_PID=$!
wait "$CHILD_PID"

if (( NUM_TILES > 1 )); then
  TILE_OUTPUT="$CKPT_DIR/${OUTPUT_BASENAME}_tile$(printf '%03d' "$TILE_IDX").tif"
  if [[ ! -s "$TILE_OUTPUT" ]]; then
    echo "ERROR: expected tile output missing or empty: $TILE_OUTPUT" >&2
    exit 1
  fi
  trap - ERR TERM
  echo "=== Done: $STATE tile $TILE_IDX/$NUM_TILES ==="
  ls -lh "$TILE_OUTPUT"
else
  if [[ ! -s "$RAW_LOCAL" ]]; then
    echo "ERROR: expected output missing or empty: $RAW_LOCAL" >&2
    exit 1
  fi
  mv -f "$RAW_LOCAL" "$FINAL_DIR/"
  MMAP_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.ckpt.mmap"
  [[ -f "$MMAP_LOCAL" ]] && rm -f "$MMAP_LOCAL" || true
  trap - ERR TERM
  echo "=== Done ==="
  echo "Output: $FINAL_DIR/${OUTPUT_BASENAME}.tif"
fi
```

- [ ] **Step 2: Verify bash syntax**

```bash
bash -n "$HOME/embeddings-health/code/embedding_generation/slurm/run_olmoearth_embed_state_array.sbatch" && echo "syntax OK"
```

- [ ] **Step 3: Create `run_olmoearth_embed_merge.sbatch`**

```bash
#!/bin/bash
#SBATCH --job-name=oe-merge
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=code/embedding_generation/slurm/logs/oe_merge_%A_%a.out
#SBATCH --error=code/embedding_generation/slurm/logs/oe_merge_%A_%a.err

# Merge tile embeddings into the final OlmoEarth Base embedding for one
# state. STATE_LIST and NUM_TILES_LIST (both colon-separated, same
# order/length) are set by submit_olmoearth_embed_all_states.sh;
# SLURM_ARRAY_TASK_ID indexes into both. Only states with NUM_TILES > 1 are
# included in this array — single-tile states are already handled by the
# tile task itself.
#
# This job is submitted with --dependency=afterany:<tile-array-id> so it
# runs after all tile jobs complete. embed.py --merge-only exits cleanly
# (a SKIP message, not an error) if any tiles are still missing — the
# resubmit chain will retry those tiles and re-run this merge.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"
VARIANT="${VARIANT:-v1_1-Base}"

: "${SCRATCH:?SCRATCH is not set}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$SCRATCH/embeddings-health/olmoearth_embeddings}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
: "${STATE_LIST:?STATE_LIST must be set to a colon-separated ordered state list}"
: "${NUM_TILES_LIST:?NUM_TILES_LIST must be set to a colon-separated list matching STATE_LIST}"

IFS=: read -ra STATES <<< "$STATE_LIST"
IFS=: read -ra NUM_TILES_ARR <<< "$NUM_TILES_LIST"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
STATE="${STATES[$TASK_ID]}"
NUM_TILES="${NUM_TILES_ARR[$TASK_ID]}"

CKPT_DIR="$SCRATCH/embeddings-health/checkpoints/olmoearth/$STATE"
FINAL_DIR="$FINAL_OUT_DIR/$STATE"
mkdir -p "$FINAL_DIR" "$CACHE_ROOT"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-${SLURMD_NODENAME:-$(hostname -s)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if command -v module >/dev/null 2>&1; then
  module load devel
  module load gcc/14.2.0
fi
export CC="$(command -v gcc)" CXX="$(command -v g++)"

if ! command -v uv >/dev/null 2>&1; then
  UV_INSTALL_DIR="${UV_INSTALL_DIR:-$CACHE_ROOT/uv-bin}"
  [[ -x "$UV_INSTALL_DIR/uv" ]] && export PATH="$UV_INSTALL_DIR:$PATH" || { echo "ERROR: uv not found." >&2; exit 1; }
fi

SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"
OUTPUT_BASENAME="olmoearth_${SAFE_VARIANT}_${STATE}_${YEAR}"

echo "Node:  $(hostname -s)"
echo "State: $STATE  (merge-only, $NUM_TILES tiles)"
echo ""

cd "$EMBED_DIR"
uv run --python 3.11 python embed.py \
  --model      olmoearth \
  --variant    "$VARIANT" \
  --output     "$CKPT_DIR/${OUTPUT_BASENAME}.tif" \
  --num-tiles  "$NUM_TILES" \
  --year       "$YEAR" \
  --merge-only

MERGED="$CKPT_DIR/${OUTPUT_BASENAME}.tif"
if [[ -s "$MERGED" ]]; then
  mv -f "$MERGED" "$FINAL_DIR/"
  MMAP_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.ckpt.mmap"
  [[ -f "$MMAP_LOCAL" ]] && rm -f "$MMAP_LOCAL" || true
  echo "=== Merge complete: $STATE ==="
  ls -lh "$FINAL_DIR/${OUTPUT_BASENAME}.tif"
else
  echo "=== Merge skipped or failed for $STATE — tiles not yet complete ==="
fi
```

- [ ] **Step 4: Verify bash syntax**

```bash
bash -n "$HOME/embeddings-health/code/embedding_generation/slurm/run_olmoearth_embed_merge.sbatch" && echo "syntax OK"
```

- [ ] **Step 5: Replace the whole file `submit_olmoearth_embed_all_states.sh`**

Replace the entire contents of `code/embedding_generation/slurm/submit_olmoearth_embed_all_states.sh` with:

```bash
#!/bin/bash
# Submit OlmoEarth Base embedding inference for all states with a complete
# composite, split into per-state tiles sized to a target chip count so
# every tile finishes within the sbatch script's fixed walltime — no more
# picking a bigger walltime tier for huge states (Sherlock's ceiling is
# fixed regardless, and Base's ~0.75s/chip rate means large states need tens
# of hours of total inference no single job could hold anyway). Safe to
# re-run — states/tiles with existing output are skipped.
#
# Usage:
#   bash submit_olmoearth_embed_all_states.sh
#   DRY_RUN=1 bash submit_olmoearth_embed_all_states.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
YEAR="${YEAR:-2022}"
VARIANT="${VARIANT:-v1_1-Base}"

: "${SCRATCH:?Set SCRATCH before submitting.}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$SCRATCH/embeddings-health/olmoearth_embeddings}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
LOG_DIR="${LOG_DIR:-$SCRATCH/embeddings-health/logs}"
CKPT_ROOT="$SCRATCH/embeddings-health/checkpoints/olmoearth"
TILE_SCRIPT="$SCRIPT_DIR/run_olmoearth_embed_state_array.sbatch"
MERGE_SCRIPT="$SCRIPT_DIR/run_olmoearth_embed_merge.sbatch"

# OlmoEarth chip size in input pixels at 10 m resolution.
CHIP_SIZE=128

# Target chip count per tile — sized from the measured ~0.75s/chip rate on NY
# (~90 min of inference + setup/merge margin within the 4h walltime). Real
# data, but from one state's observation — validate via srun (Task 9 Step 7
# of the embed-tiling plan) and adjust if a tile runs close to the limit.
TARGET_CHIPS_PER_TILE="${TARGET_CHIPS_PER_TILE:-7000}"

TASK_FILE="$SCRATCH/embeddings-health/cache/oe_tile_tasks_${YEAR}.txt"

LOADED_GDAL=0
if command -v module >/dev/null 2>&1 && ! command -v gdalinfo >/dev/null 2>&1; then
  module load devel 2>/dev/null || true
  module load physics gdal/3.10.2 2>/dev/null && LOADED_GDAL=1 || true
fi

get_chips() {
  local tif="$1"
  if ! command -v gdalinfo >/dev/null 2>&1; then
    echo "0"; return
  fi
  local size_line width height w_chips h_chips
  size_line=$(gdalinfo "$tif" 2>/dev/null | grep "^Size is" || true)
  if [[ -z "$size_line" ]]; then echo "0"; return; fi
  width=$(echo  "$size_line" | sed 's/Size is //' | cut -d',' -f1 | tr -d ' ')
  height=$(echo "$size_line" | sed 's/Size is //' | cut -d',' -f2 | tr -d ' ')
  w_chips=$(( (width  + CHIP_SIZE - 1) / CHIP_SIZE ))
  h_chips=$(( (height + CHIP_SIZE - 1) / CHIP_SIZE ))
  echo $(( w_chips * h_chips ))
}

get_h_chips() {
  local tif="$1"
  if ! command -v gdalinfo >/dev/null 2>&1; then
    echo "1"; return
  fi
  local size_line height
  size_line=$(gdalinfo "$tif" 2>/dev/null | grep "^Size is" || true)
  if [[ -z "$size_line" ]]; then echo "1"; return; fi
  height=$(echo "$size_line" | sed 's/Size is //' | cut -d',' -f2 | tr -d ' ')
  echo $(( (height + CHIP_SIZE - 1) / CHIP_SIZE ))
}

SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"

STATES=()
while IFS= read -r state; do
  STATES+=("$state")
done < <(
  find "$COMPOSITE_DIR" -maxdepth 1 -type f \
    -name "s2_annual_*_${YEAR}_olmoearth.tif" \
    -exec basename {} \; \
    | sed -E "s/^s2_annual_(.*)_${YEAR}_olmoearth\.tif$/\1/" \
    | sort
)

if (( ${#STATES[@]} == 0 )); then
  echo "ERROR: no OlmoEarth composites found in $COMPOSITE_DIR for year $YEAR" >&2
  exit 1
fi

MERGE_STATES=()
MERGE_NUM_TILES=()
> "$TASK_FILE"

total_tiles=0
skipped_states=0

for STATE in "${STATES[@]}"; do
  FINAL_TIF="$FINAL_OUT_DIR/$STATE/olmoearth_${SAFE_VARIANT}_${STATE}_${YEAR}.tif"
  if [[ -s "$FINAL_TIF" ]]; then
    (( skipped_states++ )) || true
    continue
  fi

  COMPOSITE="$COMPOSITE_DIR/s2_annual_${STATE}_${YEAR}_olmoearth.tif"
  CHIPS=$(get_chips "$COMPOSITE")
  H_CHIPS=$(get_h_chips "$COMPOSITE")

  if (( CHIPS == 0 )); then
    NUM_TILES=1
  else
    NUM_TILES=$(( (CHIPS + TARGET_CHIPS_PER_TILE - 1) / TARGET_CHIPS_PER_TILE ))
    (( NUM_TILES < 1 )) && NUM_TILES=1
    (( NUM_TILES > H_CHIPS )) && NUM_TILES=$H_CHIPS
  fi

  if (( NUM_TILES > 1 )); then
    MERGE_STATES+=("$STATE")
    MERGE_NUM_TILES+=("$NUM_TILES")
  fi

  CKPT_DIR="$CKPT_ROOT/$STATE"
  OUTPUT_BASENAME="olmoearth_${SAFE_VARIANT}_${STATE}_${YEAR}"

  for (( idx=0; idx<NUM_TILES; idx++ )); do
    if (( NUM_TILES > 1 )); then
      tile_path="$CKPT_DIR/${OUTPUT_BASENAME}_tile$(printf '%03d' "$idx").tif"
      [[ -s "$tile_path" ]] && continue
    fi
    echo "$STATE $idx $NUM_TILES" >> "$TASK_FILE"
    (( total_tiles++ )) || true
  done
done

if (( LOADED_GDAL )); then
  module unload gdal/3.10.2 2>/dev/null || true
fi

echo "Repo:          $REPO_DIR"
echo "Composites:    $COMPOSITE_DIR"
echo "Outputs:       $FINAL_OUT_DIR"
echo "Variant:       $VARIANT"
echo "Year:          $YEAR"
echo "Target chips/tile: $TARGET_CHIPS_PER_TILE"
echo ""
echo "States skipped (embedding exists): $skipped_states / ${#STATES[@]}"
echo "States needing a merge job:        ${#MERGE_STATES[@]}"
echo "Tile tasks to submit:              $total_tiles"
echo ""

if (( total_tiles == 0 && ${#MERGE_STATES[@]} == 0 )); then
  echo "All OlmoEarth Base embeddings complete. Nothing to submit."
  exit 0
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; not submitting."
  echo ""
  echo "First 10 tasks in $TASK_FILE:"
  head -10 "$TASK_FILE"
  exit 0
fi

mkdir -p "$LOG_DIR" "$FINAL_OUT_DIR"
export REPO_DIR COMPOSITE_DIR FINAL_OUT_DIR CACHE_ROOT YEAR VARIANT

MAX_ARRAY=1000
TILE_JOB_IDS=()
if (( total_tiles > 0 )); then
  batch=0
  offset=0
  while (( offset < total_tiles )); do
    end=$(( offset + MAX_ARRAY - 1 ))
    (( end >= total_tiles )) && end=$(( total_tiles - 1 ))
    count=$(( end - offset + 1 ))

    batch_file="${TASK_FILE%.txt}_batch${batch}.txt"
    sed -n "$((offset + 1)),$((end + 1))p" "$TASK_FILE" > "$batch_file"

    export TILE_TASK_FILE="$batch_file"
    JOB_ID=$(cd "$REPO_DIR" && sbatch \
      --export=ALL \
      --array="0-$(( count - 1 ))%200" \
      --output="$LOG_DIR/oe_embed_%A_%a.out" \
      --error="$LOG_DIR/oe_embed_%A_%a.err" \
      --parsable \
      "$TILE_SCRIPT" | cut -d';' -f1)
    echo "Submitted tile batch $batch: job $JOB_ID  ($count tasks, ≤200 concurrent)"
    TILE_JOB_IDS+=("$JOB_ID")
    (( batch++ )) || true
    (( offset += MAX_ARRAY )) || true
  done
fi
TILE_JOB_ID=$(IFS=:; echo "${TILE_JOB_IDS[*]}")

if (( ${#MERGE_STATES[@]} > 0 )); then
  MERGE_STATE_LIST=$(IFS=:; echo "${MERGE_STATES[*]}")
  MERGE_NUM_TILES_LIST=$(IFS=:; echo "${MERGE_NUM_TILES[*]}")
  MERGE_LAST_IDX=$(( ${#MERGE_STATES[@]} - 1 ))

  MERGE_DEP=""
  [[ -n "$TILE_JOB_ID" ]] && MERGE_DEP="--dependency=afterany:${TILE_JOB_ID}"

  export STATE_LIST="$MERGE_STATE_LIST"
  export NUM_TILES_LIST="$MERGE_NUM_TILES_LIST"
  MERGE_JOB_ID=$(cd "$REPO_DIR" && sbatch \
    --export=ALL \
    $MERGE_DEP \
    --array="0-${MERGE_LAST_IDX}" \
    --output="$LOG_DIR/oe_merge_%A_%a.out" \
    --error="$LOG_DIR/oe_merge_%A_%a.err" \
    --parsable \
    "$MERGE_SCRIPT" | cut -d';' -f1)
  echo "Submitted merge array job  $MERGE_JOB_ID  (${#MERGE_STATES[@]} states)"
  [[ -n "$TILE_JOB_ID" ]] && echo "  → depends on tile job $TILE_JOB_ID"
fi

ALL_DEPS="${TILE_JOB_ID}"
[[ -n "${MERGE_JOB_ID:-}" ]] && ALL_DEPS="${ALL_DEPS}:${MERGE_JOB_ID}"
SELF="$(realpath "${BASH_SOURCE[0]}")"
RESUBMIT_ID=$(sbatch \
  --dependency="afterany:${ALL_DEPS}" \
  --job-name=oe-embed-resubmit \
  --partition=normal \
  --time=00:10:00 \
  --mem=4G \
  --cpus-per-task=1 \
  --output="$LOG_DIR/oe_embed_resubmit_%j.out" \
  --error="$LOG_DIR/oe_embed_resubmit_%j.err" \
  --export=ALL \
  --parsable \
  --wrap="bash '$SELF'" | cut -d';' -f1)
echo "Resubmit job $RESUBMIT_ID scheduled after tiles+merge (cancel with: scancel $RESUBMIT_ID)"
```

- [ ] **Step 6: Verify bash syntax and dry-run**

```bash
bash -n "$HOME/embeddings-health/code/embedding_generation/slurm/submit_olmoearth_embed_all_states.sh" && echo "syntax OK"
DRY_RUN=1 bash "$HOME/embeddings-health/code/embedding_generation/slurm/submit_olmoearth_embed_all_states.sh"
```

- [ ] **Step 7: srun end-to-end validation on a real stuck Base state**

Pick a currently-stuck Base state — NY is a good choice since it's the one with the measured ~0.75s/chip rate this plan's `TARGET_CHIPS_PER_TILE=7000` was derived from:

```bash
STATE=NY
ls -lh "$SCRATCH/embeddings-health/olmoearth_composites/s2_annual_${STATE}_2022_olmoearth.tif"
```

Dry-run the submit script scoped to just that state's tile count:

```bash
DRY_RUN=1 bash "$HOME/embeddings-health/code/embedding_generation/slurm/submit_olmoearth_embed_all_states.sh" 2>&1 | grep -A2 "^$STATE " || true
grep "^$STATE " "$SCRATCH/embeddings-health/cache/oe_tile_tasks_2022.txt"
```

Expected: several lines `$STATE <idx> <NUM_TILES>` — note `NUM_TILES`.

Run tile 0 interactively via srun and confirm it finishes comfortably within the walltime:

```bash
TILE_TASK_FILE=/tmp/one_task.txt
echo "$STATE 0 <NUM_TILES from above>" > "$TILE_TASK_FILE"
export TILE_TASK_FILE REPO_DIR="$HOME/embeddings-health" YEAR=2022 VARIANT=v1_1-Base \
       COMPOSITE_DIR="$SCRATCH/embeddings-health/olmoearth_composites" \
       FINAL_OUT_DIR="$SCRATCH/embeddings-health/olmoearth_embeddings" \
       CACHE_ROOT="$SCRATCH/embeddings-health/cache"
srun -p gpu -G 1 -c 8 --mem=128G --time=04:00:00 \
  bash "$HOME/embeddings-health/code/embedding_generation/slurm/run_olmoearth_embed_state_array.sbatch"
```

Expected: completes with `=== Done: $STATE tile 0/<N> ===` well before the 4-hour limit. If it runs close to the limit, lower `TARGET_CHIPS_PER_TILE` in `submit_olmoearth_embed_all_states.sh` and re-dry-run before proceeding.

Confirm kill-and-resume works mid-tile: re-run the same `srun` command, but in a second terminal send `scancel <jobid>` (found via `squeue --me`) partway through, then re-run the identical `srun` command again. Expected: the second run's log shows `Resuming from memmap: N chips already processed` (or `Resuming from checkpoint:`) with `N > 0`.

Clean up the test tile output:

```bash
CKPT_DIR="$SCRATCH/embeddings-health/checkpoints/olmoearth/$STATE"
rm -f "$CKPT_DIR"/olmoearth_v1_1-Base_${STATE}_2022_tile000*
rm -f /tmp/one_task.txt
```

- [ ] **Step 8: Commit**

```bash
cd "$HOME/embeddings-health"
git add code/embedding_generation/slurm/run_olmoearth_embed_state_array.sbatch \
        code/embedding_generation/slurm/run_olmoearth_embed_merge.sbatch \
        code/embedding_generation/slurm/submit_olmoearth_embed_all_states.sh
git commit -m "feat: apply tile/merge pattern to oe-embed"
```

---

## Task 10: Manual go/no-go — real batch submission

**This is a checkpoint, not an automated step.** Do not run these submissions unattended.

- [ ] Review Tasks 3, 7, and the Clay/Base equivalents of Task 7 with the user — confirm all srun validations passed and `TARGET_CHIPS_PER_TILE` values were adjusted where needed.
- [ ] With explicit user go-ahead, submit real batch arrays **one pipeline at a time, in this order**: `bash submit_olmoearth_nano_embed_all_states.sh` → wait for it to substantially progress/complete → `bash submit_clay_embed_all_states.sh` → `bash submit_olmoearth_embed_all_states.sh`.
- [ ] After each submission, spot-check `squeue --me` and `sacct` for the first handful of completed tasks to confirm tiles are finishing well within their walltime (not just barely), before moving on to the next pipeline.
- [ ] Monitor `$SCRATCH` usage (`sh_quota`) over the following days to confirm the disk-leak fix (staging-leak memory) plus this tiling fix together stop the repeated multi-TB growth pattern.
