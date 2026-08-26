"""Shared fixtures.

Most tests use a synthetic chip so the suite runs with neither a ``.SAFE`` scene
nor a network download. Anything that genuinely needs a checkpoint is marked
``integration`` and skipped unless ``olmoearth_pretrain`` is importable.
"""

from __future__ import annotations

import importlib.util
from typing import Iterator

import numpy as np
import pytest

from olmoearth_activations.config import ModelConfig, TapConfig, TileConfig
from olmoearth_activations.encode import EmbeddingResult

#: Sentinel-2 L2A band count for the modality this package works in.
N_BANDS = 12

#: Skip marker for tests that need the upstream package importable.
requires_olmoearth = pytest.mark.skipif(
    importlib.util.find_spec("olmoearth_pretrain") is None,
    reason="olmoearth_pretrain is not installed",
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator, so failures are reproducible."""
    return np.random.default_rng(20260821)


@pytest.fixture
def synthetic_chip(rng: np.random.Generator) -> np.ndarray:
    """A 64x64x12 chip with plausible reflectance values.

    Values sit in the 0-10000 digital-number range that Sentinel-2 L2A COGs use,
    with a per-band offset so the bands are distinguishable -- which is what lets
    a band-order test detect a permutation.
    """
    base = rng.uniform(200.0, 4000.0, size=(64, 64, N_BANDS))
    per_band_offset = np.arange(N_BANDS, dtype=np.float64) * 100.0
    return (base + per_band_offset).astype(np.float32)


@pytest.fixture
def band_signature_chip() -> np.ndarray:
    """A chip where every band is a distinct constant.

    Band ``i`` is filled with ``(i + 1) * 500``. Any reordering of the channel
    axis is then immediately visible, which is the point: a wrong band order
    raises no error in the model, it only produces wrong numbers.
    """
    chip = np.zeros((16, 16, N_BANDS), dtype=np.float32)
    for i in range(N_BANDS):
        chip[:, :, i] = (i + 1) * 500.0
    return chip


@pytest.fixture
def tile_cfg() -> TileConfig:
    """A small chip config matching the synthetic fixtures."""
    return TileConfig(chip_px=64, patch_px=4, date=(15, 6, 2022))


@pytest.fixture
def synthetic_result(rng: np.random.Generator) -> EmbeddingResult:
    """A hand-built result with a known structure.

    The grid is 16x16 with 8 dimensions and 3 taps. Dimension 0 is a horizontal
    ramp, so ``dimension_map`` and ``top_patches`` have a checkable answer;
    dimension 1 is constant, so ``most_variable_dims`` must rank it last.
    """
    n_taps, height, width, dim = 3, 16, 16, 8
    acts = rng.normal(0.0, 0.1, size=(n_taps, height, width, dim)).astype(np.float32)
    ramp = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    for t in range(n_taps):
        acts[t, :, :, 0] = ramp
        acts[t, :, :, 1] = 7.0
    return EmbeddingResult(
        embeddings=acts[-1].copy(),
        activations=acts,
        tap_labels=["proj", "blk1", "blk2"],
        tap_depths=[0, 1, 2],
        grid_shape=(height, width),
        meta={"name": "synthetic", "lat": 38.9, "lon": -77.0},
    )


@pytest.fixture(scope="session")
def model() -> Iterator[object]:
    """A loaded model, for integration tests.

    Downloads a checkpoint on first use. Skips -- rather than fails -- when the
    package is missing or the configured model id is not in the installed
    ``ModelID`` enum, because that is an environment fact, not a code defect.
    """
    pytest.importorskip("olmoearth_pretrain")
    from olmoearth_activations.loader import OlmoEarthModel

    try:
        yield OlmoEarthModel.load(ModelConfig())
    except RuntimeError as exc:  # unknown ModelID member, or no blocks found
        pytest.skip(f"cannot load model: {exc}")


@pytest.fixture
def tap_cfg_token_exit() -> TapConfig:
    """Token-exit taps at every depth."""
    return TapConfig(method="token_exit")


@pytest.fixture
def tap_cfg_hooks() -> TapConfig:
    """Hook taps at every block."""
    return TapConfig(method="hooks")
