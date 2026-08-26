"""Config validation, and the normalization/band-order properties.

The normalization tests are the important ones. Both strategies use fixed
per-band constants, so normalizing a region and then slicing must equal slicing
and then normalizing. The prototype's ``sweep_patch`` docstring worries that
per-chip normalization could manufacture a fake edge effect; this is the test
that settles it.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from olmoearth_activations.config import (
    ModelConfig,
    RunConfig,
    TapConfig,
    TileConfig,
)
from olmoearth_activations.tiles import normalize_chip
from tests.conftest import N_BANDS, requires_olmoearth


# ------------------------------------------------------------------- config


def test_chip_must_divide_by_patch() -> None:
    """A non-divisible chip would give a ragged token grid; reject it early."""
    with pytest.raises(ValueError, match="divisible"):
        TileConfig(chip_px=65, patch_px=4)


def test_expected_grid_side() -> None:
    """The cross-check value, not the authority."""
    assert TileConfig(chip_px=128, patch_px=4).expected_grid_side == 32


def test_rejects_bad_date() -> None:
    """The date must be a triple."""
    with pytest.raises(ValueError, match="day, month, year"):
        TileConfig(date=(2022, 6))  # type: ignore[arg-type]


def test_fingerprint_changes_with_settings() -> None:
    """Two configs that differ must not share a fingerprint."""
    a = RunConfig()
    b = RunConfig(tile=TileConfig(chip_px=128))
    assert a.fingerprint() != b.fingerprint()
    # ...and must be stable for identical settings.
    assert a.fingerprint() == RunConfig().fingerprint()


def test_yaml_round_trip(tmp_path) -> None:
    """A config written and read back must be unchanged."""
    cfg = RunConfig(
        model=ModelConfig(model_id="OlmoEarth-v1_2-Nano", hf_revision="abc123"),
        tile=TileConfig(chip_px=128, patch_px=4, normalize="predefined"),
        tap=TapConfig(method="hooks", depths=(1, 2)),
        save_dtype="float16",
    )
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    loaded = RunConfig.from_yaml(path)
    assert loaded == cfg
    assert loaded.fingerprint() == cfg.fingerprint()


def test_unknown_top_level_key_is_rejected(tmp_path) -> None:
    """A silently-ignored typo would leave a default in place unnoticed."""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"tiles": {"chip_px": 128}}))
    with pytest.raises(ValueError, match="unknown top-level config keys"):
        RunConfig.from_yaml(path)


def test_unknown_section_key_is_rejected(tmp_path) -> None:
    """Same, one level down."""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"tile": {"chip_size": 128}}))
    with pytest.raises(ValueError, match="unknown keys in config section 'tile'"):
        RunConfig.from_yaml(path)


def test_yaml_lists_become_tuples(tmp_path) -> None:
    """YAML has no tuples; the loader must coerce so configs stay hashable."""
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump({"tile": {"date": [1, 2, 2020]}, "tap": {"depths": [0, 2]}})
    )
    cfg = RunConfig.from_yaml(path)
    assert cfg.tile.date == (1, 2, 2020)
    assert cfg.tap.depths == (0, 2)


# ------------------------------------------------------------ normalization


def test_normalize_none_is_passthrough(synthetic_chip: np.ndarray) -> None:
    """The 'none' strategy must not touch the values."""
    out = normalize_chip(synthetic_chip, "none")
    np.testing.assert_array_equal(out, synthetic_chip.astype(np.float32))


@requires_olmoearth
@pytest.mark.parametrize("strategy", ["computed", "predefined"])
def test_normalization_is_stateless(
    synthetic_chip: np.ndarray, strategy: str
) -> None:
    """Normalize-then-slice must equal slice-then-normalize.

    This is what makes it safe to normalize a whole region once and cut
    overlapping chips out of it, as the position sweep does.
    """
    region = synthetic_chip
    whole = normalize_chip(region, strategy)  # type: ignore[arg-type]
    window = (slice(8, 40), slice(12, 44))
    sliced_then_normed = normalize_chip(region[window], strategy)  # type: ignore[arg-type]
    np.testing.assert_allclose(
        whole[window], sliced_then_normed, rtol=1e-6, atol=1e-6
    )


@requires_olmoearth
def test_normalization_changes_scale_substantially(
    synthetic_chip: np.ndarray,
) -> None:
    """Normalized input must be order-unity, not order-thousands.

    This guards the finding that motivated this package: the production
    embedding pipeline feeds raw digital numbers to a model trained on
    normalized input. If this assertion ever fails, check what changed before
    trusting any comparison against production embeddings.
    """
    raw = synthetic_chip
    normed = normalize_chip(raw, "computed")
    assert np.abs(raw).mean() > 100.0, "fixture should be in DN units"
    assert np.abs(normed).mean() < 5.0, (
        f"normalized values should be order-unity, got mean "
        f"{np.abs(normed).mean():.3f}"
    )


@requires_olmoearth
def test_normalization_rejects_unknown_strategy(
    synthetic_chip: np.ndarray,
) -> None:
    """Unknown strategies must list the valid ones."""
    with pytest.raises(ValueError, match="unknown normalize strategy"):
        normalize_chip(synthetic_chip, "zscore")  # type: ignore[arg-type]


# -------------------------------------------------------------- band order


@requires_olmoearth
def test_band_order_is_the_expected_sequence() -> None:
    """Pin the band order explicitly.

    Not alphabetical, and a permutation raises no error in the model -- it only
    produces wrong numbers. If this test fails, the upstream modality spec
    changed and every stored embedding needs rechecking.
    """
    from olmoearth_activations.tiles import default_band_order

    assert default_band_order() == [
        "B02",
        "B03",
        "B04",
        "B08",
        "B05",
        "B06",
        "B07",
        "B8A",
        "B11",
        "B12",
        "B01",
        "B09",
    ]


@requires_olmoearth
def test_band_signature_survives_normalization(
    band_signature_chip: np.ndarray,
) -> None:
    """Per-band constants must stay distinguishable and stay in order.

    Each band's normalization constants differ, so the normalized values are not
    equal across bands -- but they must remain constant *within* a band, and the
    channel axis must not be permuted.
    """
    from olmoearth_activations.tiles import default_band_order

    assert band_signature_chip.shape[-1] == len(default_band_order()) == N_BANDS
    normed = normalize_chip(band_signature_chip, "computed")

    for i in range(N_BANDS):
        band = normed[:, :, i]
        assert np.allclose(band, band.flat[0]), (
            f"band {i} should be constant after normalization but has spread "
            f"{band.max() - band.min():.3g}"
        )

    # A permutation of the input channels must produce a different result --
    # otherwise this test could not detect one.
    permuted = normalize_chip(band_signature_chip[:, :, ::-1], "computed")
    assert not np.allclose(normed, permuted)
