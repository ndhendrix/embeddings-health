"""Tests for the analysis functions.

The synthetic result fixture has a deliberately known structure -- dimension 0
is a horizontal ramp, dimension 1 is constant -- so these tests check real
answers rather than just shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from olmoearth_activations import analysis
from olmoearth_activations.encode import EmbeddingResult


def test_dimension_map_returns_the_ramp(
    synthetic_result: EmbeddingResult,
) -> None:
    """Dimension 0 is a horizontal ramp, so every row must be 0..W-1."""
    values = analysis.dimension_map(synthetic_result, 0, "blk1")
    height, width = synthetic_result.grid_shape
    assert values.shape == (height, width)
    for row in range(height):
        np.testing.assert_allclose(values[row], np.arange(width))


def test_dimension_map_rejects_bad_dim(
    synthetic_result: EmbeddingResult,
) -> None:
    """An out-of-range dimension must say what the width is."""
    with pytest.raises(IndexError, match="token width 8"):
        analysis.dimension_map(synthetic_result, 99)


def test_top_patches_finds_the_ramp_maximum(
    synthetic_result: EmbeddingResult,
) -> None:
    """The highest values on a horizontal ramp are all in the last column."""
    top = analysis.top_patches(synthetic_result, 0, k=5, tap="blk1")
    assert list(top.columns) == ["rank", "row", "col", "token_index", "value"]
    assert len(top) == 5
    width = synthetic_result.grid_shape[1]
    assert (top["col"] == width - 1).all()
    # Ranked strongest first.
    assert top["value"].is_monotonic_decreasing
    # token_index must be consistent with (row, col).
    np.testing.assert_array_equal(
        top["token_index"].to_numpy(), top["row"].to_numpy() * width + top["col"].to_numpy()
    )


def test_top_patches_lowest(synthetic_result: EmbeddingResult) -> None:
    """``lowest=True`` must pick the other end of the ramp."""
    bottom = analysis.top_patches(synthetic_result, 0, k=4, tap="blk1", lowest=True)
    assert (bottom["col"] == 0).all()


def test_top_patches_clips_k(synthetic_result: EmbeddingResult) -> None:
    """Asking for more patches than exist returns all of them, not an error."""
    top = analysis.top_patches(synthetic_result, 0, k=10_000)
    assert len(top) == synthetic_result.n_patches


def test_most_variable_dims_ranks_ramp_first_constant_last(
    synthetic_result: EmbeddingResult,
) -> None:
    """The ramp varies most; the constant dimension must not rank above noise."""
    stats = analysis.most_variable_dims(synthetic_result, k=8, tap="blk1")
    assert list(stats.columns) == ["dim", "std", "range", "min", "max", "mean"]
    assert int(stats.iloc[0]["dim"]) == 0
    assert int(stats.iloc[-1]["dim"]) == 1
    # The constant dimension must report zero spread, which is the signal that
    # a rescaled heatmap of it would be pure noise amplification.
    constant = stats[stats["dim"] == 1].iloc[0]
    assert constant["std"] == pytest.approx(0.0)
    assert constant["range"] == pytest.approx(0.0)


def test_project_onto_recovers_a_basis_direction(
    synthetic_result: EmbeddingResult,
) -> None:
    """Projecting onto e_0 must reproduce dimension 0's map."""
    direction = np.zeros(synthetic_result.embed_dim, dtype=np.float32)
    direction[0] = 1.0
    projected = analysis.project_onto(synthetic_result, direction, "blk1")
    np.testing.assert_allclose(
        projected, analysis.dimension_map(synthetic_result, 0, "blk1"), atol=1e-5
    )


def test_project_onto_normalizes_by_default(
    synthetic_result: EmbeddingResult,
) -> None:
    """Scaling the direction must not scale the output when normalizing."""
    direction = np.zeros(synthetic_result.embed_dim, dtype=np.float32)
    direction[0] = 5.0
    normalized = analysis.project_onto(synthetic_result, direction, "blk1")
    raw = analysis.project_onto(
        synthetic_result, direction, "blk1", unit_normalize=False
    )
    np.testing.assert_allclose(raw, normalized * 5.0, rtol=1e-5)


def test_project_onto_rejects_wrong_width(
    synthetic_result: EmbeddingResult,
) -> None:
    """A mismatched direction must name both widths."""
    with pytest.raises(ValueError, match="length 3.*token width 8"):
        analysis.project_onto(synthetic_result, np.ones(3, dtype=np.float32))


def test_project_onto_rejects_zero_vector(
    synthetic_result: EmbeddingResult,
) -> None:
    """Normalizing a zero vector is undefined; say so."""
    with pytest.raises(ValueError, match="zero vector"):
        analysis.project_onto(
            synthetic_result, np.zeros(synthetic_result.embed_dim, dtype=np.float32)
        )


def test_depth_trajectory_shape_and_bounds(
    synthetic_result: EmbeddingResult,
) -> None:
    """One patch across depth is (n_taps, D), and bad coords must raise."""
    traj = analysis.depth_trajectory(synthetic_result, 3, 4)
    assert traj.shape == (len(synthetic_result.tap_labels), synthetic_result.embed_dim)
    # Dimension 0 is the same ramp at every tap, so column 0 is constant.
    np.testing.assert_allclose(traj[:, 0], 4.0)
    with pytest.raises(IndexError, match="outside the 16x16 token grid"):
        analysis.depth_trajectory(synthetic_result, 99, 0)


def test_depth_drift_has_one_row_per_gap(
    synthetic_result: EmbeddingResult,
) -> None:
    """Consecutive-tap drift has n_taps-1 rows and finite cosines."""
    drift = analysis.depth_drift(synthetic_result)
    assert len(drift) == len(synthetic_result.tap_labels) - 1
    assert list(drift["from_tap"]) == ["proj", "blk1"]
    assert np.isfinite(drift["mean_cosine"]).all()
    assert (drift["mean_cosine"] <= 1.0 + 1e-6).all()


@pytest.mark.parametrize("how", ["mean", "median", "min", "max", "std"])
def test_pool_shapes(synthetic_result: EmbeddingResult, how: str) -> None:
    """Every pooling statistic returns a (D,) vector."""
    vec = analysis.pool(synthetic_result, "blk1", how)  # type: ignore[arg-type]
    assert vec.shape == (synthetic_result.embed_dim,)


def test_pool_mean_of_ramp(synthetic_result: EmbeddingResult) -> None:
    """The mean of a 0..15 ramp is 7.5."""
    vec = analysis.pool(synthetic_result, "blk1", "mean")
    assert vec[0] == pytest.approx(7.5)
    assert vec[1] == pytest.approx(7.0)


def test_pool_rejects_unknown_statistic(
    synthetic_result: EmbeddingResult,
) -> None:
    """Unknown statistics must list the valid ones."""
    with pytest.raises(ValueError, match="unknown pooling statistic"):
        analysis.pool(synthetic_result, "blk1", "mode")  # type: ignore[arg-type]


def test_pool_all_column_schema(synthetic_result: EmbeddingResult) -> None:
    """Column names follow the production {DIM}_{STAT} schema."""
    frame = analysis.pool_all(synthetic_result, "blk1")
    assert len(frame) == 1
    assert frame.shape[1] == synthetic_result.embed_dim * 5
    assert "D000_MEAN" in frame.columns
    assert "D000_MINIMUM" in frame.columns
    assert "D007_STD" in frame.columns


def test_compare_locations_diagonal_is_one(
    synthetic_result: EmbeddingResult,
) -> None:
    """A chip is identical to itself, and labels come from metadata."""
    other = EmbeddingResult(
        embeddings=synthetic_result.embeddings * -1.0,
        activations=synthetic_result.activations * -1.0,
        tap_labels=synthetic_result.tap_labels,
        tap_depths=synthetic_result.tap_depths,
        grid_shape=synthetic_result.grid_shape,
        meta={"name": "flipped"},
    )
    sim = analysis.compare_locations([synthetic_result, other], "blk1")
    assert list(sim.index) == ["synthetic", "flipped"]
    np.testing.assert_allclose(np.diag(sim.to_numpy()), 1.0, atol=1e-5)
    # A sign-flipped chip must be anti-similar, not similar.
    assert sim.iloc[0, 1] == pytest.approx(-1.0, abs=1e-5)


def test_compare_locations_needs_two(
    synthetic_result: EmbeddingResult,
) -> None:
    """One chip is not a comparison."""
    with pytest.raises(ValueError, match="at least two"):
        analysis.compare_locations([synthetic_result])
