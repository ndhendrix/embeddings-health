"""Shape contract and round-trip tests for :class:`EmbeddingResult`.

These need neither a checkpoint nor the upstream package, which is deliberate:
the reshape from flat tokens to a spatial grid is the single easiest place to
silently misalign every patch, so it should be testable everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from olmoearth_activations.encode import EmbeddingResult, tap_label


def test_grid_and_tokens_are_consistent(synthetic_result: EmbeddingResult) -> None:
    """``tokens()`` must be a row-major flattening of ``grid()``."""
    grid = synthetic_result.grid("blk1")
    tokens = synthetic_result.tokens("blk1")
    height, width = synthetic_result.grid_shape

    assert grid.shape == (height, width, synthetic_result.embed_dim)
    assert tokens.shape == (height * width, synthetic_result.embed_dim)
    np.testing.assert_array_equal(tokens, grid.reshape(-1, grid.shape[-1]))

    # Patch (row, col) must be token row*W + col. If this ever inverts, every
    # spatial claim downstream silently transposes.
    for row, col in ((0, 0), (0, 3), (2, 0), (height - 1, width - 1)):
        np.testing.assert_array_equal(
            tokens[row * width + col], grid[row, col]
        )


def test_grid_shape_matches_embeddings(synthetic_result: EmbeddingResult) -> None:
    """``grid_shape`` is not allowed to disagree with the arrays."""
    assert synthetic_result.grid_shape == synthetic_result.embeddings.shape[:2]
    assert synthetic_result.n_patches == 16 * 16


def test_tap_selection_by_label_and_index(
    synthetic_result: EmbeddingResult,
) -> None:
    """Labels, positive indices and negative indices must agree."""
    assert synthetic_result.tap_index("proj") == 0
    assert synthetic_result.tap_index("blk2") == 2
    assert synthetic_result.tap_index(-1) == 2
    np.testing.assert_array_equal(
        synthetic_result.grid("blk2"), synthetic_result.grid(-1)
    )


def test_unknown_tap_label_lists_available(
    synthetic_result: EmbeddingResult,
) -> None:
    """An unhelpful KeyError here would be a real usability failure."""
    with pytest.raises(KeyError, match=r"blk9.*available"):
        synthetic_result.grid("blk9")


def test_out_of_range_tap_index(synthetic_result: EmbeddingResult) -> None:
    """Index errors must state the range."""
    with pytest.raises(IndexError, match="out of range"):
        synthetic_result.grid(7)


def test_deepest_tap_matches_embeddings(
    synthetic_result: EmbeddingResult,
) -> None:
    """The default tap is the deepest one, which is the final embedding."""
    np.testing.assert_array_equal(
        synthetic_result.grid(-1), synthetic_result.embeddings
    )


@pytest.mark.parametrize(
    ("bad_kwargs", "match"),
    [
        ({"embeddings": np.zeros((4, 4), np.float32)}, "must be"),
        ({"activations": np.zeros((4, 4, 4), np.float32)}, "must be"),
        ({"tap_labels": ["only_one"]}, "tap labels"),
        ({"grid_shape": (2, 2)}, "does not match"),
    ],
)
def test_validation_rejects_inconsistent_inputs(
    synthetic_result: EmbeddingResult, bad_kwargs: dict, match: str
) -> None:
    """Constructing an inconsistent result must fail loudly, not later."""
    kwargs = {
        "embeddings": synthetic_result.embeddings,
        "activations": synthetic_result.activations,
        "tap_labels": synthetic_result.tap_labels,
        "tap_depths": synthetic_result.tap_depths,
        "grid_shape": synthetic_result.grid_shape,
    }
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError, match=match):
        EmbeddingResult(**kwargs)


def test_save_load_round_trip(
    synthetic_result: EmbeddingResult, tmp_path
) -> None:
    """A saved result must come back identical, with its manifest beside it."""
    path = synthetic_result.save(tmp_path / "r.npz")
    manifest = tmp_path / "r_manifest.json"
    assert manifest.exists(), "save() must write a manifest sidecar"

    loaded = EmbeddingResult.load(path)
    np.testing.assert_allclose(loaded.embeddings, synthetic_result.embeddings)
    np.testing.assert_allclose(loaded.activations, synthetic_result.activations)
    assert loaded.tap_labels == synthetic_result.tap_labels
    assert loaded.tap_depths == synthetic_result.tap_depths
    assert loaded.grid_shape == synthetic_result.grid_shape
    assert loaded.meta["name"] == "synthetic"


def test_float16_save_is_lossy_but_close(
    synthetic_result: EmbeddingResult, tmp_path
) -> None:
    """float16 output should shrink the file without wrecking the values."""
    path = synthetic_result.save(tmp_path / "half.npz", dtype="float16")
    loaded = EmbeddingResult.load(path)
    np.testing.assert_allclose(
        loaded.activations, synthetic_result.activations, rtol=1e-2, atol=1e-2
    )


def test_tap_label_naming() -> None:
    """Depth 0 is the projection; deeper taps are named by block."""
    assert tap_label(0) == "proj"
    assert tap_label(1) == "blk1"
    assert tap_label(12) == "blk12"
