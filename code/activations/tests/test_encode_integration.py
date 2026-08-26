"""Integration tests that need a real checkpoint.

Marked ``integration`` and skipped when ``olmoearth_pretrain`` is missing or the
configured model id is absent from the installed ``ModelID`` enum. Run with::

    pytest -m integration

The tap-agreement test is the important one. It does not assume a relationship
between hook taps and ``token_exit`` taps -- it measures one and reports it, so
whatever the truth is ends up written down rather than guessed.
"""

from __future__ import annotations

import numpy as np
import pytest

from olmoearth_activations.config import TapConfig, TileConfig
from olmoearth_activations.encode import Encoder

pytestmark = pytest.mark.integration


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-patch cosine similarity between two (H, W, D) grids."""
    x = a.reshape(-1, a.shape[-1])
    y = b.reshape(-1, b.shape[-1])
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    return float(np.mean((x * y).sum(axis=1) / np.maximum(denom, 1e-12)))


def test_grid_shape_matches_encoder(
    model, synthetic_chip: np.ndarray, tile_cfg: TileConfig
) -> None:
    """The reported grid must match the encoder's own output, and the taps must
    all share it."""
    encoder = Encoder(model, tile_cfg, TapConfig(method="token_exit"))
    result = encoder.encode(synthetic_chip)

    side = tile_cfg.chip_px // tile_cfg.patch_px
    assert result.grid_shape == (side, side), (
        f"expected a {side}x{side} token grid for a {tile_cfg.chip_px} px chip "
        f"at patch {tile_cfg.patch_px}, got {result.grid_shape}"
    )
    assert result.embed_dim == model.embed_dim
    assert result.activations.shape == (
        len(result.tap_labels),
        side,
        side,
        model.embed_dim,
    )
    assert result.tokens("proj").shape == (side * side, model.embed_dim)
    # token_exit should give depth+1 taps: the projection plus one per block.
    assert result.tap_depths == list(range(model.depth + 1))
    assert result.tap_labels[0] == "proj"
    assert result.tap_labels[-1] == f"blk{model.depth}"


def test_determinism_and_eval_mode(
    model, synthetic_chip: np.ndarray, tile_cfg: TileConfig
) -> None:
    """Two encodes of one chip must be bitwise identical."""
    assert model.module.training is False
    encoder = Encoder(model, tile_cfg, TapConfig(method="token_exit"))
    first = encoder.encode(synthetic_chip)
    second = encoder.encode(synthetic_chip)
    assert model.module.training is False
    np.testing.assert_array_equal(first.embeddings, second.embeddings)
    np.testing.assert_array_equal(first.activations, second.activations)


def test_fast_pass_does_not_change_output(
    model, synthetic_chip: np.ndarray, tile_cfg: TileConfig
) -> None:
    """fast_pass is documented as a performance switch only.

    In eval mode with an all-ONLINE_ENCODER mask, the attention mask is None
    either way and the remove/add-masked-tokens round trip is an identity. If
    this fails, the claim in encode.py's docstring is wrong and the depth sweep
    needs re-running with the slow path.
    """
    fast = Encoder(model, tile_cfg, TapConfig(fast_pass=True)).encode(synthetic_chip)
    slow = Encoder(model, tile_cfg, TapConfig(fast_pass=False)).encode(synthetic_chip)
    max_diff = float(np.max(np.abs(fast.embeddings - slow.embeddings)))
    np.testing.assert_allclose(
        fast.embeddings,
        slow.embeddings,
        rtol=0,
        atol=0,
        err_msg=(
            f"fast_pass changed the output; max abs difference {max_diff:.3e}. "
            f"Treat fast_pass as semantically meaningful and re-check every "
            f"stored activation."
        ),
    )


def test_token_exit_full_depth_equals_plain_pass(
    model, synthetic_chip: np.ndarray, tile_cfg: TileConfig
) -> None:
    """The deepest token_exit tap is reused from the plain pass.

    ``Encoder.encode`` relies on that equivalence to save a forward pass, so it
    needs holding honest: ask for full depth explicitly and compare against the
    final embedding.
    """
    encoder = Encoder(
        model, tile_cfg, TapConfig(method="token_exit", depths=(model.depth,))
    )
    result = encoder.encode(synthetic_chip)
    np.testing.assert_array_equal(result.grid(-1), result.embeddings)


def test_hooks_and_token_exit_relationship(
    model, synthetic_chip: np.ndarray, tile_cfg: TileConfig
) -> None:
    """Measure and report how hook taps relate to token_exit taps.

    Hooks capture the residual stream *before* the encoder's final LayerNorm,
    so exact equality is not expected. What we assert is the weaker,
    genuinely-informative claim: the two are highly correlated per patch, i.e.
    they differ by something close to a per-token rescaling rather than being
    different representations.
    """
    exit_result = Encoder(
        model, tile_cfg, TapConfig(method="token_exit")
    ).encode(synthetic_chip)
    hook_result = Encoder(model, tile_cfg, TapConfig(method="hooks")).encode(
        synthetic_chip
    )

    assert hook_result.tap_depths == list(range(1, model.depth + 1)), (
        "hooks cannot produce depth 0, so they should start at depth 1"
    )

    report: list[str] = []
    for depth in hook_result.tap_depths:
        hook_grid = hook_result.grid(f"blk{depth}")
        exit_grid = exit_result.grid(f"blk{depth}")
        cos = _cosine(hook_grid, exit_grid)
        max_abs = float(np.max(np.abs(hook_grid - exit_grid)))
        norm_ratio = float(
            np.mean(np.linalg.norm(hook_grid.reshape(-1, model.embed_dim), axis=1))
            / max(
                np.mean(
                    np.linalg.norm(exit_grid.reshape(-1, model.embed_dim), axis=1)
                ),
                1e-12,
            )
        )
        report.append(
            f"blk{depth}: cosine={cos:.6f} max_abs_diff={max_abs:.4e} "
            f"norm_ratio={norm_ratio:.4f}"
        )

    message = "hooks vs token_exit per depth:\n  " + "\n  ".join(report)
    # Deliberately loose: this documents the relationship rather than pinning a
    # number nobody has verified. Tighten it once the real values are known.
    deepest_cos = _cosine(
        hook_result.grid(f"blk{model.depth}"),
        exit_result.grid(f"blk{model.depth}"),
    )
    assert deepest_cos > 0.9, (
        f"hook and token_exit taps at the deepest layer should be strongly "
        f"aligned, got cosine {deepest_cos:.4f}.\n{message}"
    )
    print("\n" + message)


def test_no_hook_leakage(
    model, synthetic_chip: np.ndarray, tile_cfg: TileConfig
) -> None:
    """No forward hooks may survive an encode call.

    A leaked hook would keep writing into a dead store on every later forward
    pass -- a slow memory leak and a source of confusing cross-talk.
    """
    encoder = Encoder(model, tile_cfg, TapConfig(method="hooks"))

    def hook_count() -> int:
        return sum(
            len(model.submodule(name)._forward_hooks)
            for name in model.block_names
        )

    before = hook_count()
    encoder.encode(synthetic_chip)
    assert hook_count() == before, "encode() left forward hooks registered"
    assert encoder._store == {}, "encode() left activations in the store"


def test_hooks_leak_nothing_on_error(model, tile_cfg: TileConfig) -> None:
    """Hooks must be removed even when the forward pass raises."""
    encoder = Encoder(model, tile_cfg, TapConfig(method="hooks"))

    def hook_count() -> int:
        return sum(
            len(model.submodule(name)._forward_hooks)
            for name in model.block_names
        )

    before = hook_count()
    with pytest.raises(Exception):
        # Wrong channel count: the patch embedding should reject this.
        encoder.encode(np.zeros((64, 64, 3), dtype=np.float32))
    assert hook_count() == before, "a failed encode() left forward hooks behind"


def test_squeeze_policy_rejects_multi_timestep(
    model, synthetic_chip: np.ndarray, tile_cfg: TileConfig
) -> None:
    """extra_dims='squeeze' must refuse to average time silently."""
    two_step = np.stack([synthetic_chip, synthetic_chip], axis=2)
    chip = two_step[None].transpose(0, 1, 2, 3, 4)  # (1, H, W, T=2, C)
    encoder = Encoder(model, tile_cfg, TapConfig(extra_dims="squeeze"))
    with pytest.raises(ValueError, match="refuses to average"):
        encoder.encode(chip)


def test_depths_filter_is_respected(
    model, synthetic_chip: np.ndarray, tile_cfg: TileConfig
) -> None:
    """Requesting a subset of depths must return exactly that subset."""
    encoder = Encoder(
        model, tile_cfg, TapConfig(method="token_exit", depths=(0, model.depth))
    )
    result = encoder.encode(synthetic_chip)
    assert result.tap_depths == [0, model.depth]
    assert result.tap_labels == ["proj", f"blk{model.depth}"]


def test_hooks_drop_depth_zero_with_warning(
    model, synthetic_chip: np.ndarray, tile_cfg: TileConfig, caplog
) -> None:
    """Depth 0 is unreachable by hooks; it must be dropped, loudly."""
    encoder = Encoder(
        model, tile_cfg, TapConfig(method="hooks", depths=(0, 1))
    )
    with caplog.at_level("WARNING"):
        result = encoder.encode(synthetic_chip)
    assert result.tap_depths == [1]
    assert any("dropping requested depths" in r.message for r in caplog.records)
