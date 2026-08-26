"""What fires where: turning per-patch activations into answerable questions.

Every function here is pure -- it takes :class:`~olmoearth_activations.encode.EmbeddingResult`
objects and returns arrays or DataFrames, and never runs the model. That split
matters because it means analysis can be re-run cheaply on saved ``.npz`` files
without a GPU or a checkpoint.

A note on default taps
----------------------
The default tap is ``-1``, the deepest available, rather than a literal like
``"blk4"``. Nano has four blocks and Base has twelve; hardcoding a label would
silently break the moment the model changes. Pass an explicit label when you
want a specific depth.
"""

from __future__ import annotations

import logging
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from olmoearth_activations.encode import EmbeddingResult

logger = logging.getLogger(__name__)

PoolHow = Literal["mean", "median", "min", "max", "std"]


def dimension_map(
    result: EmbeddingResult, dim: int, tap: str | int = -1
) -> np.ndarray:
    """Spatial map of a single activation dimension.

    Args:
        result: An encoded chip.
        dim: Which dimension of the token vector to read.
        tap: Tap label or index. Defaults to the deepest tap.

    Returns:
        ``(H', W')`` float32 array of that dimension's value at each patch.

    Raises:
        IndexError: If ``dim`` is out of range for the token width.
    """
    grid = result.grid(tap)
    if not 0 <= dim < grid.shape[-1]:
        raise IndexError(
            f"dimension {dim} out of range for token width {grid.shape[-1]}"
        )
    return grid[:, :, dim]


def top_patches(
    result: EmbeddingResult,
    dim: int,
    k: int = 10,
    tap: str | int = -1,
    *,
    lowest: bool = False,
) -> pd.DataFrame:
    """The patches where one dimension is most (or least) active.

    Args:
        result: An encoded chip.
        dim: Which dimension to rank by.
        k: How many patches to return. Clipped to the number available.
        tap: Tap label or index.
        lowest: Return the least-active patches instead of the most-active.

    Returns:
        DataFrame with columns ``rank``, ``row``, ``col``, ``token_index``,
        ``value``, ordered strongest first.
    """
    values = dimension_map(result, dim, tap)
    height, width = values.shape
    flat = values.reshape(-1)
    k = int(min(max(k, 0), flat.size))
    order = np.argsort(flat)
    picks = order[:k] if lowest else order[::-1][:k]

    return pd.DataFrame(
        {
            "rank": np.arange(1, len(picks) + 1),
            "row": picks // width,
            "col": picks % width,
            "token_index": picks,
            "value": flat[picks],
        }
    )


def most_variable_dims(
    result: EmbeddingResult, k: int = 20, tap: str | int = -1
) -> pd.DataFrame:
    """Rank dimensions by how much they vary across the chip.

    With 128 dimensions and no prior about which matter, spatial variance is the
    cheapest useful filter: a dimension that is flat across the whole chip is
    not telling you anything about *this* place, whatever its absolute value.

    The raw range is reported alongside the standard deviation deliberately. Any
    per-dimension rescaling for display will stretch a nearly-flat dimension to
    full contrast and make noise look like structure, so the range is what tells
    you whether a dramatic-looking map is real.

    Args:
        result: An encoded chip.
        k: How many dimensions to return. Clipped to the token width.
        tap: Tap label or index.

    Returns:
        DataFrame with columns ``dim``, ``std``, ``range``, ``min``, ``max``,
        ``mean``, sorted by ``std`` descending.
    """
    tokens = result.tokens(tap)
    k = int(min(max(k, 0), tokens.shape[1]))
    stats = pd.DataFrame(
        {
            "dim": np.arange(tokens.shape[1]),
            "std": tokens.std(axis=0),
            "range": tokens.max(axis=0) - tokens.min(axis=0),
            "min": tokens.min(axis=0),
            "max": tokens.max(axis=0),
            "mean": tokens.mean(axis=0),
        }
    )
    return (
        stats.sort_values("std", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )


def project_onto(
    result: EmbeddingResult,
    direction: np.ndarray,
    tap: str | int = -1,
    *,
    unit_normalize: bool = True,
) -> np.ndarray:
    """Map each patch's projection onto an arbitrary direction.

    This is the hook for later supervised work: hand it a fitted ridge weight
    vector and it returns the per-patch strength of whatever that vector encodes,
    ready to be thresholded for max-activating exemplars.

    Args:
        result: An encoded chip.
        direction: ``(D,)`` vector in the same space as the chosen tap.
        tap: Tap label or index.
        unit_normalize: Scale ``direction`` to unit length first, so projections
            are comparable across different directions. Turn off if the vector's
            magnitude is itself meaningful.

    Returns:
        ``(H', W')`` float32 array of projections.

    Raises:
        ValueError: If ``direction`` is not 1-D of the right width, or is zero.
    """
    grid = result.grid(tap)
    vec = np.asarray(direction, dtype=np.float32).reshape(-1)
    if vec.shape[0] != grid.shape[-1]:
        raise ValueError(
            f"direction has length {vec.shape[0]} but tap {tap!r} has token "
            f"width {grid.shape[-1]}"
        )
    if unit_normalize:
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            raise ValueError("direction is the zero vector; cannot normalize")
        vec = vec / norm
    return grid @ vec


def depth_trajectory(
    result: EmbeddingResult, row: int, col: int
) -> np.ndarray:
    """How one patch's vector changes with depth.

    Args:
        result: An encoded chip.
        row: Patch row in the token grid.
        col: Patch column in the token grid.

    Returns:
        ``(n_taps, D)`` float32 array, shallow tap first.

    Raises:
        IndexError: If the coordinates are outside the token grid.
    """
    height, width = result.grid_shape
    if not (0 <= row < height and 0 <= col < width):
        raise IndexError(
            f"patch ({row}, {col}) is outside the {height}x{width} token grid"
        )
    return result.activations[:, row, col, :]


def depth_drift(result: EmbeddingResult) -> pd.DataFrame:
    """Cosine similarity between consecutive taps, averaged over patches.

    A cheap read on where in the network the representation actually changes. If
    consecutive taps are near-identical, that depth is doing little, and a flat
    R²-versus-depth curve later on will have an obvious explanation.

    Caveat: for hook-derived taps this is confounded by the changing token norms
    described in :mod:`olmoearth_activations.encode`. Cosine is scale-invariant
    per patch, so it is the least-bad choice, but prefer ``token_exit`` taps.

    Args:
        result: An encoded chip.

    Returns:
        DataFrame with columns ``from_tap``, ``to_tap``, ``mean_cosine``,
        ``min_cosine``, ``max_cosine``.
    """
    rows: list[dict[str, object]] = []
    for i in range(len(result.tap_labels) - 1):
        a = result.activations[i].reshape(-1, result.embed_dim)
        b = result.activations[i + 1].reshape(-1, result.embed_dim)
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cos = np.where(denom > 0, (a * b).sum(axis=1) / np.maximum(denom, 1e-12), np.nan)
        rows.append(
            {
                "from_tap": result.tap_labels[i],
                "to_tap": result.tap_labels[i + 1],
                "mean_cosine": float(np.nanmean(cos)),
                "min_cosine": float(np.nanmin(cos)),
                "max_cosine": float(np.nanmax(cos)),
            }
        )
    return pd.DataFrame(rows)


def pool(
    result: EmbeddingResult, tap: str | int = -1, how: PoolHow = "mean"
) -> np.ndarray:
    """Reduce a tap's patches to one chip-level vector.

    Pooling lives here rather than in ``encode`` on purpose: the encoder keeps
    every patch tied to its location, and collapsing that is an analysis choice
    the caller should have to make explicitly.

    Args:
        result: An encoded chip.
        tap: Tap label or index.
        how: One of ``mean``, ``median``, ``min``, ``max``, ``std``. These are
            the same five statistics the production tract-aggregation step
            computes, so a pooled vector here is comparable to a tract feature.

    Returns:
        ``(D,)`` float32 array.

    Raises:
        ValueError: On an unknown pooling statistic.
    """
    tokens = result.tokens(tap)
    funcs = {
        "mean": np.mean,
        "median": np.median,
        "min": np.min,
        "max": np.max,
        "std": np.std,
    }
    if how not in funcs:
        raise ValueError(
            f"unknown pooling statistic {how!r}; expected one of {sorted(funcs)}"
        )
    return np.asarray(funcs[how](tokens, axis=0), dtype=np.float32)


def pool_all(
    result: EmbeddingResult, tap: str | int = -1
) -> pd.DataFrame:
    """All five pooling statistics for one tap, as a tidy one-row frame.

    Column names follow the production aggregation schema, ``{DIM}_{STAT}``, so
    these can be concatenated with tract-level features without renaming.

    Args:
        result: An encoded chip.
        tap: Tap label or index.

    Returns:
        Single-row DataFrame with ``D * 5`` columns.
    """
    stats: dict[str, float] = {}
    for how in ("mean", "median", "min", "max", "std"):
        vec = pool(result, tap, how)  # type: ignore[arg-type]
        suffix = {"min": "MINIMUM", "max": "MAXIMUM"}.get(how, how.upper())
        for i, value in enumerate(vec):
            stats[f"D{i:03d}_{suffix}"] = float(value)
    return pd.DataFrame([stats])


def compare_locations(
    results: Sequence[EmbeddingResult],
    tap: str | int = -1,
    *,
    how: PoolHow = "mean",
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Cosine similarity between the pooled vectors of several chips.

    The intended use is the urban-versus-rural contrast: encode a handful of
    chips, and read off whether the model separates them at all before doing
    anything more elaborate.

    Args:
        results: Encoded chips.
        tap: Tap label or index.
        how: Pooling statistic to compare on.
        labels: Row/column labels. Defaults to ``name`` from each result's
            metadata, else its lat/lon, else its position.

    Returns:
        Square DataFrame of cosine similarities.

    Raises:
        ValueError: If fewer than two results are given, or token widths differ.
    """
    if len(results) < 2:
        raise ValueError(
            f"compare_locations needs at least two results, got {len(results)}"
        )

    vectors = [pool(r, tap, how) for r in results]
    widths = {v.shape[0] for v in vectors}
    if len(widths) != 1:
        raise ValueError(
            f"results have differing token widths {sorted(widths)}; they are "
            f"not comparable"
        )

    if labels is None:
        labels = [_result_label(r, i) for i, r in enumerate(results)]

    matrix = np.stack(vectors, axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.maximum(norms, 1e-12)
    sim = unit @ unit.T
    return pd.DataFrame(sim, index=list(labels), columns=list(labels))


def _result_label(result: EmbeddingResult, index: int) -> str:
    """Best available human-readable label for a result."""
    name = result.meta.get("name")
    if name:
        return str(name)
    lat, lon = result.meta.get("lat"), result.meta.get("lon")
    if lat is not None and lon is not None:
        return f"{float(lat):.4f},{float(lon):.4f}"
    return f"chip{index}"
