"""Figures for per-patch activations.

``matplotlib`` only, with the Agg backend selected before ``pyplot`` is
imported so these work headless on a cluster node. Every function accepts an
optional ``ax`` and returns the figure, so panels can be composed.

The two-panel idiom
-------------------
:func:`plot_feature_matrix` reproduces the prototype's raw-versus-column-normalized
pair, and keeps its warning, which is worth restating: rescaling each dimension
to its own min/max makes change *within* a dimension visible, but it also
stretches a dimension that barely moves to full contrast. A dramatic-looking
column may be amplified noise. So the raw per-dimension ranges are always
returned alongside the figure, and the caller should look at them before
believing anything.
"""

from __future__ import annotations

import logging
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from olmoearth_activations import analysis  # noqa: E402
from olmoearth_activations.encode import EmbeddingResult  # noqa: E402

logger = logging.getLogger(__name__)

#: Indices of the true-colour bands within the model's Sentinel-2 band order
#: ``B02 B03 B04 B08 ...``: red is B04, green B03, blue B02.
RGB_BAND_INDICES = (2, 1, 0)


def rgb_composite(
    chip: np.ndarray, *, percentile_clip: tuple[float, float] = (2.0, 98.0)
) -> np.ndarray:
    """Build a display-ready true-colour image from a chip.

    Args:
        chip: ``(H, W, C)`` array in the model's band order.
        percentile_clip: Low/high percentiles for contrast stretching. Satellite
            reflectance is heavily skewed, so a linear stretch on raw values is
            almost always unreadable.

    Returns:
        ``(H, W, 3)`` float array in [0, 1].

    Raises:
        ValueError: If the chip has too few channels to contain B04/B03/B02.
    """
    arr = np.asarray(chip, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] <= max(RGB_BAND_INDICES):
        raise ValueError(
            f"expected an (H, W, C) chip with at least "
            f"{max(RGB_BAND_INDICES) + 1} channels, got shape {arr.shape}"
        )
    rgb = arr[:, :, list(RGB_BAND_INDICES)]
    lo, hi = np.nanpercentile(rgb, percentile_clip)
    if hi <= lo:
        return np.zeros_like(rgb)
    return np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)


def _resolve_ax(ax: Axes | None, **fig_kw: object) -> tuple[Figure, Axes]:
    """Return a (figure, axes) pair, creating one if needed."""
    if ax is not None:
        return ax.get_figure(), ax
    fig, new_ax = plt.subplots(**fig_kw)  # type: ignore[arg-type]
    return fig, new_ax


def plot_dimension_map(
    result: EmbeddingResult,
    dim: int,
    tap: str | int = -1,
    *,
    chip: np.ndarray | None = None,
    ax: Axes | None = None,
    cmap: str = "viridis",
) -> Figure:
    """Heatmap of one activation dimension, optionally beside the imagery.

    Args:
        result: An encoded chip.
        dim: Which dimension to map.
        tap: Tap label or index.
        chip: Optional ``(H, W, C)`` chip. If given, the true-colour composite is
            drawn to the left of the heatmap, which is the only way to judge
            whether an activation pattern corresponds to anything on the ground.
        ax: Draw into this axes instead of creating a figure. Ignored when
            ``chip`` is supplied, since that needs two axes.
        cmap: Matplotlib colormap name.

    Returns:
        The figure.
    """
    values = analysis.dimension_map(result, dim, tap)
    label = result.tap_labels[result.tap_index(tap)]

    if chip is not None:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
        axes[0].imshow(rgb_composite(chip), interpolation="nearest")
        axes[0].set_title("RGB composite (B04/B03/B02)")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        heat_ax = axes[1]
    else:
        fig, heat_ax = _resolve_ax(ax, figsize=(5, 4.2))

    image = heat_ax.imshow(values, cmap=cmap, interpolation="nearest")
    heat_ax.set_title(f"dim {dim} at {label}")
    heat_ax.set_xlabel("patch column")
    heat_ax.set_ylabel("patch row")
    fig.colorbar(image, ax=heat_ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_top_dimensions(
    result: EmbeddingResult,
    k: int = 9,
    tap: str | int = -1,
    *,
    ncols: int = 3,
    cmap: str = "viridis",
) -> tuple[Figure, "object"]:
    """Small multiples of the most spatially variable dimensions.

    Each panel is scaled to its own range, so the panel titles carry the raw
    standard deviation and range -- without them a flat dimension looks as
    structured as a real one.

    Args:
        result: An encoded chip.
        k: How many dimensions to show.
        tap: Tap label or index.
        ncols: Panels per row.
        cmap: Matplotlib colormap name.

    Returns:
        ``(figure, stats)`` where ``stats`` is the DataFrame from
        :func:`~olmoearth_activations.analysis.most_variable_dims`.
    """
    stats = analysis.most_variable_dims(result, k=k, tap=tap)
    n = len(stats)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.1 * ncols, 3.2 * nrows), squeeze=False
    )
    label = result.tap_labels[result.tap_index(tap)]

    for pos, row in enumerate(stats.itertuples(index=False)):
        ax = axes[pos // ncols][pos % ncols]
        values = analysis.dimension_map(result, int(row.dim), tap)
        ax.imshow(values, cmap=cmap, interpolation="nearest")
        ax.set_title(
            f"dim {int(row.dim)}\nsd={row.std:.3g} range={row.range:.3g}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for pos in range(n, nrows * ncols):
        axes[pos // ncols][pos % ncols].axis("off")

    fig.suptitle(f"most spatially variable dimensions at {label}", fontsize=11)
    fig.tight_layout()
    return fig, stats


def plot_depth_comparison(
    result: EmbeddingResult,
    dim: int,
    *,
    shared_scale: bool = True,
    cmap: str = "viridis",
) -> Figure:
    """The same dimension across every tap.

    Args:
        result: An encoded chip.
        dim: Which dimension to follow.
        shared_scale: Use one colour scale across all taps. This is the honest
            default -- it shows that early and late taps may sit at completely
            different magnitudes, which is exactly what you need to know before
            reading a depth sweep. Turn it off to see within-tap structure.
        cmap: Matplotlib colormap name.

    Returns:
        The figure.
    """
    n = len(result.tap_labels)
    maps = [
        analysis.dimension_map(result, dim, tap_i) for tap_i in range(n)
    ]
    vmin = float(min(m.min() for m in maps)) if shared_scale else None
    vmax = float(max(m.max() for m in maps)) if shared_scale else None

    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.3), squeeze=False)
    last = None
    for i, (label, values) in enumerate(zip(result.tap_labels, maps, strict=True)):
        ax = axes[0][i]
        last = ax.imshow(
            values, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax
        )
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    if shared_scale and last is not None:
        fig.colorbar(last, ax=axes[0].tolist(), fraction=0.02, pad=0.02)
    scale_note = "shared colour scale" if shared_scale else "per-tap colour scale"
    fig.suptitle(f"dim {dim} across depth ({scale_note})", fontsize=11)
    fig.tight_layout()
    return fig


def plot_feature_matrix(
    rows: np.ndarray,
    *,
    row_label: str = "position",
    title: str = "",
    cmap: str = "Blues",
) -> tuple[Figure, np.ndarray]:
    """The prototype's two-panel raw-versus-column-normalized idiom.

    The left panel uses one colour scale for every feature. That is the honest
    picture and it is usually hard to read: features sit at very different
    baselines, so the image is dominated by *which* feature a column is rather
    than by anything varying down the rows.

    The right panel scales each feature to its own range across the rows, which
    makes change down a column visible -- the effect usually being looked for.
    The cost is that a feature which barely moves is stretched to full contrast
    too, so a dramatic-looking column may be amplified noise. The raw ranges are
    returned so those columns can be identified.

    Args:
        rows: ``(n_rows, n_features)`` array.
        row_label: Y-axis label.
        title: Figure title.
        cmap: Matplotlib colormap name.

    Returns:
        ``(figure, column_ranges)`` where ``column_ranges`` is ``(n_features,)``.
    """
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D (rows, features) array, got {arr.shape}")

    lo = arr.min(axis=0, keepdims=True)
    hi = arr.max(axis=0, keepdims=True)
    spread = hi - lo
    normed = (arr - lo) / np.where(spread == 0, 1.0, spread)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for ax, data, sub in (
        (axes[0], arr, "raw values, one colour scale for all features"),
        (
            axes[1],
            normed,
            "NORMALISED BY COLUMN: each feature scaled to its own min/max",
        ),
    ):
        image = ax.imshow(data, cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_xlabel("feature dimension")
        ax.set_ylabel(row_label)
        ax.set_title(sub, fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig, spread.ravel()


def plot_location_similarity(
    matrix: "object", *, ax: Axes | None = None, cmap: str = "magma"
) -> Figure:
    """Heatmap of a :func:`~olmoearth_activations.analysis.compare_locations` matrix.

    Args:
        matrix: Square DataFrame of cosine similarities.
        ax: Optional axes.
        cmap: Matplotlib colormap name.

    Returns:
        The figure.
    """
    values = np.asarray(matrix, dtype=np.float32)  # type: ignore[call-overload]
    labels: Sequence[str] = list(getattr(matrix, "index", range(len(values))))
    fig, use_ax = _resolve_ax(ax, figsize=(1.1 * len(labels) + 3, 1.0 * len(labels) + 2))
    image = use_ax.imshow(values, cmap=cmap, interpolation="nearest")
    use_ax.set_xticks(range(len(labels)))
    use_ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    use_ax.set_yticks(range(len(labels)))
    use_ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            use_ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if values[i, j] < values.max() * 0.6 else "black",
            )
    use_ax.set_title("pooled-vector cosine similarity")
    fig.colorbar(image, ax=use_ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def save(fig: Figure, path: str, *, dpi: int = 150) -> str:
    """Write a figure and close it.

    Args:
        fig: The figure.
        path: Output path.
        dpi: Resolution.

    Returns:
        The path written.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path
