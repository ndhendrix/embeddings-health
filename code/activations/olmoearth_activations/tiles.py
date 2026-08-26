"""Reading imagery into model-ready arrays.

The reading logic here is ported essentially unchanged from the prototype
``extract_activations.load_tile`` and ``sweep_patch.load_region``. Three of its
decisions are correct and load-bearing, and are preserved deliberately:

1. **Band order comes from the modality spec, never from a glob.** Sentinel-2
   stores one file per band and the model's expected order is not alphabetical.
   Getting it wrong raises no error, it just produces wrong numbers.
2. **Band B02 defines the reference grid.** It is 10 m/px, the finest available,
   and every other band is resampled onto it. The model expects all bands at a
   uniform ground sample distance.
3. **The window is checked against the scene bounds before reading**, with an
   error that tells the caller what to do about it.

Normalization is stateless
--------------------------
Both ``Strategy.COMPUTED`` and ``Strategy.PREDEFINED`` use fixed per-band
constants shipped in the package's ``norm_configs/*.json``; neither derives
anything from the array it is given. So normalizing a large region once and
slicing chips out of it is *numerically identical* to normalizing each chip
separately. The prototype's ``sweep_patch.normalize_region`` docstring worries
that per-chip normalization could manufacture a fake edge effect -- that worry
is unfounded, and it is recorded here so nobody re-derives it.
"""

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import rasterio
import rasterio.windows
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as warp_transform

from olmoearth_activations.config import NormalizeStrategy, TileConfig

logger = logging.getLogger(__name__)


def default_band_order() -> list[str]:
    """Return the Sentinel-2 L2A band order the model expects.

    Imported lazily so that array-only code paths -- and the synthetic-fixture
    tests -- do not require ``olmoearth_pretrain`` to be installed.
    """
    from olmoearth_pretrain.data.constants import Modality

    return list(Modality.SENTINEL2_L2A.band_order)


def default_scene_dir() -> Path:
    """Return the repo-local directory to look for ``*.SAFE`` scenes in.

    Resolves to ``<repo>/data/scenes``. The alternative -- defaulting to the
    working directory -- means the same command looks somewhere different
    depending on where it was launched from, and puts imagery outside the
    project entirely when a caller passes a path like ``~/scenes``.

    ``data/`` is already gitignored, which matters here: one Sentinel-2 L2A
    scene is roughly a gigabyte and must never reach a commit.

    Prefer ``locations.scene_dir_for(name)`` for a named location: one scene per
    subdirectory keeps the band glob unambiguous, which this flat default
    cannot.

    Returns:
        ``<repo>/data/scenes``, or ``scenes`` relative to the working directory
        if this file is not inside a repository.
    """
    from olmoearth_activations.locations import scenes_root

    return scenes_root()


@runtime_checkable
class TileSource(Protocol):
    """Anything that can produce a chip centred on a coordinate.

    Implementations return ``(H, W, C)`` float arrays with ``C`` channels in the
    model's band order and raw (un-normalized) values. Keeping normalization out
    of the source means a source can be swapped -- a .SAFE scene now, an annual
    median composite later -- without touching the rest of the package.
    """

    def read(self, lat: float, lon: float, size_px: int) -> np.ndarray:
        """Read a ``size_px`` square chip centred on ``(lat, lon)``."""
        ...


@dataclass
class SafeSceneSource:
    """Reads chips from a Sentinel-2 ``.SAFE`` scene directory.

    Attributes:
        scene_dir: Directory containing one or more ``*.SAFE`` folders. Defaults
            to the current working directory, matching the prototype's
            behaviour.
        band_order: Channel order to read. Defaults to the modality spec's.
    """

    scene_dir: Path = Path()
    band_order: Sequence[str] | None = None

    def __post_init__(self) -> None:
        """Resolve the band order once, at construction."""
        self._bands: list[str] = list(self.band_order or default_band_order())

    # ------------------------------------------------------------ helpers

    def _band_paths(self) -> list[str]:
        """Locate one .jp2 per band, in band order.

        Raises:
            FileNotFoundError: If any band has no matching file.
        """
        paths: list[str] = []
        for band_name in self._bands:
            pattern = str(
                Path(self.scene_dir)
                / "*.SAFE"
                / "GRANULE"
                / "*"
                / "IMG_DATA"
                / "*"
                / f"*_{band_name}_*.jp2"
            )
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(
                    f"no file found for band {band_name} under "
                    f"{Path(self.scene_dir).resolve()}. Expected a .SAFE folder "
                    f"matching: {pattern}"
                )
            if len(matches) > 1:
                logger.warning(
                    "band %s matched %d files, using %s",
                    band_name,
                    len(matches),
                    matches[0],
                )
            paths.append(matches[0])
        return paths

    def _read_window(
        self, paths: Sequence[str], window: rasterio.windows.Window
    ) -> np.ndarray:
        """Read one window from every band, resampled onto B02's grid.

        Returns:
            ``(H, W, C)`` float32 array in band order.
        """
        with rasterio.open(paths[0]) as ref:
            crs, transform = ref.crs, ref.transform
            width, height = ref.width, ref.height

        out = np.zeros(
            (len(paths), int(window.height), int(window.width)), dtype=np.float32
        )
        for i, path in enumerate(paths):
            with rasterio.open(path) as src, WarpedVRT(
                src,
                crs=crs,
                transform=transform,
                width=width,
                height=height,
                resampling=Resampling.bilinear,
            ) as vrt:
                # Read only the window, not the whole scene.
                out[i] = vrt.read(1, window=window)
        return np.transpose(out, (1, 2, 0))

    def _locate(self, lat: float, lon: float) -> tuple[int, int, int, int]:
        """Return ``(row, col, scene_height, scene_width)`` for a coordinate."""
        paths = self._band_paths()
        with rasterio.open(paths[0]) as ref:
            xs, ys = warp_transform("EPSG:4326", ref.crs, [lon], [lat])
            row, col = ref.index(xs[0], ys[0])
            return int(row), int(col), ref.height, ref.width

    # --------------------------------------------------------------- api

    def read(self, lat: float, lon: float, size_px: int) -> np.ndarray:
        """Read a square chip centred on a coordinate.

        Args:
            lat: Latitude in degrees (EPSG:4326).
            lon: Longitude in degrees (EPSG:4326).
            size_px: Side length in pixels on B02's 10 m grid.

        Returns:
            ``(size_px, size_px, C)`` float32 array of raw band values.

        Raises:
            ValueError: If the requested window falls outside the scene.
        """
        paths = self._band_paths()
        row, col, height, width = self._locate(lat, lon)
        row0, col0 = row - size_px // 2, col - size_px // 2
        _check_window(row0, col0, size_px, size_px, height, width)
        window = rasterio.windows.Window(col0, row0, size_px, size_px)
        chip = self._read_window(paths, window)
        logger.info(
            "read %dx%d chip at (%.5f, %.5f) -> shape %s",
            size_px,
            size_px,
            lat,
            lon,
            chip.shape,
        )
        return chip


@dataclass
class RegionReader:
    """Reads a rectangular region with explicit offset placement.

    This is what sliding-window experiments need: rather than centring the read
    on a coordinate, it places the coordinate at a caller-chosen offset inside
    the region, so chips can then be sliced out of the region at controlled
    positions. Ported from ``sweep_patch.load_region``.

    Attributes:
        source: The scene to read from.
    """

    source: SafeSceneSource

    def read_region(
        self,
        lat: float,
        lon: float,
        height_px: int,
        width_px: int,
        row_offset: int,
        col_offset: int,
    ) -> RegionResult:
        """Read a region positioned so ``(lat, lon)`` lands at a given offset.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            height_px: Region height in pixels.
            width_px: Region width in pixels.
            row_offset: Row within the region at which ``lat`` should land.
            col_offset: Column within the region at which ``lon`` should land.

        Returns:
            The region array plus the offsets needed to interpret it.

        Raises:
            ValueError: If the region falls outside the scene.
        """
        paths = self.source._band_paths()
        row_t, col_t, scene_h, scene_w = self.source._locate(lat, lon)
        row0, col0 = row_t - row_offset, col_t - col_offset
        _check_window(row0, col0, height_px, width_px, scene_h, scene_w)
        window = rasterio.windows.Window(col0, row0, width_px, height_px)
        array = self.source._read_window(paths, window)
        logger.info("read region shape %s", array.shape)
        return RegionResult(
            array=array,
            origin_rowcol=(row0, col0),
            target_pixel_rowcol=(row_t, col_t),
        )


@dataclass
class RegionResult:
    """A region read plus the geometry needed to slice chips from it.

    Attributes:
        array: ``(H, W, C)`` raw band values.
        origin_rowcol: Scene pixel coordinates of the region's top-left corner.
        target_pixel_rowcol: Scene pixel coordinates of the requested point.
    """

    array: np.ndarray
    origin_rowcol: tuple[int, int]
    target_pixel_rowcol: tuple[int, int]


def _check_window(
    row0: int,
    col0: int,
    height_px: int,
    width_px: int,
    scene_h: int,
    scene_w: int,
) -> None:
    """Raise if a window is not fully inside the scene.

    Raises:
        ValueError: With the requested and available extents, so the caller can
            tell whether to move the point or download another scene.
    """
    fits = (
        0 <= row0
        and 0 <= col0
        and row0 + height_px <= scene_h
        and col0 + width_px <= scene_w
    )
    if not fits:
        raise ValueError(
            f"the {height_px}x{width_px} px window at (row={row0}, col={col0}) "
            f"falls outside this scene, which is {scene_h}x{scene_w} px. Pick a "
            f"location further from the scene edge, or download a scene that "
            f"covers it."
        )


# ------------------------------------------------------------- normalization


def normalize_chip(
    chip: np.ndarray,
    strategy: NormalizeStrategy,
    *,
    std_multiplier: float = 2.0,
) -> np.ndarray:
    """Apply the model's normalization to a raw chip.

    Both strategies use fixed per-band constants and are stateless, so this is
    safe to apply to a whole region before slicing (see the module docstring).

    Args:
        chip: Array whose last axis is channels in the model's band order.
        strategy: ``"computed"`` (matches the pretraining dataset pipeline),
            ``"predefined"`` (min/max scaling), or ``"none"`` (passthrough).
        std_multiplier: Only used by ``"computed"``. The library default is 2,
            giving ``(x - (mean - 2*std)) / (4*std)``.

    Returns:
        float32 array of the same shape.

    Raises:
        ValueError: On an unknown strategy.
    """
    if strategy == "none":
        # Passthrough is provided so the production pipeline's (unnormalized)
        # input path can be reproduced deliberately. It is off-distribution for
        # the model; do not make it the default.
        logger.warning(
            "normalize='none': feeding raw values to a model trained on "
            "normalized input. This reproduces the production embedding "
            "pipeline but is off-distribution."
        )
        return chip.astype(np.float32, copy=False)

    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.data.normalize import Normalizer, Strategy

    strategies = {
        "computed": Strategy.COMPUTED,
        "predefined": Strategy.PREDEFINED,
    }
    if strategy not in strategies:
        raise ValueError(
            f"unknown normalize strategy {strategy!r}; expected one of "
            f"{sorted([*strategies, 'none'])}"
        )
    normalizer = Normalizer(strategies[strategy], std_multiplier=std_multiplier)
    out = normalizer.normalize(Modality.SENTINEL2_L2A, chip)
    return np.asarray(out, dtype=np.float32)


def load_chip(
    source: TileSource, lat: float, lon: float, cfg: TileConfig
) -> np.ndarray:
    """Read and normalize a chip in one step.

    Args:
        source: Where to read from.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        cfg: Chip size and normalization settings.

    Returns:
        ``(chip_px, chip_px, C)`` float32 normalized array.
    """
    raw = source.read(lat, lon, cfg.chip_px)
    return normalize_chip(raw, cfg.normalize)
