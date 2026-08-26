"""Embedding and intermediate-activation extraction for OlmoEarth models.

The package is organised around one idea: embeddings and activations are two
views of a single forward pass, so :meth:`~olmoearth_activations.encode.Encoder.encode`
returns both, and every patch stays tied to its location until an analysis
function is explicitly asked to pool it away.

Typical use::

    from olmoearth_activations import (
        Encoder, OlmoEarthModel, RunConfig, SafeSceneSource, analysis,
        encode_location,
    )

    cfg = RunConfig.from_yaml("configs/default.yaml")
    model = OlmoEarthModel.load(cfg.model)
    source = SafeSceneSource(scene_dir="path/to/scenes")
    result = encode_location(model, source, 38.9032, -77.0370, cfg)

    print(analysis.most_variable_dims(result, k=10))

Lazy imports
------------
Top-level names resolve on first access (PEP 562) rather than at import time.
That is deliberate rather than fussy: the heavy dependencies split cleanly by
task. Producing activations needs torch and a checkpoint; *analysing* saved
``.npz`` results needs neither, and configuration needs only the standard
library. Eager imports would force every consumer to satisfy the union, so
``import olmoearth_activations.config`` would fail on a machine without numpy --
which is exactly the kind of friction that stops people from reading their own
results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

#: Maps each public name to the submodule that defines it.
_EXPORTS: dict[str, str] = {
    # config -- standard library plus PyYAML only
    "ModelConfig": "config",
    "RunConfig": "config",
    "TapConfig": "config",
    "TileConfig": "config",
    # encode -- numpy; torch only when a forward pass actually runs
    "EmbeddingResult": "encode",
    "Encoder": "encode",
    "encode_location": "encode",
    "tap_label": "encode",
    # loader -- torch, and olmoearth_pretrain when loading
    "OlmoEarthModel": "loader",
    "resolve_device": "loader",
    # locations -- standard library plus PyYAML only
    "Location": "locations",
    "load_locations": "locations",
    "scene_dir_for": "locations",
    # tiles -- rasterio
    "RegionReader": "tiles",
    "SafeSceneSource": "tiles",
    "TileSource": "tiles",
    "default_scene_dir": "tiles",
    "load_chip": "tiles",
    "normalize_chip": "tiles",
    # analysis -- numpy and pandas
    "compare_locations": "analysis",
    "depth_drift": "analysis",
    "depth_trajectory": "analysis",
    "dimension_map": "analysis",
    "most_variable_dims": "analysis",
    "pool": "analysis",
    "pool_all": "analysis",
    "project_onto": "analysis",
    "top_patches": "analysis",
}

#: Submodules reachable as attributes, e.g. ``oea.analysis``.
_SUBMODULES = frozenset(
    {"analysis", "config", "encode", "loader", "locations", "tiles", "viz"}
)

if TYPE_CHECKING:  # pragma: no cover -- for type checkers and IDEs only
    from olmoearth_activations import (
        analysis,
        config,
        encode,
        loader,
        locations,
        tiles,
        viz,
    )
    from olmoearth_activations.locations import (
        Location,
        load_locations,
        scene_dir_for,
    )
    from olmoearth_activations.analysis import (
        compare_locations,
        depth_drift,
        depth_trajectory,
        dimension_map,
        most_variable_dims,
        pool,
        pool_all,
        project_onto,
        top_patches,
    )
    from olmoearth_activations.config import (
        ModelConfig,
        RunConfig,
        TapConfig,
        TileConfig,
    )
    from olmoearth_activations.encode import (
        EmbeddingResult,
        Encoder,
        encode_location,
        tap_label,
    )
    from olmoearth_activations.loader import OlmoEarthModel, resolve_device
    from olmoearth_activations.tiles import (
        RegionReader,
        SafeSceneSource,
        TileSource,
        default_scene_dir,
        load_chip,
        normalize_chip,
    )


def __getattr__(name: str) -> Any:
    """Resolve a public name on first access.

    Args:
        name: The attribute being looked up.

    Returns:
        The submodule or the object it defines.

    Raises:
        AttributeError: If the name is not part of the public API.
    """
    import importlib

    if name in _SUBMODULES:
        return importlib.import_module(f"olmoearth_activations.{name}")
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            f"module 'olmoearth_activations' has no attribute {name!r}"
        )
    module = importlib.import_module(f"olmoearth_activations.{module_name}")
    return getattr(module, name)


def __dir__() -> list[str]:
    """List the public API, for tab completion."""
    return sorted({*_EXPORTS, *_SUBMODULES, "__version__"})


__all__ = [*sorted(_EXPORTS), *sorted(_SUBMODULES), "__version__"]
