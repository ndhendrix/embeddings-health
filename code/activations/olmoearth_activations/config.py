"""Frozen configuration objects for the OlmoEarth activation toolkit.

Every number that can change a token value lives here. Nothing elsewhere in the
package may hard-code a chip size, a patch size, a date, a normalization
constant, or a device -- if you find yourself wanting to, add a field instead.

Configs are frozen dataclasses so a resolved config can be hashed and written
into a run manifest, and so no downstream code can mutate the settings a result
was computed under.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

NormalizeStrategy = Literal["computed", "predefined", "none"]
TapMethod = Literal["token_exit", "hooks"]
ExtraDimPolicy = Literal["squeeze", "mean"]


@dataclass(frozen=True)
class ModelConfig:
    """Which checkpoint to load and where to run it.

    Attributes:
        model_id: An ``olmoearth_pretrain.model_loader.ModelID`` *value* string,
            e.g. ``"OlmoEarth-v1_2-Nano"``. Resolved to an enum member by
            :mod:`olmoearth_activations.loader`; see that module for what
            happens when the installed package predates the member.
        hf_revision: HuggingFace commit SHA to pin the weights to. Strongly
            recommended: without it, ``main`` moves and the numbers move with
            it. ``None`` means "whatever ``main`` is right now".
        package_commit: The ``olmoearth_pretrain`` git commit installed in this
            environment. Not used for loading -- recorded in the run manifest so
            a result can be traced back to a package version. The library gives
            us no reliable way to introspect this, so it is supplied by hand.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
        local_path: If set, load from this directory (expects ``config.json``
            and ``weights.pth``) instead of from HuggingFace. Takes precedence
            over ``model_id``.
    """

    model_id: str = "OlmoEarth-v1_2-Nano"
    hf_revision: str | None = None
    package_commit: str | None = None
    device: str = "auto"
    local_path: Path | None = None


@dataclass(frozen=True)
class TileConfig:
    """How imagery is turned into a model input array.

    Attributes:
        chip_px: Side length in pixels of the square chip fed to the encoder.
            Must be divisible by ``patch_px``. Note that chip size is a real
            modelling choice, not just a batching detail: a token near a chip
            edge has less surrounding context than one at the centre, so
            changing this changes token values.
        patch_px: Pixels per patch. 4 at 10 m input gives 40 m tokens. The
            checkpoint's ``max_patch_size`` bounds this above.
        date: ``(day, month, year)`` passed as the sample timestamp. The month
            encoding is additive, so for a median composite with no real month
            this is a constant offset on every token -- harmless for relative
            comparisons, but it must match whatever produced the embeddings you
            are comparing against.
        normalize: Which ``Normalizer`` strategy to apply, or ``"none"`` to pass
            raw values through. ``"computed"`` matches what the pretraining
            dataset pipeline applies and is the right default.
        input_resolution_m: Ground sample distance of the input imagery, metres.
            Recorded in metadata; the model's own default is used for the
            forward pass unless this differs from it.
    """

    chip_px: int = 64
    patch_px: int = 4
    date: tuple[int, int, int] = (15, 6, 2022)
    normalize: NormalizeStrategy = "computed"
    input_resolution_m: int = 10

    def __post_init__(self) -> None:
        """Validate the chip/patch relationship early, not at forward time."""
        if self.chip_px <= 0 or self.patch_px <= 0:
            raise ValueError(
                f"chip_px and patch_px must be positive, got "
                f"chip_px={self.chip_px}, patch_px={self.patch_px}"
            )
        if self.chip_px % self.patch_px != 0:
            raise ValueError(
                f"chip_px must be divisible by patch_px so the token grid is "
                f"square and complete, got chip_px={self.chip_px}, "
                f"patch_px={self.patch_px} (remainder "
                f"{self.chip_px % self.patch_px})"
            )
        if len(self.date) != 3:
            raise ValueError(
                f"date must be a (day, month, year) triple, got {self.date!r}"
            )

    @property
    def expected_grid_side(self) -> int:
        """Token grid side length implied by the chip and patch sizes.

        This is a *cross-check* only. The authoritative grid shape comes from
        the encoder's own output; see :mod:`olmoearth_activations.encode`.
        """
        return self.chip_px // self.patch_px


@dataclass(frozen=True)
class TapConfig:
    """Which intermediate representations to capture, and how.

    Attributes:
        method: ``"token_exit"`` uses the encoder's ``token_exit_cfg`` argument,
            one forward pass per depth, and returns representations that have
            all passed the encoder's final LayerNorm -- so they are on a common
            scale. ``"hooks"`` registers forward hooks on the encoder blocks and
            gets every depth from a single pass, but captures the residual
            stream *before* that final LayerNorm, so the taps are not on a
            common scale and the deepest tap is not identical to the encoder's
            returned tokens.
        depths: Which depths to capture. ``0`` is the patch projection before
            positional/month/modality encodings; ``k >= 1`` is after block ``k``.
            ``None`` means every depth from 0 to the model's depth inclusive.
            Ignored -- with a warning -- for ``method="hooks"``, which can only
            produce depths 1..depth.
        fast_pass: Passed through to the encoder. A performance switch only: in
            eval mode with an all-ONLINE_ENCODER mask it does not change the
            numbers. Leave it True and let the test suite hold that claim
            honest.
        extra_dims: What to do with the time and band-set axes of the returned
            token grid. ``"squeeze"`` requires them to be singleton and drops
            them, which is the honest choice for a single-timestep,
            single-band-set configuration. ``"mean"`` averages them, matching
            what the production embedding pipeline does.
    """

    method: TapMethod = "token_exit"
    depths: tuple[int, ...] | None = None
    fast_pass: bool = True
    extra_dims: ExtraDimPolicy = "squeeze"


@dataclass(frozen=True)
class RunConfig:
    """The three configs together, plus run-level bookkeeping.

    Attributes:
        model: Checkpoint and device settings.
        tile: Imagery-to-array settings.
        tap: Activation capture settings.
        save_dtype: ``"float32"`` or ``"float16"``. In-memory arrays are always
            float32; this applies only when writing to disk.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    tile: TileConfig = field(default_factory=TileConfig)
    tap: TapConfig = field(default_factory=TapConfig)
    save_dtype: Literal["float32", "float16"] = "float32"

    # ---------------------------------------------------------------- io

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Load a config from a YAML file.

        Unknown keys are rejected rather than ignored -- a typo in a config file
        that silently leaves a default in place is exactly the kind of bug that
        is invisible until the numbers are already in a manuscript.

        Args:
            path: Path to a YAML file with optional ``model``, ``tile``, ``tap``
                and ``save_dtype`` keys.

        Returns:
            The resolved config.

        Raises:
            ValueError: If the file contains keys that do not map to fields.
        """
        with Path(path).open() as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunConfig:
        """Build a config from a nested plain dict. See :meth:`from_yaml`."""
        sections: dict[str, Any] = {
            "model": ModelConfig,
            "tile": TileConfig,
            "tap": TapConfig,
        }
        unknown = set(raw) - set(sections) - {"save_dtype"}
        if unknown:
            raise ValueError(
                f"unknown top-level config keys {sorted(unknown)}; "
                f"expected any of {sorted(set(sections) | {'save_dtype'})}"
            )

        built: dict[str, Any] = {}
        for name, klass in sections.items():
            section = raw.get(name) or {}
            if not isinstance(section, dict):
                raise ValueError(f"config section '{name}' must be a mapping")
            valid = {f for f in klass.__dataclass_fields__}
            bad = set(section) - valid
            if bad:
                raise ValueError(
                    f"unknown keys in config section '{name}': {sorted(bad)}; "
                    f"expected any of {sorted(valid)}"
                )
            # YAML gives lists where the dataclasses want tuples, and strings
            # where ModelConfig wants a Path.
            coerced = dict(section)
            for key in ("date", "depths"):
                if isinstance(coerced.get(key), list):
                    coerced[key] = tuple(coerced[key])
            if name == "model" and coerced.get("local_path") is not None:
                coerced["local_path"] = Path(coerced["local_path"])
            built[name] = klass(**coerced)

        if "save_dtype" in raw:
            built["save_dtype"] = raw["save_dtype"]
        return cls(**built)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable nested dict of every setting."""

        def _plain(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return list(value)
            return value

        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict):
                out[key] = {k: _plain(v) for k, v in value.items()}
            else:
                out[key] = _plain(value)
        return out

    def to_yaml(self, path: str | Path) -> None:
        """Write this config to a YAML file."""
        with Path(path).open("w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    # ---------------------------------------------------------------- misc

    def fingerprint(self) -> str:
        """Short stable hash of every setting, for tagging outputs.

        Two results with the same fingerprint were computed under identical
        settings. Two with different fingerprints were not, and are not
        comparable without thought.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def with_(self, **changes: Any) -> RunConfig:
        """Return a copy with top-level fields replaced."""
        return replace(self, **changes)
