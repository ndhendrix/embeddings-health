"""The forward pass: one call, both embeddings and activations.

Embeddings and activations are not two steps. They are two views of the same
forward pass -- the embedding is the encoder's final per-patch output, and the
activations are the per-patch outputs at intermediate depths. So
:meth:`Encoder.encode` returns both, and there is deliberately no "now go get
the activations" second call that would run the model twice.

Two capture mechanisms, with a real difference
----------------------------------------------
``token_exit`` asks the encoder to exit tokens at depth ``k``. Every returned
representation has passed the encoder's final LayerNorm, so the taps are on a
common scale and are directly comparable across depth. It costs one forward pass
per depth, which is nothing for a 4-block, 128-dim model.

``hooks`` registers forward hooks on the encoder blocks and gets every depth
from a single pass. But block outputs are the residual stream *before* the
encoder's final LayerNorm. Two consequences worth stating plainly: the deepest
hooked tap is **not** identical to the encoder's returned tokens, and the taps
are **not** on a common scale, so an R²-versus-depth curve built from hooks is
partly measuring changing token norms. Hooks also cannot produce depth 0, the
patch projection, because there is no block to hang them on.

``token_exit`` is therefore the default. Hooks remain available because they are
the mechanism that does not depend on ``token_exit_cfg`` behaving as documented.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from olmoearth_activations.config import RunConfig, TapConfig, TileConfig
from olmoearth_activations.loader import MODALITY_KEY, OlmoEarthModel

if TYPE_CHECKING:  # pragma: no cover
    import torch

# torch is imported lazily inside the methods that need it. That keeps
# EmbeddingResult, analysis and viz usable -- and testable -- in an environment
# with no torch installed, which matters because saved .npz results are analysed
# far more often than they are produced.

logger = logging.getLogger(__name__)

#: Label used for the pre-attention patch projection (token_exit depth 0).
PROJ_LABEL = "proj"


def tap_label(depth: int) -> str:
    """Human-readable label for a tap depth.

    Args:
        depth: 0 for the patch projection, ``k >= 1`` for "after block k".

    Returns:
        ``"proj"`` for depth 0, otherwise ``"blk{k}"``.
    """
    return PROJ_LABEL if depth == 0 else f"blk{depth}"


@dataclass
class EmbeddingResult:
    """Per-patch embeddings and activations for one chip.

    Attributes:
        embeddings: ``(H', W', D)`` final-layer output, one vector per patch.
        activations: ``(n_taps, H', W', D)`` per-depth outputs, tap-major.
        tap_labels: One label per tap, aligned with ``activations``' first axis.
        tap_depths: The integer depth of each tap, aligned likewise.
        grid_shape: ``(H', W')``.
        meta: Provenance -- coordinates, date, model id, config fingerprint.
    """

    embeddings: np.ndarray
    activations: np.ndarray
    tap_labels: list[str]
    tap_depths: list[int]
    grid_shape: tuple[int, int]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that the arrays and labels agree."""
        if self.embeddings.ndim != 3:
            raise ValueError(
                f"embeddings must be (H', W', D), got shape "
                f"{self.embeddings.shape}"
            )
        if self.activations.ndim != 4:
            raise ValueError(
                f"activations must be (n_taps, H', W', D), got shape "
                f"{self.activations.shape}"
            )
        if len(self.tap_labels) != self.activations.shape[0]:
            raise ValueError(
                f"got {len(self.tap_labels)} tap labels but "
                f"{self.activations.shape[0]} tap arrays"
            )
        if len(self.tap_depths) != len(self.tap_labels):
            raise ValueError(
                f"got {len(self.tap_depths)} tap depths but "
                f"{len(self.tap_labels)} tap labels"
            )
        if tuple(self.grid_shape) != self.embeddings.shape[:2]:
            raise ValueError(
                f"grid_shape {tuple(self.grid_shape)} does not match "
                f"embeddings spatial shape {self.embeddings.shape[:2]}"
            )

    # -------------------------------------------------------------- views

    @property
    def embed_dim(self) -> int:
        """Token width."""
        return int(self.embeddings.shape[-1])

    @property
    def n_patches(self) -> int:
        """Number of patches in the chip."""
        return int(self.grid_shape[0] * self.grid_shape[1])

    def tap_index(self, tap: str | int = -1) -> int:
        """Resolve a tap selector to an index into ``activations``.

        Args:
            tap: A label such as ``"blk4"`` or ``"proj"``, a depth-matching
                integer, or a negative integer indexing from the end.

        Returns:
            An index into the tap axis.

        Raises:
            KeyError: If a label is not present, listing what is available.
            IndexError: If an integer index is out of range.
        """
        if isinstance(tap, str):
            if tap not in self.tap_labels:
                raise KeyError(
                    f"no tap labelled {tap!r}; available taps are "
                    f"{self.tap_labels}"
                )
            return self.tap_labels.index(tap)
        n = len(self.tap_labels)
        idx = tap if tap >= 0 else n + tap
        if not 0 <= idx < n:
            raise IndexError(
                f"tap index {tap} out of range for {n} taps ({self.tap_labels})"
            )
        return idx

    def grid(self, tap: str | int = -1) -> np.ndarray:
        """Spatial ``(H', W', D)`` view of one tap."""
        return self.activations[self.tap_index(tap)]

    def tokens(self, tap: str | int = -1) -> np.ndarray:
        """Flat ``(H'*W', D)`` view of one tap, in row-major grid order.

        Patch ``(row, col)`` is token ``row * W' + col``.
        """
        arr = self.grid(tap)
        return arr.reshape(-1, arr.shape[-1])

    # ---------------------------------------------------------------- io

    def save(self, path: str | Path, *, dtype: str = "float32") -> Path:
        """Write to a ``.npz`` with a stable schema, plus a manifest sidecar.

        Args:
            path: Output ``.npz`` path. A ``<stem>_manifest.json`` is written
                alongside it.
            dtype: ``"float32"`` or ``"float16"``. float16 halves the file at
                the cost of precision; the in-memory arrays stay float32.

        Returns:
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np_dtype = np.float16 if dtype == "float16" else np.float32
        np.savez_compressed(
            path,
            embeddings=self.embeddings.astype(np_dtype),
            activations=self.activations.astype(np_dtype),
            tap_labels=np.array(self.tap_labels, dtype=object),
            tap_depths=np.array(self.tap_depths, dtype=np.int32),
            grid_shape=np.array(self.grid_shape, dtype=np.int32),
            meta=json.dumps(self.meta),
        )
        manifest = path.with_name(f"{path.stem}_manifest.json")
        manifest.write_text(json.dumps(self.meta, indent=2, default=str))
        logger.info("wrote %s and %s", path, manifest)
        return path

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingResult:
        """Read back a result written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=True) as data:
            return cls(
                embeddings=data["embeddings"].astype(np.float32),
                activations=data["activations"].astype(np.float32),
                tap_labels=[str(x) for x in data["tap_labels"].tolist()],
                tap_depths=[int(x) for x in data["tap_depths"].tolist()],
                grid_shape=tuple(int(x) for x in data["grid_shape"]),  # type: ignore[arg-type]
                meta=json.loads(str(data["meta"])),
            )


class Encoder:
    """Runs the forward pass and assembles :class:`EmbeddingResult` objects."""

    def __init__(
        self,
        model: OlmoEarthModel,
        tile_cfg: TileConfig | None = None,
        tap_cfg: TapConfig | None = None,
    ) -> None:
        """Initialize the encoder.

        Args:
            model: A loaded model.
            tile_cfg: Chip/patch/date settings.
            tap_cfg: Which depths to capture and how.
        """
        self.model = model
        self.tile_cfg = tile_cfg or TileConfig()
        self.tap_cfg = tap_cfg or TapConfig()
        self._store: dict[str, torch.Tensor] = {}

    # ----------------------------------------------------------- plumbing

    @property
    def requested_depths(self) -> tuple[int, ...]:
        """The depths to capture, resolved against the model and the method.

        For ``token_exit`` this is 0..depth inclusive by default. For ``hooks``
        it is 1..depth, since there is no block to hook for the projection.
        """
        depth = self.model.depth
        if self.tap_cfg.method == "hooks":
            achievable = tuple(range(1, depth + 1))
        else:
            achievable = tuple(range(0, depth + 1))

        if self.tap_cfg.depths is None:
            return achievable

        wanted = tuple(sorted(set(self.tap_cfg.depths)))
        keep = tuple(d for d in wanted if d in achievable)
        dropped = tuple(d for d in wanted if d not in achievable)
        if dropped:
            logger.warning(
                "dropping requested depths %s: method=%r can only produce %s",
                dropped,
                self.tap_cfg.method,
                achievable,
            )
        if not keep:
            raise ValueError(
                f"none of the requested depths {wanted} are achievable with "
                f"method={self.tap_cfg.method!r}, which supports {achievable}"
            )
        return keep

    def _build_sample(
        self, chip: np.ndarray, latlon: tuple[float, float] | None
    ) -> Any:
        """Wrap a chip in a ``MaskedOlmoEarthSample``.

        Uses the library's own ``from_olmoearthsample`` helper rather than
        hand-building a mask. The helper fills every mask with
        ``MaskValue.ONLINE_ENCODER`` and sizes its last axis from the modality
        spec's band-set count -- a number this package must never hardcode,
        because a checkpoint's ``tokenization_config`` can override the
        tokenizer to a single band group while the spec still reports three.

        Args:
            chip: ``(H, W, C)`` or ``(B, H, W, T, C)`` float array.
            latlon: Optional ``(lat, lon)``. The production embedding pipeline
                passes this; ``latlon`` is absent from the checkpoints'
                ``supported_modality_names``, so it is very likely inert, but it
                is exposed here so the two paths can be compared rather than
                assumed equivalent.

        Returns:
            A ``MaskedOlmoEarthSample`` on the model's device.
        """
        from olmoearth_pretrain.datatypes import (
            MaskedOlmoEarthSample,
            OlmoEarthSample,
        )
        import torch

        arr = np.asarray(chip, dtype=np.float32)
        if arr.ndim == 3:
            # (H, W, C) -> (B=1, H, W, T=1, C)
            arr = arr[None, :, :, None, :]
        if arr.ndim != 5:
            raise ValueError(
                f"chip must be (H, W, C) or (B, H, W, T, C), got shape "
                f"{np.asarray(chip).shape}"
            )

        device = self.model.device
        chip_t = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        n_time = chip_t.shape[3]
        day, month, year = self.tile_cfg.date
        ts = (
            torch.tensor([day, month, year], dtype=torch.long, device=device)
            .view(1, 1, 3)
            .expand(chip_t.shape[0], n_time, 3)
            .contiguous()
        )

        kwargs: dict[str, Any] = {"sentinel2_l2a": chip_t, "timestamps": ts}
        if latlon is not None:
            kwargs["latlon"] = torch.tensor(
                [list(latlon)], dtype=torch.float32, device=device
            ).expand(chip_t.shape[0], 2).contiguous()

        sample = OlmoEarthSample(**kwargs)
        return MaskedOlmoEarthSample.from_olmoearthsample(sample)

    def _forward(self, sample: Any, token_exit_depth: int | None) -> torch.Tensor:
        """Run the encoder once and return the Sentinel-2 token grid.

        Args:
            sample: A masked sample.
            token_exit_depth: If not None, exit tokens at this depth.

        Returns:
            Tensor shaped ``(B, H', W', T, S, D)``.
        """
        # The encoder refuses token_exit_cfg and fast_pass together, so a tap
        # that needs an early exit cannot also take the fast path. fast_pass is
        # a performance switch that is asserted not to change the output
        # (tests/test_encode_integration.py), so dropping it for exiting depths
        # costs time and nothing else -- and the deepest tap, which passes no
        # token_exit_cfg at all, still gets it.
        exiting = token_exit_depth is not None
        kwargs: dict[str, Any] = {
            "patch_size": self.tile_cfg.patch_px,
            "fast_pass": self.tap_cfg.fast_pass and not exiting,
        }
        if exiting:
            kwargs["token_exit_cfg"] = {MODALITY_KEY: token_exit_depth}

        out = self.model.encoder(sample, **kwargs)
        try:
            tokens = out["tokens_and_masks"].sentinel2_l2a
        except (KeyError, TypeError, AttributeError) as exc:
            raise RuntimeError(
                f"could not read out['tokens_and_masks'].sentinel2_l2a from the "
                f"encoder output. Got type {type(out)!r} with keys "
                f"{list(out) if hasattr(out, 'keys') else 'n/a'}. The encoder's "
                f"return contract has changed; update Encoder._forward."
            ) from exc
        if tokens.ndim != 6:
            raise RuntimeError(
                f"expected encoder tokens shaped (B, H', W', T, S, D), got "
                f"{tuple(tokens.shape)}"
            )
        return tokens

    @contextlib.contextmanager
    def _hooks(self) -> Iterator[dict[str, torch.Tensor]]:
        """Register block forward hooks for the duration of the block.

        Hooks are removed on the way out even if the body raises, and the store
        is cleared on entry, so nothing leaks between calls.

        Yields:
            The activation store, keyed by fully-qualified module name.
        """
        import torch

        self._store = {}
        handles = []

        def make_hook(name: str) -> Any:
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                # detach: inference only. cpu: keep accelerator memory free.
                self._store[name] = tensor.detach().to("cpu", torch.float32)

            return hook

        try:
            for name in self.model.block_names:
                handles.append(
                    self.model.submodule(name).register_forward_hook(
                        make_hook(name)
                    )
                )
            logger.debug("registered %d block hooks", len(handles))
            yield self._store
        finally:
            for handle in handles:
                handle.remove()
            self._store = {}

    # ------------------------------------------------------------ reshape

    def _reduce_extra(self, tokens: torch.Tensor) -> np.ndarray:
        """Collapse the time and band-set axes of a ``(B,H',W',T,S,D)`` tensor.

        The spatial axes are never touched -- keeping every patch tied to its
        location is the whole point of this package.

        Args:
            tokens: ``(B, H', W', T, S, D)`` with ``B == 1``.

        Returns:
            ``(H', W', D)`` float32 array.

        Raises:
            ValueError: If T or S is non-singleton and the policy is
                ``"squeeze"``, which refuses to average silently.
        """
        import torch

        if tokens.shape[0] != 1:
            raise ValueError(
                f"expected batch size 1 in _reduce_extra, got "
                f"{tokens.shape[0]}"
            )
        _, h, w, n_time, n_bandsets, dim = tokens.shape
        if n_time == 1 and n_bandsets == 1:
            reduced = tokens[0, :, :, 0, 0, :]
        elif self.tap_cfg.extra_dims == "mean":
            reduced = tokens[0].mean(dim=(2, 3))
        else:
            raise ValueError(
                f"token grid has T={n_time}, S={n_bandsets}, which are not both "
                f"singleton, and extra_dims='squeeze' refuses to average them "
                f"silently. Set TapConfig(extra_dims='mean') to average over "
                f"time and band sets -- which is what the production embedding "
                f"pipeline does -- or use a single-timestep, single-band-set "
                f"input."
            )
        out = reduced.detach().to("cpu", torch.float32).numpy()
        if out.shape != (h, w, dim):
            raise RuntimeError(
                f"internal reshape error: expected {(h, w, dim)}, got "
                f"{out.shape}"
            )
        return out

    def _hook_to_grid(
        self, flat: torch.Tensor, grid: tuple[int, int, int, int, int]
    ) -> np.ndarray:
        """Reshape a hooked ``(B, N, D)`` tensor onto the token grid.

        The grid shape is taken from the encoder's own returned tensor, not
        computed from the chip and patch sizes, so any register tokens or
        padding the model adds show up here as a loud mismatch rather than a
        silent misalignment.

        Args:
            flat: ``(B, N, D)`` hook output.
            grid: ``(H', W', T, S, D)`` from the encoder's returned tokens.

        Returns:
            ``(H', W', D)`` float32 array.
        """
        h, w, n_time, n_bandsets, dim = grid
        expected_n = h * w * n_time * n_bandsets
        if flat.ndim != 3:
            raise RuntimeError(
                f"expected a hooked block output shaped (B, N, D), got "
                f"{tuple(flat.shape)}. Block outputs are bare tensors in the "
                f"versions this package was written against."
            )
        if flat.shape[1] != expected_n:
            raise RuntimeError(
                f"hooked block produced {flat.shape[1]} tokens but the encoder's "
                f"grid implies {expected_n} "
                f"(H'={h}, W'={w}, T={n_time}, S={n_bandsets}). The usual cause "
                f"is register tokens being prepended: check the checkpoint's "
                f"num_register_tokens. Reshaping onto the grid would silently "
                f"misalign every patch, so this is fatal."
            )
        if flat.shape[2] != dim:
            raise RuntimeError(
                f"hooked block width {flat.shape[2]} does not match the "
                f"encoder's token width {dim}"
            )
        reshaped = flat.view(flat.shape[0], h, w, n_time, n_bandsets, dim)
        return self._reduce_extra(reshaped)

    # ---------------------------------------------------------------- api

    def encode(
        self,
        chip: np.ndarray,
        *,
        latlon: tuple[float, float] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> EmbeddingResult:
        """Run one chip and return its embeddings and activations.

        Args:
            chip: ``(H, W, C)`` or ``(B, H, W, T, C)`` normalized float array
                with channels in the model's band order.
            latlon: Optional ``(lat, lon)`` for the sample and the metadata.
            extra_meta: Extra provenance to merge into ``result.meta``.

        Returns:
            The result, with taps ordered shallow to deep.
        """
        import torch

        if self.model.module.training:
            raise RuntimeError(
                "model is in training mode; activations would be "
                "non-deterministic (band dropout is gated on self.training)"
            )
        with torch.no_grad():
            return self._encode_nograd(chip, latlon, extra_meta)

    def _encode_nograd(
        self,
        chip: np.ndarray,
        latlon: tuple[float, float] | None,
        extra_meta: dict[str, Any] | None,
    ) -> EmbeddingResult:
        """Body of :meth:`encode`, run inside ``torch.no_grad()``."""

        sample = self._build_sample(chip, latlon)
        depths = self.requested_depths

        taps: list[np.ndarray] = []
        labels: list[str] = []
        captured: dict[str, torch.Tensor] = {}

        if self.tap_cfg.method == "hooks":
            # One pass total: the hooks fire during it and its return value is
            # the final embedding. Running a second, unhooked pass here would
            # throw away the single advantage hooks have.
            with self._hooks() as store:
                final_tokens = self._forward(sample, token_exit_depth=None)
                captured = dict(store)
                # _reduce_extra must happen before the context exits only if it
                # touched the store; it does not, but keep the read local anyway.
                grid_shape6 = final_tokens.shape
        else:
            final_tokens = self._forward(sample, token_exit_depth=None)
            grid_shape6 = final_tokens.shape

        _, h, w, n_time, n_bandsets, dim = grid_shape6
        grid5 = (h, w, n_time, n_bandsets, dim)
        embeddings = self._reduce_extra(final_tokens)

        expected_side = self.tile_cfg.expected_grid_side
        if (h, w) != (expected_side, expected_side):
            logger.warning(
                "encoder returned a %dx%d token grid but chip_px//patch_px is "
                "%d; trusting the encoder. Check chip_px and patch_px.",
                h,
                w,
                expected_side,
            )

        if self.tap_cfg.method == "hooks":
            missing = [n for n in self.model.block_names if n not in captured]
            if missing:
                raise RuntimeError(
                    f"hooks did not fire for {missing}; the forward pass may "
                    f"not have reached every block"
                )
            for depth in depths:
                name = self.model.block_names[depth - 1]
                taps.append(self._hook_to_grid(captured[name], grid5))
                labels.append(tap_label(depth))
        else:
            for depth in depths:
                if depth == self.model.depth:
                    # token_exit at full depth selects the full-depth tokens,
                    # which is exactly the plain pass. Reuse it rather than
                    # paying for an identical forward -- and note that the
                    # tap-agreement test is what holds that equivalence honest.
                    taps.append(embeddings)
                else:
                    tokens = self._forward(sample, token_exit_depth=depth)
                    taps.append(self._reduce_extra(tokens))
                labels.append(tap_label(depth))

        meta: dict[str, Any] = {
            "lat": latlon[0] if latlon else None,
            "lon": latlon[1] if latlon else None,
            "date": list(self.tile_cfg.date),
            "chip_px": self.tile_cfg.chip_px,
            "patch_px": self.tile_cfg.patch_px,
            "normalize": self.tile_cfg.normalize,
            "tap_method": self.tap_cfg.method,
            "fast_pass": self.tap_cfg.fast_pass,
            "extra_dims": self.tap_cfg.extra_dims,
            "grid_shape": [int(h), int(w)],
            "n_time": int(n_time),
            "n_band_sets": int(n_bandsets),
            "hook_caveat": (
                "hook taps are pre-final-LayerNorm and not on a common scale "
                "across depth"
                if self.tap_cfg.method == "hooks"
                else None
            ),
            **self.model.manifest(),
            **(extra_meta or {}),
        }

        return EmbeddingResult(
            embeddings=embeddings,
            activations=np.stack(taps, axis=0),
            tap_labels=labels,
            tap_depths=list(depths),
            grid_shape=(int(h), int(w)),
            meta=meta,
        )

    def encode_batch(
        self,
        chips: Iterable[np.ndarray],
        *,
        latlons: Iterable[tuple[float, float] | None] | None = None,
    ) -> list[EmbeddingResult]:
        """Encode several chips.

        Chips are run one at a time rather than stacked into a batch. That is
        deliberate: the chips in an interpretability run generally come from
        different places and are inspected individually, and per-chip forwards
        keep the grid-shape validation and the metadata unambiguous. If throughput
        ever matters more than clarity, batch inside ``_forward``.

        Args:
            chips: Normalized chip arrays.
            latlons: Optional coordinates, one per chip.

        Returns:
            One result per chip, in input order.
        """
        chip_list = list(chips)
        coords: list[tuple[float, float] | None]
        if latlons is None:
            coords = [None] * len(chip_list)
        else:
            coords = list(latlons)
            if len(coords) != len(chip_list):
                raise ValueError(
                    f"got {len(chip_list)} chips but {len(coords)} latlons"
                )
        return [
            self.encode(chip, latlon=coord)
            for chip, coord in zip(chip_list, coords, strict=True)
        ]


def encode_location(
    model: OlmoEarthModel,
    source: Any,
    lat: float,
    lon: float,
    cfg: RunConfig,
) -> EmbeddingResult:
    """Read a chip at a coordinate and encode it.

    Args:
        model: A loaded model.
        source: A :class:`~olmoearth_activations.tiles.TileSource`.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        cfg: The full run config.

    Returns:
        The encoded result, with the config fingerprint in its metadata.
    """
    from olmoearth_activations.tiles import load_chip

    chip = load_chip(source, lat, lon, cfg.tile)
    encoder = Encoder(model, cfg.tile, cfg.tap)
    return encoder.encode(
        chip,
        latlon=(lat, lon),
        extra_meta={"config_fingerprint": cfg.fingerprint()},
    )
