"""Model loading, device placement, and discovery of architecture facts.

Nothing here hard-codes depth, embedding dimension, or block count. All three
are read off the loaded module, because the point of this package is to work
across Nano/Tiny/Small/Base without edits.

A note on the ``ModelID`` enum
------------------------------
The enum's membership depends on the *installed* ``olmoearth_pretrain`` version,
not on the checkpoints available on HuggingFace. Older installs carry only the
v1 members and will reject ``"OlmoEarth-v1_2-Nano"`` outright. When that
happens :func:`OlmoEarthModel.load` raises with the installed member list and a
concrete remedy, rather than quietly loading a different model. See
``allow_path_fallback`` for the documented, opt-in escape hatch.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn

# torch is imported lazily so that importing this module -- and therefore the
# package -- does not require a working torch install. Analysis of saved
# results needs neither torch nor a checkpoint.

from olmoearth_activations.config import ModelConfig

logger = logging.getLogger(__name__)

#: Matches exactly ``encoder.blocks.<int>`` and nothing else. The anchoring
#: matters: ``re.fullmatch`` against this pattern excludes
#: ``target_encoder.blocks.N`` and ``decoder.blocks.N``, which are present in
#: the released checkpoint but are not the online encoder we want to tap.
_BLOCK_NAME_RE = re.compile(r"encoder\.blocks\.(\d+)")

#: The modality this package works in. Sentinel-2 L2A only; the multimodal
#: machinery is inert for a single-sensor annual composite.
MODALITY_KEY = "sentinel2_l2a"


def resolve_device(spec: str) -> torch.device:
    """Turn a device spec into a concrete device.

    Args:
        spec: ``"auto"``, or anything ``torch.device`` accepts.

    Returns:
        A concrete device. ``"auto"`` prefers CUDA, then Apple MPS, then CPU.
    """
    import torch

    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and (
        torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class OlmoEarthModel:
    """A loaded OlmoEarth model plus the architecture facts derived from it.

    Attributes:
        module: The full ``LatentMIM`` module. The released checkpoint contains
            ``encoder``, ``decoder`` and ``target_encoder``, so all three are
            available here without retraining.
        device: Where the module lives.
        cfg: The config it was loaded under.
        model_value: The resolved model identifier string.
    """

    module: nn.Module
    device: torch.device
    cfg: ModelConfig
    model_value: str

    # ---------------------------------------------------------------- load

    @classmethod
    def load(
        cls,
        cfg: ModelConfig | None = None,
        *,
        allow_path_fallback: bool = False,
    ) -> OlmoEarthModel:
        """Load a checkpoint, move it to the device, and put it in eval mode.

        Args:
            cfg: Model settings. Defaults to :class:`ModelConfig` defaults.
            allow_path_fallback: If the installed ``ModelID`` enum has no member
                for ``cfg.model_id``, fetch ``config.json`` and ``weights.pth``
                from ``allenai/{model_id}`` directly and build via
                ``load_model_from_path``. Off by default: a missing enum member
                usually means the installed package is older than the
                checkpoint and cannot build its config either, and failing
                loudly is better than a confusing error deeper in.

        Returns:
            The loaded model wrapper.

        Raises:
            RuntimeError: If the model id is not available and the fallback is
                not enabled, or if the module has no tappable encoder blocks.
        """
        cfg = cfg or ModelConfig()
        device = resolve_device(cfg.device)

        if cfg.local_path is not None:
            module = cls._load_from_path(cfg.local_path)
            model_value = str(cfg.local_path)
        else:
            module, model_value = cls._load_from_hub(
                cfg, allow_path_fallback=allow_path_fallback
            )

        # eval() disables band dropout (gated on self.training) and any other
        # stochastic training-time behaviour, which is what makes activations
        # reproducible. Assert rather than trust.
        module.eval()
        if module.training:
            raise RuntimeError(
                "module.training is True after calling .eval(); refusing to "
                "extract activations from a model in training mode"
            )
        module.to(device)

        wrapper = cls(
            module=module, device=device, cfg=cfg, model_value=model_value
        )
        # Fail here, at load, rather than at the first forward pass.
        _ = wrapper.block_names
        logger.info(
            "loaded %s on %s: depth=%d embed_dim=%d",
            model_value,
            device,
            wrapper.depth,
            wrapper.embed_dim,
        )
        return wrapper

    @staticmethod
    def _load_from_path(path: Path) -> nn.Module:
        """Build a model from a local directory of ``config.json``/``weights.pth``."""
        from olmoearth_pretrain.model_loader import load_model_from_path

        if not Path(path).is_dir():
            raise RuntimeError(f"local_path is not a directory: {path}")
        logger.info("loading checkpoint from local path %s", path)
        return load_model_from_path(Path(path))

    @staticmethod
    def _load_from_hub(
        cfg: ModelConfig, *, allow_path_fallback: bool
    ) -> tuple[nn.Module, str]:
        """Build a model from HuggingFace, by enum member where possible."""
        from olmoearth_pretrain.model_loader import (
            ModelID,
            load_model_from_id,
        )

        available = {member.value: member for member in ModelID}
        if cfg.model_id in available:
            member = available[cfg.model_id]
            if cfg.hf_revision is not None:
                # load_model_from_id has no revision parameter, so honouring a
                # pin means taking the explicit-download route instead.
                logger.info(
                    "hf_revision is set, so downloading %s at revision %s "
                    "explicitly rather than via load_model_from_id",
                    cfg.model_id,
                    cfg.hf_revision,
                )
                return (
                    OlmoEarthModel._download_and_build(
                        cfg.model_id, cfg.hf_revision
                    ),
                    cfg.model_id,
                )
            logger.info("loading %s via ModelID.%s", cfg.model_id, member.name)
            return load_model_from_id(member), cfg.model_id

        message = (
            f"the installed olmoearth_pretrain has no ModelID member for "
            f"{cfg.model_id!r}.\n"
            f"Installed members: {sorted(available)}\n"
            f"This normally means the installed package predates the "
            f"checkpoint. An older package usually cannot build a newer "
            f"config.json either -- v1.1 and v1.2 configs carry keys "
            f"(patch_embed_hidden_sizes, band_dropout_rate, spatial_pos_encoding, "
            f"rope_*) that an older EncoderConfig will reject -- so upgrading "
            f"is the real fix:\n"
            f"  uv pip install --no-deps "
            f"'git+https://github.com/allenai/olmoearth_pretrain@<COMMIT_SHA>'\n"
            f"Pin the commit and record it in ModelConfig.package_commit.\n"
            f"To attempt the download-and-build route anyway, pass "
            f"allow_path_fallback=True."
        )
        if not allow_path_fallback:
            raise RuntimeError(message)

        logger.warning(
            "%s\n-- allow_path_fallback=True, attempting direct download", message
        )
        return (
            OlmoEarthModel._download_and_build(cfg.model_id, cfg.hf_revision),
            cfg.model_id,
        )

    @staticmethod
    def _download_and_build(model_id: str, revision: str | None) -> nn.Module:
        """Fetch config+weights from the hub and build without the enum.

        ``config.json`` and ``weights.pth`` land in the same snapshot directory,
        which is what ``load_model_from_path`` expects.
        """
        from huggingface_hub import hf_hub_download
        from olmoearth_pretrain.model_loader import load_model_from_path

        repo_id = f"allenai/{model_id}"
        logger.info("downloading %s (revision=%s)", repo_id, revision or "main")
        config_local = hf_hub_download(
            repo_id=repo_id, filename="config.json", revision=revision
        )
        hf_hub_download(
            repo_id=repo_id, filename="weights.pth", revision=revision
        )
        return load_model_from_path(Path(config_local).parent)

    # ------------------------------------------------------- architecture

    @property
    def encoder(self) -> nn.Module:
        """The online encoder submodule."""
        return self.module.encoder

    @property
    def block_names(self) -> list[str]:
        """Fully-qualified names of the encoder blocks, in layer order.

        Discovered by regex over ``named_modules()`` rather than assumed, so the
        same code works for Nano (4 blocks) and Base (12).

        Raises:
            RuntimeError: If no encoder blocks are found, which means the module
                layout is not what this package understands.
        """
        ids: set[int] = set()
        for name, _ in self.module.named_modules():
            match = _BLOCK_NAME_RE.fullmatch(name)
            if match:
                ids.add(int(match.group(1)))
        if not ids:
            raise RuntimeError(
                "found no modules matching 'encoder.blocks.<int>'. The model "
                "layout is not what this package expects; inspect "
                "list(model.module.named_modules()) and update loader."
                "_BLOCK_NAME_RE."
            )
        return [f"encoder.blocks.{i}" for i in sorted(ids)]

    @property
    def depth(self) -> int:
        """Number of encoder blocks."""
        return len(self.block_names)

    @property
    def embed_dim(self) -> int:
        """Encoder token width, read from the module rather than a table."""
        for attr in ("embedding_size", "embed_dim"):
            value = getattr(self.encoder, attr, None)
            if isinstance(value, int):
                return value
        # Fall back to the final LayerNorm's normalized shape.
        norm = getattr(self.encoder, "norm", None)
        shape = getattr(norm, "normalized_shape", None)
        if shape:
            return int(shape[-1])
        raise RuntimeError(
            "could not determine the encoder embedding dimension from the "
            "module; expected encoder.embedding_size, encoder.embed_dim, or "
            "encoder.norm.normalized_shape"
        )

    @property
    def num_band_sets(self) -> int:
        """Band-set count for Sentinel-2 L2A, per the installed modality spec.

        This is what sizes the mask's last axis. Note that a checkpoint's
        ``tokenization_config`` may override the *tokenizer* to a single band
        group while the modality spec still reports three; that mismatch is
        real and is why masks should be built by the library helper rather than
        by hand. See :mod:`olmoearth_activations.encode`.
        """
        from olmoearth_pretrain.data.constants import Modality

        return int(Modality.SENTINEL2_L2A.num_band_sets)

    @property
    def band_order(self) -> list[str]:
        """The required input channel order for Sentinel-2 L2A.

        Not alphabetical. Currently
        ``B02 B03 B04 B08 B05 B06 B07 B8A B11 B12 B01 B09``. A wrong order
        produces no error, only wrong numbers, so always iterate this.
        """
        from olmoearth_pretrain.data.constants import Modality

        return list(Modality.SENTINEL2_L2A.band_order)

    def submodule(self, name: str) -> nn.Module:
        """Look up a submodule by its fully-qualified name."""
        return self.module.get_submodule(name)

    def manifest(self) -> dict[str, Any]:
        """Provenance fields for a run manifest."""
        import torch

        return {
            "model_id": self.model_value,
            "hf_revision": self.cfg.hf_revision,
            "package_commit": self.cfg.package_commit,
            "device": str(self.device),
            "depth": self.depth,
            "embed_dim": self.embed_dim,
            "num_band_sets": self.num_band_sets,
            "band_order": self.band_order,
            "block_names": self.block_names,
            "torch_version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
        }
