"""Supported model artifacts and geometry."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelSpec:
    family: str
    variant: str | None
    repository: str
    revision: str
    dimensions: int
    chip_pixels: int
    patch_pixels: int
    batch_size: int

MODELS = {
    "olmoearth-v1.2-nano": ModelSpec("olmoearth", "v1_2-Nano", "allenai/OlmoEarth-v1_2-Nano", "e1f693ae2a7d5b57871a978e9d09e22d05206747", 128, 128, 4, 32),
    "olmoearth-v1.2-base": ModelSpec("olmoearth", "v1_2-Base", "allenai/OlmoEarth-v1_2-Base", "581aa9baaa7aed4348c0903617eb92ee9f89e2ec", 768, 128, 4, 8),
    "clay-1.5": ModelSpec("clay", None, "made-with-clay/Clay", "70200ebcccdf67bf2a0cb9984c77ddee26c10ed2", 1024, 256, 8, 2),
}

def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise ValueError(f"unknown model {key!r}; choose from {', '.join(MODELS)}")
    return MODELS[key]
