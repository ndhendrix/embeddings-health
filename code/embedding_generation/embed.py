"""
Run OlmoEarth, Clay, or Prithvi-EO-2.0 inference on a composite GeoTIFF.

Processes the raster in non-overlapping chip-sized windows, extracts spatial
patch embeddings from the encoder, assembles them into a multi-band COG, then
optionally PCA-compresses to 64 dimensions.

OlmoEarth output resolution:   40 m/pixel  (patch_size=4 × 10 m input)
Clay v1.5 output resolution:   80 m/pixel  (patch_size=8 × 10 m input)
Prithvi output resolution:    480 m/pixel  (patch_size=16 × 30 m input)

Usage (OlmoEarth v1.1-Base or Nano):
  python embed.py --model olmoearth --variant v1_1-Base \\
    --input outputs/composites/s2_annual_RI_2022_olmoearth.tif \\
    --output outputs/embeddings/olmoearth_RI_2022.tif

  python embed.py --model olmoearth --variant Nano \\
    --input outputs/composites/s2_annual_RI_2022_olmoearth.tif \\
    --output outputs/embeddings/olmoearth_Nano_RI_2022.tif

Usage (Clay v1.5 — single 12-band OlmoEarth composite; 10 bands selected internally):
  python embed.py --model clay \\
    --input outputs/composites/s2_annual_RI_2022_olmoearth.tif \\
    --output outputs/embeddings/clay_v1.5_RI_2022.tif

Usage (Prithvi tiny — 3 seasonal composites; 4th frame auto-padded):
  python embed.py --model prithvi --variant tiny \\
    --input outputs/composites/s2_spring_RI_2022_prithvi.tif \\
            outputs/composites/s2_summer_RI_2022_prithvi.tif \\
            outputs/composites/s2_fall_RI_2022_prithvi.tif \\
    --output outputs/embeddings/prithvi_tiny_RI_2022.tif \\
    --raw-output outputs/embeddings/prithvi_tiny_RI_2022_raw.tif

Flags:
  --variant STR         Model variant. OlmoEarth: Base (default) / Large / Nano /
                        v1_1-Base / v1_1-Large. Prithvi: tiny (default) / 300M-TL /
                        300M / 600M. Unused for Clay (always v1.5 large).
  --pca                 Apply PCA compression after inference (opt-in; default is raw output).
  --raw-output PATH     Also write the pre-PCA raw embedding COG to this path.
                        Only meaningful when --pca is set; ignored otherwise (--output
                        already contains raw embeddings).
  --pca-dims N          PCA target dimensionality (default 64). Requires --pca.
  --pca-model PATH      Path to a pre-fitted .pkl PCA; if absent a new per-state PCA is
                        fitted and saved next to the output TIF. Requires --pca.
                        For nationally comparable embeddings, fit a single PCA across all
                        states with fit_national_pca.py and apply it at aggregation time.
  --force               Delete any existing output and checkpoint files before
                        starting. Without this flag, checkpoints are resumed
                        automatically.
  --test-chips N        Process only the first N chips (local debug mode).
  --batch-size N        Chips per GPU batch (default 8).
  --checkpoint-every N  Save a recovery checkpoint every N chips (default 500).
                        A .ckpt.npy and .ckpt.n sidecar are written next to the
                        output file and deleted on clean completion. If the job
                        is interrupted and restarted with the same --output path,
                        inference resumes from the last checkpoint automatically.

Frame-count mismatch (Prithvi only):
  If the model's config.num_frames exceeds the number of --input TIFs, the last
  TIF is repeated to fill the gap (with a printed warning). This lets you run the
  tiny-TL variant (num_frames=4) with only 3 seasonal composites.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils.cog_writer import write_cog


# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------
OLMOEARTH_REPO = {
    "Base":       "OlmoEarth-v1-Base",
    "Large":      "OlmoEarth-v1-Large",
    "v1_1-Base":  "OlmoEarth-v1_1-Base",
    "v1_1-Large": "OlmoEarth-v1_1-Large",
    "Nano":       "OlmoEarth-v1_1-Nano",
}
# Switch the output accumulation array to a disk-backed memmap when the array
# would exceed this size — prevents OOM on large states at 80m/768-dim output.
_MEMMAP_THRESHOLD_BYTES = 8 * 1024 ** 3
PRITHVI_REPO = {
    "tiny": "ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL",
    "300M-TL": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
    "300M": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
    "600M": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
}

# ---------------------------------------------------------------------------
# OlmoEarth parameters (verified from allenai/OlmoEarth-v1-Base config.json
# and confirmed by dry-run with random weights)
# ---------------------------------------------------------------------------
# chip_px should be divisible by patch_size (4)
OLMOEARTH_CHIP_PX = 128
# patch_size = pixels per patch (NOT tokens per chip).
# max_patch_size=8; using 4 gives 40m output from 10m input (in-distribution).
# For a 128×128 chip: 128/4 = 32 spatial tokens per axis.
OLMOEARTH_PATCH_SIZE = 4          # pixels per patch
OLMOEARTH_EMBED_DIM = 768         # encoder embedding_size (Base / v1_1-Base / Large variants)
# Effective output resolution = patch_size × input_resolution = 4 × 10m = 40m
OLMOEARTH_STRIDE_PX = 4           # output stride in input pixels

# Embedding dimension per variant (verified from each model's config.json).
# Nano embedding_size=128 confirmed from allenai/OlmoEarth-v1_1-Nano config.json.
OLMOEARTH_EMBED_DIMS = {
    "Base":       768,
    "Large":      1024,
    "v1_1-Base":  768,
    "v1_1-Large": 1024,
    "Nano":       128,
}

# ---------------------------------------------------------------------------
# Clay v1.5 parameters (verified from clay-foundation.github.io spec page
# and made-with-clay/Clay v1.5/clay-v1.5.ckpt config)
# ---------------------------------------------------------------------------
# Clay uses 256×256 chips at 10 m GSD; patch_size=8 → 32×32 patches per chip.
CLAY_CHIP_PX = 256
CLAY_PATCH_SIZE = 8               # pixels per patch
CLAY_EMBED_DIM = 1024             # ViT-Large encoder dim
# Effective output resolution = patch_size × input_resolution = 8 × 10m = 80m
CLAY_STRIDE_PX = 8                # output stride in input pixels

# Clay uses 10 Sentinel-2 L2A bands.  The 12-band OlmoEarth composite stores
# bands as [B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09]
# (indices 0-11).  Clay expects [B02, B03, B04, B05, B06, B07, B08, B8A, B11,
# B12] — note B08 and B05-B07 are reordered relative to the composite.
CLAY_BAND_INDICES = [0, 1, 2, 4, 5, 6, 3, 7, 8, 9]  # composite → Clay order

# Per-band normalization statistics for Sentinel-2 L2A from Clay metadata.yaml.
# Units match composite pixel values (surface reflectance × 10000, i.e. DN).
# Order: blue, green, red, rededge1, rededge2, rededge3, nir, nir08, swir16, swir22
CLAY_MEANS = np.array(
    [1105., 1355., 1552., 1887., 2422., 2630., 2743., 2785., 2388., 1835.],
    dtype="float32",
)
CLAY_STDS = np.array(
    [1809., 1757., 1888., 1870., 1732., 1697., 1742., 1648., 1470., 1379.],
    dtype="float32",
)
# Centre wavelengths in micrometres, matching CLAY_BAND_INDICES order.
CLAY_WAVELENGTHS = torch.tensor(
    [0.493, 0.560, 0.665, 0.704, 0.740, 0.783, 0.842, 0.865, 1.610, 2.190],
    dtype=torch.float32,
)

# ---------------------------------------------------------------------------
# Prithvi parameters
# TODO: verify timestep count and normalization scale from
#       ibm-nasa-geospatial/Prithvi-EO-2.0-300M config.json on HuggingFace.
# ---------------------------------------------------------------------------
PRITHVI_CHIP_PX = 224
PRITHVI_TIMESTEPS = 3             # default seasons produced by composite.py (spring/summer/fall)
PRITHVI_PATCH_SIZE = 16           # spatial patch size in pixels
# NOTE: the number of temporal frames fed to the model is read from model.config.num_frames
# at load time (via load_prithvi()), NOT from PRITHVI_TIMESTEPS. The tiny-TL variant needs 4.
# Normalization (S2 DN units, i.e. reflectance × 10000).
# Bands in order: B02, B03, B04, B8A, B11, B12
# TODO: confirm these against the model card mean/std values.
PRITHVI_MEANS = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype="float32")
PRITHVI_STDS  = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype="float32")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_olmoearth(variant: str = "Base"):
    """Return model using olmoearth_pretrain.model_loader.

    Note: we call model.encoder directly (bypassing the LatentMIM top-level
    and the broken eval_wrapper import chain) to get spatial patch embeddings.

    Uses load_model_from_path + hf_hub_download to bypass the ModelID enum,
    which only covers v1 and would reject v1.1 model names.
    """
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from olmoearth_pretrain.model_loader import load_model_from_path

    model_name = OLMOEARTH_REPO.get(variant)
    if model_name is None:
        raise SystemExit(
            f"Unknown OlmoEarth variant '{variant}'. "
            f"Valid variants: {list(OLMOEARTH_REPO)}"
        )
    repo_id = f"allenai/{model_name}"
    print(f"Loading OlmoEarth {variant} ({repo_id}) from HuggingFace…")
    # Download config and weights; both land in the same snapshot directory.
    config_local = hf_hub_download(repo_id=repo_id, filename="config.json")
    hf_hub_download(repo_id=repo_id, filename="weights.pth")
    model = load_model_from_path(Path(config_local).parent)
    model.eval()
    return model


def load_prithvi(variant: str = "tiny") -> tuple:
    """Return (model, embed_dim, num_frames, model_family).

    model_family is one of:
      'prithvi_mae'    — raw PyTorch weights + prithvi_mae.py (tiny-TL and similar)
      'hf_transformers'— HuggingFace AutoModel with trust_remote_code (300M, 600M)

    Dispatches based on the structure of config.json in the HuggingFace repo:
      architecture key present → prithvi_mae
      otherwise                → hf_transformers
    """
    import json
    from huggingface_hub import hf_hub_download

    repo = PRITHVI_REPO[variant]
    print(f"Loading Prithvi-EO-2.0-{variant} from {repo}…")
    cfg_path = hf_hub_download(repo_id=repo, filename="config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    if "architecture" in cfg_dict:
        return _load_prithvi_mae(repo, cfg_dict)
    else:
        return _load_prithvi_hf(repo, cfg_dict)


def _load_prithvi_mae(repo: str, cfg_dict: dict) -> tuple:
    """Load a raw-weights Prithvi model (tiny-TL style).

    Downloads the full repo snapshot, imports PrithviMAE from prithvi_mae.py,
    instantiates with pretrained_cfg, and loads the .pt checkpoint.
    """
    import sys, json, torch
    from pathlib import Path
    from huggingface_hub import snapshot_download

    repo_dir = snapshot_download(repo)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from prithvi_mae import PrithviMAE  # type: ignore  # noqa: PLC0415

    pcfg = cfg_dict["pretrained_cfg"]
    embed_dim  = int(cfg_dict.get("num_features", pcfg.get("embed_dim", 192)))
    num_frames = int(pcfg.get("num_frames", 4))
    print(f"  Config: embed_dim={embed_dim}  num_frames={num_frames}")

    # PrithviMAE.__init__ has **kwargs so extra keys (bands/mean/std/origin_url) are ignored.
    model = PrithviMAE(**pcfg)

    # Load weights; replace fixed pos_embed so the model handles variable num_frames.
    weights_path = next(Path(repo_dir).glob("*.pt"))
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    for k in list(state_dict.keys()):
        if k == "encoder.pos_embed":
            state_dict[k] = model.encoder.pos_embed
        elif k == "decoder.decoder_pos_embed":
            state_dict[k] = model.decoder.decoder_pos_embed
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"  Loaded weights from {weights_path.name}")
    return model, embed_dim, num_frames, "prithvi_mae"


def _load_prithvi_hf(repo: str, cfg_dict: dict) -> tuple:
    """Load a Prithvi model via HuggingFace AutoModel (300M / 600M)."""
    from transformers import AutoModel, AutoConfig  # type: ignore

    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
    embed_dim  = int(getattr(cfg, "embed_dim", None) or getattr(cfg, "hidden_size", 768))
    num_frames = int(getattr(cfg, "num_frames", 3))
    print(f"  Config: embed_dim={embed_dim}  num_frames={num_frames}")
    model = AutoModel.from_pretrained(repo, trust_remote_code=True)
    model.eval()
    return model, embed_dim, num_frames, "hf_transformers"


# ---------------------------------------------------------------------------
# Clay v1.5 model loading
# ---------------------------------------------------------------------------

def load_clay():
    """Download Clay v1.5 checkpoint and return the encoder (inference-only).

    Uses the bundled clay_encoder.py (extracted from the Clay Foundation Model
    GitHub repo, Apache 2.0) to avoid the broken claymodel PyPI wheel and the
    heavyweight lightning/vit-pytorch dependencies it pulls in.

    The checkpoint is a Lightning .ckpt file (~5 GB).  Only the 'model.encoder.*'
    weights are loaded into the ClayEncoder; decoder and teacher are discarded.
    """
    import math as _math
    from huggingface_hub import hf_hub_download
    from clay_encoder import ClayEncoder

    ckpt_path = hf_hub_download(
        repo_id="made-with-clay/Clay",
        filename="v1.5/clay-v1.5.ckpt",
    )
    print(f"Loading Clay v1.5 checkpoint from {ckpt_path}…")
    # weights_only=False: Lightning checkpoints store Python dicts/scalars in
    # addition to tensors, so unpickling requires full deserialization.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder_state = {
        k[len("model.encoder."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.encoder.")
    }
    encoder = ClayEncoder(
        mask_ratio=0.0,    # no masking at inference → all patches returned
        patch_size=CLAY_PATCH_SIZE,
        shuffle=False,     # deterministic row-major patch order
        dim=CLAY_EMBED_DIM,
        depth=24,
        heads=16,
        dim_head=64,
        mlp_ratio=4,
    )
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=True)
    if missing:
        print(f"  WARNING: missing keys: {missing[:5]}…")
    encoder.eval()
    print(f"  Clay v1.5 encoder loaded — {CLAY_EMBED_DIM}-dim, patch_size={CLAY_PATCH_SIZE}")
    return encoder


# ---------------------------------------------------------------------------
# NaN imputation
# ---------------------------------------------------------------------------

def _impute_nan(chips: np.ndarray) -> np.ndarray:
    """Replace NaN pixels with the per-band mean of valid pixels in that chip.

    Works for any shape (..., C, H, W). Each chip × band combination is
    imputed independently. If an entire band is NaN for a chip, fills with 0.

    Transformer attention is not NaN-safe — even one NaN pixel in a patch
    corrupts the full chip embedding via the softmax operation.
    """
    out = chips.copy()
    C = out.shape[-3]
    hw = out.shape[-2] * out.shape[-1]
    flat = out.reshape(-1, C, hw)          # (N, C, H*W)
    for i in range(flat.shape[0]):
        for c in range(C):
            band = flat[i, c]
            nan_mask = np.isnan(band)
            if nan_mask.any():
                fill = band[~nan_mask].mean() if not nan_mask.all() else 0.0
                band[nan_mask] = fill
    return out.reshape(chips.shape)


# ---------------------------------------------------------------------------
# OlmoEarth batch inference
# ---------------------------------------------------------------------------

def _make_timestamps(batch_size: int, year: int, device: torch.device) -> torch.Tensor:
    """Return (B, 1, 3) tensor with [day=15, month=6 (July, 0-indexed), year]."""
    ts = torch.tensor([[15, 6, year]], dtype=torch.long, device=device)
    return ts.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 1, 3)


def run_olmoearth_batch(
    model,
    chips: np.ndarray,    # (B, 12, chip_px, chip_px) float32
    latlons: np.ndarray,  # (B, 2) float32 — chip centre lat/lon
    device: torch.device,
    year: int = 2022,
) -> np.ndarray:
    """Return (B, P_H, P_W, 768) float32 spatial patch embeddings.

    P_H = P_W = chip_px / OLMOEARTH_PATCH_SIZE = 128/4 = 32.
    Effective spatial resolution: 40m.
    """
    from olmoearth_pretrain.datatypes import OlmoEarthSample, MaskedOlmoEarthSample

    chips = _impute_nan(chips)             # fill NaN before model sees them
    B, C, H, W = chips.shape
    # Permute (B, C, H, W) → (B, H, W, T=1, C)
    chip_t = torch.from_numpy(chips).permute(0, 2, 3, 1).unsqueeze(3).to(device)
    ts = _make_timestamps(B, year, device)  # (B, 1, 3)
    ll = torch.from_numpy(latlons).to(device)  # (B, 2)

    sample = OlmoEarthSample(sentinel2_l2a=chip_t, timestamps=ts, latlon=ll)
    masked = MaskedOlmoEarthSample.from_olmoearthsample(sample)

    with torch.no_grad():
        # Call model.encoder directly (not the full LatentMIM forward).
        # fast_pass=True skips the target encoder branch (inference-only).
        enc_out = model.encoder(masked, patch_size=OLMOEARTH_PATCH_SIZE, fast_pass=True)
        tokens_and_masks = enc_out["tokens_and_masks"]
        # sentinel2_l2a tokens: (B, P_H, P_W, T, Band_Sets, D)
        # Average over T (dim=-3) and Band_Sets (dim=-2) → (B, P_H, P_W, D)
        tokens = tokens_and_masks.sentinel2_l2a
        spatial_emb = tokens.mean(dim=(-2, -3))

    return spatial_emb.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Prithvi batch inference
# ---------------------------------------------------------------------------

def run_prithvi_batch(
    model,
    chips: np.ndarray,   # (B, T, 6, chip_px, chip_px) float32, already normalised
    device: torch.device,
    model_family: str = "prithvi_mae",
) -> np.ndarray:
    """Return (B, grid, grid, embed_dim) float32 spatial patch embeddings.

    grid = chip_px // PRITHVI_PATCH_SIZE = 224 // 16 = 14.
    Handles two model families:
      prithvi_mae     — raw PrithviMAE weights (tiny-TL); input (B,C,T,H,W);
                        calls model.forward_features() → list of hidden states.
      hf_transformers — HuggingFace AutoModel (300M/600M); input pixel_values (B,T,C,H,W);
                        reads output.last_hidden_state.
    """
    chips = _impute_nan(chips)          # fill NaN before model sees them
    B, T, C, H, W = chips.shape

    if model_family == "prithvi_mae":
        # PrithviMAE expects (B, C, T, H, W) — channels before time.
        tensor = torch.from_numpy(chips).permute(0, 2, 1, 3, 4).to(device)
        with torch.no_grad():
            features = model.forward_features(tensor)
        # forward_features returns a list; last element is final-layer hidden states.
        # Shape: (B, 1 + T*grid*grid, D) — first token is CLS.
        hidden = features[-1][:, 1:, :]    # drop CLS → (B, T*grid*grid, D)
        D = hidden.shape[-1]
        grid = int(round((hidden.shape[1] / T) ** 0.5))
    else:
        # HuggingFace: expects pixel_values (B, T, C, H, W)
        tensor = torch.from_numpy(chips).to(device)
        with torch.no_grad():
            output = model(pixel_values=tensor)
        hidden = output.last_hidden_state   # (B, N, D)
        D = hidden.shape[-1]
        grid = int(round((hidden.shape[1] / T) ** 0.5))

    # Average temporal tokens → spatial embedding map (B, grid, grid, D)
    spatial = hidden.reshape(B, T, grid, grid, D).mean(dim=1)
    return spatial.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Clay batch inference
# ---------------------------------------------------------------------------

def _make_clay_time_latlon(
    batch_size: int,
    latlons: np.ndarray,   # (B, 2) float32 — [lat, lon] in degrees
    year: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (time, latlon) encoding tensors for Clay.

    Clay encodes time as [sin_week, cos_week, sin_hour, cos_hour] and
    latlon as [sin_lat_rad, cos_lat_rad, sin_lon_rad, cos_lon_rad].
    Annual composites use week=26 (mid-year) and hour=12 (noon).
    """
    import math as _math
    week, hour = 26, 12
    sin_w = _math.sin(week * 2 * _math.pi / 52)
    cos_w = _math.cos(week * 2 * _math.pi / 52)
    sin_h = _math.sin(hour * 2 * _math.pi / 24)
    cos_h = _math.cos(hour * 2 * _math.pi / 24)
    time = torch.tensor(
        [[sin_w, cos_w, sin_h, cos_h]] * batch_size,
        dtype=torch.float32, device=device,
    )

    lat_rad = latlons[:, 0] * np.pi / 180
    lon_rad = latlons[:, 1] * np.pi / 180
    latlon_enc = np.stack(
        [np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)],
        axis=1,
    ).astype("float32")
    latlon = torch.from_numpy(latlon_enc).to(device)
    return time, latlon


def run_clay_batch(
    model,
    chips: np.ndarray,    # (B, 12, 256, 256) float32 — raw DN from OlmoEarth composite
    latlons: np.ndarray,  # (B, 2) float32 — chip centre [lat, lon] in degrees
    device: torch.device,
    year: int = 2022,
) -> np.ndarray:
    """Return (B, 32, 32, 1024) float32 spatial patch embeddings.

    Selects and reorders 10 of the 12 composite bands (CLAY_BAND_INDICES),
    normalises with Clay's Sentinel-2 L2A statistics, and runs the Clay
    ViT-Large encoder with mask_ratio=0, shuffle=False so all 32×32=1024
    patch tokens are returned in row-major spatial order.
    """
    # Band selection and reorder: composite order → Clay order
    chips_clay = chips[:, CLAY_BAND_INDICES, :, :]  # (B, 10, 256, 256)
    chips_clay = _impute_nan(chips_clay)

    # Normalise: (pixel - mean) / std, broadcasting over H×W
    chips_norm = (chips_clay - CLAY_MEANS[:, None, None]) / CLAY_STDS[:, None, None]

    B = chips_norm.shape[0]
    pixels = torch.from_numpy(chips_norm).to(device)  # (B, 10, 256, 256)

    time, latlon = _make_clay_time_latlon(B, latlons, year, device)

    datacube = {
        "pixels": pixels,
        "time":   time,
        "latlon": latlon,
        "gsd":    torch.tensor(10.0, device=device),
        "waves":  CLAY_WAVELENGTHS.to(device),
    }

    with torch.no_grad():
        encoded, *_ = model(datacube)   # (B, 1025, 1024) — index 0 is CLS

    # Drop CLS; reshape flat token sequence back to 2-D spatial grid
    spatial = encoded[:, 1:, :].reshape(B, 32, 32, CLAY_EMBED_DIM)
    return spatial.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Chip iteration helpers
# ---------------------------------------------------------------------------

def tile_row_bounds(n_row_chips: int, tile_index: int, num_tiles: int) -> tuple[int, int]:
    """Split n_row_chips chip-rows into num_tiles contiguous, roughly-equal bands.

    Returns (row_start, row_end) as chip-row indices (row_end exclusive) for
    tile_index. Splitting by row only (not a 2-D grid) is sufficient here:
    unlike composite.py's per-tile STAC searches, embed.py does windowed reads
    against an already-materialized raster, so there's no locality benefit to
    square tiles — only the total per-tile chip count matters.
    """
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if tile_index < 0 or tile_index >= num_tiles:
        raise ValueError(f"tile_index {tile_index} out of range [0, {num_tiles})")
    if num_tiles > n_row_chips:
        raise ValueError(
            f"num_tiles ({num_tiles}) exceeds n_row_chips ({n_row_chips}); reduce num_tiles."
        )
    base, rem = divmod(n_row_chips, num_tiles)
    if tile_index < rem:
        row_start = tile_index * (base + 1)
        row_end = row_start + (base + 1)
    else:
        row_start = rem * (base + 1) + (tile_index - rem) * base
        row_end = row_start + base
    return row_start, row_end


def iter_chips(src: rasterio.DatasetReader, chip_px: int,
               row_px_bounds: tuple[int, int] | None = None):
    """Yield (row_off, col_off, win, chip_data) for non-overlapping chips.

    Edge chips are zero-padded to chip_px × chip_px.
    chip_data: (C, chip_px, chip_px) float32.
    row_px_bounds, if given, restricts iteration to pixel rows
    [row_px_bounds[0], row_px_bounds[1]) of the full raster — used for tiling.
    Columns are never restricted (tiling is row-band only).
    """
    h, w = src.height, src.width
    row_start, row_end = row_px_bounds if row_px_bounds is not None else (0, h)
    for row_off in range(row_start, row_end, chip_px):
        for col_off in range(0, w, chip_px):
            read_h = min(chip_px, h - row_off)
            read_w = min(chip_px, w - col_off)
            win = Window(col_off, row_off, read_w, read_h)
            data = src.read(window=win).astype("float32")
            if read_h < chip_px or read_w < chip_px:
                pad = np.zeros((data.shape[0], chip_px, chip_px), dtype="float32")
                pad[:, :read_h, :read_w] = data
                data = pad
            yield row_off, col_off, win, data


def chips_to_batches(chip_iter, batch_size: int):
    batch = []
    for item in chip_iter:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def chip_center_latlon(
    src_transform: Affine,
    src_crs,
    row_off: int,
    col_off: int,
    chip_px: int,
) -> tuple[float, float]:
    """Return (lat, lon) of a chip's centre pixel."""
    from pyproj import Transformer
    cx = src_transform.c + (col_off + chip_px / 2) * src_transform.a
    cy = src_transform.f + (row_off + chip_px / 2) * src_transform.e
    if str(src_crs).upper().startswith("EPSG:4326"):
        return cy, cx
    t = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(cx, cy)
    return lat, lon


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def checkpoint_save(out: np.ndarray, n_done: int, ckpt_path: Path) -> None:
    """Atomically save the in-progress embedding map and chip count."""
    tmp = ckpt_path.with_suffix(".tmp.npy")
    np.save(tmp, out)
    tmp.rename(ckpt_path)
    ckpt_path.with_suffix(".n").write_text(str(n_done))


def checkpoint_load(ckpt_path: Path) -> tuple[np.ndarray, int] | tuple[None, int]:
    """Return (out_array, n_done) if a valid checkpoint exists, else (None, 0)."""
    n_path = ckpt_path.with_suffix(".n")
    if not ckpt_path.exists() or not n_path.exists():
        return None, 0
    try:
        out = np.load(ckpt_path)
        n_done = int(n_path.read_text().strip())
        print(f"Resuming from checkpoint: {n_done} chips already processed  ({ckpt_path})")
        return out, n_done
    except Exception as e:
        print(f"Warning: checkpoint unreadable ({e}), starting from scratch.")
        return None, 0


def checkpoint_delete(ckpt_path: Path) -> None:
    ckpt_path.unlink(missing_ok=True)
    ckpt_path.with_suffix(".n").unlink(missing_ok=True)
    ckpt_path.with_suffix(".mmap").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------------

def fit_pca(embedding_map: np.ndarray, n_components: int, sample_frac: float = 0.05) -> PCA:
    """Fit PCA on a sample of valid spatial embedding vectors.

    embedding_map: (D, H, W) float32.
    """
    D, H, W = embedding_map.shape
    flat = embedding_map.reshape(D, -1).T       # (N, D)
    valid = ~np.isnan(flat).any(axis=1)
    flat = flat[valid]
    n_sample = max(n_components * 20, int(len(flat) * sample_frac))
    idx = np.random.choice(len(flat), min(n_sample, len(flat)), replace=False)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(flat[idx])
    return pca


def apply_pca(embedding_map: np.ndarray, pca: PCA) -> np.ndarray:
    D, H, W = embedding_map.shape
    flat = embedding_map.reshape(D, -1).T       # (N, D)
    valid = ~np.isnan(flat).any(axis=1)
    out = np.full((flat.shape[0], pca.n_components_), np.nan, dtype="float32")
    if valid.any():
        out[valid] = pca.transform(flat[valid]).astype("float32")
    return out.T.reshape(pca.n_components_, H, W).astype("float32")


# ---------------------------------------------------------------------------
# Full-raster embedding (OlmoEarth)
# ---------------------------------------------------------------------------

def embed_olmoearth(
    src: rasterio.DatasetReader,
    model,
    device: torch.device,
    batch_size: int,
    test_chips: int | None,
    year: int,
    ckpt_path: Path | None = None,
    checkpoint_every: int = 500,
    embed_dim: int = OLMOEARTH_EMBED_DIM,
) -> np.ndarray:
    """Return (embed_dim, H_out, W_out) embedding map at 40m effective resolution.

    Output shape: (embed_dim, ceil(H/4), ceil(W/4)).
    Periodically checkpoints to ckpt_path so interrupted jobs can resume.

    When the output array would exceed _MEMMAP_THRESHOLD_BYTES (8 GB by default),
    a disk-backed numpy memmap is used automatically so large states (TX, CA, MT…)
    don't OOM. The memmap file lives next to the checkpoint as <ckpt>.mmap and is
    removed by checkpoint_delete() after the COG is written.
    """
    h, w = src.height, src.width
    stride = OLMOEARTH_STRIDE_PX
    out_h = (h + stride - 1) // stride
    out_w = (w + stride - 1) // stride
    shape = (embed_dim, out_h, out_w)

    array_bytes = embed_dim * out_h * out_w * 4
    use_memmap = array_bytes > _MEMMAP_THRESHOLD_BYTES
    mmap_path = ckpt_path.with_suffix(".mmap") if (use_memmap and ckpt_path) else None

    # Resume from checkpoint if available
    out, n_skip = None, 0
    if ckpt_path:
        n_path = ckpt_path.with_suffix(".n")
        if n_path.exists():
            try:
                n_skip = int(n_path.read_text().strip())
                if use_memmap and mmap_path and mmap_path.exists():
                    out = np.memmap(str(mmap_path), dtype="float32", mode="r+", shape=shape)
                    print(f"Resuming from memmap: {n_skip} chips done  ({mmap_path})")
                elif not use_memmap:
                    out, n_skip = checkpoint_load(ckpt_path)
            except Exception as e:
                print(f"Warning: checkpoint unreadable ({e}); starting from scratch.")
                n_skip, out = 0, None

    if out is None:
        if use_memmap:
            print(f"Output array {array_bytes / 1024**3:.1f} GB — using disk-backed memmap"
                  f" ({mmap_path})")
            out = np.memmap(str(mmap_path), dtype="float32", mode="w+", shape=shape)
        else:
            out = np.full(shape, np.nan, dtype="float32")
        out[:] = np.nan

    n_processed = 0
    for batch in tqdm(chips_to_batches(iter_chips(src, OLMOEARTH_CHIP_PX), batch_size),
                      desc=f"OlmoEarth chips (dim={embed_dim})", initial=n_skip // batch_size):
        # Skip chips already completed in a previous run
        if n_processed < n_skip:
            n_processed += len(batch)
            continue
        if test_chips is not None and (n_processed - n_skip) >= test_chips:
            break

        rows, cols, wins, datas = zip(*batch)
        chips = np.stack(datas, axis=0)

        latlons = np.array([
            chip_center_latlon(src.transform, src.crs, r, c, OLMOEARTH_CHIP_PX)
            for r, c in zip(rows, cols)
        ], dtype="float32")

        spatial = run_olmoearth_batch(model, chips, latlons, device, year)

        for i, (row_off, col_off, win) in enumerate(zip(rows, cols, wins)):
            out_r = row_off // stride
            out_c = col_off // stride
            valid_h = int(np.ceil(win.height / stride))
            valid_w = int(np.ceil(win.width  / stride))
            emb = spatial[i, :valid_h, :valid_w, :].transpose(2, 0, 1)
            out[:, out_r:out_r + valid_h, out_c:out_c + valid_w] = emb

        n_processed += len(batch)

        if ckpt_path and (n_processed % checkpoint_every == 0):
            if use_memmap:
                out.flush()
                ckpt_path.with_suffix(".n").write_text(str(n_processed))
            else:
                checkpoint_save(out, n_processed, ckpt_path)

    if use_memmap:
        out.flush()
    return out


# ---------------------------------------------------------------------------
# Full-raster embedding (Prithvi)
# ---------------------------------------------------------------------------

def embed_prithvi(
    srcs: list[rasterio.DatasetReader],
    model,
    embed_dim: int,
    device: torch.device,
    batch_size: int,
    test_chips: int | None,
    ckpt_path: Path | None = None,
    checkpoint_every: int = 500,
    model_family: str = "prithvi_mae",
) -> np.ndarray:
    """Return (embed_dim, H_out, W_out) embedding map averaged over temporal frames."""
    h, w = srcs[0].height, srcs[0].width
    stride = PRITHVI_PATCH_SIZE
    out_h = (h + stride - 1) // stride
    out_w = (w + stride - 1) // stride

    out, n_skip = (None, 0)
    if ckpt_path:
        out, n_skip = checkpoint_load(ckpt_path)
    if out is None:
        out = np.full((embed_dim, out_h, out_w), np.nan, dtype="float32")

    n_processed = 0
    chip_iters = [iter_chips(src, PRITHVI_CHIP_PX) for src in srcs]

    def multi_iter():
        for items in zip(*chip_iters):
            yield items

    n_timesteps = len(srcs)
    for batch in tqdm(chips_to_batches(multi_iter(), batch_size),
                      desc="Prithvi chips", initial=n_skip // batch_size):
        if n_processed < n_skip:
            n_processed += len(batch)
            continue
        if test_chips is not None and (n_processed - n_skip) >= test_chips:
            break

        rows  = [b[0][0] for b in batch]
        cols  = [b[0][1] for b in batch]
        wins  = [b[0][2] for b in batch]

        season_chips = []
        for t in range(n_timesteps):
            raw = np.stack([b[t][3] for b in batch], axis=0)
            norm = (raw - PRITHVI_MEANS[:, None, None]) / PRITHVI_STDS[:, None, None]
            season_chips.append(norm)
        chips = np.stack(season_chips, axis=1)

        spatial = run_prithvi_batch(model, chips, device, model_family=model_family)

        for i, (row_off, col_off, win) in enumerate(zip(rows, cols, wins)):
            out_r = row_off // stride
            out_c = col_off // stride
            valid_h = int(np.ceil(win.height / stride))
            valid_w = int(np.ceil(win.width  / stride))
            emb = spatial[i, :valid_h, :valid_w, :].transpose(2, 0, 1)
            out[:, out_r:out_r + valid_h, out_c:out_c + valid_w] = emb

        n_processed += len(batch)

        if ckpt_path and (n_processed % checkpoint_every == 0):
            checkpoint_save(out, n_processed, ckpt_path)

    return out


# ---------------------------------------------------------------------------
# Full-raster embedding (Clay)
# ---------------------------------------------------------------------------

def embed_clay(
    src: rasterio.DatasetReader,
    model,
    device: torch.device,
    batch_size: int,
    test_chips: int | None,
    year: int,
    ckpt_path: Path | None = None,
    checkpoint_every: int = 500,
) -> np.ndarray:
    """Return (1024, H_out, W_out) embedding map at 80m effective resolution.

    Output shape: (1024, ceil(H/8), ceil(W/8)).
    Reuses the memmap + checkpointing strategy of embed_olmoearth for large
    states where the output array would exceed _MEMMAP_THRESHOLD_BYTES.
    """
    h, w = src.height, src.width
    stride = CLAY_STRIDE_PX
    out_h = (h + stride - 1) // stride
    out_w = (w + stride - 1) // stride
    shape = (CLAY_EMBED_DIM, out_h, out_w)

    array_bytes = CLAY_EMBED_DIM * out_h * out_w * 4
    use_memmap = array_bytes > _MEMMAP_THRESHOLD_BYTES
    mmap_path = ckpt_path.with_suffix(".mmap") if (use_memmap and ckpt_path) else None

    out, n_skip = None, 0
    if ckpt_path:
        n_path = ckpt_path.with_suffix(".n")
        if n_path.exists():
            try:
                n_skip = int(n_path.read_text().strip())
                if use_memmap and mmap_path and mmap_path.exists():
                    out = np.memmap(str(mmap_path), dtype="float32", mode="r+", shape=shape)
                    print(f"Resuming from memmap: {n_skip} chips done  ({mmap_path})")
                elif not use_memmap:
                    out, n_skip = checkpoint_load(ckpt_path)
            except Exception as e:
                print(f"Warning: checkpoint unreadable ({e}); starting from scratch.")
                n_skip, out = 0, None

    if out is None:
        if use_memmap:
            print(f"Output array {array_bytes / 1024**3:.1f} GB — using disk-backed memmap"
                  f" ({mmap_path})")
            out = np.memmap(str(mmap_path), dtype="float32", mode="w+", shape=shape)
        else:
            out = np.full(shape, np.nan, dtype="float32")
        out[:] = np.nan

    n_processed = 0
    for batch in tqdm(chips_to_batches(iter_chips(src, CLAY_CHIP_PX), batch_size),
                      desc="Clay chips", initial=n_skip // batch_size):
        if n_processed < n_skip:
            n_processed += len(batch)
            continue
        if test_chips is not None and (n_processed - n_skip) >= test_chips:
            break

        rows, cols, wins, datas = zip(*batch)
        chips = np.stack(datas, axis=0)

        latlons = np.array([
            chip_center_latlon(src.transform, src.crs, r, c, CLAY_CHIP_PX)
            for r, c in zip(rows, cols)
        ], dtype="float32")

        spatial = run_clay_batch(model, chips, latlons, device, year)

        for i, (row_off, col_off, win) in enumerate(zip(rows, cols, wins)):
            out_r = row_off // stride
            out_c = col_off // stride
            valid_h = int(np.ceil(win.height / stride))
            valid_w = int(np.ceil(win.width  / stride))
            emb = spatial[i, :valid_h, :valid_w, :].transpose(2, 0, 1)
            out[:, out_r:out_r + valid_h, out_c:out_c + valid_w] = emb

        n_processed += len(batch)

        if ckpt_path and (n_processed % checkpoint_every == 0):
            if use_memmap:
                out.flush()
                ckpt_path.with_suffix(".n").write_text(str(n_processed))
            else:
                checkpoint_save(out, n_processed, ckpt_path)

    if use_memmap:
        out.flush()
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=["olmoearth", "prithvi", "clay"], required=True)
    parser.add_argument("--input", nargs="+", required=True,
                        help="Composite TIF(s): one for OlmoEarth/Clay; one-to-four for Prithvi.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, default=None,
                        help="Also write the pre-PCA raw embeddings to this COG path. "
                             "Useful for re-running PCA/UMAP without redoing GPU inference. "
                             "Ignored when --no-pca is set (--output already has raw embeddings).")
    parser.add_argument("--pca", action="store_true",
                        help="Apply PCA after inference. Default is raw output. "
                             "For nationally comparable embeddings, omit this flag and "
                             "apply a national PCA at aggregation time instead.")
    parser.add_argument("--pca-dims", type=int, default=64)
    parser.add_argument("--pca-model", type=Path, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Delete existing output and checkpoint files before starting. "
                             "Without this flag, an existing checkpoint is resumed automatically.")
    parser.add_argument("--test-chips", type=int, default=None,
                        help="Limit to first N chips (debug).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--variant", default=None,
                        help="Model variant. OlmoEarth: Base (default) / Large. "
                             "Prithvi: tiny (default) / 300M-TL / 300M / 600M.")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="Save recovery checkpoint every N chips (default 500).")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if args.test_chips:
        print(f"DEBUG: processing only first {args.test_chips} chips.")

    # Resolve checkpoint path early so --force can clean it up before model loading.
    ckpt_path = args.output.with_suffix(".ckpt.npy")

    if args.force:
        for p in [args.output, ckpt_path, ckpt_path.with_suffix(".n"),
                  ckpt_path.with_suffix(".mmap")]:
            if p and p.exists():
                p.unlink()
        if args.raw_output and args.raw_output.exists():
            args.raw_output.unlink()
        print("--force: cleared existing outputs and checkpoints.")

    if args.model == "olmoearth":
        variant = args.variant or "Base"
        model = load_olmoearth(variant)
        model = model.to(device)

        if len(args.input) != 1:
            raise SystemExit("OlmoEarth requires exactly one --input TIF.")

        embed_dim = OLMOEARTH_EMBED_DIMS.get(variant, OLMOEARTH_EMBED_DIM)
        with rasterio.open(args.input[0]) as src:
            print(f"Input: {args.input[0]}  shape={src.count}×{src.height}×{src.width}  CRS={src.crs}")
            raw = embed_olmoearth(src, model, device, args.batch_size,
                                  args.test_chips, args.year,
                                  ckpt_path=ckpt_path,
                                  checkpoint_every=args.checkpoint_every,
                                  embed_dim=embed_dim)
            transform_in = src.transform
            crs_in = src.crs

        stride = OLMOEARTH_STRIDE_PX
        col_prefix = "OE"

    elif args.model == "clay":
        model = load_clay()
        model = model.to(device)

        if len(args.input) != 1:
            raise SystemExit(
                "Clay requires exactly one --input TIF "
                "(12-band OlmoEarth composite; 10 bands selected internally)."
            )

        with rasterio.open(args.input[0]) as src:
            if src.count < 10:
                raise SystemExit(
                    f"Clay requires at least 10 bands; input has {src.count}."
                )
            print(f"Input: {args.input[0]}  shape={src.count}×{src.height}×{src.width}  CRS={src.crs}")
            raw = embed_clay(src, model, device, args.batch_size,
                             args.test_chips, args.year,
                             ckpt_path=ckpt_path,
                             checkpoint_every=args.checkpoint_every)
            transform_in = src.transform
            crs_in = src.crs

        embed_dim = CLAY_EMBED_DIM
        stride = CLAY_STRIDE_PX
        col_prefix = "CL"

    elif args.model == "prithvi":
        variant = args.variant or "tiny"
        if variant not in PRITHVI_REPO:
            valid = ", ".join(PRITHVI_REPO)
            raise SystemExit(f"Unknown Prithvi variant '{variant}'. Valid variants: {valid}.")
        model, embed_dim, num_frames, model_family = load_prithvi(variant)
        model = model.to(device)

        # Adjust input list to match num_frames expected by the model.
        inputs = list(args.input)
        if len(inputs) < num_frames:
            n_pad = num_frames - len(inputs)
            print(f"WARNING: {variant} model expects {num_frames} temporal frames but only "
                  f"{len(inputs)} --input TIF(s) provided. Repeating the last TIF {n_pad} "
                  f"time(s) to pad. Consider generating a winter composite for a proper 4th frame.")
            inputs = inputs + [inputs[-1]] * n_pad
        elif len(inputs) > num_frames:
            print(f"WARNING: {variant} model expects {num_frames} frames but {len(inputs)} "
                  f"--input TIFs were provided; using only the first {num_frames}.")
            inputs = inputs[:num_frames]

        srcs = [rasterio.open(p) for p in inputs]
        print(f"Inputs ({len(srcs)} frames): {inputs}")
        raw = embed_prithvi(srcs, model, embed_dim, device, args.batch_size,
                            args.test_chips,
                            ckpt_path=ckpt_path,
                            checkpoint_every=args.checkpoint_every,
                            model_family=model_family)
        transform_in = srcs[0].transform
        crs_in = srcs[0].crs
        for s in srcs:
            s.close()

        stride = PRITHVI_PATCH_SIZE
        col_prefix = "PR"

    print(f"Raw embedding map: {raw.shape}  (D × H_out × W_out)")

    # Scale geotransform to output (patch-level) resolution — needed for both raw and PCA writes.
    out_transform = Affine(
        transform_in.a * stride, transform_in.b, transform_in.c,
        transform_in.d, transform_in.e * stride, transform_in.f,
    )

    raw_band_names = [f"{col_prefix}{i:04d}" for i in range(embed_dim)]

    # Optionally save the pre-PCA raw embeddings alongside the PCA output.
    if args.raw_output and args.pca:
        print(f"Writing raw embedding COG: {args.raw_output}  {raw.shape}")
        write_cog(raw, out_transform, crs_in, args.raw_output, band_names=raw_band_names, overviews=False)

    # PCA compression (opt-in via --pca; default is raw output).
    if not args.pca:
        final = raw
        band_names = raw_band_names
    else:
        if args.pca_model and args.pca_model.exists():
            with open(args.pca_model, "rb") as f:
                pca = pickle.load(f)
            print(f"Loaded PCA from {args.pca_model}")
        else:
            print(f"Fitting PCA ({args.pca_dims} components)…")
            pca = fit_pca(raw, n_components=args.pca_dims)
            pca_path = args.output.with_suffix(".pca.pkl")
            pca_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pca_path, "wb") as f:
                pickle.dump(pca, f)
            var_exp = pca.explained_variance_ratio_.sum()
            print(f"PCA saved → {pca_path}  (explained variance: {var_exp:.1%})")

        final = apply_pca(raw, pca)
        band_names = [f"{col_prefix}{i:02d}" for i in range(args.pca_dims)]

    print(f"Writing COG: {args.output}  {final.shape}")
    write_cog(final, out_transform, crs_in, args.output, band_names=band_names, overviews=False)
    checkpoint_delete(ckpt_path)
    print("Done.")


if __name__ == "__main__":
    main()
