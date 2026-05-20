"""
Run OlmoEarth or Prithvi-EO-2.0 inference on a composite GeoTIFF.

Processes the raster in non-overlapping 128×128 pixel chips, extracts spatial
patch embeddings from the encoder, assembles them into a multi-band COG, then
optionally PCA-compresses to 64 dimensions.

OlmoEarth output resolution:  160 m/pixel  (patch_size=8 × 10 m input = 16 px/token × 10 m)
Prithvi output resolution:     ~480 m/pixel (patch_size=16 × 30 m input = 480 m/token)

Usage (OlmoEarth):
  python embed.py --model olmoearth \\
    --input outputs/composites/s2_annual_RI_2022_olmoearth.tif \\
    --output outputs/embeddings/olmoearth_RI_2022.tif

Usage (Prithvi — three seasonal composites):
  python embed.py --model prithvi \\
    --input outputs/composites/s2_spring_RI_2022_prithvi.tif \\
            outputs/composites/s2_summer_RI_2022_prithvi.tif \\
            outputs/composites/s2_fall_RI_2022_prithvi.tif \\
    --output outputs/embeddings/prithvi_RI_2022.tif

Flags:
  --no-pca              Store raw embeddings (768 for OlmoEarth; 1024 for Prithvi).
  --pca-dims N          PCA target dimensionality (default 64).
  --pca-model PATH      Path to pre-fitted .pkl PCA; if absent a new PCA is fitted
                        and saved next to the output TIF.
  --test-chips N        Process only the first N chips (local debug mode).
  --batch-size N        Chips per GPU batch (default 8).
  --variant STR         Base/Large for OlmoEarth; 300M/600M for Prithvi.
  --checkpoint-every N  Save a recovery checkpoint every N chips (default 500).
                        A .ckpt.npy and .ckpt.n sidecar are written next to the
                        output file and deleted on clean completion. If the job
                        is interrupted and restarted with the same --output path,
                        inference resumes from the last checkpoint automatically.
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
    "Base":  "OlmoEarth-v1-Base",
    "Large": "OlmoEarth-v1-Large",
}
PRITHVI_REPO = {
    "300M": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
    "600M": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
}

# ---------------------------------------------------------------------------
# OlmoEarth parameters (verified from allenai/OlmoEarth-v1-Base config.json
# and confirmed by dry-run with random weights)
# ---------------------------------------------------------------------------
# chip_px should be divisible by patch_size (8)
OLMOEARTH_CHIP_PX = 128
# patch_size = pixels per patch (NOT tokens per chip).
# max_patch_size=8 means 8 pixels per patch.
# For a 128×128 chip: 128/8 = 16 spatial tokens per axis.
OLMOEARTH_PATCH_SIZE = 8          # pixels per patch
OLMOEARTH_EMBED_DIM = 768         # encoder embedding_size (Base variant)
# Effective output resolution = patch_size × input_resolution = 8 × 10m = 80m
OLMOEARTH_STRIDE_PX = 8           # output stride in input pixels

# ---------------------------------------------------------------------------
# Prithvi parameters
# TODO: verify timestep count and normalization scale from
#       ibm-nasa-geospatial/Prithvi-EO-2.0-300M config.json on HuggingFace.
# ---------------------------------------------------------------------------
PRITHVI_CHIP_PX = 224
PRITHVI_TIMESTEPS = 3             # spring, summer, fall
PRITHVI_PATCH_SIZE = 16           # spatial patch size in pixels
PRITHVI_EMBED_DIM_BY_VARIANT = {"300M": 768, "600M": 1024}

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
    """
    from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

    model_id = ModelID(f"OlmoEarth-v1-{variant}")
    print(f"Loading OlmoEarth {variant} from HuggingFace ({model_id.repo_id()})…")
    model = load_model_from_id(model_id)
    model.eval()
    return model


def load_prithvi(variant: str = "300M"):
    """Return (model,) using transformers AutoModel with trust_remote_code."""
    from transformers import AutoModel  # type: ignore

    repo = PRITHVI_REPO[variant]
    print(f"Loading Prithvi-EO-2.0-{variant} from {repo}…")
    model = AutoModel.from_pretrained(repo, trust_remote_code=True)
    model.eval()
    return (model,)


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

    P_H = P_W = chip_px / OLMOEARTH_PATCH_SIZE = 128/8 = 16.
    Effective spatial resolution: 80m.
    """
    from olmoearth_pretrain.datatypes import OlmoEarthSample, MaskedOlmoEarthSample
    from olmoearth_pretrain.nn.flexi_vit import PoolingType

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
        # pool_spatially averages over T and Band_Sets dims → (B, P_H, P_W, D)
        spatial_emb = tokens_and_masks.pool_spatially(PoolingType.MEAN)

    return spatial_emb.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Prithvi batch inference
# ---------------------------------------------------------------------------

def run_prithvi_batch(
    model,
    chips: np.ndarray,   # (B, T, 6, chip_px, chip_px) float32, already normalised
    device: torch.device,
) -> np.ndarray:
    """Return (B, grid, grid, embed_dim) float32 spatial patch embeddings.

    grid = chip_px // PRITHVI_PATCH_SIZE.
    TODO: verify output structure from ibm-nasa-geospatial/Prithvi-EO-2.0-300M
          — the actual attribute name and shape of the encoder's hidden state.
    """
    tensor = torch.from_numpy(chips).to(device)   # (B, T, C, H, W)
    with torch.no_grad():
        output = model(pixel_values=tensor)

    # AutoModel from Prithvi returns last_hidden_state: (B, T * H_p * W_p, D)
    # TODO: confirm the output attribute name and shape from the model card.
    hidden = output.last_hidden_state   # (B, num_tokens, D)
    B, N, D = hidden.shape
    T = chips.shape[1]
    grid = int(round((N / T) ** 0.5))
    # Reshape to (B, T, grid, grid, D) then average over T
    spatial = hidden.reshape(B, T, grid, grid, D).mean(dim=1)  # (B, grid, grid, D)
    return spatial.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Chip iteration helpers
# ---------------------------------------------------------------------------

def iter_chips(src: rasterio.DatasetReader, chip_px: int):
    """Yield (row_off, col_off, win, chip_data) for non-overlapping chips.

    Edge chips are zero-padded to chip_px × chip_px.
    chip_data: (C, chip_px, chip_px) float32.
    """
    h, w = src.height, src.width
    for row_off in range(0, h, chip_px):
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
    out = pca.transform(flat)                   # (N, K)
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
) -> np.ndarray:
    """Return (768, H_out, W_out) embedding map at 80m effective resolution.

    Output shape: (768, ceil(H/8), ceil(W/8)) — one 768-dim vector per 8×8 pixel (80m) patch.
    Periodically checkpoints to ckpt_path so interrupted jobs can resume.
    """
    h, w = src.height, src.width
    stride = OLMOEARTH_STRIDE_PX
    out_h = (h + stride - 1) // stride
    out_w = (w + stride - 1) // stride

    # Resume from checkpoint if available
    out, n_skip = (None, 0)
    if ckpt_path:
        out, n_skip = checkpoint_load(ckpt_path)
    if out is None:
        out = np.full((OLMOEARTH_EMBED_DIM, out_h, out_w), np.nan, dtype="float32")

    n_processed = 0
    for batch in tqdm(chips_to_batches(iter_chips(src, OLMOEARTH_CHIP_PX), batch_size),
                      desc="OlmoEarth chips", initial=n_skip):
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
            checkpoint_save(out, n_processed, ckpt_path)

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
) -> np.ndarray:
    """Return (embed_dim, H_out, W_out) embedding map averaged over 3 seasons."""
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

    for batch in tqdm(chips_to_batches(multi_iter(), batch_size),
                      desc="Prithvi chips", initial=n_skip):
        if n_processed < n_skip:
            n_processed += len(batch)
            continue
        if test_chips is not None and (n_processed - n_skip) >= test_chips:
            break

        rows  = [b[0][0] for b in batch]
        cols  = [b[0][1] for b in batch]
        wins  = [b[0][2] for b in batch]

        season_chips = []
        for t in range(PRITHVI_TIMESTEPS):
            raw = np.stack([b[t][3] for b in batch], axis=0)
            norm = (raw - PRITHVI_MEANS[:, None, None]) / PRITHVI_STDS[:, None, None]
            season_chips.append(norm)
        chips = np.stack(season_chips, axis=1)

        spatial = run_prithvi_batch(model, chips, device)

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
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=["olmoearth", "prithvi"], required=True)
    parser.add_argument("--input", nargs="+", required=True,
                        help="Composite TIF(s): one for OlmoEarth; three for Prithvi.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-pca", action="store_true",
                        help="Skip PCA; store raw embeddings.")
    parser.add_argument("--pca-dims", type=int, default=64)
    parser.add_argument("--pca-model", type=Path, default=None)
    parser.add_argument("--test-chips", type=int, default=None,
                        help="Limit to first N chips (debug).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--variant", default=None)
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

    if args.model == "olmoearth":
        variant = args.variant or "Base"
        model = load_olmoearth(variant)
        model = model.to(device)

        if len(args.input) != 1:
            raise SystemExit("OlmoEarth requires exactly one --input TIF.")

        ckpt_path = args.output.with_suffix(".ckpt.npy")
        with rasterio.open(args.input[0]) as src:
            print(f"Input: {args.input[0]}  shape={src.count}×{src.height}×{src.width}  CRS={src.crs}")
            raw = embed_olmoearth(src, model, device, args.batch_size,
                                  args.test_chips, args.year,
                                  ckpt_path=ckpt_path,
                                  checkpoint_every=args.checkpoint_every)
            transform_in = src.transform
            crs_in = src.crs

        embed_dim = OLMOEARTH_EMBED_DIM
        stride = OLMOEARTH_STRIDE_PX
        col_prefix = "OE"

    elif args.model == "prithvi":
        variant = args.variant or "300M"
        (model,) = load_prithvi(variant)
        model = model.to(device)

        if len(args.input) != 3:
            raise SystemExit("Prithvi requires exactly three --input TIFs (spring, summer, fall).")

        ckpt_path = args.output.with_suffix(".ckpt.npy")
        srcs = [rasterio.open(p) for p in args.input]
        print(f"Inputs: {args.input}")
        embed_dim = PRITHVI_EMBED_DIM_BY_VARIANT[variant]
        raw = embed_prithvi(srcs, model, embed_dim, device, args.batch_size,
                            args.test_chips,
                            ckpt_path=ckpt_path,
                            checkpoint_every=args.checkpoint_every)
        transform_in = srcs[0].transform
        crs_in = srcs[0].crs
        for s in srcs:
            s.close()

        stride = PRITHVI_PATCH_SIZE
        col_prefix = "PR"

    print(f"Raw embedding map: {raw.shape}  (D × H_out × W_out)")

    # PCA compression
    if args.no_pca:
        final = raw
        band_names = [f"{col_prefix}{i:04d}" for i in range(embed_dim)]
    else:
        if args.pca_model and args.pca_model.exists():
            with open(args.pca_model, "rb") as f:
                pca = pickle.load(f)
            print(f"Loaded PCA from {args.pca_model}")
        else:
            print(f"Fitting PCA ({args.pca_dims} components)…")
            pca = fit_pca(raw, n_components=args.pca_dims)
            pca_path = args.output.with_suffix(".pca.pkl")
            with open(pca_path, "wb") as f:
                pickle.dump(pca, f)
            var_exp = pca.explained_variance_ratio_.sum()
            print(f"PCA saved → {pca_path}  (explained variance: {var_exp:.1%})")

        final = apply_pca(raw, pca)
        band_names = [f"{col_prefix}{i:02d}" for i in range(args.pca_dims)]

    # Scale geotransform to output (patch-level) resolution
    out_transform = Affine(
        transform_in.a * stride, transform_in.b, transform_in.c,
        transform_in.d, transform_in.e * stride, transform_in.f,
    )

    print(f"Writing COG: {args.output}  {final.shape}")
    write_cog(final, out_transform, crs_in, args.output, band_names=band_names)
    checkpoint_delete(ckpt_path)
    print("Done.")


if __name__ == "__main__":
    main()
