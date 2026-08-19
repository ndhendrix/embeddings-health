"""
Input: a Sentinel-2 L2A scene unzipped as a .SAFE folder in the working dir.,
     must first configure input tile with latitude, longitude, etc.
"""

import re
import glob

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform

import torch

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy


# CONFIGURE INPUT TILE HERE

LAT, LON = 38.903240, -77.036964       # Latitude/Longitude
SIZE = 64                              # tile size in pixels
PATCH = 4                              # patch size

DATE = (3, 7, 2026)                   # Day, month, year


def load_model():
    model = load_model_from_id(ModelID.OLMOEARTH_V1_2_NANO)
    model.eval()   # disables dropout (used in training mode to prevent overfitting) so activations are deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"model loaded on {device}")
    return model, device


def find_encoder_blocks(model):
    """
    Discover the encoder blocks by name instead of assuming there are 12. 
    We match module names of the exact form 'encoder.blocks.<number>' (not
    decoder.* or target_encoder.*, which are training-only artifacts we don't
    want). Returns the block names sorted 0, 1, 2, ... so hooks and stacking
    stay in layer order.
    """
    block_ids = set()
    for name, _ in model.named_modules():
        m = re.fullmatch(r"encoder\.blocks\.(\d+)", name)
        if m:
            block_ids.add(int(m.group(1)))
    names = [f"encoder.blocks.{i}" for i in sorted(block_ids)]
    if not names:
        raise RuntimeError("No 'encoder.blocks.N' modules found -- check the model.")
    print(f"found {len(names)} encoder blocks: {names[0]} .. {names[-1]}")
    return names

def load_tile(lat, lon, size):
    """
    Read a `size` x `size` tile centered on (lat, lon) from a .SAFE scene in the
    working directory. Sentinel-2 stores each band as its own .jp2 at possibly
    different resolutions; we resample them all onto band B02's 10 m grid.
    """
    # band_order is the exact sequence OlmoEarth expects -- wrong order is a
    # silent bug, so we always iterate it rather than glob blindly.
    fnames = []
    for band_name in Modality.SENTINEL2_L2A.band_order:
        matches = glob.glob(f"*.SAFE/GRANULE/*/IMG_DATA/*/*_{band_name}_*.jp2")
        if not matches:
            raise FileNotFoundError(
                f"No file for band {band_name}. Is the .SAFE folder in this dir?"
            )
        fnames.append(matches[0])

    # First band (B02, 10 m/px) defines the reference grid and locates our point.
    with rasterio.open(fnames[0]) as src:
        crs, transform = src.crs, src.transform
        width, height = src.width, src.height
        xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])  # GPS -> scene CRS
        row, col = src.index(xs[0], ys[0])                       # -> pixel row/col

    row0, col0 = row - size // 2, col - size // 2
    if not (0 <= row0 and row0 + size <= height and 0 <= col0 and col0 + size <= width):
        raise ValueError(
            "Requested tile falls outside this scene. Pick a location inside the "
            "downloaded scene, or download another."
        )
    window = rasterio.windows.Window(col0, row0, size, size)

    image = np.zeros((len(fnames), size, size), dtype=np.int32)
    for i, fname in enumerate(fnames):
        with rasterio.open(fname) as src:
            with WarpedVRT(
                src, crs=crs, transform=transform,
                width=width, height=height,
                resampling=Resampling.bilinear,
            ) as vrt:
                image[i] = vrt.read(1, window=window)   # only the tile, not the scene

    # (12, H, W) -> (1, H, W, 1, 12) = (Batch, Height, Width, Time, Channels)
    image = image.transpose(1, 2, 0)[None, :, :, None, :]
    print(f"tile loaded, raw shape {image.shape}")
    return image


def normalize_tile(image):
    # Scale raw band values into the range the model was trained on.
    return Normalizer(Strategy.COMPUTED).normalize(Modality.SENTINEL2_L2A, image)


def register_hooks(model, block_names, store):
    
    def make_hook(name):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            # detach: no gradients (inference). cpu: keep GPU memory free.
            store[name] = out.detach().cpu()
        return hook

    handles = [model.get_submodule(n).register_forward_hook(make_hook(n))
               for n in block_names]
    print(f"registered {len(handles)} hooks")
    return handles


def run_model(model, device, image, date):
    # Bundle image + mask + timestamp and run the encoder (hooks capture activations).
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=torch.tensor(image, dtype=torch.float32, device=device),
        # mask last axis = 3 band-sets; ONLINE_ENCODER = actually encode everything
        sentinel2_l2a_mask=torch.ones(
            (1, image.shape[1], image.shape[2], 1, 3), device=device
        ) * MaskValue.ONLINE_ENCODER.value,
        timestamps=torch.tensor(date, device=device)[None, None, :],
    )
    with torch.no_grad():
        out = model.encoder(sample, fast_pass=True, patch_size=PATCH)
        final = out["tokens_and_masks"].sentinel2_l2a
    print(f"forward pass done, final output shape {tuple(final.shape)}")
    return final


# Stack the captured activations per-patch

def stack_per_patch(store, block_names):
    """
    Combine the captured activations into (n_layers, N, D), keeping the token
    axis N (the flattened spatial grid). We do NOT average it away -- that's the
    whole point of per-patch: each of the N vectors stays tied to a location, so
    later analysis can say WHICH patches a concept fires on.

    Note: to turn the flat N tokens back into a 2D map (H' x W') for heatmaps,
    reshape once you know H' and W' from the final output shape above.
    """
    ordered = [store[n] for n in block_names]            # each (1, N, D)
    per_patch = torch.stack(ordered, dim=0).squeeze(1)   # (n_layers, N, D)
    print(f"per_patch shape {tuple(per_patch.shape)}  (layers, tokens, width)")
    return per_patch


def remove_hooks(handles):
    for h in handles:
        h.remove()
    handles.clear()


# Main
def main():
    model, device = load_model()
    block_names = find_encoder_blocks(model)     # discover depth from the model

    image = load_tile(LAT, LON, SIZE)
    image = normalize_tile(image)

    acts = {}
    handles = register_hooks(model, block_names, acts)
    try:
        acts.clear()
        run_model(model, device, image, DATE)
        per_patch = stack_per_patch(acts, block_names)
    finally:
        remove_hooks(handles)   # runs even if something above errored

    # Per-patch is large: (layers x N x D) per tile. At scale, watch disk/dtype
    # (float16 halves it) and save a MANIFEST row per tile mapping the file back
    # to (tile id, lat, lon, date) -- that's what lets these vectors be lined up
    # against health data later.
    np.savez(
        "tile_activations.npz",
        per_patch=per_patch.numpy(),
        lat=LAT, lon=LON, date=np.array(DATE),
    )
    print("saved tile_activations_nano_perpatch.npz")


if __name__ == "__main__":
    main()