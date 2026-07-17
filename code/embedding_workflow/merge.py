"""Strict, non-destructive, atomic mosaic of aligned embedding tiles."""
import argparse
import os
from pathlib import Path
import rasterio
from rasterio.windows import from_bounds

def _validate_coverage(windows, width: int, height: int) -> None:
    """Validate rectangular ownership without a width*height bitmap."""
    rectangles = []
    for window in windows:
        x0, y0 = int(window.col_off), int(window.row_off)
        x1, y1 = x0 + int(window.width), y0 + int(window.height)
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            raise ValueError("tile falls outside mosaic bounds")
        rectangles.append((x0, y0, x1, y1))
    for i, a in enumerate(rectangles):
        for b in rectangles[i + 1:]:
            if max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3]):
                raise ValueError("tile ownership rectangles overlap")
    area = sum((x1-x0)*(y1-y0) for x0,y0,x1,y1 in rectangles)
    if area != width * height:
        raise ValueError(f"tile set covers {area} of {width * height} pixels")

def merge_tiles(paths: list[Path], output: Path, band_chunk: int = 16) -> None:
    if not paths:
        raise ValueError("no tile paths supplied")
    sources = [rasterio.open(path) for path in paths]
    try:
        first = sources[0]
        for src in sources[1:]:
            if src.crs != first.crs or src.count != first.count or src.dtypes != first.dtypes or src.res != first.res:
                raise ValueError(f"incompatible tile schema: {src.name}")
        left, bottom = min(s.bounds.left for s in sources), min(s.bounds.bottom for s in sources)
        right, top = max(s.bounds.right for s in sources), max(s.bounds.top for s in sources)
        transform = rasterio.transform.from_origin(left, top, *first.res)
        width, height = round((right-left)/first.res[0]), round((top-bottom)/first.res[1])
        profile = first.profile.copy()
        profile.update(width=width, height=height, transform=transform, tiled=True,
                       compress="deflate", BIGTIFF="IF_SAFER", nodata=first.nodata)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.partial")
        temporary.unlink(missing_ok=True)
        windows = [from_bounds(*src.bounds, transform=transform).round_offsets().round_lengths() for src in sources]
        for src, window in zip(sources, windows):
            if window.width != src.width or window.height != src.height:
                raise ValueError(f"unaligned tile: {src.name}")
        _validate_coverage(windows, width, height)
        with rasterio.open(temporary, "w", **profile) as dst:
            dst.update_tags(**first.tags())
            for band, description in enumerate(first.descriptions, 1):
                if description: dst.set_band_description(band, description)
            for src, window in zip(sources, windows):
                for _, block in src.block_windows(1):
                    target = rasterio.windows.Window(window.col_off+block.col_off, window.row_off+block.row_off, block.width, block.height)
                    for start in range(1, src.count + 1, band_chunk):
                        indexes = list(range(start, min(src.count + 1, start + band_chunk)))
                        dst.write(src.read(indexes, window=block), indexes=indexes, window=target)
        os.replace(temporary, output)
    finally:
        for src in sources: src.close()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--band-chunk", type=int, default=16)
    a = p.parse_args(); merge_tiles(a.tiles, a.output, a.band_chunk)

if __name__ == "__main__": main()
