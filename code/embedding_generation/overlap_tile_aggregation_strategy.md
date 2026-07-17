# Overlapping Tile Inference and Tract Aggregation Strategy

This note documents a proposed production strategy for reducing chip-edge
effects while keeping the endpoint focused on census-tract embeddings.

## Summary

Use a fixed projected tile grid, run embedding inference over each tile with a
halo and overlapping chips, keep only tile-owned output pixels, aggregate those
pixels directly to intersecting census tracts, and then reduce tract partials
across tiles.

The main pipeline becomes:

```text
fixed tile grid
  -> read composite tile + halo
  -> run overlapping chip inference
  -> crop or blend overlapping predictions
  -> keep only the tile interior
  -> aggregate tile-owned pixels to tracts
  -> reduce per-tile tract partials into final tract rows
```

This avoids making a giant merged state embedding raster a required
intermediate.

## Tile Size

A practical default is **100 km x 100 km** tiles in a projected CRS.

At common input resolutions:

| Resolution | Pixels per 100 km tile |
|---:|---:|
| 10 m | 10,000 x 10,000 |
| 30 m | 3,334 x 3,334 |
| 40 m output | 2,500 x 2,500 |
| 80 m output | 1,250 x 1,250 |
| 480 m output | 209 x 209 |

Why 100 km:

- It is close to Sentinel-2 MGRS tile scale, but easier to define as a clean
  projected grid.
- It keeps compositing and inference jobs moderately sized.
- It gives predictable Slurm walltime and memory behavior.
- It makes halos easy because neighboring tiles are known from integer indices.

An alternative is to use native Sentinel-2 MGRS tiles, which are roughly
110 km x 110 km. MGRS has the advantage of matching Sentinel-2 acquisition
organization, but a custom 100 km UTM grid is simpler to make stable across
models and downstream products.

## Tile Reference System

Each tile should be referenced by projected coordinates, not by a lat/lon center
alone.

Recommended tile ID:

```text
utm{zone}_{epsg}_x{x_index}_y{y_index}_v{grid_version}
```

Example:

```text
utm10_EPSG32610_x005_y014_v1
```

The tile record should include:

- `tile_id`
- `grid_version`
- `crs`, for example `EPSG:32610`
- `utm_zone`
- `x_index`, `y_index`
- `xmin`, `ymin`, `xmax`, `ymax` in projected metres
- optional `lon_min`, `lat_min`, `lon_max`, `lat_max` for STAC search
- `tile_size_m`, for example `100000`

The projected bounds are authoritative. Lat/lon bounds are derived from them
for querying STAC catalogs and for human inspection.

## Defining the Fixed Grid

For each UTM zone covering CONUS:

1. Choose an origin aligned to a multiple of `tile_size_m`.
2. Transform the CONUS/state extent into the UTM CRS.
3. Generate all grid cells intersecting the target geography.
4. Store the full tile manifest.

A tile's projected bounds are:

```text
xmin = grid_origin_x + x_index * tile_size_m
xmax = xmin + tile_size_m
ymin = grid_origin_y + y_index * tile_size_m
ymax = ymin + tile_size_m
```

This is deterministic as long as `grid_origin_*`, `tile_size_m`, CRS, and
`grid_version` are fixed.

## Halo Size

When using overlapping chips, each processing task should read more than the
tile interior. This extra margin is the halo.

The halo must be large enough that predictions retained inside the tile interior
never depend on artificial padding or missing neighboring context.

For center-crop inference:

```text
halo_input_pixels >= discarded_edge_input_pixels
halo_metres = halo_input_pixels * input_resolution_m
```

Example for OlmoEarth-like 10 m input:

- chip size: 128 px
- stride: 64 px
- keep central 64 px
- discard 32 px on each side
- required halo: at least 32 input px = 320 m

In practice, round halos up to a simple value such as 512 m or 1 km so tile
edges are robust to model variants.

## Overlapping Inference Options

### Option A: Center Crop

Run overlapping chips, but only keep the central region of each chip's output.
Discard edge tokens.

This is the recommended default because:

- It is simple.
- Each output location has a single owner.
- It avoids double-weighting pixels during tract aggregation.
- It directly targets chip-edge artifacts.

### Option B: Weighted Blend

Run overlapping chips and blend predictions using weights that are largest near
the chip center and smallest near chip edges.

This can be smoother, but it requires accumulating:

```text
weighted_sum[band, row, col]
weight_sum[row, col]
```

Then:

```text
embedding = weighted_sum / weight_sum
```

This is still compatible with tile-level tract aggregation, but it is more
complex than center-crop.

### Option C: Treat Overlaps as Repeated Samples

This is not recommended. If overlapping predictions are directly aggregated as
independent pixels, tract summaries become implicitly weighted by chip coverage,
and central areas may count more than edge areas.

## Tile Ownership Rule

Even though each task reads `tile + halo`, it should only write or aggregate
pixels whose centers fall inside the tile's interior bounds:

```text
xmin <= pixel_center_x < xmax
ymin <= pixel_center_y < ymax
```

This prevents neighboring tile jobs from double-counting the same output
location.

For pixels exactly on a boundary, use half-open intervals as above so ownership
is deterministic.

## Census Tract Intersection

Before aggregation, build a tract/tile intersection table:

```text
tile_id, GEOID, intersection_geometry
```

Procedure:

1. Load tract geometries.
2. Reproject tracts to each tile CRS.
3. Select tracts whose geometry intersects the tile interior bounds.
4. Store the intersection geometry or store `GEOID` plus enough metadata to
   recompute the intersection.

During tile aggregation:

1. Open the tile-owned embedding pixels.
2. For each intersecting `GEOID`, rasterize the tract geometry onto the tile
   output grid.
3. Select pixels whose centers fall inside the tract.
4. Compute partial statistics for each embedding dimension.

The key is that aggregation is keyed by `GEOID`, not by county or state. A tract
that intersects multiple tiles will produce multiple partial rows with the same
`GEOID`.

## Partial Statistics

For each tile and tract, write a partial record:

```text
tile_id
GEOID
year
model
variant
dimension
count
sum
sumsq
min
max
```

These are sufficient to exactly reduce:

- mean
- minimum
- maximum
- standard deviation

Reduction across tiles:

```text
total_count = sum(count)
total_sum = sum(sum)
total_sumsq = sum(sumsq)
mean = total_sum / total_count
variance = total_sumsq / total_count - mean^2
std = sqrt(max(variance, 0))
min = min(tile_min)
max = max(tile_max)
```

This produces the final census-tract feature columns.

## Median

Median is the one statistic that does not reduce exactly from simple partials.

Options:

1. Omit median from routine production runs.
2. Use approximate quantile sketches.
3. Store per-tile pixel values and compute exact medians in a slower final pass.

For large-scale production, mean/std/min/max are much more natural for this
tile-reduction architecture.

## County Role

Counties can still be useful for reporting, QA, or scheduling summaries, but
they should not be the core raster processing unit.

Reasons:

- Counties are irregular polygons, while model inference prefers rectangular
  arrays.
- County sizes vary dramatically, which creates uneven Slurm tasks.
- County boundaries can force duplicated image reads and artificial chip edges.
- Fixed tiles are reusable across states, counties, tracts, models, and years.

Tracts can be merged correctly without counties because every partial statistic
is keyed by `GEOID` and reduced across all tiles intersecting that `GEOID`.

## Validation Checks

For each tile:

- confirm CRS and transform match the tile manifest
- confirm output pixel centers are within tile interior bounds
- confirm no duplicate `(tile_id, GEOID, dimension)` partial records
- record finite-pixel fraction and count per tract

For the final tract table:

- confirm every expected `GEOID` is present or explicitly listed as missing
- confirm tract pixel counts are positive
- confirm feature dimension count matches the model/PCA configuration
- confirm no duplicate `GEOID, year, model, variant` rows

## Recommended Default

Start with:

- 100 km fixed UTM tiles
- 1 km halo
- overlapping chips with center-crop ownership
- tile-level tract partials for mean/std/min/max
- state-level COG merging only as an optional QA artifact

This is the simplest design that supports smaller stride inference while
avoiding chip-edge artifacts and unnecessary large-state embedding merges.

## Optional Tract Tensor Endpoint

The tile strategy can also support methods that need the spatial structure of
each tract, such as a CNN head over the tract embedding field.

In that case, keep tile embeddings and add a second endpoint:

```text
tile embeddings
  -> tract tensor crops
  -> downstream spatial models
```

For each tract, write an artifact containing:

- `GEOID`
- embedding tensor crop, for example `(height, width, embedding_dim)`
- binary tract mask with the same `(height, width)` spatial shape
- affine transform for the crop
- CRS
- model, variant, year, PCA/raw embedding metadata
- source `tile_id` list

The mask is important. Census tracts are irregular polygons, while CNNs expect
rectangular tensors. Pixels outside the tract but inside the crop should be
masked so padding or neighboring geography is not treated as valid signal.

### Expected Size

Approximate average tract area is around 40 km2, though the distribution has a
long rural tail.

Approximate tensor sizes for an average tract:

| Embedding product | Output resolution | Approx. pixels | Raw float32 tensor |
|---|---:|---:|---:|
| 64-d PCA at 40 m | 40 m | 25,000 | ~6 MB |
| 768-d raw at 40 m | 40 m | 25,000 | ~77 MB |
| 1024-d raw at 80 m | 80 m | 6,250 | ~26 MB |
| 192-d raw at 480 m | 480 m | 170 | ~0.13 MB |

Most tracts should intersect one 100 km tile. Some will cross two tiles, and a
small number near tile corners may cross four. Large rural tracts are the main
outliers and may need special handling, such as chunked crops or maximum-size
guards.

### How to Build Tract Tensors

Do not reconstruct tract tensors by repeatedly opening large merged state
rasters. Instead, build them from tile embeddings:

1. For each tile, find intersecting tracts.
2. Read the tile-owned embedding pixels.
3. For each tract, crop the minimal bounding window covering the tract
   intersection.
4. Rasterize the tract geometry into a mask on that crop grid.
5. Write or append the crop to a tract-level artifact.
6. If the tract crosses multiple tiles, stitch the tile-local crops into one
   tract crop using projected coordinates.

This is efficient because reads are tile-local and can be batched by tile.

### Storage Formats

Recommended formats:

- **Zarr** for chunked tensor storage and repeated random access.
- **WebDataset/tar shards** for PyTorch training workflows.
- **COG** only when a tract crop needs GIS inspection.

Parquet is still appropriate for summary statistics, but it is awkward for
image-like tensors.

### Relationship to Summary Stats

The tensor endpoint should be parallel to, not a replacement for, the summary
statistics endpoint:

```text
tile embeddings -> tract summary stats
tile embeddings -> tract tensor crops
```

This allows conventional regressions, tree models, and spatial neural models to
use the same underlying embedding tiles.
