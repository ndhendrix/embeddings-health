#!/bin/bash
#SBATCH --job-name=embed-ri-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/embed_ri_%j.out
#SBATCH --error=logs/embed_ri_%j.err

set -euo pipefail

# Adjust REPO_DIR to wherever you've cloned this project on the cluster.
REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
DATA_DIR="$EMBED_DIR/outputs"
YEAR=2022
STATE=RI

cd "$EMBED_DIR"
mkdir -p logs "$DATA_DIR/composites" "$DATA_DIR/embeddings"

echo "=== Step 1: S2 composite (OlmoEarth) ==="
uv run python composite.py --state $STATE --year $YEAR --model olmoearth \
  --output-dir "$DATA_DIR/composites"

echo "=== Step 2: S2 composites (Prithvi) ==="
uv run python composite.py --state $STATE --year $YEAR --model prithvi \
  --output-dir "$DATA_DIR/composites"

echo "=== Step 3: OlmoEarth inference ==="
uv run python embed.py --model olmoearth \
  --input "$DATA_DIR/composites/s2_annual_${STATE}_${YEAR}_olmoearth.tif" \
  --output "$DATA_DIR/embeddings/olmoearth_${STATE}_${YEAR}.tif" \
  --batch-size 32

echo "=== Step 4: Prithvi inference ==="
uv run python embed.py --model prithvi \
  --input "$DATA_DIR/composites/s2_spring_${STATE}_${YEAR}_prithvi.tif" \
         "$DATA_DIR/composites/s2_summer_${STATE}_${YEAR}_prithvi.tif" \
         "$DATA_DIR/composites/s2_fall_${STATE}_${YEAR}_prithvi.tif" \
  --output "$DATA_DIR/embeddings/prithvi_${STATE}_${YEAR}.tif" \
  --batch-size 32

echo "=== Step 5: Aggregate to census tracts ==="
# Download RI census tracts if not present
TRACTS="$DATA_DIR/tracts/tl_2020_44_tract.gpkg"
if [ ! -f "$TRACTS" ]; then
  mkdir -p "$DATA_DIR/tracts"
  echo "Downloading RI census tract shapefile…"
  curl -L "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_44_tract.zip" \
       -o /tmp/ri_tracts.zip
  unzip -o /tmp/ri_tracts.zip -d /tmp/ri_tracts/
  uv run python -c "
import geopandas as gpd
gdf = gpd.read_file('/tmp/ri_tracts/tl_2020_44_tract.shp')
gdf.to_file('$TRACTS', driver='GPKG')
print('Saved', len(gdf), 'tracts')
"
fi

uv run python aggregate.py \
  --embedding "$DATA_DIR/embeddings/olmoearth_${STATE}_${YEAR}.tif" \
  --tracts "$TRACTS" \
  --output "$REPO_DIR/data/olmoearth_embeddings_${STATE}_${YEAR}.csv" \
  --year $YEAR --model olmoearth

uv run python aggregate.py \
  --embedding "$DATA_DIR/embeddings/prithvi_${STATE}_${YEAR}.tif" \
  --tracts "$TRACTS" \
  --output "$REPO_DIR/data/prithvi_embeddings_${STATE}_${YEAR}.csv" \
  --year $YEAR --model prithvi

echo "=== All done ==="
