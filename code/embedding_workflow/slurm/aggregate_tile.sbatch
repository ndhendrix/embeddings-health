#!/bin/bash
#SBATCH --job-name=tract-partial
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"; : "${SCRATCH:?}"; : "${MODEL:?}"; : "${STATE:?}"; : "${NUM_TILES:?}"
YEAR="${YEAR:-2022}"; TRACT_YEAR="${TRACT_YEAR:-2020}"; INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -n "${TASK_FILE:-}" ]]; then read -r STATE TILE_INDEX NUM_TILES TASK_GRID_ROWS TASK_GRID_COLS <<< "$(sed -n "$((INDEX+1))p" "$TASK_FILE")"; INDEX="$TILE_INDEX"; fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow}"; CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
declare -A FIPS=([RI]=44 [AL]=01 [AZ]=04 [AR]=05 [CA]=06 [CO]=08 [CT]=09 [DE]=10 [DC]=11 [FL]=12 [GA]=13 [ID]=16 [IL]=17 [IN]=18 [IA]=19 [KS]=20 [KY]=21 [LA]=22 [ME]=23 [MD]=24 [MA]=25 [MI]=26 [MN]=27 [MS]=28 [MO]=29 [MT]=30 [NE]=31 [NV]=32 [NH]=33 [NJ]=34 [NM]=35 [NY]=36 [NC]=37 [ND]=38 [OH]=39 [OK]=40 [OR]=41 [PA]=42 [SC]=45 [SD]=46 [TN]=47 [TX]=48 [UT]=49 [VT]=50 [VA]=51 [WA]=53 [WV]=54 [WI]=55 [WY]=56)
if [[ "${PRODUCT_MODE:-}" == overlap ]]; then PRODUCT_STEM="${MODEL}_overlap-center50_${STATE}_${YEAR}"; else PRODUCT_STEM="${PRODUCT_STEM:-${MODEL}_${STATE}_${YEAR}}"; fi
AGG_SUFFIX="${AGG_SUFFIX:-}"
PARTIAL_STEM="${PARTIAL_STEM:-${PRODUCT_STEM}${AGG_SUFFIX}}"
printf -v suffix "_tile%03d.tif" "$INDEX"; TILE="$OUTPUT_ROOT/$MODEL/$STATE/${PRODUCT_STEM}${suffix}"
(( NUM_TILES == 1 )) && TILE="$OUTPUT_ROOT/$MODEL/$STATE/${PRODUCT_STEM}.tif"
PARTIAL="$OUTPUT_ROOT/$MODEL/$STATE/partials/$PARTIAL_STEM/tile$(printf '%03d' "$INDEX").npz"
TRACTS="${TRACTS:-$SCRATCH/embeddings-health/data/census_tracts/tl_${TRACT_YEAR}_${FIPS[$STATE]}_tract.zip}"
module load devel; module load gcc/14.2.0
export UV_CACHE_DIR="$CACHE_ROOT/uv" UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-cpu" GDAL_CACHEMAX="${GDAL_CACHEMAX:-512}"
cd "$REPO_DIR/code/embedding_generation"; uv sync --python 3.11
PCA_ARGS=()
if [[ -n "${PCA_MODEL:-}" ]]; then
  [[ -s "$PCA_MODEL" ]] || { echo "ERROR: PCA_MODEL not found: $PCA_MODEL" >&2; exit 1; }
  PCA_ARGS=(--pca-model "$PCA_MODEL")
fi
uv run --python 3.11 python ../embedding_workflow/aggregate_tiles.py partial --tile "$TILE" --tracts "$TRACTS" --output "$PARTIAL" --band-chunk "${BAND_CHUNK:-64}" "${PCA_ARGS[@]}"
