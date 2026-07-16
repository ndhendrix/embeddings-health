#!/bin/bash
#SBATCH --job-name=clay-aggregate
#SBATCH -p normal
#SBATCH -c 25
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=code/embedding_generation/slurm/logs/clay_aggregate_%A_%a.out
#SBATCH --error=code/embedding_generation/slurm/logs/clay_aggregate_%A_%a.err

# Per-state Clay tract aggregation, mirroring run_aggregate_state_array.sbatch
# (Prithvi) but for Clay's flat, single-variant layout (no tiny/300M-TL split).
#
# STATE_LIST must be set by the submit script: colon-separated sorted list of
# all states to consider for aggregation. SLURM_ARRAY_TASK_ID indexes into it.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"
TRACT_YEAR="${TRACT_YEAR:-2020}"   # TIGER/Line vintage

: "${SCRATCH:?SCRATCH is not set}"
EMBED_OUT_DIR="${EMBED_OUT_DIR:-$SCRATCH/embeddings-health/clay_embeddings}"
TRACT_DIR="${TRACT_DIR:-$SCRATCH/embeddings-health/data/census_tracts}"
AGG_OUT_DIR="${AGG_OUT_DIR:-$SCRATCH/embeddings-health/clay_aggregated}"
NATIONAL_PCA_DIR="${NATIONAL_PCA_DIR:-$SCRATCH/embeddings-health/clay_aggregated/national_pca}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

: "${STATE_LIST:?STATE_LIST must be set to a colon-separated ordered state list}"
IFS=: read -ra STATES <<< "$STATE_LIST"

if (( TASK_ID < 0 || TASK_ID >= ${#STATES[@]} )); then
  echo "ERROR: TASK_ID=$TASK_ID out of range (${#STATES[@]} states)" >&2
  exit 1
fi

STATE="${STATES[$TASK_ID]}"

declare -A STATE_FIPS=(
  [AL]=01 [AZ]=04 [AR]=05 [CA]=06 [CO]=08 [CT]=09
  [DC]=11 [DE]=10 [FL]=12 [GA]=13 [ID]=16 [IL]=17
  [IN]=18 [IA]=19 [KS]=20 [KY]=21 [LA]=22 [ME]=23
  [MD]=24 [MA]=25 [MI]=26 [MN]=27 [MS]=28 [MO]=29
  [MT]=30 [NE]=31 [NV]=32 [NH]=33 [NJ]=34 [NM]=35
  [NY]=36 [NC]=37 [ND]=38 [OH]=39 [OK]=40 [OR]=41
  [PA]=42 [RI]=44 [SC]=45 [SD]=46 [TN]=47 [TX]=48
  [UT]=49 [VT]=50 [VA]=51 [WA]=53 [WI]=55 [WV]=54 [WY]=56
)

FIPS="${STATE_FIPS[$STATE]:?No FIPS code for '$STATE'}"
TRACT_ZIP="$TRACT_DIR/tl_${TRACT_YEAR}_${FIPS}_tract.zip"

if [[ ! -f "$TRACT_ZIP" ]]; then
  echo "ERROR: tract file not found: $TRACT_ZIP" >&2
  echo "  Run: sbatch download_census_tracts.sbatch" >&2
  exit 1
fi

mkdir -p "$AGG_OUT_DIR" "$CACHE_ROOT"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-cpu"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if command -v module >/dev/null 2>&1; then
  module load devel
  module load gcc/14.2.0
fi
export CC="$(command -v gcc)" CXX="$(command -v g++)"

if ! command -v uv >/dev/null 2>&1; then
  UV_INSTALL_DIR="${UV_INSTALL_DIR:-$CACHE_ROOT/uv-bin}"
  if [[ -x "$UV_INSTALL_DIR/uv" ]]; then
    export PATH="$UV_INSTALL_DIR:$PATH"
  else
    echo "ERROR: uv not found and cannot be bootstrapped. Run an embedding job first." >&2
    exit 1
  fi
fi

echo "=== Clay tract aggregation ==="
echo "State:  $STATE ($FIPS)"
echo "Tracts: $TRACT_ZIP"
echo "Year:   $YEAR"

cd "$EMBED_DIR"

EMBED_TIF="$EMBED_OUT_DIR/$STATE/clay_v1.5_${STATE}_${YEAR}.tif"
if [[ ! -s "$EMBED_TIF" ]]; then
  echo "SKIP $STATE — no complete embedding found at $EMBED_TIF"
  exit 0
fi

AGG_CSV="$AGG_OUT_DIR/clay_v1.5_${STATE}_${YEAR}_tracts.csv"
if [[ -s "$AGG_CSV" ]]; then
  echo "SKIP $STATE — already aggregated ($AGG_CSV)"
  exit 0
fi

# Use the national PCA model when available so all states share the same embedding space.
PCA_MODEL="$NATIONAL_PCA_DIR/clay_v1.5_national_pca.pkl"
PCA_MODEL_ARG=""
if [[ -f "$PCA_MODEL" ]]; then
  PCA_MODEL_ARG="--pca-model $PCA_MODEL"
else
  echo "  NOTE: national PCA not found at $PCA_MODEL — aggregating raw dimensions"
fi

mkdir -p "$(dirname "$AGG_CSV")"
echo ""
echo "--- $STATE  (embedding: $(basename "$EMBED_TIF")) ---"
# shellcheck disable=SC2086
uv run --python 3.11 python aggregate.py \
  --embedding   "$EMBED_TIF" \
  --tracts      "$TRACT_ZIP" \
  --output      "$AGG_CSV" \
  --year        "$YEAR" \
  --model       clay \
  $PCA_MODEL_ARG

echo "Saved: $AGG_CSV  ($(wc -l < "$AGG_CSV") rows)"
echo ""
echo "=== Done: $STATE ==="
