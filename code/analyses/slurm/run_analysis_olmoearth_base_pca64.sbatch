#!/bin/bash
#SBATCH --job-name=emb-health-oe-base-pca64
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=code/analyses/slurm/logs/oe_base_pca64_%j.out
#SBATCH --error=code/analyses/slurm/logs/oe_base_pca64_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?SCRATCH env var is not set}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
AGG_DIR="$SCRATCH/embeddings-health/olmoearth_base_aggregated_pca64"
COMBINED="$AGG_DIR/olmoearth_v1.2_base_pca64_2022_all_tracts.csv"

mkdir -p "$CACHE_ROOT" "$AGG_DIR" "$REPO_DIR/code/analyses/slurm/logs"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-analysis"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

module load devel
module load gcc/14.2.0
export CC="$(command -v gcc)" CXX="$(command -v g++)"

state_csvs=("$AGG_DIR"/olmoearth-v1.2-base_overlap-center50_*_2022_pca64_tracts.csv)
qa_reports=("$AGG_DIR"/olmoearth-v1.2-base_overlap-center50_*_2022_pca64_tracts.validation.json)
(( ${#state_csvs[@]} == 49 )) || { echo "ERROR: expected 49 state CSVs, found ${#state_csvs[@]}" >&2; exit 1; }
(( ${#qa_reports[@]} == 49 )) || { echo "ERROR: expected 49 QA reports, found ${#qa_reports[@]}" >&2; exit 1; }

jq -e -s '
  length == 49 and
  all(.[]; (.status | startswith("complete"))) and
  all(.[]; .embedding_dimensions == 64) and
  all(.[]; .year == 2022) and
  all(.[]; .unexpected_tracts == 0) and
  all(.[]; (.duplicate_expected_geoids | length) == 0) and
  ([.[].expected_tracts] | add) == 83776 and
  ([.[].tracts_with_valid_pixels] | add) == 82506 and
  ([.[].zero_valid_pixel_tracts] | add) == 1270
' "${qa_reports[@]}" >/dev/null

validate_combined() {
  local path="$1" lines columns
  [[ -s "$path" ]] || return 1
  lines=$(wc -l < "$path")
  columns=$(awk -F',' 'NR == 1 {print NF; exit}' "$path")
  (( lines == 83777 && columns == 259 ))
}

echo "=== OlmoEarth Base PCA64 analysis ==="
echo "REPO_DIR    : $REPO_DIR"
echo "SCRATCH     : $SCRATCH"
echo "AGG_DIR     : $AGG_DIR"
echo "EMBEDDINGS  : $COMBINED"
echo "OUTPUTS     : $SCRATCH/embeddings-health/outputs/olmoearth_base_pca64"

cd "$REPO_DIR/code/analyses"
if [[ ! -s "$COMBINED" ]]; then
  temporary="${COMBINED}.tmp.${SLURM_JOB_ID}"
  trap 'rm -f "$temporary"' EXIT
  uv run --python 3.11 python prepare_embeddings.py \
    --input-dir "$AGG_DIR" \
    --output "$temporary" \
    --no-area
  validate_combined "$temporary" || {
    echo "ERROR: combined PCA64 CSV failed row/column QA" >&2
    exit 1
  }
  mv "$temporary" "$COMBINED"
  trap - EXIT
else
  echo "Combined file already exists: $COMBINED"
fi
validate_combined "$COMBINED" || {
  echo "ERROR: existing PCA64 CSV failed row/column QA" >&2
  exit 1
}

uv run --python 3.11 python analyses_sherlock.py \
  --model olmoearth_base_pca64 \
  --no-cache

echo "=== Done ==="
