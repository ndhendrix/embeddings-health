#!/bin/bash
#SBATCH --job-name=emb-health-oe-base
#SBATCH -p normal
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=code/analyses/slurm/logs/oe_base_%j.out
#SBATCH --error=code/analyses/slurm/logs/oe_base_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?SCRATCH env var is not set}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
STATE_ROOT="$SCRATCH/embeddings-health/embedding_workflow_overlap_v1/olmoearth-v1.2-base"
AGG_DIR="$SCRATCH/embeddings-health/olmoearth_base_aggregated"
FLAT_DIR="$AGG_DIR/state_csvs"
COMBINED="$AGG_DIR/olmoearth_v1.2_base_2022_all_tracts.csv"

mkdir -p "$CACHE_ROOT" "$AGG_DIR" "$FLAT_DIR" "$REPO_DIR/code/analyses/slurm/logs"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-analysis"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

module load devel
module load gcc/14.2.0
export CC="$(command -v gcc)" CXX="$(command -v g++)"

state_csvs=("$STATE_ROOT"/*/olmoearth-v1.2-base_overlap-center50_*_2022_tracts.csv)
qa_reports=("$STATE_ROOT"/*/olmoearth-v1.2-base_overlap-center50_*_2022_tracts.validation.json)
(( ${#state_csvs[@]} == 49 )) || { echo "ERROR: expected 49 state CSVs, found ${#state_csvs[@]}" >&2; exit 1; }
(( ${#qa_reports[@]} == 49 )) || { echo "ERROR: expected 49 QA reports, found ${#qa_reports[@]}" >&2; exit 1; }

jq -e -s '
  length == 49 and
  all(.[]; (.status | startswith("complete"))) and
  all(.[]; .embedding_dimensions == 768) and
  all(.[]; .year == 2022) and
  all(.[]; .unexpected_tracts == 0) and
  all(.[]; (.duplicate_expected_geoids | length) == 0) and
  ([.[].expected_tracts] | add) == 83776 and
  ([.[].tracts_with_valid_pixels] | add) == 82506 and
  ([.[].zero_valid_pixel_tracts] | add) == 1270
' "${qa_reports[@]}" >/dev/null

for csv in "${state_csvs[@]}"; do
  ln -sfn "$csv" "$FLAT_DIR/$(basename "$csv")"
done
flat_csvs=("$FLAT_DIR"/*.csv)
(( ${#flat_csvs[@]} == 49 )) || { echo "ERROR: expected 49 staged state CSVs, found ${#flat_csvs[@]}" >&2; exit 1; }

validate_combined() {
  local path="$1" lines columns
  [[ -s "$path" ]] || return 1
  lines=$(wc -l < "$path")
  columns=$(awk -F',' 'NR == 1 {print NF; exit}' "$path")
  (( lines == 83777 && columns == 3075 ))
}

echo "=== OlmoEarth Base analysis ==="
echo "REPO_DIR    : $REPO_DIR"
echo "SCRATCH     : $SCRATCH"
echo "STATE_ROOT  : $STATE_ROOT"
echo "EMBEDDINGS  : $COMBINED"
echo "OUTPUTS     : $SCRATCH/embeddings-health/outputs/olmoearth_base"

cd "$REPO_DIR/code/analyses"

if [[ ! -s "$COMBINED" ]]; then
  temp_combined="${COMBINED}.tmp.${SLURM_JOB_ID}"
  trap 'rm -f "$temp_combined"' EXIT
  echo "--- Combining 49 OE-Base state CSVs ---"
  uv run --python 3.11 python prepare_embeddings.py \
    --input-dir "$FLAT_DIR" \
    --output "$temp_combined" \
    --no-area
  validate_combined "$temp_combined" || {
    echo "ERROR: combined OE-Base CSV failed row/column QA" >&2
    exit 1
  }
  mv "$temp_combined" "$COMBINED"
  trap - EXIT
else
  echo "Combined file already exists: $COMBINED"
fi

validate_combined "$COMBINED" || {
  echo "ERROR: existing combined OE-Base CSV failed row/column QA" >&2
  exit 1
}

echo "--- Running full OE-Base analysis ---"
uv run --python 3.11 python analyses_sherlock.py \
  --model olmoearth_base \
  --no-cache

echo "=== Done ==="
