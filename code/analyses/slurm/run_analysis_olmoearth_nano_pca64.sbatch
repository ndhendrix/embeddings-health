#!/bin/bash
#SBATCH --job-name=emb-health-oe-nano
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=code/analyses/slurm/logs/oe_nano_pca64_%j.out
#SBATCH --error=code/analyses/slurm/logs/oe_nano_pca64_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?SCRATCH env var is not set}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
ANALYSIS_MODEL="olmoearth_nano_pca64"
AGG_DIR="$SCRATCH/embeddings-health/olmoearth_nano_aggregated_pca64"
COMBINED="$AGG_DIR/olmoearth_v1.2_nano_pca64_2022_all_tracts.csv"

mkdir -p "$CACHE_ROOT" "$REPO_DIR/code/analyses/slurm/logs"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-analysis"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

module load devel
module load gcc/14.2.0
export CC="$(command -v gcc)" CXX="$(command -v g++)"

state_csvs=("$AGG_DIR"/olmoearth-v1.2-nano_overlap-center50_*_2022_pca64_tracts.csv)
qa_reports=("$AGG_DIR"/olmoearth-v1.2-nano_overlap-center50_*_2022_pca64_tracts.validation.json)
(( ${#state_csvs[@]} == 49 )) || { echo "ERROR: expected 49 state CSVs, found ${#state_csvs[@]}" >&2; exit 1; }
(( ${#qa_reports[@]} == 49 )) || { echo "ERROR: expected 49 QA reports, found ${#qa_reports[@]}" >&2; exit 1; }
jq -e -s '
  length == 49 and
  all(.[]; (.status | startswith("complete"))) and
  all(.[]; .embedding_dimensions == 64) and
  all(.[]; .unexpected_tracts == 0) and
  all(.[]; (.duplicate_expected_geoids | length) == 0)
' "${qa_reports[@]}" >/dev/null
[[ -s "$COMBINED" ]] || { echo "ERROR: missing combined embeddings: $COMBINED" >&2; exit 1; }

echo "=== OlmoEarth Nano PCA-64 analysis ==="
echo "REPO_DIR    : $REPO_DIR"
echo "SCRATCH     : $SCRATCH"
echo "AGG_DIR     : $AGG_DIR"
echo "EMBEDDINGS  : $COMBINED"
echo "OUTPUTS     : $SCRATCH/embeddings-health/outputs/$ANALYSIS_MODEL"

cd "$REPO_DIR/code/analyses"
uv run --python 3.11 python analyses_sherlock.py --model "$ANALYSIS_MODEL"

echo "=== Done ==="
