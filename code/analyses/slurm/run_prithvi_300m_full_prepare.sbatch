#!/bin/bash
#SBATCH --job-name=prithvi-300m-prep
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

: "${SCRATCH:?SCRATCH must be set}"
: "${RUN_ROOT:?RUN_ROOT must be set}"

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
ROOT="$SCRATCH/embeddings-health"
STATE_DIR="${STATE_DIR:-$ROOT/prithvi_aggregated_full/300M-TL}"
COMBINED="${COMBINED:-$ROOT/prithvi_aggregated_full/prithvi_300M-TL_2022_all_tracts.csv}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/cache/prithvi_300m_tl_full_prepared}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$ROOT/outputs/prithvi_300m_tl_full}"
PYTHON="${PYTHON:-$ROOT/cache/venv-3.11-cpu/bin/python}"
AREA_SOURCE="${AREA_SOURCE:-$ROOT/alphaearth/alphaearth_embeddings.csv}"

mkdir -p "$RUN_ROOT" "$PREPARED_DIR" "$OUTPUTS_DIR" "$(dirname "$COMBINED")"

state_files=$(find "$STATE_DIR" -maxdepth 1 -type f -name 'prithvi_300M-TL_*_2022_tracts.csv' -size +0c | wc -l)
if (( state_files != 49 )); then
  echo "ERROR: expected 49 nonempty state aggregates, found $state_files" >&2
  exit 1
fi

if [[ ! -s "$COMBINED" ]]; then
  "$PYTHON" "$REPO_DIR/code/analyses/prepare_embeddings.py" \
    --input-dir "$STATE_DIR" \
    --output "$COMBINED" \
    --no-area
fi

TABLE="$PREPARED_DIR/analysis_data.parquet"
METADATA="$PREPARED_DIR/metadata.json"
if [[ ! -s "$TABLE" || ! -s "$METADATA" || "$COMBINED" -nt "$TABLE" || "$COMBINED" -nt "$METADATA" ]]; then
  "$PYTHON" "$REPO_DIR/code/analyses/analyses_sherlock.py" \
    --model prithvi_300m_tl \
    --embeddings-path "$COMBINED" \
    --outputs-dir "$OUTPUTS_DIR" \
    --stage prepare \
    --prepared-dir "$PREPARED_DIR"
else
  echo "Prepared data are current: $PREPARED_DIR"
fi

"$PYTHON" "$REPO_DIR/code/analyses/analyses_sharded.py" plan \
  --prepared-dir "$PREPARED_DIR" \
  --run-root "$RUN_ROOT" \
  --q1-batch-size "${Q1_BATCH_SIZE:-10}" \
  --q23-batch-size "${Q23_BATCH_SIZE:-5}"

echo "Preparation complete"
echo "Combined: $COMBINED"
echo "Prepared: $PREPARED_DIR"
echo "Plan:     $RUN_ROOT/plan.json"
