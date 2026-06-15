#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
YEAR="${YEAR:-2022}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data/prithvi_data}"

if [[ -z "${FINAL_OUT_DIR:-}" ]]; then
  : "${SCRATCH:?Set SCRATCH or FINAL_OUT_DIR before submitting.}"
  FINAL_OUT_DIR="$SCRATCH/embeddings-health/prithvi_embeddings"
fi

SCRIPT="$SCRIPT_DIR/run_prithvi_state_array.sbatch"
LOG_DIR="$SCRIPT_DIR/logs"
NUM_MODELS=2

STATES=()
while IFS= read -r state; do
  STATES+=("$state")
done < <(
  find "$DATA_DIR" -maxdepth 1 -type f -name "s2_spring_*_${YEAR}_prithvi.tif" -exec basename {} \; \
    | sed -E "s/^s2_spring_(.*)_${YEAR}_prithvi\.tif$/\1/" \
    | sort
)

if (( ${#STATES[@]} == 0 )); then
  echo "ERROR: no spring composites found in $DATA_DIR for year $YEAR" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$FINAL_OUT_DIR"

NUM_STATES="${#STATES[@]}"
MAX_TASK=$((NUM_STATES * NUM_MODELS - 1))

echo "Repo: $REPO_DIR"
echo "Data: $DATA_DIR"
echo "Final outputs: $FINAL_OUT_DIR"
echo "Year: $YEAR"
echo "States (${NUM_STATES}): ${STATES[*]}"
echo "Models: tiny 300M-TL"
echo "Array: 0-$MAX_TASK ($((MAX_TASK + 1)) tasks)"

export REPO_DIR DATA_DIR FINAL_OUT_DIR YEAR

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; not submitting."
  echo "Command: cd $REPO_DIR && sbatch --export=ALL --array=0-$MAX_TASK $SCRIPT"
  exit 0
fi

cd "$REPO_DIR"
sbatch --export=ALL --array="0-$MAX_TASK" "$SCRIPT"
