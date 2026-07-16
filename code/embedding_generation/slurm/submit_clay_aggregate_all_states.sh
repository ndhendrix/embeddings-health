#!/bin/bash
# Submit Clay tract aggregation for all states with a complete embedding,
# mirroring submit_aggregate_all_states.sh (Prithvi) but for Clay's flat,
# single-variant layout. Safe to re-run — states with existing CSVs are
# skipped by the array job.
#
# Prerequisites:
#   1. sbatch download_census_tracts.sbatch  (downloads TIGER/Line tract ZIPs)
#   2. At least some Clay embedding jobs complete (clay_embeddings/<STATE>/)
#   3. sbatch fit_clay_national_pca.sbatch (recommended, for cross-state comparability;
#      aggregation falls back to raw 1024-dim output if the PCA model isn't found yet)
#
# Usage:
#   bash submit_clay_aggregate_all_states.sh
#   DRY_RUN=1 bash submit_clay_aggregate_all_states.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
YEAR="${YEAR:-2022}"

: "${SCRATCH:?Set SCRATCH before submitting.}"
EMBED_OUT_DIR="${EMBED_OUT_DIR:-$SCRATCH/embeddings-health/clay_embeddings}"
AGG_OUT_DIR="${AGG_OUT_DIR:-$SCRATCH/embeddings-health/clay_aggregated}"
NATIONAL_PCA_DIR="${NATIONAL_PCA_DIR:-$SCRATCH/embeddings-health/clay_aggregated/national_pca}"
TRACT_DIR="${TRACT_DIR:-$SCRATCH/embeddings-health/data/census_tracts}"
LOG_DIR="${LOG_DIR:-$SCRATCH/embeddings-health/logs}"
SCRIPT="$SCRIPT_DIR/run_clay_aggregate_state_array.sbatch"

# Discover states with a complete Clay embedding.
STATES=()
for state_dir in "$EMBED_OUT_DIR"/*/; do
  [[ -d "$state_dir" ]] || continue
  state=$(basename "$state_dir")
  tif="$state_dir/clay_v1.5_${state}_${YEAR}.tif"
  [[ -s "$tif" ]] && STATES+=("$state")
done

IFS=$'\n' STATES=($(printf '%s\n' "${STATES[@]}" | sort -u)); unset IFS

if (( ${#STATES[@]} == 0 )); then
  echo "No states with complete Clay embeddings found in $EMBED_OUT_DIR"
  exit 1
fi

# Check which states still need aggregation.
INCOMPLETE_TASKS=()
for i in "${!STATES[@]}"; do
  state="${STATES[$i]}"
  csv="$AGG_OUT_DIR/clay_v1.5_${state}_${YEAR}_tracts.csv"
  [[ -s "$csv" ]] || INCOMPLETE_TASKS+=("$i")
done

STATE_LIST=$(IFS=:; echo "${STATES[*]}")

echo "Repo:        $REPO_DIR"
echo "Embeddings:  $EMBED_OUT_DIR"
echo "Tracts:      $TRACT_DIR"
echo "Output:      $AGG_OUT_DIR"
echo "Year:        $YEAR"
echo ""
echo "States with complete embeddings: ${#STATES[@]}"
echo "Tasks needing aggregation:       ${#INCOMPLETE_TASKS[@]}"
echo ""

PCA_STATUS="missing"
[[ -f "$NATIONAL_PCA_DIR/clay_v1.5_national_pca.pkl" ]] && PCA_STATUS="ready"
echo "National PCA: $PCA_STATUS ($NATIONAL_PCA_DIR/clay_v1.5_national_pca.pkl)"
echo ""

if (( ${#INCOMPLETE_TASKS[@]} == 0 )); then
  echo "All Clay aggregations complete. Nothing to submit."
  exit 0
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; not submitting."
  echo "Incomplete states:"
  for i in "${INCOMPLETE_TASKS[@]}"; do
    printf "  [%2d] %s\n" "$i" "${STATES[$i]}"
  done
  exit 0
fi

mkdir -p "$LOG_DIR" "$AGG_OUT_DIR"
TASK_ARRAY=$(IFS=,; echo "${INCOMPLETE_TASKS[*]}")

export REPO_DIR EMBED_OUT_DIR AGG_OUT_DIR NATIONAL_PCA_DIR TRACT_DIR YEAR STATE_LIST

JOB_ID=$(cd "$REPO_DIR" && sbatch \
  --export=ALL \
  --array="$TASK_ARRAY" \
  --output="$LOG_DIR/clay_aggregate_%A_%a.out" \
  --error="$LOG_DIR/clay_aggregate_%A_%a.err" \
  --parsable \
  "$SCRIPT" | cut -d';' -f1)

echo ""
echo "Submitted Clay aggregation job $JOB_ID  (${#INCOMPLETE_TASKS[@]} tasks)"
echo "Logs: $LOG_DIR/clay_aggregate_${JOB_ID}_*.out"
