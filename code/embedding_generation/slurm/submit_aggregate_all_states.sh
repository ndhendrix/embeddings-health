#!/bin/bash
# Submit tract aggregation for all states with at least one complete embedding.
# Safe to re-run — states with existing CSVs are skipped by the array job.
#
# Prerequisites:
#   1. sbatch download_census_tracts.sbatch  (downloads TIGER/Line tract ZIPs)
#   2. At least some embedding jobs complete  (prithvi_embeddings/tiny/ or 300M-TL/)
#
# Usage:
#   bash submit_aggregate_all_states.sh
#   DRY_RUN=1 bash submit_aggregate_all_states.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
YEAR="${YEAR:-2022}"

: "${SCRATCH:?Set SCRATCH before submitting.}"
EMBED_OUT_DIR="${EMBED_OUT_DIR:-$SCRATCH/embeddings-health/prithvi_embeddings}"
AGG_OUT_DIR="${AGG_OUT_DIR:-$SCRATCH/embeddings-health/prithvi_aggregated}"
NATIONAL_PCA_DIR="${NATIONAL_PCA_DIR:-$SCRATCH/embeddings-health/prithvi_aggregated/national_pca}"
TRACT_DIR="${TRACT_DIR:-$SCRATCH/embeddings-health/data/census_tracts}"
LOG_DIR="${LOG_DIR:-$SCRATCH/embeddings-health/logs}"
SCRIPT="$SCRIPT_DIR/run_aggregate_state_array.sbatch"

MODEL_VARIANTS=("tiny" "300M-TL")

# Discover states with at least one complete embedding variant
STATES=()
_has_raw_tif() {
  local variant="$1" state="$2" safe_var="$3"
  local raw="$EMBED_OUT_DIR/$variant/$state/prithvi_${safe_var}_${state}_${YEAR}_raw.tif"
  local plain="$EMBED_OUT_DIR/$variant/$state/prithvi_${safe_var}_${state}_${YEAR}.tif"
  [[ -s "$raw" || -s "$plain" ]]
}

for state_dir in "$EMBED_OUT_DIR/tiny"/*/; do
  [[ -d "$state_dir" ]] || continue
  state=$(basename "$state_dir")
  for VARIANT in "${MODEL_VARIANTS[@]}"; do
    SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"
    if _has_raw_tif "$VARIANT" "$state" "$SAFE_VARIANT"; then
      STATES+=("$state")
      break  # at least one variant done — include this state
    fi
  done
done

# Sort and deduplicate
IFS=$'\n' STATES=($(printf '%s\n' "${STATES[@]}" | sort -u)); unset IFS

if (( ${#STATES[@]} == 0 )); then
  echo "No states with complete embeddings found in $EMBED_OUT_DIR"
  exit 1
fi

# Check which states still need aggregation (at least one variant incomplete)
INCOMPLETE_TASKS=()
for i in "${!STATES[@]}"; do
  state="${STATES[$i]}"
  needs_work=0
  for VARIANT in "${MODEL_VARIANTS[@]}"; do
    SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"
    csv="$AGG_OUT_DIR/$VARIANT/prithvi_${SAFE_VARIANT}_${state}_${YEAR}_tracts.csv"
    if _has_raw_tif "$VARIANT" "$state" "$SAFE_VARIANT" && [[ ! -s "$csv" ]]; then
      needs_work=1
      break
    fi
  done
  (( needs_work )) && INCOMPLETE_TASKS+=("$i")
done

# Build STATE_LIST for export (colon-separated, same order as STATES array)
STATE_LIST=$(IFS=:; echo "${STATES[*]}")

echo "Repo:        $REPO_DIR"
echo "Embeddings:  $EMBED_OUT_DIR"
echo "Tracts:      $TRACT_DIR"
echo "Output:      $AGG_OUT_DIR"
echo "Year:        $YEAR"
echo ""
echo "States with any complete embedding: ${#STATES[@]}"
echo "Tasks needing aggregation:          ${#INCOMPLETE_TASKS[@]}"
echo ""

if (( ${#INCOMPLETE_TASKS[@]} == 0 )); then
  echo "All aggregations complete. Nothing to submit."
  exit 0
fi

echo "Incomplete states:"
for i in "${INCOMPLETE_TASKS[@]}"; do
  state="${STATES[$i]}"
  printf "  [%2d] %s\n" "$i" "$state"
  for VARIANT in "${MODEL_VARIANTS[@]}"; do
    SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"
    csv="$AGG_OUT_DIR/$VARIANT/prithvi_${SAFE_VARIANT}_${state}_${YEAR}_tracts.csv"
    tif_done=$( _has_raw_tif "$VARIANT" "$state" "$SAFE_VARIANT" && echo "embed=done" || echo "embed=pending" )
    csv_done=$( [[ -s "$csv" ]] && echo "agg=done" || echo "agg=todo" )
    pca_done=$( [[ -f "$NATIONAL_PCA_DIR/prithvi_${SAFE_VARIANT}_national_pca.pkl" ]] && echo "pca=ready" || echo "pca=missing" )
    printf "       %-10s  %s  %s  %s\n" "$VARIANT" "$tif_done" "$csv_done" "$pca_done"
  done
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo ""
  echo "DRY_RUN=1; not submitting."
  exit 0
fi

mkdir -p "$LOG_DIR" "$AGG_OUT_DIR"
TASK_ARRAY=$(IFS=,; echo "${INCOMPLETE_TASKS[*]}")

export REPO_DIR EMBED_OUT_DIR AGG_OUT_DIR NATIONAL_PCA_DIR TRACT_DIR YEAR STATE_LIST

JOB_ID=$(cd "$REPO_DIR" && sbatch \
  --export=ALL \
  --array="$TASK_ARRAY" \
  --output="$LOG_DIR/aggregate_%A_%a.out" \
  --error="$LOG_DIR/aggregate_%A_%a.err" \
  --parsable \
  "$SCRIPT" | cut -d';' -f1)

echo ""
echo "Submitted aggregation job $JOB_ID  (${#INCOMPLETE_TASKS[@]} tasks)"
echo "Logs: $LOG_DIR/aggregate_${JOB_ID}_*.out"
