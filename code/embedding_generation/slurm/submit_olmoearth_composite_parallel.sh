#!/bin/bash
# Submit per-tile composite jobs for all incomplete states in parallel.
#
# Each Slurm array task processes exactly one 100km×100km tile for one state,
# so all remaining tiles across all states run concurrently (subject to cluster
# capacity). After all tile jobs finish, one merge job per state assembles the
# tile TIFs into the final composite.
#
# Existing tile TIFs are skipped (composite.py --tile-index is idempotent).
# States whose final composite TIF already exists are skipped entirely.
#
# Usage:
#   bash submit_olmoearth_composite_parallel.sh
#   DRY_RUN=1 bash submit_olmoearth_composite_parallel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
YEAR="${YEAR:-2022}"

: "${SCRATCH:?Set SCRATCH before submitting.}"
OUT_DIR="${OUT_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
LOG_DIR="${LOG_DIR:-$SCRATCH/embeddings-health/logs}"
TILE_SCRIPT="$SCRIPT_DIR/run_olmoearth_composite_tile.sbatch"
MERGE_SCRIPT="$SCRIPT_DIR/run_olmoearth_composite_merge.sbatch"

# Task list file written to SCRATCH so compute nodes can read it.
TASK_FILE="$SCRATCH/embeddings-health/cache/oe_tile_tasks_${YEAR}.txt"

ALL_STATES=(
  AL AR AZ CA CO CT DC DE FL GA IA ID IL IN KS KY LA MA MD ME
  MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD
  TN TX UT VA VT WA WI WV WY
)

mkdir -p "$OUT_DIR" "$LOG_DIR" "$(dirname "$TASK_FILE")"

# This script also runs at the tail of the resubmit chain via
# `sbatch --wrap="bash $SELF"` (see below), which starts in a clean batch
# environment — unlike an interactive login shell, `devel`/`uv` are not
# pre-loaded there (see slurm/README.md, "Login shell vs. batch job
# environment"). get_tile_count() below needs `uv` either way, so set up the
# same bootstrapping the tile/merge jobs use.
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-$(hostname -s)"
if command -v module >/dev/null 2>&1; then
  module load devel
  module load gcc/14.2.0
fi
export CC="$(command -v gcc || true)" CXX="$(command -v g++ || true)"
if ! command -v uv >/dev/null 2>&1; then
  UV_INSTALL_DIR="${UV_INSTALL_DIR:-$CACHE_ROOT/uv-bin}"
  [[ -x "$UV_INSTALL_DIR/uv" ]] && export PATH="$UV_INSTALL_DIR:$PATH" || { echo "ERROR: uv not found." >&2; exit 1; }
fi

# ------------------------------------------------------------------
# Ask composite.py for the true tile count directly (--print-tile-count runs
# split_bbox_into_tiles() with no network access and exits immediately). This
# is the same tiling logic the tile and merge jobs use, so the submit script
# can never drift out of sync with it — no area-based estimate, no scraping
# prior run logs for a banner line whose format can change independently.
# ------------------------------------------------------------------
get_tile_count() {
  local state="$1"
  (cd "$REPO_DIR/code/embedding_generation" && uv run --python 3.11 python composite.py \
    --state "$state" --max-tile-km 100 --print-tile-count)
}

# ------------------------------------------------------------------
# Build the task list: one line per remaining tile
# ------------------------------------------------------------------
MERGE_STATES=()   # states that need a merge job
> "$TASK_FILE"    # truncate / create

total_tiles=0
skipped_states=0

for state in "${ALL_STATES[@]}"; do
  final_tif="$OUT_DIR/s2_annual_${state}_${YEAR}_olmoearth.tif"
  if [[ -s "$final_tif" ]]; then
    (( skipped_states++ )) || true
    continue
  fi

  n_tiles=$(get_tile_count "$state")
  MERGE_STATES+=("$state")

  for (( idx=0; idx<n_tiles; idx++ )); do
    # Skip tiles that already exist
    tile_path="$OUT_DIR/s2_annual_${state}_${YEAR}_olmoearth_tile$(printf '%03d' "$idx").tif"
    # For single-tile states the final TIF path IS the output (no _tile suffix) —
    # checked above, so we always add single-tile states here.
    if [[ $n_tiles -gt 1 && -s "$tile_path" ]]; then
      continue
    fi
    echo "$state $idx" >> "$TASK_FILE"
    (( total_tiles++ )) || true
  done
done

echo "Repo:          $REPO_DIR"
echo "Output:        $OUT_DIR"
echo "Year:          $YEAR"
echo "Task file:     $TASK_FILE"
echo ""
echo "States skipped (composite exists): $skipped_states / ${#ALL_STATES[@]}"
echo "States needing merge:              ${#MERGE_STATES[@]}"
echo "Tile jobs to submit:               $total_tiles"
echo ""

if (( total_tiles == 0 && ${#MERGE_STATES[@]} == 0 )); then
  echo "All OlmoEarth composites present in $OUT_DIR — nothing to submit."
  if [[ "${EMBED_ON_COMPLETE:-0}" == "1" ]]; then
    echo ""
    echo "All composites complete — submitting embedding trigger..."
    mkdir -p "$LOG_DIR"
    sbatch \
      --export=ALL \
      --output="$LOG_DIR/start_embeddings_%j.out" \
      --error="$LOG_DIR/start_embeddings_%j.err" \
      "$SCRIPT_DIR/start_embeddings.sbatch"
  fi
  exit 0
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; not submitting."
  echo ""
  echo "First 10 tasks in $TASK_FILE:"
  head -10 "$TASK_FILE"
  exit 0
fi

export REPO_DIR OUT_DIR CACHE_ROOT YEAR

# ------------------------------------------------------------------
# Submit tile arrays in batches of 1000 (Sherlock max_array_tasks=1000)
# Each batch gets its own task file containing only its tasks.
# ------------------------------------------------------------------
MAX_ARRAY=1000
TILE_JOB_IDS=()
if (( total_tiles > 0 )); then
  batch=0
  offset=0
  while (( offset < total_tiles )); do
    end=$(( offset + MAX_ARRAY - 1 ))
    (( end >= total_tiles )) && end=$(( total_tiles - 1 ))
    count=$(( end - offset + 1 ))

    # Write this batch's task file (lines offset+1 to end+1 of the main file)
    batch_file="${TASK_FILE%.txt}_batch${batch}.txt"
    sed -n "$((offset + 1)),$((end + 1))p" "$TASK_FILE" > "$batch_file"

    export TILE_TASK_FILE="$batch_file"
    JOB_ID=$(cd "$REPO_DIR" && sbatch \
      --export=ALL \
      --array="0-$(( count - 1 ))%200" \
      --output="$LOG_DIR/oe_tile_%A_%a.out" \
      --error="$LOG_DIR/oe_tile_%A_%a.err" \
      --parsable \
      "$TILE_SCRIPT" | cut -d';' -f1)
    echo "Submitted tile batch $batch: job $JOB_ID  ($count tasks, ≤200 concurrent)"
    TILE_JOB_IDS+=("$JOB_ID")
    (( batch++ )) || true
    (( offset += MAX_ARRAY )) || true
  done
fi
TILE_JOB_ID=$(IFS=:; echo "${TILE_JOB_IDS[*]}")

# ------------------------------------------------------------------
# Submit merge array — one task per state, after all tile jobs finish
# ------------------------------------------------------------------
if (( ${#MERGE_STATES[@]} > 0 )); then
  MERGE_STATE_LIST=$(IFS=:; echo "${MERGE_STATES[*]}")
  MERGE_LAST_IDX=$(( ${#MERGE_STATES[@]} - 1 ))

  MERGE_DEP=""
  [[ -n "$TILE_JOB_ID" ]] && MERGE_DEP="--dependency=afterany:${TILE_JOB_ID}"

  export STATE_LIST="$MERGE_STATE_LIST"
  MERGE_JOB_ID=$(cd "$REPO_DIR" && sbatch \
    --export=ALL \
    $MERGE_DEP \
    --array="0-${MERGE_LAST_IDX}" \
    --output="$LOG_DIR/oe_merge_%A_%a.out" \
    --error="$LOG_DIR/oe_merge_%A_%a.err" \
    --parsable \
    "$MERGE_SCRIPT" | cut -d';' -f1)
  echo "Submitted merge array job  $MERGE_JOB_ID  (${#MERGE_STATES[@]} states)"
  [[ -n "$TILE_JOB_ID" ]] && echo "  → depends on tile job $TILE_JOB_ID"

  # Resubmit chain: after merges complete, re-run this script to pick up any
  # tiles that failed and need a retry, and trigger embed submissions.
  SELF="$(realpath "${BASH_SOURCE[0]}")"
  RESUB_DEP="afterany:${MERGE_JOB_ID}"
  RESUB_ID=$(sbatch \
    --dependency="$RESUB_DEP" \
    --job-name=oe-tile-resubmit \
    --partition=normal \
    --time=00:10:00 \
    --mem=4G \
    --cpus-per-task=1 \
    --output="$LOG_DIR/oe_tile_resubmit_%j.out" \
    --error="$LOG_DIR/oe_tile_resubmit_%j.err" \
    --export=ALL \
    --parsable \
    --wrap="bash '$SELF'" | cut -d';' -f1)
  echo "Resubmit job $RESUB_ID after merges (cancel: scancel $RESUB_ID)"
fi
