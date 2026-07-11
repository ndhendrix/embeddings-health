#!/bin/bash
# Submit Clay v1.5 embedding inference for all states with a complete
# composite, split into per-state tiles sized to a target chip count so
# every tile finishes within the sbatch script's fixed walltime. Safe to
# re-run — states/tiles with existing output are skipped.
#
# Usage:
#   bash submit_clay_embed_all_states.sh
#   DRY_RUN=1 bash submit_clay_embed_all_states.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
YEAR="${YEAR:-2022}"

: "${SCRATCH:?Set SCRATCH before submitting.}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$SCRATCH/embeddings-health/clay_embeddings}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
LOG_DIR="${LOG_DIR:-$SCRATCH/embeddings-health/logs}"
CKPT_ROOT="$SCRATCH/embeddings-health/checkpoints/clay"
TILE_SCRIPT="$SCRIPT_DIR/run_clay_embed_state_array.sbatch"
MERGE_SCRIPT="$SCRIPT_DIR/run_clay_embed_merge.sbatch"

# Clay chip size: 256×256 px.
CHIP_SIZE=256

# Target chip count per tile — placeholder pending real measurement (Clay's
# throughput hasn't been benchmarked the way Base/Nano have; validate via
# srun per Task 9 Step 7 of the embed-tiling plan and adjust if a tile runs
# close to the 4-hour walltime).
TARGET_CHIPS_PER_TILE="${TARGET_CHIPS_PER_TILE:-20000}"

TASK_FILE="$SCRATCH/embeddings-health/cache/clay_tile_tasks_${YEAR}.txt"

LOADED_GDAL=0
if command -v module >/dev/null 2>&1 && ! command -v gdalinfo >/dev/null 2>&1; then
  module load devel 2>/dev/null || true
  module load physics gdal/3.10.2 2>/dev/null && LOADED_GDAL=1 || true
fi

get_chips() {
  local tif="$1"
  if ! command -v gdalinfo >/dev/null 2>&1; then
    echo "0"; return
  fi
  local size_line width height w_chips h_chips
  size_line=$(gdalinfo "$tif" 2>/dev/null | grep "^Size is" || true)
  if [[ -z "$size_line" ]]; then echo "0"; return; fi
  width=$(echo  "$size_line" | sed 's/Size is //' | cut -d',' -f1 | tr -d ' ')
  height=$(echo "$size_line" | sed 's/Size is //' | cut -d',' -f2 | tr -d ' ')
  w_chips=$(( (width  + CHIP_SIZE - 1) / CHIP_SIZE ))
  h_chips=$(( (height + CHIP_SIZE - 1) / CHIP_SIZE ))
  echo $(( w_chips * h_chips ))
}

get_h_chips() {
  local tif="$1"
  if ! command -v gdalinfo >/dev/null 2>&1; then
    echo "1"; return
  fi
  local size_line height
  size_line=$(gdalinfo "$tif" 2>/dev/null | grep "^Size is" || true)
  if [[ -z "$size_line" ]]; then echo "1"; return; fi
  height=$(echo "$size_line" | sed 's/Size is //' | cut -d',' -f2 | tr -d ' ')
  echo $(( (height + CHIP_SIZE - 1) / CHIP_SIZE ))
}

STATES=()
while IFS= read -r state; do
  STATES+=("$state")
done < <(
  find "$COMPOSITE_DIR" -maxdepth 1 -type f \
    -name "s2_annual_*_${YEAR}_olmoearth.tif" \
    -exec basename {} \; \
    | sed -E "s/^s2_annual_(.*)_${YEAR}_olmoearth\.tif$/\1/" \
    | sort
)

if (( ${#STATES[@]} == 0 )); then
  echo "ERROR: no OlmoEarth composites found in $COMPOSITE_DIR for year $YEAR" >&2
  exit 1
fi

MERGE_STATES=()
MERGE_NUM_TILES=()
> "$TASK_FILE"

total_tiles=0
skipped_states=0

for STATE in "${STATES[@]}"; do
  FINAL_TIF="$FINAL_OUT_DIR/$STATE/clay_v1.5_${STATE}_${YEAR}.tif"
  if [[ -s "$FINAL_TIF" ]]; then
    (( skipped_states++ )) || true
    continue
  fi

  COMPOSITE="$COMPOSITE_DIR/s2_annual_${STATE}_${YEAR}_olmoearth.tif"
  CHIPS=$(get_chips "$COMPOSITE")
  H_CHIPS=$(get_h_chips "$COMPOSITE")

  if (( CHIPS == 0 )); then
    NUM_TILES=1
  else
    NUM_TILES=$(( (CHIPS + TARGET_CHIPS_PER_TILE - 1) / TARGET_CHIPS_PER_TILE ))
    (( NUM_TILES < 1 )) && NUM_TILES=1
    (( NUM_TILES > H_CHIPS )) && NUM_TILES=$H_CHIPS
  fi

  if (( NUM_TILES > 1 )); then
    MERGE_STATES+=("$STATE")
    MERGE_NUM_TILES+=("$NUM_TILES")
  fi

  CKPT_DIR="$CKPT_ROOT/$STATE"
  OUTPUT_BASENAME="clay_v1.5_${STATE}_${YEAR}"

  for (( idx=0; idx<NUM_TILES; idx++ )); do
    if (( NUM_TILES > 1 )); then
      tile_path="$CKPT_DIR/${OUTPUT_BASENAME}_tile$(printf '%03d' "$idx").tif"
      [[ -s "$tile_path" ]] && continue
    fi
    echo "$STATE $idx $NUM_TILES" >> "$TASK_FILE"
    (( total_tiles++ )) || true
  done
done

if (( LOADED_GDAL )); then
  module unload gdal/3.10.2 2>/dev/null || true
fi

echo "Repo:          $REPO_DIR"
echo "Composites:    $COMPOSITE_DIR"
echo "Outputs:       $FINAL_OUT_DIR"
echo "Year:          $YEAR"
echo "Target chips/tile: $TARGET_CHIPS_PER_TILE"
echo ""
echo "States skipped (embedding exists): $skipped_states / ${#STATES[@]}"
echo "States needing a merge job:        ${#MERGE_STATES[@]}"
echo "Tile tasks to submit:              $total_tiles"
echo ""

if (( total_tiles == 0 && ${#MERGE_STATES[@]} == 0 )); then
  echo "All Clay v1.5 embeddings complete. Nothing to submit."
  exit 0
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; not submitting."
  echo ""
  echo "First 10 tasks in $TASK_FILE:"
  head -10 "$TASK_FILE"
  exit 0
fi

mkdir -p "$LOG_DIR" "$FINAL_OUT_DIR"
export REPO_DIR COMPOSITE_DIR FINAL_OUT_DIR CACHE_ROOT YEAR

MAX_ARRAY=1000

# Sherlock's gpu-partition QOS caps this account at ~100 total submitted jobs
# at once (MaxSubmitJobsPerUser), shared across all concurrently-running
# pipelines (Nano/Clay/Base each submit their own tile arrays). A single
# state can need dozens of tiles, so total_tiles can vastly exceed that
# ceiling on its own. Cap how many tiles this invocation actually submits;
# the resubmit chain re-scans real $SCRATCH state on its next run and picks
# up whatever tiles are still missing, converging over several waves.
#
# Clay is well behind (27% complete vs. Base's 16% and Nano's 71% as of
# 2026-07-11), so it gets a larger share of the shared ~100-job quota than
# Nano, sized proportional to each pipeline's remaining state count
# (41/36/14 of 49 total). See the matching comment in
# submit_olmoearth_nano_embed_all_states.sh — the three pipelines' caps
# (Base 45 / Clay 40 / Nano 15) are sized together and sum to the full
# ~100-job quota (composite work is nearly done, so it no longer needs a
# meaningful share).
MAX_SUBMIT_PER_RUN="${MAX_SUBMIT_PER_RUN:-40}"
SUBMIT_COUNT=$(( total_tiles < MAX_SUBMIT_PER_RUN ? total_tiles : MAX_SUBMIT_PER_RUN ))
if (( SUBMIT_COUNT < total_tiles )); then
  echo "NOTE: only submitting $SUBMIT_COUNT/$total_tiles tile tasks this run" \
       "(MAX_SUBMIT_PER_RUN=$MAX_SUBMIT_PER_RUN); the resubmit chain will pick up the rest."
fi

TILE_JOB_IDS=()
if (( SUBMIT_COUNT > 0 )); then
  batch=0
  offset=0
  while (( offset < SUBMIT_COUNT )); do
    end=$(( offset + MAX_ARRAY - 1 ))
    (( end >= SUBMIT_COUNT )) && end=$(( SUBMIT_COUNT - 1 ))
    count=$(( end - offset + 1 ))

    batch_file="${TASK_FILE%.txt}_batch${batch}.txt"
    sed -n "$((offset + 1)),$((end + 1))p" "$TASK_FILE" > "$batch_file"

    export TILE_TASK_FILE="$batch_file"
    JOB_ID=$(cd "$REPO_DIR" && sbatch \
      --export=ALL \
      --array="0-$(( count - 1 ))%200" \
      --output="$LOG_DIR/clay_embed_%A_%a.out" \
      --error="$LOG_DIR/clay_embed_%A_%a.err" \
      --parsable \
      "$TILE_SCRIPT" | cut -d';' -f1)
    echo "Submitted tile batch $batch: job $JOB_ID  ($count tasks, ≤200 concurrent)"
    TILE_JOB_IDS+=("$JOB_ID")
    (( batch++ )) || true
    (( offset += MAX_ARRAY )) || true
  done
fi
# ${arr[*]:-} (not ${arr[*]}) — Sherlock's default bash predates 4.4 and
# treats a zero-element array as unset under `set -u`, so an unguarded
# expansion here crashes with "unbound variable" whenever a cycle submits
# no new tile tasks (e.g. all tiles done, only a merge still pending).
TILE_JOB_ID=$(IFS=:; echo "${TILE_JOB_IDS[*]:-}")

if (( ${#MERGE_STATES[@]} > 0 )); then
  MERGE_STATE_LIST=$(IFS=:; echo "${MERGE_STATES[*]}")
  MERGE_NUM_TILES_LIST=$(IFS=:; echo "${MERGE_NUM_TILES[*]}")
  MERGE_LAST_IDX=$(( ${#MERGE_STATES[@]} - 1 ))

  MERGE_DEP=""
  [[ -n "$TILE_JOB_ID" ]] && MERGE_DEP="--dependency=afterany:${TILE_JOB_ID}"

  export STATE_LIST="$MERGE_STATE_LIST"
  export NUM_TILES_LIST="$MERGE_NUM_TILES_LIST"
  MERGE_JOB_ID=$(cd "$REPO_DIR" && sbatch \
    --export=ALL \
    $MERGE_DEP \
    --array="0-${MERGE_LAST_IDX}" \
    --output="$LOG_DIR/clay_merge_%A_%a.out" \
    --error="$LOG_DIR/clay_merge_%A_%a.err" \
    --parsable \
    "$MERGE_SCRIPT" | cut -d';' -f1)
  echo "Submitted merge array job  $MERGE_JOB_ID  (${#MERGE_STATES[@]} states)"
  [[ -n "$TILE_JOB_ID" ]] && echo "  → depends on tile job $TILE_JOB_ID"
fi

# Built as only-the-non-empty-parts, joined by ':' — TILE_JOB_ID is now
# legitimately empty when a cycle submits no new tile tasks (see the
# TILE_JOB_IDS[*]:- fix above), and naively prepending it would leave a
# leading ':' that sbatch --dependency rejects as malformed.
ALL_DEPS="${TILE_JOB_ID}"
if [[ -n "${MERGE_JOB_ID:-}" ]]; then
  if [[ -n "$ALL_DEPS" ]]; then
    ALL_DEPS="${ALL_DEPS}:${MERGE_JOB_ID}"
  else
    ALL_DEPS="${MERGE_JOB_ID}"
  fi
fi
SELF="$(realpath "${BASH_SOURCE[0]}")"
RESUBMIT_ID=$(sbatch \
  --dependency="afterany:${ALL_DEPS}" \
  --job-name=clay-embed-resubmit \
  --partition=normal \
  --time=00:10:00 \
  --mem=4G \
  --cpus-per-task=1 \
  --output="$LOG_DIR/clay_embed_resubmit_%j.out" \
  --error="$LOG_DIR/clay_embed_resubmit_%j.err" \
  --export=ALL \
  --parsable \
  --wrap="bash '$SELF'" | cut -d';' -f1)
echo "Resubmit job $RESUBMIT_ID scheduled after tiles+merge (cancel with: scancel $RESUBMIT_ID)"
