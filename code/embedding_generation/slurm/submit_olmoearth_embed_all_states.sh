#!/bin/bash
# Submit OlmoEarth Base embedding inference for all states with a complete
# composite, split into per-state tiles sized to a target chip count so
# every tile finishes within the sbatch script's fixed walltime — no more
# picking a bigger walltime tier for huge states (Sherlock's ceiling is
# fixed regardless, and Base's ~0.75s/chip rate means large states need tens
# of hours of total inference no single job could hold anyway). Safe to
# re-run — states/tiles with existing output are skipped.
#
# Usage:
#   bash submit_olmoearth_embed_all_states.sh
#   DRY_RUN=1 bash submit_olmoearth_embed_all_states.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
YEAR="${YEAR:-2022}"
VARIANT="${VARIANT:-v1_1-Base}"

: "${SCRATCH:?Set SCRATCH before submitting.}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$SCRATCH/embeddings-health/olmoearth_embeddings}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
LOG_DIR="${LOG_DIR:-$SCRATCH/embeddings-health/logs}"
CKPT_ROOT="$SCRATCH/embeddings-health/checkpoints/olmoearth"
TILE_SCRIPT="$SCRIPT_DIR/run_olmoearth_embed_state_array.sbatch"
MERGE_SCRIPT="$SCRIPT_DIR/run_olmoearth_embed_merge.sbatch"

# Merge resource escalation: the merge sbatch defaults to a small --mem and
# the standard 8h --time so most states (observed <1GB actual usage, and fast
# once tile_merge.py reads whole blocks instead of band-by-band) don't force
# Sherlock's normal-partition MaxMemPerCPU=8000 ratio into allocating far more
# CPUs than the merge code needs. A handful of states may still OOM (a few
# physically enormous ones, e.g. GA, CA) or still run past 8h even after the
# block-read fix. Detect states whose merge task OOM'd or timed out in the
# array this script submitted last cycle and force them onto a high-mem +
# long-time tier from now on, persisted across resubmits in $CACHE_ROOT.
mkdir -p "$CACHE_ROOT"
HIGHMEM_STATES_FILE="$CACHE_ROOT/oe_merge_highmem_states.txt"
LAST_MERGE_JOB_FILE="$CACHE_ROOT/oe_merge_last_job.txt"
touch "$HIGHMEM_STATES_FILE"

# States whose merge from the *last* cycle is still RUNNING/PENDING right
# now. Rebuilt fresh every invocation (unlike HIGHMEM_STATES_FILE, this is a
# point-in-time fact, not a durable classification) and used below to skip
# re-submitting a merge for a state that already has one in flight -- this
# is what makes it safe to decouple the resubmit chain from waiting on
# merge completion (see ALL_DEPS below): tile-inference can resubmit as
# soon as tiles finish without accidentally double-submitting a merge for a
# state whose merge from the previous cycle is still going.
INFLIGHT_STATES_FILE="$CACHE_ROOT/oe_merge_inflight_states.txt"
> "$INFLIGHT_STATES_FILE"

if [[ -s "$LAST_MERGE_JOB_FILE" ]]; then
  IFS=: read -ra LAST_MERGE_JOB_IDS < "$LAST_MERGE_JOB_FILE"
  for JOB_ID in "${LAST_MERGE_JOB_IDS[@]}"; do
    [[ -z "$JOB_ID" ]] && continue
    while IFS='|' read -r FULL_JOBID STATE_OUT; do
      [[ -z "$FULL_JOBID" ]] && continue
      LOG_OUT="$LOG_DIR/oe_merge_${FULL_JOBID}.out"
      [[ -f "$LOG_OUT" ]] || continue
      THIS_STATE=$(grep -m1 "^State: " "$LOG_OUT" | awk '{print $2}')
      [[ -z "$THIS_STATE" ]] && continue
      if [[ "$STATE_OUT" == "RUNNING" || "$STATE_OUT" == "PENDING" ]]; then
        echo "$THIS_STATE" >> "$INFLIGHT_STATES_FILE"
      elif ! grep -qxF "$THIS_STATE" "$HIGHMEM_STATES_FILE"; then
        echo "$THIS_STATE" >> "$HIGHMEM_STATES_FILE"
        echo "Merge $STATE_OUT detected for $THIS_STATE (job $FULL_JOBID) -- escalating to high-mem/long-time tier for future merges"
      fi
    done < <(sacct -j "$JOB_ID" -X --noheader --parsable2 -o JobID,State,ExitCode 2>/dev/null | \
      awk -F'|' '$2 ~ /^RUNNING$/ || $2 ~ /^PENDING$/ || $2 ~ /^OUT_OF_ME/ || $2 == "TIMEOUT" || ($2 == "FAILED" && $3 ~ /:9$/) {print $1"|"$2}')
  done
fi

# OlmoEarth chip size in input pixels at 10 m resolution.
CHIP_SIZE=128

# Target chip count per tile — sized from the measured ~0.75s/chip rate on NY
# (~90 min of inference + setup/merge margin within the 4h walltime). Real
# data, but from one state's observation — validate via srun (Task 10 Step 7
# of the embed-tiling plan) and adjust if a tile runs close to the limit.
TARGET_CHIPS_PER_TILE="${TARGET_CHIPS_PER_TILE:-7000}"

TASK_FILE="$SCRATCH/embeddings-health/cache/oe_embed_tile_tasks_${YEAR}.txt"

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

SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"

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
  FINAL_TIF="$FINAL_OUT_DIR/$STATE/olmoearth_${SAFE_VARIANT}_${STATE}_${YEAR}.tif"
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

  if (( NUM_TILES > 1 )) && ! grep -qxF "$STATE" "$INFLIGHT_STATES_FILE"; then
    MERGE_STATES+=("$STATE")
    MERGE_NUM_TILES+=("$NUM_TILES")
  fi

  CKPT_DIR="$CKPT_ROOT/$STATE"
  OUTPUT_BASENAME="olmoearth_${SAFE_VARIANT}_${STATE}_${YEAR}"

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
echo "Variant:       $VARIANT"
echo "Year:          $YEAR"
echo "Target chips/tile: $TARGET_CHIPS_PER_TILE"
echo ""
echo "States skipped (embedding exists): $skipped_states / ${#STATES[@]}"
echo "States needing a merge job:        ${#MERGE_STATES[@]}"
echo "Tile tasks to submit:              $total_tiles"
echo ""

if (( total_tiles == 0 && ${#MERGE_STATES[@]} == 0 )); then
  echo "All OlmoEarth Base embeddings complete. Nothing to submit."
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
export REPO_DIR COMPOSITE_DIR FINAL_OUT_DIR CACHE_ROOT YEAR VARIANT

MAX_ARRAY=1000

# Sherlock's gpu-partition QOS caps this account at ~100 total submitted jobs
# at once (MaxSubmitJobsPerUser), shared across all concurrently-running
# pipelines (Nano/Clay/Base each submit their own tile arrays). A single
# state can need dozens of tiles, so total_tiles can vastly exceed that
# ceiling on its own. Cap how many tiles this invocation actually submits;
# the resubmit chain re-scans real $SCRATCH state on its next run and picks
# up whatever tiles are still missing, converging over several waves.
#
# Base is the only pipeline still doing real tile inference (Clay finished
# all tiles and is merge-only as of 2026-07-13; Nano has 12 states left).
# Base gets the lion's share of the shared ~100-job quota — Nano is capped
# at 6 (just enough to keep its last few states trickling forward) and
# Clay's cap below is moot (0 tile tasks left, so it never actually submits
# against the quota regardless of its configured value).
MAX_SUBMIT_PER_RUN="${MAX_SUBMIT_PER_RUN:-90}"
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
      --output="$LOG_DIR/oe_embed_%A_%a.out" \
      --error="$LOG_DIR/oe_embed_%A_%a.err" \
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

# Split states needing a merge into the default tier and the escalated
# tier (states that previously OOM'd or timed out — see detection block above).
NORMAL_MERGE_STATES=()
NORMAL_MERGE_NUM_TILES=()
HIGH_MERGE_STATES=()
HIGH_MERGE_NUM_TILES=()
for i in "${!MERGE_STATES[@]}"; do
  if grep -qxF "${MERGE_STATES[$i]}" "$HIGHMEM_STATES_FILE"; then
    HIGH_MERGE_STATES+=("${MERGE_STATES[$i]}")
    HIGH_MERGE_NUM_TILES+=("${MERGE_NUM_TILES[$i]}")
  else
    NORMAL_MERGE_STATES+=("${MERGE_STATES[$i]}")
    NORMAL_MERGE_NUM_TILES+=("${MERGE_NUM_TILES[$i]}")
  fi
done

MERGE_JOB_IDS=()

if (( ${#NORMAL_MERGE_STATES[@]} > 0 )); then
  MERGE_STATE_LIST=$(IFS=:; echo "${NORMAL_MERGE_STATES[*]}")
  MERGE_NUM_TILES_LIST=$(IFS=:; echo "${NORMAL_MERGE_NUM_TILES[*]}")
  MERGE_LAST_IDX=$(( ${#NORMAL_MERGE_STATES[@]} - 1 ))

  MERGE_DEP=""
  [[ -n "$TILE_JOB_ID" ]] && MERGE_DEP="--dependency=afterany:${TILE_JOB_ID}"

  export STATE_LIST="$MERGE_STATE_LIST"
  export NUM_TILES_LIST="$MERGE_NUM_TILES_LIST"
  JOB_ID=$(cd "$REPO_DIR" && sbatch \
    --export=ALL \
    $MERGE_DEP \
    --array="0-${MERGE_LAST_IDX}" \
    --output="$LOG_DIR/oe_merge_%A_%a.out" \
    --error="$LOG_DIR/oe_merge_%A_%a.err" \
    --parsable \
    "$MERGE_SCRIPT" | cut -d';' -f1)
  echo "Submitted merge array job  $JOB_ID  (${#NORMAL_MERGE_STATES[@]} states, default mem tier)"
  [[ -n "$TILE_JOB_ID" ]] && echo "  → depends on tile job $TILE_JOB_ID"
  MERGE_JOB_IDS+=("$JOB_ID")
fi

if (( ${#HIGH_MERGE_STATES[@]} > 0 )); then
  MERGE_STATE_LIST=$(IFS=:; echo "${HIGH_MERGE_STATES[*]}")
  MERGE_NUM_TILES_LIST=$(IFS=:; echo "${HIGH_MERGE_NUM_TILES[*]}")
  MERGE_LAST_IDX=$(( ${#HIGH_MERGE_STATES[@]} - 1 ))

  MERGE_DEP=""
  [[ -n "$TILE_JOB_ID" ]] && MERGE_DEP="--dependency=afterany:${TILE_JOB_ID}"

  export STATE_LIST="$MERGE_STATE_LIST"
  export NUM_TILES_LIST="$MERGE_NUM_TILES_LIST"
  JOB_ID=$(cd "$REPO_DIR" && sbatch \
    --export=ALL \
    $MERGE_DEP \
    --array="0-${MERGE_LAST_IDX}" \
    --mem=64G \
    --cpus-per-task=9 \
    --time=1-00:00:00 \
    --output="$LOG_DIR/oe_merge_%A_%a.out" \
    --error="$LOG_DIR/oe_merge_%A_%a.err" \
    --parsable \
    "$MERGE_SCRIPT" | cut -d';' -f1)
  echo "Submitted merge array job  $JOB_ID  (${#HIGH_MERGE_STATES[@]} states: ${HIGH_MERGE_STATES[*]} — high-mem/long-time tier)"
  [[ -n "$TILE_JOB_ID" ]] && echo "  → depends on tile job $TILE_JOB_ID"
  MERGE_JOB_IDS+=("$JOB_ID")
fi

# Persist this cycle's merge job IDs so the next resubmit can check their
# outcomes and grow the high-mem list before it rebuilds MERGE_STATES.
(IFS=:; echo "${MERGE_JOB_IDS[*]:-}") > "$LAST_MERGE_JOB_FILE"

# Deliberately depend on tiles only, NOT on merge completion: merges (esp.
# the high-mem/long-time tier) can run for up to 24h, and GPU tile-inference
# has nothing to do with whether a merge has finished. Waiting on merges here
# would leave the whole GPU pipeline idle for as long as the slowest merge in
# flight -- confirmed happening in production (Base sat with zero new tile
# jobs for hours while 24h-tier merges ran). INFLIGHT_STATES_FILE above is
# what makes this safe: a state whose merge is still running won't be
# resubmitted by the next cycle just because this resubmit didn't wait for it.
# Only fall back to depending on this cycle's merges if there were no new
# tile tasks to depend on instead (e.g. Clay, which is merge-only), so the
# resubmit doesn't tight-loop with nothing new to wait on.
if [[ -n "$TILE_JOB_ID" ]]; then
  ALL_DEPS="$TILE_JOB_ID"
else
  ALL_DEPS=$(IFS=:; echo "${MERGE_JOB_IDS[*]:-}")
fi
SELF="$(realpath "${BASH_SOURCE[0]}")"
RESUBMIT_ID=$(sbatch \
  --dependency="afterany:${ALL_DEPS}" \
  --job-name=oe-embed-resubmit \
  --partition=normal \
  --time=00:10:00 \
  --mem=4G \
  --cpus-per-task=1 \
  --output="$LOG_DIR/oe_embed_resubmit_%j.out" \
  --error="$LOG_DIR/oe_embed_resubmit_%j.err" \
  --export=ALL \
  --parsable \
  --wrap="bash '$SELF'" | cut -d';' -f1)
echo "Resubmit job $RESUBMIT_ID scheduled after tiles (not merges — see ALL_DEPS above) (cancel with: scancel $RESUBMIT_ID)"
