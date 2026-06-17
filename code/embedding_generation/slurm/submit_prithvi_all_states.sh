#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
YEAR="${YEAR:-2022}"
DATA_DIR="${DATA_DIR:-$SCRATCH/embeddings-health/data/prithvi_data}"

if [[ -z "${FINAL_OUT_DIR:-}" ]]; then
  : "${SCRATCH:?Set SCRATCH or FINAL_OUT_DIR before submitting.}"
  FINAL_OUT_DIR="$SCRATCH/embeddings-health/prithvi_embeddings"
fi

SCRIPT="$SCRIPT_DIR/run_prithvi_state_array.sbatch"
LOG_DIR="$SCRIPT_DIR/logs"
MODEL_VARIANTS=("tiny" "300M-TL")
NUM_MODELS="${#MODEL_VARIANTS[@]}"
SBATCH_MEM="${SBATCH_MEM:-128G}"

# Prithvi chip size in pixels, calibrated from the RI trial run (165 chips
# for a 2409x3295 TIF → ceil(2409/224)*ceil(3295/224) = 11*15 = 165).
CHIP_SIZE=224

# Walltime tiers based on 300M-TL timing from RI (165 chips, 2m37s) with a
# 1.5x safety margin. All thresholds are in chips (ceil(w/224)*ceil(h/224)).
#
#  <=  500 chips  →  30 min   (DC, RI, DE)
#  <= 2000 chips  →   1 hr    (CT, NJ, MD, VT, NH, MA, HI)
#  <= 6000 chips  →   3 hr    (most mid-size states)
#  <= 14000 chips →   6 hr    (large: CO, WY, OR, WA, NM, AZ, NV, MT)
#  >  14000 chips →  12 hr    (CA, TX; may still use checkpoints+resubmit)
chips_to_walltime() {
  local chips="$1"
  if   (( chips == 0 ));     then echo "01:00:00"  # gdalinfo unavailable; safe default
  elif (( chips <= 500 ));   then echo "00:30:00"
  elif (( chips <= 2000 ));  then echo "01:00:00"
  elif (( chips <= 6000 ));  then echo "03:00:00"
  elif (( chips <= 14000 )); then echo "06:00:00"
  else                            echo "12:00:00"
  fi
}

# Load GDAL to read TIF dimensions; unload before exporting env to jobs so
# GDAL's library paths don't leak into the compute node environment.
LOADED_GDAL=0
if command -v module >/dev/null 2>&1 && ! command -v gdalinfo >/dev/null 2>&1; then
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

# Discover states from spring TIF files in DATA_DIR
STATES=()
while IFS= read -r state; do
  STATES+=("$state")
done < <(
  find "$DATA_DIR" -maxdepth 1 -type f -name "s2_spring_*_${YEAR}_prithvi.tif" \
    -exec basename {} \; \
    | sed -E "s/^s2_spring_(.*)_${YEAR}_prithvi\.tif$/\1/" \
    | sort
)

if (( ${#STATES[@]} == 0 )); then
  echo "ERROR: no spring composites found in $DATA_DIR for year $YEAR" >&2
  exit 1
fi

# Assign each incomplete task to a walltime tier
declare -A TIER_TASKS   # walltime -> space-separated task IDs
declare -A TIER_STATES  # walltime -> space-separated state names (for display)

for i in "${!STATES[@]}"; do
  STATE="${STATES[$i]}"
  SPRING="$DATA_DIR/s2_spring_${STATE}_${YEAR}_prithvi.tif"
  CHIPS=$(get_chips "$SPRING")
  WALLTIME=$(chips_to_walltime "$CHIPS")

  HAS_INCOMPLETE=0
  for j in "${!MODEL_VARIANTS[@]}"; do
    VARIANT="${MODEL_VARIANTS[$j]}"
    SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"
    FINAL_TIF="$FINAL_OUT_DIR/$VARIANT/$STATE/prithvi_${SAFE_VARIANT}_${STATE}_${YEAR}.tif"
    if [[ ! -s "$FINAL_TIF" ]]; then
      TIER_TASKS["$WALLTIME"]+=" $(( i * NUM_MODELS + j ))"
      HAS_INCOMPLETE=1
    fi
  done
  (( HAS_INCOMPLETE )) && TIER_STATES["$WALLTIME"]+=" $STATE" || true
done

# Unload GDAL before exporting environment to jobs
if (( LOADED_GDAL )); then
  module unload gdal/3.10.2 2>/dev/null || true
fi

# Summary
NUM_STATES="${#STATES[@]}"
NUM_TOTAL=$(( NUM_STATES * NUM_MODELS ))
NUM_INCOMPLETE=0
for wt in "${!TIER_TASKS[@]}"; do
  read -ra _ids <<< "${TIER_TASKS[$wt]}"
  NUM_INCOMPLETE=$(( NUM_INCOMPLETE + ${#_ids[@]} ))
done
NUM_COMPLETE=$(( NUM_TOTAL - NUM_INCOMPLETE ))

echo "Repo:          $REPO_DIR"
echo "Data:          $DATA_DIR"
echo "Final outputs: $FINAL_OUT_DIR"
echo "Year:          $YEAR"
echo "Complete:      $NUM_COMPLETE / $NUM_TOTAL  (${NUM_STATES} states, ${NUM_MODELS} models)"
echo "Mem:           $SBATCH_MEM"

if (( NUM_INCOMPLETE == 0 )); then
  echo "All state/model pairs complete. Nothing to submit."
  exit 0
fi

echo ""
echo "Walltime tiers:"
for wt in $(echo "${!TIER_TASKS[@]}" | tr ' ' '\n' | sort); do
  read -ra _ids <<< "${TIER_TASKS[$wt]}"
  printf "  %-12s  %3d tasks   states:%s\n" "$wt" "${#_ids[@]}" "${TIER_STATES[$wt]:-}"
done

mkdir -p "$LOG_DIR" "$FINAL_OUT_DIR"
export REPO_DIR DATA_DIR FINAL_OUT_DIR YEAR

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo ""
  echo "DRY_RUN=1; not submitting."
  exit 0
fi

# Submit one array per tier
JOB_IDS=()
for wt in "${!TIER_TASKS[@]}"; do
  read -ra _ids <<< "${TIER_TASKS[$wt]}"
  TASK_ARRAY=$(IFS=,; echo "${_ids[*]}")
  JOB_ID=$(cd "$REPO_DIR" && sbatch \
    --export=ALL --time="$wt" --mem="$SBATCH_MEM" \
    --array="$TASK_ARRAY" --parsable "$SCRIPT" | cut -d';' -f1)
  echo "Submitted job $JOB_ID  time=$wt  ${#_ids[@]} tasks"
  JOB_IDS+=("$JOB_ID")
done

# Single resubmit fires after ALL tier arrays complete; re-tiers remaining
# work and exits cleanly once everything is done.
# To stop the chain early: scancel <resubmit_job_id>
DEPENDENCY="afterany:$(IFS=:; echo "${JOB_IDS[*]}")"
SELF="$(realpath "${BASH_SOURCE[0]}")"
RESUBMIT_ID=$(sbatch \
  --dependency="$DEPENDENCY" \
  --job-name=prithvi-resubmit \
  --partition=normal \
  --time=00:10:00 \
  --mem=4G \
  --cpus-per-task=1 \
  --output="$LOG_DIR/resubmit_%j.out" \
  --error="$LOG_DIR/resubmit_%j.err" \
  --export=ALL \
  --parsable \
  --wrap="bash '$SELF'" | cut -d';' -f1)
echo "Resubmit job $RESUBMIT_ID scheduled after all tiers (cancel with: scancel $RESUBMIT_ID)"
