#!/bin/bash
# Submit Clay v1.5 embedding inference for all states with a complete composite.
# Safe to re-run — states with existing embedding TIFs are skipped.
#
# Clay uses 256×256 chips (vs 128×128 for OlmoEarth) so chip count is ~4×
# lower, but the model is ViT-Large (24 layers) so per-chip time is longer.
# Walltime tiers are calibrated accordingly.
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
SCRIPT="$SCRIPT_DIR/run_clay_embed_state_array.sbatch"
SBATCH_MEM="${SBATCH_MEM:-128G}"

# Clay chip size: 256×256 px.  Chip count is roughly (W/256)*(H/256).
CHIP_SIZE=256

# Walltime tiers calibrated for Clay ViT-Large at 256-px chip size.
# ~4× fewer chips than OlmoEarth Base, but per-chip latency is higher.
chips_to_walltime() {
  local chips="$1"
  if   (( chips == 0 ));       then echo "04:00:00"  # gdalinfo unavailable — be safe
  elif (( chips <= 1250 ));    then echo "01:00:00"  # tiny states (DC, RI, DE)
  elif (( chips <= 12500 ));   then echo "02:00:00"  # small-medium states
  elif (( chips <= 50000 ));   then echo "04:00:00"  # large states
  else                              echo "06:00:00"  # very large (TX, CA, MT)
  fi
}

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

# Discover states with a composite but without a complete Clay embedding.
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

declare -A TIER_TASKS
declare -A TIER_STATES

for i in "${!STATES[@]}"; do
  STATE="${STATES[$i]}"
  FINAL_TIF="$FINAL_OUT_DIR/$STATE/clay_v1.5_${STATE}_${YEAR}.tif"
  if [[ -s "$FINAL_TIF" ]]; then
    continue  # already done
  fi
  COMPOSITE="$COMPOSITE_DIR/s2_annual_${STATE}_${YEAR}_olmoearth.tif"
  CHIPS=$(get_chips "$COMPOSITE")
  WALLTIME=$(chips_to_walltime "$CHIPS")
  TIER_TASKS["$WALLTIME"]+=" $i"
  TIER_STATES["$WALLTIME"]+=" $STATE"
done

if (( LOADED_GDAL )); then
  module unload gdal/3.10.2 2>/dev/null || true
fi

NUM_STATES="${#STATES[@]}"
NUM_INCOMPLETE=0
for wt in "${!TIER_TASKS[@]}"; do
  read -ra _ids <<< "${TIER_TASKS[$wt]}"
  NUM_INCOMPLETE=$(( NUM_INCOMPLETE + ${#_ids[@]} ))
done
NUM_COMPLETE=$(( NUM_STATES - NUM_INCOMPLETE ))

STATE_LIST=$(IFS=:; echo "${STATES[*]}")

echo "Repo:        $REPO_DIR"
echo "Composites:  $COMPOSITE_DIR"
echo "Outputs:     $FINAL_OUT_DIR"
echo "Year:        $YEAR"
echo "Mem:         $SBATCH_MEM"
echo "Complete:    $NUM_COMPLETE / $NUM_STATES"
echo ""

if (( NUM_INCOMPLETE == 0 )); then
  echo "All Clay v1.5 embeddings complete. Nothing to submit."
  exit 0
fi

echo "Walltime tiers:"
for wt in $(echo "${!TIER_TASKS[@]}" | tr ' ' '\n' | sort); do
  read -ra _ids <<< "${TIER_TASKS[$wt]}"
  printf "  %-12s  %3d tasks   states:%s\n" "$wt" "${#_ids[@]}" "${TIER_STATES[$wt]:-}"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo ""
  echo "DRY_RUN=1; not submitting."
  exit 0
fi

mkdir -p "$LOG_DIR" "$FINAL_OUT_DIR"
export REPO_DIR COMPOSITE_DIR FINAL_OUT_DIR CACHE_ROOT YEAR STATE_LIST

JOB_IDS=()
for wt in "${!TIER_TASKS[@]}"; do
  read -ra _ids <<< "${TIER_TASKS[$wt]}"
  TASK_ARRAY=$(IFS=,; echo "${_ids[*]}")
  JOB_ID=$(cd "$REPO_DIR" && sbatch \
    --export=ALL \
    --time="$wt" \
    --mem="$SBATCH_MEM" \
    --array="$TASK_ARRAY" \
    --output="$LOG_DIR/clay_embed_%A_%a.out" \
    --error="$LOG_DIR/clay_embed_%A_%a.err" \
    --parsable \
    "$SCRIPT" | cut -d';' -f1)
  echo "Submitted job $JOB_ID  time=$wt  ${#_ids[@]} tasks"
  JOB_IDS+=("$JOB_ID")
done

DEPENDENCY="afterany:$(IFS=:; echo "${JOB_IDS[*]}")"
SELF="$(realpath "${BASH_SOURCE[0]}")"
RESUBMIT_ID=$(sbatch \
  --dependency="$DEPENDENCY" \
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
echo "Resubmit job $RESUBMIT_ID scheduled after all tiers (cancel with: scancel $RESUBMIT_ID)"
