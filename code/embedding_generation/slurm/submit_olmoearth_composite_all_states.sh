#!/bin/bash
# Submit OlmoEarth Sentinel-2 annual composite jobs for all CONUS states.
# Safe to re-run — states with existing composites are skipped by the array job.
#
# Uses a resubmit chain (same pattern as submit_prithvi_all_states.sh) so that
# large states requiring more than one walltime slot continue automatically.
# All individual jobs are capped at 8 hours; the chain runs until every state
# has a complete composite TIF.
#
# Usage:
#   bash submit_olmoearth_composite_all_states.sh
#   DRY_RUN=1 bash submit_olmoearth_composite_all_states.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
YEAR="${YEAR:-2022}"

: "${SCRATCH:?Set SCRATCH before submitting.}"
OUT_DIR="${OUT_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
LOG_DIR="$SCRIPT_DIR/logs"
SCRIPT="$SCRIPT_DIR/run_olmoearth_composite_array.sbatch"

# Land area (km²) used to pick a walltime tier.
# States with large land area may not finish in one slot; the resubmit chain
# handles continuation via composite.py's built-in tile-level resumption.
declare -A STATE_AREA_KM2=(
  [AL]=131426 [AZ]=294207 [AR]=134771 [CA]=403466 [CO]=268431
  [CT]=12542  [DC]=159    [DE]=5047   [FL]=138887 [GA]=148959
  [ID]=214045 [IL]=143793 [IN]=92789  [IA]=144701 [KS]=211754
  [KY]=102269 [LA]=111898 [ME]=79884  [MD]=25142  [MA]=20202
  [MI]=146435 [MN]=206232 [MS]=121531 [MO]=178040 [MT]=376962
  [NE]=198974 [NV]=284332 [NH]=23187  [NJ]=19047  [NM]=314161
  [NY]=122057 [NC]=125920 [ND]=178711 [OH]=105829 [OK]=177847
  [OR]=248608 [PA]=115883 [RI]=2678   [SC]=77857  [SD]=196350
  [TN]=106798 [TX]=676587 [UT]=212818 [VT]=23871  [VA]=102279
  [WA]=172119 [WV]=62259  [WI]=140268 [WY]=251470
)

# All 48 contiguous states + DC (sorted)
ALL_STATES=(
  AL AR AZ CA CO CT DC DE FL GA IA ID IL IN KS KY LA MA MD ME
  MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD
  TN TX UT VA VT WA WI WV WY
)

area_to_walltime() {
  local area="$1"
  # Calibrated from RI (2,678 km²) = 3h16m at 3 scenes/month, 10m resolution.
  # Time scales roughly linearly with area per tile; large states use the 8h cap
  # and the resubmit chain continues until all tiles are done.
  if   (( area <= 10000 ));  then echo "04:00:00"   # DC, RI, DE, CT, NJ (~1 tile)
  elif (( area <= 50000 ));  then echo "06:00:00"   # NH, VT, MD, MA
  elif (( area <= 150000 )); then echo "08:00:00"   # most mid-size states
  else                            echo "08:00:00"   # large states; resubmit chain handles rest
fi
}

# Discover which states still need a composite
mkdir -p "$OUT_DIR" "$LOG_DIR"

INCOMPLETE_STATES=()
for state in "${ALL_STATES[@]}"; do
  tif="$OUT_DIR/s2_annual_${state}_${YEAR}_olmoearth.tif"
  if [[ ! -s "$tif" ]]; then
    INCOMPLETE_STATES+=("$state")
  fi
done

if (( ${#INCOMPLETE_STATES[@]} == 0 )); then
  echo "All OlmoEarth composites present in $OUT_DIR — nothing to submit."
  exit 0
fi

# Sort into walltime tiers
declare -A TIER_TASKS   # walltime -> space-separated task indices
declare -A TIER_STATES  # walltime -> space-separated state names (display only)

# Build the full ordered state list (needed so SLURM_ARRAY_TASK_ID → state is stable)
STATE_LIST=$(IFS=:; echo "${INCOMPLETE_STATES[*]}")

for i in "${!INCOMPLETE_STATES[@]}"; do
  state="${INCOMPLETE_STATES[$i]}"
  area="${STATE_AREA_KM2[$state]:-100000}"
  wt=$(area_to_walltime "$area")
  TIER_TASKS["$wt"]+=" $i"
  TIER_STATES["$wt"]+=" $state"
done

echo "Repo:    $REPO_DIR"
echo "Output:  $OUT_DIR"
echo "Year:    $YEAR"
echo ""
echo "States needing composites: ${#INCOMPLETE_STATES[@]} / ${#ALL_STATES[@]}"
echo ""
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

export REPO_DIR OUT_DIR CACHE_ROOT YEAR STATE_LIST

JOB_IDS=()
for wt in "${!TIER_TASKS[@]}"; do
  read -ra _ids <<< "${TIER_TASKS[$wt]}"
  TASK_ARRAY=$(IFS=,; echo "${_ids[*]}")
  JOB_ID=$(cd "$REPO_DIR" && sbatch \
    --export=ALL \
    --time="$wt" \
    --array="$TASK_ARRAY" \
    --parsable \
    "$SCRIPT" | cut -d';' -f1)
  echo "Submitted job $JOB_ID  time=$wt  ${#_ids[@]} tasks"
  JOB_IDS+=("$JOB_ID")
done

# Schedule a resubmit that fires after all tier arrays finish, re-discovers
# remaining work, and exits cleanly once all composites are present.
DEPENDENCY="afterany:$(IFS=:; echo "${JOB_IDS[*]}")"
SELF="$(realpath "${BASH_SOURCE[0]}")"
RESUBMIT_ID=$(sbatch \
  --dependency="$DEPENDENCY" \
  --job-name=oe-composite-resubmit \
  --partition=normal \
  --time=00:10:00 \
  --mem=4G \
  --cpus-per-task=1 \
  --output="$LOG_DIR/oe_composite_resubmit_%j.out" \
  --error="$LOG_DIR/oe_composite_resubmit_%j.err" \
  --export=ALL \
  --parsable \
  --wrap="bash '$SELF'" | cut -d';' -f1)
echo "Resubmit job $RESUBMIT_ID scheduled after all tiers (cancel with: scancel $RESUBMIT_ID)"
