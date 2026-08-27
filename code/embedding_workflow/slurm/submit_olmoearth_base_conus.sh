#!/bin/bash
# Retile CONUS sources, then start resumable OlmoEarth Base inference.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
: "${SCRATCH:?Set SCRATCH before submitting.}"

YEAR="${YEAR:-2022}"
SOURCE_ROOT="${SOURCE_ROOT:-$SCRATCH/embeddings-health/olmoearth_composites}"
REPAIR_ROOT="${REPAIR_ROOT:-$SCRATCH/embeddings-health/olmoearth_composites_repair_lowcoverage_20260725}"
REPAIR_QA_ROOT="${REPAIR_QA_ROOT:-$SCRATCH/embeddings-health/qa/source_composites_repair_dense_final_20260801}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites_tiled512}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
CACHE="$CACHE_ROOT/olmoearth_base_overlap"
LOG_DIR="${LOG_DIR:-$REPO_DIR/code/embedding_workflow/slurm/logs}"
CONUS_STATES="AL AR AZ CA CO CT DC DE FL GA IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY"
MAX_RETILE_CONCURRENT="${MAX_RETILE_CONCURRENT:-8}"
RUN_ID="manual_$(date +%Y%m%dT%H%M%S)_$$"
RETILE_TASK_FILE="$CACHE/retile_${YEAR}_${RUN_ID}.tsv"

mkdir -p "$CACHE" "$COMPOSITE_DIR" "$LOG_DIR"
: > "$RETILE_TASK_FILE"
included=()
for state in $CONUS_STATES; do
  source="$SOURCE_ROOT/s2_annual_${state}_${YEAR}_olmoearth.tif"
  source_qa="canonical"
  if [[ ! -s "$source" ]]; then
    source="$REPAIR_ROOT/s2_annual_${state}_${YEAR}_olmoearth.tif"
    qa="$REPAIR_QA_ROOT/$state.json"
    [[ -s "$source" && -s "$qa" ]] || { echo "missing source or repair QA for $state" >&2; exit 1; }
    source_qa=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "$qa")
    if [[ "$source_qa" != pass && "$state" != WY ]]; then
      echo "repair QA is not pass for $state: $source_qa" >&2
      exit 1
    fi
    if [[ "$state" == WY && "$source_qa" != pass ]]; then
      echo "including WY by explicit low-density coverage exception: $source_qa"
    fi
  fi
  target="$COMPOSITE_DIR/s2_annual_${state}_${YEAR}_olmoearth.tif"
  printf '%s\t%s\t%s\t%s\n' "$state" "$source" "$target" "$source_qa" >> "$RETILE_TASK_FILE"
  included+=("$state")
done

count=$(wc -l < "$RETILE_TASK_FILE")
states="${included[*]}"
echo "Prepared $count source-retiling tasks for: $states"
echo "Retile task file: $RETILE_TASK_FILE"
if [[ "${DRY_RUN:-0}" == 1 || "${CONFIRM_OE_BASE_CONUS_SUBMIT:-0}" != 1 ]]; then
  echo "DRY RUN (set CONFIRM_OE_BASE_CONUS_SUBMIT=1 to submit)"
  exit 0
fi

export REPO_DIR SCRATCH YEAR CACHE_ROOT COMPOSITE_DIR RETILE_TASK_FILE LOG_DIR
retile=$(sbatch --parsable --array="0-$((count - 1))%$MAX_RETILE_CONCURRENT" --export=ALL \
  --output="$LOG_DIR/oe_base_retile_%A_%a.out" \
  --error="$LOG_DIR/oe_base_retile_%A_%a.err" \
  "$SCRIPT_DIR/retile_overlap_source.sbatch")
retile="${retile%%;*}"

export STATES="$states" EXCLUDE_STATES="" STATE_SET=conus CONFIRM_OE_BASE_SUBMIT=1
controller=$(sbatch --parsable --job-name=oe-base-start --partition=normal \
  --dependency="afterok:$retile" --time=01:00:00 --mem=4G --cpus-per-task=1 \
  --export=ALL \
  --output="$LOG_DIR/oe_base_start_%j.out" \
  --error="$LOG_DIR/oe_base_start_%j.err" \
  --wrap="bash '$SCRIPT_DIR/submit_olmoearth_base_overlap_resubmit.sh'")
controller="${controller%%;*}"
echo "Submitted Base source retiling=$retile and dependent inference controller=$controller"
