#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRATCH="${SCRATCH:-/scratch/users/nhendrix}"
YEAR="${YEAR:-2022}"
MODEL="${MODEL:-olmoearth-v1.2-base}"
TARGET_BLOCKS_PER_TILE="${TARGET_BLOCKS_PER_TILE:-4000}"
STATE_SET="${STATE_SET:-conus}"
STATES="${STATES:-AL AR AZ CA CO CT DC DE FL GA IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY}"

COMPOSITES="${COMPOSITES:-$SCRATCH/embeddings-health/olmoearth_composites_tiled512}"
OUTPUT_ROOT="${OVERLAP_OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow_overlap_v1}"
PY="${PY:-$SCRATCH/embeddings-health/cache/venv-3.11-cpu/bin/python}"

STAMP="$(date +%Y%m%d_%H%M%S)"
TASK_FILE="${TASK_FILE:-/tmp/oe_base_tasks_${YEAR}_${STAMP}.txt}"
ALL_TASK_FILE="${ALL_TASK_FILE:-/tmp/oe_base_all_tasks_${YEAR}_${STAMP}.txt}"
STATE_FILE="${STATE_FILE:-/tmp/oe_base_states_${YEAR}_${STAMP}.txt}"
MANIFEST="${MANIFEST:-/tmp/oe_base_manifest_${YEAR}_${STAMP}.json}"

read -r -a state_args <<< "$STATES"
missing_sources=()
available_sources=0
for state in "${state_args[@]}"; do
  source="$COMPOSITES/s2_annual_${state}_${YEAR}_olmoearth.tif"
  if [[ -s "$source" ]]; then
    ((available_sources += 1))
  else
    missing_sources+=("$state")
  fi
done

echo "Current all-state OE-Base status:"
echo "- Model: $MODEL"
echo "- Tiled composites: $COMPOSITES"
echo "- Source preparation: $available_sources/${#state_args[@]} complete"
if (( ${#missing_sources[@]} > 0 )); then
  echo "- Sources still being tiled: ${missing_sources[*]}"
  echo "- Embedding inference is waiting for source preparation."
else
  "$PY" "$REPO_DIR/code/embedding_workflow/plan_overlap_tasks.py" \
    --model "$MODEL" \
    --composites "$COMPOSITES" \
    --output-root "$OUTPUT_ROOT" \
    --task-file "$TASK_FILE" \
    --all-task-file "$ALL_TASK_FILE" \
    --state-file "$STATE_FILE" \
    --manifest "$MANIFEST" \
    --year "$YEAR" \
    --target-blocks "$TARGET_BLOCKS_PER_TILE" \
    --max-tasks 100000 \
    --tiling rectangular \
    --states "${state_args[@]}" >/dev/null

  summary="$(jq -r '
    ([.states[].num_tiles] | add) as $total |
    ([.states[].missing_tiles | length] | add) as $missing |
    ($total - $missing) as $complete |
    "Total expected tiles: \($total)\nComplete / validated: \($complete)\nNot done yet: \($missing)\nOverall complete: \((($complete / $total * 100) * 100 | round) / 100)%"
  ' "$MANIFEST")"
  echo "$summary" | sed 's/^/- /'

  echo
  echo "Incomplete states:"
  jq -r '
    .states[]
    | select((.missing_tiles | length) > 0)
    | "\(.state)\t\(.missing_tiles | length)\t\(.num_tiles - (.missing_tiles | length))/\(.num_tiles)"
  ' "$MANIFEST" | awk 'BEGIN {printf "%-3s %10s   %s\n", "ST", "remaining", "done"} {printf "%-3s %10d   %s\n", $1, $2, $3}'
fi

echo
echo "Live Slurm status:"
squeue --me || true

report_wave() {
  local job_name="$1" label="$2" wave
  wave="$(squeue --me --noheader --name="$job_name" --format='%i' 2>/dev/null \
    | sed -E 's/(\[.*|_.*)//g' | sort -n | tail -1 || true)"
  [[ -n "$wave" ]] || return 0
  echo
  echo "$label: $wave"
  sacct -j "$wave" --format=JobID,State -P 2>/dev/null \
    | awk -F'|' -v wave="$wave" '$1 ~ "^" wave "_[0-9]+$" {c[$2]++} END {for (s in c) print s, c[s]}' \
    | sort || true
}

report_wave oe-base-retile "Source-retiling array"
report_wave oe-base-overlap "Current inference wave"

watcher="$(squeue --me --noheader --name=oe-base-overlap-resubmit --format='%A %T %M' 2>/dev/null | sort -n | tail -1 || true)"
starter="$(squeue --me --noheader --name=oe-base-start --format='%A %T %M' 2>/dev/null | sort -n | tail -1 || true)"
[[ -z "$starter" ]] || { echo; echo "Initial controller: $starter"; }
[[ -z "$watcher" ]] || { echo; echo "Watcher/controller: $watcher"; }

if [[ -s "$MANIFEST" ]]; then
  echo
  echo "Temporary manifest: $MANIFEST"
fi
