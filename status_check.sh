#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRATCH="${SCRATCH:-/scratch/users/nhendrix}"
YEAR="${YEAR:-2022}"
MODEL="${MODEL:-olmoearth-v1.2-nano}"
TARGET_BLOCKS_PER_TILE="${TARGET_BLOCKS_PER_TILE:-20000}"
STATE_SET="${STATE_SET:-final49}"
STATES="${STATES:-AL AR AZ CA CO CT DC DE FL GA IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY}"

DEFAULT_COMPOSITES="$SCRATCH/embeddings-health/olmoearth_composites_final_20260802"
if [[ ! -d "$DEFAULT_COMPOSITES" ]]; then
  DEFAULT_COMPOSITES="$SCRATCH/embeddings-health/olmoearth_composites"
fi
COMPOSITES="${COMPOSITES:-$DEFAULT_COMPOSITES}"
OUTPUT_ROOT="${OVERLAP_OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow_overlap_v1}"
PY="${PY:-$SCRATCH/embeddings-health/cache/venv-3.11-cpu/bin/python}"

STAMP="$(date +%Y%m%d_%H%M%S)"
TASK_FILE="${TASK_FILE:-/tmp/oe_nano_tasks_${YEAR}_${STAMP}.txt}"
ALL_TASK_FILE="${ALL_TASK_FILE:-/tmp/oe_nano_all_tasks_${YEAR}_${STAMP}.txt}"
STATE_FILE="${STATE_FILE:-/tmp/oe_nano_states_${YEAR}_${STAMP}.txt}"
MANIFEST="${MANIFEST:-/tmp/oe_nano_manifest_${YEAR}_${STAMP}.json}"

read -r -a state_args <<< "$STATES"
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

echo "Current all-state OE-Nano status:"
echo "- Model: $MODEL"
echo "- Composites: $COMPOSITES"
echo "$summary" | sed 's/^/- /'
echo
echo "Incomplete states:"
jq -r '
  .states[]
  | select((.missing_tiles | length) > 0)
  | "\(.state)\t\(.missing_tiles | length)\t\(.num_tiles - (.missing_tiles | length))/\(.num_tiles)"
' "$MANIFEST" | awk 'BEGIN {printf "%-3s %10s   %s\n", "ST", "remaining", "done"} {printf "%-3s %10d   %s\n", $1, $2, $3}'

echo
echo "Live Slurm status:"
squeue --me || true

current_wave="$(squeue --me --noheader --name=oe-nano-overlap --format='%i' 2>/dev/null | sed -E 's/(\[.*|_.*)//g' | sort -n | tail -1 || true)"
watcher="$(squeue --me --noheader --name=oe-nano-overlap-resubmit --format='%A %T %M' 2>/dev/null | sort -n | tail -1 || true)"

if [[ -n "$current_wave" ]]; then
  echo
  echo "Current inference wave: $current_wave"
  sacct -j "$current_wave" --format=JobID,State -P 2>/dev/null \
    | awk -F'|' -v wave="$current_wave" '$1 ~ "^" wave "_[0-9]+$" {c[$2]++} END {for (s in c) print s, c[s]}' \
    | sort || true
fi

if [[ -n "$watcher" ]]; then
  echo
  echo "Watcher/controller: $watcher"
fi

echo
echo "Temporary manifest: $MANIFEST"
