#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
: "${MODEL:?MODEL is required}"; : "${STATE:?STATE is required}"
NUM_TILES="${NUM_TILES:-4}"; YEAR="${YEAR:-2022}"
export REPO_DIR MODEL STATE NUM_TILES YEAR
mkdir -p "$SCRIPT_DIR/logs"
tiles=$(cd "$REPO_DIR" && sbatch --parsable --export=ALL --array="0-$((NUM_TILES-1))" --output="$SCRIPT_DIR/logs/${MODEL}_${STATE}_tile_%A_%a.out" --error="$SCRIPT_DIR/logs/${MODEL}_${STATE}_tile_%A_%a.err" "$SCRIPT_DIR/run_state_tile.sbatch"); tiles="${tiles%%;*}"
merge=$(cd "$REPO_DIR" && sbatch --parsable --export=ALL --dependency="afterok:$tiles" --output="$SCRIPT_DIR/logs/${MODEL}_${STATE}_merge_%j.out" --error="$SCRIPT_DIR/logs/${MODEL}_${STATE}_merge_%j.err" "$SCRIPT_DIR/merge_state.sbatch")
merge="${merge%%;*}"
aggregate=""; partial=""
if [[ "${AGGREGATE:-0}" == 1 ]]; then
  partial=$(cd "$REPO_DIR" && sbatch --parsable --export=ALL --dependency="afterok:$tiles" --array="0-$((NUM_TILES-1))" --output="$SCRIPT_DIR/logs/${MODEL}_${STATE}_partial_%A_%a.out" --error="$SCRIPT_DIR/logs/${MODEL}_${STATE}_partial_%A_%a.err" "$SCRIPT_DIR/aggregate_tile.sbatch"); partial="${partial%%;*}"
  aggregate=$(cd "$REPO_DIR" && sbatch --parsable --export=ALL --dependency="afterok:$partial" --output="$SCRIPT_DIR/logs/${MODEL}_${STATE}_reduce_%j.out" --error="$SCRIPT_DIR/logs/${MODEL}_${STATE}_reduce_%j.err" "$SCRIPT_DIR/reduce_tracts.sbatch"); aggregate="${aggregate%%;*}"
fi
echo "Submitted $MODEL $STATE: tiles=$tiles merge=$merge${partial:+ partials=$partial}${aggregate:+ reduce=$aggregate}"
