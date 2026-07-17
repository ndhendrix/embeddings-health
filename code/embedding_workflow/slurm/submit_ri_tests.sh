#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
mkdir -p "$SCRIPT_DIR/logs"
job=$(cd "$REPO_DIR" && sbatch --parsable --export=ALL "$SCRIPT_DIR/test_ri_merge.sbatch")
echo "Submitted RI exact merge test: ${job%%;*}"
memory=$(cd "$REPO_DIR" && sbatch --parsable --export=ALL "$SCRIPT_DIR/test_ri_memory.sbatch")
echo "Submitted RI bounded-memory merge+aggregation test: ${memory%%;*}"
if [[ "${SUBMIT_GPU_SMOKE:-0}" == 1 ]]; then
  for model in olmoearth-v1.2-nano olmoearth-v1.2-base clay-1.5; do
    MODEL="$model" STATE=RI NUM_TILES=2 TEST_CHIPS=2 bash "$SCRIPT_DIR/submit_state.sh"
  done
fi
