#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"; : "${SCRATCH:?}"
YEAR="${YEAR:-2022}"; MODEL=clay-1.5; OUTPUT_ROOT="${OVERLAP_OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow_overlap_v1}"; COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
CACHE="$SCRATCH/embeddings-health/cache/clay_overlap"; mkdir -p "$CACHE" "$SCRIPT_DIR/logs"
EASTERN_STATES="CT DC DE FL GA IN KY MA MD ME MI NC NH NJ NY OH PA RI SC TN VA VT WV"
if [[ "${STATE_SET:-}" == eastern ]]; then STATES="${STATES:-$EASTERN_STATES}"; fi
LABEL="${STATE_SET:-all}"; [[ -n "${STATES:-}" ]] && LABEL="${LABEL}_$(tr ' ' '_' <<< "$STATES" | cut -c1-80)"
TASK_FILE="$CACHE/tasks_${YEAR}_${LABEL}.txt"; MANIFEST="$CACHE/manifest_${YEAR}_${LABEL}.json"
PY="$SCRATCH/embeddings-health/cache/venv-3.11-cpu/bin/python"
planner=("$PY" "$REPO_DIR/code/embedding_workflow/plan_clay_tasks.py" --composites "$COMPOSITE_DIR" --output-root "$OUTPUT_ROOT" --task-file "$TASK_FILE" --manifest "$MANIFEST" --year "$YEAR" --target-blocks "${TARGET_BLOCKS_PER_TILE:-1200}" --max-tasks "${MAX_TASKS_PER_WAVE:-80}" --tiling "${TILING:-rectangular}")
if [[ -n "${STATES:-}" ]]; then read -r -a state_args <<< "$STATES"; planner+=(--states "${state_args[@]}"); fi
"${planner[@]}"
COUNT=$(wc -l < "$TASK_FILE"); echo "Planned $COUNT Clay tile tasks; manifest: $MANIFEST"
[[ "$COUNT" -gt 0 ]] || exit 0
if [[ "${CONFIRM_CLAY_SUBMIT:-0}" != 1 ]]; then echo "DRY RUN (set CONFIRM_CLAY_SUBMIT=1 to submit)"; sed -n '1,20p' "$TASK_FILE"; exit 0; fi
export REPO_DIR YEAR MODEL OUTPUT_ROOT OVERLAP_OUTPUT_ROOT="$OUTPUT_ROOT" COMPOSITE_DIR TASK_FILE
INFER=$(cd "$REPO_DIR" && sbatch --parsable --export=ALL --array="0-$((COUNT-1))%${MAX_CONCURRENT:-20}" --output="$SCRIPT_DIR/logs/clay_overlap_%A_%a.out" --error="$SCRIPT_DIR/logs/clay_overlap_%A_%a.err" "$SCRIPT_DIR/run_overlap_tile.sbatch"); INFER="${INFER%%;*}"
export PRODUCT_MODE=overlap
PARTIAL=$(cd "$REPO_DIR" && sbatch --parsable --export=ALL --dependency="afterok:$INFER" --array="0-$((COUNT-1))%${MAX_CPU_CONCURRENT:-20}" --output="$SCRIPT_DIR/logs/clay_partial_%A_%a.out" --error="$SCRIPT_DIR/logs/clay_partial_%A_%a.err" "$SCRIPT_DIR/aggregate_tile.sbatch"); PARTIAL="${PARTIAL%%;*}"
echo "Submitted Clay wave: inference=$INFER partials=$PARTIAL"
