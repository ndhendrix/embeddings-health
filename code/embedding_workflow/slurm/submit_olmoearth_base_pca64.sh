#!/bin/bash
# Fit national PCA64, aggregate all OE-Base tiles in bounded parallel chunks,
# reduce state tract products, then run the PCA64 analysis.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
: "${SCRATCH:?Set SCRATCH before submitting.}"

YEAR="${YEAR:-2022}"
MODEL="olmoearth-v1.2-base"
STATES="AL AR AZ CA CO CT DC DE FL GA IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY"
OUTPUT_ROOT="${OVERLAP_OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow_overlap_v1}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites_tiled512}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
CACHE="$CACHE_ROOT/olmoearth_base_overlap"
AGG_DIR="$SCRATCH/embeddings-health/olmoearth_base_aggregated_pca64"
PCA_MODEL="$AGG_DIR/national_pca/olmoearth_v1.2_base_national_pca.pkl"
LOG_DIR="$REPO_DIR/code/embedding_workflow/slurm/logs"
ANALYSIS_LOG_DIR="$REPO_DIR/code/analyses/slurm/logs"
CHUNKS="${CHUNKS:-32}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
RUN_ID="pca64_$(date +%Y%m%dT%H%M%S)_$$"
TASK_FILE="$CACHE/tasks_2022_conus_$RUN_ID.txt"
ALL_TASK_FILE="$CACHE/all_tasks_2022_conus_$RUN_ID.txt"
STATE_FILE="$CACHE/states_2022_conus_$RUN_ID.txt"
MANIFEST="$CACHE/manifest_2022_conus_$RUN_ID.json"
CHUNK_DIR="$CACHE/pca64_task_chunks_$RUN_ID"
PY="$CACHE_ROOT/venv-3.11-cpu/bin/python"

mkdir -p "$CACHE" "$AGG_DIR/national_pca" "$LOG_DIR" "$ANALYSIS_LOG_DIR" "$CHUNK_DIR"
read -r -a state_args <<< "$STATES"
"$PY" "$REPO_DIR/code/embedding_workflow/plan_overlap_tasks.py" \
  --model "$MODEL" \
  --composites "$COMPOSITE_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --task-file "$TASK_FILE" \
  --all-task-file "$ALL_TASK_FILE" \
  --state-file "$STATE_FILE" \
  --manifest "$MANIFEST" \
  --year "$YEAR" \
  --target-blocks 4000 \
  --max-tasks 100000 \
  --tiling rectangular \
  --states "${state_args[@]}"

jq -e '
  .selected_states == 49 and
  .total_tiles == 8867 and
  .total_missing_tiles == 0 and
  .dimensions == 768
' "$MANIFEST" >/dev/null
(( $(wc -l < "$ALL_TASK_FILE") == 8867 )) || { echo "ERROR: expected 8867 PCA tasks" >&2; exit 1; }
(( $(wc -l < "$STATE_FILE") == 49 )) || { echo "ERROR: expected 49 states" >&2; exit 1; }
(( CHUNKS >= 1 && CHUNKS <= 99 )) || { echo "ERROR: CHUNKS must be between 1 and 99" >&2; exit 1; }
(( MAX_CONCURRENT >= 1 && MAX_CONCURRENT <= CHUNKS )) || { echo "ERROR: invalid MAX_CONCURRENT" >&2; exit 1; }

for ((i=0; i<CHUNKS; i++)); do
  printf -v chunk '%s/chunk_%02d.txt' "$CHUNK_DIR" "$i"
  : > "$chunk"
done
awk -v dir="$CHUNK_DIR" -v chunks="$CHUNKS" '
  {path=sprintf("%s/chunk_%02d.txt", dir, (NR-1)%chunks); print >> path}
' "$ALL_TASK_FILE"
for ((i=0; i<CHUNKS; i++)); do
  printf -v chunk '%s/chunk_%02d.txt' "$CHUNK_DIR" "$i"
  [[ -s "$chunk" ]] || { echo "ERROR: empty task chunk: $chunk" >&2; exit 1; }
done

echo "OE-Base PCA64 plan: 49 states, 8867 tiles, $CHUNKS chunks, concurrency $MAX_CONCURRENT"
echo "PCA model: $PCA_MODEL"
echo "Aggregation output: $AGG_DIR"
if [[ "${DRY_RUN:-0}" == 1 || "${CONFIRM_OE_BASE_PCA64_SUBMIT:-0}" != 1 ]]; then
  echo "DRY RUN (set CONFIRM_OE_BASE_PCA64_SUBMIT=1 to submit)"
  exit 0
fi

common_export="ALL,REPO_DIR=$REPO_DIR,SCRATCH=$SCRATCH,MODEL=$MODEL,YEAR=$YEAR,OUTPUT_ROOT=$OUTPUT_ROOT,CACHE_ROOT=$CACHE_ROOT"
fit=$(sbatch --parsable \
  --job-name=oe-base-pca64-fit \
  --output="$LOG_DIR/oe_base_pca64_fit_%j.out" \
  --error="$LOG_DIR/oe_base_pca64_fit_%j.err" \
  --export="$common_export,TASK_FILE=$ALL_TASK_FILE,PCA_MODEL=$PCA_MODEL,PCA_INPUT_DIMS=768,PCA_COMPONENTS=64" \
  "$SCRIPT_DIR/fit_overlap_pca64.sbatch")
fit="${fit%%;*}"

partials=$(sbatch --parsable \
  --dependency="afterok:$fit" \
  --array="0-$((CHUNKS - 1))%$MAX_CONCURRENT" \
  --job-name=oe-base-pca64-partials \
  --output="$LOG_DIR/oe_base_pca64_partials_%A_%a.out" \
  --error="$LOG_DIR/oe_base_pca64_partials_%A_%a.err" \
  --export="$common_export,CHUNK_DIR=$CHUNK_DIR,AGG_SUFFIX=_pca64,PCA_MODEL=$PCA_MODEL,WORKERS=4,BAND_CHUNK=64" \
  "$SCRIPT_DIR/run_pca64_chunk.sbatch")
partials="${partials%%;*}"

reduce=$(sbatch --parsable \
  --dependency="afterok:$partials" \
  --job-name=oe-base-pca64-reduce \
  --output="$LOG_DIR/oe_base_pca64_reduce_%j.out" \
  --error="$LOG_DIR/oe_base_pca64_reduce_%j.err" \
  --export="$common_export,STATE_FILE=$STATE_FILE,AGG_SUFFIX=_pca64,AGG_OUT_DIR=$AGG_DIR,FEATURE_PREFIX=PC" \
  "$SCRIPT_DIR/reduce_complete_states_driver.sbatch")
reduce="${reduce%%;*}"

analysis=$(sbatch --parsable \
  --dependency="afterok:$reduce" \
  --job-name=emb-health-oe-base-pca64 \
  --output="$ANALYSIS_LOG_DIR/oe_base_pca64_%j.out" \
  --error="$ANALYSIS_LOG_DIR/oe_base_pca64_%j.err" \
  --export=ALL,REPO_DIR="$REPO_DIR",SCRATCH="$SCRATCH",CACHE_ROOT="$CACHE_ROOT" \
  "$REPO_DIR/code/analyses/slurm/run_analysis_olmoearth_base_pca64.sbatch")
analysis="${analysis%%;*}"

jobs_file="$CACHE/pca64_pipeline_jobs_$RUN_ID.txt"
printf 'fit=%s\npartials=%s\nreduce=%s\nanalysis=%s\n' \
  "$fit" "$partials" "$reduce" "$analysis" > "$jobs_file"
echo "Submitted PCA fit=$fit partials=$partials reduce=$reduce analysis=$analysis"
echo "Pipeline record: $jobs_file"
