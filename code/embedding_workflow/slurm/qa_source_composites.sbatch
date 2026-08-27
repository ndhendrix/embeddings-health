#!/bin/bash
#SBATCH --job-name=qa-src-comp
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=code/embedding_workflow/slurm/logs/qa_source_composites_%A_%a.out
#SBATCH --error=code/embedding_workflow/slurm/logs/qa_source_composites_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?SCRATCH is not set}"
: "${STATE_LIST:?STATE_LIST must be a colon-separated state list}"

IFS=: read -ra STATES <<< "$STATE_LIST"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
STATE="${STATES[$TASK_ID]}"

CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
PY="${PY:-$CACHE_ROOT/venv-3.11-cpu/bin/python}"
COMPOSITE_ROOT="${COMPOSITE_ROOT:-$SCRATCH/embeddings-health/olmoearth_composites}"
TRACT_ROOT="${TRACT_ROOT:-$SCRATCH/embeddings-health/data/census_tracts}"
QA_OUTPUT_DIR="${QA_OUTPUT_DIR:-$SCRATCH/embeddings-health/qa/source_composites_${YEAR:-2022}}"
YEAR="${YEAR:-2022}"
TRACT_YEAR="${TRACT_YEAR:-2020}"
WINDOW_HALF="${WINDOW_HALF:-16}"
FAIL_BELOW_PCT="${FAIL_BELOW_PCT:-90}"

cd "$REPO_DIR"
"$PY" code/embedding_workflow/qa_source_composites.py \
  --state "$STATE" \
  --composite-root "$COMPOSITE_ROOT" \
  --tract-root "$TRACT_ROOT" \
  --output-dir "$QA_OUTPUT_DIR" \
  --year "$YEAR" \
  --tract-year "$TRACT_YEAR" \
  --window-half "$WINDOW_HALF" \
  --fail-below-pct "$FAIL_BELOW_PCT"
