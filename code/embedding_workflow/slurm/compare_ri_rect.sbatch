#!/bin/bash
#SBATCH --job-name=ri-rect-compare
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=code/embedding_workflow/slurm/logs/ri_rect_compare_%j.out
#SBATCH --error=code/embedding_workflow/slurm/logs/ri_rect_compare_%j.err
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"; : "${REFERENCE:?}"; : "${CANDIDATE:?}"; : "${COMPARE_JSON:?}"; : "${SCRATCH:?}"
module load devel; module load gcc/14.2.0
export UV_PROJECT_ENVIRONMENT="$SCRATCH/embeddings-health/cache/venv-3.11-cpu" UV_CACHE_DIR="$SCRATCH/embeddings-health/cache/uv" GDAL_CACHEMAX=512
cd "$REPO_DIR/code/embedding_generation"; uv sync --python 3.11
extra=(--rtol "${RTOL:-2e-5}" --atol "${ATOL:-1e-5}")
[[ -n "${REFERENCE_CSV:-}" ]] && extra+=(--reference-csv "$REFERENCE_CSV" --candidate-csv "$CANDIDATE_CSV")
uv run --python 3.11 python ../embedding_workflow/compare_overlap.py --reference "$REFERENCE" --candidate "$CANDIDATE" --output "$COMPARE_JSON" ${extra[@]:-}
