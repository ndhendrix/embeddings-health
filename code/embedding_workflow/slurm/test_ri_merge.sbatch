#!/bin/bash
#SBATCH --job-name=ri-merge-test
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=code/embedding_workflow/slurm/logs/ri_merge_%j.out
#SBATCH --error=code/embedding_workflow/slurm/logs/ri_merge_%j.err
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?SCRATCH is not set}"
INPUT="${INPUT:-$SCRATCH/embeddings-health/olmoearth_composites/s2_annual_RI_2022_olmoearth.tif}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
module load devel
module load gcc/14.2.0
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-cpu"
WORK="${L_SCRATCH:-/tmp}/ri-merge-$SLURM_JOB_ID"
trap 'rm -rf "$WORK"' EXIT
cd "$REPO_DIR/code/embedding_generation"
uv sync --python 3.11
uv run --python 3.11 python ../embedding_workflow/tests/test_ri_merge.py --input "$INPUT" --work-dir "$WORK"
