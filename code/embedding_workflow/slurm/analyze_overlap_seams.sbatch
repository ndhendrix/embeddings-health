#!/bin/bash
#SBATCH --job-name=overlap-seams
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=code/embedding_workflow/slurm/logs/overlap_seams_%j.out
#SBATCH --error=code/embedding_workflow/slurm/logs/overlap_seams_%j.err
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"; : "${RASTER:?}"; : "${SEAM_JSON:?}"; : "${SCRATCH:?}"
module load devel; module load gcc/14.2.0
export UV_PROJECT_ENVIRONMENT="$SCRATCH/embeddings-health/cache/venv-3.11-cpu" UV_CACHE_DIR="$SCRATCH/embeddings-health/cache/uv" GDAL_CACHEMAX=512
cd "$REPO_DIR/code/embedding_generation"; uv sync --python 3.11
uv run --python 3.11 python ../embedding_workflow/analyze_overlap_seams.py --raster "$RASTER" --output "$SEAM_JSON"
