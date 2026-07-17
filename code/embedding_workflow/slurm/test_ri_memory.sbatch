#!/bin/bash
#SBATCH --job-name=ri-memory-test
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=code/embedding_workflow/slurm/logs/ri_memory_%j.out
#SBATCH --error=code/embedding_workflow/slurm/logs/ri_memory_%j.err
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"; : "${SCRATCH:?SCRATCH is not set}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"; ROOT="$SCRATCH/embeddings-health/embedding_workflow/olmoearth-v1.2-nano/RI"
WORK="${L_SCRATCH:-/tmp}/ri-memory-$SLURM_JOB_ID"; mkdir -p "$WORK"; trap 'rm -rf "$WORK"' EXIT
module load devel; module load gcc/14.2.0
export UV_CACHE_DIR="$CACHE_ROOT/uv" UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-cpu" GDAL_CACHEMAX=512
cd "$REPO_DIR/code/embedding_generation"; uv sync --python 3.11
/usr/bin/time -v uv run --python 3.11 python ../embedding_workflow/merge.py --band-chunk 16 --tiles "$ROOT/olmoearth-v1.2-nano_RI_2022_tile000.tif" "$ROOT/olmoearth-v1.2-nano_RI_2022_tile001.tif" --output "$WORK/merged.tif"
/usr/bin/time -v uv run --python 3.11 python ../embedding_workflow/tests/test_ri_aggregation.py --merged "$WORK/merged.tif" --tiles "$ROOT/olmoearth-v1.2-nano_RI_2022_tile000.tif" "$ROOT/olmoearth-v1.2-nano_RI_2022_tile001.tif" --tracts "$SCRATCH/embeddings-health/data/census_tracts/tl_2020_44_tract.zip" --work-dir "$WORK/aggregation"
