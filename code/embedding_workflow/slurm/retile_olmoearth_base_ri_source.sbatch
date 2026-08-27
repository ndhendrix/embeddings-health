#!/bin/bash
#SBATCH --job-name=oe-base-retile
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=code/embedding_workflow/slurm/logs/oe_base_retile_%j.out
#SBATCH --error=code/embedding_workflow/slurm/logs/oe_base_retile_%j.err
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
INPUT="${INPUT:-$SCRATCH/embeddings-health/olmoearth_composites/s2_annual_RI_2022_olmoearth.tif}"
OUTPUT="${OUTPUT:-$SCRATCH/embeddings-health/olmoearth_base_ri_experiment_v2/source/s2_annual_RI_2022_olmoearth_tiled512_zstd.tif}"

mkdir -p "$(dirname "$OUTPUT")" "$CACHE_ROOT" "$REPO_DIR/code/embedding_workflow/slurm/logs"
module load devel
module load gcc/14.2.0
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-cpu"
export GDAL_CACHEMAX=2048

cd "$REPO_DIR/code/embedding_generation"
uv sync --python 3.11
uv run --python 3.11 python ../embedding_workflow/retile_source.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --block-size 512
