#!/bin/bash
#SBATCH --job-name=providence-rgb
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=code/embedding_workflow/slurm/logs/providence_rgb_%j.out
#SBATCH --error=code/embedding_workflow/slurm/logs/providence_rgb_%j.err
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"; : "${SCRATCH:?}"
export PROVIDENCE_ROOT="${PROVIDENCE_ROOT:-$SCRATCH/embeddings-health/providence_overlap_v1}"
export MPLBACKEND=Agg
cd "$REPO_DIR/code/figures"
"$SCRATCH/embeddings-health/cache/venv-3.11-cpu/bin/python" providence_overlap_grid.py
