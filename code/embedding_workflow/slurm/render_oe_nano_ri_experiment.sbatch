#!/bin/bash
#SBATCH --job-name=oe-nano-ri-rgb
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=code/embedding_workflow/slurm/logs/oe_nano_ri_rgb_%j.out
#SBATCH --error=code/embedding_workflow/slurm/logs/oe_nano_ri_rgb_%j.err
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"; : "${SCRATCH:?}"; : "${OE_NANO_RI_ROOT:?}"
export MPLBACKEND=Agg
cd "$REPO_DIR/code/figures"
"$SCRATCH/embeddings-health/cache/venv-3.11-cpu/bin/python" oe_nano_ri_experiment_grid.py
