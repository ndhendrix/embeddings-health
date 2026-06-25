#!/bin/bash
#SBATCH --job-name=composite-dc
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=code/embedding_generation/slurm/logs/composite_dc_%j.out
#SBATCH --error=code/embedding_generation/slurm/logs/composite_dc_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"

: "${SCRATCH:?SCRATCH is not set.}"
OUT_DIR="${OUT_DIR:-$SCRATCH/embeddings-health/data/prithvi_data}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"

mkdir -p "$OUT_DIR" "$CACHE_ROOT"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-${SLURMD_NODENAME:-$(hostname -s)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if command -v module >/dev/null 2>&1; then
  module load devel
  module load gcc/14.2.0
fi
export CC="$(command -v gcc)" CXX="$(command -v g++)"

if ! command -v uv >/dev/null 2>&1; then
  UV_INSTALL_DIR="${UV_INSTALL_DIR:-$CACHE_ROOT/uv-bin}"
  if [[ -x "$UV_INSTALL_DIR/uv" ]]; then
    export PATH="$UV_INSTALL_DIR:$PATH"
  else
    echo "ERROR: uv not found. Run an embedding job first to bootstrap it, or set UV_INSTALL_DIR." >&2
    exit 1
  fi
fi

echo "uv: $(command -v uv)  $(uv --version)"
echo "gcc: $(gcc --version | head -n 1)"
echo "=== Composite DC ==="
echo "Output dir: $OUT_DIR"
echo "Year:       $YEAR"

cd "$EMBED_DIR"

uv run --python 3.11 python composite.py \
  --state DC \
  --model prithvi \
  --year "$YEAR" \
  --output-dir "$OUT_DIR"

echo ""
echo "=== Done — output files ==="
ls -lh "$OUT_DIR"/s2_*_DC_${YEAR}_prithvi.tif
