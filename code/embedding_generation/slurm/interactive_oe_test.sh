#!/bin/bash
# Interactive test: load OlmoEarth v1.1-Base and run on 3 chips of RI.
# Run via: srun -p gpu -G 1 -c 8 --mem=64G --time=00:20:00 bash <this file>
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR=2022
VARIANT="v1_1-Base"
STATE=RI

: "${SCRATCH:?SCRATCH not set}"
COMPOSITE="$SCRATCH/embeddings-health/olmoearth_composites/s2_annual_${STATE}_${YEAR}_olmoearth.tif"
CACHE_ROOT="$SCRATCH/embeddings-health/cache"
CKPT_DIR="$SCRATCH/embeddings-health/checkpoints/olmoearth/${STATE}_test"
OUT="$CKPT_DIR/oe_test_${STATE}_${YEAR}.tif"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-$(hostname -s)"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export OMP_NUM_THREADS=8

module load devel
module load gcc/14.2.0
module load rust/1.90.0
module load cuda/12.6.1 || true
export CC="$(command -v gcc)" CXX="$(command -v g++)"

UV_INSTALL_DIR="$CACHE_ROOT/uv-bin"
[[ -x "$UV_INSTALL_DIR/uv" ]] && export PATH="$UV_INSTALL_DIR:$PATH"

echo "=== Environment ==="
echo "Node:  $(hostname -s)"
echo "uv:    $(command -v uv && uv --version || echo NOT FOUND)"
echo "gcc:   $(gcc --version | head -n 1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo ""

(cd "$EMBED_DIR" && uv sync --python 3.11 2>&1 | tail -3)

if ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import olmoearth_pretrain" 2>/dev/null; then
  echo "Installing olmoearth_pretrain..."
  VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install --no-deps \
    git+https://github.com/allenai/olmoearth_pretrain
  VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install \
    "einops>=0.7.0" "huggingface_hub" "numpy>=1.26.4" "universal-pathlib>=0.2.5"
fi

mkdir -p "$CKPT_DIR"

echo "=== Running embed.py --test-chips 3 ==="
cd "$EMBED_DIR"
uv run --python 3.11 python embed.py \
  --model     olmoearth \
  --variant   "$VARIANT" \
  --input     "$COMPOSITE" \
  --output    "$OUT" \
  --year      "$YEAR" \
  --batch-size 4 \
  --test-chips 3 \
  --force

echo ""
echo "=== Output ==="
ls -lh "$OUT" 2>/dev/null && python3 -c "
import rasterio, numpy as np
with rasterio.open('$OUT') as s:
    print(f'Shape: {s.count} bands × {s.height} × {s.width}')
    print(f'Resolution: {s.res[0]:.1f} m')
    d = s.read()
    print(f'Value range: {np.nanmin(d):.4f} – {np.nanmax(d):.4f}')
" || true
