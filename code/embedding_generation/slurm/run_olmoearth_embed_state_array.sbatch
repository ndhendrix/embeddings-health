#!/bin/bash
#SBATCH --job-name=oe-embed
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=code/embedding_generation/slurm/logs/oe_embed_%A_%a.out
#SBATCH --error=code/embedding_generation/slurm/logs/oe_embed_%A_%a.err

# Per-tile OlmoEarth Base embedding job. SLURM_ARRAY_TASK_ID indexes into
# TILE_TASK_FILE, a newline-delimited file with lines of the form:
#   STATE TILE_IDX NUM_TILES
# created by submit_olmoearth_embed_all_states.sh.
#
# Reads the composite directly from $SCRATCH with windowed reads (no local
# staging) — each tile only touches its own row-band, so there's no
# multi-hundred-GB copy to make or leak. Checkpoints every CHECKPOINT_EVERY
# chips so a timed-out tile resumes mid-tile rather than from scratch. Base's
# per-chip cost (~0.75s/chip measured on NY) is the main reason tiling is
# needed here at all — large states need tens of hours of total inference,
# far more than any single job's walltime, so bounding chip count per task
# via tiling (rather than trying to fit a whole state in one job) is what
# makes forward progress possible.
#
# NUM_TILES=1 states (the common case for small/medium states) behave exactly
# as before: embed.py writes directly to the final name, no tile suffix, and
# this script moves it straight to FINAL_DIR. NUM_TILES>1 states leave their
# tile output in CKPT_DIR for run_olmoearth_embed_merge.sbatch to mosaic.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"
VARIANT="${VARIANT:-v1_1-Base}"

: "${SCRATCH:?SCRATCH is not set}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$SCRATCH/embeddings-health/olmoearth_embeddings}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
: "${TILE_TASK_FILE:?TILE_TASK_FILE must be set to the task list file path}"

LINE=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "$TILE_TASK_FILE")
STATE=$(echo "$LINE" | awk '{print $1}')
TILE_IDX=$(echo "$LINE" | awk '{print $2}')
NUM_TILES=$(echo "$LINE" | awk '{print $3}')

if [[ -z "$STATE" || -z "$TILE_IDX" || -z "$NUM_TILES" ]]; then
  echo "ERROR: could not parse line $(( SLURM_ARRAY_TASK_ID + 1 )) of $TILE_TASK_FILE: '$LINE'" >&2
  exit 1
fi

COMPOSITE="$COMPOSITE_DIR/s2_annual_${STATE}_${YEAR}_olmoearth.tif"

if [[ ! -s "$COMPOSITE" ]]; then
  echo "ERROR: composite not found or empty: $COMPOSITE" >&2
  exit 1
fi

CKPT_DIR="$SCRATCH/embeddings-health/checkpoints/olmoearth/$STATE"
FINAL_DIR="$FINAL_OUT_DIR/$STATE"

PREEMPTED=0
CHILD_PID=""

handle_preempt() {
  PREEMPTED=1
  echo "Job preempted or timed out — checkpoint preserved at: $CKPT_DIR" >&2
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$CHILD_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$CHILD_PID" 2>/dev/null || true
  fi
}

preserve_failure() {
  status=$?
  if [[ "$PREEMPTED" == 1 ]]; then
    exit 0
  fi
  echo "ERROR: task failed with status $status — partial outputs in $CKPT_DIR" >&2
  exit "$status"
}

trap handle_preempt TERM
trap preserve_failure ERR

mkdir -p "$CKPT_DIR" "$FINAL_DIR" "$CACHE_ROOT"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_DATA_DIR="$CACHE_ROOT/uv-data"
export UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-${SLURMD_NODENAME:-$(hostname -s)}"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if command -v module >/dev/null 2>&1; then
  module load devel
  module load gcc/14.2.0
  module load rust/1.90.0
  module load cuda/12.6.1 || true
fi
export CC="$(command -v gcc)" CXX="$(command -v g++)"

if ! command -v uv >/dev/null 2>&1; then
  UV_INSTALL_DIR="${UV_INSTALL_DIR:-$CACHE_ROOT/uv-bin}"
  mkdir -p "$UV_INSTALL_DIR"
  if [[ -x "$UV_INSTALL_DIR/uv" ]]; then
    export PATH="$UV_INSTALL_DIR:$PATH"
  else
    echo "Installing uv into $UV_INSTALL_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
    export PATH="$UV_INSTALL_DIR:$PATH"
  fi
fi

echo "uv: $(command -v uv)"
uv --version

(cd "$EMBED_DIR" && uv sync --python 3.11 2>&1 | tail -5)

# olmoearth_pretrain is excluded from pyproject.toml because its metadata pins
# torch>=2.7, which conflicts with Sherlock's manylinux baseline. --no-deps
# bypasses that metadata check; the library works fine with torch<2.7 at
# runtime.
if ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import olmoearth_pretrain" 2>/dev/null; then
  echo "Installing olmoearth_pretrain (--no-deps to skip torch>=2.7 metadata conflict)..."
  VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install --no-deps \
    git+https://github.com/allenai/olmoearth_pretrain
  VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT" uv pip install \
    "einops>=0.7.0" "huggingface_hub" "numpy>=1.26.4" "universal-pathlib>=0.2.5"
  echo "olmoearth_pretrain installed."
else
  echo "olmoearth_pretrain already present — skipping install."
fi

SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"
OUTPUT_BASENAME="olmoearth_${SAFE_VARIANT}_${STATE}_${YEAR}"
RAW_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.tif"

BATCH_SIZE="${BATCH_SIZE:-32}"

TEST_CHIPS_ARGS=""
if [[ -n "${TEST_CHIPS:-}" ]]; then
  TEST_CHIPS_ARGS="--test-chips $TEST_CHIPS"
fi

FORCE_FLAG=""
if [[ -n "${FORCE_RESTART:-}" ]]; then
  FORCE_FLAG="--force"
fi

echo "=== OlmoEarth Base embedding task ==="
echo "State:      $STATE"
echo "Tile:       $TILE_IDX / $NUM_TILES"
echo "Year:       $YEAR"
echo "Variant:    $VARIANT"
echo "Composite:  $COMPOSITE"
echo "Ckpt dir:   $CKPT_DIR"
echo "Final dir:  $FINAL_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Test chips: ${TEST_CHIPS:-none}"
nvidia-smi || true

cd "$EMBED_DIR"

uv run --python 3.11 python embed.py \
  --model        olmoearth \
  --variant      "$VARIANT" \
  --input        "$COMPOSITE" \
  --output       "$RAW_LOCAL" \
  --tile-index   "$TILE_IDX" \
  --num-tiles    "$NUM_TILES" \
  --year         "$YEAR" \
  --batch-size   "$BATCH_SIZE" \
  --checkpoint-every "${CHECKPOINT_EVERY:-25}" \
  $FORCE_FLAG \
  $TEST_CHIPS_ARGS &
CHILD_PID=$!
wait "$CHILD_PID"

if (( NUM_TILES > 1 )); then
  TILE_OUTPUT="$CKPT_DIR/${OUTPUT_BASENAME}_tile$(printf '%03d' "$TILE_IDX").tif"
  if [[ ! -s "$TILE_OUTPUT" ]]; then
    echo "ERROR: expected tile output missing or empty: $TILE_OUTPUT" >&2
    exit 1
  fi
  trap - ERR TERM
  echo "=== Done: $STATE tile $TILE_IDX/$NUM_TILES ==="
  ls -lh "$TILE_OUTPUT"
else
  if [[ ! -s "$RAW_LOCAL" ]]; then
    echo "ERROR: expected output missing or empty: $RAW_LOCAL" >&2
    exit 1
  fi
  mv -f "$RAW_LOCAL" "$FINAL_DIR/"
  MMAP_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.ckpt.mmap"
  [[ -f "$MMAP_LOCAL" ]] && rm -f "$MMAP_LOCAL" || true
  trap - ERR TERM
  echo "=== Done ==="
  echo "Output: $FINAL_DIR/${OUTPUT_BASENAME}.tif"
fi
