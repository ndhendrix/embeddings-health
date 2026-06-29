#!/bin/bash
#SBATCH --job-name=oe-nano-embed
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=code/embedding_generation/slurm/logs/oe_nano_embed_%A_%a.out
#SBATCH --error=code/embedding_generation/slurm/logs/oe_nano_embed_%A_%a.err

# OlmoEarth Nano embedding job — identical setup to the Base script but for
# the Nano variant (embedding_size=128 vs 768).
#
# STATE_LIST must be set by the submit script (colon-separated state abbrevs).
# SLURM_ARRAY_TASK_ID indexes into it.
#
# Nano is ~6× smaller than Base so 64 GB RAM is sufficient for all CONUS states.
# Checkpoints are written every CHECKPOINT_EVERY chips (default 25).

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"
VARIANT="${VARIANT:-Nano}"

: "${SCRATCH:?SCRATCH is not set}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$SCRATCH/embeddings-health/olmoearth_nano_embeddings}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"

: "${STATE_LIST:?STATE_LIST must be set to a colon-separated ordered state list}"
IFS=: read -ra STATES <<< "$STATE_LIST"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID < 0 || TASK_ID >= ${#STATES[@]} )); then
  echo "ERROR: TASK_ID=$TASK_ID out of range (${#STATES[@]} states)" >&2
  exit 1
fi

STATE="${STATES[$TASK_ID]}"
COMPOSITE="$COMPOSITE_DIR/s2_annual_${STATE}_${YEAR}_olmoearth.tif"

if [[ ! -s "$COMPOSITE" ]]; then
  echo "ERROR: composite not found or empty: $COMPOSITE" >&2
  exit 1
fi

# Stage composite to fast node-local SSD
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  STAGE_DIR="$SLURM_TMPDIR/oe_nano_${STATE}_${YEAR}"
else
  STAGE_DIR="$SCRATCH/embeddings-health/staging/${SLURM_JOB_ID:-manual}_${TASK_ID}"
fi

INPUT_STAGE="$STAGE_DIR/input"
CKPT_DIR="$SCRATCH/embeddings-health/checkpoints/olmoearth_nano/$STATE"
FINAL_DIR="$FINAL_OUT_DIR/$STATE"

PREEMPTED=0

cleanup_stage() {
  if [[ -n "${SLURM_TMPDIR:-}" && "$STAGE_DIR" == "$SLURM_TMPDIR/"* ]] || \
     [[ -n "${SCRATCH:-}" && "$STAGE_DIR" == "$SCRATCH/embeddings-health/staging/"* ]]; then
    rm -rf "$STAGE_DIR" || true
  else
    echo "Refusing to remove unexpected staging path: $STAGE_DIR" >&2
  fi
}

cleanup_success() {
  cleanup_stage
  rm -rf "$CKPT_DIR" || true
}

handle_preempt() {
  PREEMPTED=1
  echo "Job preempted or timed out — checkpoint preserved at: $CKPT_DIR" >&2
}

preserve_failure() {
  status=$?
  if [[ "$PREEMPTED" == 1 ]]; then
    cleanup_stage
    exit 0
  fi
  echo "ERROR: task failed with status $status — partial outputs in $CKPT_DIR" >&2
  cleanup_stage
  exit "$status"
}

trap handle_preempt TERM
trap preserve_failure ERR

mkdir -p "$INPUT_STAGE" "$CKPT_DIR" "$FINAL_DIR" "$CACHE_ROOT"

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

# olmoearth_pretrain is needed for all OlmoEarth variants including Nano.
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

cp "$COMPOSITE" "$INPUT_STAGE/"
STAGED_COMPOSITE="$INPUT_STAGE/$(basename "$COMPOSITE")"

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

echo "=== OlmoEarth Nano embedding task ==="
echo "State:      $STATE"
echo "Year:       $YEAR"
echo "Variant:    $VARIANT"
echo "Task ID:    $TASK_ID"
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
  --input        "$STAGED_COMPOSITE" \
  --output       "$RAW_LOCAL" \
  --year         "$YEAR" \
  --batch-size   "$BATCH_SIZE" \
  --checkpoint-every "${CHECKPOINT_EVERY:-25}" \
  $FORCE_FLAG \
  $TEST_CHIPS_ARGS

if [[ ! -s "$RAW_LOCAL" ]]; then
  echo "ERROR: expected output missing or empty: $RAW_LOCAL" >&2
  exit 1
fi

mv -f "$RAW_LOCAL" "$FINAL_DIR/"
MMAP_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.ckpt.mmap"
[[ -f "$MMAP_LOCAL" ]] && rm -f "$MMAP_LOCAL" || true

trap - ERR TERM
cleanup_success

echo "=== Done ==="
echo "Output: $FINAL_DIR/${OUTPUT_BASENAME}.tif"
