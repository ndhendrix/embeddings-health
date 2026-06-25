#!/bin/bash
#SBATCH --job-name=prithvi-embed
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=code/embedding_generation/slurm/logs/prithvi_embed_%A_%a.out
#SBATCH --error=code/embedding_generation/slurm/logs/prithvi_embed_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"
DATA_DIR="${DATA_DIR:-$SCRATCH/embeddings-health/data/prithvi_data}"

if [[ -z "${FINAL_OUT_DIR:-}" ]]; then
  : "${SCRATCH:?Set SCRATCH or FINAL_OUT_DIR before submitting.}"
  FINAL_OUT_DIR="$SCRATCH/embeddings-health/prithvi_embeddings"
fi

if [[ -z "${CACHE_ROOT:-}" ]]; then
  : "${SCRATCH:?Set SCRATCH or CACHE_ROOT before submitting.}"
  CACHE_ROOT="$SCRATCH/embeddings-health/cache"
fi

MODEL_VARIANTS=("tiny" "300M-TL")
NUM_MODELS="${#MODEL_VARIANTS[@]}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
STATE_INDEX=$((TASK_ID / NUM_MODELS))
MODEL_INDEX=$((TASK_ID % NUM_MODELS))

STATES=()
while IFS= read -r state; do
  STATES+=("$state")
done < <(
  find "$DATA_DIR" -maxdepth 1 -type f -name "s2_spring_*_${YEAR}_prithvi.tif" -exec basename {} \; \
    | sed -E "s/^s2_spring_(.*)_${YEAR}_prithvi\.tif$/\1/" \
    | sort
)

if (( ${#STATES[@]} == 0 )); then
  echo "ERROR: no spring composites found in $DATA_DIR for year $YEAR" >&2
  exit 1
fi

if (( STATE_INDEX < 0 || STATE_INDEX >= ${#STATES[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK_ID maps to state index $STATE_INDEX, but only ${#STATES[@]} states were found." >&2
  exit 1
fi

STATE="${STATES[$STATE_INDEX]}"
VARIANT="${MODEL_VARIANTS[$MODEL_INDEX]}"

SPRING="$DATA_DIR/s2_spring_${STATE}_${YEAR}_prithvi.tif"
SUMMER="$DATA_DIR/s2_summer_${STATE}_${YEAR}_prithvi.tif"
FALL="$DATA_DIR/s2_fall_${STATE}_${YEAR}_prithvi.tif"

for input in "$SPRING" "$SUMMER" "$FALL"; do
  if [[ ! -s "$input" ]]; then
    echo "ERROR: missing or empty input composite: $input" >&2
    exit 1
  fi
done

# Input staging: fast local SSD for read-heavy inference; wiped at job end
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  STAGE_DIR="$SLURM_TMPDIR/prithvi_${STATE}_${VARIANT}_${YEAR}"
else
  : "${SCRATCH:?Set SCRATCH when SLURM_TMPDIR is unavailable.}"
  STAGE_DIR="$SCRATCH/embeddings-health/staging/${SLURM_JOB_ID:-manual}_${TASK_ID}"
fi

INPUT_STAGE="$STAGE_DIR/input"

# Checkpoint and output dir on $SCRATCH so they survive preemption/timeout
CKPT_DIR="$SCRATCH/embeddings-health/checkpoints/$VARIANT/$STATE"
FINAL_DIR="$FINAL_OUT_DIR/$VARIANT/$STATE"

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

echo "gcc: $(command -v gcc || true)"
gcc --version | head -n 1 || true

if ! command -v uv >/dev/null 2>&1; then
  UV_INSTALL_DIR="${UV_INSTALL_DIR:-$CACHE_ROOT/uv-bin}"
  mkdir -p "$UV_INSTALL_DIR"

  if [[ -x "$UV_INSTALL_DIR/uv" ]]; then
    export PATH="$UV_INSTALL_DIR:$PATH"
  else
    echo "uv is not available on PATH; installing uv into $UV_INSTALL_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
    export PATH="$UV_INSTALL_DIR:$PATH"
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not available after module load/bootstrap. Set UV_INSTALL_DIR or install uv before submitting." >&2
  exit 1
fi

echo "uv: $(command -v uv)"
uv --version

cp "$SPRING" "$INPUT_STAGE/"
cp "$SUMMER" "$INPUT_STAGE/"
cp "$FALL" "$INPUT_STAGE/"

STAGED_SPRING="$INPUT_STAGE/$(basename "$SPRING")"
STAGED_SUMMER="$INPUT_STAGE/$(basename "$SUMMER")"
STAGED_FALL="$INPUT_STAGE/$(basename "$FALL")"

SAFE_VARIANT="${VARIANT//[^A-Za-z0-9._-]/_}"
OUTPUT_BASENAME="prithvi_${SAFE_VARIANT}_${STATE}_${YEAR}"
RAW_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.tif"

if [[ -n "${BATCH_SIZE:-}" ]]; then
  RUN_BATCH_SIZE="$BATCH_SIZE"
elif [[ "$VARIANT" == "tiny" ]]; then
  RUN_BATCH_SIZE="${BATCH_SIZE_TINY:-8}"
else
  RUN_BATCH_SIZE="${BATCH_SIZE_300M_TL:-1}"
fi

TEST_CHIPS_ARGS=""
if [[ -n "${TEST_CHIPS:-}" ]]; then
  TEST_CHIPS_ARGS="--test-chips $TEST_CHIPS"
fi

FORCE_FLAG=""
if [[ -n "${FORCE_RESTART:-}" ]]; then
  FORCE_FLAG="--force"
fi

echo "=== Prithvi embedding task ==="
echo "State: $STATE"
echo "Year: $YEAR"
echo "Variant: $VARIANT"
echo "Task ID: $TASK_ID"
echo "Input dir: $DATA_DIR"
echo "Stage dir: $STAGE_DIR"
echo "Checkpoint dir: $CKPT_DIR"
echo "Final dir: $FINAL_DIR"
echo "Batch size: $RUN_BATCH_SIZE"
echo "Test chips: ${TEST_CHIPS:-none}"
nvidia-smi || true

cd "$EMBED_DIR"

uv run --python 3.11 python embed.py \
  --model prithvi \
  --variant "$VARIANT" \
  --input "$STAGED_SPRING" "$STAGED_SUMMER" "$STAGED_FALL" \
  --output "$RAW_LOCAL" \
  --year "$YEAR" \
  --batch-size "$RUN_BATCH_SIZE" \
  --checkpoint-every "${CHECKPOINT_EVERY:-25}" \
  $FORCE_FLAG \
  $TEST_CHIPS_ARGS

if [[ ! -s "$RAW_LOCAL" ]]; then
  echo "ERROR: expected output missing or empty: $RAW_LOCAL" >&2
  exit 1
fi

mv -f "$RAW_LOCAL" "$FINAL_DIR/"

trap - ERR TERM
cleanup_success

echo "=== Done ==="
echo "Outputs moved to $FINAL_DIR"
