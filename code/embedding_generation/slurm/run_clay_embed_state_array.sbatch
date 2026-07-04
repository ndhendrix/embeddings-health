#!/bin/bash
#SBATCH --job-name=clay-embed
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=code/embedding_generation/slurm/logs/clay_embed_%A_%a.out
#SBATCH --error=code/embedding_generation/slurm/logs/clay_embed_%A_%a.err

# Clay v1.5 embedding job.
#
# Uses the same 12-band OlmoEarth composites as the OlmoEarth Base/Nano jobs;
# embed.py selects and reorders 10 bands internally.
#
# Clay uses 256×256 chips (vs 128×128 for OlmoEarth) so chip count is ~4×
# lower, but the model is ViT-Large (24 layers, 1024-dim) and the output is
# also 1024-dimensional — large states will use the disk-backed memmap path.
#
# No extra package installs needed: clay_encoder.py is bundled in the repo
# and only depends on torch + einops (already in the virtualenv).
# The Clay checkpoint (~5 GB) is downloaded from HuggingFace on first run
# and cached in $HF_HOME.
#
# STATE_LIST must be set by the submit script (colon-separated state abbrevs).

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
EMBED_DIR="$REPO_DIR/code/embedding_generation"
YEAR="${YEAR:-2022}"

: "${SCRATCH:?SCRATCH is not set}"
COMPOSITE_DIR="${COMPOSITE_DIR:-$SCRATCH/embeddings-health/olmoearth_composites}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$SCRATCH/embeddings-health/clay_embeddings}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"

: "${STATE_LIST:?STATE_LIST must be set to a colon-separated ordered state list}"
IFS=: read -ra STATES <<< "$STATE_LIST"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID < 0 || TASK_ID >= ${#STATES[@]} )); then
  echo "ERROR: TASK_ID=$TASK_ID out of range (${#STATES[@]} states)" >&2
  exit 1
fi

STATE="${STATES[$TASK_ID]}"
# Clay reuses the OlmoEarth 12-band composites; 10 bands are selected in embed.py.
COMPOSITE="$COMPOSITE_DIR/s2_annual_${STATE}_${YEAR}_olmoearth.tif"

if [[ ! -s "$COMPOSITE" ]]; then
  echo "ERROR: composite not found or empty: $COMPOSITE" >&2
  exit 1
fi

# Staging: prefer node-local SSD (auto-wiped at job end regardless of how the
# job dies) when the composite fits; Sherlock's node-local mount is
# $L_SCRATCH, not $SLURM_TMPDIR (which Sherlock never sets). Fall back to
# $SCRATCH for composites too big for local disk.
STAGE_ON_LOCAL=0
if [[ -n "${L_SCRATCH:-}" ]]; then
  COMPOSITE_BYTES=$(stat -c%s "$COMPOSITE")
  AVAIL_BYTES=$(df --output=avail -B1 "$L_SCRATCH" 2>/dev/null | tail -1)
  if [[ -n "$AVAIL_BYTES" ]] && (( COMPOSITE_BYTES < AVAIL_BYTES * 8 / 10 )); then
    STAGE_ON_LOCAL=1
  fi
fi

if [[ "$STAGE_ON_LOCAL" == 1 ]]; then
  STAGE_DIR="$L_SCRATCH/clay_${STATE}_${YEAR}"
else
  STAGE_DIR="$SCRATCH/embeddings-health/staging/${SLURM_JOB_ID:-manual}_${TASK_ID}"
fi

INPUT_STAGE="$STAGE_DIR/input"
CKPT_DIR="$SCRATCH/embeddings-health/checkpoints/clay/$STATE"
FINAL_DIR="$FINAL_OUT_DIR/$STATE"

PREEMPTED=0

cleanup_stage() {
  if [[ -n "${L_SCRATCH:-}" && "$STAGE_DIR" == "$L_SCRATCH/"* ]] || \
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

CHILD_PID=""

# Slurm's --signal=B:TERM@600 delivers this 10 minutes before the hard time
# limit. Forward it to the embed.py child and give it up to 30s to exit
# gracefully (it may write a final checkpoint) before force-killing, so the
# remaining buffer is available for cleanup_stage() instead of being eaten by
# a GPU process that doesn't respond promptly to SIGTERM. Without this, Slurm's
# eventual SIGKILL hits the whole process group at once and no trap ever runs.
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

# Clay needs no special extra installs: clay_encoder.py is bundled in the repo
# and only requires torch + einops, both of which are in pyproject.toml.
# The 5 GB Clay checkpoint is cached in $HF_HOME on first run.
(cd "$EMBED_DIR" && uv sync --python 3.11 2>&1 | tail -5)

cp "$COMPOSITE" "$INPUT_STAGE/"
STAGED_COMPOSITE="$INPUT_STAGE/$(basename "$COMPOSITE")"

OUTPUT_BASENAME="clay_v1.5_${STATE}_${YEAR}"
RAW_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.tif"

# Clay 256-px chips → smaller batch sizes than OlmoEarth 128-px chips.
BATCH_SIZE="${BATCH_SIZE:-8}"

TEST_CHIPS_ARGS=""
if [[ -n "${TEST_CHIPS:-}" ]]; then
  TEST_CHIPS_ARGS="--test-chips $TEST_CHIPS"
fi

FORCE_FLAG=""
if [[ -n "${FORCE_RESTART:-}" ]]; then
  FORCE_FLAG="--force"
fi

echo "=== Clay v1.5 embedding task ==="
echo "State:      $STATE"
echo "Year:       $YEAR"
echo "Task ID:    $TASK_ID"
echo "Composite:  $COMPOSITE"
echo "Ckpt dir:   $CKPT_DIR"
echo "Final dir:  $FINAL_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Test chips: ${TEST_CHIPS:-none}"
nvidia-smi || true

cd "$EMBED_DIR"

# Run in background and wait: bash's `wait` builtin returns as soon as a
# trapped signal arrives, whereas a foreground command can block trap
# execution until the child exits on its own. This is what lets
# handle_preempt() run promptly on the --signal=B:TERM@600 warning.
uv run --python 3.11 python embed.py \
  --model        clay \
  --input        "$STAGED_COMPOSITE" \
  --output       "$RAW_LOCAL" \
  --year         "$YEAR" \
  --batch-size   "$BATCH_SIZE" \
  --checkpoint-every "${CHECKPOINT_EVERY:-25}" \
  $FORCE_FLAG \
  $TEST_CHIPS_ARGS &
CHILD_PID=$!
wait "$CHILD_PID"

if [[ ! -s "$RAW_LOCAL" ]]; then
  echo "ERROR: expected output missing or empty: $RAW_LOCAL" >&2
  # exit (rather than a failing command) does not trigger the ERR trap, so
  # clean up the staging copy explicitly here instead of relying on preserve_failure.
  cleanup_stage
  exit 1
fi

mv -f "$RAW_LOCAL" "$FINAL_DIR/"
MMAP_LOCAL="$CKPT_DIR/${OUTPUT_BASENAME}.ckpt.mmap"
[[ -f "$MMAP_LOCAL" ]] && rm -f "$MMAP_LOCAL" || true

trap - ERR TERM
cleanup_success

echo "=== Done ==="
echo "Output: $FINAL_DIR/${OUTPUT_BASENAME}.tif"
