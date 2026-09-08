#!/bin/bash
#SBATCH --job-name=prithvi-300m-dispatch
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH --time=00:30:00

set -euo pipefail

: "${SCRATCH:?SCRATCH must be set}"
: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${PREPARED_DIR:?PREPARED_DIR must be set}"
: "${OUTPUTS_DIR:?OUTPUTS_DIR must be set}"

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
ROOT="$SCRATCH/embeddings-health"
PYTHON="${PYTHON:-$ROOT/cache/venv-3.11-cpu/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
PLAN="$RUN_ROOT/plan.json"
WORKER="$REPO_DIR/code/analyses/slurm/run_prithvi_300m_full_shard.sbatch"
FINALIZER="$REPO_DIR/code/analyses/slurm/run_prithvi_300m_full_finalize.sbatch"

[[ -s "$PLAN" ]] || { echo "ERROR: missing plan: $PLAN" >&2; exit 1; }
mkdir -p "$LOG_DIR"

q1_tasks=$(jq -r '.tasks.q1' "$PLAN")
q23_tasks=$(jq -r '.tasks.q23' "$PLAN")
q4_state_tasks=$(jq -r '.tasks.q4_state' "$PLAN")
q4_decile_tasks=$(jq -r '.tasks.q4_decile' "$PLAN")
for count in "$q1_tasks" "$q23_tasks" "$q4_state_tasks" "$q4_decile_tasks"; do
  (( count > 0 )) || { echo "ERROR: plan contains an empty stage" >&2; exit 1; }
done

common_export="ALL,RUN_ROOT=$RUN_ROOT,PREPARED_DIR=$PREPARED_DIR,OUTPUTS_DIR=$OUTPUTS_DIR,PYTHON=$PYTHON,REPO_DIR=$REPO_DIR"
run_id=$(basename "$RUN_ROOT")

q1_job=$(sbatch --parsable \
  --array="0-$((q1_tasks - 1))%${Q1_CONCURRENCY:-6}" \
  --cpus-per-task=8 --mem=32G --time=03:00:00 \
  --output="$LOG_DIR/prithvi_300m_q1_${run_id}_%A_%a.out" \
  --error="$LOG_DIR/prithvi_300m_q1_${run_id}_%A_%a.err" \
  --export="$common_export,STAGE=q1" "$WORKER" | cut -d';' -f1)

q23_job=$(sbatch --parsable \
  --array="0-$((q23_tasks - 1))%${Q23_CONCURRENCY:-4}" \
  --cpus-per-task=8 --mem=32G --time=03:00:00 \
  --output="$LOG_DIR/prithvi_300m_q23_${run_id}_%A_%a.out" \
  --error="$LOG_DIR/prithvi_300m_q23_${run_id}_%A_%a.err" \
  --export="$common_export,STAGE=q23" "$WORKER" | cut -d';' -f1)

q4_state_job=$(sbatch --parsable \
  --array="0-$((q4_state_tasks - 1))%${Q4_STATE_CONCURRENCY:-12}" \
  --cpus-per-task=4 --mem=16G --time=02:00:00 \
  --output="$LOG_DIR/prithvi_300m_q4state_${run_id}_%A_%a.out" \
  --error="$LOG_DIR/prithvi_300m_q4state_${run_id}_%A_%a.err" \
  --export="$common_export,STAGE=q4_state" "$WORKER" | cut -d';' -f1)

q4_decile_job=$(sbatch --parsable \
  --array="0-$((q4_decile_tasks - 1))%${Q4_DECILE_CONCURRENCY:-4}" \
  --cpus-per-task=4 --mem=16G --time=02:00:00 \
  --output="$LOG_DIR/prithvi_300m_q4decile_${run_id}_%A_%a.out" \
  --error="$LOG_DIR/prithvi_300m_q4decile_${run_id}_%A_%a.err" \
  --export="$common_export,STAGE=q4_decile" "$WORKER" | cut -d';' -f1)

final_job=$(sbatch --parsable \
  --dependency="afterany:$q1_job:$q23_job:$q4_state_job:$q4_decile_job" \
  --output="$LOG_DIR/prithvi_300m_final_${run_id}_%j.out" \
  --error="$LOG_DIR/prithvi_300m_final_${run_id}_%j.err" \
  --export="$common_export" "$FINALIZER" | cut -d';' -f1)

{
  echo "Q1_JOB=$q1_job"
  echo "Q23_JOB=$q23_job"
  echo "Q4_STATE_JOB=$q4_state_job"
  echo "Q4_DECILE_JOB=$q4_decile_job"
  echo "FINAL_JOB=$final_job"
} >> "$RUN_ROOT/jobs.txt"

echo "Q1 array:       $q1_job ($q1_tasks tasks)"
echo "Q2/Q3 array:    $q23_job ($q23_tasks tasks)"
echo "Q4 state array: $q4_state_job ($q4_state_tasks tasks)"
echo "Q4 decile array:$q4_decile_job ($q4_decile_tasks tasks)"
echo "Finalizer:      $final_job"
