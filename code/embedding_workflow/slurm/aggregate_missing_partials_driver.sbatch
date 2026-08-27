#!/bin/bash
#SBATCH --job-name=tract-partial-driver
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?SCRATCH is not set}"
: "${MODEL:?MODEL is required}"
: "${TASK_FILE:?TASK_FILE is required}"

YEAR="${YEAR:-2022}"
TRACT_YEAR="${TRACT_YEAR:-2020}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
BAND_CHUNK="${BAND_CHUNK:-64}"

fips_for_state() {
  case "$1" in
    AL) echo 01 ;; AZ) echo 04 ;; AR) echo 05 ;; CA) echo 06 ;; CO) echo 08 ;;
    CT) echo 09 ;; DE) echo 10 ;; DC) echo 11 ;; FL) echo 12 ;; GA) echo 13 ;;
    ID) echo 16 ;; IL) echo 17 ;; IN) echo 18 ;; IA) echo 19 ;; KS) echo 20 ;;
    KY) echo 21 ;; LA) echo 22 ;; ME) echo 23 ;; MD) echo 24 ;; MA) echo 25 ;;
    MI) echo 26 ;; MN) echo 27 ;; MS) echo 28 ;; MO) echo 29 ;; MT) echo 30 ;;
    NE) echo 31 ;; NV) echo 32 ;; NH) echo 33 ;; NJ) echo 34 ;; NM) echo 35 ;;
    NY) echo 36 ;; NC) echo 37 ;; ND) echo 38 ;; OH) echo 39 ;; OK) echo 40 ;;
    OR) echo 41 ;; PA) echo 42 ;; RI) echo 44 ;; SC) echo 45 ;; SD) echo 46 ;;
    TN) echo 47 ;; TX) echo 48 ;; UT) echo 49 ;; VT) echo 50 ;; VA) echo 51 ;;
    WA) echo 53 ;; WV) echo 54 ;; WI) echo 55 ;; WY) echo 56 ;;
    *) return 1 ;;
  esac
}

process_task() {
  local state="$1" index="$2" num_tiles="$3"
  local stem="${MODEL}_overlap-center50_${state}_${YEAR}"
  local partial_stem="${stem}${AGG_SUFFIX:-}"
  local dir="$OUTPUT_ROOT/$MODEL/$state"
  local tile partial fips tracts

  printf -v tile '%s/%s_tile%03d.tif' "$dir" "$stem" "$index"
  if (( num_tiles == 1 )); then
    tile="$dir/$stem.tif"
  fi
  printf -v partial '%s/partials/%s/tile%03d.npz' "$dir" "$partial_stem" "$index"

  if [[ -s "$partial" ]]; then
    echo "skip existing partial: $partial"
    return 0
  fi
  [[ -s "$tile" ]] || { echo "missing tile: $tile" >&2; return 1; }
  if [[ ! -s "${tile%.tif}.validation.json" ]]; then
    if [[ "${ALLOW_MISSING_VALIDATION:-0}" == 1 ]]; then
      echo "warning: missing validation accepted by override: ${tile%.tif}.validation.json" >&2
    else
      echo "missing validation: ${tile%.tif}.validation.json" >&2
      return 1
    fi
  fi

  fips="$(fips_for_state "$state")" || { echo "unknown state FIPS: $state" >&2; return 1; }
  tracts="${TRACTS:-$SCRATCH/embeddings-health/data/census_tracts/tl_${TRACT_YEAR}_${fips}_tract.zip}"
  echo "aggregate partial: $state tile $index/$((num_tiles - 1)) -> $partial_stem"
  local command=(
    "$UV_PROJECT_ENVIRONMENT/bin/python"
    ../embedding_workflow/aggregate_tiles.py partial
    --tile "$tile"
    --tracts "$tracts"
    --output "$partial"
    --band-chunk "$BAND_CHUNK"
  )
  if [[ -n "${PCA_MODEL:-}" ]]; then
    [[ -s "$PCA_MODEL" ]] || { echo "ERROR: PCA_MODEL not found: $PCA_MODEL" >&2; return 1; }
    command+=(--pca-model "$PCA_MODEL")
  fi
  "${command[@]}"
}

worker_loop() {
  local worker="$1" line=0 state index num_tiles rows cols
  while read -r state index num_tiles rows cols; do
    (( line += 1 ))
    (( (line - 1) % WORKERS == worker )) || continue
    process_task "$state" "$index" "$num_tiles"
  done < "$TASK_FILE"
}

module load devel
module load gcc/14.2.0
export UV_CACHE_DIR="$CACHE_ROOT/uv" UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-cpu" GDAL_CACHEMAX="${GDAL_CACHEMAX:-512}"

cd "$REPO_DIR/code/embedding_generation"
uv sync --python 3.11

echo "task file: $TASK_FILE"
echo "workers: $WORKERS"
for ((worker=0; worker<WORKERS; worker++)); do
  worker_loop "$worker" &
done

status=0
for job in $(jobs -p); do
  wait "$job" || status=1
done
exit "$status"
