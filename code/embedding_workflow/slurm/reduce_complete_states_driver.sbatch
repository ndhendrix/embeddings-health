#!/bin/bash
#SBATCH --job-name=tract-reduce-driver
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?SCRATCH is not set}"
: "${MODEL:?MODEL is required}"
: "${STATE_FILE:?STATE_FILE is required}"

YEAR="${YEAR:-2022}"
TRACT_YEAR="${TRACT_YEAR:-2020}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"

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

module load devel
module load gcc/14.2.0
export UV_CACHE_DIR="$CACHE_ROOT/uv" UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-cpu"
prefix="${FEATURE_PREFIX:-OE}"
if [[ -z "${FEATURE_PREFIX:-}" && "$MODEL" == clay-* ]]; then prefix=CL; fi

cd "$REPO_DIR/code/embedding_generation"
uv sync --python 3.11

while read -r state num_tiles rows cols; do
  stem="${MODEL}_overlap-center50_${state}_${YEAR}"
  agg_stem="${stem}${AGG_SUFFIX:-}"
  dir="$OUTPUT_ROOT/$MODEL/$state"
  if [[ -n "${AGG_OUT_DIR:-}" ]]; then
    mkdir -p "$AGG_OUT_DIR"
    csv="$AGG_OUT_DIR/${agg_stem}_tracts.csv"
  else
    csv="$dir/${agg_stem}_tracts.csv"
  fi
  qa_json="${csv%.csv}.validation.json"
  if [[ -s "$csv" && -s "$qa_json" ]]; then
    echo "skip existing reduced output: $csv"
    continue
  fi

  partials=()
  for ((i=0; i<num_tiles; i++)); do
    printf -v partial '%s/partials/%s/tile%03d.npz' "$dir" "$agg_stem" "$i"
    [[ -s "$partial" ]] || { echo "missing partial: $partial" >&2; exit 1; }
    partials+=("$partial")
  done

  fips="$(fips_for_state "$state")" || { echo "unknown state FIPS: $state" >&2; exit 1; }
  tracts="${TRACTS:-$SCRATCH/embeddings-health/data/census_tracts/tl_${TRACT_YEAR}_${fips}_tract.zip}"
  echo "reduce state: $state partials=$num_tiles"
  "$UV_PROJECT_ENVIRONMENT/bin/python" \
    ../embedding_workflow/aggregate_tiles.py reduce \
    --partials "${partials[@]}" \
    --output "$csv" \
    --year "$YEAR" \
    --prefix "$prefix" \
    --tracts "$tracts" \
    --qa-json "$qa_json"
done < "$STATE_FILE"
