#!/bin/bash
#SBATCH --job-name=embed-aggregate
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?SCRATCH is not set}"; : "${MODEL:?MODEL is required}"; : "${STATE:?STATE is required}"
YEAR="${YEAR:-2022}"; TRACT_YEAR="${TRACT_YEAR:-2020}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
declare -A FIPS=([AL]=01 [AZ]=04 [AR]=05 [CA]=06 [CO]=08 [CT]=09 [DE]=10 [DC]=11 [FL]=12 [GA]=13 [ID]=16 [IL]=17 [IN]=18 [IA]=19 [KS]=20 [KY]=21 [LA]=22 [ME]=23 [MD]=24 [MA]=25 [MI]=26 [MN]=27 [MS]=28 [MO]=29 [MT]=30 [NE]=31 [NV]=32 [NH]=33 [NJ]=34 [NM]=35 [NY]=36 [NC]=37 [ND]=38 [OH]=39 [OK]=40 [OR]=41 [PA]=42 [RI]=44 [SC]=45 [SD]=46 [TN]=47 [TX]=48 [UT]=49 [VT]=50 [VA]=51 [WA]=53 [WV]=54 [WI]=55 [WY]=56)
EMBEDDING="$OUTPUT_ROOT/$MODEL/$STATE/${MODEL}_${STATE}_${YEAR}.tif"
TRACTS="${TRACTS:-$SCRATCH/embeddings-health/data/census_tracts/tl_${TRACT_YEAR}_${FIPS[$STATE]}_tract.zip}"
CSV="$OUTPUT_ROOT/$MODEL/$STATE/${MODEL}_${STATE}_${YEAR}_tracts.csv"
family=olmoearth; [[ "$MODEL" == clay-* ]] && family=clay
module load devel; module load gcc/14.2.0
export UV_CACHE_DIR="$CACHE_ROOT/uv" UV_PROJECT_ENVIRONMENT="$CACHE_ROOT/venv-3.11-cpu"
cd "$REPO_DIR/code/embedding_generation"; uv sync --python 3.11
uv run --python 3.11 python ../embedding_workflow/aggregate.py --embedding "$EMBEDDING" --tracts "$TRACTS" --output "$CSV" --year "$YEAR" --model "$family"
