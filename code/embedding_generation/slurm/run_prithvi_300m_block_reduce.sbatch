#!/bin/bash
#SBATCH --job-name=prithvi-300m-reduce
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${MANIFEST:?MANIFEST must point to the block-task TSV}"
: "${STATE_FILE:?STATE_FILE must list states to reduce}"
: "${PARTIAL_ROOT:?PARTIAL_ROOT is required}"
: "${AGG_OUT_DIR:?AGG_OUT_DIR is required}"
: "${TRACT_DIR:?TRACT_DIR is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

PYTHON="${PYTHON:-$SCRATCH/embeddings-health/cache/venv-3.11-cpu/bin/python}"
YEAR="${YEAR:-2022}"
TRACT_YEAR="${TRACT_YEAR:-2020}"

STATE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$STATE_FILE")
[[ -n "$STATE" ]] || {
  echo "ERROR: no state for reducer task $SLURM_ARRAY_TASK_ID" >&2
  exit 1
}

declare -A STATE_FIPS=(
  [AL]=01 [AZ]=04 [AR]=05 [CA]=06 [CO]=08 [CT]=09 [DE]=10 [DC]=11
  [FL]=12 [GA]=13 [ID]=16 [IL]=17 [IN]=18 [IA]=19 [KS]=20 [KY]=21
  [LA]=22 [ME]=23 [MD]=24 [MA]=25 [MI]=26 [MN]=27 [MS]=28 [MO]=29
  [MT]=30 [NE]=31 [NV]=32 [NH]=33 [NJ]=34 [NM]=35 [NY]=36 [NC]=37
  [ND]=38 [OH]=39 [OK]=40 [OR]=41 [PA]=42 [RI]=44 [SC]=45 [SD]=46
  [TN]=47 [TX]=48 [UT]=49 [VT]=50 [VA]=51 [WA]=53 [WV]=54 [WI]=55
  [WY]=56
)

EXPECTED=$(awk -F $'\t' -v state="$STATE" '$1 == state {count++} END {print count+0}' "$MANIFEST")
shopt -s nullglob
PARTIALS=("$PARTIAL_ROOT/$STATE"/block_r*_c*.npz)
shopt -u nullglob
(( ${#PARTIALS[@]} == EXPECTED )) || {
  echo "ERROR: $STATE expected $EXPECTED partials, found ${#PARTIALS[@]}" >&2
  exit 1
}

TRACTS="$TRACT_DIR/tl_${TRACT_YEAR}_${STATE_FIPS[$STATE]}_tract.zip"
OUTPUT="$AGG_OUT_DIR/prithvi_300M-TL_${STATE}_${YEAR}_tracts.csv"
QA_JSON="$AGG_OUT_DIR/prithvi_300M-TL_${STATE}_${YEAR}_tracts.validation.json"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$PYTHON" "$REPO_DIR/code/embedding_workflow/aggregate_tiles.py" reduce \
  --partials "${PARTIALS[@]}" \
  --output "$OUTPUT" \
  --year "$YEAR" \
  --prefix PR \
  --tracts "$TRACTS" \
  --qa-json "$QA_JSON"

jq -e '
  .embedding_dimensions == 1024 and
  .unexpected_tracts == 0 and
  (.duplicate_expected_geoids | length) == 0 and
  .expected_tracts == (.tracts_with_valid_pixels + .zero_valid_pixel_tracts)
' "$QA_JSON" >/dev/null

columns=$(awk -F',' 'NR == 1 {print NF; exit}' "$OUTPUT")
(( columns == 4099 )) || {
  echo "ERROR: $OUTPUT has $columns columns; expected 4099" >&2
  exit 1
}

echo "Complete: $STATE ($EXPECTED blocks, $columns columns)"
