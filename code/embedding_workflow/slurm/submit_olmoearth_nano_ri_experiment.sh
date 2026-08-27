#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"; : "${SCRATCH:?}"
MODEL=olmoearth-v1.2-nano; STATE=RI; YEAR="${YEAR:-2022}"
INPUT="${INPUT:-$SCRATCH/embeddings-health/olmoearth_composites/s2_annual_RI_${YEAR}_olmoearth.tif}"
BASE="${OE_NANO_RI_ROOT:-$SCRATCH/embeddings-health/olmoearth_nano_ri_experiment_v2}"
REF_ROOT="$BASE/reference"; RECT_ROOT="$BASE/rect2x2"; STEM="${MODEL}_overlap-center50_RI_${YEAR}"
REF="$REF_ROOT/$MODEL/RI/$STEM.tif"; RECT="$RECT_ROOT/$MODEL/RI/$STEM.tif"
COMPARE="$BASE/reference_vs_rect2x2.json"; REF_SEAMS="$BASE/reference_seams.json"; RECT_SEAMS="$BASE/rect2x2_seams.json"
REF_CSV="$REF_ROOT/$MODEL/RI/${STEM}_tracts.csv"; RECT_CSV="$RECT_ROOT/$MODEL/RI/${STEM}_tracts.csv"
echo "OlmoEarth Nano RI center-crop experiment (not submitted by default)"
echo "reference: $REF"; echo "rectangular merge: $RECT"; echo "comparison: $COMPARE"; echo "seam diagnostics: $REF_SEAMS $RECT_SEAMS"
if [[ "${CONFIRM_OE_NANO_RI_SUBMIT:-0}" != 1 ]]; then
  echo "DRY RUN. Set CONFIRM_OE_NANO_RI_SUBMIT=1 to submit."
  echo "Jobs: 1 reference GPU + 4 rectangular GPU tasks + merge + tract partial/reduce + equivalence, seam, and PCA RGB diagnostics"
  exit 0
fi
mkdir -p "$SCRIPT_DIR/logs"; export REPO_DIR MODEL STATE YEAR INPUT PRODUCT_MODE=overlap
REF_JOB=$(OVERLAP_OUTPUT_ROOT="$REF_ROOT" NUM_TILES=1 sbatch --parsable --export=ALL --output="$SCRIPT_DIR/logs/oe_nano_RI_reference_%j.out" --error="$SCRIPT_DIR/logs/oe_nano_RI_reference_%j.err" "$SCRIPT_DIR/run_overlap_tile.sbatch"); REF_JOB="${REF_JOB%%;*}"
RECT_JOB=$(OVERLAP_OUTPUT_ROOT="$RECT_ROOT" NUM_TILES=4 GRID_ROWS=2 GRID_COLS=2 sbatch --parsable --export=ALL --array="0-3" --output="$SCRIPT_DIR/logs/oe_nano_RI_rect_%A_%a.out" --error="$SCRIPT_DIR/logs/oe_nano_RI_rect_%A_%a.err" "$SCRIPT_DIR/run_overlap_tile.sbatch"); RECT_JOB="${RECT_JOB%%;*}"
tiles=("${RECT%.tif}_tile000.tif" "${RECT%.tif}_tile001.tif" "${RECT%.tif}_tile002.tif" "${RECT%.tif}_tile003.tif")
MERGE=$(sbatch --parsable --export=ALL --dependency="afterok:$RECT_JOB" --output="$SCRIPT_DIR/logs/oe_nano_RI_rect_merge_%j.out" --error="$SCRIPT_DIR/logs/oe_nano_RI_rect_merge_%j.err" --wrap="module load devel; module load gcc/14.2.0; export UV_PROJECT_ENVIRONMENT='$SCRATCH/embeddings-health/cache/venv-3.11-cpu'; cd '$REPO_DIR/code/embedding_generation'; uv run --python 3.11 python ../embedding_workflow/merge.py --tiles ${tiles[*]} --output '$RECT'"); MERGE="${MERGE%%;*}"
REF_PART=$(OUTPUT_ROOT="$REF_ROOT" NUM_TILES=1 SLURM_ARRAY_TASK_ID=0 sbatch --parsable --export=ALL --dependency="afterok:$REF_JOB" "$SCRIPT_DIR/aggregate_tile.sbatch"); REF_PART="${REF_PART%%;*}"
RECT_PART=$(OUTPUT_ROOT="$RECT_ROOT" NUM_TILES=4 sbatch --parsable --export=ALL --dependency="afterok:$RECT_JOB" --array="0-3" "$SCRIPT_DIR/aggregate_tile.sbatch"); RECT_PART="${RECT_PART%%;*}"
REF_REDUCE=$(OUTPUT_ROOT="$REF_ROOT" PRODUCT_STEM="$STEM" NUM_TILES=1 sbatch --parsable --export=ALL --dependency="afterok:$REF_PART" "$SCRIPT_DIR/reduce_tracts.sbatch"); REF_REDUCE="${REF_REDUCE%%;*}"
RECT_REDUCE=$(OUTPUT_ROOT="$RECT_ROOT" PRODUCT_STEM="$STEM" NUM_TILES=4 sbatch --parsable --export=ALL --dependency="afterok:$RECT_PART" "$SCRIPT_DIR/reduce_tracts.sbatch"); RECT_REDUCE="${RECT_REDUCE%%;*}"
COMPARE_JOB=$(REFERENCE="$REF" CANDIDATE="$RECT" REFERENCE_CSV="$REF_CSV" CANDIDATE_CSV="$RECT_CSV" COMPARE_JSON="$COMPARE" sbatch --parsable --export=ALL --dependency="afterok:$MERGE:$REF_REDUCE:$RECT_REDUCE" "$SCRIPT_DIR/compare_ri_rect.sbatch"); COMPARE_JOB="${COMPARE_JOB%%;*}"
REF_SEAM_JOB=$(RASTER="$REF" SEAM_JSON="$REF_SEAMS" sbatch --parsable --export=ALL --dependency="afterok:$REF_JOB" "$SCRIPT_DIR/analyze_overlap_seams.sbatch"); REF_SEAM_JOB="${REF_SEAM_JOB%%;*}"
RECT_SEAM_JOB=$(RASTER="$RECT" SEAM_JSON="$RECT_SEAMS" sbatch --parsable --export=ALL --dependency="afterok:$MERGE" "$SCRIPT_DIR/analyze_overlap_seams.sbatch"); RECT_SEAM_JOB="${RECT_SEAM_JOB%%;*}"
VISUAL_JOB=$(OE_NANO_RI_ROOT="$BASE" sbatch --parsable --export=ALL --dependency="afterok:$REF_JOB:$MERGE" "$SCRIPT_DIR/render_oe_nano_ri_experiment.sbatch"); VISUAL_JOB="${VISUAL_JOB%%;*}"
echo "Submitted: reference=$REF_JOB rect=$RECT_JOB merge=$MERGE ref_part=$REF_PART rect_part=$RECT_PART ref_reduce=$REF_REDUCE rect_reduce=$RECT_REDUCE compare=$COMPARE_JOB ref_seams=$REF_SEAM_JOB rect_seams=$RECT_SEAM_JOB visual=$VISUAL_JOB"
