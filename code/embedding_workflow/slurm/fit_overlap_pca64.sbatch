#!/bin/bash
#SBATCH --job-name=overlap-pca64-fit
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/embeddings-health}"
: "${SCRATCH:?SCRATCH is not set}"
: "${MODEL:?MODEL is required}"
: "${TASK_FILE:?TASK_FILE is required}"
: "${PCA_MODEL:?PCA_MODEL is required}"

YEAR="${YEAR:-2022}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/embeddings-health/embedding_workflow_overlap_v1}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH/embeddings-health/cache}"
PY="${PY:-$CACHE_ROOT/venv-3.11-cpu/bin/python}"
PCA_INPUT_DIMS="${PCA_INPUT_DIMS:-768}"
PCA_COMPONENTS="${PCA_COMPONENTS:-64}"

module load devel
module load gcc/14.2.0
export PYTHONUNBUFFERED=1

validate_pca() {
  "$PY" - "$1" "$PCA_INPUT_DIMS" "$PCA_COMPONENTS" <<'PY'
import pickle
import sys
import numpy as np

path, expected_features, expected_components = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, "rb") as handle:
    pca = pickle.load(handle)
assert int(pca.n_features_in_) == expected_features
assert int(pca.n_components_) == expected_components
assert np.isfinite(pca.components_).all()
assert np.isfinite(pca.mean_).all()
print(f"validated PCA: {pca.n_features_in_} -> {pca.n_components_}; "
      f"explained_variance={pca.explained_variance_ratio_.sum():.6f}")
PY
}

if [[ -s "$PCA_MODEL" ]] && validate_pca "$PCA_MODEL"; then
  echo "Validated PCA model already exists: $PCA_MODEL"
  exit 0
fi

temporary="${PCA_MODEL}.tmp.${SLURM_JOB_ID}"
trap 'rm -f "$temporary"' EXIT
"$PY" "$REPO_DIR/code/embedding_workflow/fit_overlap_national_pca.py" \
  --task-file "$TASK_FILE" \
  --output-root "$OUTPUT_ROOT" \
  --model "$MODEL" \
  --year "$YEAR" \
  --output "$temporary" \
  --n-components "$PCA_COMPONENTS" \
  --total-samples "${TOTAL_SAMPLES:-500000}" \
  --min-samples-per-state "${MIN_SAMPLES_PER_STATE:-500}" \
  --max-rasters-per-state "${MAX_RASTERS_PER_STATE:-16}" \
  --block-size "${PCA_BLOCK_SIZE:-256}"
validate_pca "$temporary"
mkdir -p "$(dirname "$PCA_MODEL")"
mv "$temporary" "$PCA_MODEL"
trap - EXIT
echo "PCA model saved: $PCA_MODEL"
