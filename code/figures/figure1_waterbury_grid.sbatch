#!/bin/bash
#SBATCH --job-name=fig1-waterbury
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/nhendrix/embeddings-health/code/figures/logs/fig1_waterbury_%j.out
#SBATCH --error=/home/users/nhendrix/embeddings-health/code/figures/logs/fig1_waterbury_%j.err

set -euo pipefail

PYTHON="$SCRATCH/embeddings-health/cache/venv-3.11-cpu/bin/python"
FIGURES_DIR="$HOME/embeddings-health/code/figures"
SRC_BASE="$SCRATCH/embeddings-health"

echo "Node: $(hostname)"
echo "Start: $(date)"

# Stage all embedding TIFs to L_SCRATCH (local SSD) for fast windowed reads.
# Band-sequential COGs require ~3000 scattered seeks on a 44 GB Lustre file;
# L_SCRATCH turns those into sequential reads on local NVMe.
STAGE="$L_SCRATCH/tifs"
mkdir -p \
  "$STAGE/olmoearth_nano_embeddings/CT" \
  "$STAGE/olmoearth_embeddings/CT" \
  "$STAGE/clay_embeddings/CT" \
  "$STAGE/prithvi_embeddings/tiny/CT" \
  "$STAGE/prithvi_embeddings/300M-TL/CT"

echo "Staging TIFs to L_SCRATCH ($STAGE) ..."

cp "$SRC_BASE/olmoearth_nano_embeddings/CT/olmoearth_Nano_CT_2022.tif" \
   "$STAGE/olmoearth_nano_embeddings/CT/" &
cp "$SRC_BASE/olmoearth_embeddings/CT/olmoearth_v1_1-Base_CT_2022.tif" \
   "$STAGE/olmoearth_embeddings/CT/" &
cp "$SRC_BASE/clay_embeddings/CT/clay_v1.5_CT_2022.tif" \
   "$STAGE/clay_embeddings/CT/" &
cp "$SRC_BASE/prithvi_embeddings/tiny/CT/prithvi_tiny_CT_2022_raw.tif" \
   "$STAGE/prithvi_embeddings/tiny/CT/" &
cp "$SRC_BASE/prithvi_embeddings/300M-TL/CT/prithvi_300M-TL_CT_2022_raw.tif" \
   "$STAGE/prithvi_embeddings/300M-TL/CT/" &

wait
echo "Staging done: $(date)"
df -h "$L_SCRATCH"

export TIF_BASE="$STAGE"
mkdir -p "$SRC_BASE/figures"

"$PYTHON" -u "$FIGURES_DIR/figure1_waterbury_grid.py"

echo "Done: $(date)"
