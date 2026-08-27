#!/bin/bash
#SBATCH --job-name=fig1-dc
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/nhendrix/embeddings-health/code/figures/logs/fig1_dc_%j.out
#SBATCH --error=/home/users/nhendrix/embeddings-health/code/figures/logs/fig1_dc_%j.err

set -euo pipefail

PYTHON="$SCRATCH/embeddings-health/cache/venv-3.11-cpu/bin/python"
FIGURES_DIR="$HOME/embeddings-health/code/figures"
SRC="$SCRATCH/embeddings-health"

echo "Node: $(hostname)"
echo "Start: $(date)"

# Stage smaller files to L_SCRATCH (local NVMe) for fast random-seek I/O.
# Clay DC/MD/VA are too large (338M + 60G + 159G) to stage; read from Lustre.
STAGE="$L_SCRATCH/tifs"
mkdir -p \
  "$STAGE/olmoearth_nano_embeddings/DC" \
  "$STAGE/olmoearth_nano_embeddings/MD" \
  "$STAGE/olmoearth_nano_embeddings/VA" \
  "$STAGE/prithvi_embeddings/tiny/DC" \
  "$STAGE/prithvi_embeddings/tiny/MD" \
  "$STAGE/prithvi_embeddings/tiny/VA" \
  "$STAGE/prithvi_embeddings/300M-TL/DC" \
  "$STAGE/prithvi_embeddings/300M-TL/MD" \
  "$STAGE/prithvi_embeddings/300M-TL/VA"

echo "Staging TIFs to L_SCRATCH ($STAGE) ..."

cp "$SRC/olmoearth_nano_embeddings/DC/olmoearth_Nano_DC_2022.tif" \
   "$STAGE/olmoearth_nano_embeddings/DC/" &
cp "$SRC/olmoearth_nano_embeddings/MD/olmoearth_Nano_MD_2022.tif" \
   "$STAGE/olmoearth_nano_embeddings/MD/" &
cp "$SRC/olmoearth_nano_embeddings/VA/olmoearth_Nano_VA_2022.tif" \
   "$STAGE/olmoearth_nano_embeddings/VA/" &

cp "$SRC/prithvi_embeddings/tiny/DC/prithvi_tiny_DC_2022_raw.tif" \
   "$STAGE/prithvi_embeddings/tiny/DC/" &
cp "$SRC/prithvi_embeddings/tiny/MD/prithvi_tiny_MD_2022_raw.tif" \
   "$STAGE/prithvi_embeddings/tiny/MD/" &
cp "$SRC/prithvi_embeddings/tiny/VA/prithvi_tiny_VA_2022_raw.tif" \
   "$STAGE/prithvi_embeddings/tiny/VA/" &

cp "$SRC/prithvi_embeddings/300M-TL/DC/prithvi_300M-TL_DC_2022_raw.tif" \
   "$STAGE/prithvi_embeddings/300M-TL/DC/" &
cp "$SRC/prithvi_embeddings/300M-TL/MD/prithvi_300M-TL_MD_2022_raw.tif" \
   "$STAGE/prithvi_embeddings/300M-TL/MD/" &
cp "$SRC/prithvi_embeddings/300M-TL/VA/prithvi_300M-TL_VA_2022.tif" \
   "$STAGE/prithvi_embeddings/300M-TL/VA/" &

wait
echo "Staging done: $(date)"
df -h "$L_SCRATCH"

export STAGE_DIR="$STAGE"
mkdir -p "$SRC/figures"

"$PYTHON" -u "$FIGURES_DIR/figure1_dc_grid.py"

echo "Done: $(date)"
