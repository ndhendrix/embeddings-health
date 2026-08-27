#!/bin/bash
#SBATCH --job-name=fig1-pa
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/nhendrix/embeddings-health/code/figures/logs/fig1_pa_%j.out
#SBATCH --error=/home/users/nhendrix/embeddings-health/code/figures/logs/fig1_pa_%j.err

set -euo pipefail

PYTHON="$SCRATCH/embeddings-health/cache/venv-3.11-cpu/bin/python"
FIGURES_DIR="$HOME/embeddings-health/code/figures"
SRC="$SCRATCH/embeddings-health"

echo "Node: $(hostname)"
echo "Start: $(date)"

# Stage smaller CA files to L_SCRATCH.
# Clay CA (835G) is too large to stage; read from Lustre.
STAGE="$L_SCRATCH/tifs"
mkdir -p \
  "$STAGE/olmoearth_nano_embeddings/CA" \
  "$STAGE/prithvi_embeddings/tiny/CA" \
  "$STAGE/prithvi_embeddings/300M-TL/CA"

echo "Staging TIFs to L_SCRATCH ($STAGE) ..."

cp "$SRC/olmoearth_nano_embeddings/CA/olmoearth_Nano_CA_2022.tif" \
   "$STAGE/olmoearth_nano_embeddings/CA/" &
cp "$SRC/prithvi_embeddings/tiny/CA/prithvi_tiny_CA_2022_raw.tif" \
   "$STAGE/prithvi_embeddings/tiny/CA/" &
cp "$SRC/prithvi_embeddings/300M-TL/CA/prithvi_300M-TL_CA_2022.tif" \
   "$STAGE/prithvi_embeddings/300M-TL/CA/" &

wait
echo "Staging done: $(date)"
df -h "$L_SCRATCH"

export STAGE_DIR="$STAGE"
mkdir -p "$SRC/figures"

"$PYTHON" -u "$FIGURES_DIR/figure1_pa_grid.py"

echo "Done: $(date)"
