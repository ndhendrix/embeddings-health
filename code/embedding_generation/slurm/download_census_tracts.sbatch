#!/bin/bash
#SBATCH --job-name=dl-census-tracts
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=code/embedding_generation/slurm/logs/download_tracts_%j.out
#SBATCH --error=code/embedding_generation/slurm/logs/download_tracts_%j.err

set -euo pipefail

: "${SCRATCH:?SCRATCH is not set}"
TRACT_DIR="${TRACT_DIR:-$SCRATCH/embeddings-health/data/census_tracts}"
YEAR="${YEAR:-2020}"   # TIGER/Line vintage (2020 tracts match AlphaEarth)

mkdir -p "$TRACT_DIR"

declare -A STATE_FIPS=(
  [AL]=01 [AZ]=04 [AR]=05 [CA]=06 [CO]=08 [CT]=09
  [DC]=11 [DE]=10 [FL]=12 [GA]=13 [ID]=16 [IL]=17
  [IN]=18 [IA]=19 [KS]=20 [KY]=21 [LA]=22 [ME]=23
  [MD]=24 [MA]=25 [MI]=26 [MN]=27 [MS]=28 [MO]=29
  [MT]=30 [NE]=31 [NV]=32 [NH]=33 [NJ]=34 [NM]=35
  [NY]=36 [NC]=37 [ND]=38 [OH]=39 [OK]=40 [OR]=41
  [PA]=42 [RI]=44 [SC]=45 [SD]=46 [TN]=47 [TX]=48
  [UT]=49 [VT]=50 [VA]=51 [WA]=53 [WI]=55 [WV]=54 [WY]=56
)

BASE_URL="https://www2.census.gov/geo/tiger/TIGER${YEAR}/TRACT"
OK=0; SKIP=0; FAIL=0

for state in $(echo "${!STATE_FIPS[@]}" | tr ' ' '\n' | sort); do
  fips="${STATE_FIPS[$state]}"
  dest="$TRACT_DIR/tl_${YEAR}_${fips}_tract.zip"
  if [[ -f "$dest" ]]; then
    echo "SKIP $state ($fips)"
    (( ++SKIP ))
    continue
  fi
  url="$BASE_URL/tl_${YEAR}_${fips}_tract.zip"
  echo "Downloading $state ($fips)..."
  if curl -fsSL --retry 3 --retry-delay 5 -o "${dest}.tmp" "$url"; then
    mv "${dest}.tmp" "$dest"
    echo "OK   $state ($fips) → $(du -sh "$dest" | cut -f1)"
    (( ++OK ))
  else
    echo "FAIL $state ($fips)" >&2
    rm -f "${dest}.tmp"
    (( ++FAIL ))
  fi
done

echo ""
echo "=== Done: $OK downloaded, $SKIP skipped, $FAIL failed ==="
ls "$TRACT_DIR"/*.zip 2>/dev/null | wc -l
echo "zip files in $TRACT_DIR"
