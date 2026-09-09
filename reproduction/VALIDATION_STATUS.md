# Validation status: 2026-09-09

The corrected four-task AlphaEarth smoke run passes **36 of 36 checks**, with
no missing tasks. This validates the representative checks, not every included
paper result. Nothing has been uploaded to Zenodo.

## Confirmed corrections

The author's Sherlock diagnostic matched the local analysis code, input-file
checksums, prepared feature values, ACS target values, tract membership, and row
order. Its GroupKFold state assignments differed when state sizes were tied.
Using those exact assignments locally reproduced the median-family-income ACS
mean R² and fold SD to floating-point precision.

The state and area-decile analyses use the original post-join PLACES row order,
which differs from the ACS source order. Preserving it reproduces the Alabama
and first-area-decile reference results at reported precision.

| Comparison | Corrected reproduction | Paper reference |
|---|---:|---:|
| ACS median-family-income mean R² | 0.2180365391 | 0.2180365391 |
| ACS median-family-income fold SD | 0.1077939883 | 0.1077939883 |
| Alabama mean embedding R² | 0.3389330506 | 0.3389 |
| First tract-area decile mean embedding R² | 0.4162494800 | 0.4162 |

The held-out PLACES and residualized-outcome smoke checks also pass. Reference
values and comparison precision were not changed to obtain this result.
The actual `replicate.sh` entry point passed all 36 smoke checks again after the
final dependency change, in its locked Python 3.11.13 environment. The broader
174-check run preceded that dependency change; it has not been repeated under
the final lockfile.
A broader run also passed all 174 checks across the 58 AlphaEarth state/decile
tasks. Thirteen automated tests pass, including new regression checks for frozen folds,
changed sample identities, and separate ACS/PLACES row order.

The recorded Sherlock numerical library versions were also tested in an isolated
local environment. Those versions alone did not resolve the differences; explicit
fold assignments and correct post-join order did. The release lock now uses the
recorded numerical versions and plotting packages with Linux glibc 2.17 wheels.
LightGBM requires a local source build on this older Linux target; GCC/OpenMP
must be available before setup. This compatibility resolution is not a completed
Linux installation or full Slurm run.

## Prithvi-300M cleanup

The refreshed aggregate has 76,755 rows, down from 83,680. All 6,925 removed rows
had null values for every retained embedding feature. None belonged to the analysis
sample. Every retained value is unchanged. The Parquet file was rebuilt with a
full source-to-Parquet equality check, and its data-lock checksum was updated.

## Full ACS fold export

The export from Sherlock job 42605275 passed its confirmed-case guard and contains
all 138 requested model/target mappings. Its request checksum, sample fingerprints,
state counts, mapping coverage, and previously confirmed AlphaEarth folds passed
the importer checks. The complete file is now staged and pinned in data-lock.json.
All 23 shared AlphaEarth ACS targets were subsequently refitted locally under
the final lockfile. All 138 report checks passed, with no missing tasks or
discrepancies. The remaining five models still require ACS numerical validation.

## Remaining release gates

The full 726-task local validation started on September 9. An interim check of
49 completed AlphaEarth tasks (23 ACS and 26 PLACES/residual tasks) passed
762 of 762 comparisons after correcting a reporting-only calculation-order
difference. The original R table script rounds the index R² before computing
the sequential R² and unexplained-variance percentage; the reproduction now
does the same in reporting, leaving raw fitted scores unchanged. All 14
automated tests pass, including a regression for this behavior. References and
comparison tolerances were not changed. The interim check reassembled existing
raw fits with the corrected reporter; it is not a completed full validation.
The full run is being restarted in `outputs/full-validation-v2` under the new
code fingerprint, without reusing the earlier task cache.

1. Complete validation across every included model, outcome, and result family,
   including a full run on the intended Linux/Slurm release environment. Slurm
   orchestration has been tested with a simulated scheduler, not a full live run.
2. Publish the permitted data bundle and set its exact Zenodo version record ID.
3. Select the software license and complete source-specific data notices.

The colleague supplied the tract aggregates. The author generated the August
results and their September Sherlock rerun; earlier attribution of the fitted
results to the colleague was incorrect.
