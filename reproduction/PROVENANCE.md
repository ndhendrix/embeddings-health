# Data and analysis provenance

The tract aggregates were supplied by a collaborator. The paper's author ran the
analyses. Embeddings retain means, minima, maxima, and standard deviations;
medians, pixel counts, and PCA64 inputs are excluded.

The analysis implementation derives from `code/analyses/analyses_sherlock.py`
in the research repository; table presentation follows `code/figures/tables.R`.
The reference tables identify numerical comparison targets, not cached model fits.
The exact execution revision for every original results directory was not recorded.

`analysis_samples.parquet` contains tract membership and two explicit row orders:
ACS source order and the original post-join PLACES order. These affect shuffled
splits. State files are concatenated in sorted filename order while preserving
within-file order for ACS. Index-source tract identifiers establish PLACES order;
no source index values are included.

`acs_folds.json` preserves all 138 model/target state assignments from the
Sherlock export, with sample fingerprints checked before fitting. The export
recorded NumPy 1.26.4 and scikit-learn 1.3.0 on an Intel Xeon Gold 5118 node.
Explicit assignments avoid regenerating platform-dependent ordering of tied
state-group sizes.

Prithvi-300M excludes 6,925 rows whose retained features were all null. Retained
feature values are unchanged; none of the removed rows was in the analysis sample.

The data record's `manifest.json` documents source file hashes, column mappings,
coverage, transformations, and available provenance. Precise checkpoint revisions
were not recorded in the supplied CSVs. The code's `data-lock.json` pins every
required download by SHA-256. Source notices govern third-party materials.

Numerical results can depend on platform/compiler details. Preserve the locked
environment and use [VALIDATION.md](VALIDATION.md) to assess each run.

The original Prithvi-300M prepared data use Float32 predictors, including ALAND
and AWATER. The runner applies this conversion at fitting time. A paired ACCESS2
diagnostic reproduced the original full-run score with Float32 predictors while
holding sample/order, outcomes, holdouts, and model parameters fixed. Broader
numerical agreement must be assessed from the resulting run reports.

Tiny's median-family-income folds place Rhode Island in fold 2 and South Dakota
in fold 4. Its income-disparity folds place Tennessee in fold 1 and Massachusetts
in fold 5. The paired full-input diagnostic confirmed these assignments reproduce
the original ACS means and fold SDs. `acs_folds.json` records the correction and
its evidence hash; all other model/target assignments are preserved.
