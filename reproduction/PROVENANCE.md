# Provenance

Prepared from the embeddings-health analysis workspace on 2026-09-08.
Authoritative numerical targets: the user-designated `outputs/tables` directory.
Model mapping uses `code/figures/tables.R`, including the Prithvi `_full` reruns.
ACS transformations and model settings derive from `code/analyses/analyses_sherlock.py`.
The data package was built from `data/gfm_agg`, excluding medians and pixel counts.

The source workspace's August 27 lineage audit could not establish the precise
execution revision for every final results directory. Validation discrepancies
are therefore reported, never replaced by cached expected values.

Data download uses Zenodo public file endpoints and code-pinned SHA-256 hashes:
https://developers.zenodo.org/
LightGBM documents platform/compiler sensitivity of results:
https://lightgbm.readthedocs.io/en/stable/Parameters.html#deterministic

A small `analysis_samples.parquet` supplements the original data bundle with
tract IDs, membership flags, ACS source order, and the original post-join order
for PLACES/heterogeneity. It includes no index values. For state-file collections,
the supplied files are concatenated in sorted filename order, with within-file
order retained for ACS. The original PLACES inner join follows the index-source
tract-ID order; freezing this order reproduces the AlphaEarth heterogeneity
diagnostic results. The state-group ACS folds are separately frozen in
`acs_folds.json`, with each target sample fingerprint checked before fitting.

The author clarified that a colleague supplied the tract aggregates, downloaded
on June 29, and that the author reran the analyses on August 19 after excluding
medians. On September 9 the author supplied another Sherlock run. Its state and
decile CSVs are byte-identical to the August results; the other CSVs contain only
a few small numerical changes. The local Slurm script requests a fresh fit and
excludes AK/HI. The returned Sherlock diagnostic confirmed identical source code, source inputs,
prepared features, ACS target values, and row order. Only the ACS fold assignments
differed. Reusing Sherlock folds locally matches the paper ACS mean and SD to
floating-point precision. Restoring the original post-join row order also matches
the Alabama and first-area-decile references at reported precision. These are
verified representative checks, not verification of every model and outcome. Earlier attribution of the result files
to the colleague was incorrect; the colleague supplied the aggregates.

All 138 ACS state assignments were imported from the validated Sherlock export
(job 42605275). The exporter recorded NumPy 1.26.4 and scikit-learn 1.3.0 on an
Intel Xeon Gold 5118 node. The successful export matches the previously confirmed
AlphaEarth case; the earlier failed export does not establish its own cause
because that run did not preserve an environment report.
