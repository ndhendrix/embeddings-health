# AlphaEarth paired diagnostic

This runs only the five-fold median-family-income analysis, using the current
`analyses_sherlock.py` data preparation and fitting function directly. It excludes
AK/HI and asserts that the embedding statistics are MEAN, MINIMUM, MAXIMUM, and
STD. It preserves the runner's feature order, pandas conversion, and model
parameters. It never reads cached scores or writes to the paper result folders.

## Run on Sherlock

From the repository root, after pulling the branch containing this diagnostic:

```bash
mkdir -p code/analyses/slurm/logs
sbatch code/analyses/slurm/run_alphaearth_diagnostic.sbatch
```

The launcher uses the same environment setup and eight CPUs as the regular
AlphaEarth launcher. Reports are saved to:

```text
$SCRATCH/embeddings-health/diagnostics/alphaearth_<job ID>/
```

Return that directory's `report.json` and `tracts.csv`, along with the job's
`.out` and `.err` logs. A failed job can leave a partial directory; only a completed
job with `report.json` contains the full diagnostic. Every run requires a new
output directory. No original data files need to be returned.

## Run locally

Use the existing project environment, so its versions can be compared with the
Sherlock environment. From the repository root:

```bash
OMP_NUM_THREADS=8 uv run python code/analyses/diagnostics/alphaearth.py \
  --data-root data \
  --embeddings data/gfm_agg/alphaearth_aggregated/alphaearth_embeddings.csv \
  --output outputs/reproduction_audit/alphaearth_diagnostic_local
```

`--data-root` contains the original PLACES CSV and `social_risk_indices/`.
The ACS source defaults to the repository's `data/acs.csv`; use `--acs` to override
it. Sherlock defaults to `<data-root>/alphaearth/alphaearth_embeddings.csv`.
The original preparation reads the ReADI source to preserve its joins, but this
diagnostic exports no ReADI, SDI, or SVI values.

## Compare

This comparison uses only the Python standard library:

```bash
python3 code/analyses/diagnostics/alphaearth.py --compare \
  outputs/reproduction_audit/alphaearth_diagnostic_local \
  /path/to/downloaded/sherlock/diagnostic
```

Exit 0 means all compared fields match; exit 2 means at least one differs.
Platform and environment differences are expected between macOS and Linux and
will produce exit 2 even if scores agree. Inspect the individual fields.

- Source and input checksums identify different code or source files.
- Ordered and sorted tract/feature checksums distinguish row order from values.
- Per-feature checksums help locate a changed predictor.
- `tracts.csv` records row order, tract ID, state, held-out fold, observed ACS
  median family income, and out-of-fold prediction.
- Fold reports contain held-out states, sample sizes, R², effective LightGBM
  parameters, and fitted model checksums.
- Environment details include numerical package versions and thread libraries.

Compare inputs, features, and folds before attributing score differences to the
platform. File hashes alone may differ because of harmless CSV formatting; the
prepared numerical checksums provide a second comparison. The diagnostic is
intentionally coupled to two markers in the original analysis source and fails
if those markers are removed. It does not alter the reproduction package or the
paper's validation targets.

## Export the remaining ACS folds

The successful paired diagnostic established that some tied state sizes receive
platform-dependent GroupKFold assignments. The paper reproduction therefore needs
explicit assignments for every model and shared ACS target.

On Sherlock, in the same environment as the successful diagnostic:

```bash
mkdir -p code/analyses/slurm/logs
sbatch code/analyses/slurm/export_acs_folds.sbatch
```

Return `$SCRATCH/embeddings-health/diagnostics/acs_folds_<job ID>.json`.
This runs no models and reads no original embeddings or restricted-index data.
It uses the provided tract counts by state for 138 model/target combinations.
GroupKFold's state allocation depends on these counts, not within-state row order.
A built-in check requires exact agreement with the previously confirmed AlphaEarth
median-family-income folds. An environment that fails this check cannot export.

### If the fold export fails validation

The exporter keeps the original guard. A failed match now writes
`acs_folds_<job ID>.diagnostic.json` alongside the intended output path, then exits
with an error. Return that diagnostic file and the job logs. It contains candidate
assignments, exact state differences, numerical package versions, CPU details,
and NumPy runtime information. It is explicitly marked failed and cannot be
imported as a release manifest. No normal `acs_folds_<job ID>.json` is created.

The successful original AlphaEarth diagnostic recorded NumPy 1.26.4,
scikit-learn 1.3.0, and an OpenBLAS CPU designation of SkylakeX. That designation
is a diagnostic clue, not a verified Slurm constraint or exact node model.
The new report allows comparison of both software and node CPU capabilities.

## Prithvi-300M full-run lineage inspection

This diagnostic inspects the prepared directory and run root recorded in the
final full-dimensional results. It inventories files, reads table schemas and
array shapes, extracts small saved configuration summaries and relevant source
lines, and compares available feature lists with the reproduction configuration.
It does not fit models, change inputs, or transfer tract-level data into the report.
A complete inspection is not a numerical validation pass.

On Sherlock, from the separate reproduction checkout:

```bash
cd "$SCRATCH/embeddings-health-reproduction"
git pull --ff-only origin codex/reproduction-package
sbatch code/analyses/slurm/inspect_prithvi_full.sbatch
```

It uses the existing `reproduction/.venv` from the completed reproduction run,
requests one CPU, 4 GB, and 15 minutes, and writes:

- `$SCRATCH/prithvi-full-diagnostic-JOBID.json`
- `prithvi-lineage-JOBID.log` in the submission directory.

Download the JSON report after completion. If the job fails, include the log.
Missing directories and inspection errors are saved in the report with
`inspection_status: incomplete` and exit code 2. This evidence is still useful.
Existing reports are never overwritten.

Defaults follow the original result metadata. If your paths differ, set the
relevant variables before submission (paths must be visible on compute nodes):

```bash
export ORIGINAL_REPO=/absolute/path/to/original/embeddings-health
export PREPARED_DIR=/absolute/path/to/prithvi_300m_tl_full_prepared
export RUN_ROOT=/absolute/path/to/original/sharded/run
sbatch code/analyses/slurm/inspect_prithvi_full.sbatch
```

`REPRO_REPO` defaults to `$SCRATCH/embeddings-health-reproduction`;
`ORIGINAL_REPO` defaults to `$HOME/embeddings-health`. The launcher records both
prepared/run locations even if absent. It does not infer replacement directories.

## Paired Prithvi-300M predictor-precision comparison

After the lineage report identifies Float32 predictors, run:

```bash
cd "$SCRATCH/embeddings-health-reproduction"
git pull --ff-only origin codex/reproduction-package
sbatch code/analyses/slurm/compare_prithvi_precision.sbatch
```

This fits the direct ACCESS2 model twice, once with Float64 predictors and once
with all predictors (including ALAND/AWATER) cast to Float32. Both use identical
outcomes, feature order, tract order, holdout states, model parameters, and four
threads. It first compares the original prepared file's sample/order, targets,
feature list and holdout metadata, and checks every predictor value against the
Float32-cast shared inputs in column blocks. No original script is executed.
The paper reference is read for comparison only after fitting.

The job requests 4 CPUs, 32 GB and a two-hour time limit. It uses the existing
reproduction environment. It writes `$SCRATCH/prithvi-precision-JOBID/report.json`
and `prithvi-precision-JOBID.log` in the submission directory. Download the report;
include the log if the job fails. Intermediate reports have status `running`;
only `comparison_complete` means both fits and reference checks finished. Neither
status certifies the full reproduction. Output directories cannot be reused.

`REPRO_REPO`, `DATA_DIR`, and `PREPARED_DIR` may override the default paths in the
launcher. No production data, code, cached task results, or expected scores are
modified. A synthetic end-to-end check exercises alignment, both fit paths,
report creation, and overwrite protection; the real-data comparison runs on Sherlock.

## Prithvi Tiny: two discrepant ACS targets

On Sherlock, from the reproduction checkout:

```bash
cd "$SCRATCH/embeddings-health-reproduction"
git pull --ff-only origin codex/reproduction-package
sbatch code/analyses/slurm/compare_prithvi_tiny_acs.sbatch
```

The job compares median family income and income disparity. It checks current
original-script preparation against shared inputs by tract ID, including sample
membership, row order, feature order, predictor values, and target values. It
compares frozen ACS folds with GroupKFold assignments generated on the current
node. It fits Float64 and Float32 predictor variants with the same frozen folds;
if relevant, it also fits current-fold and original-preparation variants. All
fits use the reproduction estimator parameters, which are recorded alongside
the original fit-function source. This compares current preparations; it does
not claim to reconstruct an unrecorded historical environment.

The original preparation section is executed from your original repository's
`code/analyses/analyses_sherlock.py`; no analysis-output writing or model-fitting
section of that script is executed. Original input paths must exist. No data are
automatically downloaded. Tiny's original default uses no explicit AK/HI exclusion;
any membership differences are reported instead of silently aligning the samples.

Default paths follow the original Tiny ACS launcher: original code under
`$HOME/embeddings-health`, source files under `$SCRATCH/embeddings-health`, and
Tiny's full state CSVs under `prithvi_aggregated_full/tiny/`.
The diagnostic combines these into its own output directory, preserving values
and within-file order and recording sorted filenames and source hashes. This
order is explicit, but is not assumed to be the historical combined-file order.
The older `prithvi_aggregated/prithvi_tiny_2022_all_tracts.csv` contains PCA64
features and must not be used for the full-dimensional comparison.
Override `ORIGINAL_REPO`, `ORIGINAL_DATA`, `EMBEDDINGS`, `DATA_DIR`, or `REPRO_REPO`
if needed. Feature discrepancies can reveal a stale combined CSV from PCA64 work.
The existing reproduction environment is used; the job requests 4 CPUs, 32 GB,
and two hours. It can run alongside the Prithvi-300M validation with separate outputs.

Return `$SCRATCH/prithvi-tiny-acs-diagnostic-JOBID/report.json`. If the job fails,
include `tiny-acs-diagnostic-JOBID.log` from the submission directory. Per-target
tract CSVs contain public ACS targets and fold assignments for follow-up if needed.
Only `comparison_complete` means the entire diagnostic finished. A synthetic
end-to-end check exercised two targets, mismatched fold assignments, precision
variants, original preparation, and output overwrite protection. No real fits
were run on the Mac.

## Tiny historical-run evidence

The full-input comparison matched predictor/target values and current state
folds but not the two historical ACS reference scores. To locate saved evidence:

```bash
cd "$SCRATCH/embeddings-health-reproduction"
git pull --ff-only origin codex/reproduction-package
sbatch code/analyses/slurm/inspect_tiny_history.sbatch
```

Return `$SCRATCH/tiny-history-JOBID.json`; if the job fails, also return
`tiny-history-JOBID.log` from the submission directory. The job requests one CPU,
2 GB, and ten minutes. It does not run models or execute historical scripts.

The audit searches original analysis scripts/logs, Tiny output directories,
Tiny-specific cache folders, and top-level scheduler logs. It extracts relevant
log/launcher lines, source hashes and timestamps, the two target rows from saved
ACS result files, recent relevant git history, and current cached environment
package versions. Current files and package versions do not establish what ran
historically. Truncation, missing search locations, and read errors are explicit.
Retain the full logs locally for follow-up; the report contains bounded excerpts.

Override `ORIGINAL_REPO`, `ORIGINAL_DATA`, or `REPRO_REPO` before submission if
these differ from the preceding diagnostic. No analysis inputs, fitted outputs,
or environment files are changed. A synthetic check exercises discovery, cache
messages, relevant result selection, redaction, and overwrite protection.

## Prithvi-300M income-disparity fold comparison

After the Float32 full-model run, test its remaining ACS discrepancy:

```bash
cd "$SCRATCH/embeddings-health-reproduction"
git pull --ff-only origin codex/reproduction-package
sbatch code/analyses/slurm/compare_prithvi_300m_income_folds.sbatch
```

First copy the latest release `acs_folds.json` into the shared data directory if
not already done. The current code pins the Tiny-corrected file, even though
this diagnostic changes only 300M folds in memory. A mismatched download fails
its checksum check before fitting; never bypass that check.

The job fits five folds twice for income disparity, with Float32 predictors:
current frozen assignments versus MA moved from fold 1 to 5 and TN from 5 to 1.
It checks that these states have equal nonzero sample counts and the expected
baseline assignments. Input matrices, outcomes, and other assignments are fixed.
No release files or reference scores are modified. It requests four CPUs, 32 GB,
and a two-hour time limit. A small synthetic test checks the swap and its guards;
real fitting runs on Sherlock.

Return `$SCRATCH/prithvi-300m-income-folds-JOBID/report.json` and, if the job fails,
`prithvi-income-folds-JOBID.log` from the submission directory. Only
`comparison_complete` means both variants finished. A matched single target is
not a complete validation run. `DATA_DIR` and `REPRO_REPO` override shared paths.
