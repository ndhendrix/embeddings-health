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
