# Embeddings and health: reproduction package

One command reruns the analyses supported by the shared tract-level data,
creates scoped tables and figures, and compares the numbers against the paper.
**A completed run is successful only when
`outputs/validation.json` reports `passed` for `all_included_analyses`.**
A smoke-test pass is not a full replication claim.

In this research repository, first `cd reproduction`. For Sherlock setup, data
transfer, and job submission, see [SHERLOCK.md](SHERLOCK.md).

## Run locally

Clone this repository. Place the exact Zenodo files directly in `data/`, then:

```bash
./replicate.sh
```

The script creates an isolated Python 3.11.13 environment with the committed lockfile,
checks the data hashes, fits the models, assembles tables/figures, and writes a
validation report. If uv is absent, the script installs uv 0.10.0 in a local
`.bootstrap` environment using Python 3. It does not install into system Python.
An internet connection is needed for first-time environment setup.
LightGBM 4.3.0 needs a source build on Apple Silicon and older Linux systems.
Install a C/C++ compiler with OpenMP support first; uv handles Python build
dependencies. On macOS, install Xcode command-line tools and `brew install libomp`.
On Sherlock, load the same GCC module used by the successful diagnostic before
running setup (`module load devel gcc/14.2.0`). Other clusters should use their
local compiler module. The remaining pinned dependencies have compatible wheels
for Linux x86-64 with glibc 2.17.

To download pinned files from a public Zenodo record without an API token:

```bash
./replicate.sh --record-id NUMERIC_VERSION_RECORD_ID
```

Use an exact version record, not a moving latest-version identifier. Downloads
are accepted only if their SHA-256 hashes match `data-lock.json`. Completed files
are reused; interrupted files are downloaded again on retry. No private data are
uploaded by this program. Manual and automatic downloads use the same checks.

Other useful commands:

```bash
./replicate.sh --check-only
./replicate.sh --smoke --output-dir outputs/smoke
./replicate.sh --data-dir /path/to/downloads --threads 4 --workers 1
./replicate.sh --models alphaearth --stages places
./replicate.sh --fresh
```

The full plan contains 726 tasks: 240 PLACES/residual tasks, 138 ACS tasks,
288 state tasks, and 60 decile tasks. A full run is computationally substantial. The default uses one task at a time
and four LightGBM threads. Large-model tasks can use many GB of RAM; do not increase
`--workers` without budgeting memory per worker. `--smoke` runs four AlphaEarth
tasks: ACCESS2 direct/residual fits, median-family-income ACS, Alabama, and the
first tract-area decile. State/decile tasks use all 40 PLACES outcomes, not toy data.
Smoke and other subsets should use separate output directories.

Matching completed tasks are resumed automatically. Data hashes, code, model
settings, numerical-library versions, Python/platform information, thread count,
and a payload checksum determine whether a completed task can be reused.
`--fresh` explicitly refits. A failed fresh task cannot fall back to old results.

## Run on Slurm

Read [SLURM.md](SLURM.md) for the execution layout, cluster-specific settings,
shared-storage requirements, scheduler limitations, and recovery instructions.
Use [SHERLOCK.md](SHERLOCK.md) only for the Stanford-specific example.

From a Linux login node with uv or Python 3, using a shared filesystem:

```bash
./replicate.sh --slurm --partition YOUR_PARTITION --account YOUR_ACCOUNT \
  --threads 4 --array-limit 12 --memory 32G --time-limit 12:00:00
```

Partition and account are optional if your cluster supplies defaults. Setup and
all data downloads happen before submission. An array uses the same numerical
tasks as local execution; a dependent collector runs after the array finishes,
even if tasks failed, to report missing outputs. There are no Stanford-specific
paths, module names, or scratch assumptions. Each worker rechecks its inputs.
The installed environment and input/output directories must be visible on compute
nodes. Submission prints job IDs; it does not claim the jobs have completed.

Local and Slurm runs use the same functions and parameters. Slurm submission has
unit/integration coverage using a simulated scheduler; execution on a real cluster
still requires a release validation run.

## Included analyses

| Analysis | Generated output | Comparison |
|---|---|---|
| Embeddings predict PLACES | `tables/table_s1_embeddings_only.csv` | Embedding-only rows of S1 |
| Embeddings predict supplied index-adjusted residuals | `tables/table_s2_figure3_incremental_r2.csv`, S2 wide tables, S5 | Corresponding S2/S5 values |
| Embeddings predict 23 available ACS concepts | `tables/acs_included_long.csv`, `table_s6_acs_included.csv` | Corresponding S6 means and fold SDs |
| Embedding performance by state | `tables/table_s4_embeddings_only.csv` | Embedding columns of S4 |
| Embedding performance by tract-area decile | `tables/table_s3_embeddings_only.csv` | Embedding columns of S3 |

Training/test membership, source row order, feature order, transformations, and
model parameters are preserved explicitly. The first-stage residualization is
not refitted: supplied training residuals are in-sample, and test residuals are
from held-out states. Each embedding model uses its own recorded tract sample.
The PLACES 2025 release supplies five 2022 measures and 35 additional 2023 measures;
actual years are in the data. No outcome year is silently changed by this code.

The ACS routines preserve the paper code's variable definitions and missing-value
rules, including its order of transformations. This package does not silently
repair or reinterpret published variables. Standard deviations use `ddof=0`
across the five state-grouped CV folds. State/decile fits use shuffled two-fold
CV with the original smaller LightGBM settings.

## Intentionally excluded

- ReADI/SDI/SVI values, index-only fits, and joint index-plus-embedding fits.
- Index-based state and tract-size comparisons.
- The two ReADI-sourced ACS targets (`MEDMORT`, `NOINT`).
- PCA64 sensitivity analyses and Table S7.
- Pixel-level imagery and map figures.
- Standalone refitting of index-only workbook results (Table S8).

S9's sequential-analysis quantities are represented by the regenerated S2 and S5
CSV tables; this package does not reproduce the original Excel workbook formatting.
Reference files are validation inputs, never a fallback for missing computation.

## What “match” means

All included numerical values must agree at the precision printed in the paper:
four decimal places for performance tables and three for S6 means/SDs. Sample
counts must match exactly. The program does not loosen tolerances to obtain a pass.
Unrounded results and fold scores are retained in `outputs/tasks/`. A difference,
failed task, or incomplete requested run produces a nonzero exit status and a
report. Expected tables are never read by the fitting module.

Table S2 preserves the original R script's calculation order: round the index
R² to four decimals, then combine it with the unrounded additional variance
for the sequential R² and unexplained-variance percentage. Table S5 summarizes
these displayed S2 values. Raw task results and plot inputs retain full precision.

`outputs/REPORT.md`, `validation.json`, and `validation_checks.csv` show the scope,
every tested value, and exclusions. Scoped PNG/PDF figures are redrawn from computed
data with the paper's model colors and with excluded index curves removed. They
are not byte-identical reproductions of the original images or page layouts.
Floating-point results can depend on platform/compiler details, so the environment
is recorded in the run report. Assess each execution using its numerical checks.

## Tests and repository contents

```bash
uv run --locked --python 3.11.13 python -m unittest discover -s tests -v
```

`analysis.py` fits models, `acs.py` contains the retained source transformations,
`reporting.py` assembles outputs and comparisons, and `replication.py` coordinates
execution. The `reference/` directory contains small published result tables.
The code repo excludes large data, environments, outputs, credentials, and the
maintainer-only scripts that read restricted indices.

Use the exact Zenodo version record ID for downloads. Do not replace pinned
checksums to accept different inputs. See [VALIDATION.md](VALIDATION.md) for
acceptance criteria and the data record for source attributions.

## Explicit folds and analysis order

The shared inputs now preserve two different row orders: `source_order` for ACS
and `q2_order` for PLACES and heterogeneity. The latter preserves the original
post-join order before shuffled splitting. ACS assignments come from
`acs_folds.json`; the runner validates each sample fingerprint and never silently
regenerates missing folds. This avoids platform-dependent ordering of equal-sized
state groups. All 138 model/target mappings are pinned, with sample fingerprints
checked against the shared inputs. Prithvi-300M excludes all-null source rows;
none of those rows belongs to the analysis sample.

## License

The reproduction code and author-written documentation are available under the
[MIT License](LICENSE). Third-party datasets, reference tables, and upstream
materials are excluded from this grant and retain their applicable terms.
See the data record's SOURCE_NOTICES.md for attribution and provenance limitations.

## Prithvi-300M predictor precision

Prithvi-300M fits use Float32 embedding and area predictors, matching the original
full-dimensional analysis. Conversion occurs only when constructing model inputs;
shared Parquet values, outcome precision, and area-based grouping/reporting are
preserved. Other models retain their existing predictor precision.
