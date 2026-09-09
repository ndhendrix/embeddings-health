# Validate on Sherlock

For other servers, use [the general Slurm guide](SLURM.md). Its settings table
explains which paths, modules, queues, and resource requests must change.

This branch contains the standalone reproduction package inside `reproduction/`.
The data deposit is not published yet: cloning GitHub supplies code and reference
results, but does not supply the 6.8 GB analysis data. Full Linux/Slurm validation
is still pending. Use a separate clone to preserve the existing analysis checkout.

## 1. Clone on Sherlock

On a Sherlock login node:

```bash
cd "$SCRATCH"
git clone --branch codex/reproduction-package --single-branch \
  https://github.com/ndhendrix/embeddings-health.git embeddings-health-reproduction
mkdir -p "$SCRATCH/embeddings-health-reproduction-data"
printf '%s\n' "$SCRATCH/embeddings-health-reproduction-data"
```

Save the absolute data path printed by the final command. For subsequent code
updates, with no validation jobs running:

```bash
cd "$SCRATCH/embeddings-health-reproduction"
git pull --ff-only origin codex/reproduction-package
```

Do not change this checkout or its environment while jobs run. Scratch storage
is temporary; retain the original data elsewhere and copy final reports back.

## 2. Transfer data from the Mac

Run this in a terminal **on the Mac**, replacing `YOUR_SUNET_ID` and
`ABSOLUTE_DATA_PATH` with your login and the path printed above:

```bash
rsync -avP \
  -e "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  /Users/nathanielhendrix/Documents/Current_Projects/embeddings-health/outputs/reproduction_package/ \
  YOUR_SUNET_ID@sherlock.stanford.edu:ABSOLUTE_DATA_PATH/
```

Keep the trailing slash on the source directory. This transfers the staged
bundle directly into the destination. Repeat the same command if interrupted;
rsync reuses transferred files. Complete the transfer before submitting jobs.
Authentication may require Stanford's usual interactive verification.

## 3. Prepare and verify on Sherlock

```bash
cd "$SCRATCH/embeddings-health-reproduction/reproduction"
module load devel gcc/14.2.0
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
unset UV_PROJECT_ENVIRONMENT
./replicate.sh --data-dir "$SCRATCH/embeddings-health-reproduction-data" --check-only
```

This builds the isolated environment and checks all 17 required input files.
The compiler/OpenMP environment is inherited by submitted jobs. Environment setup
requires internet access. It performs no model fitting on the login node.

## 4. Submit a smoke test

```bash
./replicate.sh --slurm --smoke --partition normal \
  --data-dir "$SCRATCH/embeddings-health-reproduction-data" \
  --output-dir "$SCRATCH/embeddings-health-reproduction-smoke" \
  --threads 4 --array-limit 4 --memory 32G --time-limit 12:00:00
```

Record both printed job IDs. Once the array and dependent validation job finish,
read `$SCRATCH/embeddings-health-reproduction-smoke/REPORT.md`. Expect 36/36 checks
and zero missing tasks. A submission confirmation is not a validation pass.

## 5. Submit the full validation after the smoke test passes

```bash
./replicate.sh --slurm --partition normal \
  --data-dir "$SCRATCH/embeddings-health-reproduction-data" \
  --output-dir "$SCRATCH/embeddings-health-reproduction-full" \
  --threads 4 --array-limit 12 --memory 32G --time-limit 12:00:00
```

This submits 726 tasks, with at most 12 running simultaneously, each requesting
4 CPUs and 32 GB. Add `--account YOUR_ACCOUNT` if required by your allocation.
Queue delays and actual resource needs remain to be measured in this validation.
The dependent collector runs even after task failures and reports missing or
mismatched results. Once submission succeeds, you may disconnect or close your
laptop; execution is on Sherlock.

```bash
squeue -u "$USER"
cat "$SCRATCH/embeddings-health-reproduction-full/REPORT.md"
```

The report is created by the collector after the array ends. Final success requires
`validation.json` to report `passed` and scope `all_included_analyses`, with all
726 tasks present. Preserve `validation.json`, `validation_checks.csv`, `REPORT.md`,
the plan, task JSONs, tables, figures, and Slurm logs. If tasks fail, inspect their
logs before retrying; identical reruns reuse valid completed task results.
