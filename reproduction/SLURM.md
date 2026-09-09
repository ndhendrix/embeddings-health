# Adapting the reproduction package to your Slurm cluster

Start with [README.md](README.md) for data and analysis scope. [SHERLOCK.md](SHERLOCK.md)
is a Stanford example, not a portable cluster configuration. Real-cluster validation
is still pending; see [VALIDATION_STATUS.md](VALIDATION_STATUS.md).

## How the run is organized

1. Run `replicate.sh --slurm` on a host allowed to install dependencies, read the
   shared inputs, and submit jobs. It sets up the locked Python environment,
   downloads missing inputs if a published record ID is supplied, verifies their
   hashes, and writes an immutable task plan. It does not fit models on this host.
2. The launcher submits an array: one independent analysis task per array index.
   The full plan has 726 tasks (240 PLACES/residual, 138 ACS, 288 state, 60 decile).
   Workers use the same analysis functions as local execution, verify their input
   files, and save individual result JSONs. They use CPUs, not GPUs or MPI.
3. A separate collector is submitted with an `afterany` dependency on the array.
   It assembles tables and plots and checks computed values against the paper,
   including when some tasks failed. Submission alone is not successful replication.

The generated `worker-plan-*.sh`, `collect-plan-*.sh`, and `plan-*.json` files
are in your output directory. The scripts contain absolute code, Python, and plan
paths. They invoke the installed Python directly, without activating a shell
virtual environment, loading modules, or installing packages on compute nodes.

## Settings to adapt

| Setting | What to choose for your server |
|---|---|
| Code location | A stable shared directory accessible at the same absolute path from submission and compute nodes. Inside this research repository, enter `reproduction/`; in the standalone archive, enter its top-level directory. |
| `--data-dir` | Absolute shared path containing the 17 pinned inputs directly, without an extra enclosing folder. |
| `--output-dir` | Writable shared path with room for task files, logs, tables, and figures. Use separate smoke and full-run directories. |
| Python/environment | Keep the committed Python 3.11.13 and dependency lock. Install using this cluster's platform; do not copy a Mac virtual environment to Linux. |
| Compiler/modules | Load your site's C/C++ compiler and OpenMP runtime before setup and submission. Sherlock's `devel gcc/14.2.0` is only an example. |
| `--partition` | Your permitted CPU queue. Omit if the scheduler has an appropriate default. |
| `--account` | Your allocation/account, if required. Omit if a default applies. |
| `--threads` | CPUs requested per array task and LightGBM thread count; default 4. Changing it changes the numerical-run fingerprint. |
| `--array-limit` | Maximum simultaneous array tasks; default 12. This is concurrency, not total array size. |
| `--memory` | Memory requested for each array task, default `32G`. This is not memory per CPU or total for the whole array. |
| `--time-limit` | Wall-clock limit per array task, default `12:00:00`, not a deadline for the entire run. |
| Collector | Currently fixed at 4 GB and 30 minutes; uses the same partition/account. Worker memory/time flags do not change these collector settings. |

At the example limits, up to 12 workers request 48 CPUs and 384 GB collectively.
These are requested resources, not measured consumption or a guarantee that every
model fits in the allocation. Start with the smoke test, then review full-run logs.
`--workers` only controls local execution; use `--array-limit` for Slurm.

## Storage and environment requirements

Code, the Python environment, its base interpreter, input data, and outputs must
all be accessible to compute nodes. The launcher normally creates `.venv/` and
`.cache/` within the code directory. `UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR`,
`UV_PYTHON_INSTALL_DIR`, `MPLCONFIGDIR`, and `XDG_CACHE_HOME` can redirect these
locations; choose shared locations and keep them stable throughout the run.
Unset an inherited `UV_PROJECT_ENVIRONMENT` if it points to another project.

Use scratch or project storage according to local quotas and retention rules.
Node-local temporary storage alone is insufficient for this implementation:
there is no automatic per-node staging or copy-back. Keep a durable data copy and
archive final outputs before scratch expires. Workers hash the files they use;
large concurrent arrays can generate substantial shared-storage traffic.

Environment setup needs package/Python download access. Workers themselves do
not download packages or data. If compute nodes have no internet, prepare the
shared environment and inputs on an allowed host first. If login-node compilation
is prohibited, use a site-approved build or interactive allocation. Completely
offline installation is not automated here; arrange cached dependencies with
local support without changing the pinned versions.

The generated scripts rely on the submission environment's exported compiler
runtime paths. If your site strips exports or requires module initialization inside
batch scripts, adapt **both** generated-script templates in `replication.py` before
submission. Include site shell initialization/module loads before their Python
command. Do not edit code or rebuild the environment while a run is active.

## Portable submission example

Replace every `/shared/...`, `YOUR_PARTITION`, and `YOUR_ACCOUNT` value below.
Load your site's compiler/OpenMP modules first, following local instructions.

```bash
cd /shared/path/to/reproduction
unset UV_PROJECT_ENVIRONMENT
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
./replicate.sh --data-dir /shared/path/to/data --check-only

./replicate.sh --slurm --smoke \
  --data-dir /shared/path/to/data --output-dir /shared/path/to/smoke \
  --partition YOUR_PARTITION --account YOUR_ACCOUNT \
  --threads 4 --array-limit 4 --memory 32G --time-limit 12:00:00
```

After the smoke report passes all 36 checks with zero missing tasks, submit the
full run with the same settings, omit `--smoke`, change the output directory to
`/shared/path/to/full`, and choose a suitable `--array-limit` (for example 12).
Save both printed job IDs. Once both jobs are submitted, you can disconnect.

## Site-specific scheduler policies

The CLI exposes partition, account, worker CPU/memory/time, and array concurrency.
It does **not** expose QoS, node constraints, reservations, custom `srun` wrappers,
containers, or a configurable collector resource request. If your site requires
these, edit the `command` (array) and `collect` (collector) argument lists in
`replication.py` before starting, or use site-approved `SBATCH_*` environment
settings where appropriate. Ensure both jobs satisfy the policy. Record your
local changes and rerun the smoke test; preserve analysis parameters, frozen
folds, input checksums, and comparison precision.

The full array needs 726 indices. A lower cluster array-size limit is not solved
by reducing `--array-limit`. The package does not automatically chunk oversized
arrays. Ask your administrator about limits; model/stage subsets are available
but produce selected-check reports and are not equivalent to the full validation.

Scheduler option semantics and exported environment behavior are documented in
[SchedMD's sbatch reference](https://slurm.schedmd.com/sbatch.html) and
[job-array guide](https://slurm.schedmd.com/job_array.html).

## Monitor, diagnose, and resume

```bash
squeue -u "$USER"
# Replace ARRAY_JOB_ID with the printed array job ID:
sacct -j ARRAY_JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Accounting availability depends on the site. Look in the output directory for
`slurm-ARRAY_JOB_ID_TASK_INDEX.log` and `validation-COLLECTOR_JOB_ID.log`.
`REPORT.md` and `validation.json` are written by the collector. Final success
requires `status: passed`, scope `all_included_analyses`, all 726 tasks present,
and no discrepancies. Smoke/subset success is a narrower claim.

| Symptom | Next step |
|---|---|
| SSH broken pipe while uploading | Repeat the same `rsync -avP` transfer; use SSH keepalives as in the Sherlock guide. Finish transfer and run `--check-only` before submission. |
| Invalid partition/account or pending policy limit | Check permitted queues, allocation, and limits with local support. |
| Missing Python/library or module on compute nodes | Check shared interpreter paths, runtime libraries, and exported environment. |
| Input hash mismatch | Repair the transfer or obtain the correct release data; do not replace the expected hash to bypass validation. |
| OUT_OF_MEMORY or TIMEOUT | Increase the relevant per-task request within site limits and rerun in the same output directory after the previous run finishes. |
| Numerical discrepancy | Preserve task JSONs, plan, logs, and environment details; investigate without changing references or tolerances. |
| Array submitted but collector submission failed | Record the array ID, resolve the collector's scheduler error, and let the array finish before resubmitting the same command. Valid tasks are reused. |

A retry submits the requested array again, but valid completed tasks are reused
when code, data, environment, parameters, thread count, and payload checks match.
Changing scheduler memory/time/concurrency alone does not change that fingerprint.
Do not overlap two runs writing into the same output directory. `--fresh` refits
all selected tasks; it is not needed for normal recovery. Code/environment changes
can invalidate cached results, so retain an unchanged checkout for active jobs.

Archive the plan, task JSONs, validation report/checks, tables, figures, and logs.
A report on scratch is not a permanent research record.
