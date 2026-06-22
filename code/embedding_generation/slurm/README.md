# Slurm Pipeline — Sherlock HPC Notes

Operational notes from running the Prithvi embedding pipeline on Stanford's
Sherlock cluster. The official docs are at https://www.sherlock.stanford.edu/docs/
and are authoritative; this file captures project-specific lessons and gotchas.

---

## Module loading

### Categories must be loaded first

Sherlock modules are grouped into categories (`devel`, `math`, `physics`,
`chemistry`, `biology`, etc.). Only `devel` and `math` are loaded in the
default login environment. Most science/domain modules live under a category
that must be loaded first:

```bash
ml spider gdal            # find the right version and its prerequisites
ml physics gdal/3.10.2    # load category then module
```

Always use `ml spider <name>` rather than `ml av` — spider searches all
categories including ones not currently loaded.

### Hidden GCC dependencies

Several modules list `gcc/12.4.0` (from `devel`) as a prerequisite but do
not declare it explicitly in the module file. The load silently fails if
`devel` is not already active. Example: `gdal/3.10.2` requires `devel` to
be loaded first, even though `ml spider gdal/3.10.2` only lists `physics`
as a prerequisite.

**Lesson**: If a module load succeeds from your login shell but fails in a
batch job, load `devel` explicitly before it:

```bash
module load devel
module load physics gdal/3.10.2
```

### Login shell vs. batch job environment

Your interactive login shell has `devel` and `math` pre-loaded. A batch job
started with `sbatch --wrap="bash script.sh"` starts in a **clean
environment** without those categories. This caused `module load physics
gdal/3.10.2` to fail silently (stderr was redirected to `/dev/null`) in the
resubmit chain, leaving `gdalinfo` unavailable and all walltime estimates
defaulting to the fallback value.

Any submit script intended to run both interactively AND as a batch job must
load its own prerequisites explicitly.

---

## Slurm array jobs

### Walltime is per-task, not per-array

`--time=06:00:00` on an array job gives each task 6 hours from when **that
task starts**, not from when the array was submitted. Tasks in the same array
that start at different times (as GPUs become available) each get their own
independent 6-hour clock.

### Each task gets two IDs

An array task has both an **array-scoped ID** (`30490799_65`) and a unique
**individual job ID** (`30522587`). Log files written with `%A_%a` use the
array-scoped form; `$SLURM_JOB_ID` inside the running task is the individual
ID. Staging directories created by the sbatch use `$SLURM_JOB_ID`, so they
won't match the log filename.

### The resubmit chain

`submit_prithvi_all_states.sh` submits one array per walltime tier, then
schedules a resubmit job with `--dependency=afterany:<all_tier_job_ids>`.
The resubmit job runs the submit script again as a batch job, which re-tiers
any still-incomplete states and submits new arrays. Key properties:

- The resubmit job itself runs on `--partition=normal` with `--time=00:10:00`.
- It has `--export=ALL` so `DATA_DIR`, `FINAL_OUT_DIR`, `YEAR`, and
  `REPO_DIR` are inherited from the original submission environment.
- Cancel the resubmit job to stop the chain: `scancel <resubmit_job_id>`.
  The running inference tasks continue until they finish or hit their limits.
- New arrays submitted by the resubmit chain get fresh job IDs, so log files
  are renamed (e.g., `prithvi_embed_30530000_65.out`).

---

## Storage performance

### Setup overhead

Node-local SSD (`$L_SCRATCH`) is fast; Lustre (`$SCRATCH`) and NFS (`$HOME`)
are not. Our sbatch stages input TIFs from `$SCRATCH` to `$L_SCRATCH` before
running inference. Depending on file sizes and network load:

- A fresh venv build on a new node takes 30–60 minutes (all packages
  compiled from scratch or installed via wheel). The venv is cached at
  `$SCRATCH/embeddings-health/cache/venv-3.11` and reused on subsequent
  runs on the same node family — but only if that exact path is accessible.
- Staging large input TIFs (3 × seasonal composites) can take 30–90 minutes
  under high Lustre load.

In practice, **total setup overhead per job is 1–5 hours**, which must be
included in walltime estimates. The walltime tiers in
`submit_prithvi_all_states.sh` include a floor to account for this.

### $HOME is NFS — keep it small

`$HOME` has a 15 GB quota. Job logs, Python caches, venvs, and output data
all go to `$SCRATCH`. Override cache dirs in batch jobs:

```bash
export UV_CACHE_DIR="$SCRATCH/embeddings-health/cache/uv"
export HF_HOME="$SCRATCH/embeddings-health/cache/hf"
export UV_PROJECT_ENVIRONMENT="$SCRATCH/embeddings-health/cache/venv-3.11"
```

### $SCRATCH purge

Files on Lustre (`$SCRATCH`, `$GROUP_SCRATCH`) are deleted after 90 days of
inactivity. The timer resets only on real writes — `touch`, `chmod`, and
rename do **not** reset it. Move completed outputs to `$OAK` or another
backed-up location.

---

## TIFF size limits

Classic TIFF has a 4 GB file size limit. Large states (TX at 192 dims,
NY/CA/OR at 1024 dims) exceed this. Use `BIGTIFF="IF_SAFER"` in rasterio
open/copy calls; this activates BigTIFF automatically only when needed:

```python
rasterio.open(path, "w", ..., BIGTIFF="IF_SAFER")
rio_copy(src, dst, ..., BIGTIFF="IF_SAFER")
```

`IF_SAFER` is preferable to `YES` because it leaves smaller files as
standard TIFFs (broader tool compatibility).

---

## Recovering a job killed during COG finalisation

The COG write flow is: inference → write flat `.tmp.tif` → build overviews
in-place on `.tmp.tif` → `rio_copy` to final `.tif` → delete `.tmp.tif`.

If a job is killed during the overview build or copy step, the `.tmp.tif`
survives and contains the full inference output. Recovery: submit a small
CPU-only job (`-p normal`, no GPU, `--mem=64G`, `--time=03:00:00`) that
calls `_add_overviews_and_copy_as_cog` directly on the `.tmp.tif`:

```python
import rasterio
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy

with rasterio.open(tmp_path, "r+") as src:
    src.build_overviews([2, 4, 8, 16, 32], Resampling.average)
    src.update_tags(ns="rio_overview", resampling="average")

rio_copy(tmp_path, final_path, driver="GTiff", compress="lzw",
         copy_src_overviews=True, tiled=True,
         blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
```

---

## Walltime estimation

`submit_prithvi_all_states.sh` estimates chip count from the input TIF
dimensions via `gdalinfo`, then maps to a walltime tier. Two failure modes:

1. **gdalinfo unavailable**: if `module load physics gdal/3.10.2` fails
   (e.g., in a resubmit batch job without `devel` loaded), the function
   returns 0 chips. The `chips=0` tier is set conservatively (3h) and the
   script falls back to estimating from the `300M-TL` checkpoint `.npy`
   file size if available.

2. **Tier too low**: all tiers include a floor to absorb 1–5h of per-job
   setup overhead. When in doubt, a job that times out resumes from its
   checkpoint, so false-low estimates are recoverable.

Current tiers (see `chips_to_walltime` in the submit script for latest):

| Chips (gdalinfo ceil estimate) | Walltime |
|---:|:---|
| 0 (unknown) | 3h |
| ≤ 500 | 1h30m |
| ≤ 2,000 | 2h |
| ≤ 8,000 | 4h |
| ≤ 20,000 | 8h |
| > 20,000 | 12h |

---

## Useful commands

```bash
squeue --me                          # your jobs; PD=pending R=running
squeue --me -o "%.18i %.12j %.8T %.10M %.10l %R"  # with time and limit
seff <jobid>                         # CPU/mem efficiency post-mortem
sh_quota                             # your storage usage
sh_quota -g                          # group storage usage
ml spider <name>                     # find module and its prerequisites
sh_dev -g 1                          # quick interactive GPU shell
salloc -p gpu --gpus 1               # interactive GPU allocation
scancel <jobid>                      # cancel a job or resubmit chain
```
