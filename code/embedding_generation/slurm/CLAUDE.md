# Slurm directory

This directory contains job scripts for the Prithvi/OlmoEarth embedding
pipeline on Stanford's Sherlock HPC cluster.

**Before writing or editing any Slurm-related code, read `README.md` in this
directory.** It documents hard-won operational lessons including:

- Module loading gotchas (load `devel` before `physics gdal/3.10.2`)
- Login-shell vs. batch-job environment differences
- How to run one-off scripts on a compute node with `srun` instead of `sbatch`
- Walltime estimation logic and the resubmit chain
- Storage tiers and setup overhead per job
- COG recovery procedure for jobs killed during finalisation
