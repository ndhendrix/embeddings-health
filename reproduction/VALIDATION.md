# Verify a reproduction run

Run `./replicate.sh --check-only` to verify all required data checksums.
Run `./replicate.sh --smoke --output-dir outputs/smoke` for four representative
AlphaEarth tasks. A successful smoke run has 36 matching checks and no missing
tasks. It does not check every model or outcome.

For the complete included analysis set, run `./replicate.sh` or follow
[SLURM.md](SLURM.md). A successful complete run has `status: passed` and
`scope: all_included_analyses` in `validation.json`, with all 726 tasks present
and no discrepancies. Inspect `REPORT.md` and `validation_checks.csv` for details.

The comparison contract is exact equality at the paper's reported precision:
four decimals for performance tables, three for ACS summary formatting, and
exact sample counts. Raw fitted values and fold scores remain in task JSONs.
Reference values are comparison targets, never substitutes for computation.
Figure files reproduce the included numerical content, not the original image bytes.

This document defines acceptance criteria; it does not certify a particular run.
Retain the report, plan, logs, and task results as evidence for your execution.
See README.md for excluded analyses and PROVENANCE.md for data lineage.
