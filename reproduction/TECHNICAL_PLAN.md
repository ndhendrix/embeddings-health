# Implementation and validation plan

Approved scope: one-command reproduction of analyses supported by the shared data;
PCA64, restricted-index fits and unavailable ACS targets are excluded.

1. Freeze allowed inputs, feature orders, tract membership/order, and model settings.
2. Create a small standalone Python project with a locked environment.
3. Make manual and Zenodo downloads converge on the same SHA-256 verification.
4. Run independent model/outcome, ACS, state, and decile tasks locally or as a Slurm array.
5. Assemble only computed results and render scoped plots.
6. Compare every requested included result at paper precision and fail on missing
   tasks, altered inputs, or discrepancies. Never copy reference values into results.
7. Exercise integrity/failure tests, a real-data smoke run, and clean environment
   setup. Diagnose any mismatch before calling the package a verified reproduction.
8. Complete a full Linux/Slurm run and record runtime/resource evidence before
   declaring all included analyses exactly reproduced.

## September 9 continuation

1. Preserve the supplied Sherlock diagnostic and its confirmed ACS folds. Compare
   the original and current aggregate inputs before replacing any data assets.
2. Test the remaining smoke discrepancies using the recorded Sherlock numerical
   library versions in an isolated local environment. Freeze explicit ACS state
   assignments with sample-count validation rather than recomputing tie order.
3. Rebuild only changed inputs, refresh checksums, and verify unchanged analysis
   membership and retained feature values after the Prithvi no-data cleanup.
4. Run real-data checks against the unchanged paper references. Expand validation
   only after the specific diagnostic passes; record any unresolved families.
