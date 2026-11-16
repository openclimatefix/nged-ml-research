# ruff: noqa: I001
"""Experiment 701, GAM side: extend the dates of experiment 377's modules to the 2026-10 origins.

Import this module first in every stage script, before `e00377_ens`, `e00377_run`, `e00377_tall`, or
`e00378_*`, with the folder of experiment 377 and `ml-experiments/00700-*` on `PYTHONPATH`. It
imports `e00701_gam_extend` (the `studies.dp_features` and `studies.dp_gam` dates), then:

- `e00377_common.LAST_RUN` and `e00377s2_common.LAST_RUN` move to 2026-10-31, so `run_dates()` holds
  944 runs (experiment 377 stores 821);
- `e00700_horizon` sets the stage 1 window to 678 half-hours (the 12:00 UTC origin plus 14 days);
- `e00377s3_stack.LAST_FOLD_MONTH` moves to 2026-10-01;
- `e00377_common.CACHE_DIR` and `e00377s2_common.CACHE_DIR` point at this experiment's extended
  feature cache, so the cache of experiment 106 is neither read nor written.

The stored `e00377_common.monthly_cuts` and `e00377_tall.monthly_cuts` read the cuts of the stored
fits, and `e00377_tall.LAST_WEEKLY_CUT` serves the weekly cadence only. No script of this folder
calls any of them for the new months; `e00701_gam_refit.py` takes the cuts from `MONTHS`.
"""

import e00377_common as c377  # ty: ignore[unresolved-import]
import e00377s2_common as c377s2  # ty: ignore[unresolved-import]
import e00377s3_stack as s3  # ty: ignore[unresolved-import]
import e00701_gam_extend as ext
import e00700_horizon  # noqa: F401  # ty: ignore[unresolved-import]  (sets the 678 half-hour window before e00377_ens and e00377_run import)

c377.LAST_RUN = ext.LAST_RUN  # ty: ignore[invalid-assignment]
c377s2.LAST_RUN = ext.LAST_RUN  # ty: ignore[invalid-assignment]
s3.LAST_FOLD_MONTH = ext.MONTHS[-1]  # ty: ignore[invalid-assignment]
c377.CACHE_DIR = ext.CACHE_DIR  # ty: ignore[invalid-assignment]
c377s2.CACHE_DIR = ext.CACHE_DIR  # ty: ignore[invalid-assignment]
