# ruff: noqa: ANN001, D103, E402, E501, I001, PT018
"""Experiment 640: leakage assertions for the 25 fit-age folds (CPU only, no fit).

For each fold `YYYY-MM-AA` (development month, fit age `AA` weeks) the script builds the fold's
fine-tune rows and pre-training rows with the power hidden from the cutoff, exactly as the training
phase does, and asserts:

- the cutoff is the month start minus `AA` weeks, and the scored window is the whole real month;
- the cleaned power is NaN from the cutoff and finite before it;
- the newest fine-tune row (valid time and origin) is older than the cutoff by the usual refit age
  (`regimes.REFIT_CUTOFF_AGE`, 3.5 h), and so is every lag input read by a row (a lag is read at or
  before its row's origin);
- the newest pre-training row (valid time and origin) is older than the cutoff by the same age;
- the B1 statistics and the placebo day structure are cut at the cutoff's grid index;
- the test rows' origins all lie inside the real month.

Usage: python e00640_check.py [--fold 2025-10-08 | --all]. `--all` runs each fold in its own process.
Run it through `gpu.sh`-style environment (`source ml-experiments/00524*/env.sh` first).
"""

import argparse
import subprocess
import sys
from datetime import UTC, datetime

_p = argparse.ArgumentParser()
_p.add_argument("--fold")
_p.add_argument("--all", action="store_true")
MINE = _p.parse_args()

if MINE.all:
    import e00640_fold as fold_lib

    failed = []
    for month in fold_lib.MONTHS:
        for age in fold_lib.AGES_WEEKS:
            name = fold_lib.fold_name(month=month, age_weeks=age)
            done = subprocess.run([sys.executable, __file__, "--fold", name], check=False)
            if done.returncode:
                failed.append(name)
    assert not failed, f"leakage check failed for {failed}"
    print("ALL 25 FOLDS PASSED")
    raise SystemExit(0)

assert MINE.fold, "give --fold or --all"
FOLD = MINE.fold
import os

os.environ["E00640_PHASE"] = "train"
sys.argv = ["e00574_fit.py", "fit", "--arm", "P5", "--months", FOLD, "--seeds", "0"]

import e00574_fit as e  # ty: ignore[unresolved-import]
import numpy as np

import e00640_fold as fold

fold.install(e, folds=[FOLD])
c, k = e.c, e.k
month, age = fold.parse(FOLD)
cutoff = fold.cutoff_of(fold=FOLD)
age_limit = c.regimes.REFIT_CUTOFF_AGE
step_us, grid_us = c.prep422.STEP_US, c.prep422.GRID_START_US


def time_of(*, idx) -> datetime:
    return datetime.fromtimestamp((int(idx) * step_us + grid_us) / 1e6, UTC)


# 1. the cutoff and the scored window
real_start = datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC)
assert (real_start - cutoff).days == 7 * age, (FOLD, cutoff)
assert c.month_start(month=FOLD) == cutoff
assert k.cutoffs(month=FOLD, cadence="d") == [cutoff]
lo, hi = k.window_of(month=FOLD, cadence="d", index=0)
assert lo == c.us(t=real_start) and hi == c.us(t=c.month_stop(month=FOLD)), (lo, hi)
assert c.month_stop(month=FOLD) > real_start
assert e.fit_until(month=FOLD) == (c.us(t=cutoff) - grid_us) // step_us, "B1 cut index"
origins = e.prep.test_origins(month=FOLD)
assert origins and all(real_start <= o < c.month_stop(month=FOLD) for o in origins), "test origins"
assert len(origins) >= 28, len(origins)

# 2. the hidden power
clean = c.clean_stack()
cut = int((c.us(t=cutoff) - grid_us) // step_us)
assert not np.isfinite(clean[:, cut:]).any(), "power from the cutoff is not hidden"
assert np.isfinite(clean[:, :cut]).any(), "all power is hidden"
last_finite = int(np.nonzero(np.isfinite(clean).any(axis=0))[0].max())
assert last_finite < cut, (last_finite, cut)

# 3. the fine-tune rows (experiment 510's rows at the fold's refit, no later cutoff)
mr = e._build_month(month=FOLD, clean=clean)
limit_us = c.us(t=cutoff - age_limit)
assert mr.valid_us.max() <= limit_us, (
    f"fine-tune row valid {datetime.fromtimestamp(mr.valid_us.max() / 1e6, UTC)} after cutoff - {age_limit}"
)
assert mr.origin_us.max() <= limit_us, "fine-tune row origin after the cut"
origin_idx = (mr.origin_us - grid_us) // step_us
assert origin_idx.max() < cut and mr.target_idx.max() < cut
# a lag is read at or before its row's origin, so its grid time is older than the origin
assert time_of(idx=origin_idx.max()) <= cutoff - age_limit
assert len(mr.y) == mr.n_base and np.isfinite(mr.y).all(), "a fine-tune target is missing"
print(
    f"{FOLD}: newest fine-tune row valid {datetime.fromtimestamp(mr.valid_us.max() / 1e6, UTC):%Y-%m-%d %H:%M}, "
    f"origin {datetime.fromtimestamp(mr.origin_us.max() / 1e6, UTC):%Y-%m-%d %H:%M}; cutoff {cutoff:%Y-%m-%d}",
    flush=True,
)

# 4. the pre-training rows
pre = e.r448.pretrain_rows(rows=mr.base, refit=cutoff, clean=clean)
latest_valid = time_of(idx=pre.target_idx.max())
latest_origin = time_of(idx=pre.origin_idx.max())
assert latest_valid <= cutoff - age_limit, (latest_valid, cutoff)
assert latest_origin <= cutoff - age_limit, (latest_origin, cutoff)
print(
    f"{FOLD}: newest pre-training row valid {latest_valid:%Y-%m-%d %H:%M}, origin {latest_origin:%Y-%m-%d %H:%M}",
    flush=True,
)

# 5. the test rows are the real month's, whatever the cutoff
test = e._build_test(month=FOLD, standardiser=mr.base.standardiser)
assert test.origin.min() >= lo and test.origin.max() < hi, "test rows outside the real month"
assert test.valid.min() > test.origin.min()
first = datetime.fromtimestamp(test.origin.min() / 1e6, UTC)
print(
    f"{FOLD}: test origins from {first:%Y-%m-%d} ({len(test.y) if hasattr(test, 'y') else len(test.origin):,} rows)",
    flush=True,
)
print(f"{FOLD}: LEAKAGE CHECK PASSED")
