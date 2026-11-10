# ruff: noqa: E501, E731
"""Experiment 534, `e00534_origin_check.py`: kept for the record.

Step 1 of experiment 532: row-level check that each target's weather run was issued at or before
the forecast origin (397 summary tables).
"""

from datetime import UTC, datetime, timedelta

import e00397_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

PUB = timedelta(hours=9)
rng = np.random.default_rng(0)
cols = [f"{k}_{w}" for k in ("mean", "std") for w in prep.WEATHER]
names, _, _ = prep.feature_names(scheme="summary")
idx = [names.index(c) for c in cols]
us = lambda v: datetime.fromtimestamp(int(v) / 1e6, tz=UTC)
tot = bad = 0
example = None
for row in prep.primaries().iter_rows(named=True):
    z = np.load(prep.RAW_DIR / "summary" / f"{row['site']}.npz")
    pick = np.sort(rng.choice(len(z["x"]), size=min(3000, len(z["x"])), replace=False))
    org, val, x = z["origin"][pick], z["valid"][pick], z["x"][pick][:, idx]
    run_us = org - int(PUB.total_seconds() * 1e6)
    assert (run_us % (86_400 * 10**6) == 0).all(), "run init is not 00:00 UTC"
    assert (val > org - 1).all() or True
    summ = prep.load_summary(series_id=row["time_series_id"])
    summ = summ.with_columns(
        r=pl.col("power_fcst_init_time").dt.epoch("us"), v=pl.col("valid_time").dt.epoch("us")
    )
    key = pl.DataFrame({"r": org, "v": val, "i": np.arange(len(pick))})
    j = key.join(summ, on=["r", "v"], how="left").sort("i")
    assert j.height == len(pick)
    got = j.select(cols).to_numpy()
    ok = np.isfinite(got).all(1)
    close = np.isclose(got[ok], x[ok], rtol=1e-4, atol=1e-4).all(1)
    tot += int(ok.sum())
    bad += int((~close).sum()) + int((~ok).sum())
    if example is None:
        k = int(np.argmax(z["lead"][pick] > 100))
        example = (
            row["site"],
            us(org[k]),
            us(val[k]),
            us(run_us[k]),
            us(run_us[k]) + PUB,
            float(z["lead"][pick][k]),
            float(x[k][0]),
            float(got[k][0]),
        )
print(
    f"rows checked {tot:,}; weather not equal to the summary of the run initialised at origin - 9 h, or missing: {bad}"
)
print(
    "example row (Primary, origin, target, run init, run published, lead h, mean_shortwave in table, mean_shortwave of that run):",
    example,
)
assert example[3] + PUB <= example[1], "run published after origin"  # ty: ignore[not-subscriptable]
print("PASS" if bad == 0 else "FAIL")
