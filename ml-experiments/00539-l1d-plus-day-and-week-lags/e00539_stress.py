# ruff: noqa: E501, E741
"""Injected-shift leak of R-DW (539), R1, R9 (538) and L-1d (511): sign x change in the median forecast, points of capacity, by block age and lead band."""

from pathlib import Path

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DD = Path(str(ML_EXPERIMENTS_DATA_DIR))
W = DD / "00539-l1d-plus-day-and-week-lags/work/fits"
O = DD / "00538-l1d-plus-recent-level-and-lags/work/fits"
L = DD / "00511-lagless-network-refit-cadence-half-life-and-mixed-pretraining/work/fits/L-1d"
MONTHS = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]
rows = []
for arm, folder in (("L-1d", L), ("R1", O / "R1"), ("R9", O / "R9"), ("R-DW", W / "R-DW")):
    for m in MONTHS:
        for day in range(14, 21):
            ctrl = pl.read_parquet(folder / f"{m}_d{day}_s0.parquet").select(*KEYS, c="q50")
            for sign, tag in ((1, "p"), (-1, "n")):
                p = folder / f"{m}_d{day}_s0_stress{tag}.parquet"
                if not p.exists():
                    continue
                j = ctrl.join(pl.read_parquet(p).select(*KEYS, s="q50"), on=KEYS).with_columns(
                    lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
                    leak=sign * 100.0 * (pl.col("s") - pl.col("c")),
                )
                for lab, cond in {
                    "all": pl.lit(True),
                    "day1": pl.col("lead") <= 24,
                    "days2_3": (pl.col("lead") > 24) & (pl.col("lead") <= 72),
                    "days4_10": pl.col("lead") > 72,
                }.items():
                    rows.append(
                        {
                            "arm": arm,
                            "month": m,
                            "block_age_days": day - 14,
                            "sign": sign,
                            "band": lab,
                            "leak": j.filter(cond)["leak"].mean(),
                            "n": j.filter(cond).height,
                        }
                    )
t = pl.DataFrame(rows)
t.write_csv(DD / "00539-l1d-plus-day-and-week-lags/scores/stress_leak_long.csv")
fold = t.group_by("arm", "month", "band").agg(pl.col("leak").mean())
s = (
    fold.group_by("arm", "band")
    .agg(leak=pl.col("leak").mean(), lo=pl.col("leak").min(), hi=pl.col("leak").max())
    .sort("band", "arm")
)
s.write_csv(DD / "00539-l1d-plus-day-and-week-lags/scores/stress_leak.csv")
byage = (
    t.filter(pl.col("band") == "all")
    .group_by("arm", "block_age_days")
    .agg(leak=pl.col("leak").mean())
    .sort("arm", "block_age_days")
)
byage.write_csv(DD / "00539-l1d-plus-day-and-week-lags/scores/stress_leak_by_age.csv")
with pl.Config(tbl_rows=60):
    print(s.with_columns(pl.col(pl.Float64).round(3)))
    print(
        byage.pivot(on="arm", index="block_age_days", values="leak").with_columns(
            pl.col(pl.Float64).round(3)
        )
    )
