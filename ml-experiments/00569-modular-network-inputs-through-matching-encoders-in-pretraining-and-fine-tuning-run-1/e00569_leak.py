# ruff: noqa: ANN201, D103, E501
"""Injected 7-day shift leak of experiment 569's arms at cut day 14 (the formula of experiment 554's stress script).

The leak is the sign times the change in the median forecast, in points of capacity, over the rows
of the window after the cutoff, by lead band.
"""

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

D = (
    ML_EXPERIMENTS_DATA_DIR
    / "00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1"
)
MONTHS = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]
ARMS = ("M0", "B1", "B1p", "B3a", "B3ap")


def main():
    rows = []
    for arm in ARMS:
        folder = D / "work/fits" / arm
        for m in MONTHS:
            ctrl_path = folder / f"{m}_d14_s0.parquet"
            if not ctrl_path.exists():
                continue
            ctrl = pl.read_parquet(ctrl_path).select(*KEYS, c="q50")
            for sign, tag in ((1, "p"), (-1, "n")):
                p = folder / f"{m}_d14_s0_stress{tag}.parquet"
                if not p.exists():
                    continue
                j = ctrl.join(pl.read_parquet(p).select(*KEYS, s="q50"), on=KEYS).with_columns(
                    lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
                    leak=sign * 100.0 * (pl.col("s") - pl.col("c")),
                )
                for lab, cond in {"all": pl.lit(True), "day1": pl.col("lead") <= 24, "days2_3": (pl.col("lead") > 24) & (pl.col("lead") <= 72), "days4_10": pl.col("lead") > 72}.items():  # fmt: skip
                    sub = j.filter(cond)
                    rows.append({"arm": arm, "month": m, "sign": sign, "band": lab, "leak": sub["leak"].mean(), "n": sub.height})  # fmt: skip
    t = pl.DataFrame(rows)
    fold = t.group_by("arm", "month", "band").agg(pl.col("leak").mean())
    s = fold.group_by("arm", "band").agg(leak=pl.col("leak").mean(), lo=pl.col("leak").min(), hi=pl.col("leak").max(), folds=pl.len()).sort("band", "arm")  # fmt: skip
    s.write_csv(D / "leak.csv")
    with pl.Config(tbl_rows=40):
        print(s.with_columns(pl.col(pl.Float64).round(3)))


main()
