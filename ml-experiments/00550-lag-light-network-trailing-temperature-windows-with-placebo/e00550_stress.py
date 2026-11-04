# ruff: noqa: E501
"""Injected-shift leak of the experiment 550 arms beside R-DW and L-1d: sign x change in the median forecast.

Mirrors experiment 548's `e00548_stress.py`. Points of capacity, by block age and lead group, seed 0,
five folds, 511's stress world (a 7-day block of shifted power before day 14 of each month).
"""

from pathlib import Path
from typing import Final

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DD: Final[Path] = ML_EXPERIMENTS_DATA_DIR
OUT: Final[Path] = (
    DD / "00550-lag-light-network-trailing-temperature-windows-with-placebo" / "scores"
)
ARMS: Final[dict[str, Path]] = {
    "RDW-T": DD
    / "00550-lag-light-network-trailing-temperature-windows-with-placebo/work/fits/RDW-T",
    "RDW-Tnull": DD
    / "00550-lag-light-network-trailing-temperature-windows-with-placebo/work/fits/RDW-Tnull",
    "E": DD
    / "00548-lag-free-release-candidate-assembly-with-trailing-temperature-windows/work/fits/E",
    "R-DW": DD / "00539-l1d-plus-day-and-week-lags/work/fits/R-DW",
    "L-1d": DD
    / "00511-lagless-network-refit-cadence-half-life-and-mixed-pretraining/work/fits/L-1d",
}
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]


def main() -> None:
    """Write the leak by block age and by lead group."""
    rows = []
    for arm, folder in ARMS.items():
        for m in MONTHS:
            for day in range(14, 21):
                ctrl = pl.read_parquet(folder / f"{m}_d{day}_s0.parquet").select(*KEYS, c="q50")
                for sign, tag in ((1, "p"), (-1, "n")):
                    p = folder / f"{m}_d{day}_s0_stress{tag}.parquet"
                    if not p.exists():
                        continue
                    j = ctrl.join(pl.read_parquet(p).select(*KEYS, s="q50"), on=KEYS).with_columns(
                        lead=(
                            pl.col("valid_time") - pl.col("power_fcst_init_time")
                        ).dt.total_hours(),
                        leak=sign * 100.0 * (pl.col("s") - pl.col("c")),
                    )
                    groups = {"all": pl.lit(value=True), "day1": pl.col("lead") <= 24, "days2_3": (pl.col("lead") > 24) & (pl.col("lead") <= 72), "days4_10": pl.col("lead") > 72}  # fmt: skip
                    for lab, cond in groups.items():
                        sub = j.filter(cond)
                        rows.append({"arm": arm, "month": m, "block_age_days": day - 14, "sign": sign, "band": lab, "leak": sub["leak"].mean(), "n": sub.height})  # fmt: skip
    t = pl.DataFrame(rows)
    t.write_csv(OUT / "stress_leak_long.csv")
    fold = t.group_by("arm", "month", "band").agg(pl.col("leak").mean())
    s = (
        fold.group_by("arm", "band")
        .agg(leak=pl.col("leak").mean(), lo=pl.col("leak").min(), hi=pl.col("leak").max())
        .sort("band", "arm")
    )
    s.write_csv(OUT / "stress_leak.csv")
    with pl.Config(tbl_rows=60):
        print(s.with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
