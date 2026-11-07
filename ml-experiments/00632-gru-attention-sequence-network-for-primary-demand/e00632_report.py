# ruff: noqa: E501, D103, PERF401
"""Experiment 632: tables of the results (CRPS by day, tails, contrasts with month-block intervals).

Usage: python e00632_report.py
"""

import numpy as np
import polars as pl
from e00632_score import EXPORT, MONTHS, ROOT, arm_forecast, score
from studies.bootstrap import bootstrap_month_totals

SEEDS = (0, 1, 2)
ARMS = [
    "seq",
    "P2-pre",
    "K3-pre",
    "no_attn",
    "no_gru",
    "seq_nb",
    "seq_ls",
    "seq_h3w",
    "seq_h3w_ly",
    "P2-hist6",
    "climatology",
]
EXPORT_ARMS = ["K1", "Candidate", "K3"]
COUNTED = [("seq", "P2-pre"), ("seq", "P2-hist6"), ("P2-pre", "K3-pre"), ("seq", "K3-pre")]
LEVEL = 100.0 - 5.0 / len(COUNTED)


def present(arm: str) -> bool:
    return all(list((ROOT / "predictions" / arm).glob(f"{m}_s*.parquet")) for m in MONTHS)


def scored(arm: str, seeds: tuple[int, ...] | None = None) -> pl.DataFrame:
    parts = []
    for m in MONTHS:
        parts.append(score(arm_df=arm_forecast(arm=arm, month=m, seeds=seeds), month=m))
    return pl.concat(parts).with_columns(arm=pl.lit(arm))


def export_scored(arm: str) -> pl.DataFrame:
    parts = []
    for m in MONTHS:
        a = pl.read_parquet(EXPORT / f"{arm}_{m}.parquet").select(
            "time_series_id", "power_fcst_init_time", "valid_time", "lead_hours",
            "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
        )  # fmt: skip
        parts.append(score(arm_df=a, month=m))
    return pl.concat(parts).with_columns(arm=pl.lit(f"{arm} (export, ENS weather)"))


def by_day(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("arm", "group", "day")
        .agg(
            pl.col("crps").mean(),
            pl.col("mae").mean(),
            pl.col("hit").mean(),
            pl.col("above90").mean(),
        )
        .sort("group", "arm", "day")
    )


def contrast(
    treatment: str, reference: str, group: str, lo_day: int = 2
) -> tuple[float, float, float]:
    sums: list = []
    counts = np.zeros(0)
    for s in SEEDS:
        t, r = scored(treatment, (s,)), scored(reference, (s,))
        t = t.filter((pl.col("group") == group) & (pl.col("day") >= lo_day))
        r = r.filter((pl.col("group") == group) & (pl.col("day") >= lo_day))
        j = t.join(
            r,
            on=["time_series_id", "power_fcst_init_time", "valid_time"],
            suffix="_r",
            validate="1:1",
        )
        assert j.height == t.height == r.height
        g = (
            j.group_by("month")
            .agg((pl.col("crps") - pl.col("crps_r")).sum().alias("d"), pl.len().alias("n"))
            .sort("month")
        )
        sums.append(g["d"].to_numpy())
        counts = g["n"].to_numpy()
    lo, hi = bootstrap_month_totals(month_sums=np.stack(sums), month_counts=counts, level=LEVEL)
    return float(np.stack(sums).sum(axis=1).mean() / counts.sum()), lo, hi


def main() -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(12)
    pl.Config.set_fmt_str_lengths(40)
    frames = [scored(a) for a in ARMS if present(a)] + [export_scored(a) for a in EXPORT_ARMS]
    allf = pl.concat(frames, how="diagonal_relaxed")
    t = by_day(allf).filter(pl.col("day") <= 7)
    out = ROOT / "tables"
    out.mkdir(exist_ok=True)
    t.write_csv(out / "crps_by_day.csv")
    for g in ("eight MVA Primaries", "six MW Primaries"):
        print(g)
        print(
            t.filter(pl.col("group") == g)
            .pivot(on="arm", index="day", values="crps")
            .with_columns(pl.exclude("day").round(2))
        )
        print(
            t.filter(pl.col("group") == g)
            .pivot(on="arm", index="day", values="hit")
            .with_columns(pl.exclude("day").round(3)),
            "coverage q05-q95",
        )
    print("pooled days 2-7 CRPS / MAE / above-q90 share")
    pooled = (
        allf.filter(pl.col("day") >= 2)
        .group_by("arm", "group")
        .agg(
            pl.col("crps").mean().round(3),
            pl.col("mae").mean().round(3),
            pl.col("above90").mean().round(3),
            pl.col("hit").mean().round(3),
        )
        .sort("group", "arm")
    )
    print(pooled)
    pooled.write_csv(out / "pooled_days2_7.csv")
    print(
        f"contrasts, mean paired CRPS difference (treatment - reference), days 2-7, {LEVEL:.2f}% month-block interval"
    )
    rows = []
    for tr, rf in COUNTED:
        if not (present(tr) and present(rf)):
            continue
        for g in ("eight MVA Primaries", "six MW Primaries"):
            diff, lo, hi = contrast(tr, rf, g)
            rows.append(
                {
                    "treatment": tr,
                    "reference": rf,
                    "group": g,
                    "diff": round(diff, 3),
                    "lo": round(lo, 3),
                    "hi": round(hi, 3),
                }
            )
    c = pl.DataFrame(rows)
    print(c)
    c.write_csv(out / "contrasts.csv")


if __name__ == "__main__":
    main()
