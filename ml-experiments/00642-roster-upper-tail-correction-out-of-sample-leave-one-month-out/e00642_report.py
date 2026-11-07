"""Experiment 642: tables, intervals, and the pre-registered bar from the scored per-row frames."""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00630_common import NETWORKS, bootstrap_ratio, day_table  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00642-roster-upper-tail-correction-out-of-sample-leave-one-month-out"
)
ARMS: Final[tuple[str, ...]] = ("base", "S", "B", "C", "Q", "M")
SHORT_HISTORY: Final[tuple[str, ...]] = ("D15", "D18")
IND: Final[tuple[str, ...]] = ("a90", "a95", "b10", "b05")
N_BOOT: Final[int] = 20_000


def value_cols() -> list[str]:
    """All per-arm columns that are averaged."""
    return [f"{i}_{a}" for a in ARMS for i in (*IND, "crps", "tw")]


def means(*, df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Mean of every per-arm column per key, with the CRPS and twCRPS ratios to the base arm."""
    cols = value_cols()
    agg = (
        df.group_by(keys).agg(
            pl.len().alias("n"), *[pl.col(c).cast(pl.Float64).mean() for c in cols]
        )
        if keys
        else df.select(pl.len().alias("n"), *[pl.col(c).cast(pl.Float64).mean() for c in cols])
    )
    return agg.with_columns(
        *[(pl.col(f"crps_{a}") / pl.col("crps_base")).alias(f"crps_ratio_{a}") for a in ARMS],
        *[(pl.col(f"tw_{a}") / pl.col("tw_base")).alias(f"tw_ratio_{a}") for a in ARMS],
    )


def month_block_interval(*, df: pl.DataFrame, n_boot: int = N_BOOT) -> pl.DataFrame:
    """Month-block bootstrap of the pooled q90 and q95 exceedance and the CRPS ratio per arm."""
    months = sorted(df["month"].unique().to_list())
    sums = {
        m: df.filter(pl.col("month") == m)
        .select(pl.len().alias("n"), *[pl.col(c).cast(pl.Float64).sum() for c in value_cols()])
        .row(0, named=True)
        for m in months
    }
    rng = np.random.default_rng(642)
    pick = rng.integers(0, len(months), size=(n_boot, len(months)))
    rows = []
    for a in ARMS:
        stats = {}
        for name in (f"a90_{a}", f"a95_{a}", f"crps_{a}"):
            arr = np.array([sums[m][name] for m in months])
            n = np.array([sums[m]["n"] for m in months])
            stats[name] = arr[pick].sum(axis=1) / n[pick].sum(axis=1)
        base = np.array([sums[m]["crps_base"] for m in months])[pick].sum(axis=1)
        crps_arr = np.array([sums[m][f"crps_{a}"] for m in months])[pick].sum(axis=1)
        ratio = crps_arr / base
        rows.append(
            {
                "arm": a,
                **{
                    f"{k}_{s}": float(np.quantile(v, q))
                    for k, v in stats.items()
                    if k.startswith("a")
                    for s, q in (("mlo", 0.025), ("mhi", 0.975))
                },
                "crps_ratio_mlo": float(np.quantile(ratio, 0.025)),
                "crps_ratio_mhi": float(np.quantile(ratio, 0.975)),
            }
        )
    return pl.DataFrame(rows)


def day_interval(*, df: pl.DataFrame) -> pl.DataFrame:
    """Day-resampling intervals (within month) of the pooled exceedance and CRPS ratios per arm."""
    cols = value_cols()
    days = day_table(df=df, keys=[], values=cols)
    ratios = {f"crps_ratio_{a}": (f"crps_{a}", "crps_base") for a in ARMS if a != "base"}
    return bootstrap_ratio(days=days, keys=[], values=cols, ratios=ratios, n_boot=N_BOOT, seed=642)


def long_form(*, wide: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """One row per key and arm."""
    parts = [
        wide.select(
            *keys,
            "n",
            pl.lit(a).alias("arm"),
            pl.col(f"a90_{a}").alias("a90"),
            pl.col(f"a95_{a}").alias("a95"),
            pl.col(f"b10_{a}").alias("b10"),
            pl.col(f"b05_{a}").alias("b05"),
            pl.col(f"crps_{a}").alias("crps"),
            pl.col(f"crps_ratio_{a}").alias("crps_ratio"),
            pl.col(f"tw_ratio_{a}").alias("tw_ratio"),
        )
        for a in ARMS
    ]
    return pl.concat(parts)


def bar(*, pooled: pl.DataFrame, by_month: pl.DataFrame) -> pl.DataFrame:
    """Evaluate the pre-registered bar (a)-(d) for each arm of one network."""
    worst = by_month.group_by("arm").agg(
        pl.col("a90").min().alias("a90_min_month"), pl.col("a90").max().alias("a90_max_month")
    )
    out = pooled.join(worst, on="arm").with_columns(
        a_ok=pl.col("a90").is_between(0.085, 0.115),
        b_ok=pl.col("a95").is_between(0.035, 0.065),
        c_ok=pl.col("crps_ratio") <= 1.01,
        d_ok=(pl.col("a90_min_month") >= 0.07) & (pl.col("a90_max_month") <= 0.13),
        dist=(pl.col("a90") - 0.10).abs() + (pl.col("a95") - 0.05).abs(),
    )
    return out.with_columns(
        passes=pl.col("a_ok") & pl.col("b_ok") & pl.col("c_ok") & pl.col("d_ok")
    )


def main() -> None:
    """Write every table and print the bar."""
    OUT.mkdir(parents=True, exist_ok=True)
    tab = OUT / "tables"
    tab.mkdir(exist_ok=True)
    bars = []
    for network in NETWORKS:
        df = pl.read_parquet(OUT / f"scored_{network}.parquet")
        ef = df.filter(pl.col("event_free"))
        ef14 = ef.filter(~pl.col("primary_name").is_in(SHORT_HISTORY))
        results = {}
        for tag, sub in (("ef16", ef), ("ef14", ef14), ("all16", df)):
            pooled = long_form(wide=means(df=sub, keys=[]), keys=[])
            results[tag] = pooled
            pooled.with_columns(subset=pl.lit(tag)).write_csv(tab / f"pooled_{tag}_{network}.csv")
        by_month = long_form(wide=means(df=ef, keys=["month"]), keys=["month"])
        by_month.write_csv(tab / f"by_month_{network}.csv")
        for name, keys in (
            ("band", ["band"]),
            ("group", ["group"]),
            ("primary", ["primary_name"]),
            ("cold_night", ["cold_night"]),
            ("block", ["block"]),
        ):
            long_form(wide=means(df=ef, keys=keys), keys=keys).write_csv(
                tab / f"by_{name}_{network}.csv"
            )
        day_iv = day_interval(df=ef)
        day_iv.write_csv(tab / f"day_intervals_{network}.csv")
        month_iv = month_block_interval(df=ef)
        month_iv.write_csv(tab / f"month_block_intervals_{network}.csv")
        b = bar(pooled=results["ef16"], by_month=by_month).with_columns(
            network=pl.lit(network), basis=pl.lit("16 Primaries")
        )
        b14 = bar(
            pooled=results["ef14"],
            by_month=long_form(wide=means(df=ef14, keys=["month"]), keys=["month"]),
        ).with_columns(network=pl.lit(network), basis=pl.lit("14 Primaries"))
        bars += [b, b14]
        corrected = (
            df.group_by("month")
            .agg(pl.col("corrected_C").mean().alias("C"), pl.col("corrected_M").mean().alias("M"))
            .sort("month")
        )
        corrected.write_csv(tab / f"corrected_fraction_by_month_{network}.csv")
        df.group_by("band").agg(
            pl.col("corrected_C").mean().alias("C"), pl.col("corrected_M").mean().alias("M")
        ).sort("band").write_csv(tab / f"corrected_fraction_by_band_{network}.csv")
        print(f"\n== {network}: corrected fraction by month")
        print(corrected)
        with pl.Config(tbl_rows=60, tbl_cols=30, fmt_float="mixed"):
            print(f"== {network}: pooled event-free, 16 Primaries")
            print(
                results["ef16"].select(
                    "arm",
                    "n",
                    pl.col("a90", "a95", "b10", "b05", "crps_ratio", "tw_ratio").round(4),
                )
            )
            print("by month q90:")
            print(
                by_month.pivot(on="arm", index="month", values="a90")
                .sort("month")
                .with_columns(pl.exclude("month").round(3))
            )
            print("by month q95:")
            print(
                by_month.pivot(on="arm", index="month", values="a95")
                .sort("month")
                .with_columns(pl.exclude("month").round(3))
            )
    allbar = pl.concat(bars)
    allbar.write_csv(tab / "bar.csv")
    with pl.Config(tbl_rows=80, tbl_cols=30, fmt_float="mixed"):
        print(
            allbar.select(
                "network",
                "basis",
                "arm",
                pl.col("a90", "a95", "crps_ratio", "a90_min_month", "a90_max_month").round(4),
                "a_ok",
                "b_ok",
                "c_ok",
                "d_ok",
                "passes",
            )
        )


if __name__ == "__main__":
    main()
