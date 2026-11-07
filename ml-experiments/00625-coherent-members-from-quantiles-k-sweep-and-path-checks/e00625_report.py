"""Experiment 625: contrasts with month-block bootstrap intervals, printed as a report."""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00625-coherent-members-from-quantiles-k-sweep-and-path-checks"
    / "work"
)
ALPHA: Final[float] = 0.0167
N_BOOT: Final[int] = 20000
RNG = np.random.default_rng(625)


def month(col: str = "power_fcst_init_time") -> pl.Expr:
    return pl.col(col).dt.strftime("%Y-%m").alias("month")


def boot(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Return the relative difference of mean(a) against mean(b) with a month-block interval."""
    n = len(a)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    r = a[idx].mean(1) / b[idx].mean(1) - 1
    return float(a.mean() / b.mean() - 1), *map(float, np.quantile(r, [ALPHA / 2, 1 - ALPHA / 2]))


def monthly(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return df.with_columns(month()).group_by(["month", "arm", "k"]).agg(pl.col(col).mean())


def contrast(m: pl.DataFrame, *, arm_a: str, k_a: int, arm_b: str, k_b: int, col: str) -> str:
    a = m.filter((pl.col("arm") == arm_a) & (pl.col("k") == k_a)).sort("month")
    b = m.filter((pl.col("arm") == arm_b) & (pl.col("k") == k_b)).sort("month")
    j = a.join(b, on="month", suffix="_b")
    est, lo, hi = boot(j[col].to_numpy(), j[f"{col}_b"].to_numpy())
    return f"{arm_a}@{k_a} vs {arm_b}@{k_b}: {est * 100:+.1f}% ({lo * 100:+.1f}% to {hi * 100:+.1f}%), abs {j[col].mean():.3f} vs {j[f'{col}_b'].mean():.3f}, {len(j)} months"


def main() -> None:
    coh = pl.read_parquet(W / "coherence.parquet")
    print("runs", coh["power_fcst_init_time"].n_unique())
    base = coh.filter((pl.col("window") == "days_1_10") & (pl.col("slice") == "all"))
    # per-run family score (spatial: mean over half-hours already; label = family)
    sp = (
        base.filter(pl.col("block") == "spatial")
        .group_by(["power_fcst_init_time", "arm", "k", "label"])
        .agg(pl.col("variogram").mean())
    )
    groups = {
        "farms": sp.filter(pl.col("label").is_in(["wind", "solar"]))
        .group_by(["power_fcst_init_time", "arm", "k"])
        .agg(pl.col("variogram").mean()),
        "demand": sp.filter(pl.col("label") == "demand").select(
            "power_fcst_init_time", "arm", "k", "variogram"
        ),
        "wind": sp.filter(pl.col("label") == "wind").select(
            "power_fcst_init_time", "arm", "k", "variogram"
        ),
        "solar": sp.filter(pl.col("label") == "solar").select(
            "power_fcst_init_time", "arm", "k", "variogram"
        ),
    }
    joint = base.filter(pl.col("block") == "joint").select(
        "power_fcst_init_time", "arm", "k", "variogram"
    )
    groups["joint"] = joint
    tmp = (
        base.filter(pl.col("block") == "temporal")
        .group_by(["power_fcst_init_time", "arm", "k"])
        .agg(pl.col("variogram").mean())
    )
    groups["temporal"] = tmp
    for name, df in groups.items():
        m = monthly(df, "variogram")
        print(
            f"\n== variogram, spatial/{name} (days 1-10)"
            if name not in ("joint", "temporal")
            else f"\n== variogram, {name} (days 1-10)"
        )
        for arm in ("J2", "J3", "J5", "SSR", "J6"):
            if name == "demand" and arm == "J6":
                continue
            print(" ", contrast(m, arm_a=arm, k_a=51, arm_b="J0", k_b=51, col="variogram"))
        print("  K sweep against K=51 of the same arm:")
        for arm in ("J3", "SSR", "J2", "J5"):
            for k in (5, 10, 20):
                print("   ", contrast(m, arm_a=arm, k_a=k, arm_b=arm, k_b=51, col="variogram"))
    # J3 vs J6 on farms
    m = monthly(groups["farms"], "variogram")
    print("\nMW vs SS farms:", contrast(m, arm_a="J6", k_a=51, arm_b="J3", k_b=51, col="variogram"))

    crps = pl.read_parquet(W / "crps.parquet").filter(pl.col("window") == "days_1_10")
    print("\n== CRPS of K paths against the direct 51-value CRPS (days 1-10)")
    for fam in ("demand", "wind", "solar"):
        c = crps.filter(pl.col("family") == fam).with_columns(month())
        for arm in ("J0", "J3", "SSR", "J2", "J5", "J6"):
            line = []
            for k in (5, 10, 20, 51):
                d = (
                    c.filter((pl.col("arm") == arm) & (pl.col("k") == k))
                    .group_by("month")
                    .agg(pl.col("crps").mean(), pl.col("direct").mean())
                    .sort("month")
                )
                if d.height:
                    est, lo, hi = boot(d["crps"].to_numpy(), d["direct"].to_numpy())
                    line.append(f"K{k} {est * 100:+.2f}% ({lo * 100:+.2f},{hi * 100:+.2f})")
            if line:
                print(f"  {fam} {arm}: " + "; ".join(line))
    cov = pl.read_parquet(W / "coverage.parquet").filter(pl.col("window") == "days_1_10")
    print("\n== coverage of member spread (days 1-10), mean over runs")
    print(
        cov.filter(pl.col("arm").is_in(["J0", "J3", "J2", "SSR"]) & pl.col("k").is_in([5, 10, 51]))
        .group_by(["family", "arm", "k", "level"])
        .agg(pl.col("coverage").mean())
        .sort(["family", "level", "arm", "k"])
        .pivot(on="level", index=["family", "arm", "k"], values="coverage")
        .to_pandas()
        .round(3)
        .to_string()
        if False
        else cov.filter(
            pl.col("arm").is_in(["J0", "J3", "J2", "SSR"]) & pl.col("k").is_in([5, 10, 51])
        )
        .group_by(["family", "arm", "k", "level"])
        .agg(pl.col("coverage").mean())
        .sort(["family", "level", "arm", "k"])
    )
    sky = pl.read_parquet(W / "sky.parquet")
    thr = sky.filter(pl.col("arm") == "OBS")["spread"].quantile(0.75)
    days = sky.filter((pl.col("arm") == "OBS") & (pl.col("spread") >= thr)).select(
        "power_fcst_init_time", "day"
    )
    top = sky.join(days, on=["power_fcst_init_time", "day"]).with_columns(month())
    print(
        f"\n== opposite-sky share on top-quarter-spread days (spread >= {thr:.2f}), {days.height} run-days"
    )
    for arm in ("OBS", "J0", "J2", "J3", "SSR", "J5", "J6"):
        for k in (0,) if arm == "OBS" else (10, 51):
            d = (
                top.filter((pl.col("arm") == arm) & (pl.col("k") == k))
                .group_by("month")
                .agg(pl.col("opposite").sum(), pl.col("total").sum())
            )
            sh = (d["opposite"] / d["total"]).to_numpy()
            idx = RNG.integers(0, len(sh), size=(N_BOOT, len(sh)))
            q = np.quantile(sh[idx].mean(1), [0.025, 0.975])
            print(f"  {arm}@{k}: {sh.mean() * 100:.2f}% ({q[0] * 100:.2f}, {q[1] * 100:.2f})")
    acf = pl.read_parquet(W / "acf.parquet")
    print("\n== lag-1 autocorrelation of step changes")
    print(
        acf.filter(pl.col("k").is_in([0, 10, 51]))
        .group_by(["window", "family", "arm", "k"])
        .agg(pl.col("acf1").mean())
        .sort(["window", "family", "arm", "k"])
    )


if __name__ == "__main__":
    main()
