"""Part C of experiment 395: does ensemble spread help calibration when weather uncertainty is high?

Compares experiment 389's `S1M_raw` (monthly refit with the mean and standard deviation of the
ensemble `temperature_2m` and `shortwave`) with `S0M_raw` (the plain monthly member) on bins that
experiment 394 does not cover: the top decile of the ensemble temperature spread, and the
regime-uncertainty bins (the share of the 51 members whose daily mean temperature on the week's
coldest day is below 5 degrees C). Both are known at the forecast origin. Nothing is refitted.

    ... e00395_part_c.py build | report
"""
# ruff: noqa: E501

import sys
import warnings
from pathlib import Path
from typing import Final

import e00395_build as b395  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = b395.WORK
E389: Final[Path] = (
    DATA / "00389-xgboost-tail-calibration-and-ensemble-spread-columns" / "work" / "arms"
)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
HEADLINE: Final[str] = "lead_days_1_10"
COLD_C: Final[float] = 5.0
HALF_HOUR: Final[np.timedelta64] = np.timedelta64(30, "m")
STEPS_PER_DAY: Final[int] = 48
STORES: Final[dict[str, Path]] = {
    **dict.fromkeys(
        ("D5", "D9", "D10", "D11", "D14", "D16", "D18"),
        DATA
        / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
        / "work"
        / "ens_store",
    ),
    **dict.fromkeys(
        ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"),
        DATA / "00374-mva-fold-stage-two-at-eight-primaries" / "work" / "ens_store",
    ),
    "D15": DATA / "00377-demand-v1-first-run-feasibility" / "work" / "d15" / "ens_store",
}
REGIME_BINS: Final[list[str]] = [
    "all warm (0)",
    "0 to 25%",
    "25 to 75% (split)",
    "75 to 100%",
    "all cold (1)",
]


def ensemble_features(*, label: str) -> pl.DataFrame:
    """Return per run and step: the temperature spread, and per run the cold-member share."""
    with np.load(STORES[label] / f"{label}.npz") as stored:
        origins = stored["origins"]
        temperature = stored["temperature"].astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = np.nanstd(temperature, axis=1)
        week = temperature[:, :, : 7 * STEPS_PER_DAY].reshape(
            len(origins), temperature.shape[1], 7, STEPS_PER_DAY
        )
        daily = np.nanmean(week, axis=3)  # run, member, day
        coldest = np.nanargmin(np.nanmean(daily, axis=1), axis=1)
        cold_share = np.array(
            [np.nanmean(daily[i, :, coldest[i]] < COLD_C) for i in range(len(origins))]
        )
    n_runs, n_steps = spread.shape
    return pl.DataFrame(
        {
            "power_fcst_init_time": np.repeat(origins, n_steps),
            "step": np.tile(np.arange(n_steps), n_runs),
            "temp_std": spread.ravel(),
            "cold_share": np.repeat(cold_share, n_steps),
        }
    ).with_columns(pl.col("power_fcst_init_time").dt.replace_time_zone("UTC"))


def build() -> None:
    """Join the spread and regime features to the S0M and S1M scored rows and write them."""
    from e00377_adapter import series_table  # ty: ignore[unresolved-import]

    series = series_table()
    table = pl.read_parquet(WORK / "b_slice_table.parquet")
    feats = []
    for label, sid in zip(
        series["label"].to_list(), series["time_series_id"].to_list(), strict=True
    ):
        feats.append(
            ensemble_features(label=label).with_columns(
                time_series_id=pl.lit(sid, dtype=series.schema["time_series_id"])
            )
        )
        print("features", label, flush=True)
    features = pl.concat(feats)
    for name in ("S0M_raw", "S1M_raw"):
        rows = pl.read_parquet(WORK / f"b_rows_{name}.parquet")
        stored = pl.read_parquet(E389 / f"{name}.parquet").select(
            *KEYS, width90=(pl.col("q95") - pl.col("q05")) * 100
        )
        out = (
            rows.join(stored, on=KEYS, how="left", validate="1:1")
            .with_columns(
                step=(
                    (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() // 30
                )
                - 1
            )
            .join(
                features,
                on=["time_series_id", "power_fcst_init_time", "step"],
                how="left",
                validate="m:1",
            )
        )
        out.write_parquet(WORK / f"c_rows_{name}.parquet")
        print(name, out.height, out["temp_std"].is_null().sum(), flush=True)
    print(table.height)


def with_bins(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return the rows with the lead band, spread tercile, top-decile flag, and regime bin."""
    band = (
        pl.when(pl.col("lead_hours") < 72).then(pl.lit("days 1-3"))
        .when(pl.col("lead_hours") < 168).then(pl.lit("days 4-7"))
        .otherwise(pl.lit("days 8-10"))
    )  # fmt: skip
    rows = rows.with_columns(band=band)
    rank = pl.col("temp_std").rank("average").over("time_series_id", "band") / pl.len().over(
        "time_series_id", "band"
    )
    return rows.with_columns(
        spread_rank=rank,
        spread_bin=pl.when(rank > 0.9).then(pl.lit("top decile"))
        .when(rank > 2 / 3).then(pl.lit("upper tercile"))
        .when(rank > 1 / 3).then(pl.lit("middle tercile"))
        .otherwise(pl.lit("lower tercile")),
        regime=pl.when(pl.col("cold_share") == 0).then(pl.lit(REGIME_BINS[0]))
        .when(pl.col("cold_share") < 0.25).then(pl.lit(REGIME_BINS[1]))
        .when(pl.col("cold_share") <= 0.75).then(pl.lit(REGIME_BINS[2]))
        .when(pl.col("cold_share") < 1).then(pl.lit(REGIME_BINS[3]))
        .otherwise(pl.lit(REGIME_BINS[4])),
    )  # fmt: skip


def day_block_interval(
    *, diff: pl.DataFrame, days: int = 7, n_resamples: int = 2000
) -> tuple[float, float]:
    """Return a 95% interval of the mean difference from resampling blocks of `days` origin days."""
    block = (
        diff.with_columns(block=(pl.col("power_fcst_init_time").dt.epoch("d") // days))
        .group_by("block")
        .agg(s=pl.col("d").sum(), n=pl.len())
    )
    sums, counts = block["s"].to_numpy(), block["n"].to_numpy()
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(sums), size=(n_resamples, len(sums)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report() -> None:
    """Print the Part C tables."""
    pl.Config.set_tbl_rows(100)
    table = pl.read_parquet(WORK / "b_slice_table.parquet")
    arms = {
        n: with_bins(
            rows=pl.read_parquet(WORK / f"c_rows_{n}.parquet").join(
                table, on=KEYS, how="left", validate="1:1"
            )
        ).filter(pl.col(HEADLINE))
        for n in ("S0M_raw", "S1M_raw")
    }
    s0, s1 = arms["S0M_raw"], arms["S1M_raw"]
    bins = {
        f"spread {b}": pl.col("spread_bin") == b
        for b in ("lower tercile", "middle tercile", "upper tercile", "top decile")
    }
    bins.update({f"regime {b}": pl.col("regime") == b for b in REGIME_BINS})
    bins["all rows"] = pl.lit(True)
    joined = s1.join(
        s0.select(
            *KEYS,
            **{
                f"{c}_0": c
                for c in (
                    "crps",
                    "pinball_q05",
                    "pinball_q95",
                    "pinball_q01",
                    "pinball_q99",
                    "coverage_q05",
                    "coverage_q95",
                    "coverage_q01",
                    "coverage_q99",
                    "width90",
                    "mae_median",
                )
            },
        ),
        on=KEYS,
    )
    print("== C1: S1M minus S0M by bin (all Primaries pooled; days 1-10 rows) ==")
    for name, cond in bins.items():
        sub = joined.filter(cond)
        if not sub.height:
            continue
        line = {"rows": sub.height}
        for c in ("coverage_q05", "coverage_q95", "coverage_q01", "coverage_q99"):
            line[c] = f"{sub[f'{c}_0'].mean():.3f}->{sub[c].mean():.3f}"
        for c in ("crps", "pinball_q95", "pinball_q99"):
            line[f"d_{c}"] = round(sub[c].mean() - sub[f"{c}_0"].mean(), 4)  # ty: ignore[unsupported-operator]
        r = metrics.month_bootstrap_contrast(
            treatment=s1,
            reference=s0,
            metric="crps",
            where=cond,
            slice_table=table.join(s1.select(*KEYS, "spread_bin", "regime"), on=KEYS, how="inner"),
        )
        d = sub.select(*KEYS, d=pl.col("crps") - pl.col("crps_0"))
        lo, hi = day_block_interval(diff=d)
        line["crps_diff_month_ci"] = (
            f"{r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}), {r['n_months']} months"
            if "n_months" in r
            else f"{r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f})"
        )
        line["day_block_ci"] = f"({lo:+.3f} to {hi:+.3f})"
        print(name, line)
    print("\n== C2: interval width against realised absolute error of the median (points) ==")
    for arm, frame in arms.items():
        for name, cond in bins.items():
            sub = frame.filter(cond)
            if sub.height:
                print(
                    arm,
                    name,
                    "width90",
                    round(sub["width90"].mean(), 3),  # ty: ignore[no-matching-overload]
                    "mean |median error|",
                    round(sub["mae_median"].mean(), 3),  # ty: ignore[no-matching-overload]
                    "ratio",
                    round(sub["width90"].mean() / sub["mae_median"].mean(), 2),  # ty: ignore[unsupported-operator]
                    "rank corr",
                    round(
                        sub.select(pl.corr("width90", "mae_median", method="spearman")).item(), 3
                    ),
                )
    print("\n== C3: rows per regime bin and per Primary-month check ==")
    print(
        s1.group_by("regime")
        .agg(rows=pl.len(), days=pl.col("power_fcst_init_time").dt.date().n_unique())
        .sort("regime")
    )
    print("S1M gain by regime and lead band:")
    for b in ("days 1-3", "days 4-7", "days 8-10"):
        for rg in REGIME_BINS:
            sub = joined.filter(pl.col("band") == b, pl.col("regime") == rg)
            if sub.height:
                print(
                    b,
                    rg,
                    sub.height,
                    round(sub["crps"].mean() - sub["crps_0"].mean(), 4),  # ty: ignore[unsupported-operator]
                    "q95 cov",
                    round(sub["coverage_q95_0"].mean(), 3),  # ty: ignore[no-matching-overload]
                    "->",
                    round(sub["coverage_q95"].mean(), 3),  # ty: ignore[no-matching-overload]
                )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    {"build": build, "report": report}[sys.argv[1]]()
