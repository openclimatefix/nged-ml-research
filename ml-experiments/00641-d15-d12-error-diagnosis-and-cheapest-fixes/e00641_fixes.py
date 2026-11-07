# ruff: noqa: D103, PTH207, RET504, PLW2901, PT018, E501
"""Experiment 641 step 3: score level-term fixes on K1's stored forecasts (CPU only).

Fixes (each shifts all 11 quantiles by a causal amount known at the origin):
  F1: the trailing 14-day mean error of K1's lead 24-48 h forecasts, per Primary.
  F2: the same, but per Primary and half-hour of the local day (the mean of the 14 previous days'
      errors at that half-hour).
Free comparisons: K1 (control), the Candidate, K3, and an oracle that removes each valid day's mean
error (not causal; a bound).
Causal rule: a forecast issued at origin T uses only errors at valid times <= T - 24 h.
"""

import glob
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

EXPORT: Final = Path(
    "/home/jack/dev/nged-substation-forecast/data/ml-experiments/backtest_export_roster_2026-11-07"
)
OUT: Final = Path(
    "/home/jack/dev/nged-substation-forecast/data/ml-experiments/00641-d15-d12-error-diagnosis-and-cheapest-fixes/tables"
)
LEVELS: Final = [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99]
QCOLS: Final = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
CUT: Final = pl.datetime(2025, 3, 1, time_zone="UTC")
WINDOW: Final = "14d"
MIN_ROWS: Final = 48  # half-hours of known errors required before a correction is applied
SEED: Final = 0
N_BOOT: Final = 4000


def load(network: str) -> pl.DataFrame:
    sub = pl.read_csv(EXPORT / "substations.csv").select(
        "time_series_id", "primary_name", "capacity"
    )
    df = pl.concat(
        [pl.read_parquet(f) for f in sorted(glob.glob(str(EXPORT / f"{network}_*.parquet")))]
    )
    df = df.join(sub, on="time_series_id").filter(
        pl.col("observed_power").is_not_null(),
        pl.col("lead_hours") >= 24,
        pl.col("lead_hours") < 240,
        ~(pl.col("primary_name").is_in(["D15", "D18"]) & (pl.col("valid_time") < CUT)),
    )
    return df


def crps(df: pl.DataFrame, shift: str | None = None) -> pl.Series:
    y = pl.col("observed_power")
    s = pl.col(shift) if shift else pl.lit(0.0)
    terms = [
        pl.max_horizontal(t * (y - (pl.col(c) + s)), (t - 1) * (y - (pl.col(c) + s)))
        for t, c in zip(LEVELS, QCOLS, strict=True)
    ]
    return df.select(
        (pl.sum_horizontal(terms) * 2 / len(LEVELS) / pl.col("capacity") * 100).alias("c")
    )["c"]


def with_fixes(k1: pl.DataFrame) -> pl.DataFrame:
    ref = (
        k1.filter(pl.col("lead_hours") < 48)
        .group_by("time_series_id", "valid_time")
        .agg((pl.col("observed_power") - pl.col("q50")).mean().alias("e"))
        .sort("time_series_id", "valid_time")
        .with_columns(
            pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.hour().alias("h"),
            pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.minute().alias("m"),
        )
        .with_columns((pl.col("h") * 2 + pl.col("m") // 30).alias("hh"))
    )
    r1 = ref.with_columns(
        pl.col("e")
        .rolling_mean_by("valid_time", window_size=WINDOW, closed="both")
        .over("time_series_id")
        .alias("f1"),
        pl.lit(1).alias("_n"),
    )
    r1 = r1.with_columns(
        pl.col("_n")
        .rolling_sum_by("valid_time", window_size=WINDOW, closed="both")
        .over("time_series_id")
        .alias("n1"),
        pl.col("e")
        .rolling_mean_by("valid_time", window_size=WINDOW, closed="both")
        .over("time_series_id", "hh")
        .alias("f2"),
        pl.col("_n")
        .rolling_sum_by("valid_time", window_size=WINDOW, closed="both")
        .over("time_series_id", "hh")
        .alias("n2"),
    )
    base = (
        k1.with_columns(
            (pl.col("power_fcst_init_time") - pl.duration(hours=24)).alias("cutoff"),
            pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.hour().alias("_h"),
            pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.minute().alias("_m"),
        )
        .with_columns((pl.col("_h") * 2 + pl.col("_m") // 30).alias("hh"))
        .sort("time_series_id", "cutoff")
    )
    # F1: latest rolling mean at or before the cutoff
    f1 = r1.select(
        "time_series_id",
        pl.col("valid_time").alias("cutoff"),
        pl.col("valid_time").alias("vt1"),
        "f1",
        "n1",
    ).sort("time_series_id", "cutoff")
    base = base.join_asof(f1, on="cutoff", by="time_series_id", strategy="backward")
    # F2: for each (series, half-hour of day) the latest rolling mean whose valid time is <= cutoff
    f2 = r1.select(
        "time_series_id",
        "hh",
        pl.col("valid_time").alias("cutoff"),
        pl.col("valid_time").alias("vt2"),
        "f2",
        "n2",
    ).sort("time_series_id", "hh", "cutoff")
    base = base.sort("time_series_id", "hh", "cutoff").join_asof(
        f2, on="cutoff", by=["time_series_id", "hh"], strategy="backward"
    )
    # a stale correction (last known error older than 14 days) is not used
    base = base.with_columns(
        pl.when(
            (pl.col("n1") >= MIN_ROWS) & ((pl.col("cutoff") - pl.col("vt1")) <= pl.duration(days=3))
        )
        .then(pl.col("f1"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("s1"),
        pl.when((pl.col("n2") >= 5) & ((pl.col("cutoff") - pl.col("vt2")) <= pl.duration(days=3)))
        .then(pl.col("f2"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("s2"),
    ).with_columns(pl.col("s1").ne(0.0).alias("applied1"))
    # staleness: the latest known error must be within 3 days of the cutoff
    return base


def block_boot(
    diff: np.ndarray, week: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Mean of diff and a 95% interval over week blocks."""
    weeks = np.unique(week)
    sums = np.array([diff[week == w].sum() for w in weeks])
    cnt = np.array([(week == w).sum() for w in weeks])
    idx = rng.integers(0, len(weeks), size=(N_BOOT, len(weeks)))
    m = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    rng = np.random.default_rng(SEED)
    k1 = load("K1")
    # unit test of the causal rule: perturb observed power after a mid-March cutoff
    base = with_fixes(k1)
    split = pl.datetime(2026, 3, 15, 9, time_zone="UTC")
    pert = k1.with_columns(
        pl.when(pl.col("valid_time") > split)
        .then(pl.col("observed_power") + 1000)
        .otherwise(pl.col("observed_power"))
        .alias("observed_power")
    )
    pb = with_fixes(pert)
    key = ["time_series_id", "power_fcst_init_time", "valid_time"]
    chk = base.select(*key, "s1", "s2").join(
        pb.select(*key, pl.col("s1").alias("p1"), pl.col("s2").alias("p2")), on=key
    )
    # origins whose cutoff (origin minus 24 h) is at or before the split must be unchanged
    early = chk.filter(pl.col("power_fcst_init_time") <= split + pl.duration(hours=24))
    late = chk.filter(pl.col("power_fcst_init_time") > split + pl.duration(days=3))
    d_early = (early["s1"] - early["p1"]).abs().max()
    d_late = (late["s1"] - late["p1"]).abs().max()
    assert early.height > 0 and late.height > 0
    assert d_early < 1e-9 and (early["s2"] - early["p2"]).abs().max() < 1e-9  # ty: ignore[unsupported-operator]
    assert d_late > 1.0  # ty: ignore[unsupported-operator]  # the test can fail: later origins do see it
    print("causal unit test passed:", early.height, "unchanged rows,", late.height, "changed rows")

    for name in ("Candidate", "K3"):
        o = load(name).select(key + [pl.col(c).alias(f"{name}_{c}") for c in QCOLS])
        base = base.join(o, on=key, how="left")
    out = base.with_columns(
        crps(base).alias("k1"),
        crps(base, "s1").alias("f1c"),
        crps(base, "s2").alias("f2c"),
    )
    for name in ("Candidate", "K3"):
        tmp = base.select(
            [pl.col(f"{name}_{c}").alias(c) for c in QCOLS] + ["observed_power", "capacity"]
        )
        out = out.with_columns(crps(tmp).alias(name.lower()))
    day = pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.date()
    out = out.with_columns(
        (pl.col("observed_power") - pl.col("q50"))
        .mean()
        .over("time_series_id", day)
        .alias("dayerr")
    )
    out = out.with_columns(crps(out, "dayerr").alias("oracle"))
    out = out.with_columns(pl.col("valid_time").dt.truncate("1w").dt.epoch("d").alias("week"))
    out.select(
        "primary_name",
        "valid_time",
        "power_fcst_init_time",
        "lead_hours",
        "k1",
        "f1c",
        "f2c",
        "candidate",
        "k3",
        "oracle",
        "applied1",
        "week",
    ).write_parquet(OUT / "fix_scores.parquet")
    rows = []
    for p, d in out.group_by("primary_name"):
        p = p[0]
        wk = d["week"].to_numpy()
        row = {"primary": p, "k1": d["k1"].mean()}
        for a in ("f1c", "f2c", "candidate", "k3", "oracle"):
            diff = (d[a] - d["k1"]).to_numpy()
            m, lo, hi = block_boot(diff, wk, rng)
            row[a] = d[a].mean()
            row[a + "_pct"] = 100 * m / row["k1"]  # ty: ignore[unsupported-operator]
            row[a + "_lo"] = 100 * lo / row["k1"]  # ty: ignore[unsupported-operator]
            row[a + "_hi"] = 100 * hi / row["k1"]  # ty: ignore[unsupported-operator]
        row["applied1"] = d["applied1"].mean()
        rows.append(row)
    t = pl.DataFrame(rows).sort("k1", descending=True)
    t.write_csv(OUT / "fix_by_primary.csv")
    with pl.Config(tbl_rows=20, tbl_cols=30, float_precision=2):
        print(
            t.select(
                "primary",
                "k1",
                "applied1",
                "f1c_pct",
                "f1c_lo",
                "f1c_hi",
                "f2c_pct",
                "f2c_lo",
                "f2c_hi",
                "oracle_pct",
                "candidate_pct",
                "k3_pct",
            )
        )
    oth = out.filter(~pl.col("primary_name").is_in(["D15", "D12"]))
    pooled = []
    for a in ("f1c", "f2c"):
        rng_pool = np.random.default_rng(SEED + 1)
        m, lo, hi = block_boot((oth[a] - oth["k1"]).to_numpy(), oth["week"].to_numpy(), rng_pool)
        base_mean = oth["k1"].mean()
        pooled.append(
            {
                "fix": a,
                "pct": 100 * m / base_mean,  # ty: ignore[unsupported-operator]
                "lo": 100 * lo / base_mean,  # ty: ignore[unsupported-operator]
                "hi": 100 * hi / base_mean,  # ty: ignore[unsupported-operator]
            }
        )  # ty: ignore[unsupported-operator]
        print(
            f"other 14 Primaries, {a}: change {100 * m / oth['k1'].mean():.2f}% [{100 * lo / oth['k1'].mean():.2f}, {100 * hi / oth['k1'].mean():.2f}]"  # ty: ignore[unsupported-operator]
        )
    pl.DataFrame(pooled).write_csv(OUT / "fix_other_14_pooled.csv")


if __name__ == "__main__":
    main()
