# ruff: noqa: E501, PERF401
r"""Score experiment 526 (placebo repeat of the trailing weather windows) on 10 winter and 4 summer folds.

Rows (`experiment_526_v1`) follow experiment 515's rules (`experiment_515_winter_v1`): every forecast
row of a fold's test month with a lead up to 240 hours, a valid time before 2026-07-01, a cleaned
actual (flagged half-hours and the D15 and D18 start cut are NaN), and no labelled event; the rows
every scored core arm holds. The eight MVA-only and the six signed-MW Primaries are the gating
groups; D15 and D18 are report-only.

Arms are the quantile average of seeds 0 to 4 (seeds 0 to 2 of experiment 515's arms are read from
experiment 515 and experiment 513's `P0`; the rest are this experiment's). `<arm>@11` is the average
of seeds 0 to 10 and exists for W0, W-T, and W-Tnull. Quantiles are widened by 1.15 about the median
for CRPS and the tail scores, as in experiment 515. The mean-forecast errors use the unwidened mixture
mean (`mean_forecast`) and the unwidened median.

Contrast: first arm minus second arm, negative favours the first. Intervals are month-block
bootstraps over the folds of the slice (each fold carrying its seed mean), and a t interval over the
fold contrasts. Bonferroni: the primary is 1 test at 0.05 and the secondary family is 21 tests
(`secondary_family`). Everything else is descriptive and uncorrected. Slices come from the stored
ensemble mean at the valid time (experiment 515's `slice_context`), never from an arm's columns.
`all`, the cold slices, and the horizon bands use the 10 winter folds; `all_summer` and the warm
slices use the 4 summer folds.

Commands: `rows` (build per-fold row metrics), `baseline` (minimum detectable effects), `stats`
(contrasts, tails, bands, mean-forecast checks).
"""

import argparse
import math
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00404_score as score404  # ty: ignore[unresolved-import]
import e00435_tailcore as tail  # ty: ignore[unresolved-import]
import e00470_score as s470  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00515_score as s515  # ty: ignore[unresolved-import]
import e00526_fit as fit
import numpy as np
import polars as pl

W: Final[Path] = fit.W
SCORE_DIR: Final[Path] = W / "scores"
LEVELS: Final[tuple[str, ...]] = s470.LEVELS
WIDEN: Final[float] = 1.15
MAX_LEAD_HOURS: Final[float] = 240.0
POINTS: Final[float] = 100.0
GROUPS: Final[dict[str, str]] = {"eight": "mva_eight", "six": "signed_six", "report": "report_only"}
CORE_ARMS: Final[tuple[str, ...]] = ("W0", "W-T", "W-Tnull", "W-null", "W-noise3")
OTHER_ARMS: Final[tuple[str, ...]] = ("W-S", "W-X", fit.SEAM_ARM)
EXTENSION_ARMS: Final[tuple[str, ...]] = ("W0", "W-T", "W-Tnull")
CORE_SEEDS: Final[tuple[int, ...]] = fit.ALL_SEEDS
EXTENSION_SEEDS: Final[tuple[int, ...]] = (*fit.ALL_SEEDS, *fit.EXTENSION_SEEDS)
TAG_11: Final[str] = "@11"
PRIMARY_TESTS: Final[int] = 1
SECONDARY_TESTS: Final[int] = 21
RESAMPLES: Final[int] = 20_000
METRICS: Final[tuple[str, ...]] = ("crps", "pin_q95", "pin_q99", "cov_q95", "cov_q99", "ae_median", "ae_mean", "mean_minus_median")  # fmt: skip
SLICES: Final[tuple[str, ...]] = (
    "all", "cold_night", "cold_sunny", "cold_overcast", "warm_sunny", "warm_overcast", "all_summer",
    "christmas", "day1", "days2_3", "days4_10",
)  # fmt: skip
SUMMER_SLICES: Final[frozenset[str]] = frozenset({"warm_sunny", "warm_overcast", "all_summer"})
LEAVE_OUT: Final[tuple[str, ...]] = ("2024-11",)


def read_seed(*, arm: str, month: str, seed: int) -> pl.DataFrame | None:
    """Return one seed's forecast (11 quantiles and `mean_forecast`, which may be null), or None."""
    path = fit.stored_forecast(arm=arm, month=month, seed=seed)
    if not path.exists():
        return None
    frame = pl.read_parquet(path)
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    if "mean_forecast" not in frame.columns:
        refit = fit.refit_forecast(arm=arm, month=month, seed=seed)
        if refit.exists():
            frame = frame.join(pl.read_parquet(refit).select(*keys, "mean_forecast"), on=keys, how="left", maintain_order="left")  # fmt: skip
        else:
            frame = frame.with_columns(mean_forecast=pl.lit(None, dtype=pl.Float32))
    return frame


def have_seeds(*, arm: str, month: str, seeds: tuple[int, ...]) -> bool:
    """Return whether every seed's forecast file exists."""
    return all(fit.stored_forecast(arm=arm, month=month, seed=s).exists() for s in seeds)


def arm_jobs(*, month: str) -> dict[str, tuple[str, tuple[int, ...]]]:  # noqa: PLR0912
    """Return the scoring jobs of a fold: name -> (arm, seeds). Core names are plain arm names."""
    jobs: dict[str, tuple[str, tuple[int, ...]]] = {}
    for arm in (*CORE_ARMS, *OTHER_ARMS):
        if arm == fit.SEAM_ARM:  # noqa: SIM108
            seeds = CORE_SEEDS if month in fit.WINTER else ()
        else:
            seeds = CORE_SEEDS
        if seeds and have_seeds(arm=arm, month=month, seeds=seeds):
            jobs[arm] = (arm, seeds)
    if month in fit.WINTER:
        for arm in EXTENSION_ARMS:
            if have_seeds(arm=arm, month=month, seeds=EXTENSION_SEEDS):
                jobs[f"{arm}{TAG_11}"] = (arm, EXTENSION_SEEDS)
    singles = {}
    for name, (arm, seeds) in jobs.items():
        if name.endswith(TAG_11):
            continue
        for s in seeds:
            singles[f"{arm}-s{s}"] = (arm, (s,))
    for arm in EXTENSION_ARMS:
        if f"{arm}{TAG_11}" in jobs:
            for s in fit.EXTENSION_SEEDS:
                singles[f"{arm}-s{s}"] = (arm, (s,))
    return jobs | singles


def job_metrics(
    *, base: pl.DataFrame, arm: str, month: str, seeds: tuple[int, ...]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return the per-row metrics of the seed-averaged forecast of an arm, and the mask of rows it holds."""
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    frames = []
    for s in seeds:
        raw = read_seed(arm=arm, month=month, seed=s)
        assert raw is not None
        frames.append(base.select(*keys).join(raw, on=keys, how="left", maintain_order="left"))
    stacked = np.stack([f.select(list(LEVELS)).to_numpy().astype(np.float64) for f in frames]).mean(
        axis=0
    )
    mean_stack = np.stack([f["mean_forecast"].to_numpy().astype(np.float64) for f in frames]).mean(
        axis=0
    )
    ok = np.isfinite(stacked).all(axis=1)
    actual = base["actual"].to_numpy().astype(np.float64)
    wide = pl.DataFrame({"actual": actual}).hstack(
        pl.DataFrame(stacked, schema=list(LEVELS), orient="row")
    )
    widened = tail.widen(
        frame=wide.with_columns(pl.col(q).fill_null(0.0) for q in LEVELS), factor=WIDEN
    )
    out = dict(s470.row_metrics(frame=widened).items())
    median = stacked[:, LEVELS.index("q50")]
    out["ae_median"] = POINTS * np.abs(median - actual)
    out["ae_mean"] = POINTS * np.abs(mean_stack - actual)
    out["mean_minus_median"] = POINTS * (mean_stack - median)
    return {k: np.where(ok, v, np.nan) for k, v in out.items()}, ok


def month_rows(
    *, month: str, clean: np.ndarray, excluded: np.ndarray, sites: pl.DataFrame
) -> pl.DataFrame | None:
    """Return a fold's per-row metrics of every arm and seed job on the common rows."""
    jobs = arm_jobs(month=month)
    if "W0" not in jobs:
        return None
    first = pl.read_parquet(fit.stored_forecast(arm="W0", month=month, seed=0))
    keys = first.select("time_series_id", "power_fcst_init_time", "valid_time")
    keys = keys.join(sites.select("time_series_id", "primary", "site"), on="time_series_id", how="left", maintain_order="left")  # fmt: skip
    grid = c.prep422
    origin = ((keys["power_fcst_init_time"].dt.epoch("us").to_numpy() - grid.GRID_START_US) // grid.STEP_US).astype(np.int64)  # fmt: skip
    target = ((keys["valid_time"].dt.epoch("us").to_numpy() - grid.GRID_START_US) // grid.STEP_US).astype(np.int64)  # fmt: skip
    primary = keys["primary"].to_numpy().astype(np.int64)
    actual = clean[primary, target].astype(np.float64)
    lead = (target - origin) / 2.0
    keep = (
        np.isfinite(actual) & (lead > 0) & (lead <= MAX_LEAD_HOURS) & ~excluded[primary, target]
        & (keys["valid_time"].dt.epoch("us").to_numpy() < c.us(t=common.HOLDOUT))
    )  # fmt: skip
    group = np.where(keys["site"].is_in(["D15", "D18"]).to_numpy(), "report_only", np.where(keys["site"].is_in(score404.SIX_SIGNED).to_numpy(), "signed_six", "mva_eight"))  # fmt: skip
    base = keys.with_columns(actual=pl.Series(actual), lead_hours=pl.Series(lead), group=pl.Series(group)).filter(pl.Series(keep))  # fmt: skip
    ids = ["time_series_id", "power_fcst_init_time", "valid_time"]
    base = base.join(
        s515.slice_context(keys=base.select(*ids)), on=ids, how="left", maintain_order="left"
    )
    wide = base.select(*ids, "actual", "lead_hours", "group", "cold_night", "cold_sunny", "cold_overcast", "warm_sunny", "warm_overcast", "christmas")  # fmt: skip
    cols: dict[str, np.ndarray] = {}
    every = np.ones(base.height, dtype=bool)
    for name, (arm, seeds) in jobs.items():
        metrics, ok = job_metrics(base=base, arm=arm, month=month, seeds=seeds)
        if name in CORE_ARMS:
            every &= ok
        for metric, values in metrics.items():
            cols[f"{name}__{metric}"] = values
    out = wide.hstack(pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float32) for k, v in cols.items()}))
    return out.with_columns(common_rows=pl.Series(every), month=pl.lit(month), season=pl.lit("summer" if month in fit.SUMMER else "winter")).filter(pl.col("common_rows"))  # fmt: skip


def rows_command() -> None:
    """Write each fold's per-row metrics under `scores/`."""
    clean = c.clean_stack()
    padded = score404.real_event_mask()
    excluded = np.stack([score404.post_event_mask(padded=m) for m in padded])
    sites = prep.primaries().select("primary", "time_series_id", "site")
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    for month in (*fit.WINTER, *fit.SUMMER):
        frame = month_rows(month=month, clean=clean, excluded=excluded, sites=sites)
        if frame is None:
            print("skipped", month, "(no W0 forecasts)", flush=True)
            continue
        jobs = arm_jobs(month=month)
        missing = [a for a in CORE_ARMS if a not in jobs]
        frame.write_parquet(SCORE_DIR / f"rows_{month}.parquet")
        print(f"wrote {month}: {frame.height} rows; arms {sorted(n for n in jobs if '-s' not in n)}; core arms missing: {missing}", flush=True)  # fmt: skip


def load_rows() -> pl.DataFrame:
    """Return the rows of every scored fold."""
    return pl.concat(
        [pl.read_parquet(p) for p in sorted(SCORE_DIR.glob("rows_*.parquet"))], how="diagonal"
    )


def mask_of(
    *, frame: pl.DataFrame, group: str, slice_name: str, exclude: tuple[str, ...] = ()
) -> pl.Series:
    """Return the boolean row mask of a group and slice (winter folds, except the summer slices)."""
    lead = frame["lead_hours"]
    flag = {
        "all": frame["season"] == "winter",
        "all_summer": frame["season"] == "summer",
        "day1": (lead <= 24) & (frame["season"] == "winter"),
        "days2_3": (lead > 24) & (lead <= 72) & (frame["season"] == "winter"),
        "days4_10": (lead > 72) & (frame["season"] == "winter"),
    }.get(slice_name)
    if flag is None:
        flag = frame[slice_name]
        flag = flag & (frame["season"] == ("summer" if slice_name in SUMMER_SLICES else "winter"))
    out = (frame["group"] == GROUPS[group]) & flag
    for month in exclude:
        out = out & (frame["month"] != month)
    return out


def fold_table(
    *, frame: pl.DataFrame, name: str, metric: str, mask: pl.Series
) -> dict[str, tuple[float, int]]:
    """Return per fold the sum and count of a metric over the masked rows with a finite value."""
    out = {}
    column = f"{name}__{metric}"
    for month in sorted(frame["month"].unique().to_list()):
        sel = mask & (frame["month"] == month)
        values = frame.filter(sel)[column].to_numpy() if column in frame.columns else np.array([])
        out[month] = (float(np.nansum(values)), int(np.isfinite(values).sum()))
    return out


def contrast(
    *, frame: pl.DataFrame, a: str, b: str, group: str, slice_name: str, metric: str = "crps", tests: int = 1,
    exclude: tuple[str, ...] = (),
) -> dict:  # fmt: skip
    """Return `a` minus `b`: pooled, a fold-block bootstrap interval at Bonferroni level, a t interval, signs."""
    mask = mask_of(frame=frame, group=group, slice_name=slice_name, exclude=exclude)
    both = mask & frame[f"{a}__{metric}"].is_finite() & frame[f"{b}__{metric}"].is_finite() if f"{a}__{metric}" in frame.columns and f"{b}__{metric}" in frame.columns else mask & False  # fmt: skip
    ta = fold_table(frame=frame, name=a, metric=metric, mask=both)
    tb = fold_table(frame=frame, name=b, metric=metric, mask=both)
    months = [m for m in ta if ta[m][1] > 0]
    record = {"a": a, "b": b, "group": group, "slice": slice_name, "metric": metric, "tests": tests, "excluded": ",".join(exclude), "folds": len(months)}  # fmt: skip
    if not months:
        return record | {"rows": 0, "contrast": float("nan"), "lo": float("nan"), "hi": float("nan"), "t_lo": float("nan"), "t_hi": float("nan"), "negative_folds": 0}  # fmt: skip
    sa = np.array([ta[m][0] for m in months])
    sb = np.array([tb[m][0] for m in months])
    n = np.array([ta[m][1] for m in months], dtype=np.float64)
    point = (sa.sum() - sb.sum()) / n.sum()
    rng = np.random.default_rng(526)
    draws = rng.integers(0, len(months), size=(RESAMPLES, len(months)))
    boot = (sa[draws].sum(axis=1) - sb[draws].sum(axis=1)) / n[draws].sum(axis=1)
    alpha = 0.05 / (2 * tests)
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    per_fold = (sa - sb) / n
    t_lo = t_hi = float("nan")
    if len(months) > 1:
        from scipy import stats

        half = (
            stats.t.ppf(1.0 - alpha, len(months) - 1)
            * per_fold.std(ddof=1)
            / math.sqrt(len(months))
        )
        t_lo, t_hi = float(per_fold.mean() - half), float(per_fold.mean() + half)
    return record | {"rows": int(n.sum()), "contrast": float(point), "lo": float(lo), "hi": float(hi), "t_lo": t_lo, "t_hi": t_hi, "negative_folds": int((per_fold < 0).sum())}  # fmt: skip


def mde(
    *, frame: pl.DataFrame, arms: list[str], seeds: tuple[int, ...], group: str, slice_name: str
) -> dict:
    """Return the minimum detectable effect of a difference of two seed means from the single-seed spread."""
    mask = mask_of(frame=frame, group=group, slice_name=slice_name)
    residuals = []
    for arm in arms:
        per_seed = [
            fold_table(frame=frame, name=f"{arm}-s{s}", metric="crps", mask=mask) for s in seeds
        ]
        months = [m for m in sorted(per_seed[0]) if all(p[m][1] > 0 for p in per_seed)]
        if not months:
            continue
        values = np.array([[p[m][0] / p[m][1] for m in months] for p in per_seed]).T
        residuals.append(values - values.mean(axis=1, keepdims=True))
    if not residuals:
        return {"group": group, "slice": slice_name, "seeds": len(seeds), "folds": 0, "seed_sd": float("nan"), "mde": float("nan"), "mde_pooled": float("nan")}  # fmt: skip
    resid = np.concatenate(residuals)
    folds = resid.shape[0] // len(residuals)
    sd = math.sqrt((resid**2).sum() / (len(resid) * (len(seeds) - 1)))
    return {"group": group, "slice": slice_name, "seeds": len(seeds), "folds": folds, "seed_sd": sd, "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(len(seeds)), "mde_pooled": 1.96 * math.sqrt(2.0) * sd / math.sqrt(len(seeds) * folds)}  # fmt: skip


def secondary_family() -> list[tuple[str, str, str, str]]:
    """Return the 21 pre-registered secondary tests as (first arm, second arm, group, slice)."""
    family = [
        ("W-T", "W-null", "eight", "all"), ("W-T", "W-null", "six", "all"),
        ("W-T", "W-null", "eight", "cold_night"), ("W-T", "W-null", "six", "cold_night"),
        ("W-T", "W-null", "six", "cold_sunny"),
    ]  # fmt: skip
    for second in ("W-noise3", "W-Tnull"):
        for group in ("eight", "six"):
            for slice_name in ("all", "cold_night", "cold_sunny", "warm_sunny"):
                family.append(("W-T", second, group, slice_name))
    assert len(family) == SECONDARY_TESTS
    return family


def baseline_command() -> None:
    """Print the minimum detectable effects at 5 and 11 seeds, and the row counts, before any contrast."""
    frame = load_rows()
    print(f"folds scored: {sorted(frame['month'].unique().to_list())}")
    records = []
    for g in GROUPS:
        for s in ("all", "cold_night"):
            records.append(mde(frame=frame, arms=[a for a in CORE_ARMS if f"{a}-s0__crps" in frame.columns], seeds=CORE_SEEDS, group=g, slice_name=s))  # fmt: skip
            if f"W0{TAG_11}__crps" in frame.columns:
                records.append(mde(frame=frame, arms=list(EXTENSION_ARMS), seeds=EXTENSION_SEEDS, group=g, slice_name=s))  # fmt: skip
    table = pl.DataFrame(records)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    table.write_csv(SCORE_DIR / "mde.csv")
    print(table)
    print(frame.group_by("month", "group").agg(pl.len().alias("rows")).sort("month", "group"))


def stats_command() -> None:  # noqa: PLR0912
    """Write and print the primary, the secondary family, descriptive contrasts, tails, bands, and mean checks."""
    frame = load_rows()
    names = {n.split("__")[0] for n in frame.columns if "__crps" in n}
    have = lambda n: n in names  # noqa: E731
    records = []
    primary = contrast(
        frame=frame,
        a="W-T",
        b="W-Tnull",
        group="eight",
        slice_name="cold_night",
        tests=PRIMARY_TESTS,
    )
    records.append(primary | {"family": "primary"})
    if have(f"W-T{TAG_11}") and have(f"W-Tnull{TAG_11}"):
        records.append(contrast(frame=frame, a=f"W-T{TAG_11}", b=f"W-Tnull{TAG_11}", group="eight", slice_name="cold_night", tests=PRIMARY_TESTS) | {"family": "primary_11_seeds"})  # fmt: skip
        records.append(contrast(frame=frame, a=f"W-T{TAG_11}", b=f"W-Tnull{TAG_11}", group="eight", slice_name="cold_night", tests=PRIMARY_TESTS, exclude=LEAVE_OUT) | {"family": "primary_11_seeds_without_2024-11"})  # fmt: skip
    if have(f"W-T{TAG_11}") and have(f"W0{TAG_11}"):
        for g in ("eight", "six"):
            records.append(contrast(frame=frame, a=f"W-T{TAG_11}", b=f"W0{TAG_11}", group=g, slice_name="all") | {"family": "adoption_ii_11_seeds"})  # fmt: skip
    records.append(contrast(frame=frame, a="W-T", b="W-Tnull", group="eight", slice_name="cold_night", tests=PRIMARY_TESTS, exclude=LEAVE_OUT) | {"family": "primary_without_2024-11"})  # fmt: skip
    for a, b, g, s in secondary_family():
        if have(a) and have(b):
            records.append(contrast(frame=frame, a=a, b=b, group=g, slice_name=s, tests=SECONDARY_TESTS) | {"family": "secondary"})  # fmt: skip
    pairs = [("W-T", "W0"), ("W-Tnull", "W0"), ("W-null", "W0"), ("W-noise3", "W0"), ("W-S", "W0"), ("W-X", "W0"), ("W-null", "W-Tnull"), ("W-T", "W-noise3"), (fit.SEAM_ARM, "W-T")]  # fmt: skip
    for a, b in pairs:
        if not (have(a) and have(b)):
            continue
        for g in GROUPS:
            for s in SLICES:
                records.append(
                    contrast(frame=frame, a=a, b=b, group=g, slice_name=s)
                    | {"family": "descriptive"}
                )
            for s in ("all", "cold_night"):
                for metric in ("pin_q95", "pin_q99", "cov_q95", "cov_q99", "ae_mean", "ae_median"):
                    records.append(contrast(frame=frame, a=a, b=b, group=g, slice_name=s, metric=metric) | {"family": "descriptive"})  # fmt: skip
    table = pl.DataFrame(records, infer_schema_length=None)
    table.write_csv(SCORE_DIR / "contrasts.csv")
    with pl.Config(tbl_rows=500, tbl_cols=24, fmt_str_lengths=30, tbl_width_chars=250):
        print(table.filter(pl.col("family").str.starts_with("primary")))
        print(table.filter((pl.col("family") == "secondary") & (pl.col("metric") == "crps")))
    absolute = []
    for name in sorted(names):
        if "-s" in name.rsplit("@", 1)[0][-4:]:
            continue
        for g in GROUPS:
            for s in ("all", "cold_night", "warm_sunny"):
                row = {"arm": name, "group": g, "slice": s}
                for metric in ("crps", "ae_mean", "ae_median", "mean_minus_median"):
                    t = fold_table(
                        frame=frame,
                        name=name,
                        metric=metric,
                        mask=mask_of(frame=frame, group=g, slice_name=s),
                    )
                    row[metric] = sum(v[0] for v in t.values()) / max(
                        sum(v[1] for v in t.values()), 1
                    )
                absolute.append(row)
    absolute_table = pl.DataFrame(absolute)
    absolute_table.write_csv(SCORE_DIR / "absolute.csv")
    with pl.Config(tbl_rows=300, tbl_cols=12):
        print(absolute_table)
    folds = []
    for a, b in (
        ("W-T", "W-Tnull"),
        ("W-T", "W0"),
        ("W-Tnull", "W0"),
        ("W-null", "W0"),
        ("W-noise3", "W0"),
    ):
        if not (have(a) and have(b)):
            continue
        for g in ("eight", "six"):
            mask = mask_of(frame=frame, group=g, slice_name="cold_night")
            ta = fold_table(frame=frame, name=a, metric="crps", mask=mask)
            tb = fold_table(frame=frame, name=b, metric="crps", mask=mask)
            folds.append(
                {"a": a, "b": b, "group": g}
                | {
                    m: round((ta[m][0] - tb[m][0]) / ta[m][1], 3)
                    for m in sorted(ta)
                    if ta[m][1] > 0
                }
            )
    folds_table = pl.DataFrame(folds, infer_schema_length=None)
    folds_table.write_csv(SCORE_DIR / "per_fold_cold_night.csv")
    with pl.Config(tbl_cols=20, tbl_width_chars=250):
        print(folds_table)
    print(
        f"Bonferroni: primary {PRIMARY_TESTS} test at 0.05; secondary family {SECONDARY_TESTS} tests; the rest descriptive"
    )


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("rows", "baseline", "stats"))
    args = parser.parse_args()
    {"rows": rows_command, "baseline": baseline_command, "stats": stats_command}[args.command]()


if __name__ == "__main__":
    main()
