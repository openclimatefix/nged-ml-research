# ruff: noqa: E501
"""Experiment 611: score the BSP/GSP pilot forecasts and write the Parquet export.

Reads each design's per-seed forecasts (capacity fractions), averages the seeds in quantile space,
converts to MW, joins the cleaned actual, and writes:

- `scores/rows.parquet`: one row per forecast row with CRPS, coverage, and width;
- `scores/summary.csv`: CRPS (points of effective capacity), q05-q95 coverage, and q05-q95 width by
  design, lead band, group, series, and month;
- `scores/contrasts.csv`: paired month-block contrasts with Bonferroni intervals and the minimum
  detectable effect;
- `export/<design>_<month>.parquet`: the forecasts in the layout of `backtest_export_2026-11-06/`.

Stored references come from experiment 87 (BG2, `climatology`, `manual_heuristic`) and cover
whatever test months and leads that experiment stored; the rows they lack are reported as missing.

Usage: python e00611_score.py DESIGN [DESIGN ...]   (designs: D1, D2)
"""

import itertools
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00386_adapter as adapter  # ty: ignore[unresolved-import]
import e00386_score_a as stage_a  # ty: ignore[unresolved-import]
import e00611_pool as pool
import numpy as np
import polars as pl

OUT: Final[Path] = pool.WORK / "scores"
EXPORT: Final[Path] = pool.WORK / "export"
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
BANDS: Final[tuple[tuple[int, int], ...]] = ((0, 24), (24, 120), (120, 240))
LEVELS: Final[list[str]] = list(common.LEVEL_COLUMNS)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
LABELS: Final[dict[int, str]] = {}
"""Filled by `series()`: time_series_id to site label (D1, D6, D13, D17)."""
KIND: Final[dict[str, str]] = {"D13": "BSP", "D17": "BSP", "D1": "GSP", "D6": "GSP"}


def series() -> pl.DataFrame:
    """Return the 4 series: id, label, kind, capacity in MW."""
    table = adapter.series_table().with_columns(
        kind=pl.col("label").replace(KIND), unit=pl.lit("MW")
    )
    LABELS.update(dict(zip(table["time_series_id"], table["label"], strict=True)))
    return table


def seed_average(*, files: list[Path]) -> pl.DataFrame:
    """Return the mean over files of the quantiles (sorted again), keyed by series, origin, and valid time."""
    frame = (
        pl.concat([pl.read_parquet(f) for f in files])
        .group_by(KEYS)
        .agg(*[pl.col(c).mean() for c in LEVELS])
        .sort(KEYS)
    )
    values = np.sort(frame.select(LEVELS).to_numpy(), axis=1)
    return frame.with_columns(*[pl.Series(c, values[:, i]) for i, c in enumerate(LEVELS)])


def load_design(*, design: str, table: pl.DataFrame) -> pl.DataFrame:
    """Return a design's seed-averaged forecast for all 4 series in MW, with `month` and `lead_hours`."""
    parts = []
    for group in pool.GROUPS:
        folder = pool.WORK / "work" / group / "seed_preds" / {"D1": "O2", "D2": "N1"}[design]
        by_month: dict[str, list[Path]] = {}
        for f in sorted(folder.glob("*_s*.parquet")):
            by_month.setdefault(f.stem.split("_s")[0], []).append(f)
        for month, files in by_month.items():
            parts.append(seed_average(files=files).with_columns(month=pl.lit(month)))
    frame = pl.concat(parts).join(
        table.select("time_series_id", "capacity_mw"), on="time_series_id"
    )
    frame = frame.with_columns(*[(pl.col(c) * pl.col("capacity_mw")).alias(c) for c in LEVELS])
    return frame.with_columns(
        lead_hours=(
            (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 3600
        )
    )


def actuals(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return the cleaned actual power (MW) of each series on the half-hour grid."""
    out = []
    for group in pool.GROUPS:
        pool.install(group=group)
        pool._table = None
        pool.GROUP = group
        clean = pool.cleaned_arrays()
        grid = common.grid_times()
        for site in pool.GROUPS[group]:
            row = table.filter(pl.col("label") == site).row(0, named=True)
            power = clean[site].astype(np.float64) * row["capacity_mw"]
            out.append(
                pl.DataFrame({"valid_time": grid, "actual": power}).with_columns(
                    time_series_id=pl.lit(row["time_series_id"]).cast(pl.Int32)
                )
            )
    return pl.concat(out).filter(pl.col("actual").is_finite())


def pinball_crps(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `crps` (MW; atom-mass form), `covered` (q05 <= actual <= q95), and `width` (q95 - q05)."""
    q = frame.select(LEVELS).to_numpy()
    q = np.sort(q, axis=1)
    y = frame["actual"].to_numpy()[:, None]
    diff = y - q
    levels = common.LEVELS[None, :]
    pin = np.where(diff >= 0, diff * levels, -diff * (1.0 - levels))
    masses = common.ATOM_MASSES
    crps = 2.0 * (pin * masses[None, :]).sum(axis=1) / masses.sum()
    i05, i95 = LEVELS.index("q05"), LEVELS.index("q95")
    return frame.with_columns(
        crps=pl.Series(crps),
        covered=pl.Series((q[:, i05] <= y[:, 0]) & (y[:, 0] <= q[:, i95])),
        width=pl.Series(q[:, i95] - q[:, i05]),
        below_q95=pl.Series(y[:, 0] <= q[:, i95]),
    )


def band_label(*, lo: int, hi: int) -> str:
    """Return the label of a lead band."""
    return f"{lo}-{hi}h"


def add_bands(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add a `band` column (the lead bands are half-open: lo <= lead < hi)."""
    expr = pl.lit(None, dtype=pl.String)
    for lo, hi in BANDS:
        expr = (
            pl.when((pl.col("lead_hours") >= lo) & (pl.col("lead_hours") < hi))
            .then(pl.lit(band_label(lo=lo, hi=hi)))
            .otherwise(expr)
        )
    return frame.with_columns(band=expr).filter(pl.col("band").is_not_null())


def references(*, table: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return experiment 87's stored BG2, climatology, and manual heuristic at the 4 series, in MW."""
    out = {}
    for arm, frame in stage_a.reference_frames(
        series=table.select("time_series_id", "label", "capacity_mw")
    ).items():
        out[arm] = frame.with_columns(
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
            lead_hours=(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 3600
            ),
        ).filter(pl.col("month").is_in(MONTHS))
    return out


def month_block_contrast(*, diff: pl.DataFrame, n_tests: int) -> dict[str, float]:
    """Bound a paired difference by resampling the 5 months, at the Bonferroni level for `n_tests`.

    `diff` has one row per month with `d` (mean CRPS difference, treatment minus reference). With
    5 blocks all 5^5 resamples are enumerated, so the interval is exact for this scheme. The
    minimum detectable effect is (z for the Bonferroni level + z for 80% power) x the standard error
    of the mean of the monthly differences.
    """
    d = diff["d"].to_numpy()
    n = len(d)
    means = np.sort([np.mean([d[i] for i in idx]) for idx in itertools.product(range(n), repeat=n)])
    alpha = 0.05 / n_tests
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    nd = NormalDist()
    mde = (nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(0.8)) * se
    return {
        "difference": float(d.mean()),
        "lower": float(lo),
        "upper": float(hi),
        "mde": float(mde),
        "n_months": n,
    }


def write_export(
    *, arms: dict[str, pl.DataFrame], act: pl.DataFrame, table: pl.DataFrame, names: list[str]
) -> None:
    """Write one Parquet per design and month in the layout of `backtest_export_2026-11-06/`."""
    EXPORT.mkdir(parents=True, exist_ok=True)
    for name in names:
        frame = (
            arms[name]
            .join(act, on=["time_series_id", "valid_time"], how="left")
            .rename({"actual": "observed_power"})
        )
        frame = frame.with_columns(unit=pl.lit("MW")).select(
            "time_series_id",
            "power_fcst_init_time",
            "valid_time",
            "lead_hours",
            *LEVELS,
            "observed_power",
            "unit",
            "month",
        )
        for (month,), part in frame.group_by("month"):
            part.drop("month").sort(KEYS[:1] + KEYS[1:]).write_parquet(
                EXPORT / f"{name}_{month}.parquet", compression="zstd"
            )
    table.select(
        "time_series_id",
        name=pl.col("label"),
        kind=pl.col("kind"),
        capacity_mw=pl.col("capacity_mw"),
        unit=pl.col("unit"),
    ).write_csv(EXPORT / "substations.csv")


def main(designs: list[str]) -> None:
    """Score the designs and the references, and write the tables and the export."""
    table = series()
    OUT.mkdir(parents=True, exist_ok=True)
    act = actuals(table=table)
    arms: dict[str, pl.DataFrame] = {}
    for d in designs:
        arms[f"design{d[1:]}"] = load_design(design=d, table=table)
    names = list(arms)
    for name, frame in references(table=table).items():
        arms[name] = frame.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
    scored = []
    for name, frame in arms.items():
        j = frame.join(act, on=["time_series_id", "valid_time"], how="inner")
        j = add_bands(frame=pinball_crps(frame=j)).with_columns(
            arm=pl.lit(name),
            label=pl.col("time_series_id").replace_strict(LABELS, return_dtype=pl.String),
            cap=pl.col("time_series_id").replace_strict(
                dict(zip(table["time_series_id"], table["capacity_mw"], strict=True)),
                return_dtype=pl.Float64,
            ),
        )
        scored.append(
            j.with_columns(
                crps_points=pl.col("crps") / pl.col("cap") * 100,
                width_points=pl.col("width") / pl.col("cap") * 100,
                group=pl.col("label").replace(KIND),
            ).select(
                "arm",
                "label",
                "group",
                "month",
                "band",
                "power_fcst_init_time",
                "valid_time",
                "lead_hours",
                "actual",
                "crps_points",
                "width_points",
                "width",
                "covered",
                "below_q95",
            )
        )
    rows = pl.concat(scored)
    write_export(arms=arms, act=act, table=table, names=names)
    rows.write_parquet(OUT / "rows.parquet")
    agg = [
        pl.col("crps_points").mean().alias("crps_points"),
        pl.col("covered").mean().alias("coverage_q05_q95"),
        pl.col("below_q95").mean().alias("share_below_q95"),
        pl.col("width_points").mean().alias("width_points"),
        pl.col("width").mean().alias("width_mw"),
        pl.len().alias("n"),
    ]
    rows.group_by("arm", "group", "label", "band", "month").agg(*agg).sort(
        "arm", "label", "band", "month"
    ).write_csv(OUT / "summary_by_month.csv")
    for by, name in ((["arm", "group", "band"], "group"), (["arm", "label", "band"], "series")):
        rows.group_by(*by).agg(*agg).sort(*by).write_csv(OUT / f"summary_by_{name}.csv")
    # Paired contrasts: only rows scored by both arms.
    contrasts = []
    pairs = [(a, b) for a in names for b in ("climatology", "bg2", "manual_heuristic")] + [
        (a, b) for a, b in itertools.permutations(names, 2) if a > b
    ]
    keys = ["label", "power_fcst_init_time", "valid_time"]
    pending = []
    for tr, ref in pairs:
        for grp, band in itertools.product(
            ("BSP", "GSP"), [band_label(lo=lo, hi=hi) for lo, hi in BANDS]
        ):
            sub = rows.filter(pl.col("group") == grp, pl.col("band") == band)
            both = sub.filter(pl.col("arm") == tr).join(
                sub.filter(pl.col("arm") == ref), on=keys, suffix="_ref"
            )
            if both.is_empty():
                contrasts.append(
                    {
                        "treatment": tr,
                        "reference": ref,
                        "group": grp,
                        "band": band,
                        "note": "reference missing",
                    }
                )
                continue
            per_month = (
                both.group_by("month")
                .agg(
                    (pl.col("crps_points") - pl.col("crps_points_ref")).mean().alias("d"),
                    pl.len().alias("n"),
                )
                .sort("month")
            )
            pending.append((tr, ref, grp, band, per_month))
    n_tests = len(pending)
    for tr, ref, grp, band, per_month in pending:
        res = month_block_contrast(diff=per_month, n_tests=n_tests)
        contrasts.append(
            {
                "treatment": tr,
                "reference": ref,
                "group": grp,
                "band": band,
                "n_rows": int(per_month["n"].sum()),
                "bonferroni_count": n_tests,
                **res,
            }
        )
    pl.DataFrame(contrasts, infer_schema_length=None).write_csv(OUT / "contrasts.csv")
    print(f"Bonferroni count {n_tests}")
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=200):
        print(pl.read_csv(OUT / "summary_by_group.csv"))
        print(pl.read_csv(OUT / "summary_by_series.csv"))
        print(pl.read_csv(OUT / "contrasts.csv"))


if __name__ == "__main__":
    main(sys.argv[1:])
