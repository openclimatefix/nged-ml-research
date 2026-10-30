# ruff: noqa: E501
"""Score the experiment 389 arms at all 16 Primaries under demand-v1 and print every number quoted.

Arms come from `e00389_calibrate.py` (`W_*` for the weekly member, `B_*` for experiment 381's weekly
bundle: `raw` is factor 1.0, `base` is factor 1.15, `t2` and `t3` the per-Primary arms) plus any
refitted arm named in `--extra` (files `work/arms/<name>.parquet` in fractions of capacity). The
references are BG2, `manual_heuristic`, and `climatology` of experiment 87.

    PYTHONPATH=ml-experiments/00379-*:ml-experiments/00087-*:ml-experiments/00389-* \
        DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00389-*/e00389_score.py \
        [--extra S1 ...] [--with-stack]
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00379_score as base  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00389-xgboost-tail-calibration-and-ensemble-spread-columns" / "work"
)
HEADLINE: Final[str] = base.HEADLINE_BAND
BANDS: Final[tuple[str, ...]] = (
    HEADLINE,
    "lead_day_1",
    "lead_days_5_7",
    "lead_days_8_10",
    "lead_days_11_14",
)
GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "signed-MW six": ("D5", "D9", "D10", "D11", "D14", "D16"),
    "MVA-only eight": ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"),
    "MVA-only without D8": ("D2", "D3", "D4", "D12", "D19", "D21", "D22"),
    "D15 and D18": ("D15", "D18"),
}
TAILS: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q95", 0.95),
    ("q99", 0.99),
)
CONTRASTS: Final[dict[str, tuple[str, ...]]] = {
    "W_base": ("W_raw", "W_t2", "W_t3"),
    "B_base": ("B_raw", "B_t2", "B_t3"),
    "S0M_base": ("S0M_raw", "S0M_t2", "S0M_t3", "S1M_base"),
    "S1M_base": ("S1M_raw", "S1M_t2", "S1M_t3"),
    "S0M_raw": ("S1M_raw",),
    "S0M_t2": ("S1M_t2",),
    "S0M_t3": ("S1M_t3",),
}
"""Each reference arm and the arms scored minus it."""
FAMILIES: Final[dict[str, tuple[str, ...]]] = {
    "W": ("W_raw", "W_t2", "W_t3"),
    "B": ("B_raw", "B_t2", "B_t3"),
    "S0M": ("S0M_raw", "S0M_t2", "S0M_t3"),
    "S1M": ("S1M_raw", "S1M_t2", "S1M_t3"),
}
BASES: Final[dict[str, str]] = {
    "W": "W_base",
    "B": "B_base",
    "S0M": "S0M_base",
    "S1M": "S1M_base",
}
N_BOOT: Final[int] = 20_000
SLICES: Final[tuple[str, ...]] = ("cold_nights", "sunny_april_june")


def arm_frames(*, series: pl.DataFrame, extra: list[str]) -> dict[str, pl.DataFrame]:
    """Return every arm's forecast in MW, leads to 336 hours."""
    out = {}
    for family, arms in FAMILIES.items():
        for name in (BASES[family], *arms):
            out[name] = adapter.forecast_frame(
                stored=pl.read_parquet(WORK / "arms" / f"{name}.parquet"),
                series=series,
                unit="fraction",
            )
    for name in extra:
        out[name] = adapter.forecast_frame(
            stored=pl.read_parquet(WORK / "arms" / f"{name}.parquet"),
            series=series,
            unit="fraction",
        )
    for arm in ("bg2", "manual_heuristic", "climatology"):
        stored = pl.read_parquet(base.STORED_87 / f"{base.STORED[arm]}.parquet").filter(
            pl.col("time_series_id").is_in(series["time_series_id"].to_list()),
            pl.col("lead_hours") <= base.MAX_LEAD_HOURS,
        )
        out[arm] = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    return out


def coverage_error_contrast(
    *,
    treatment: pl.DataFrame,
    reference: pl.DataFrame,
    labels: tuple[str, ...],
    name: str,
    nominal: float,
) -> tuple[float, float, float, float, float]:
    """Return the coverage of both arms and the paired change in |coverage - nominal| with a 99% interval.

    Whole calendar months of the forecast origin are resampled together for both arms.
    """
    cols = ["power_fcst_init_time", "valid_time", "label", name]
    keep = pl.col("label").is_in(list(labels)) & pl.col(HEADLINE)
    a = treatment.filter(keep).select(cols).rename({name: "t"})
    b = reference.filter(keep).select(cols).rename({name: "r"})
    j = a.join(b, on=["power_fcst_init_time", "valid_time", "label"], validate="1:1")
    month = j["power_fcst_init_time"].dt.strftime("%Y-%m")
    g = (
        j.with_columns(month=month)
        .group_by("month")
        .agg(t=pl.col("t").sum(), r=pl.col("r").sum(), n=pl.len())
    )
    t, r, n = g["t"].to_numpy(), g["r"].to_numpy(), g["n"].to_numpy()
    rng = np.random.default_rng(389)
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    cov_t = t[idx].sum(axis=1) / n[idx].sum(axis=1)
    cov_r = r[idx].sum(axis=1) / n[idx].sum(axis=1)
    diff = np.abs(cov_t - nominal) - np.abs(cov_r - nominal)
    point = abs(t.sum() / n.sum() - nominal) - abs(r.sum() / n.sum() - nominal)
    lo, hi = np.percentile(diff, [0.5, 99.5])
    return t.sum() / n.sum(), r.sum() / n.sum(), float(point), float(lo), float(hi)


def main() -> None:  # noqa: PLR0912
    """Print the scores."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", nargs="*", default=[])
    parser.add_argument("--with-stack", action="store_true")
    args = parser.parse_args()
    pl.Config.set_tbl_rows(400)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(260)
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=base.CACHE_DIR)
    actuals, context = adapter.actuals_and_context(
        cache=cache,
        series=series,
        first=datetime(2024, 4, 1, tzinfo=UTC),
        last=datetime(2026, 7, 1, tzinfo=UTC),
    )
    events = adapter.events_from_masks(cache=cache, series=series)
    first_readings = {
        int(sid): next(
            t for t, p in zip(cache.times_series(), cache.sites[lab].present, strict=True) if p
        )
        for sid, lab, _ in series.iter_rows()
    }
    thresholds = metrics.twcrps_thresholds(
        history=actuals.rename({"actual": "power_mw"}), series=series, before=base.SCORED_FIRST
    )
    arms = arm_frames(series=series, extra=args.extra)
    if args.with_stack:
        arms = {
            "W_base": arms["W_base"],
            "B_base": arms["B_base"],
            "stack_377": base.stack_frame(series=series),
        }
    table = metrics.shared_slice_table(
        forecasts=arms,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    )
    print(f"arms: {list(arms)}\nshared half-hours: {table.height}")
    rows = {}
    for name, frame in arms.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        rows[name] = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        )
    tag = "stack" if args.with_stack else "main"
    out = WORK / "scores"
    out.mkdir(exist_ok=True)
    pl.concat([r.with_columns(arm=pl.lit(n)) for n, r in rows.items()]).write_parquet(
        out / f"rows_{tag}.parquet"
    )
    value_cols = [
        "crps",
        "mae_median",
        "twcrps",
        *(f"pinball_{q}" for q, _ in TAILS),
        *(f"coverage_{q}" for q, _ in TAILS),
    ]

    for gname, labels in GROUPS.items():
        for band in BANDS:
            parts = [
                r.filter(pl.col("label").is_in(list(labels)), pl.col(band)).select(
                    pl.lit(n).alias("arm"), *[pl.col(c).mean() for c in value_cols]
                )
                for n, r in rows.items()
            ]
            print(f"\n== {gname}, slice {band}: pooled means (points; coverage as fraction) ==")
            print(pl.concat(parts))
    for gname, labels in GROUPS.items():
        for sl in SLICES:
            parts = [
                r.filter(pl.col("label").is_in(list(labels)), pl.col(sl), pl.col(HEADLINE)).select(
                    pl.lit(n).alias("arm"),
                    pl.len().alias("n"),
                    *[
                        pl.col(c).mean()
                        for c in (
                            "crps",
                            "coverage_q95",
                            "coverage_q99",
                            "coverage_q05",
                            "coverage_q01",
                        )
                    ],
                )
                for n, r in rows.items()
            ]
            print(f"\n== {gname}, {sl}, days 1 to 10 ==")
            print(pl.concat(parts))
    print("\n== Per Primary, days 1 to 10: CRPS and coverage q01 q05 q95 q99 ==")
    parts = [
        r.filter(pl.col(HEADLINE))
        .group_by("label")
        .agg(pl.col("crps").mean(), *[pl.col(f"coverage_{q}").mean().alias(q) for q, _ in TAILS])
        .with_columns(arm=pl.lit(n))
        for n, r in rows.items()
    ]
    print(pl.concat(parts).sort("label", "arm"))
    if args.with_stack:
        return
    for ref, treat in CONTRASTS.items():
        for name in treat:
            for gname, labels in GROUPS.items():
                where = pl.col("label").is_in(list(labels))
                c = metrics.headline_month_bootstrap_contrast(
                    treatment=rows[name],
                    reference=rows[ref],
                    metric="crps",
                    where=where & pl.col(HEADLINE),
                )
                print(f"\nCRPS {name} minus {ref}, {gname}: {c}")
                for q, level in TAILS:
                    cov = coverage_error_contrast(
                        treatment=rows[name],
                        reference=rows[ref],
                        labels=labels,
                        name=f"coverage_{q}",
                        nominal=level,
                    )
                    print(
                        f"  {q}: cov {name} {cov[0]:.4f}, {ref} {cov[1]:.4f}; coverage-error change {cov[2]:+.4f} (99%: {cov[3]:+.4f} to {cov[4]:+.4f})"
                    )
            for gname in ("signed-MW six", "MVA-only eight"):
                labels = GROUPS[gname]
                for left in labels:
                    where = pl.col("label").is_in([x for x in labels if x != left]) & pl.col(
                        HEADLINE
                    )
                    c = metrics.month_bootstrap_contrast(
                        treatment=rows[name], reference=rows[ref], metric="crps", where=where
                    )
                    print(
                        f"  leave out {left} ({gname}): CRPS {name} minus {ref} {c['difference']:+.4f} ({c['lower']:+.4f} to {c['upper']:+.4f})"
                    )


if __name__ == "__main__":
    main()
