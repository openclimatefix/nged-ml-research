# ruff: noqa: D103, E501, PT018, ANN201
"""Experiment 704: per-Primary tables for arms A ("proxy") and B ("drop"), CPU only.

Each table is experiment 667's table (`inputs/trailing437/<site>.parquet`, read only) with the two
previous-day CAMS columns `cams_prev_day_mean` and `cams_prev_day_csi` replaced, written to
`inputs/trailing437_<arm>/<site>.parquet` in this experiment's data folder:

- `drop`: both columns are 0.0.
- `proxy`: both columns come from the analysis-proxy shortwave of experiment 437's `site_table`
  (the first day of each run, from the ensemble-summary `mean_shortwave`). For an origin, the mean
  of the 48 half-hours of the previous UTC day (stamps 00:30 to 00:00 of the origin's day, period
  ends), and their sum over the sum of `f437.clear_sky_on_grid`; NaN if any of the 48 half-hours
  has no finite shortwave or the clear-sky sum is 0. Constant within a run.

Usage: python e00704_tables.py {build|check|causality} {proxy|drop} [SITE ...]
`build` skips a table that exists; `check` compares each table with experiment 667's (every other
column bit-equal on every key) and, for proxy, with CAMS; `causality` (proxy, site D2) rebuilds the
columns from summaries with all later runs deleted.
"""

import sys
import time
from datetime import UTC, datetime
from typing import Final

import e00437_features as f437  # ty: ignore[unresolved-import]
import e00667_tables as t667  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00704_common import previous_day_columns  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00704-cams-free-inference-for-the-primaries-step-5-network"
ROOT: Final = ML_EXPERIMENTS_DATA_DIR / NAME
SOURCE: Final = t667.OUT
"""Experiment 667's tables (read only)."""
CAMS_COLUMNS: Final = ("cams_prev_day_mean", "cams_prev_day_csi")


def arm_dir(*, arm: str):
    return ROOT / "inputs" / f"trailing437_{arm}"


def proxy_by_origin(*, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the sorted origins (us) of a Primary's summary, and each origin's two proxy columns.

    The analysis-proxy shortwave is built exactly as `f437.site_table` builds `history_s`.
    """
    summary = (
        f437.prep.load_summary(series_id=int(row["time_series_id"]))
        .select(
            origin=pl.col("power_fcst_init_time").dt.epoch("us"),
            valid=pl.col("valid_time").dt.epoch("us"),
            shortwave=pl.col("mean_shortwave").cast(pl.Float64),
        )
        .sort("origin", "valid")
    )
    origin_us, valid_us = summary["origin"].to_numpy(), summary["valid"].to_numpy()
    shortwave = summary["shortwave"].to_numpy()
    oi, vi = f437.grid_of(us=origin_us), f437.grid_of(us=valid_us)
    history = np.full(f437.common.grid_times().len(), np.nan)
    first_day = (vi - oi >= 1) & (vi - oi <= f437.STEPS_PER_DAY)
    if len(np.unique(vi[first_day])) != int(first_day.sum()):
        msg = "two runs' first days overlap on the grid"
        raise ValueError(msg)
    history[vi[first_day]] = shortwave[first_day]
    clear = f437.clear_sky_on_grid(
        latitude=float(row["latitude"]), longitude=float(row["longitude"])
    )
    origins = np.unique(origin_us)
    mean, csi = previous_day_columns(shortwave=history, clear=clear, origin_us=origins)
    return origins, mean, csi


def arm_table(*, arm: str, row: dict, source: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 667's table `source` with the two CAMS columns of `arm`."""
    if arm == "drop":
        return source.with_columns(*(pl.lit(0.0, dtype=pl.Float32).alias(c) for c in CAMS_COLUMNS))
    origins, mean, csi = proxy_by_origin(row=row)
    wanted = source["origin_us"].to_numpy()
    at = np.clip(np.searchsorted(origins, wanted), 0, len(origins) - 1)
    assert (origins[at] == wanted).all(), "a table origin has no summary run"
    return source.with_columns(
        pl.Series(CAMS_COLUMNS[0], mean[at].astype(np.float32)),
        pl.Series(CAMS_COLUMNS[1], csi[at].astype(np.float32)),
    )


def selected(*, sites: list[str]) -> list[dict]:
    rows = list(f437.prep.primaries().iter_rows(named=True))
    return [r for r in rows if not sites or r["site"] in sites]


def build(*, arm: str, sites: list[str]) -> None:
    out = arm_dir(arm=arm)
    out.mkdir(parents=True, exist_ok=True)
    for row in selected(sites=sites):
        target = out / f"{row['site']}.parquet"
        if target.exists():
            print(row["site"], "exists", flush=True)
            continue
        started = time.monotonic()
        table = arm_table(
            arm=arm, row=row, source=pl.read_parquet(SOURCE / f"{row['site']}.parquet")
        )
        tmp = target.with_name(target.name + ".tmp")
        table.write_parquet(tmp)
        tmp.rename(target)
        share = {c: int(table[c].is_finite().sum()) / table.height for c in CAMS_COLUMNS}
        print(
            row["site"],
            table.height,
            "rows",
            f"{time.monotonic() - started:.1f} s",
            "finite",
            share,
            flush=True,
        )


def same(*, a: np.ndarray, b: np.ndarray) -> bool:
    return bool(((a == b) | (np.isnan(a) & np.isnan(b))).all())


def per_origin(*, table: pl.DataFrame) -> pl.DataFrame:
    return (
        table.group_by("origin_us")
        .agg(*(pl.col(c).first() for c in CAMS_COLUMNS))
        .sort("origin_us")
    )


def check(*, arm: str, sites: list[str]) -> None:
    others = [c for c in (*f437.INPUT_COLUMNS, *f437.DIAGNOSTIC_COLUMNS) if c not in CAMS_COLUMNS]
    for row in selected(sites=sites):
        old = pl.read_parquet(SOURCE / f"{row['site']}.parquet")
        new = pl.read_parquet(arm_dir(arm=arm) / f"{row['site']}.parquet")
        assert old.columns == new.columns and old.height == new.height, row["site"]
        for key in ("origin_us", "valid_us"):
            assert (old[key] == new[key]).all(), (row["site"], key)
        bad = [c for c in others if not same(a=old[c].to_numpy(), b=new[c].to_numpy())]
        assert not bad, (row["site"], "columns differ from experiment 667's", bad)
        assert all(new[c].dtype == pl.Float32 for c in CAMS_COLUMNS)
        new_by = per_origin(table=new)
        print(
            row["site"],
            "same keys, the other",
            len(others),
            "columns bit-equal; finite share of rows",
            {c: round(int(new[c].is_finite().sum()) / new.height, 6) for c in CAMS_COLUMNS},
            "origins not finite:",
            int((~new_by[CAMS_COLUMNS[0]].is_finite()).sum()),
            "of",
            new_by.height,
            flush=True,
        )
        if arm == "drop":
            assert (new[CAMS_COLUMNS[0]] == 0).all()
            assert (new[CAMS_COLUMNS[1]] == 0).all()
            continue
        old_by = per_origin(table=old)
        for c in CAMS_COLUMNS:
            a, b = old_by[c].to_numpy().astype(np.float64), new_by[c].to_numpy().astype(np.float64)
            ok = np.isfinite(a) & np.isfinite(b)
            print(
                f"  {c} vs CAMS over {int(ok.sum())} origins: correlation {np.corrcoef(a[ok], b[ok])[0, 1]:.3f}, "
                f"mean {b[ok].mean():.3f} vs {a[ok].mean():.3f}, std {b[ok].std():.3f} vs {a[ok].std():.3f}",
                flush=True,
            )


def causality(*, arm: str, sites: list[str]) -> None:
    assert arm == "proxy", "only arm A reads the summaries"
    row = selected(sites=sites or ["D2"])[0]
    new = pl.read_parquet(arm_dir(arm=arm) / f"{row['site']}.parquet")
    full = f437.prep.load_summary(series_id=int(row["time_series_id"]))
    original = f437.prep.load_summary
    cuts = (
        datetime(2025, 3, 10, 9, tzinfo=UTC),
        datetime(2026, 7, 1, 9, tzinfo=UTC),
        datetime(2026, 10, 31, 9, tzinfo=UTC),
    )
    for cut in cuts:
        truncated = full.filter(pl.col("power_fcst_init_time") <= cut)
        f437.prep.load_summary = lambda *, series_id, t=truncated: t  # ty: ignore[invalid-assignment]
        try:
            origins, mean, csi = proxy_by_origin(row=row)
        finally:
            f437.prep.load_summary = original
        key = int(cut.timestamp() * 1_000_000)
        at = int(np.searchsorted(origins, key))
        assert origins[at] == key
        stored = new.filter(pl.col("origin_us") == key)
        assert stored.height > 0
        worst = 0.0
        for c, rebuilt in zip(CAMS_COLUMNS, (mean[at], csi[at]), strict=True):
            values = stored[c].to_numpy().astype(np.float64)
            assert np.allclose(values, values[0], equal_nan=True), "a column varies within a run"
            if np.isnan(rebuilt) or np.isnan(values[0]):
                assert np.isnan(rebuilt)
                assert np.isnan(values[0]), (c, rebuilt, values[0])
            else:
                worst = max(worst, abs(values[0] - float(rebuilt)) / max(abs(values[0]), 1.0))
        print(
            "run",
            cut,
            "largest relative difference with all later runs deleted:",
            worst,
            "values",
            mean[at],
            csi[at],
            flush=True,
        )
        assert worst <= 1e-6, "a column depends on a later run"  # float32 storage


if __name__ == "__main__":
    command, arm_name, *site_list = sys.argv[1:]
    assert arm_name in {"proxy", "drop"}, arm_name
    t667.setup()
    f437.TRAILING_DIR = arm_dir(arm=arm_name)  # never experiment 667's folder
    {"build": build, "check": check, "causality": causality}[command](arm=arm_name, sites=site_list)
