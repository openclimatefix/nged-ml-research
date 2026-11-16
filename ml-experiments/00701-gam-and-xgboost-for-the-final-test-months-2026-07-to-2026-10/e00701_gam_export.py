# ruff: noqa: E501, I001
"""Experiment 701, GAM side: write default 3's forecasts of the origin months 2026-07 to 2026-10 in experiment 677's layout.

    PYTHONPATH=<folders of 377, 700, 701> DATA_PATH_INTERNAL=<real data folder> \
        uv run --no-sync python ml-experiments/00701-*/e00701_gam_export.py

Reads the sandbox outputs of `e00701_gam_*.py` and experiment 378's arms (`e00701_gam_arms.py`), and writes
`<ROOT>/backtests/gam_final_test_sp1.parquet` beside experiment 700's `gam_year_sp1.parquet`, in the same
columns, units, and 12:00 UTC origin relabel (`e00700_export.to_contract`, imported unchanged).
Set `E00701_ROOT_BACKTESTS` to write elsewhere.

**The forecast is Primary demand research default 3 as experiment 378 assembles it**, with the stage 2
stack of experiment 377 beyond 240 hours, exactly as `e00700_export.gam_rows` does:

- up to 168 hours after the native 09:00 UTC origin: the solar summary S (`x1`, `low_lead_<source>.parquet`);
- above 168 and up to 240 hours: S and the daylight heuristic blend (`d3`, `long_lead_<source>.parquet`);
- above 240 hours: stage 2 without S (`stack_d_<site>.parquet`).

Fifteen Primaries are written; D15 has no stage 2 file. The script asserts that no quantile is `null`,
that every Primary has an origin on each of the 123 days (every day of the months), and that no origin
is missing 672 half-hours beyond the gaps of the actual power readings.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

import e00701_gam_extend377  # noqa: F401  (dates and paths first)
import e00701_gam_extend as ext
import e00700_export as e700  # ty: ignore[unresolved-import]
import polars as pl

Q = e700.Q
SOURCE_OF = e700.SOURCE_OF
KEYS = ["power_fcst_init_time", "valid_time"]
START, END = ext.FIRST_ORIGIN_MONTH, ext.END
STORED_377 = ext.REAL_DATA / "ml-experiments" / "00377-demand-v1-first-run-feasibility" / "work"
BACKTESTS = Path(os.environ.get("E00701_ROOT_BACKTESTS", ext.ROOT / "backtests"))


def gam_rows(
    *,
    start: datetime = START,
    end: datetime = END,
    sites: list[str] | None = None,
    low_frames: dict[str, pl.DataFrame] | None = None,
    long_frames: dict[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Return default 3's quantiles at the Primaries, in points, with leads from the native origin.

    `low_frames` and `long_frames` (keyed by source) replace the arms' files, for the smoke test.
    """
    w377 = ext.SANDBOX_ML / "00377-demand-v1-first-run-feasibility" / "work"
    w378 = (
        ext.SANDBOX_ML
        / "00378-primary-demand-default-3-at-all-16-primaries-under-demand-v1"
        / "work"
        / "arms"
    )
    frames = []
    for site, source in SOURCE_OF.items():
        if sites is not None and site not in sites:
            continue
        name = "stack_d_donor_D8" if site == "D8" else f"stack_d_{site}"
        base = pl.read_parquet(w377 / source / "arms" / "a0" / "dp_preds_s2" / f"{name}.parquet")
        # A Primary has no forecast before its first month in experiment 377's stored file.
        first_init = pl.read_parquet(
            STORED_377 / source / "arms" / "a0" / "dp_preds_s2" / f"{name}.parquet",
            columns=["power_fcst_init_time"],
        )["power_fcst_init_time"].min()
        assert isinstance(first_init, datetime), name
        first = first_init.replace(day=1, hour=0, minute=0)
        base = base.filter(
            (pl.col("power_fcst_init_time") >= max(first, start))
            & (pl.col("seed") == 0)
            & (pl.col("power_fcst_init_time") < end)
        )
        low = (
            low_frames[source]
            if low_frames
            else pl.read_parquet(w378 / f"low_lead_{source}.parquet")
        ).filter(pl.col("site") == site)
        long = (
            long_frames[source]
            if long_frames
            else pl.read_parquet(w378 / f"long_lead_{source}.parquet")
        ).filter(pl.col("site") == site)
        joined = base.join(
            low.select(*KEYS, *[pl.col(f"x1_{q}").alias(f"low_{q}") for q in Q]),
            on=KEYS,
            how="left",
        ).join(
            long.select(*KEYS, *[pl.col(f"d3_{q}").alias(f"long_{q}") for q in Q]),
            on=KEYS,
            how="left",
        )
        short = pl.col("lead_hours") <= 168.0
        mid = pl.col("lead_hours") <= 240.0
        out = joined.select(
            pl.lit(site).alias("site"),
            *KEYS,
            "lead_hours",
            *[
                pl.when(short)
                .then(pl.col(f"low_{q}"))
                .when(mid)
                .then(pl.col(f"long_{q}"))
                .otherwise(pl.col(q))
                .alias(q)
                for q in Q
            ],
        )
        assert out.select(Q).null_count().sum_horizontal().item() == 0, site
        frames.append(out)
    return pl.concat(frames)


def write(*, frame: pl.DataFrame, name: str, step: str) -> Path:
    """Write a file in experiment 677's layout (seed 0, `NRA` rows) under `BACKTESTS`."""
    out = frame.with_columns(
        seed=pl.lit(0, dtype=pl.Int64),
        step=pl.lit(step),
        version=pl.lit("research"),
        NRA_or_prevailing=pl.lit("NRA"),
    )
    path = BACKTESTS / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    out.write_parquet(temporary)
    temporary.rename(path)
    per = out.group_by("time_series_id", "power_fcst_init_time").len()["len"].value_counts()
    print(
        name,
        out.height,
        "rows; Primaries",
        out["time_series_id"].n_unique(),
        "; origins",
        out["power_fcst_init_time"].n_unique(),
        "; rows per origin",
        per.to_dicts(),
    )
    return path


def main() -> None:
    """Assemble and write the file, then check its coverage."""
    frame = e700.to_contract(frame=gam_rows(), unit="points")
    days = (END - START).days
    origins = frame.group_by("time_series_id").agg(n=pl.col("power_fcst_init_time").n_unique())
    assert (origins["n"] == days).all(), origins
    assert frame["power_fcst_init_time"].min() == datetime(2026, 7, 1, 12, tzinfo=UTC)
    assert frame["power_fcst_init_time"].max() == datetime(2026, 10, 31, 12, tzinfo=UTC)
    assert frame["valid_time"].max() <= datetime(2026, 11, 14, 12, tzinfo=UTC)
    write(frame=frame, name="gam_final_test_sp1", step="gam")


if __name__ == "__main__":
    main()
