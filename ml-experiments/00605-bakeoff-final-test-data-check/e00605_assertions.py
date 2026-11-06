# ruff: noqa: D103
"""Row-by-row assertions on the test origins: weather run timing and lag ages.

For every daily forecast origin from 2026-07-01 to 2026-10-31, assert that (1) the ENS run used is
the newest run issued at or before the origin minus 9 h, that it exists in the production table,
and that every target after the origin lies inside the run's 15-day horizon; and (2) every capped
day and week lag (experiment 558's recipe, the one the v3 designs and the 24-hour-cap designs
share) is stamped at least 24 h before the origin, and report how many lags hold a reading.
Reads observed power and ENS run metadata only. Writes `assertions.json` and `lag_presence.parquet`.
"""

import json
from datetime import UTC, datetime, timedelta
from typing import Final

import e00558_lags as lags  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00605_power import OUT_DIR, series_labels
from studies.sources import REPO_DATA_DIR

PUBLICATION_DELAY: Final[timedelta] = timedelta(hours=9)
FIRST_ORIGIN_DAY: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
LAST_ORIGIN_DAY: Final[datetime] = datetime(2026, 10, 31, tzinfo=UTC)
GRID_START: Final[datetime] = datetime(2026, 5, 1, tzinfo=UTC)
HORIZON_STEPS: Final[int] = 480  # 10 days of half-hours
STEP: Final[timedelta] = timedelta(minutes=30)


def main() -> None:
    runs = pl.read_parquet(OUT_DIR / "ens_runs.parquet")
    run_set = set(runs["init_time"].to_list())
    days = int((LAST_ORIGIN_DAY - FIRST_ORIGIN_DAY).days) + 1
    out: dict[str, object] = {"origins": days}
    # 1. Weather run timing. The origin is the 00:00 UTC run plus 9 h; check the newest run issued
    # at or before (origin - 9 h) is that run, and that it exists.
    bad = []
    for k in range(days):
        run = FIRST_ORIGIN_DAY + timedelta(days=k)
        origin = run + PUBLICATION_DELAY
        newest_allowed = origin - PUBLICATION_DELAY
        used = max(r for r in run_set if r <= newest_allowed)
        if used != run:
            bad.append(str(run))
    out["origins_whose_run_is_not_the_same_day_run"] = bad
    horizon = runs.filter(pl.col("init_time") >= FIRST_ORIGIN_DAY)
    out["min_run_horizon_hours"] = int(horizon["max_lead_h"].min())
    out["runs_with_fewer_than_51_members"] = int((horizon["members"] < 51).sum())
    # 2. Lag ages on the half-hour grid.
    ids = series_labels().filter(pl.col("group").str.starts_with("Primary"))
    power = (
        pl.scan_delta(str(REPO_DATA_DIR / "NGED" / "cleaned_power_time_series.delta"))
        .filter(pl.col("time_series_id").is_in(ids["time_series_id"].to_list()))
        .filter(pl.col("time") >= GRID_START)
        .filter(pl.col("drop_reason").is_null())
        .collect()
    )
    n_steps = int((datetime(2026, 11, 7, tzinfo=UTC) - GRID_START) / STEP)
    grid = np.full((ids.height, n_steps), np.nan, dtype=np.float32)
    row_of = {sid: i for i, sid in enumerate(ids["time_series_id"].to_list())}
    idx = (power["time"].dt.epoch("us").to_numpy() - int(GRID_START.timestamp() * 1e6)) // (
        30 * 60 * 10**6
    )
    keep = (idx >= 0) & (idx < n_steps)
    rows = np.array([row_of[s] for s in power["time_series_id"].to_list()])
    grid[rows[keep], idx[keep]] = power["power"].to_numpy()[keep]
    records = []
    max_lag_age_violation = 0
    for k in range(days):
        origin = FIRST_ORIGIN_DAY + timedelta(days=k) + PUBLICATION_DELAY
        o = int((origin - GRID_START) / STEP)
        target = np.arange(o + 1, o + 1 + HORIZON_STEPS)
        origin_idx = np.full_like(target, o)
        day, week = lags.capped_lag_indices(target_idx=target, origin_idx=origin_idx)
        # the lag's own timestamp must be at least 24 h before the origin
        max_lag_age_violation += int(((o - day) < 48).sum() + ((o - week) < 48).sum())
        for name, lag_idx in (("day", day), ("week", week)):
            vals = grid[:, lag_idx]
            records.append(
                {
                    "origin": origin,
                    "month": origin.strftime("%Y-%m"),
                    "lag": name,
                    "share_present": float(np.isfinite(vals).mean()),
                    "min_share_present_primary": float(np.isfinite(vals).mean(axis=1).min()),
                }
            )
    out["capped_lag_age_violations"] = max_lag_age_violation
    pres = pl.DataFrame(records)
    pres.write_parquet(OUT_DIR / "lag_presence.parquet")
    (OUT_DIR / "assertions.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(
        pres.group_by("month", "lag")
        .agg(
            mean_present=pl.col("share_present").mean(),
            worst_origin=pl.col("share_present").min(),
            worst_primary=pl.col("min_share_present_primary").min(),
        )
        .sort("month", "lag")
    )


if __name__ == "__main__":
    main()
