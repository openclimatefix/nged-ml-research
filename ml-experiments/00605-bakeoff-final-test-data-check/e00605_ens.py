# ruff: noqa: D103
"""Completeness of the production ECMWF ENS table and of the derived inputs built from it.

For every daily run from 2025-09-01 to the newest, count the members, the grid cells, the valid
times, the longest lead, and the null share of the weather columns. Then list the newest run each
derived or pretraining input holds. Reads no forecast and no forecast error.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import polars as pl
from e00605_power import OUT_DIR
from studies.sources import REPO_DATA_DIR

NWP_URI: Final[str] = str(REPO_DATA_DIR / "NWP")
FIRST_RUN: Final[datetime] = datetime(2025, 9, 1, tzinfo=UTC)
WEATHER: Final[tuple[str, ...]] = (
    "temperature_2m",
    "downward_short_wave_radiation_flux_surface",
    "wind_u_10m",
    "wind_v_100m",
    "pressure_surface",
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    day = FIRST_RUN
    # one month at a time, to bound memory
    while day < datetime(2026, 11, 8, tzinfo=UTC):
        end = (day.replace(day=28) + timedelta(days=5)).replace(day=1)
        part = (
            pl.scan_delta(NWP_URI)
            .filter(pl.col("init_time") >= day, pl.col("init_time") < end)
            .group_by("init_time")
            .agg(
                rows=pl.len(),
                members=pl.col("ensemble_member").n_unique(),
                cells=pl.col("h3_index").n_unique(),
                valid_times=pl.col("valid_time").n_unique(),
                max_lead_h=(
                    pl.col("valid_time").max() - pl.col("init_time").first()
                ).dt.total_hours(),
                **{f"null_{c}": pl.col(c).is_null().mean() for c in WEATHER},
            )
            .collect()
        )
        frames.append(part)
        print(day.date(), part.height, flush=True)
        day = end
    runs = pl.concat(frames).sort("init_time")
    runs.write_parquet(OUT_DIR / "ens_runs.parquet")
    expected = pl.datetime_range(
        FIRST_RUN, runs["init_time"].max(), "1d", time_zone="UTC", eager=True
    )
    missing = expected.filter(~expected.is_in(runs["init_time"]))
    pl.DataFrame({"missing_run": missing}).write_parquet(OUT_DIR / "ens_missing_runs.parquet")
    print("missing runs", missing.to_list())
    print(runs.tail(3))


def newest_inputs() -> None:
    """Print the newest run each derived input holds."""
    work = next(Path(REPO_DATA_DIR / "ml-experiments").glob("00397-*")) / "work"
    for name in ("ens_summary", "ens_summary_uv", "ens_summary_tcc"):
        files = sorted((work / name).glob("init=*.parquet"))
        print(name, len(files), files[0].name, files[-1].name)


if __name__ == "__main__":
    main()
    newest_inputs()
