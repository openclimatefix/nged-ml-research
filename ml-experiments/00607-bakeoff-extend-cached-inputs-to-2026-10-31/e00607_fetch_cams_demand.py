# ruff: noqa: E501
"""Fetch CAMS irradiance at the 22 demand series for 2026-09-11 to 2026-10-31.

The published script `fetch_cams_demand_points.py` refuses a date override and rewrites the whole
table, so this wrapper reuses its request, strip-coordinates, and parsing functions for one new
window. It writes `cams_demand_points_2026-09-11_2026-10-31.parquet` beside the stored table; the
stored table is not touched. Rows are keyed by the anonymous label `D1` to `D22`.

    uv run --no-sync --with cdsapi python e00607_fetch_cams_demand.py --series <series table>
"""

import argparse
import concurrent.futures
import hashlib
from pathlib import Path
from typing import Final

import fetch_cams_demand_points as published  # ty: ignore[unresolved-import]
import polars as pl

FIRST: Final[str] = "2026-09-11"
LAST: Final[str] = "2026-10-31"
OUTPUT: Final[Path] = published.OUTPUT_DIR / "cams_demand_points_2026-09-11_2026-10-31.parquet"


def main() -> None:
    """Fetch the 22 sites and write the tidy frame."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=Path, required=True)
    args = parser.parse_args()
    series = pl.read_parquet(args.series).select("site", "latitude", "longitude")
    published.check_series(series=series)
    fingerprint = hashlib.sha256(series.sort("site").write_csv().encode()).hexdigest()
    marker = published.OUTPUT_DIR / ".series_fingerprint"
    if marker.read_text() != fingerprint:
        msg = "series table differs from the one the stored raw files were fetched with"
        raise SystemExit(msg)
    # One request per site for the whole window; the published helpers read the range from here.
    published.date_range_of = lambda *, year: f"{FIRST}/{LAST}"  # ty: ignore[invalid-assignment]
    jobs = published.build_jobs(series=series.with_columns())
    jobs = [job for job in jobs if job.year == 2026]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=published.MAX_CONCURRENT_REQUESTS
    ) as pool:
        futures = [pool.submit(published.fetch_one, job=job) for job in jobs]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            future.result()
            print(f"{done} of {len(jobs)} sites ready", flush=True)
    frame = pl.concat(
        [
            published.read_one(path=published.csv_path(site=j.site, year=j.year), site=j.site)
            for j in jobs
        ]
    )
    tidy = (
        frame.drop_nulls(subset=["time", *published.VALUE_COLUMNS])
        .filter(~pl.any_horizontal(pl.col(*published.VALUE_COLUMNS).is_nan()))
        .sort("site", "time")
    )
    tidy.write_parquet(OUTPUT)
    expected = 22 * 24 * 51
    print(
        f"wrote {tidy.height:,} rows (expected {expected:,}), {tidy['time'].min()} to {tidy['time'].max()}"
    )


if __name__ == "__main__":
    main()
