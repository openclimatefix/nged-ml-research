"""Build the daily demand frames of phase E (2025-04-01 to 2025-06-30, 20 demand series).

    uv run python ml-experiments/00709-*/e00709_e_frames.py [--out DIR] [--first DAY] [--last DAY]

Experiment 48's `build_run` makes one run's demand frame from the ECMWF ENS archive and the stage 1
pilot's observed power table, and its own command line builds only its fixed run list. This driver
calls `build_run` and `_write_sorted` unchanged for the phase E dates. The default output is
`frames_e/` in this experiment's data folder, never a folder of another experiment. A run whose file
exists is skipped, so a stopped build resumes. A rebuild of run 2025-05-29 matched the stored
snapshot frame exactly (all 250,614 rows of its 7 series, maximum difference 0.0).
"""

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import e00048_build_frames as bf  # ty: ignore[unresolved-import]
import e00048_common as common  # ty: ignore[unresolved-import]
import e00048_gen_common as gen  # ty: ignore[unresolved-import]
import e00709_features as features  # ty: ignore[unresolved-import]
import polars as pl

DEFAULT_OUT = features.DATA_DIR / "frames_e"
FIRST_RUN = datetime(2025, 4, 1, tzinfo=UTC)
LAST_RUN = datetime(2025, 6, 30, tzinfo=UTC)


def main() -> None:
    """Build the missing frames, one run at a time, on one core."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--first", default=f"{FIRST_RUN:%Y-%m-%d}")
    parser.add_argument("--last", default=f"{LAST_RUN:%Y-%m-%d}")
    args = parser.parse_args()
    first = datetime.fromisoformat(args.first).replace(tzinfo=UTC)
    last = datetime.fromisoformat(args.last).replace(tzinfo=UTC)
    assert FIRST_RUN <= first <= last <= LAST_RUN, "outside the phase E dates"
    series, sites = common.series_table(), gen.site_table()
    observed = pl.read_parquet(common.OBSERVED_PATH)
    run = first
    while run <= last:
        target = args.out / f"init={run:%Y-%m-%d}.parquet"
        if target.exists():
            print(f"{run:%Y-%m-%d}: exists", flush=True)
        else:
            started = time.monotonic()
            demand, _generators = bf.build_run(
                run=run, series=series, sites=sites, observed=observed, want_demand=True
            )
            assert demand is not None
            bf._write_sorted(frame=demand, target=target)
            print(
                f"{run:%Y-%m-%d}: {demand.height:,} rows, {time.monotonic() - started:.0f} s",
                flush=True,
            )
        run += timedelta(days=1)


if __name__ == "__main__":
    main()
