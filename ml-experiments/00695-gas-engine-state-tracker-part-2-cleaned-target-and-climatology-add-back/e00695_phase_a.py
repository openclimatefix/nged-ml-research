# ruff: noqa: E501
r"""Phase A: fix the host Primary and describe Gas A's day blocks before any fit.

    source ml-experiments/00524*/env.sh; export PYTHONPATH=src:$PYTHONPATH
    uv run --no-sync python ml-experiments/00695-*/e00695_phase_a.py

The host is the MVA-only Primary among D2, D3, and D4 whose median reading from 2025-03-24 to
2025-10-31 is closest to D8's median over the same dates on half-hours that experiment 694's
hindsight tracker marks as no unit running. Writes `host.json`. Prints counts only, no Gas A series.
"""

import json
from datetime import UTC, datetime

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00695_common import (
    D8_LABEL,
    DATA_DIR,
    HOST_CANDIDATES,
    SHARED_DIR,
    e694_flags,
    grid_index,
)
from e00695_engine import choose_host, d8_tracked_off_median, drive_series, record_days

WINDOW: tuple[datetime, datetime] = (
    datetime(2025, 3, 24, tzinfo=UTC),
    datetime(2025, 11, 1, tzinfo=UTC),
)


def series_mva() -> dict[str, np.ndarray]:
    """Return each Primary's cleaned reading (MVA) on the pipeline grid."""
    stack = c.clean_stack()
    table = prep.primaries().select("site", "time_series_id")
    capacity = pl.read_parquet(SHARED_DIR / "series_capacity.parquet").select(
        "time_series_id", "capacity"
    )
    caps = dict(
        zip(
            table.join(capacity, on="time_series_id")["site"],
            table.join(capacity, on="time_series_id")["capacity"],
            strict=True,
        )
    )
    return {
        site: stack[i].astype(np.float64) * float(caps[site]) for i, site in enumerate(c.sites())
    }


def main() -> None:
    """Choose the host and write `host.json`."""
    series = series_mva()
    flags = e694_flags()
    n_grid = len(series[D8_LABEL])
    flag = np.zeros(n_grid, dtype=bool)
    idx = (
        (
            flags["valid_time"].dt.epoch("s").to_numpy()
            - int(datetime(2019, 9, 1, tzinfo=UTC).timestamp())
        )
        // 1800
    ).astype(int)
    flag[idx] = flags["flag"].to_numpy()
    start, stop = (grid_index(when=w) for w in WINDOW)
    target = d8_tracked_off_median(y_d8=series[D8_LABEL], flag=flag, start=start, stop=stop)
    medians = {label: float(np.nanmedian(series[label][start:stop])) for label in HOST_CANDIDATES}
    host = choose_host(medians=medians, target=target)
    print(
        f"D8 median reading on tracked-off half-hours, 2025-03-24 to 2025-10-31: {target:.3f} MVA"
    )
    for label, value in medians.items():
        print(f"  {label} median reading {value:.3f} MVA, distance {abs(value - target):.3f}")
    print(f"host: {host}")
    days = record_days()
    print(
        f"Gas A whole days: {len(days['weekday'])} weekday, {len(days['weekend'])} weekend "
        f"(of 44 days; the gap and the partial end days are dropped)"
    )
    _, units = drive_series(n_grid=n_grid)
    injected = units > 0
    window = slice(
        grid_index(when=datetime(2023, 8, 15, tzinfo=UTC)),
        grid_index(when=datetime(2026, 7, 1, tzinfo=UTC)),
    )
    print(
        f"injected series: running share {injected[window].mean():.1%} of half-hours from 2023-08-15 to 2026-06-30"
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "host.json").write_text(
        json.dumps({"host": host, "medians": medians, "target": target})
    )


if __name__ == "__main__":
    main()
