# ruff: noqa: D103, E501
"""Experiment 654, Step 0: transition width at the start and end of the verified transfers (CPU)."""

import json
from datetime import UTC, datetime

import e00583_common as c  # ty: ignore[unresolved-import]
import e00654_common as c654  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

G, S = c.k578.prep.GRID_START_US, c.k578.prep.STEP_US
SUB = pl.read_csv(
    c654.ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07" / "substations.csv"
)


def idx(s: str) -> int:
    return int((datetime.fromisoformat(s).replace(tzinfo=UTC).timestamp() * 1e6 - G) // S)


def edge(*, x: np.ndarray, around: int) -> dict:
    """Return the step and its 10-90% width in half-hours near the largest jump within 48 h of `around`."""
    win = np.arange(around - 96, around + 96)
    d = np.diff(x[win])
    j = int(win[int(np.nanargmax(np.abs(d)))]) + 1
    before = np.nanmedian(x[j - 24 : j - 2])
    after = np.nanmedian(x[j + 2 : j + 24])
    step = float(after - before)
    path = (x[j - 6 : j + 7] - before) / step if step else np.full(13, np.nan)
    inside = np.flatnonzero((path > 0.1) & (path < 0.9))
    return {
        "time": str(datetime.fromtimestamp((G + j * S) / 1e6, UTC)),
        "step": step,
        "width_half_hours": len(inside),
        "largest_single_half_hour_share_of_step": float(
            np.nanmax(np.abs(np.diff(x[j - 6 : j + 7]))) / abs(step)
        )
        if step
        else None,
    }


def main() -> None:
    cl = c.k578.clean()
    sm = c.k578.sites()
    pm = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    tid = dict(zip(SUB["primary_name"].to_list(), SUB["time_series_id"].to_list(), strict=True))
    out = {}
    for pair, names, (start, end) in (
        ("D5/D16", ("D5", "D16"), ("2025-12-29T12:00", "2026-01-13T12:00")),
        ("D3/D22", ("D3", "D22"), ("2024-10-26T12:00", "2024-11-22T12:00")),
    ):
        for name in names:
            if name not in tid or tid[name] not in pm:
                out[f"{pair} {name}"] = "not among the 16 Primaries"
                continue
            x = cl[pm[tid[name]]].astype(float)
            out[f"{pair} {name}"] = {
                "start": edge(x=x, around=idx(start)),
                "end": edge(x=x, around=idx(end)),
            }
    p = c654.WORK / "score" / "edges.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


main()
