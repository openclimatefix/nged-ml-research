"""Write `primaries_calibrated.parquet`: experiment 683's arm 2 rule on the prevailing rows.

Run from the repository root with the output folder as the only argument:

    PYTHONPATH=ml-experiments/00668-*:ml-experiments/00683-* uv run --no-sync python \
        ml-experiments/00689-*/calibrated_copies/e00689_make_calibrated_primaries.py <output-folder>

The normal-running rows are copied unchanged. The prevailing rows get one scale per series, origin
month, and lead band, fitted by `fit_month` of `e00683_calibrate.py` (90 days of valid time ending
6 hours before the month starts) on the prevailing rows of the stored file, then applied with the
zero clamp of the MVA Primaries. Primaries 13 and 16 keep their 2026-06 scale from 2026-07 on (a
rule chosen after seeing those months, so post hoc), and 2025-11 is left as stored. Also writes
`primaries_calibrated_scales.parquet` beside the copy. This is the copy described under
"Calibrated copies" in the backtest folder's README.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00683_calibrate as cal  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

HOLD_IDS: Final[tuple[int, ...]] = (13, 16)
HOLD_FROM: Final[str] = "2026-07"
HOLD_SOURCE: Final[str] = "2026-06"
N_ROW_GROUPS: Final[int] = 64


def main(*, out_dir: Path) -> None:
    """Fit the prevailing scales, apply them, and write the copy and the scales table."""
    full = pl.read_parquet(lib.FOLDER / "primaries.parquet")
    prevailing_rows = np.flatnonzero((full["NRA_or_prevailing"] == "prevailing").to_numpy())
    forecast = full.filter(pl.col("NRA_or_prevailing") == "prevailing")
    ids = sorted(forecast["time_series_id"].unique().to_list())
    index = {t: i for i, t in enumerate(ids)}
    capacity = (
        pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
        .filter(pl.col("time_series_id").is_in(ids))
        .sort("time_series_id")
    )
    cap = capacity["capacity"].to_numpy().astype(np.float64)
    nonneg = np.array([u == "MVA" for u in capacity["power_unit"].to_list()])
    grid = cal.load_truth(name="primaries", ids=ids)
    row = forecast["time_series_id"].replace_strict(index).to_numpy()
    vidx = lib.grid_index(times=forecast["valid_time"])
    oidx = lib.grid_index(times=forecast["power_fcst_init_time"])
    if vidx.max() >= grid.shape[1]:
        grid = np.pad(
            grid, ((0, 0), (0, int(vidx.max()) + 1 - grid.shape[1])), constant_values=np.nan
        )
    band = cal.band_index(lead_h=(vidx - oidx) / 2.0)
    scale = 100.0 / cap[row]
    y = grid[row, vidx] * scale
    q = forecast.select(lib.QCOLS).to_numpy().astype(np.float64) * scale[:, None]
    ok = np.isfinite(y) & np.isfinite(q).all(axis=1)
    y = np.where(ok, y, 0.0)
    origin_month = forecast["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    months = sorted(set(origin_month))
    fits = {}
    for m in months:
        start = pl.Series([datetime(int(m[:4]), int(m[5:]), 1, tzinfo=UTC)])
        cutoff = int(lib.grid_index(times=start)[0]) - cal.DELAY_HALF_HOURS
        fits[m] = cal.fit_month(
            y=y, q=q, vidx=vidx, row=row, band=band, ok=ok, cutoff_v=cutoff, n_series=len(ids)
        )
    out = q.copy()
    scale_rows = []
    for m in months:
        sel = np.flatnonzero(origin_month == m)
        factors = fits[m]["scale2"].copy()
        if m >= HOLD_FROM:
            for tid in HOLD_IDS:
                factors[index[tid]] = fits[HOLD_SOURCE]["scale2"][index[tid]]
        if m == months[0]:
            factors[:] = 1.0
        out[sel] = cal.apply_arm(
            q=q[sel],
            factor=factors[row[sel], band[sel]],
            shift=np.zeros(len(sel)),
            floor_zero=nonneg[row[sel]],
        )
        scale_rows.extend(
            (tid, m, k, float(factors[i, k])) for i, tid in enumerate(ids) for k in range(4)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        scale_rows, schema=["time_series_id", "origin_month", "band", "scale"], orient="row"
    ).write_parquet(out_dir / "primaries_calibrated_scales.parquet")
    new_q = (out / scale[:, None]).astype(np.float32)
    columns = {name: full[name].to_numpy().copy() for name in lib.QCOLS}
    for k, name in enumerate(lib.QCOLS):
        columns[name][prevailing_rows] = new_q[:, k]
    result = full.with_columns([pl.Series(n, columns[n]).cast(pl.Float32) for n in lib.QCOLS])
    result.write_parquet(
        out_dir / "primaries_calibrated.parquet",
        compression="zstd",
        row_group_size=-(-full.height // N_ROW_GROUPS),
    )
    print("written", result.shape)


if __name__ == "__main__":
    main(out_dir=Path(sys.argv[1]))
