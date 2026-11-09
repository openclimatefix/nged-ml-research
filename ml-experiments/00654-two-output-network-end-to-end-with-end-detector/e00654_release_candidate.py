# ruff: noqa: E501
"""Experiment 654: write the release-candidate backtest parquet for the reference arm E0 (CPU).

E0 is experiment 652's R-haz (mixture m50, seed mean of 2 seeds) on K1's normal-running-arrangement
(NRA) quantiles. Each development month is scored by the head fitted on the other four months. K1's
quantiles and the observed power come from experiment 647's real-row head data. Values are written
in MW (or MVA for the MVA-only Primaries): the stored fraction of capacity times `capacity_mw`.

Writes `<WORK>/release_candidate/E0.parquet` and `README.md` beside it.

Usage: python e00654_release_candidate.py
"""

from datetime import timedelta
from typing import Final

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00647_common as c647  # ty: ignore[unresolved-import]
import e00650_common as c650  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00654-two-output-network-end-to-end-with-end-detector"
WORK: Final = ML_EXPERIMENTS_DATA_DIR / NAME
D647: Final = (
    c650.WORK.parent / "00647-two-output-forecast-departure-head-on-frozen-k1" / "head_data"
)
D652: Final = (
    c650.WORK.parent / "00652-two-output-forecast-with-explicit-rules-and-survival-hazard" / "fits"
)
QNAMES: Final = [f"q{int(round(t * 100)):02d}" for t in c647.TAUS]
SEEDS: Final = (0, 1)


def month_frame(*, month: str, sites: pl.DataFrame) -> pl.DataFrame:
    """Return E0's real-row forecasts for one scored month, in MW."""
    z = np.load(D647 / f"real_{month}.npz")
    fits = [np.load(D652 / f"Rhaz_m50_{month}_s{s}.npz") for s in SEEDS]
    idx = fits[0]["real_idx"]
    assert all(np.array_equal(f["real_idx"], idx) for f in fits)
    prev = np.sort(np.mean([f["real_q"] for f in fits], axis=0), axis=1)
    surv = np.mean([f["real_p"] for f in fits], axis=0)
    nra = np.sort(z["q"][idx], axis=1)
    cap = sites["capacity_mw"].to_numpy()[z["prim"][idx].astype(int)]
    ts_id = sites["time_series_id"].to_numpy()[z["prim"][idx].astype(int)]
    grid0 = k578.GRID_START
    origin = [grid0 + timedelta(minutes=30 * int(i)) for i in z["oi"][idx]]
    valid = [grid0 + timedelta(minutes=30 * int(i)) for i in z["vi"][idx]]
    cols: dict[str, np.ndarray | list] = {
        "time_series_id": ts_id.astype(np.int64),
        "power_fcst_init_time": origin,
        "valid_time": valid,
        "lead_hours": z["lead"][idx].astype(np.float32),
        "scored_month": np.full(len(idx), month),
    }
    for j, name in enumerate(QNAMES):
        cols[f"nra_{name}"] = (nra[:, j] * cap).astype(np.float32)
    for j, name in enumerate(QNAMES):
        cols[f"prevailing_{name}"] = (prev[:, j] * cap).astype(np.float32)
    cols["survival_probability"] = surv.astype(np.float32)
    cols["observed_power"] = (z["y"][idx] * cap).astype(np.float32)
    cols["flag_event_free"] = z["event_free"][idx]
    cols["flag_verified_transfer"] = z["verified"][idx]
    cols["flag_substation_zero"] = z["flagged"][idx]
    return pl.DataFrame(cols).with_columns(
        pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
    )


README: Final[str] = """# Release candidate backtest: experiment 654

Backtest forecasts for the 16 Primary substations on five development months (2025-10, 2025-12,
2026-01, 2026-03, and 2026-06), as one parquet file per arm. The months 2026-07 to 2026-10 are not
in these files.

## What each forecast is

- **NRA forecast (`nra_q*`).** The demand the Primary would draw in its normal running arrangement
  (NRA), with no load shift. It ignores shifts on purpose: while a shift is in force, the observed
  power is expected to differ from it.
- **Prevailing forecast (`prevailing_q*`).** The demand expected under the arrangement in force at the
  forecast origin. It carries a shift that is in force at the origin forward, and a survival
  probability (`survival_probability`, the probability that the shift is still on at `valid_time`)
  pulls it back to the NRA forecast over the horizon.

## What each is scored against

- The NRA forecast is scored against the observed power on rows outside load shifts
  (`flag_event_free` true). NGED's shift labels are an incomplete one-off dump, so rows flagged
  event-free can still hold unlabelled shifts.
- The prevailing forecast is scored against the observed power on all rows.
- Rows with `flag_substation_zero` true are half-hours the cleaning rule flags as an exact zero;
  drop them before scoring. `flag_verified_transfer` marks the verified D5/D16 and D3/D22 transfers.

## How to read the columns

- `q01` to `q99` are the quantile levels 0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90,
  0.95, and 0.99. Each row's 11 values are sorted, so `q50` is the median forecast.
- Powers are in MW for the signed-MW Primaries and in MVA for the MVA-only Primaries.
- `power_fcst_init_time` is the forecast origin, `valid_time` the half-hour forecast, and
  `lead_hours` the gap between them (24 to 240 hours).
- `scored_month` is the development month the row belongs to. Each month's forecast comes from a
  head fitted on the other four months, so no row is scored by a head that saw its own month.

## Caveat

The backtest covers development months only, five months in total. Intervals computed from five
months are coarse.
"""


def main() -> None:
    """Write the E0 file and the README."""
    out = WORK / "release_candidate"
    out.mkdir(parents=True, exist_ok=True)
    sites = k578.sites()
    df = pl.concat([month_frame(month=m, sites=sites) for m in c650.MONTHS])
    df.write_parquet(out / "E0.parquet")
    (out / "README.md").write_text(README)
    print(df.shape, df["time_series_id"].n_unique(), "series")
    print(df.group_by("scored_month").len().sort("scored_month"))


if __name__ == "__main__":
    main()
