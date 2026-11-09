# ruff: noqa: E501, D102, D107, ANN201, ANN202, PT018, F841, PERF401, B007, RUF100
"""Experiment 654: write the release-candidate backtest parquet files, one per arm (CPU).

E0 is experiment 652's R-haz (mixture m50, seed mean of 2 seeds) on K1's
normal-running-arrangement (NRA) quantiles. Each development month is scored by the head fitted
on the other four months. K1's quantiles and the observed power come from experiment 647's
real-row head data. Values are written in MW (or MVA for the MVA-only Primaries): the stored
fraction of capacity times `capacity_mw`.

Writes `<WORK>/release_candidate/<arm>.parquet` for E0, J2, J1, and C0, and `README.md` beside
them.

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
QNAMES: Final = [f"q{round(t * 100):02d}" for t in c647.TAUS]
SEEDS: Final = (0, 1)
FITS: Final = WORK / "fits"
ARMS: Final = ("E0", "J2", "J1", "C0")


def joint_arm(*, arm: str, month: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (row index, NRA, prevailing, survival) of a joint arm's real rows, seed mean, both cuts."""
    per_seed = []
    for s in SEEDS:
        parts = [np.load(FITS / f"{arm}_{month}_d{d}_s{s}.npz") for d in (0, 14)]
        idx = np.concatenate([p["real_idx"] for p in parts])
        o = np.argsort(idx)
        per_seed.append(
            (
                idx[o],
                *[np.concatenate([p[f"real_{k}"] for p in parts])[o] for k in ("nra", "q", "p")],
            )
        )
    assert all(np.array_equal(per_seed[0][0], p[0]) for p in per_seed)
    return (per_seed[0][0], *[np.mean([p[i] for p in per_seed], axis=0) for i in (1, 2, 3)])


def month_frame(*, month: str, sites: pl.DataFrame, arm: str = "E0") -> pl.DataFrame:
    """Return an arm's real-row forecasts for one scored month, in MW."""
    z = np.load(D647 / f"real_{month}.npz")
    if arm == "E0":
        fits = [np.load(D652 / f"Rhaz_m50_{month}_s{s}.npz") for s in SEEDS]
        idx = fits[0]["real_idx"]
        assert all(np.array_equal(f["real_idx"], idx) for f in fits)
        prev = np.sort(np.mean([f["real_q"] for f in fits], axis=0), axis=1)
        surv = np.mean([f["real_p"] for f in fits], axis=0)
        nra = np.sort(z["q"][idx], axis=1)
    else:
        idx, nra_, prev_, surv = joint_arm(arm=arm, month=month)
        nra, prev = np.sort(nra_, axis=1), np.sort(prev_, axis=1)
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

## The files

- `E0.parquet`: K1 (the normal-running-arrangement network of experiments 612 and 613) plus the
  survival-hazard head of experiment 652. Best arm on injected shifts in the development months.
- `J2.parquet`: one network, fine-tuned from K1's weights with the prevailing loss reaching the lag
  encoder, with no gate. The end-to-end arm.
- `J1.parquet`: as J2 with the prevailing loss detached from the lag encoder.
- `C0.parquet`: the same fine-tune with no prevailing head. It has the NRA columns only.

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
    for arm in ARMS:
        df = pl.concat([month_frame(month=m, sites=sites, arm=arm) for m in c650.MONTHS])
        if arm == "C0":  # no prevailing head: only the NRA columns are meaningful
            df = df.drop(
                [
                    c
                    for c in df.columns
                    if c.startswith("prevailing_") or c == "survival_probability"
                ]
            )
        df.write_parquet(out / f"{arm}.parquet")
        print(arm, df.shape, df["time_series_id"].n_unique(), "series")
    (out / "README.md").write_text(README)


if __name__ == "__main__":
    main()
