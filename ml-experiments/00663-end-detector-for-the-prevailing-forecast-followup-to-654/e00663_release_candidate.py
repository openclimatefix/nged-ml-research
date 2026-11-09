# ruff: noqa: D103, E501
"""Experiment 663: write the release-candidate backtest parquet files, one per arm (CPU).

Arms `E0`, `E0d`, `J2d` (seed mean of 2 seeds). Development months: each month is scored by a
head or detector fitted on the other four months. The window file (`<arm>_window.parquet`) holds
the final-fold arms' forecasts for 2026-07-01 to 2026-10-31 and is written only after
`final/freeze2.json` exists. Values are in MW (MVA for the MVA-only Primaries): the stored
fraction of capacity times `capacity_mw`.

Usage: python e00663_release_candidate.py [--arms E0 E0d J2d] [--window]
"""

import argparse
from datetime import timedelta

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00647_common as c647  # ty: ignore[unresolved-import]
import e00650_common as c650  # ty: ignore[unresolved-import]
import e00650_fit as f650  # ty: ignore[unresolved-import]
import e00663_common as c663
import e00663_score as sc
import numpy as np
import polars as pl

QNAMES = [f"q{round(t * 100):02d}" for t in c647.TAUS]
OUT = c663.WORK / "release_candidate"
README = """# Release candidate backtest: experiment 663 (end detector for the prevailing forecast)

Backtest forecasts for the 16 Primary substations, one parquet file per arm. The files
`<arm>.parquet` cover five development months (2025-10, 2025-12, 2026-01, 2026-03, and 2026-06).
The files `<arm>_window.parquet` cover the final-test window (2026-07-01 to 2026-10-31).

## The arms

- `E0`: K1 (the normal-running-arrangement network of experiments 612 and 613) plus the
  survival-hazard head of experiment 652.
- `E0d`: `E0` plus an end detector. The detector reads the freshest half-hours of observed power
  minus K1's forecast, at the Primary and at its electrical neighbours, and outputs the
  probability that a load shift has ended (`end_probability`). The survival is multiplied by
  one minus that probability.
- `J2d`: the end-to-end network of experiment 654 (`J2`) plus the same end detector.

## What each forecast is

- **NRA forecast (`nra_q*`).** The demand the Primary would draw in its normal running arrangement,
  with no load shift. The detector arms leave it unchanged.
- **Prevailing forecast (`prevailing_q*`).** The demand expected under the arrangement in force at
  the forecast origin. A shift in force at the origin is carried forward and a survival probability
  (`survival_probability`, the probability that the shift is still on at `valid_time`) pulls the
  forecast back to the NRA forecast over the horizon.

## What each is scored against

- The NRA forecast is scored against observed power on rows outside load shifts (`flag_event_free`).
  NGED's shift labels are an incomplete one-off dump, so event-free rows can still hold unlabelled
  shifts.
- The prevailing forecast is scored against observed power on all rows.
- Drop rows with `flag_substation_zero` true (the cleaning rule's exact zeros) before scoring.

## How to read the columns

- `q01` to `q99` are the quantile levels 0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90,
  0.95, and 0.99, sorted within each row. `q50` is the median forecast.
- Powers are in MW for signed-MW Primaries and in MVA for MVA-only Primaries.
- `power_fcst_init_time` is the forecast origin, `valid_time` the half-hour forecast, `lead_hours`
  the gap (24 to 240 hours).
- `scored_period` is the development month of the row, or `window`. Each development month's forecast
  comes from a head and detector fitted on the other four months. The window files come from a fit
  on all five development months.

## Caveats

Development months total five months, so intervals are coarse. The window files are the final-fold
fit read once.
"""


def frame(
    *,
    z: dict,
    idx: np.ndarray,
    nra: np.ndarray,
    prev: np.ndarray,
    surv: np.ndarray,
    qend: np.ndarray,
    label: str,
    sites: pl.DataFrame,
) -> pl.DataFrame:
    prim = z["prim"][idx].astype(int)
    cap = sites["capacity_mw"].to_numpy()[prim]
    at = lambda i: [k578.GRID_START + timedelta(minutes=30 * int(x)) for x in i]  # noqa: E731
    cols: dict = {
        "time_series_id": sites["time_series_id"].to_numpy()[prim].astype(np.int64),
        "power_fcst_init_time": at(z["oi"][idx]),
        "valid_time": at(z["vi"][idx]),
        "lead_hours": z["lead"][idx].astype(np.float32),
        "scored_period": np.full(len(idx), label),
    }
    for j, n in enumerate(QNAMES):
        cols[f"nra_{n}"] = (np.sort(nra, axis=1)[:, j] * cap).astype(np.float32)
    for j, n in enumerate(QNAMES):
        cols[f"prevailing_{n}"] = (np.sort(prev, axis=1)[:, j] * cap).astype(np.float32)
    cols["survival_probability"] = surv.astype(np.float32)
    cols["end_probability"] = qend.astype(np.float32)
    cols["observed_power"] = (z["y"][idx] * cap).astype(np.float32)
    cols["flag_event_free"] = z["event_free"][idx]
    cols["flag_verified_transfer"] = z["verified"][idx]
    cols["flag_substation_zero"] = z["flagged"][idx]
    return pl.DataFrame(cols).with_columns(
        pl.col("power_fcst_init_time").dt.replace_time_zone("UTC"),
        pl.col("valid_time").dt.replace_time_zone("UTC"),
    )


def month_frame(*, arm: str, month: str, sites: pl.DataFrame) -> pl.DataFrame:
    z = f650.sources(month=month, bank="new")["real"]
    rows = []
    for s in sc.SEEDS:
        r = sc.arm_rows(arm=arm, month=month, seed=s, setname="real", z=z)
        folder = "fits_e0" if arm.startswith("E0") else "fits_j2"
        qe = (
            np.load(c663.WORK / folder / f"{arm}_{month}_s{s}.npz")["real_q_end"]
            if arm in ("E0d", "J2d")
            else np.zeros(len(r["idx"]), np.float32)
        )
        rows.append((r, qe))
    idx = rows[0][0]["idx"]
    return frame(
        z=z, idx=idx, nra=np.mean([r["nra"] for r, _ in rows], axis=0), prev=np.mean([r["q"] for r, _ in rows], axis=0),
        surv=np.mean([r["p"] for r, _ in rows], axis=0), qend=np.mean([q for _, q in rows], axis=0), label=month, sites=sites,
    )  # fmt: skip


def window_frame(*, arm: str, sites: pl.DataFrame) -> pl.DataFrame:
    z = f650.load(path=c663.WORK / "window" / "real_window.npz", kind="real")
    folder = "fits_e0" if arm.startswith("E0") else "fits_j2"
    fs = [np.load(c663.WORK / folder / f"{arm}_final_s{s}.npz") for s in sc.SEEDS]
    idx = fs[0]["window_idx"]
    nra = z["q"][idx] if arm.startswith("E0") else np.mean([f["window_nra"] for f in fs], axis=0)
    qe = (
        np.mean([f["window_q_end"] for f in fs], axis=0)
        if arm in ("E0d", "J2d")
        else np.zeros(len(idx), np.float32)
    )
    return frame(
        z=z, idx=idx, nra=nra, prev=np.mean([f["window_q"] for f in fs], axis=0), surv=np.mean([f["window_p"] for f in fs], axis=0),
        qend=qe, label="window", sites=sites,
    )  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["E0", "E0d", "J2d"])
    ap.add_argument("--window", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sites = k578.sites()
    for arm in a.arms:
        df = pl.concat([month_frame(arm=arm, month=m, sites=sites) for m in c650.MONTHS])
        df.write_parquet(OUT / f"{arm}.parquet")
        print(arm, df.shape, flush=True)
        if a.window:
            dw = window_frame(arm=arm, sites=sites)
            dw.write_parquet(OUT / f"{arm}_window.parquet")
            print(arm, "window", dw.shape, flush=True)
    (OUT / "README.md").write_text(README)


if __name__ == "__main__":
    main()
