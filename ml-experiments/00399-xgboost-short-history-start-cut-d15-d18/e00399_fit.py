"""Refit experiment 381's monthly XGBoost control at D15 and D18 with a cut training start (399).

Each arm is experiment 381's `control` recipe (via its `run_job`), with two changes made by patching
experiment 87's `e00087_fit_demand` module in every worker:

- Power comes from `studies.power.scan_power` (valid times before 2026-07-01), not from the stored
  `observed.parquet`. For NCZ and NCR the removed half-hours are set to NaN in the power array, so
  they leave the training labels, and the three origin lags read them as missing.
- `known_rows` also drops the rows with valid time before the arm's cut.

Arms: `N3`, `N6` (first 3 or 6 calendar months after the first reading), `NS` (the
`new_meter_settling` rule of `studies/quick_wins/cleaning_rules.py`), `NC` (2025-03-01), `NCZ`
(NC and every reading with an absolute value of 0.009 MW or less removed), and `NCR` (NC and only
runs of 2 or more such half-hours removed). Predictions go to `work/arm_preds/monthly/<arm>/`.

```bash
PYTHONPATH=ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:studies/quick_wins \
    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00399-*/e00399_fit.py \
    --arm NC --cores 3
```
"""

import argparse
import dataclasses
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00381_fit as base  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from cleaning_rules import new_meter_settling  # ty: ignore[unresolved-import]
from studies.power import scan_power
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00399-xgboost-short-history-start-cut-d15-d18" / "work"
)
ARM_DIR: Final[Path] = WORK_DIR / "arm_preds" / "monthly"
ARMS: Final[tuple[str, ...]] = ("N3", "N6", "NS", "NC", "NCZ", "NCR")
GATE_ARM: Final[str] = "N0"
"""Refits the control with no cut and no removal in this environment: the reference for every
contrast. A first single-fit check against the stored files lives in `arm_preds_gate`."""
SITES: Final[tuple[str, ...]] = ("D15", "D18")
FIRST_REFIT: Final[dict[str, datetime]] = {
    "D15": datetime(2026, 3, 1, tzinfo=UTC),
    "D18": datetime(2026, 1, 1, tzinfo=UTC),
}
"""The first monthly refit whose runs the protocol scores, per Primary."""
POWER_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
CUT_NC: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
NEAR_ZERO_MW: Final[float] = 0.009
NEAR_ZERO_TOLERANCE_MW: Final[float] = 1e-6
TRAIN_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
ARM_ENV: Final[str] = "E399_ARM"
EXPECTED_SETTLING_CUT: Final[dict[str, str]] = {"D15": "2025-03-20", "D18": "2025-03-04"}
"""The `new_meter_settling` cut dates the brief quotes, checked to the day by `--check`."""


def scanned_power(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the cleaned half-hourly power (MW) of D15 and D18 before 2026-07-01.

    Reads power only through `studies.power.scan_power`.
    """
    ids = series.filter(pl.col("site").is_in(SITES))["time_series_id"].to_list()
    return (
        scan_power()
        .filter(pl.col("time_series_id").is_in(ids) & (pl.col("time") < POWER_END))
        .select("time_series_id", "time", "power")
        .sort("time_series_id", "time")
        .collect()
    )


def removed_times(*, power: pl.DataFrame, arm: str) -> pl.DataFrame:
    """Return the `(time_series_id, time)` pairs NCZ or NCR removes (none for the other arms)."""
    if arm not in {"NCZ", "NCR"}:
        return power.head(0).select("time_series_id", "time")
    near = (pl.col("power").abs() <= NEAR_ZERO_MW + NEAR_ZERO_TOLERANCE_MW) & (
        pl.col("time") >= CUT_NC
    )
    frame = power.sort("time_series_id", "time").with_columns(near=near)
    if arm == "NCR":
        step = pl.col("time").diff().over("time_series_id") == pl.duration(minutes=30)
        # A half-hour joins a run if it is near zero and a neighbour 30 minutes away is too.
        previous = (pl.col("near").shift(1) & step).over("time_series_id")
        following = (
            pl.col("near").shift(-1)
            & (pl.col("time").diff(-1).over("time_series_id") == pl.duration(minutes=-30))
        ).over("time_series_id")
        frame = frame.with_columns(near=pl.col("near") & (previous | following).fill_null(False))
    return frame.filter(pl.col("near")).select("time_series_id", "time")


def cut_times(*, power: pl.DataFrame, series: pl.DataFrame, arm: str) -> dict[str, datetime]:
    """Return the first valid time each Primary's training rows keep under `arm`."""
    out: dict[str, datetime] = {}
    for row in series.filter(pl.col("site").is_in(SITES)).iter_rows(named=True):
        mine = power.filter(pl.col("time_series_id") == row["time_series_id"])
        first: datetime = mine["time"].min()  # ty: ignore[invalid-assignment]
        if arm in {"N3", "N6"}:
            months = int(arm[1:])
            out[row["site"]] = pl.select(pl.lit(first).dt.offset_by(f"{months}mo")).item()
        elif arm == "NS":
            dropped = new_meter_settling(power=power, train_start=TRAIN_START).filter(
                pl.col("time_series_id") == row["time_series_id"]
            )
            latest: datetime = dropped["time"].max()  # ty: ignore[invalid-assignment]
            out[row["site"]] = latest + timedelta(minutes=30)
        elif arm == GATE_ARM:
            out[row["site"]] = datetime(2000, 1, 1, tzinfo=UTC)
        else:
            out[row["site"]] = CUT_NC
    return out


def patch_arrays(*, arm: str) -> None:
    """Replace D15 and D18's power arrays with `scan_power`'s, and patch the row builders."""
    original_series_and_arrays = fd._series_and_arrays
    original_training_rows = fd.training_rows
    original_known_rows = fd.known_rows

    def series_and_arrays() -> tuple[pl.DataFrame, dict[str, common.SeriesArrays]]:
        first_call = "series" not in fd._CACHE
        series, arrays = original_series_and_arrays()
        if first_call:
            scanned = scanned_power(series=series)
            fd._CACHE["cuts"] = cut_times(power=scanned, series=series, arm=arm)
            gone = removed_times(power=scanned, arm=arm)
            power = scanned.join(gone, on=["time_series_id", "time"], how="anti")
            for row in series.filter(pl.col("site").is_in(SITES)).iter_rows(named=True):
                mine = power.filter(pl.col("time_series_id") == row["time_series_id"])
                grid = np.full(len(arrays[row["site"]].power), np.nan, dtype=np.float32)
                index = np.asarray(common.grid_index(times=mine["time"]))
                grid[index] = (mine["power"] / row["effective_capacity_mw"]).to_numpy()
                arrays[row["site"]] = dataclasses.replace(arrays[row["site"]], power=grid)
        return series, arrays

    def training_rows(
        *, row: dict, arrays: common.SeriesArrays, frozen: bool, a7: bool
    ) -> pl.DataFrame:
        rows = original_training_rows(row=row, arrays=arrays, frozen=frozen, a7=a7)
        index = np.asarray(common.grid_index(times=rows["valid_time"]))
        label = common.take(values=arrays.power, index=index) * row["effective_capacity_mw"]
        return rows.with_columns(power=pl.Series(label).cast(pl.Float32)).filter(
            pl.col("power").is_finite()
        )

    def known_rows(*, rows: pl.DataFrame, refit: datetime, config: object) -> pl.DataFrame:
        fd._series_and_arrays()
        site = _site_of(rows=rows)
        known = original_known_rows(rows=rows, refit=refit, config=config)
        return known.filter(pl.col("valid_time") >= fd._CACHE["cuts"][site])

    fd._series_and_arrays = series_and_arrays
    fd.training_rows = training_rows
    fd.known_rows = known_rows


def _site_of(*, rows: pl.DataFrame) -> str:
    """Return the site label of a frame of one series' rows."""
    series = fd._CACHE["series"]
    series_id = rows["time_series_id"][0] if "time_series_id" in rows.columns else None
    return series.filter(pl.col("time_series_id") == series_id)["site"].item()


if ARM_ENV in os.environ:
    patch_arrays(arm=os.environ[ARM_ENV])


def main() -> None:
    """Fit one arm at the scored refits of D15 and D18."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=(*ARMS, GATE_ARM), required=True)
    parser.add_argument("--cores", type=int, default=3)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--smoke", action="store_true", help="1 refit per Primary, 20 rounds")
    args = parser.parse_args()
    if os.environ.get(ARM_ENV) != args.arm:
        msg = f"set {ARM_ENV}={args.arm} before launching (workers read it at import)"
        raise SystemExit(msg)
    jobs = regimes.jobs_for(regime_names=("monthly",), origins=common.all_origins())
    out = ARM_DIR / args.arm
    rounds = common.N_ROUNDS
    todo = [
        (refit, runs, [s for s in SITES if refit >= FIRST_REFIT[s]])
        for refit, runs in jobs.items()
        if any(refit >= FIRST_REFIT[s] for s in SITES)
    ]
    if args.smoke:
        todo = todo[:1]
        rounds = base.SMOKE_ROUNDS
        out = WORK_DIR / "arm_preds_smoke" / "monthly" / args.arm
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.cores, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                base.run_job,
                arm_name="control",
                refit=refit,
                origins=runs,
                sites=sites,
                nthread=1,
                rounds=rounds,
                seed=args.seed,
                out_dir=out,
            )
            for refit, runs, sites in todo
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s ({out})")


if __name__ == "__main__":
    main()
