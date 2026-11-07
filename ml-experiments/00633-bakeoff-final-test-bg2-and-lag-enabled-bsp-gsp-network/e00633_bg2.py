# ruff: noqa: E501, D103, ANN001, D205
"""Experiment 633: BG2 as one fit at 2026-07-01 ("BG2-single-fit"), fitted and then forecast over the window.

BG2's weekly XGBoost member (experiment 87's configuration `n1h`): one booster per series, 500 rounds,
eta 0.05, row weights with a half-life of 0.5 years anchored at the refit, trained on the ensemble
control member's rows, each run forecast for all 51 members and pooled into 11 quantiles. The code of
experiment 87's `e00087_fit_demand.py` is imported, not copied. Differences from experiment 87's BG2,
all stated on the result page: one refit at the cutoff instead of weekly refits; only the weekly member
(no monthly, analogue, pooled, or frozen-stack member, no solar offset); training power taken from the
cleaned power grid for every row; D15 and D18 cut before 2025-03-01.

Two phases, set by `E00633_PHASE`:

- `train`: the power from 2026-07-01 on is set to NaN in every array before any row is built, so no
  code path can read a held-out reading. `fit` writes the boosters and a record of their training rows.
- `predict`: the window is open (after the go). `predict` loads the boosters and forecasts the 123 runs.

Usage: python e00633_bg2.py {fit|predict|lookahead} [--sites D2 D5 ...]
"""

import argparse
import hashlib
import json
import time

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00633_common as k
import numpy as np
import polars as pl
import xgboost as xgb

CONFIG = fd.CONFIGS["n1h"]
ARM = "bg2_single_fit"


def setup() -> tuple[pl.DataFrame, dict]:
    """Return the series table and the arrays, with the short-history cut and (train phase) the window hidden."""
    assert k.PHASE in {"train", "predict"}, "set E00633_PHASE to train or predict"
    k.redirect()
    series, arrays = fd._series_and_arrays()
    k.cut_short_history(arrays=arrays)
    if k.PHASE == "train":
        k.hide_window(arrays=arrays)
    return series, arrays


def training_rows(*, row: dict, arrays: dict) -> pl.DataFrame:
    """Return one series' control-member rows with cleaned power, derived columns, and lags.

    A copy of `fd.training_rows` that takes every row's power from the cleaned grid, not only the rows
    after 2025-06-30, so that a flagged half-hour is never a label.
    """
    series_id = row["time_series_id"]
    before = (
        pl.scan_parquet(common.ENS_TRAIN_PATH)
        .filter(pl.col("time_series_id") == series_id)
        .select(*common.KEYS, "ensemble_member", "power", *fd.WEATHER)
        .collect()
    )
    after = pl.concat(
        [
            pl.read_parquet(p).filter(pl.col("time_series_id") == series_id)
            for p in sorted(common.AFTER_DIR.glob("init=*.parquet"))
        ]
    )
    padded = after.with_columns(power=pl.lit(None, dtype=pl.Float32)).select(before.columns)
    rows = pl.concat([before, padded.cast(before.schema)])
    index = np.asarray(common.grid_index(times=rows["valid_time"]))
    rows = rows.with_columns(
        power=pl.Series(
            common.take(values=arrays.power, index=index) * row["effective_capacity_mw"]  # ty: ignore[unresolved-attribute]
        )
    )
    rows = rows.filter(
        pl.col("power").is_finite() & (pl.col("valid_time") > pl.col("power_fcst_init_time"))
    )
    rows = common.with_derived(frame=rows, latitude=row["latitude"], longitude=row["longitude"])
    rows = rows.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    return fd.add_lags(frame=rows, power=arrays.power, a7=False)  # ty: ignore[unresolved-attribute]


def fit(*, sites: list[str] | None) -> None:
    assert k.PHASE == "train"
    series, arrays = setup()
    out = k.BOOSTERS / ARM
    out.mkdir(parents=True, exist_ok=True)
    record = {}
    for row in series.iter_rows(named=True):
        site = row["site"]
        if sites and site not in sites:
            continue
        target = out / f"{site}_{k.CUTOFF:%Y%m%d}.ubj"
        if target.exists():
            continue
        started = time.monotonic()
        known = fd.known_rows(
            rows=training_rows(row=row, arrays=arrays[site]), refit=k.CUTOFF, config=CONFIG
        )
        assert known["valid_time"].max() <= k.CUTOFF - fd.regimes.REFIT_CUTOFF_AGE  # ty: ignore[unresolved-attribute]
        assert known["power_fcst_init_time"].max() <= k.CUTOFF
        if site in k.SHORT_HISTORY_SITES:
            assert known["valid_time"].min() >= k.SHORT_HISTORY_CUT
        booster = xgb.train(
            fd.parameters(config=CONFIG, seed=k.SEED, nthread=4),
            fd.train_inputs(known=known, row=row, refit=k.CUTOFF, config=CONFIG, offset=False),
            num_boost_round=CONFIG.rounds,
        )
        booster.save_model(str(target) + ".tmp")
        (out / f"{site}_{k.CUTOFF:%Y%m%d}.ubj.tmp").rename(target)
        record = {
            "site": site,
            "rows": known.height,
            "last_valid_time": str(known["valid_time"].max()),
            "last_run_origin": str(known["power_fcst_init_time"].max()),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "seconds": round(time.monotonic() - started),
        }
        (out / f"{site}_{k.CUTOFF:%Y%m%d}.json").write_text(json.dumps(record, indent=1))
        print(record, flush=True)


def forecast_run(*, run, row, arrays, model) -> pl.DataFrame:
    """Return one run's mixture quantiles (fractions of capacity) for one series, before the spread factor."""
    frame = pl.read_parquet(k.FRAMES / f"init={run:%Y-%m-%d}.parquet").filter(
        pl.col("time_series_id") == row["time_series_id"]
    )
    assert (frame["power_fcst_init_time"] - frame["nwp_init_time"]).unique().to_list() == [
        common.PUBLICATION_DELAY
    ]
    prepared = fd.prepared_run(run_rows=frame, row=row, arrays=arrays, a7=False)
    return fd.predict_variants(
        model=model, prepared=prepared, row=row, level=None, config=CONFIG, seed=k.SEED
    )[""]


def predict(*, sites: list[str] | None) -> None:
    assert k.PHASE == "predict"
    series, arrays = setup()
    out = k.ROOT / "preds" / ARM
    out.mkdir(parents=True, exist_ok=True)
    for row in series.iter_rows(named=True):
        site = row["site"]
        if sites and site not in sites:
            continue
        target = out / f"{site}.parquet"
        if target.exists():
            continue
        model = xgb.Booster()
        model.load_model(str(k.BOOSTERS / ARM / f"{site}_{k.CUTOFF:%Y%m%d}.ubj"))
        started = time.monotonic()
        tables = [
            forecast_run(run=run, row=row, arrays=arrays[site], model=model) for run in k.runs()
        ]
        frame = pl.concat(tables).with_columns(
            site=pl.lit(site), capacity=pl.lit(float(row["effective_capacity_mw"]))
        )
        common.atomic_write(frame=frame, target=target)
        print(site, frame.height, f"{time.monotonic() - started:.0f} s", flush=True)


def lookahead(*, sites: list[str] | None) -> None:
    """Row-by-row lookahead assertion on synthetic power: the forecast of an origin never reads
    power after origin - 3.5 h.

    For three runs of the window and two series, replace the power from 3.5 hours before the origin on with
    two different random values and check that the forecast is identical, and that it changes when the
    lag values it reads (at or before the newest usable half-hour) change. Uses no real held-out reading.
    """
    series, arrays = setup()
    rng = np.random.default_rng(633)
    for site in sites or ["D5", "D13"]:
        row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
        model = xgb.Booster()
        model.load_model(str(k.BOOSTERS / ARM / f"{site}_{k.CUTOFF:%Y%m%d}.ubj"))
        for run in (k.runs()[0], k.runs()[60], k.runs()[-1]):
            origin = common.grid_index(times=run + common.PUBLICATION_DELAY)
            newest = origin - common.NEWEST_AGE
            power = arrays[site].power
            saved = power.copy()
            power[:] = rng.uniform(0.2, 0.8, size=power.shape).astype(np.float32)
            base = forecast_run(run=run, row=row, arrays=arrays[site], model=model)
            keep = power[: newest + 1].copy()
            power[newest + 1 :] = rng.uniform(0.2, 0.8, size=power[newest + 1 :].shape)
            again = forecast_run(run=run, row=row, arrays=arrays[site], model=model)
            assert base.equals(again), (
                f"{site} {run}: a reading after the newest usable half-hour changed the forecast"
            )
            power[: newest + 1] = rng.uniform(0.2, 0.8, size=keep.shape)
            different = forecast_run(run=run, row=row, arrays=arrays[site], model=model)
            assert not base.equals(different), (
                f"{site} {run}: the lags do not read the permitted readings"
            )
            power[:] = saved
            print("lookahead ok", site, run.date(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("fit", "predict", "lookahead"))
    p.add_argument("--sites", nargs="+", default=None)
    a = p.parse_args()
    {"fit": fit, "predict": predict, "lookahead": lookahead}[a.command](sites=a.sites)
