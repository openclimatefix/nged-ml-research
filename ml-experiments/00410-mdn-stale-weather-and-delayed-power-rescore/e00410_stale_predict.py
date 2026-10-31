"""Forecast the stored networks of experiment 397 with stale weather or delayed power (410).

Reads the pooled and per-Primary plain networks of experiment 397 (summary scheme, five screen
months, three seeds) and forecasts every test origin under the arms below, writing one forecast
file per network, arm, month, and seed under this experiment's private folder. Nothing is fitted.

| Arm | Weather forecast | Power readings |
|---|---|---|
| FRESH_RAW | run of the origin | unfiltered actuals, as in experiment 397 (the reproduction gate) |
| FRESH | run of the origin | cleaned (flagged half-hours missing) |
| W24, W48 | run one or two days earlier, matched on valid time | cleaned |
| P6, P12, P24 | run of the origin | origin of the lags moved back 6, 12, 24 hours |
| DROP | run of the origin | the three origin lags missing |
| W24_P12, W24_P24 | previous day's run | origin moved back 12, 24 hours |

The lead input is always measured from the forecast origin. A delayed origin changes the lag values
and the lag-age inputs, not the lead.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand \
OMP_NUM_THREADS=2 systemd-run --user --scope -p MemoryMax=4G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00410-*/e00410_stale_predict.py --device cpu
```
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00410-mdn-stale-weather-and-delayed-power-rescore" / "work"
)
FLAGGED_KEYS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00398-load-shift-event-rescore-without-flagged-zero-actuals"
    / "flagged_keys.parquet"
)
MODELS_DIR: Final[Path] = prep.WORK_DIR / "work" / "models" / "summary"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
NETWORKS: Final[dict[str, str]] = {"pooled": "plain_pooled", "per_primary": "plain_own"}
SHORT_HISTORY_CUT: Final[datetime] = datetime(2025, 3, 1, tzinfo=prep.FIRST_RUN.tzinfo)
SHORT_HISTORY_LABELS: Final[tuple[str, ...]] = ("D15", "D18")


@dataclass(frozen=True)
class Arm:
    """One pairing of weather-forecast age and power-reading delay."""

    name: str
    weather_days: int = 0
    power_hours: int = 0
    drop_lags: bool = False
    raw_power: bool = False


ARMS: Final[tuple[Arm, ...]] = (
    Arm("FRESH_RAW", raw_power=True),
    Arm("FRESH"),
    Arm("W24", weather_days=1),
    Arm("W48", weather_days=2),
    Arm("P6", power_hours=6),
    Arm("P12", power_hours=12),
    Arm("P24", power_hours=24),
    Arm("DROP", drop_lags=True),
    Arm("W24_P12", weather_days=1, power_hours=12),
    Arm("W24_P24", weather_days=1, power_hours=24),
)


def cleaned_power(*, arrays: dict[str, common.SeriesArrays], series: pl.DataFrame) -> dict:
    """Return each site's power array with the flagged `substation_zero` half-hours set to NaN."""
    flagged = pl.read_parquet(FLAGGED_KEYS).select("time_series_id", "valid_time")
    out = {}
    for row in series.iter_rows(named=True):
        power = arrays[row["site"]].power.copy()
        times = flagged.filter(pl.col("time_series_id") == row["time_series_id"])["valid_time"]
        index = np.asarray(common.grid_index(times=times))
        power[index[(index >= 0) & (index < power.size)]] = np.nan
        out[row["site"]] = power
    return out


def lag_matrix_with_delay(
    *, power: np.ndarray, valid: pl.Series, origin: pl.Series, delay_steps: int
) -> np.ndarray:
    """Return the three origin lags with the origin moved back by `delay_steps` half-hours."""
    target = np.asarray(common.grid_index(times=valid))
    base = np.asarray(common.grid_index(times=origin)) - delay_steps
    return common.origin_lags(power=power, target=target, origin=base)


def stale_summary(*, run: datetime, series_id: int) -> pl.DataFrame | None:
    """Return a run's weather summaries for a series matched on valid time, or `None` if absent."""
    path = prep.SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet"
    if not path.exists():
        return None
    return (
        pl.read_parquet(path)
        .filter(pl.col("time_series_id") == series_id)
        .drop("time_series_id", "power_fcst_init_time")
    )


def arm_inputs(
    *,
    arm: Arm,
    prepared: pl.DataFrame,
    run: datetime,
    row: dict,
    power: np.ndarray,
) -> tuple[pl.DataFrame, np.ndarray] | None:
    """Return the keys and raw (unstandardised) inputs of a Primary's run under an arm.

    Valid times the older weather run does not cover are dropped. Returns `None` when the older
    run is absent.
    """
    frame = prepared
    if arm.weather_days:
        stale = stale_summary(
            run=run - timedelta(days=arm.weather_days),
            series_id=row["time_series_id"],
        )
        if stale is None:
            return None
        weather = [c for c in stale.columns if c.startswith(("mean_", "std_"))]
        frame = frame.drop(weather).join(
            stale, on="valid_time", how="inner", validate="1:1", maintain_order="left"
        )
    lags = lag_matrix_with_delay(
        power=power,
        valid=frame["valid_time"],
        origin=frame["power_fcst_init_time"],
        delay_steps=2 * arm.power_hours,
    )
    if arm.drop_lags:
        lags = np.full_like(lags, np.nan)
    frame = frame.with_columns(
        pl.DataFrame(lags, schema=list(common.LAG_COLUMNS), orient="row").cast(pl.Float32)
    )
    raw = prep.feature_matrix(frame=frame, scheme="summary")
    if arm.power_hours:
        names, _, _ = prep.feature_names(scheme="summary")
        ages = prep.lag_ages_hours(
            lead_hours=frame["lead_hours"].to_numpy().astype(np.float64) + arm.power_hours
        )
        for i, tag in enumerate(("origin", "day", "week")):
            raw[:, names.index(f"lag_{tag}_age")] = ages[:, i] / prep.LEAD_SCALE_HOURS
    keys = frame.select(*common.KEYS, "lead_hours")
    return keys, raw


def load_fits(
    *, network: str, month: str, seed: int, device: str
) -> tuple[dict, prep.Standardiser, np.ndarray, np.ndarray]:
    """Return (fits by Primary or None, standardiser, target mean, target std) of a stored fold."""
    arm_name = NETWORKS[network]
    folder = MODELS_DIR / arm_name / month / f"s{seed}"
    meta = json.loads((folder / "meta.json").read_text())
    setting = fit397.Setting(**meta["setting"])
    names, _, layout = prep.feature_names(scheme="summary")
    n_primaries = len(meta["target_mean"])
    with np.load(folder / "standardiser.npz") as stored:
        z = {k: stored[k].copy() for k in ("mean", "std", "mask")}
    fits = {}
    for key in meta["epochs"]:
        g = None if key == "None" else int(key)
        model = fit397.make_model(
            arm=arm_name,
            setting=setting,
            layout=layout,
            n_primaries=n_primaries,
            n_inputs=len(names),
        )
        file = folder / ("pooled.pt" if g is None else f"p{g:02d}.pt")
        model.load_state_dict(torch.load(file, map_location=device))
        model.to(device).eval()
        fits[g] = fit397.Fitted(
            model=model,
            setting=setting,
            arm=arm_name,
            seed=seed,
            epochs=meta["epochs"][key],
            val_nll=meta["val_nll"][key],
        )
    standardiser = prep.Standardiser(mean=z["mean"], std=z["std"], mask=z["mask"])
    return fits, standardiser, np.array(meta["target_mean"]), np.array(meta["target_std"])


def run_month(
    *, month: str, networks: list[str], arms: list[Arm], device: str, smoke: bool
) -> None:
    """Forecast one test month under every arm for every network and seed."""
    series, arrays = fd._series_and_arrays()
    clean = cleaned_power(
        arrays=arrays, series=series.filter(pl.col("substation_type") == "Primary")
    )
    primaries = prep.primaries()
    loaded = {
        (n, s): load_fits(network=n, month=month, seed=s, device=device)
        for n in networks
        for s in SEEDS
    }
    pieces: dict[tuple[str, str, int], list[pl.DataFrame]] = {}
    origins = prep.test_origins(month=month)
    for origin in origins[:2] if smoke else origins:
        run = origin - common.PUBLICATION_DELAY
        frame = pl.read_parquet(common.demand_frame_path(run=run))
        for row in primaries.iter_rows(named=True):
            prepared = fd.prepared_run(
                run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
                row=row,
                arrays=arrays[row["site"]],
                a7=False,
            )
            prepared = (
                prepared.filter(pl.col("ensemble_member") == pl.col("ensemble_member").min())
                .join(
                    pl.read_parquet(prep.SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet")
                    .filter(pl.col("time_series_id") == row["time_series_id"])
                    .drop("time_series_id"),
                    on=["power_fcst_init_time", "valid_time"],
                    how="inner",
                    validate="1:1",
                    maintain_order="left",
                )
                .sort("valid_time")
            )
            p = int(row["primary"])
            for arm in arms:
                power = arrays[row["site"]].power if arm.raw_power else clean[row["site"]]
                if row["site"] in SHORT_HISTORY_LABELS and not arm.raw_power:
                    power = power.copy()
                    power[: int(common.grid_index(times=SHORT_HISTORY_CUT))] = np.nan
                made = arm_inputs(arm=arm, prepared=prepared, run=run, row=row, power=power)
                if made is None:
                    continue
                keys, raw = made
                for network in networks:
                    for seed in SEEDS:
                        fits, standardiser, t_mean, t_std = loaded[(network, seed)]
                        fitted = fits.get(p, fits.get(None))
                        if fitted is None:
                            continue
                        q, _, _ = fit397.predict_z_quantiles(
                            fitted=fitted,
                            x=standardiser.apply(x=raw),
                            primary=p,
                            members=1,
                            device=device,
                        )
                        q = np.sort(q * t_std[p] + t_mean[p], axis=1).astype(np.float32)
                        pieces.setdefault((network, arm.name, seed), []).append(
                            keys.drop("lead_hours").hstack(
                                pl.DataFrame(q, schema=list(common.LEVEL_COLUMNS), orient="row")
                            )
                        )
    for (network, arm_name, seed), parts in pieces.items():
        target = (
            OUT_DIR
            / "preds"
            / network
            / arm_name
            / f"{month}_s{seed}{'_smoke' if smoke else ''}.parquet"
        )
        common.atomic_write(frame=pl.concat(parts), target=target)


def main() -> None:
    """Forecast the requested months."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(prep.SCREEN_MONTHS))
    parser.add_argument("--networks", nargs="+", default=["pooled"], choices=tuple(NETWORKS))
    parser.add_argument("--arms", nargs="+", default=[a.name for a in ARMS])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true", help="two origins, one month")
    args = parser.parse_args()
    arms = [a for a in ARMS if a.name in args.arms]
    for month in args.months[:1] if args.smoke else args.months:
        started = time.monotonic()
        run_month(
            month=month, networks=args.networks, arms=arms, device=args.device, smoke=args.smoke
        )
        print(f"{month}: {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
