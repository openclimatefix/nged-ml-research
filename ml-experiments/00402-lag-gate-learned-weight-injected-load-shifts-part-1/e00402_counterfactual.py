"""Experiment 402, step 2: real-event counterfactual and synthetic sweep on the R1 lagged boosters.

Prediction only, on experiment 392's arm R1 lagged boosters (`00392-.../work/boosters/r1_lagged`),
which are read and never written. Two sweeps:

- `real`: for each corrected labelled event (experiment 400's `events_corrected.parquet`, per
  Primary) and each daily run of the four refit months whose lags fall inside the event, the run is
  predicted three times: with the observed lags (`real`), with every reading inside the event
  replaced by experiment 391's stage-1 median (`cf`, the counterfactual with the event removed), and
  with `cf` plus a constant step of the event's depth inside the event window (`syn`). The share
  of the step that the median passes on is `(q50 - q50_cf) / step`, by the number `k` of lags
  inside the event, for target times outside every event.
- `clean`: experiment 395 Part E's harness on the same boosters: one clean run per refit month (the
  15th), a step of 2%, 5%, and 10% of capacity of both signs added to the first `k` of the three lags.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
DATA_PATH_INTERNAL=... systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00402-*/e00402_counterfactual.py \
    run --mode real --sites D5 D9 --threads 3
```
"""

import argparse
import dataclasses
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
BOOSTERS: Final[Path] = (
    DATA / "00392-xgboost-nra-load-shift-masked-targets" / "work" / "boosters" / "r1_lagged"
)
STAGE1: Final[Path] = DATA / "00391-xgboost-lag-surprise-second-stage" / "work" / "stage1" / "ens"
EVENTS: Final[Path] = (
    DATA
    / "00400-lagged-lagless-forecast-blend-around-load-shifts-stored"
    / "work"
    / "events_corrected.parquet"
)
WORK: Final[Path] = DATA / "00402-lag-gate-learned-weight-injected-load-shifts-part-1" / "work"
OUT: Final[Path] = WORK / "counterfactual"
REFITS: Final[tuple[datetime, ...]] = tuple(
    datetime(y, m, 1, tzinfo=UTC) for y, m in ((2025, 10), (2026, 1), (2026, 3), (2026, 6))
)
MAX_RUNS_PER_EVENT_REFIT: Final[int] = 6
"""Runs predicted per event and refit month, spread evenly over the runs whose lags touch it."""
DEPTHS: Final[tuple[float, ...]] = (0.02, 0.05, 0.10)
BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24h", 0, 24), ("24-72h", 24, 72), ("72-168h", 72, 168), ("168-240h", 168, 240),
)  # fmt: skip
Q50: Final[int] = common.LEVEL_COLUMNS.index("q50")
Q05: Final[int] = common.LEVEL_COLUMNS.index("q05")
Q95: Final[int] = common.LEVEL_COLUMNS.index("q95")


def quantiles(*, model: xgb.Booster, prepared: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """Return the mixture's 11 quantiles per valid time."""
    members = prepared["ensemble_member"].n_unique()
    raw = model.predict(xgb.DMatrix(prepared.select(columns).to_numpy().astype(np.float32)))
    return common.mixture_quantiles(member_quantiles=raw.reshape(-1, members, len(common.LEVELS)))


def load_model(*, site: str, refit: datetime, threads: int) -> xgb.Booster:
    """Return experiment 392's R1 lagged booster for one Primary and refit."""
    model = xgb.Booster()
    model.load_model(BOOSTERS / f"{site}_s0_{refit:%Y%m%d}.ubj")
    model.set_param({"nthread": threads, "device": "cpu"})
    return model


def origins_by_refit() -> dict[datetime, list[datetime]]:
    """Return the daily forecast origins that the monthly regime assigns to each scored refit."""
    jobs = regimes.jobs_for(regime_names=("monthly",), origins=common.all_origins())
    return {r: jobs[r] for r in REFITS}


def run_prepared(*, site_row: dict, arrays: common.SeriesArrays, origin: datetime) -> pl.DataFrame:
    """Return the run's prepared member rows for the (possibly altered) series arrays."""
    frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
    return fd.prepared_run(
        run_rows=frame.filter(pl.col("time_series_id") == site_row["time_series_id"]),
        row=site_row,
        arrays=arrays,
        a7=False,
    )


def band_of(*, lead: np.ndarray) -> np.ndarray:
    """Return the index of each lead's band in `BANDS`, or -1."""
    out = np.full(lead.shape, -1)
    for i, (_, lo, hi) in enumerate(BANDS):
        out[(lead >= lo) & (lead < hi)] = i
    return out


def stage1_median(*, site: str, capacity: float) -> np.ndarray:
    """Return the stage-1 median on experiment 87's grid as a fraction of capacity (NaN gaps)."""
    table = pl.read_parquet(STAGE1 / f"{site}.parquet", columns=["valid_time", "median_mw"])
    grid = np.full(common.grid_times().len(), np.nan)
    index = np.asarray(common.grid_index(times=table["valid_time"]))
    inside = (index >= 0) & (index < grid.size)
    grid[index[inside]] = table["median_mw"].to_numpy()[inside] / capacity
    return grid


def real_site(*, site: str, threads: int) -> pl.DataFrame:
    """Return the counterfactual sweep's per-run, per-k aggregates for one Primary."""
    series, arrays_of = fd._series_and_arrays()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    arrays = arrays_of[site]
    columns = fd.input_columns(a7=False)
    events = pl.read_parquet(EVENTS).filter(pl.col("label") == site).sort("onset_idx")
    if events.is_empty():
        return pl.DataFrame()
    capacity = float(events["capacity_mw"][0])
    median = stage1_median(site=site, capacity=capacity)
    size = arrays.power.size
    in_any = np.zeros(size, dtype=bool)
    for r in events.iter_rows(named=True):
        in_any[r["onset_idx"] : r["end_idx"] + 1] = True
    out = []
    for refit, origins in origins_by_refit().items():
        model = load_model(site=site, refit=refit, threads=threads)
        for ev in events.iter_rows(named=True):
            lo, hi = ev["onset_idx"], ev["end_idx"]
            shift = ev["depth_mw"] / capacity
            if np.isnan(median[lo : hi + 1]).any() or np.isnan(arrays.power[lo : hi + 1]).any():
                continue
            if lo < int(common.grid_index(times=refit)) <= hi:
                continue
            cf_power = arrays.power.copy()
            cf_power[lo : hi + 1] = median[lo : hi + 1]
            syn_power = cf_power.copy()
            syn_power[lo : hi + 1] += shift
            variants = {
                "real": arrays,
                "cf": dataclasses.replace(arrays, power=cf_power),
                "syn": dataclasses.replace(arrays, power=syn_power),
            }
            touching = []
            for origin in origins:
                o = int(common.grid_index(times=origin))
                target = np.arange(o + 1, o + 1 + 14 * 48)
                lag = common.origin_lags(
                    power=np.arange(size, dtype=np.float32),
                    target=target,
                    origin=np.full(target.shape, o),
                ).astype(np.int64)
                k = ((lag >= lo) & (lag <= hi)).sum(axis=1)
                if (k > 0).any():
                    touching.append(origin)
            if not touching:
                continue
            step = max(1, len(touching) // MAX_RUNS_PER_EVENT_REFIT)
            for origin in touching[::step][:MAX_RUNS_PER_EVENT_REFIT]:
                prepared = {
                    n: run_prepared(site_row=row, arrays=a, origin=origin)
                    for n, a in variants.items()
                }
                keys = prepared["real"].select(*common.KEYS).unique(maintain_order=True)
                target = np.asarray(common.grid_index(times=keys["valid_time"]))
                o = np.asarray(common.grid_index(times=keys["power_fcst_init_time"]))
                lag = common.origin_lags(
                    power=np.arange(size, dtype=np.float32), target=target, origin=o
                ).astype(np.int64)
                k = ((lag >= lo) & (lag <= hi)).sum(axis=1)
                lead = (
                    keys["valid_time"] - keys["power_fcst_init_time"]
                ).dt.total_minutes().to_numpy() / 60
                band = band_of(lead=lead)
                q = {
                    n: quantiles(model=model, prepared=p, columns=columns)
                    for n, p in prepared.items()
                }
                keep = (k > 0) & ~in_any[target] & (band >= 0)
                for kk in (1, 2, 3):
                    for b in range(len(BANDS)):
                        sel = keep & (k == kk) & (band == b)
                        if not sel.any():
                            continue
                        w = {n: (v[sel, Q95] - v[sel, Q05]).mean() for n, v in q.items()}
                        out.append({
                            "site": site, "event_id": ev["event_id"], "refit": refit,
                            "origin": origin, "k": kk, "band": BANDS[b][0], "n": int(sel.sum()),
                            "step": shift,
                            "share_real": float(((q["real"][sel, Q50] - q["cf"][sel, Q50]) / shift).mean()),
                            "share_syn": float(((q["syn"][sel, Q50] - q["cf"][sel, Q50]) / shift).mean()),
                            "w90_cf": float(w["cf"]), "w90_real": float(w["real"]), "w90_syn": float(w["syn"]),
                        })  # fmt: skip
        print(site, refit.date(), "done", flush=True)
    return pl.DataFrame(out)


def clean_site(*, site: str, threads: int) -> pl.DataFrame:
    """Return experiment 395 Part E's synthetic sweep for one Primary on the R1 lagged boosters."""
    series, arrays_of = fd._series_and_arrays()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    arrays = arrays_of[site]
    columns = fd.input_columns(a7=False)
    lag_idx = [columns.index(c) for c in common.LAG_COLUMNS]
    out = []
    for refit in REFITS:
        origin = datetime(refit.year, refit.month, 15, 0, tzinfo=UTC)
        origin = min(origins_by_refit()[refit], key=lambda x: abs((x - origin).total_seconds()))
        model = load_model(site=site, refit=refit, threads=threads)
        prepared = run_prepared(site_row=row, arrays=arrays, origin=origin)
        keys = prepared.select(*common.KEYS).unique(maintain_order=True)
        lead = (
            keys["valid_time"] - keys["power_fcst_init_time"]
        ).dt.total_minutes().to_numpy() / 60
        band = band_of(lead=lead)
        members = prepared["ensemble_member"].n_unique()
        matrix = prepared.select(columns).to_numpy().astype(np.float32)
        base = common.mixture_quantiles(
            member_quantiles=model.predict(xgb.DMatrix(matrix)).reshape(
                -1, members, len(common.LEVELS)
            )
        )
        for sign in (1, -1):
            for depth in DEPTHS:
                for k in (1, 2, 3):
                    shifted = matrix.copy()
                    for i in range(k):
                        shifted[:, lag_idx[i]] += sign * depth
                    quant = common.mixture_quantiles(
                        member_quantiles=model.predict(xgb.DMatrix(shifted)).reshape(
                            -1, members, len(common.LEVELS)
                        )
                    )
                    for b in range(len(BANDS)):
                        sel = band == b
                        out.append({
                            "site": site, "refit": refit, "sign": sign, "depth": depth, "k": k,
                            "band": BANDS[b][0], "n": int(sel.sum()),
                            "share": float(((quant[sel, Q50] - base[sel, Q50]) / (sign * depth)).mean()),
                            "w90_ratio": float(((quant[sel, Q95] - quant[sel, Q05]).mean() / (base[sel, Q95] - base[sel, Q05]).mean())),
                        })  # fmt: skip
        print(site, refit.date(), "clean done", flush=True)
    return pl.DataFrame(out)


def main() -> None:
    """Run one sweep for the named Primaries, skipping any already written."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("real", "clean"), required=True)
    parser.add_argument("--sites", nargs="+", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    out = OUT / args.mode
    out.mkdir(parents=True, exist_ok=True)
    for site in args.sites:
        target = out / f"{site}.parquet"
        if target.exists():
            continue
        frame = (real_site if args.mode == "real" else clean_site)(site=site, threads=args.threads)
        frame.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
