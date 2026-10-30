"""Experiment 414 (post hoc): experiment 411's scoring with the gaps in the stage-1 median filled.

Prediction and scoring only. The boosters are experiment 392's arm R1 lagged boosters, read and
never written. The harness is a copy of experiment 402's `e00402_counterfactual.py` `real` sweep
(itself built on experiment 401's): for each corrected labelled event of experiment 400 and each
sampled daily run whose lags fall inside the event, the run is predicted twice, with the observed
lags (`real`) and with every reading inside the event replaced by experiment 391's stage-1 median
plus the event's pre-event bias (`cf`). Where the stage-1 median has a gap, `fill_median` fills it
with the mean reading at the same half-hour of the 4 weeks before. This script
scores both forecasts against the observed power, row by row, on harm rows: rows whose target time
lies outside every event of the Primary but whose lags lie inside the event.

CRPS is the exact CRPS of the linear-between-quantiles distribution from the 11 quantiles, in
points of effective capacity (the unit of experiments 400 and 402).

```bash
DATA_PATH_INTERNAL=... systemd-run --user --scope -p MemoryMax=4G -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 \
    PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    uv run --no-sync python ml-experiments/00414-*/e00414_rescore.py --sites D5 D9 --threads 2
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
import e00414_mixture as mix  # ty: ignore[unresolved-import]
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
OUT: Final[Path] = DATA / "00414-post-hoc-gap-filled-counterfactual-lag-rescore" / "work" / "rows"
REFITS: Final[tuple[datetime, ...]] = tuple(
    datetime(y, m, 1, tzinfo=UTC) for y, m in ((2025, 10), (2026, 1), (2026, 3), (2026, 6))
)
PRE_EVENT_STEPS: Final[int] = 7 * 48
"""The stage-1 bias of each event is the mean reading minus median over the 7 days before it."""
MIN_SHARE_OF_LABEL: Final[float] = 0.25
"""Events whose measured deviation is under this share of the labelled depth are skipped."""
MAX_RUNS_PER_EVENT_REFIT: Final[int] = 6
"""Runs predicted per event and refit month, spread evenly over the runs whose lags touch it."""
MAX_LEAD_HOURS: Final[float] = 240.0
"""Harm rows are scored out to day 10, the headline band of the demand protocol."""
SCHEMA: Final[dict[str, pl.DataType | type]] = {
    "site": pl.String, "event_id": pl.Int64, "refit": pl.Datetime("us", "UTC"),
    "origin": pl.Datetime("us", "UTC"), "valid_idx": pl.Int64, "k": pl.Int64,
    "lead_hours": pl.Float64, "step": pl.Float64, "crps_real": pl.Float64, "crps_cf": pl.Float64,
}  # fmt: skip
"""Columns of a Primary's table, so that a Primary with no harm rows still writes a valid file."""
WEEK_STEPS: Final[int] = 7 * 48
"""Half-hours in a week."""
FILL_WEEKS: Final[int] = 4
"""Earlier weeks averaged to fill a gap in the stage-1 median."""
FILL_MIN_WEEKS: Final[int] = 2
"""A gap stays unfilled, and its event is skipped, when fewer weeks than this qualify."""
POINTS: Final[float] = 100.0
"""CRPS is reported in points (hundredths) of effective capacity."""


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


def stage1_median(*, site: str, capacity: float) -> np.ndarray:
    """Return the stage-1 median on experiment 87's grid as a fraction of capacity (NaN gaps)."""
    table = pl.read_parquet(STAGE1 / f"{site}.parquet", columns=["valid_time", "median_mw"])
    grid = np.full(common.grid_times().len(), np.nan)
    index = np.asarray(common.grid_index(times=table["valid_time"]))
    inside = (index >= 0) & (index < grid.size)
    grid[index[inside]] = table["median_mw"].to_numpy()[inside] / capacity
    return grid


def fill_median(*, median: np.ndarray, power: np.ndarray, in_any: np.ndarray) -> np.ndarray:
    """Return the stage-1 median with each gap filled by the same half-hour of the earlier weeks.

    The fill value for grid index `i` is the mean of `power[i - 336 * w]` for `w` in 1 to
    `FILL_WEEKS`, over the weeks whose reading is finite and outside every event. The value is
    left as NaN when fewer than `FILL_MIN_WEEKS` weeks qualify. Every reading used is older than
    index `i`, so a lag index never receives a value from after itself.
    """
    usable = np.where(in_any | ~np.isfinite(power), np.nan, power)
    stacked = np.full((FILL_WEEKS, median.size), np.nan)
    for w in range(1, FILL_WEEKS + 1):
        stacked[w - 1, WEEK_STEPS * w :] = usable[: median.size - WEEK_STEPS * w]
    count = np.isfinite(stacked).sum(axis=0)
    with np.errstate(invalid="ignore"):
        profile = np.where(count >= FILL_MIN_WEEKS, np.nanmean(stacked, axis=0), np.nan)
    return np.where(np.isnan(median), profile, median)


def crps_points(*, q: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the exact CRPS of each row's 11 quantiles, in points of effective capacity."""
    pair = mix.Pair(q1=q, q2=q, actual=actual)
    return pair.crps(w=np.ones(q.shape[0])) * POINTS


def harm_rows_site(  # noqa: PLR0912, PLR0915
    *, site: str, threads: int
) -> pl.DataFrame:
    """Return one row per scored harm row with the CRPS under real and counterfactual lags."""
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
    median = fill_median(median=median, power=arrays.power, in_any=in_any)
    out: list[pl.DataFrame] = []
    for refit, origins in origins_by_refit().items():
        model = load_model(site=site, refit=refit, threads=threads)
        for ev in events.iter_rows(named=True):
            lo, hi = ev["onset_idx"], ev["end_idx"]
            label_shift = ev["depth_mw"] / capacity
            if np.isnan(median[lo : hi + 1]).any() or np.isnan(arrays.power[lo : hi + 1]).any():
                continue
            if lo < int(common.grid_index(times=refit)) <= hi:
                continue
            before = slice(max(0, lo - PRE_EVENT_STEPS), lo)
            gap = arrays.power[before] - median[before]
            bias = float(np.nanmean(gap)) if np.isfinite(gap).any() else 0.0
            cf_power = arrays.power.copy()
            cf_power[lo : hi + 1] = median[lo : hi + 1] + bias
            shift = float(np.mean(arrays.power[lo : hi + 1] - cf_power[lo : hi + 1]))
            if shift * label_shift <= 0 or abs(shift) < MIN_SHARE_OF_LABEL * abs(label_shift):
                continue
            if abs(bias) > 0.5 * abs(shift):
                continue
            variants = {"real": arrays, "cf": dataclasses.replace(arrays, power=cf_power)}
            touching = []
            for origin in origins:
                o = int(common.grid_index(times=origin))
                target = np.arange(o + 1, o + 1 + int(MAX_LEAD_HOURS * 2))
                lag = common.origin_lags(
                    power=np.arange(size, dtype=np.float32),
                    target=target,
                    origin=np.full(target.shape, o),
                ).astype(np.int64)
                k = ((lag >= lo) & (lag <= hi)).sum(axis=1)
                if ((k > 0) & ~in_any[target]).any():
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
                actual = arrays.power[target]
                keep = (k > 0) & ~in_any[target] & (lead <= MAX_LEAD_HOURS) & np.isfinite(actual)
                if not keep.any():
                    continue
                q = {
                    n: quantiles(model=model, prepared=p, columns=columns)[keep]
                    for n, p in prepared.items()
                }
                out.append(
                    pl.DataFrame({
                        "site": site, "event_id": ev["event_id"], "refit": refit, "origin": origin,
                        "valid_idx": target[keep], "k": k[keep], "lead_hours": lead[keep],
                        "step": shift,
                        "crps_real": crps_points(q=q["real"], actual=actual[keep]),
                        "crps_cf": crps_points(q=q["cf"], actual=actual[keep]),
                    })
                )  # fmt: skip
        print(site, refit.date(), "done", flush=True)
    return pl.concat(out) if out else pl.DataFrame()


def main() -> None:
    """Score the named Primaries, skipping any already written."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for site in args.sites:
        target = OUT / f"{site}.parquet"
        if target.exists():
            continue
        frame = harm_rows_site(site=site, threads=args.threads)
        if frame.width == 0:
            frame = pl.DataFrame(schema=SCHEMA)
        frame.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
