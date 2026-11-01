"""Stage 0: the pure stage-1 GAM forecast of 14 Primaries for every ensemble run.

Reads the stored monthly per-site T-all fits of experiment 136 (six signed-MW Primaries) and of
experiment 252, arm A1 (eight MVA-only Primaries), and drives them with the calibrated ensemble
mean to 240 hours. No Kalman filter, no lagged-error input, no spread. Each run uses the latest
fit whose cut is at or before the run's `power_fcst_init_time`. Writes `gam_<site>.parquet` with
the key `(time_series_id, power_fcst_init_time, valid_time)` and the column `gam_points`, and a
`gam_cuts.json` that records each fold month's fit cut for the leakage check.

Run once per source (the experiment 252 modules read these environment variables at import):

```bash
export PYTHONPATH=ml-experiments/00252-gam-stack-at-eight-more-primaries
E00252_SOURCE=control uv run --no-sync python e00457_stage0.py --source control
E00252_SOURCE=mva E00252_ARM=a1 uv run --no-sync python e00457_stage0.py --source mva
```
"""

import argparse
import json
import pickle
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00252_common as common  # ty: ignore[unresolved-import]
import e00252_design as design  # ty: ignore[unresolved-import]
import e00252_ens as ens  # ty: ignore[unresolved-import]
import e00252_run as run252  # ty: ignore[unresolved-import]
import e00252_tall as tall  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_ens import choose_fit
from studies.dp_ens_weather import RunWindow
from studies.dp_gam import QUANTILE_COLUMNS  # noqa: F401
from studies.sources import ML_EXPERIMENTS_DATA_DIR

for _name, _module in (("tall", tall), ("design", design), ("common", common)):
    sys.modules[f"e00136_{_name}"] = _module  # the experiment 136 pickles name its own modules

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / (
    "00457-gam-stage-1-forecast-as-an-input-to-the-pooled-mixture-density-network"
)
FIRST_ORIGIN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
"""Every stored ensemble run from the first one is forecast, because the fine-tuning rows of a fold
reach back to 2024-04 and each needs its own out-of-fold GAM value."""
HORIZON_HALF_HOURS: Final[int] = 480
"""240 hours."""
SOURCES: Final[dict[str, tuple[Path, tuple[str, ...]]]] = {
    "control": (
        ML_EXPERIMENTS_DATA_DIR
        / "00136-dp-bg-stage1-assembly-on-the-ensemble-mean/work/tall_fits/monthly",
        ("D5", "D9", "D10", "D11", "D14", "D16"),
    ),
    "mva": (
        ML_EXPERIMENTS_DATA_DIR
        / "00252-gam-stack-at-eight-more-primaries/work/mva/arms/a1/tall_fits/monthly",
        ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"),
    ),
}


def forecast_site(*, site: str, fit_dir: Path) -> tuple[pl.DataFrame, list[dict]]:
    """Return the GAM mean forecast of every fold-month run at a site, and the fits it used."""
    fits = pickle.loads((fit_dir / f"{site}.pkl").read_bytes())
    context = run252.load_site(site=site)
    cache, setup = context.cache, context.setup
    features = cache.sites[site]
    frames, used = [], {}
    for index, origin in enumerate(setup.origins):
        if origin < FIRST_ORIGIN:
            continue
        run = setup.runs[index]
        if fits[0].cut > origin:
            continue  # no fit yet (D8 before 2025-07): the run has no GAM value
        fit = choose_fit(fits=fits, origin=origin)
        month_start = datetime(origin.year, origin.month, 1, tzinfo=UTC)
        if fit.cut > month_start:
            msg = f"{site}: fit cut {fit.cut} is after the scored month {month_start}"
            raise ValueError(msg)
        used[f"{origin.year}-{origin.month:02d}"] = fit.cut.isoformat()
        horizon = min(HORIZON_HALF_HOURS, len(cache.end_times) - run.origin_position - 1)
        if horizon < ens.MIN_WINDOW_HALF_HOURS:
            continue
        if index < common.PROXY_WARM_UP_RUNS:
            continue  # the analysis proxy lacks a week of past temperature before these runs
        window = RunWindow.build(
            cache=cache, context=setup.context, run=run, horizon_half_hours=horizon
        )
        driven = window.driven_caches(
            context=setup.context, run=ens.mean_run(run=run), members=[0]
        )[0]
        n = window.window.stop - window.window.start
        mean = ens.predict_mean(driven=driven, fit=fit, index=np.arange(n))
        rows = np.arange(window.window.start, window.window.stop)
        keep = features.present[rows]
        origin64 = np.datetime64(origin.replace(tzinfo=None), "us")
        frames.append(
            pl.DataFrame(
                {
                    "time_series_id": np.full(int(keep.sum()), features.table.time_series_id),
                    "power_fcst_init_time": pl.Series(
                        np.full(int(keep.sum()), origin64)
                    ).dt.replace_time_zone("UTC"),
                    "valid_time": pl.Series(cache.end_times[rows[keep]]).dt.replace_time_zone(
                        "UTC"
                    ),
                    "gam_points": mean[keep].astype(np.float32),
                }
            )
        )
    cuts = [{"site": site, "month": m, "cut": c} for m, c in sorted(used.items())]
    return pl.concat(frames), cuts


def main() -> None:
    """Forecast every site of one source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=list(SOURCES), required=True)
    parser.add_argument("--sites", nargs="+")
    args = parser.parse_args()
    fit_dir, sites = SOURCES[args.source]
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    for site in args.sites or sites:
        started = time.monotonic()
        frame, cuts = forecast_site(site=site, fit_dir=fit_dir)
        frame.write_parquet(WORK_DIR / f"gam_{site}.parquet")
        (WORK_DIR / f"gam_cuts_{site}.json").write_text(json.dumps(cuts, indent=1))
        print(f"{site}: {frame.height:,} rows, {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
