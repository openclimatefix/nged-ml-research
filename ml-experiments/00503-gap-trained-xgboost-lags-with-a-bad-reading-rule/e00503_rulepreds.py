r"""Forecast the real rows of the four refit months with the serving rules applied (experiment 503).

For each stored booster and each rule set, the three real lags of every forecast run are set to
missing where the rules of `e00503_rules.py` fire, and the booster predicts the 51-member mixture.
The outputs have the layout of `e00503_fit.py`'s forecasts, so the scoring script treats them as
arms:

- `a_iii`, `a_iii2`: experiment 402's `r0_lagged` boosters (trained without gaps), rule sets
  `iii` (zero and stuck) and `iii2` (zero, far, and stuck). These are the (a-rule) arms.
- `g_iii`, `g_iii2`: this experiment's gap-trained boosters. These are arms G-A and G-B.

It also writes, once per site and refit, `arm_preds/fire/<site>_s<seed>_<refit>.parquet`: for every
forecast key, how many of the three lags each rule would set to missing, and how many lags lack a
stage-1 value. The scoring script joins it to the event row classes.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1:\\
ml-experiments/00503-gap-trained-xgboost-lags-with-a-bad-reading-rule DATA_PATH_INTERNAL=... \\
    E503_DEVICE=cuda uv run --no-sync python ml-experiments/00503-*/e00503_rulepreds.py
```
"""

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00503_fit as fit  # ty: ignore[unresolved-import]
import e00503_rules as rules  # ty: ignore[unresolved-import]
import e00503_sweep as sweep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb

ARM_DIR: Final[Path] = fit.ARM_DIR
ARM_NAMES: Final[dict[tuple[str, str], str]] = {
    ("a", "iii"): "a_iii",
    ("a", "iii2"): "a_iii2",
    ("g", "iii"): "g_iii",
    ("g", "iii2"): "g_iii2",
}


def run_job(*, booster: str, refit: datetime, origins: list[datetime], sites: list[str]) -> str:
    """Forecast one refit's runs for one booster kind, for each rule set and site."""
    series, arrays = fd._series_and_arrays()
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    columns = fit.columns()
    positions = [columns.index(c) for c in common.LAG_COLUMNS]
    count = 0
    for site in sites:
        row, power = rows_of[site], arrays[site].power
        path = sweep.BOOSTER_DIRS[booster] / f"{site}_s{common.SEED}_{refit:%Y%m%d}.ubj"
        if not path.exists():
            continue
        model = xgb.Booster()
        model.load_model(path)
        model.set_param({"device": sweep.DEVICE, "nthread": 1})
        median, mad = rules.stage1_median(
            site=site, capacity_mw=row["effective_capacity_mw"], refit=refit
        )
        pieces: dict[str, list[pl.DataFrame]] = {r: [] for b, r in sweep.ARMS if b == booster}
        pieces.pop("none", None)
        fires: list[pl.DataFrame] = []
        for origin in origins:
            frame = pl.read_parquet(
                common.demand_frame_path(run=origin - common.PUBLICATION_DELAY)
            )
            prepared = fd.prepared_run(
                run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
                row=row,
                arrays=arrays[site],
                a7=False,
            )
            members = prepared["ensemble_member"].n_unique()
            target = np.asarray(common.grid_index(times=prepared["valid_time"]))
            origin_index = np.asarray(common.grid_index(times=prepared["power_fcst_init_time"]))
            flags = sweep.case_flags(
                power=power, target=target, origin=origin_index, median=median, mad=mad
            )
            matrix = prepared.select(columns).to_numpy().astype(np.float32)
            keys = prepared.select(*common.KEYS).unique(maintain_order=True)
            for rule_set in pieces:
                ruled = rules.apply_rules(
                    matrix=matrix,
                    lag_positions=positions,
                    flags=flags,
                    rules=rules.rules_to_apply(name=rule_set),
                )
                raw = model.predict(xgb.DMatrix(ruled))
                pieces[rule_set].append(
                    fd.mixture_frame(keys=keys, raw=raw, members=members, seed=common.SEED)
                )
            if booster == "g":
                fires.append(
                    prepared.select(*common.KEYS)
                    .with_columns(
                        **{
                            f"fire_{name}": pl.Series(flags[name].sum(axis=1).astype(np.uint8))
                            for name in (*rules.RULE_NAMES, "table_gap")
                        }
                    )
                    .unique(subset=list(common.KEYS), keep="first", maintain_order=True)
                )
        for rule_set, tables in pieces.items():
            target_path = fd.job_path(
                root=ARM_DIR,
                config=ARM_NAMES[(booster, rule_set)],
                site=site,
                seed=common.SEED,
                refit=refit,
            )
            common.atomic_write(frame=pl.concat(tables), target=target_path)
            count += 1
        if fires:
            common.atomic_write(
                frame=pl.concat(fires),
                target=fd.job_path(
                    root=ARM_DIR, config="fire", site=site, seed=common.SEED, refit=refit
                ),
            )
    return f"{booster} {refit:%Y-%m-%d}: {count} forecast tables"


def main() -> None:
    """Write the rule-applied forecasts and the firing tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=None, help="default: the signed-MW six")
    parser.add_argument("--refit-dates", nargs="+", default=list(fit.REFIT_DATES))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    sites = args.sites or sorted(fit.SIGNED_MW_LABELS, key=lambda s: int(s[1:]))
    jobs = regimes.jobs_for(regime_names=("monthly",), origins=common.all_origins())
    jobs = {r: runs for r, runs in jobs.items() if f"{r:%Y-%m-%d}" in args.refit_dates}
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(run_job, booster=booster, refit=refit, origins=runs, sites=sites)
            for refit, runs in jobs.items()
            for booster in ("a", "g")
        ]
        for done, future in enumerate(futures, start=1):
            print(f"[{done}/{len(futures)}] {future.result()}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
