"""Refit the XGBoost member of experiment 87 with ensemble spread columns (S1), or without (S0).

Reuses experiment 87's fitter (`e00087_fit_demand`, read only) and adds four input columns when
the environment variable `E389_SPREAD` is `1`: the mean and the standard deviation across the 51
ensemble members of `temperature_2m` and of `shortwave`, at the series' location and the row's
valid time. Training rows (the control member only) and prediction rows (all 51 members) read the
columns from the same ensemble store, so every member row of a run carries the same four numbers.
The stores hold all 51 members for the 821 runs from 2024-04-01 to 2026-06-30, and the training
rows start on 2024-04-01, so no training row is dropped for lack of a spread value.

- `s0`: the baseline configuration (`n1h`, weekly regime), columns off. It is the gate.
- `s1`: the same with the four columns on.

Output goes to `work/anchor_preds/<config>/` of this experiment, never to experiment 87's folder.

    E389_SPREAD=1 PYTHONPATH=ml-experiments/00087-*:ml-experiments/00389-* DATA_PATH_INTERNAL=... \
        uv run --no-sync python ml-experiments/00389-*/e00389_fit.py --config s1 [--sites D5 ...]
"""

import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_offset as offsets  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
OUT: Final[Path] = DATA / "00389-xgboost-tail-calibration-and-ensemble-spread-columns" / "work"
OUT_ROOT: Final[Path] = OUT / "anchor_preds"
SPREAD_COLUMNS: Final[tuple[str, ...]] = (
    "ens_temp_mean",
    "ens_temp_std",
    "ens_sw_mean",
    "ens_sw_std",
)
STORE_FOLDERS: Final[dict[str, Path]] = {
    **dict.fromkeys(
        ("D5", "D9", "D10", "D11", "D14", "D16", "D18"),
        DATA
        / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
        / "work"
        / "ens_store",
    ),
    **dict.fromkeys(
        ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"),
        DATA / "00374-mva-fold-stage-two-at-eight-primaries" / "work" / "ens_store",
    ),
    "D15": DATA / "00377-demand-v1-first-run-feasibility" / "work" / "d15" / "ens_store",
}
ENABLED: Final[bool] = os.environ.get("E389_SPREAD") == "1"
HALF_HOUR: Final[np.timedelta64] = np.timedelta64(30, "m")
_STORES: dict[str, dict[str, np.ndarray]] = {}

fd.CONFIGS["s0"] = replace(
    fd.CONFIGS["n1h"], name="s0", regime_names=("weekly",), groups=("Primary",)
)
fd.CONFIGS["s1"] = replace(
    fd.CONFIGS["n1h"], name="s1", regime_names=("weekly",), groups=("Primary",)
)
fd.CONFIGS["s0m"] = replace(
    fd.CONFIGS["n1h"], name="s0m", regime_names=("monthly",), groups=("Primary",)
)
fd.CONFIGS["s1m"] = replace(
    fd.CONFIGS["n1h"], name="s1m", regime_names=("monthly",), groups=("Primary",)
)
CONFIG_NAMES: Final[tuple[str, ...]] = ("s0", "s1", "s0m", "s1m")
SPREAD_CONFIGS: Final[tuple[str, ...]] = ("s1", "s1m")
BOOSTER_DIR: Final[Path] = OUT / "boosters"
_CONTEXT: dict[str, object] = {}


def store_of(*, site: str) -> dict[str, np.ndarray]:
    """Return a site's ensemble mean and standard deviation per run and step, loaded once."""
    if site not in _STORES:
        with np.load(STORE_FOLDERS[site] / f"{site}.npz") as stored:
            origins = stored["origins"]
            fields = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for name, key in (("temp", "temperature"), ("sw", "shortwave")):
                    values = stored[key].astype(np.float64)
                    fields[f"{name}_mean"] = np.nanmean(values, axis=1)
                    fields[f"{name}_std"] = np.nanstd(values, axis=1)
        _STORES[site] = {"origins": origins, **fields}
    return _STORES[site]


def attach_spread(*, frame: pl.DataFrame, site: str) -> pl.DataFrame:
    """Return the frame with the four spread columns, in the frame's row order.

    Args:
        frame: Rows with `power_fcst_init_time` and `valid_time`.
        site: The Primary whose ensemble store supplies the columns.

    Returns:
        The input frame plus `ens_temp_mean`, `ens_temp_std`, `ens_sw_mean`, and `ens_sw_std`.

    Raises:
        ValueError: If a row's run or step is not in the site's store.
    """
    store = store_of(site=site)
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    run = np.searchsorted(store["origins"], origin)
    run_ok = (run < len(store["origins"])) & (
        store["origins"][np.minimum(run, len(store["origins"]) - 1)] == origin
    )
    step = ((valid - origin) // HALF_HOUR).astype(np.int64) - 1
    width = store["temp_mean"].shape[1]
    ok = run_ok & (step >= 0) & (step < width)
    columns = {}
    for name in SPREAD_COLUMNS:
        key = name.removeprefix("ens_")
        table = store[key]
        values = np.full(frame.height, np.nan)
        values[ok] = table[run[ok], step[ok]]
        columns[name] = values
    return frame.with_columns(*[pl.Series(n, v) for n, v in columns.items()])


def _save_boosters() -> None:
    """Save every fitted booster as a .ubj file in work/boosters, named by config, site, refit."""
    real_train = fd.xgb.train
    real_inputs = fd.train_inputs

    def train_inputs(
        *, known: pl.DataFrame, row: dict, refit: datetime, config: fd.Config, offset: bool
    ) -> Any:
        _CONTEXT.update(site=row["site"], refit=refit, name=config.name + ("_o" if offset else ""))
        return real_inputs(known=known, row=row, refit=refit, config=config, offset=offset)

    def train(params: Any, dtrain: Any, *args: Any, **kwargs: Any) -> Any:
        booster = real_train(params, dtrain, *args, **kwargs)
        folder = BOOSTER_DIR / str(_CONTEXT["name"])
        folder.mkdir(parents=True, exist_ok=True)
        refit = _CONTEXT["refit"]
        booster.save_model(folder / f"{_CONTEXT['site']}_{refit:%Y%m%dT%H%M}.ubj")
        return booster

    fd.train_inputs = train_inputs
    fd.xgb = SimpleNamespace(train=train, DMatrix=fd.xgb.DMatrix, Booster=fd.xgb.Booster)


_save_boosters()

if ENABLED:
    _input_columns = fd.input_columns
    _training_rows = fd.training_rows
    _prepared_run = fd.prepared_run

    def input_columns(*, a7: bool) -> list[str]:
        """Return experiment 87's input columns followed by the four spread columns."""
        return [*_input_columns(a7=a7), *SPREAD_COLUMNS]

    def training_rows(
        *, row: dict, arrays: common.SeriesArrays, frozen: bool, a7: bool
    ) -> pl.DataFrame:
        """Return the control member's training rows with the spread columns, finite rows only."""
        rows = _training_rows(row=row, arrays=arrays, frozen=frozen, a7=a7)
        rows = attach_spread(frame=rows, site=row["site"])
        kept = rows.filter(pl.all_horizontal(pl.col(c).is_finite() for c in SPREAD_COLUMNS))
        print(
            f"{row['site']}: dropped {rows.height - kept.height:,} of {rows.height:,} rows "
            "without a spread value",
            flush=True,
        )
        return kept

    def prepared_run(
        *, run_rows: pl.DataFrame, row: dict, arrays: common.SeriesArrays, a7: bool
    ) -> pl.DataFrame:
        """Return one run's member rows with the spread columns, which must all be finite."""
        prepared = attach_spread(
            frame=_prepared_run(run_rows=run_rows, row=row, arrays=arrays, a7=a7), site=row["site"]
        )
        if (
            not prepared.select(pl.all_horizontal(pl.col(c).is_finite() for c in SPREAD_COLUMNS))
            .to_series()
            .all()
        ):
            msg = f"{row['site']}: a prediction row has no spread value"
            raise ValueError(msg)
        return prepared

    fd.input_columns = input_columns
    fd.training_rows = training_rows
    fd.prepared_run = prepared_run


def main() -> None:
    """Fit the requested configuration's weekly refits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=CONFIG_NAMES, required=True)
    parser.add_argument("--sites", nargs="+", help="series labels, default all 16 Primaries")
    parser.add_argument(
        "--offset", action="store_true", help="the offset twin at the 5 offset Primaries (vintage)"
    )
    parser.add_argument("--first-refits", type=int, help="only the first N weekly refits")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if (args.config in SPREAD_CONFIGS) != ENABLED:
        msg = "set E389_SPREAD=1 for s1 and leave it unset for s0"
        raise SystemExit(msg)
    config = fd.CONFIGS[args.config]
    series = common.series_table().filter(pl.col("substation_type") == "Primary")
    offset_sites = list(offsets.OFFSET_SITES)
    default = offset_sites if args.offset else series["site"].to_list()
    sites = args.sites or default
    jobs = fd.selected_jobs(
        config=config, regime=config.regime_names[0], origins=common.all_origins()
    )
    ordered = sorted(jobs.items())[: args.first_refits]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                fd.run_job,
                config_name=config.name,
                refit=refit,
                origins=runs,
                sites=sites,
                offset=args.offset,
                out_root=OUT_ROOT,
                rounds=config.rounds,
                nthread=args.threads,
                seed=common.SEED,
                workers=args.workers,
                log_hours=False,
            )
            for refit, runs in ordered
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
