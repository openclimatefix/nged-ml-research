r"""Bad-data injections and genuine-step sweeps for experiment 503, with the serving rules.

Every injection is made in the half-hourly readings of the Primary itself, and every lag is rebuilt
from the modified readings before the stored booster predicts, as in experiment 465's
`e00465_sweep.py` (whose readings-level modifications this script reuses). The serving rules of
`e00503_rules.py` then set flagged lag values to missing, using the modified readings, and the
booster predicts. The shift of the median is measured against the same arm and rule set on the
unmodified readings, so it is the effect of the injection given the rule.

Boosters: `a` is experiment 402's `r0_lagged` (trained without gaps) and `g` is this experiment's
`g_gap`. Rule sets: `none`, `iii` (rules zero and stuck), and `iii2` (zero, far, and stuck). The
arms are therefore `a/none` (arm (a)), `a/iii` and `a/iii2` ((a-rule)), `g/none` (G), `g/iii`
(G-A), and `g/iii2` (G-B).

Variants: `clean` (no modification; counts how often each rule fires on real rows), `stuck` (the 48
readings before the newest usable one repeat the reading before them), `dropout` (the 12 newest
usable readings are zero), `spike` (the 4 newest usable readings rise by 5, 8, or 15 times the
stage-1 residual MAD, standing for the bands 4 to 6, 6 to 10, and 10 to 20 MAD), and `step`
(5% of capacity added to the readings in `[origin - L, origin]`, both signs, L from 6 h to 7 days).

Rows are the clean rows of experiment 392's sweep at leads up to 240 hours, every sixth daily run of
each refit, with valid time before 2026-07-01. Sums per (arm, rule set, site, refit, variant, band,
widening) are written for the shift of the median, its absolute value, the corruption of the lags,
the band width and its baseline, the coverage of the actual and of the actual plus the step, and the
rule firings.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1:\\
ml-experiments/00392-xgboost-nra-load-shift-masked-targets:\\
ml-experiments/00465-median-day-and-week-lags-for-load-shift-robustness:\\
ml-experiments/00503-gap-trained-xgboost-lags-with-a-bad-reading-rule \\
    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00503-*/e00503_sweep.py \\
    --sites D5 D9 --refit-dates 2025-10-01 2026-03-01
```
"""

import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00383_fit as base  # ty: ignore[unresolved-import]
import e00392_sweep as sweep392  # ty: ignore[unresolved-import]
import e00465_sweep as sweep465  # ty: ignore[unresolved-import]
import e00503_fit as fit  # ty: ignore[unresolved-import]
import e00503_rules as rules  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb

OUT_DIR: Final[Path] = fit.WORK_DIR / "sweep"
BOOSTER_DIRS: Final[dict[str, Path]] = {
    "a": sweep465.BOOSTERS["a"],
    "g": fit.BOOSTER_DIR / fit.ARM_NAME,
}
ARMS: Final[tuple[tuple[str, str], ...]] = (
    ("a", "none"),
    ("a", "iii"),
    ("a", "iii2"),
    ("g", "none"),
    ("g", "iii"),
    ("g", "iii2"),
)
SPIKE_MADS: Final[dict[str, float]] = {"x5": 5.0, "x8": 8.0, "x15": 15.0}
STEP_SHARE: Final[float] = 0.05
WIDENINGS: Final[tuple[float, ...]] = (1.0, 1.15)
DEVICE: Final[str] = os.environ.get("E503_DEVICE", "cpu")
"""Set `E503_DEVICE=cuda` to predict on the GPU (the same boosters, the same trees)."""
BANDS = sweep465.BANDS
Q05, Q50, Q95 = sweep465.Q05, sweep465.Q50, sweep465.Q95
METRICS: Final[tuple[str, ...]] = (
    "n",
    "shift_sum",
    "abs_shift_sum",
    "corruption_sum",
    "corrupted_rows",
    "width_sum",
    "width_base_sum",
    "cover_clean_sum",
    "cover_event_sum",
    "lag_cells",
    "fire_zero",
    "fire_far",
    "fire_stuck",
    "table_gap",
    "new_fire_zero_rows",
    "new_fire_far_rows",
    "new_fire_stuck_rows",
)


def variants(*, mad: float) -> list[tuple[str, str, int, float, int]]:
    """Return the variants as `(kind, name, sign, depth, size)`."""
    out: list[tuple[str, str, int, float, int]] = [("clean", "none", 1, 0.0, 0)]
    out += [("stuck", "24h", 1, 0.0, 0), ("dropout", "6h", 1, 0.0, 0)]
    out += [("spike", name, 1, m * mad, 0) for name, m in SPIKE_MADS.items()]
    for sign in (1, -1):
        for name, steps in sweep465.STEP_LENGTHS_STEPS.items():
            out.append(("step", name, sign, sign * STEP_SHARE, steps))
    return out


def lag_matrix(*, prepared: pl.DataFrame, power: np.ndarray, columns: list[str]) -> np.ndarray:
    """Return the input matrix of `prepared` with its three lags rebuilt from `power`."""
    rebuilt = fd.add_lags(frame=prepared, power=power, a7=False)
    return rebuilt.select(columns).to_numpy().astype(np.float32)


def predict(
    *,
    model: xgb.Booster,
    matrix: np.ndarray,
    flags: dict[str, np.ndarray],
    rule_set: str,
    positions: list[int],
    members: int,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    """Return the 11 mixture quantiles per valid time after the rule set has acted.

    Two rule sets that leave the matrix identical share one prediction, kept in `cache`.
    """
    ruled = rules.apply_rules(
        matrix=matrix,
        lag_positions=positions,
        flags=flags,
        rules=rules.rules_to_apply(name=rule_set),
    )
    key = hash(ruled.tobytes())
    if key not in cache:
        cache[key] = sweep392.mixture_of(model=model, matrix=ruled, members=members)
    return cache[key]


def case_flags(
    *,
    power: np.ndarray,
    target: np.ndarray,
    origin: np.ndarray,
    median: np.ndarray,
    mad: float,
) -> dict[str, np.ndarray]:
    """Return the per-lag rule flags of every case for the readings `power`."""
    newest = origin - common.NEWEST_AGE
    return rules.rule_flags(
        power=power,
        index=rules.lag_indices(target=target, origin=origin),
        newest=newest,
        median=median,
        mad=mad,
        runs=rules.run_counts(power=power),
    )


def accumulate(
    *,
    sums: dict[tuple, np.ndarray],
    key: tuple,
    base_q: np.ndarray,
    new_q: np.ndarray,
    actual: np.ndarray,
    event: np.ndarray,
    corruption: np.ndarray,
    new_flags: dict[str, np.ndarray],
    old_flags: dict[str, np.ndarray],
    masks: list[np.ndarray],
) -> None:
    """Add one run's per-row metrics to `sums[(*key, widening)]`, one row per band."""
    ones = np.ones(actual.shape)
    corrupted = (corruption > 1e-6).astype(float)
    fires = {r: new_flags[r].sum(axis=1).astype(float) for r in rules.RULE_NAMES}
    new_rows = {
        r: (new_flags[r] & ~old_flags[r]).any(axis=1).astype(float) for r in rules.RULE_NAMES
    }
    for widening in WIDENINGS:
        b_med, n_med = base_q[:, [Q50]], new_q[:, [Q50]]
        new_w = n_med + widening * (new_q - n_med)
        base_w = b_med + widening * (base_q - b_med)
        shift = new_q[:, Q50] - base_q[:, Q50]
        per_row = [
            ones,
            shift,
            np.abs(shift),
            corruption,
            corrupted,
            new_w[:, Q95] - new_w[:, Q05],
            base_w[:, Q95] - base_w[:, Q05],
            ((new_w[:, Q05] <= actual) & (actual <= new_w[:, Q95])).astype(float),
            ((new_w[:, Q05] <= event) & (event <= new_w[:, Q95])).astype(float),
            ones * 3,
            fires["zero"],
            fires["far"],
            fires["stuck"],
            new_flags["table_gap"].sum(axis=1).astype(float),
            new_rows["zero"] * corrupted,
            new_rows["far"] * corrupted,
            new_rows["stuck"] * corrupted,
        ]
        table = sums.setdefault((*key, widening), np.zeros((len(BANDS), len(METRICS))))
        for position, mask in enumerate(masks):
            table[position] += [float(v[mask].sum()) for v in per_row]


def sweep_site(
    *,
    booster: str,
    refit: datetime,
    site: str,
    origins: list[datetime],
    frames: list[pl.DataFrame],
    row: dict,
    arrays: common.SeriesArrays,
    padded: np.ndarray,
) -> list[pl.DataFrame] | None:
    """Return the sweep sums of one stored booster at one refit, or None if it is absent."""
    path = BOOSTER_DIRS[booster] / f"{site}_s{common.SEED}_{refit:%Y%m%d}.ubj"
    if not path.exists():
        return None
    model = xgb.Booster()
    model.load_model(path)
    model.set_param({"device": DEVICE, "nthread": 1})
    columns = fit.columns()
    positions = [columns.index(c) for c in common.LAG_COLUMNS]
    median, mad = rules.stage1_median(
        site=site, capacity_mw=row["effective_capacity_mw"], refit=refit
    )
    rule_sets = [r for b, r in ARMS if b == booster]
    sums: dict[tuple, np.ndarray] = {}
    for origin_time, frame in zip(origins, frames, strict=True):
        prepared = fd.prepared_run(
            run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
            row=row,
            arrays=arrays,
            a7=False,
        )
        keep = sweep392.clean_rows(prepared=prepared, padded=padded, power=arrays.power)
        keep &= sweep392.lead_band_of(prepared=prepared) >= 0
        if not keep.any():
            continue
        prepared = prepared.filter(pl.Series(keep))
        members = prepared["ensemble_member"].n_unique()
        target = np.asarray(common.grid_index(times=prepared["valid_time"]))
        origin_index = np.asarray(common.grid_index(times=prepared["power_fcst_init_time"]))
        actual = common.take(values=arrays.power, index=target)[::members]
        masks = [m[::members] for m in sweep465.band_masks(prepared=prepared)]
        origin = int(common.grid_index(times=origin_time))
        base_matrix = lag_matrix(prepared=prepared, power=arrays.power, columns=columns)
        base_flags = case_flags(
            power=arrays.power, target=target, origin=origin_index, median=median, mad=mad
        )
        cache: dict[int, np.ndarray] = {}
        base_q = {
            r: predict(
                model=model,
                matrix=base_matrix,
                flags=base_flags,
                rule_set=r,
                positions=positions,
                members=members,
                cache=cache,
            )
            for r in rule_sets
        }
        subsample = slice(None, None, members)
        for kind, name, sign, depth, size in variants(mad=mad):
            if kind == "clean":
                modified = arrays.power
            else:
                modified = sweep465.modified_readings(
                    power=arrays.power,
                    kind=kind,
                    origin=origin,
                    size=size,
                    depth=depth,
                    first_origin=0,
                )
            matrix = lag_matrix(prepared=prepared, power=modified, columns=columns)
            flags = case_flags(
                power=modified, target=target, origin=origin_index, median=median, mad=mad
            )
            corruption = np.nansum(
                np.abs(matrix[:, positions] - base_matrix[:, positions]), axis=1
            )[subsample]
            event_level = actual + (depth if kind in ("step", "spike") else 0.0)
            for rule_set in rule_sets:
                new_q = predict(
                    model=model,
                    matrix=matrix,
                    flags=flags,
                    rule_set=rule_set,
                    positions=positions,
                    members=members,
                    cache=cache,
                )
                accumulate(
                    sums=sums,
                    key=(rule_set, kind, name, sign),
                    base_q=base_q[rule_set],
                    new_q=new_q,
                    actual=actual,
                    event=event_level,
                    corruption=corruption,
                    new_flags={k: v[subsample] for k, v in flags.items()},
                    old_flags={k: v[subsample] for k, v in base_flags.items()},
                    masks=masks,
                )
    pieces = []
    for (rule_set, kind, name, sign, widening), table in sums.items():
        pieces.append(
            pl.DataFrame(table, schema=list(METRICS), orient="row").with_columns(
                band=pl.Series([b for b, _, _ in BANDS]),
                booster=pl.lit(booster),
                rule_set=pl.lit(rule_set),
                site=pl.lit(site),
                refit=pl.lit(f"{refit:%Y-%m-%d}"),
                kind=pl.lit(kind),
                variant=pl.lit(name),
                sign=pl.lit(sign),
                widening=pl.lit(widening),
            )
        )
    return pieces


def run_job(
    *, booster: str, refit: datetime, origins: list[datetime], sites: list[str]
) -> tuple[pl.DataFrame, list[str]]:
    """Sweep every available booster of one kind at one refit."""
    series, arrays = fd._series_and_arrays()
    masks, _ = base.worker_state()
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    frames = [
        pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        for origin in origins
    ]
    pieces: list[pl.DataFrame] = []
    missing: list[str] = []
    for site in sites:
        result = sweep_site(
            booster=booster,
            refit=refit,
            site=site,
            origins=origins,
            frames=frames,
            row=rows_of[site],
            arrays=arrays[site],
            padded=masks[site],
        )
        if result is None:
            missing.append(f"{booster} {site} {refit:%Y-%m-%d}")
        else:
            pieces.extend(result)
    return (pl.concat(pieces) if pieces else pl.DataFrame()), missing


def main() -> None:
    """Run the sweeps and write the tidy sums."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boosters", nargs="+", default=["a", "g"], choices=list(BOOSTER_DIRS))
    parser.add_argument("--sites", nargs="+", default=None, help="default: the signed-MW six")
    parser.add_argument("--refit-dates", nargs="+", default=list(fit.REFIT_DATES))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "sweep.parquet")
    parser.add_argument("--max-origins", type=int, default=None, help="smoke test")
    args = parser.parse_args()
    sites = args.sites or sorted(fit.SIGNED_MW_LABELS, key=lambda s: int(s[1:]))
    jobs = regimes.jobs_for(regime_names=("monthly",), origins=common.all_origins())
    jobs = {
        refit: runs[:: sweep465.ORIGIN_STRIDE][: args.max_origins]
        for refit, runs in jobs.items()
        if f"{refit:%Y-%m-%d}" in args.refit_dates
    }
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(run_job, booster=booster, refit=refit, origins=runs, sites=sites)
            for refit, runs in jobs.items()
            for booster in args.boosters
        ]
        results, missing = [], []
        for done, future in enumerate(futures, start=1):
            frame, gone = future.result()
            results.append(frame)
            missing.extend(gone)
            print(f"[{done}/{len(futures)}] {time.monotonic() - started:.0f} s", flush=True)
    if missing:
        print(f"{len(missing)} boosters missing, for example: {missing[:3]}")
    table = pl.concat([r for r in results if r.height])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(args.output)
    print(f"done in {time.monotonic() - started:.0f} s ({args.output})")


if __name__ == "__main__":
    main()
