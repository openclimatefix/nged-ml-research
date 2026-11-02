r"""Readings-level injection sweeps for experiment 502: encroachment and bad data per lag subset.

This file is a copy of experiment 465's `e00465_sweep.py` with the arms of experiment 502. Every
injection is made in the half-hourly readings of the Primary itself, and every lag is rebuilt from
the modified readings before the stored booster predicts (see experiment 465 for the variants:
`step`, `column_k1`, `stuck`, `dropout`, `spike`, `level`). Only readings no younger than the newest
usable one reach a lag.

Arms swept (stored boosters): `rep` (experiment 502's CUDA repeat of the three original lags),
`nowk` (no week lag), and `noday` (no day lag), and, from experiment 465's stored table, arms `a`
and `b`. Experiment 465's `sweep.parquet` already holds `a` and `b`; they are not rerun.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1:\\
ml-experiments/00392-xgboost-nra-load-shift-masked-targets:\\
ml-experiments/00502-leave-one-lag-out-power-lag-ablation-xgboost-demand \\
    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00502-*/e00502_sweep.py \\
    --arms nowk noday --sites D5 D16 --refit-dates 2026-01-01 2026-06-01
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
import e00383_fit as base  # ty: ignore[unresolved-import]
import e00392_sweep as sweep392  # ty: ignore[unresolved-import]
import e00502_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb

OUT_DIR: Final[Path] = fit.WORK_DIR / "sweep"
BOOSTERS: Final[dict[str, Path]] = {
    arm: fit.BOOSTER_DIR / arm for arm in ("a_rep", "nowk", "noday")
}
REFIT_DATES: Final[tuple[str, ...]] = ("2025-10-01", "2026-01-01", "2026-03-01", "2026-06-01")
ORIGIN_STRIDE: Final[int] = 6
STEP_LENGTHS_STEPS: Final[dict[str, int]] = {
    "6h": 12,
    "24h": 48,
    "48h": 96,
    "3d": 144,
    "5d": 240,
    "7d": 336,
}
SHARES: Final[tuple[float, ...]] = (0.05, 0.025)
SIGNS: Final[tuple[int, ...]] = (1, -1)
SPIKE_MULTIPLES: Final[tuple[float, ...]] = (4.0, 20.0)
SPIKE_STEPS: Final[int] = 4
STUCK_STEPS: Final[int] = 48
DROPOUT_STEPS: Final[int] = 12
LEVEL_SHARE: Final[float] = 0.05
WIDENINGS: Final[tuple[float, ...]] = (1.0, 1.05, 1.10, 1.15)
BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24 h", 0.0, 24.0),
    ("12-36 h", 12.0, 36.0),
    ("24-72 h", 24.0, 72.0),
    ("72-168 h", 72.0, 168.0),
    ("168-240 h", 168.0, 240.0),
    ("0-240 h", 0.0, 240.0),
)
Q05, Q50, Q95 = 1, 5, 9
METRICS: Final[tuple[str, ...]] = (
    "n",
    "shift_sum",
    "width_sum",
    "width_base_sum",
    "cover_clean_sum",
    "cover_event_sum",
)
VARIANT_KINDS: Final[tuple[str, ...]] = ("step", "column_k1", "stuck", "dropout", "spike", "level")


def band_masks(*, prepared: pl.DataFrame) -> list[np.ndarray]:
    """Return one boolean mask per band of `BANDS` over the rows of `prepared`."""
    hours = (
        (prepared["valid_time"] - prepared["power_fcst_init_time"]).dt.total_seconds() / 3600.0
    ).to_numpy()
    return [(hours >= low) & (hours <= high) for _, low, high in BANDS]


def week_spread(*, power: np.ndarray, end: int) -> float:
    """Return the median absolute change from the same half-hour a week earlier, up to `end`."""
    week = 7 * common.STEPS_PER_DAY
    recent = power[max(week, end - 8 * week) : end]
    previous = power[max(week, end - 8 * week) - week : end - week]
    difference = np.abs(recent - previous)
    return float(np.nanmedian(difference[np.isfinite(difference)]))


def modified_readings(
    *, power: np.ndarray, kind: str, origin: int, size: int, depth: float, first_origin: int
) -> np.ndarray:
    """Return a copy of `power` with the variant's modification applied up to the origin.

    Args:
        power: Readings as fractions of effective capacity on the grid.
        kind: One of `VARIANT_KINDS`.
        origin: Grid index of the run's origin.
        size: Window length in half-hours (`step`), or ignored.
        depth: The step, or the spike height, as a fraction of capacity.
        first_origin: Grid index of the refit's first run (the start of a `level` change).

    Returns:
        The modified readings.
    """
    out = power.copy()
    if kind == "step":
        out[origin - size : origin + 1] += np.float32(depth)
    elif kind == "stuck":
        end = origin - common.NEWEST_AGE
        out[end - STUCK_STEPS + 1 : end + 1] = power[end - STUCK_STEPS]
    elif kind == "dropout":
        end = origin - common.NEWEST_AGE
        out[end - DROPOUT_STEPS + 1 : end + 1] = 0.0
    elif kind == "spike":
        end = origin - common.NEWEST_AGE
        out[end - SPIKE_STEPS + 1 : end + 1] += np.float32(depth)
    elif kind == "level":
        out[first_origin : origin + 1] += np.float32(depth)
    return out


def matrix_for(
    *,
    arm: str,
    prepared: pl.DataFrame,
    power: np.ndarray,
    columns: list[str],
    origin_shift: float = 0.0,
) -> np.ndarray:
    """Return the input matrix of `prepared` with its lags rebuilt from `power`."""
    rebuilt = fd.add_lags(frame=prepared, power=power, a7=False)
    if origin_shift:
        rebuilt = rebuilt.with_columns(lag_origin=pl.col("lag_origin") + float(origin_shift))
    return rebuilt.select(columns).to_numpy().astype(np.float32)


def accumulate(
    *,
    sums: dict[tuple, np.ndarray],
    key: tuple,
    base_q: np.ndarray,
    new_q: np.ndarray,
    actual: np.ndarray,
    event: np.ndarray,
    masks: list[np.ndarray],
) -> None:
    """Add one run's per-row metrics to `sums[(*key, widening)]`, one row per band."""
    for widening in WIDENINGS:
        b_med, n_med = base_q[:, [Q50]], new_q[:, [Q50]]
        base_w = b_med + widening * (base_q - b_med)
        new_w = n_med + widening * (new_q - n_med)
        per_row = [
            np.ones(actual.shape),
            new_q[:, Q50] - base_q[:, Q50],
            new_w[:, Q95] - new_w[:, Q05],
            base_w[:, Q95] - base_w[:, Q05],
            ((new_w[:, Q05] <= actual) & (actual <= new_w[:, Q95])).astype(float),
            ((new_w[:, Q05] <= event) & (event <= new_w[:, Q95])).astype(float),
        ]
        table = sums.setdefault((*key, widening), np.zeros((len(BANDS), len(METRICS))))
        for position, mask in enumerate(masks):
            table[position] += [float(v[mask].sum()) for v in per_row]


def variants(*, power: np.ndarray, origin: int, first_origin: int) -> list[tuple]:
    """Return the variants to run: tuples of `(kind, name, sign, depth, size, share)`."""
    out: list[tuple] = []
    for share in SHARES:
        for sign in SIGNS:
            for name, steps in STEP_LENGTHS_STEPS.items():
                out.append(("step", name, sign, sign * share, steps, share))
            out.append(("column_k1", "origin", sign, sign * share, 0, share))
    out += [("stuck", "24h", 1, 0.0, 0, 0.0), ("dropout", "6h", 1, 0.0, 0, 0.0)]
    spread = week_spread(power=power, end=origin - common.NEWEST_AGE)
    out += [("spike", f"x{int(m)}", 1, m * spread, 0, m) for m in SPIKE_MULTIPLES]
    out += [("level", "5pct", sign, sign * LEVEL_SHARE, 0, LEVEL_SHARE) for sign in SIGNS]
    return out


def sweep_site(
    *,
    arm: str,
    refit: datetime,
    site: str,
    origins: list[datetime],
    frames: list[pl.DataFrame],
    row: dict,
    arrays: common.SeriesArrays,
    padded: np.ndarray,
    kinds: tuple[str, ...],
) -> list[pl.DataFrame] | None:
    """Return the sweep sums of one stored booster at one refit, or None if it is absent."""
    path = BOOSTERS[arm] / f"{site}_s{common.SEED}_{refit:%Y%m%d}.ubj"
    if not path.exists():
        return None
    model = xgb.Booster()
    model.load_model(path)
    model.set_param({"device": "cpu", "nthread": 1})
    columns = fit.input_columns(arm=arm)
    sums: dict[tuple, np.ndarray] = {}
    first_origin = int(common.grid_index(times=origins[0]))
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
        actual = common.take(values=arrays.power, index=target)[::members]
        masks = [m[::members] for m in band_masks(prepared=prepared)]
        origin = int(common.grid_index(times=origin_time))
        base_matrix = matrix_for(arm=arm, prepared=prepared, power=arrays.power, columns=columns)
        base_q = sweep392.mixture_of(model=model, matrix=base_matrix, members=members)
        for kind, name, sign, depth, size, share in variants(
            power=arrays.power, origin=origin, first_origin=first_origin
        ):
            if kind not in kinds or (False):
                continue
            if kind == "column_k1":
                matrix = matrix_for(
                    arm=arm,
                    prepared=prepared,
                    power=arrays.power,
                    columns=columns,
                    origin_shift=depth,
                )
            else:
                modified = modified_readings(
                    power=arrays.power,
                    kind=kind,
                    origin=origin,
                    size=size,
                    depth=depth,
                    first_origin=first_origin,
                )
                matrix = matrix_for(arm=arm, prepared=prepared, power=modified, columns=columns)
            new_q = sweep392.mixture_of(model=model, matrix=matrix, members=members)
            day = 0
            if kind == "level":
                day = (origin - first_origin) // common.STEPS_PER_DAY
            event_level = actual + (
                depth if kind in ("step", "column_k1", "level", "spike") else 0.0
            )
            accumulate(
                sums=sums,
                key=(kind, name, sign, share, day),
                base_q=base_q,
                new_q=new_q,
                actual=actual,
                event=event_level,
                masks=masks,
            )
    pieces = []
    for (kind, name, sign, share, day, widening), table in sums.items():
        pieces.append(
            pl.DataFrame(table, schema=list(METRICS), orient="row").with_columns(
                band=pl.Series([b for b, _, _ in BANDS]),
                arm=pl.lit(arm),
                site=pl.lit(site),
                refit=pl.lit(f"{refit:%Y-%m-%d}"),
                kind=pl.lit(kind),
                variant=pl.lit(name),
                sign=pl.lit(sign),
                share=pl.lit(float(share)),
                day=pl.lit(day),
                widening=pl.lit(widening),
            )
        )
    return pieces


def run_job(
    *, arm: str, refit: datetime, origins: list[datetime], sites: list[str], kinds: tuple[str, ...]
) -> tuple[pl.DataFrame, list[str]]:
    """Sweep every available booster of one arm at one refit."""
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
            arm=arm,
            refit=refit,
            site=site,
            origins=origins,
            frames=frames,
            row=rows_of[site],
            arrays=arrays[site],
            padded=masks[site],
            kinds=kinds,
        )
        if result is None:
            missing.append(f"{arm} {site} {refit:%Y-%m-%d}")
        else:
            pieces.extend(result)
    return (pl.concat(pieces) if pieces else pl.DataFrame()), missing


def main() -> None:
    """Run the sweeps and write the tidy sums."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(BOOSTERS), choices=list(BOOSTERS))
    parser.add_argument("--kinds", nargs="+", default=list(VARIANT_KINDS), choices=VARIANT_KINDS)
    parser.add_argument("--sites", nargs="+", default=None, help="default: all 16 Primaries")
    parser.add_argument("--refit-dates", nargs="+", default=list(REFIT_DATES))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "sweep.parquet")
    parser.add_argument("--max-origins", type=int, default=None, help="smoke test")
    args = parser.parse_args()
    series = common.series_table()
    sites = args.sites or series.filter(pl.col("substation_type") == "Primary")["site"].to_list()
    jobs = regimes.jobs_for(regime_names=("monthly",), origins=common.all_origins())
    jobs = {
        refit: runs[::ORIGIN_STRIDE][: args.max_origins]
        for refit, runs in jobs.items()
        if f"{refit:%Y-%m-%d}" in args.refit_dates
    }
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                run_job, arm=arm, refit=refit, origins=runs, sites=sites, kinds=tuple(args.kinds)
            )
            for refit, runs in jobs.items()
            for arm in args.arms
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
