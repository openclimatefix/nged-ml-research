# ruff: noqa: E501
"""Experiment 706, stage 1: the look-back offset arms on the stored normal-running forecasts.

Run it from the repository root with the environment of `CLAUDE.md` set:

    uv run --no-sync python ml-experiments/00706-*/e00706_arms.py --n-jobs 4

Each arm adds one correction to all 11 stored normal-running quantiles of a Primary at every lead,
and writes the result with `studies.experiment_store.write_predictions` (nothing is overwritten; an
existing set is skipped). The correction at a row is a factor times `a + b . x`, where `a` and `b`
come from a ridge regression of the residuals of the look-back window and `x` is the weather of the
row's own ensemble run. The window ends 6 hours before the origin (3 hours for the prevailing arms
of C5, whose stored rows cap their window at 03:00 UTC), and a window with under half its half-hours
observed gives an offset of zero. Nothing newer than the origin's 06:00 UTC power or the run of
00:00 UTC on the origin's day is read.

Arms and set names (all in `data/ml-experiments/00706-*/predictions/`):

- `g`: G-lb, the primary global offset: 0.75 times the 14-day mean of the lead-band-matched residual.
- `g-sl`: G with the smallest-lead residual. `g-refit`: the smallest-lead residual from forecasts of
  the origin's own month only. `g-w7`, `g-w28`: windows of 7 and 28 days (28 from 2026-01).
- `d`: G-lb plus ridge slopes on the irradiance driver and the cold driver. `d-pv`, `d-cold`: one
  slope each. `d-plus`: also a weekend offset and three time-of-day blocks. `d-w7`, `d-w28`.
- `p`: P-cp, the prevailing arm (factor 1.0, level of the last segment of the best single change
  point in a 28-day window of daily mean residuals), window ending 03:00 UTC. `p-7d`: factor 1.0
  and a 7-day window of arm D's form. `p-cp06`, `p-7d06`: the same with the window ending at 06:00.

Row sets: `g`, `d`, and their variants cover origins 2025-12-01 to 2026-06-30; the prevailing arms and
the 28-day arms cover 2026-01-01 to 2026-06-30. `--month YYYY-MM` restricts the origins to one month,
adds `-t<YYYYMM>` to the set names, and so measures memory first. `--dry-run` computes and prints but
writes nothing.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

import e00706_common as c  # ty: ignore[unresolved-import]
import numpy as np
from studies.experiment_store import write_predictions
from studies.sources import ML_EXPERIMENTS_DATA_DIR

BOTH: tuple[int, ...] = (c.PV_DRIVER, c.COLD_DRIVER)
PREVAILING_FACTOR: float = 1.0

LOOK_BACK_ARMS: tuple[c.ArmSpec, ...] = (
    c.ArmSpec(name="g"),
    c.ArmSpec(name="g-sl", band_matched=False),
    c.ArmSpec(name="g-refit", band_matched=False, truncate_to_month=True),
    c.ArmSpec(name="g-w7", window_days=7),
    c.ArmSpec(
        name="g-w28",
        window_days=28,
        first_origin=c.FIRST_ORIGIN_JAN,
        row_set_id=c.ROW_SET_ID_FROM_JANUARY,
    ),
    c.ArmSpec(name="d", drivers=BOTH),
    c.ArmSpec(name="d-pv", drivers=(c.PV_DRIVER,)),
    c.ArmSpec(name="d-cold", drivers=(c.COLD_DRIVER,)),
    c.ArmSpec(name="d-plus", drivers=BOTH, calendar=True),
    c.ArmSpec(name="d-w7", drivers=BOTH, window_days=7),
    c.ArmSpec(
        name="d-w28",
        drivers=BOTH,
        window_days=28,
        first_origin=c.FIRST_ORIGIN_JAN,
        row_set_id=c.ROW_SET_ID_FROM_JANUARY,
    ),
)
PREVAILING_ARMS: tuple[c.ArmSpec, ...] = (
    c.ArmSpec(
        name="p-7d",
        drivers=BOTH,
        window_days=7,
        lag_half_hours=c.LAG_3H_CAP,
        factor=PREVAILING_FACTOR,
        first_origin=c.FIRST_ORIGIN_JAN,
        row_set_id=c.ROW_SET_ID_FROM_JANUARY,
    ),
    c.ArmSpec(
        name="p-7d06",
        drivers=BOTH,
        window_days=7,
        lag_half_hours=c.LAG_6H,
        factor=PREVAILING_FACTOR,
        first_origin=c.FIRST_ORIGIN_JAN,
        row_set_id=c.ROW_SET_ID_FROM_JANUARY,
    ),
)
CHANGEPOINT_ARMS: tuple[tuple[str, int], ...] = (("p", c.LAG_3H_CAP), ("p-cp06", c.LAG_6H))
"""P-cp arms: set name and the half-hours between the origin and the newest residual."""


def look_back_correction(
    *, inputs: c.Inputs, residuals: c.Residuals, spec: c.ArmSpec, n_jobs: int
) -> tuple[np.ndarray, list[c.WindowFit]]:
    """Return the correction of every series, (K, N_ORIGINS, N_H) float32, and each series' fit."""

    def one(series: int) -> tuple[np.ndarray, c.WindowFit]:
        return c.series_corrections(inputs=inputs, residuals=residuals, series=series, spec=spec)

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        results = list(pool.map(one, range(len(inputs.ids))))
    return np.stack([result[0] for result in results]).astype(np.float32), [r[1] for r in results]


def changepoint_correction(
    *, inputs: c.Inputs, residuals: c.Residuals, lag_half_hours: int
) -> np.ndarray:
    """Return the P-cp correction, (K, N_ORIGINS, N_H) float32: one level per origin, all rows."""
    levels = np.stack(
        [
            c.changepoint_levels(residual=residuals.smallest[series], lag_half_hours=lag_half_hours)
            for series in range(len(inputs.ids))
        ]
    )
    return np.broadcast_to(levels[:, :, None], (len(inputs.ids), c.N_ORIGINS, c.N_H)).astype(
        np.float32
    )


def print_fit_summary(*, name: str, fits: list[c.WindowFit], origins: np.ndarray) -> None:
    """Print how many windows were covered and the mean coefficients over the scored origins."""
    coef = np.stack([fit.coef[origins] for fit in fits])
    covered = np.stack([fit.covered[origins] for fit in fits])
    print(f"  {name}: windows covered {covered.mean():.3f} of (series, origin, band)")
    print(
        f"  {name}: mean intercept {coef[..., 0].mean():+.3f} points, SD {coef[..., 0].std():.3f}"
    )
    for index in range(1, coef.shape[-1]):
        slope = coef[..., index]
        on = covered & (slope != 0.0)
        share = float(on.sum() / max(covered.sum(), 1))
        mean = float(slope[on].mean()) if on.any() else float("nan")
        print(
            f"  {name}: slope {index} non-zero in {share:.3f} of covered windows, mean {mean:+.3f}"
        )


def write_arm(
    *,
    inputs: c.Inputs,
    name: str,
    correction: np.ndarray,
    spec_origin: int,
    row_set_id: str,
    month: str | None,
    dry_run: bool,
) -> None:
    """Add `correction` to the stored normal-running quantiles and write the set."""
    quantiles = c.apply_correction(inputs=inputs, quantiles=inputs.q_nra, correction=correction)
    origins = np.arange(spec_origin, c.N_ORIGINS)
    set_name = name
    if month is not None:
        origins = np.intersect1d(origins, c.origin_of_month(month=month))
        set_name = f"{name}-t{month.replace('-', '')}"
        row_set_id = f"{row_set_id}_{month}"
    frame = c.dense_to_frame(inputs=inputs, quantiles=quantiles, origins=origins)
    shift = np.abs(correction[:, origins, :]).mean()
    print(f"  {set_name}: {frame.height:,} rows, mean |correction| {shift:.3f} points")
    if dry_run:
        return
    destination = ML_EXPERIMENTS_DATA_DIR / f"00706-{c.SLUG}" / "predictions" / set_name
    if destination.exists():
        print(f"  {set_name}: exists, skipped")
        return
    path = write_predictions(
        experiment_id=c.EXPERIMENT_ID,
        name=set_name,
        frame=frame,
        kind="quantiles",
        row_set_id=row_set_id,
    )
    print(f"  {set_name}: wrote {path}")


def main() -> None:
    """Compute the arms and write one prediction set per arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=4, help="threads: one series per thread")
    parser.add_argument("--month", default=None, help="restrict origins to this month, YYYY-MM")
    parser.add_argument("--only", default=None, help="comma-separated set names to compute")
    parser.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    arguments = parser.parse_args()
    wanted = None if arguments.only is None else set(arguments.only.split(","))
    inputs = c.load_inputs(load_prevailing=False, n_jobs=arguments.n_jobs)
    residuals = c.build_residuals(inputs=inputs)
    print(f"LOADED peak memory {c.peak_memory_gb():.2f} GB")
    for spec in (*LOOK_BACK_ARMS, *PREVAILING_ARMS):
        if wanted is not None and spec.name not in wanted:
            continue
        print(f"ARM {spec.name}")
        correction, fits = look_back_correction(
            inputs=inputs, residuals=residuals, spec=spec, n_jobs=arguments.n_jobs
        )
        print_fit_summary(
            name=spec.name, fits=fits, origins=np.arange(spec.first_origin, c.N_ORIGINS)
        )
        write_arm(
            inputs=inputs,
            name=spec.name,
            correction=correction,
            spec_origin=spec.first_origin,
            row_set_id=spec.row_set_id,
            month=arguments.month,
            dry_run=arguments.dry_run,
        )
    for name, lag in CHANGEPOINT_ARMS:
        if wanted is not None and name not in wanted:
            continue
        print(f"ARM {name} (change point, window ends {lag / 2:.0f} h before the origin)")
        write_arm(
            inputs=inputs,
            name=name,
            correction=PREVAILING_FACTOR
            * changepoint_correction(inputs=inputs, residuals=residuals, lag_half_hours=lag),
            spec_origin=c.FIRST_ORIGIN_JAN,
            row_set_id=c.ROW_SET_ID_FROM_JANUARY,
            month=arguments.month,
            dry_run=arguments.dry_run,
        )
    print(f"PEAK MEMORY {c.peak_memory_gb():.2f} GB")


if __name__ == "__main__":
    main()
