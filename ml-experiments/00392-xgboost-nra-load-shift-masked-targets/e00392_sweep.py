r"""Lag-encroachment sweep of experiment 392's saved boosters (prediction only).

A load-shift event reaches the model through its three power lags. The sweep adds a synthetic step
of depth D to the lags of clean forecast rows and measures how far the forecast distribution
follows it. Nothing is refitted.

**Lag units.** `lag_origin`, `lag_same_time_latest_day` and `lag_same_time_latest_week` come from
`e00087_common.origin_lags` on `SeriesArrays.power`, which is the observed power divided by the
series' effective capacity (the training label is that ratio too). The lags and every forecast
quantile are therefore fractions of effective capacity, not MW. A step of `share` of capacity
is `D = sign * share` in those units, and "points of capacity" are `100 * value`.

**Encroachment level k** adds D to the lag columns in recency order: k = 0 changes no lag; k = 1
changes `lag_origin`; k = 2 also changes `lag_same_time_latest_day`; k = 3 also changes
`lag_same_time_latest_week`. A lag that is missing stays missing. Lagless forms have no lag
columns, so their forecasts are identical for every k (the control: share 0, ratio 1).

**Rows.** For each refit (2025-10-01, 2026-01-01, 2026-03-01, 2026-06-01) the daily runs that
refit serves, every third one. A row is clean when its observed target is finite, its target lies
outside the cache's padded event mask and outside the 48 hours after an event ends, none of its
three lags lies inside the padded mask, and all three lags are finite. Every arm and form is
swept on the same rows.

**Metrics** (all per arm and form, fold, Primary, lead band, k, share, sign, and whether the
5%-95% and 1%-99% bands are widened by 1.15 about the median):

- `shift_sum`: sum of (q50 at k) - (q50 at k = 0). Divide by `n` for the mean, by `n * D` for the
  share of D followed (+1 means the median moved to the event level, whichever the sign).
- `width5_sum`, `width5_base_sum`, `width1_sum`, `width1_base_sum`: summed widths of the 5%-95% and
  1%-99% bands at k and at k = 0.
- `cover_clean_sum`: rows whose actual value lies inside the 5%-95% band at k.
- `cover_event_sum`: rows whose event-level value (actual + D) lies inside that band.
- `reach_sum`: rows where q95 (positive step) or q05 (negative step) reaches actual + D at k.
- `reach_by_k_sum`: rows where it reaches actual + D at this k or at a smaller k.
- `first_k_sum`: sum over rows of the first k at which the band reaches actual + D (4 if never).

Experiment 391's R5 arm needs surprise columns recomputed from the shifted lags, so this sweep
does not cover it.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1:\\
ml-experiments/00392-xgboost-nra-load-shift-masked-targets \\
    DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 uv run --no-sync python \\
    ml-experiments/00392-xgboost-nra-load-shift-masked-targets/e00392_sweep.py --workers 2
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
import e00392_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from studies.eval_protocol.slices import MVA_ONLY_LABELS, SIGNED_MW_LABELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SWEEP_DIR: Final[Path] = fit.WORK_DIR / "sweep"
REFIT_DATES: Final[tuple[str, ...]] = ("2025-10-01", "2026-01-01", "2026-03-01", "2026-06-01")
ARM_NAMES: Final[tuple[str, ...]] = ("r0", "r1", "rc1")
FORM_NAMES: Final[tuple[str, ...]] = ("lagged", "lagless")
ORIGIN_STRIDE: Final[int] = 3
"""Every third daily run of a refit."""
LEVELS_OF_K: Final[tuple[int, ...]] = (0, 1, 2, 3)
SHARES: Final[tuple[float, ...]] = (0.02, 0.05, 0.10)
SIGNS: Final[tuple[int, ...]] = (1, -1)
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24 h", 0.0, 24.0),
    ("24-72 h", 24.0, 72.0),
    ("72-168 h", 72.0, 168.0),
    ("168-240 h", 168.0, 240.0),
)
WIDENING: Final[float] = common.SPREAD_FACTOR
"""The 1.15 spread factor, applied about the median."""
POST_EVENT_STEPS: Final[int] = fit.POST_EVENT_STEPS
Q01, Q05, Q50, Q95, Q99 = 0, 1, 5, 9, 10
METRICS: Final[tuple[str, ...]] = (
    "n",
    "shift_sum",
    "width5_sum",
    "width5_base_sum",
    "width1_sum",
    "width1_base_sum",
    "cover_clean_sum",
    "cover_event_sum",
    "reach_sum",
    "reach_by_k_sum",
    "first_k_sum",
)
GROUP_OF: Final[dict[str, str]] = {
    **dict.fromkeys(SIGNED_MW_LABELS, "signed-MW six"),
    **dict.fromkeys(MVA_ONLY_LABELS, "MVA-only eight"),
}
HEADLINE_GROUPS: Final[tuple[str, ...]] = ("signed-MW six", "MVA-only eight")
HEADLINE_SHARE: Final[float] = 0.05

Accumulator = dict[tuple[float, int, int, bool], np.ndarray]


REUSED_BOOSTER_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00402-lag-gate-learned-weight-injected-load-shifts-part-1"
    / "work"
    / "boosters"
)
"""Experiment 402's `r0_lagged` boosters, refitted on the same four folds; they reproduce
experiment 381's stored control to within 0.0011 of capacity at the two Primaries compared."""


def post_event_mask(*, padded: np.ndarray) -> np.ndarray:
    """Return the half-hours inside an event or within 48 hours after its padded end."""
    mask = padded.copy()
    for start, end in fit.runs_of(mask=padded):
        mask[start : min(end + POST_EVENT_STEPS, padded.size)] = True
    return mask


def clean_rows(*, prepared: pl.DataFrame, padded: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Return, per row of `prepared`, whether it is clean (rules in the module docstring)."""
    target = np.asarray(common.grid_index(times=prepared["valid_time"]))
    origin = np.asarray(common.grid_index(times=prepared["power_fcst_init_time"]))
    actual = common.take(values=power, index=target)
    lag_inside = base.lag_flags(mask=padded, target=target, origin=origin).any(axis=1)
    lags_finite = np.isfinite(prepared.select(common.LAG_COLUMNS).to_numpy()).all(axis=1)
    event_or_after = (
        common.take(values=post_event_mask(padded=padded).astype(np.float32), index=target) > 0.5
    )
    return np.isfinite(actual) & ~event_or_after & ~lag_inside & lags_finite


def lead_band_of(*, prepared: pl.DataFrame) -> np.ndarray:
    """Return each row's position in `LEAD_BANDS`, or -1 when it lies outside every band."""
    hours = (
        (prepared["valid_time"] - prepared["power_fcst_init_time"]).dt.total_seconds() / 3600.0
    ).to_numpy()
    band = np.full(hours.shape, -1, dtype=np.int64)
    last = len(LEAD_BANDS) - 1
    for position, (_, low, high) in enumerate(LEAD_BANDS):
        upper = hours <= high if position == last else hours < high
        band[(hours >= low) & upper] = position
    return band


def mixture_of(*, model: xgb.Booster, matrix: np.ndarray, members: int) -> np.ndarray:
    """Return the 11 mixture quantiles per valid time of the ensemble members' predictions."""
    raw = model.predict(xgb.DMatrix(matrix))
    return common.mixture_quantiles(member_quantiles=raw.reshape(-1, members, len(common.LEVELS)))


def shifted_matrix(
    *, matrix: np.ndarray, lag_positions: list[int], depth: float, level: int
) -> np.ndarray:
    """Return a copy of `matrix` with `depth` added to the `level` most recent lag columns."""
    out = matrix.copy()
    for position in lag_positions[:level]:
        out[:, position] += np.float32(depth)
    return out


def widen(*, quantiles: np.ndarray) -> np.ndarray:
    """Return `quantiles` with every spread about the median multiplied by `WIDENING`."""
    median = quantiles[:, [Q50]]
    return median + WIDENING * (quantiles - median)


def accumulate_depth(
    *,
    accumulator: Accumulator,
    quantiles_by_level: list[np.ndarray],
    actual: np.ndarray,
    band: np.ndarray,
    share: float,
    sign: int,
) -> None:
    """Add one run's per-row metrics for one step depth to `accumulator`, for every k and widening.

    Args:
        accumulator: Sums by `(share, sign, k, widened)`, each shaped (lead bands, `METRICS`).
        quantiles_by_level: The mixture quantiles (rows, 11) at each k in `LEVELS_OF_K`.
        actual: The observed value of each row, as a fraction of capacity.
        band: Each row's lead-band position, at least 0.
        share: The step depth as a share of capacity.
        sign: +1 for a rise, -1 for a fall.
    """
    event = actual + sign * share
    for widened in (False, True):
        quantiles = [widen(quantiles=q) if widened else q for q in quantiles_by_level]
        base_q = quantiles[0]
        reach = np.stack(
            [(q[:, Q95] >= event) if sign > 0 else (q[:, Q05] <= event) for q in quantiles]
        )
        reached_by = np.logical_or.accumulate(reach, axis=0)
        first_k = np.where(reached_by[-1], reached_by.argmax(axis=0), len(LEVELS_OF_K))
        for level, q in zip(LEVELS_OF_K, quantiles, strict=True):
            per_row = [
                np.ones(actual.shape),
                q[:, Q50] - base_q[:, Q50],
                q[:, Q95] - q[:, Q05],
                base_q[:, Q95] - base_q[:, Q05],
                q[:, Q99] - q[:, Q01],
                base_q[:, Q99] - base_q[:, Q01],
                ((q[:, Q05] <= actual) & (actual <= q[:, Q95])).astype(float),
                ((q[:, Q05] <= event) & (event <= q[:, Q95])).astype(float),
                reach[level].astype(float),
                reached_by[level].astype(float),
                first_k.astype(float),
            ]
            sums = accumulator.setdefault(
                (share, sign, level, widened), np.zeros((len(LEAD_BANDS), len(METRICS)))
            )
            for column, values in enumerate(per_row):
                sums[:, column] += np.bincount(band, weights=values, minlength=len(LEAD_BANDS))


def sweep_site(
    *,
    arm_form: str,
    refit: datetime,
    site: str,
    origin_frames: list[pl.DataFrame],
    row: dict,
    arrays: common.SeriesArrays,
    padded: np.ndarray,
) -> Accumulator | None:
    """Return the sweep sums of one saved booster over a refit's runs, or None if it is absent."""
    root = REUSED_BOOSTER_DIR if arm_form == "r0_lagged" else fit.BOOSTER_DIR
    path = root / arm_form / f"{site}_s{common.SEED}_{refit:%Y%m%d}.ubj"
    if not path.exists():
        return None
    model = xgb.Booster()
    model.load_model(path)
    model.set_param({"device": "cpu", "nthread": 1})
    lagged = arm_form.endswith("_lagged")
    columns = fd.input_columns(a7=False)
    if not lagged:
        columns = [c for c in columns if c not in common.LAG_COLUMNS]
    lag_positions = [columns.index(c) for c in common.LAG_COLUMNS] if lagged else []
    accumulator: Accumulator = {}
    for frame in origin_frames:
        prepared = fd.prepared_run(
            run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
            row=row,
            arrays=arrays,
            a7=False,
        )
        keep = clean_rows(prepared=prepared, padded=padded, power=arrays.power)
        keep &= lead_band_of(prepared=prepared) >= 0
        if not keep.any():
            continue
        prepared = prepared.filter(pl.Series(keep))
        members = prepared["ensemble_member"].n_unique()
        matrix = prepared.select(columns).to_numpy().astype(np.float32)
        target = np.asarray(common.grid_index(times=prepared["valid_time"]))
        actual = common.take(values=arrays.power, index=target)[::members]
        band = lead_band_of(prepared=prepared)[::members]
        base_q = mixture_of(model=model, matrix=matrix, members=members)
        for share in SHARES:
            for sign in SIGNS:
                shifted = [
                    mixture_of(
                        model=model,
                        matrix=shifted_matrix(
                            matrix=matrix,
                            lag_positions=lag_positions,
                            depth=sign * share,
                            level=level,
                        ),
                        members=members,
                    )
                    if lagged
                    else base_q
                    for level in LEVELS_OF_K[1:]
                ]
                accumulate_depth(
                    accumulator=accumulator,
                    quantiles_by_level=[base_q, *shifted],
                    actual=actual,
                    band=band,
                    share=share,
                    sign=sign,
                )
    return accumulator


def run_job(
    *, arm_form: str, refit: datetime, origins: list[datetime], sites: list[str]
) -> tuple[pl.DataFrame, list[str]]:
    """Sweep every available booster of one arm form at one refit.

    Returns:
        A tidy frame with one row per site, lead band, k, share, sign and widening, holding the
        `METRICS` sums, and the labels of the sites whose booster file is missing.
    """
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
        accumulator = sweep_site(
            arm_form=arm_form,
            refit=refit,
            site=site,
            origin_frames=frames,
            row=rows_of[site],
            arrays=arrays[site],
            padded=masks[site],
        )
        if accumulator is None:
            missing.append(f"{arm_form} {site} {refit:%Y-%m-%d}")
            continue
        for (share, sign, level, widened), sums in accumulator.items():
            piece = pl.DataFrame(sums, schema=list(METRICS), orient="row").with_columns(
                lead_band=pl.Series([name for name, _, _ in LEAD_BANDS]),
                arm_form=pl.lit(arm_form),
                site=pl.lit(site),
                refit=pl.lit(f"{refit:%Y-%m-%d}"),
                k=pl.lit(level),
                share=pl.lit(share),
                sign=pl.lit(sign),
                widened=pl.lit(widened),
            )
            pieces.append(piece)
    return (pl.concat(pieces) if pieces else pl.DataFrame()), missing


def with_ratios(*, sums: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Pool `sums` over everything not in `by` and add the per-row means and ratios."""
    pooled = sums.group_by(by).agg(pl.col(m).sum() for m in METRICS)
    return pooled.with_columns(
        shift_points=100 * pl.col("shift_sum") / pl.col("n"),
        shift_share_of_d=pl.col("shift_sum") / pl.col("n") / (pl.col("sign") * pl.col("share")),
        width5_ratio=pl.col("width5_sum") / pl.col("width5_base_sum"),
        width1_ratio=pl.col("width1_sum") / pl.col("width1_base_sum"),
        cover_clean=pl.col("cover_clean_sum") / pl.col("n"),
        cover_event=pl.col("cover_event_sum") / pl.col("n"),
        reach=pl.col("reach_sum") / pl.col("n"),
        reach_by_k=pl.col("reach_by_k_sum") / pl.col("n"),
        mean_first_k=pl.col("first_k_sum") / pl.col("n"),
    )


def print_headline(*, sweep: pl.DataFrame) -> None:
    """Print the group x arm x form x k tables for share 5%, both signs, by lead band, unwidened."""
    table = sweep.filter(
        (pl.col("share") == HEADLINE_SHARE)
        & ~pl.col("widened")
        & pl.col("group").is_in(HEADLINE_GROUPS)
    )
    by = ["group", "arm_form", "sign", "k", "share", "lead_band"]
    pooled = with_ratios(sums=table, by=by)
    order = {name: i for i, (name, _, _) in enumerate(LEAD_BANDS)}
    for metric in (
        "shift_share_of_d",
        "width5_ratio",
        "cover_clean",
        "cover_event",
        "reach",
        "mean_first_k",
    ):
        print(f"\n=== {metric} (share 5%, 5%-95% band unless stated; rows pooled over refits) ===")
        wide = (
            pooled.with_columns(band_order=pl.col("lead_band").replace_strict(order))
            .sort("band_order")
            .pivot(
                on="lead_band",
                index=["group", "arm_form", "sign", "k"],
                values=metric,
                sort_columns=False,
            )
            .sort("group", "arm_form", "sign", "k")
        )
        with pl.Config(tbl_rows=400, tbl_cols=12, float_precision=3, tbl_width_chars=140):
            print(wide)


def main() -> None:
    """Run the sweep over the requested arms, forms, refits and Primaries, and write the parquet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(ARM_NAMES), choices=ARM_NAMES)
    parser.add_argument("--forms", nargs="+", default=list(FORM_NAMES), choices=FORM_NAMES)
    parser.add_argument("--sites", nargs="+", default=None, help="default: all 16 Primaries")
    parser.add_argument("--refit-dates", nargs="+", default=list(REFIT_DATES))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=SWEEP_DIR / "sweep.parquet")
    args = parser.parse_args()
    series = common.series_table()
    sites = args.sites or series.filter(pl.col("substation_type") == "Primary")["site"].to_list()
    jobs = regimes.jobs_for(regime_names=("monthly",), origins=common.all_origins())
    jobs = {
        refit: runs[::ORIGIN_STRIDE]
        for refit, runs in jobs.items()
        if f"{refit:%Y-%m-%d}" in args.refit_dates
    }
    started = time.monotonic()
    print(
        "R5 (experiment 391's N1) needs surprise columns recomputed from the shifted lags and is "
        "NOT swept."
    )
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(run_job, arm_form=f"{arm}_{form}", refit=refit, origins=runs, sites=sites)
            for refit, runs in jobs.items()
            for arm in args.arms
            for form in args.forms
        ]
        results = []
        missing: list[str] = []
        for done, future in enumerate(futures, start=1):
            frame, gone = future.result()
            results.append(frame)
            missing.extend(gone)
            print(f"[{done}/{len(futures)}] {time.monotonic() - started:.0f} s", flush=True)
    if missing:
        print(f"{len(missing)} boosters missing, for example: {missing[:3]}")
    sweep = pl.concat([r for r in results if r.height]).with_columns(
        group=pl.col("site").replace_strict(GROUP_OF, default="short-history")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sweep.write_parquet(args.output)
    print_headline(sweep=sweep)
    print(f"done in {time.monotonic() - started:.0f} s ({args.output})")


if __name__ == "__main__":
    main()
