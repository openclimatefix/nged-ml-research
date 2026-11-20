"""Raw-matrix builders of the Primary demand arms of experiment 711: fold rows with cloud columns.

**The rows are experiment 707's rows of arm `R_F`, with the arm's extra inputs appended.** The
fine-tuning rows follow `e00707_folds.build_fold` (window `W24`), the `S_ens` rows follow
`e00707_folds.ens_rows` (window `F`), and the scored rows follow `e00707_prep.run_inputs`. Each
function takes the arm and adds, for every kept row, the columns of `e00711_rowcloud`: the summary
(real, shuffled-run placebo, or time-shifted placebo), the T3 daylight mean, the positive control's
four temperature columns (which leave the base inputs), and the `cloud_missing` flag.

**A masked row holds NaN before standardisation.** `e00707_prep.Standardiser.fit` is NaN-aware, so
the fold's standardiser uses the training rows with `cloud_missing = 0` only, and `apply` sets a
NaN input to 0. The rule reads the run date, never a cloud value.

**The origin rule is asserted on every row of every arm.** `e00711_origin.assert_primary_rows`'s
origin test runs on each row set's run and origin, real and placebo alike, and the placebo's
missing flag is asserted equal to the real flag in `e00711_rowcloud.row_extras`.

This module imports experiment 707's `e00707_prep`, which imports `torch`; the tests that import it
skip where `torch` is absent. Run every script of this folder with the folders of experiments 707
on `PYTHONPATH`.
"""

from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Final

import e00707_arms as arms707  # ty: ignore[unresolved-import]
import e00707_common as base  # ty: ignore[unresolved-import]
import e00707_folds as folds707  # ty: ignore[unresolved-import]
import e00707_prep as prep  # ty: ignore[unresolved-import]
import e00711_arms as arms
import e00711_origin as origin_rules
import e00711_placebo as placebo
import e00711_rowcloud as rowcloud
import numpy as np
import polars as pl
from e00707_models import Layout  # ty: ignore[unresolved-import]

DELAY_US: Final[int] = origin_rules.DELAY_US
TARGET_FIRST_DAY: Final[int] = placebo.day_number(run=placebo.TARGET_FIRST_RUN)
"""The day number of the first run in a donor map."""
Scored = tuple[dict, pl.DataFrame, np.ndarray, np.ndarray]
"""A series' row, its keys (`time_series_id`, origin, valid time, `lead_hours`), the raw base
matrix,
and each row's run in microseconds since the epoch."""


def _us(*, refit: datetime) -> int:
    """Return a UTC datetime as microseconds since the epoch."""
    return int(refit.timestamp() * 1_000_000)


def series_key(*, site: str) -> int:
    """Return the integer that seeds a Primary's placebo draw (the number in its label)."""
    return int(site.removeprefix("D"))


def input_plan(*, arm: arms.Arm) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return an arm's input names, standardise-mask, and the kept base-column indices."""
    base_names, base_mask, _ = prep.feature_names()
    return rowcloud.extend_columns(base_names=base_names, base_mask=base_mask, arm=arm)


def layout_for(*, names: list[str]) -> Layout:
    """Return a layout of one group over all of an arm's inputs."""
    return Layout(slices={"all": slice(0, len(names))}, n_inputs=len(names))


def donor_map(*, arm: arms.Arm, month: str, seed: int, site: str) -> np.ndarray | None:
    """Return the arm's donor map for a series, or `None` for a real arm.

    The shuffled-run map depends on the fold's cut and the fitting seed, and is stored under
    `work/placebo/`. The time-shifted map depends on neither.
    """
    if arm.placebo == "shuffled":
        return placebo.stored_shuffled_map(month=month, seed=seed, series_key=series_key(site=site))
    if arm.placebo == "shifted":
        return placebo.stored_shifted_map(series_key=series_key(site=site))
    return None


def extend_site(
    *,
    arm: arms.Arm,
    site: str,
    kept: np.ndarray,
    x: np.ndarray,
    origin: np.ndarray,
    lead: np.ndarray,
    month: str,
    seed: int,
    store: rowcloud.RunArrays | None,
) -> np.ndarray:
    """Return a series' raw input matrix with the arm's extra columns, for rows of one set.

    Args:
        arm: The arm.
        site: The Primary's label.
        kept: Indices of the base columns that the arm keeps.
        x: The raw base matrix, shape (rows, 46).
        origin: Each row's origin in microseconds since the epoch.
        lead: Each row's lead from the origin in hours.
        month: The fold's month, which sets the placebo's cut.
        seed: The fitting seed.
        store: The series' `RunArrays`, or `None` for an arm with no summary and no flag.

    Returns:
        The matrix of the arm's width.

    Raises:
        ValueError: If the origin rule fails on a row.
    """
    nwp_init = origin - DELAY_US
    origin_rules.assert_origin_after_run(nwp_init=nwp_init, origin=origin)
    if store is None:
        return x
    donors = donor_map(arm=arm, month=month, seed=seed, site=site)
    summaries, flag = rowcloud.row_extras(
        arm=arm,
        store=store,
        nwp_init=nwp_init,
        lead_hours=lead,
        donors=donors,
        donors_first_day=TARGET_FIRST_DAY,
    )
    return rowcloud.extend_matrix(
        x=x, kept=kept, summaries=summaries, flag=flag if arm.flag else None
    )


def store_for(*, arm: arms.Arm, site: str) -> rowcloud.RunArrays | None:
    """Return the series' store if the arm needs summaries or the flag, else `None`."""
    if arm.summary == "none" and not arm.flag:
        return None
    return rowcloud.load_store(site=site)


def build_fold(
    *, arm: arms.Arm, refit: datetime, seed: int, window: arms707.Window = arms707.W24
) -> folds707.Fold:
    """Return the standardised fine-tuning rows of a fold for an arm, with the standing cuts.

    As `e00707_folds.build_fold` (window `W24`), with the arm's extra columns appended to every
    kept row before the standardiser is fitted. With `arm.drop_gap`, the rows whose run lies in the
    19-month gap are dropped first, by the run-date rule.

    Args:
        arm: The arm.
        refit: The fold's cut.
        seed: The fitting seed (it seeds the shuffled-run placebo).
        window: The fine-tuning window.

    Returns:
        The fold, with `layout` covering all of the arm's inputs.

    Raises:
        ValueError: If a row is not known at the cut, or the window keeps no row.
    """
    names, mask, kept = input_plan(arm=arm)
    month = f"{refit:%Y-%m}"
    flagged = folds707.flagged_times(legacy=False)
    dropped = folds707.DroppedRows()
    parts = []
    for row in prep.primaries().iter_rows(named=True):
        z = prep.load_site(site=row["site"])
        keep = folds707.window_mask(z=z, refit=refit, window=window)
        if arm.drop_gap:
            keep &= rowcloud.drop_gap_rows(nwp_init=z["origin"] - DELAY_US)
        keep, counts = folds707.apply_cuts(
            keep=keep,
            valid=z["valid"],
            lead=z["lead"],
            site=row["site"],
            series_id=int(row["time_series_id"]),
            flagged=flagged,
        )
        dropped += counts
        if not keep.any():
            continue
        folds707.assert_no_covid(x=z["x"][keep], window=window, site=row["site"])
        x = extend_site(
            arm=arm,
            site=row["site"],
            kept=kept,
            x=z["x"][keep],
            origin=z["origin"][keep],
            lead=z["lead"][keep],
            month=month,
            seed=seed,
            store=store_for(arm=arm, site=row["site"]),
        )
        parts.append(
            {"x": x}
            | {k: z[k][keep] for k in ("y", "origin", "valid", "lead")}
            | {"primary": np.full(int(keep.sum()), row["primary"], dtype=np.int64)}
        )
        del z
    if not parts:
        msg = f"window {window.name} keeps no row at {refit:%Y-%m-%d}"
        raise ValueError(msg)
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    prep.assert_known_at_cut(valid=cat["valid"], origin=cat["origin"], refit=refit)
    held, purged = prep.held_out_weeks(origin_day=prep.origin_days(origin=cat["origin"]))
    train = ~held & ~purged
    standardiser = prep.Standardiser.fit(x=cat["x"][train], mask=mask)
    age_days = (_us(refit=refit) - cat["valid"]) / prep.US_PER_DAY
    weight = 0.5 ** (age_days / folds707.HALF_LIFE_DAYS)
    n_primaries = len(prep.primaries())
    target_mean, target_std = np.zeros(n_primaries), np.ones(n_primaries)
    for p in range(n_primaries):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep_rows = train | held
    return folds707.Fold(
        x=standardiser.apply(x=cat["x"][keep_rows]),
        y=cat["y"][keep_rows],
        primary=cat["primary"][keep_rows],
        weight=weight[keep_rows].astype(np.float32),
        is_val=held[keep_rows],
        lead=cat["lead"][keep_rows],
        origin=cat["origin"][keep_rows],
        valid=cat["valid"][keep_rows],
        standardiser=standardiser,
        target_mean=target_mean,
        target_std=target_std,
        layout=layout_for(names=names),
        dropped=dropped,
    )


def ens_rows(
    *, arm: arms.Arm, fold: folds707.Fold, refit: datetime, seed: int, window: arms707.Window
) -> tuple[folds707.Rows, folds707.DroppedRows]:
    """Return the unweighted `S_ens` rows of a window on the fold's input scale.

    As `e00707_folds.ens_rows`, with the arm's extra columns appended and the fold's standardiser
    (fitted on the fine-tuning rows with `cloud_missing = 0`) applied.

    Args:
        arm: The arm.
        fold: The fold from `build_fold`.
        refit: The fold's cut.
        seed: The fitting seed.
        window: The `S_ens` window (window `F` for `R_F`).

    Returns:
        The rows and the counts of rows that the standing cuts dropped.

    Raises:
        ValueError: If the window keeps no row.
    """
    _, _, kept = input_plan(arm=arm)
    month = f"{refit:%Y-%m}"
    flagged = folds707.flagged_times(legacy=False)
    dropped = folds707.DroppedRows()
    parts: list[folds707.Rows] = []
    for row in prep.primaries().iter_rows(named=True):
        z = prep.load_site(site=row["site"])
        keep = folds707.window_mask(z=z, refit=refit, window=window)
        held, purged = prep.held_out_weeks(origin_day=prep.origin_days(origin=z["origin"]))
        keep &= ~purged
        if arm.drop_gap:
            keep &= rowcloud.drop_gap_rows(nwp_init=z["origin"] - DELAY_US)
        keep, counts = folds707.apply_cuts(
            keep=keep,
            valid=z["valid"],
            lead=z["lead"],
            site=row["site"],
            series_id=int(row["time_series_id"]),
            flagged=flagged,
        )
        dropped += counts
        if not keep.any():
            continue
        folds707.assert_no_covid(x=z["x"][keep], window=window, site=row["site"])
        prep.assert_known_at_cut(valid=z["valid"][keep], origin=z["origin"][keep], refit=refit)
        x = extend_site(
            arm=arm,
            site=row["site"],
            kept=kept,
            x=z["x"][keep],
            origin=z["origin"][keep],
            lead=z["lead"][keep],
            month=month,
            seed=seed,
            store=store_for(arm=arm, site=row["site"]),
        )
        parts.append(
            folds707.Rows(
                x=fold.standardiser.apply(x=x),
                y=z["y"][keep],
                primary=np.full(int(keep.sum()), row["primary"], dtype=np.int64),
                is_val=held[keep],
                lead=z["lead"][keep],
                origin=z["origin"][keep],
                valid=z["valid"][keep],
            )
        )
        del z
    if not parts:
        msg = f"window {window.name} keeps no S_ens row at {refit:%Y-%m-%d}"
        raise ValueError(msg)
    out = folds707.Rows(
        **{k: np.concatenate([getattr(p, k) for p in parts]) for k in folds707.Rows.__annotations__}
    )
    return out, dropped


# --- scored rows ------------------------------------------------------------------------------


def recent_scored(*, month: str) -> Iterator[Scored]:
    """Yield, per scored run and Primary, the raw base matrix of the recent folds (2025-07 on).

    The rows are those of `e00707_prep.run_inputs`: experiment 87's stored 51-member frames joined
    to experiment 397's summaries. The matrix is returned raw, so that the arm's columns can be
    appended before the fold's standardiser is applied.
    """
    _series, arrays = base.series_and_arrays()
    for origin in prep.test_origins(month=month):
        run = origin - base.PUBLICATION_DELAY
        frame = pl.read_parquet(base.demand_frame_path(run=run))
        for row in prep.primaries().iter_rows(named=True):
            prepared = base.prepared_run(
                run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
                row=row,
                arrays=arrays[row["site"]],
            )
            summary = pl.read_parquet(prep.OLD_SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet").filter(
                pl.col("time_series_id") == row["time_series_id"]
            )
            first = prepared.filter(pl.col("ensemble_member") == pl.col("ensemble_member").min())
            prepared = first.join(
                summary.drop("time_series_id"),
                on=["power_fcst_init_time", "valid_time"],
                how="inner",
                validate="1:1",
                maintain_order="left",
            )
            origin_rules.assert_primary_rows(frame=prepared)
            keys = prepared.select(*base.KEYS, "lead_hours").unique(
                subset=list(base.KEYS), maintain_order=True
            )
            run_us = np.full(keys.height, int(run.timestamp() * 1_000_000), dtype=np.int64)
            yield row, keys, prep.feature_matrix(frame=prepared), run_us


def replicate_scored(*, month: str) -> Iterator[Scored]:
    """Yield, per Primary, the raw base matrix of the replicate fold's scored rows.

    The scored rows of a replicate fold (2023-04 to 2024-03) are the back-filled raw rows whose
    origin falls in the month. A row without an observed power or a finite weather input is absent
    from the raw rows, so the five signed-MW Primaries and the eight score the rows that have an
    actual.
    """
    first = _us(refit=prep.refit_of(month=month))
    stop = _us(refit=(prep.refit_of(month=month) + timedelta(days=32)).replace(day=1))
    for row in prep.primaries().iter_rows(named=True):
        z = prep.load_site(site=row["site"])
        keep = (z["origin"] >= first) & (z["origin"] < stop)
        if not keep.any():
            continue
        origin = z["origin"][keep]
        keys = pl.DataFrame(
            {
                "time_series_id": np.full(len(origin), row["time_series_id"], dtype=np.int64),
                "power_fcst_init_time": origin,
                "valid_time": z["valid"][keep],
                "lead_hours": z["lead"][keep].astype(np.float64),
            }
        ).with_columns(
            pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
            pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
        )
        yield row, keys, z["x"][keep], origin - DELAY_US


def scored_rows(*, arm: arms.Arm, month: str) -> Iterator[Scored]:
    """Return the scored rows of a fold: the replicate's from the raw rows, the others' from 707."""
    return (
        replicate_scored(month=month) if arm.folds == "replicate12" else recent_scored(month=month)
    )


def scored_input(
    *, arm: arms.Arm, scored: Scored, month: str, seed: int, standardiser: prep.Standardiser
) -> tuple[pl.DataFrame, np.ndarray]:
    """Return a series' scored keys and the standardised inputs of the arm.

    Args:
        arm: The arm.
        scored: One item of `scored_rows`.
        month: The fold's month.
        seed: The fitting seed.
        standardiser: The fold's standardiser (fitted on the extended fine-tuning rows).

    Returns:
        The keys (`time_series_id`, `power_fcst_init_time`, `valid_time`) and the inputs.
    """
    row, keys, x, run_us = scored
    _, _, kept = input_plan(arm=arm)
    origin = run_us + DELAY_US
    extended = extend_site(
        arm=arm,
        site=row["site"],
        kept=kept,
        x=x,
        origin=origin,
        lead=keys["lead_hours"].to_numpy(),
        month=month,
        seed=seed,
        store=store_for(arm=arm, site=row["site"]),
    )
    return keys.drop("lead_hours"), standardiser.apply(x=extended)
