"""Fold rows for experiment 710: copies of experiment 707's row builders that take a power world.

**Four functions are copied from `e00707_folds.py` and `e00707_prep.py`, because no script of
experiment 707 may be edited:** `build_fold`, `ens_rows`, `fold_pretrain`, and `run_inputs`. Each
takes `power`, a dictionary of each Primary's power on the grid by site label. With `power=None`
the rows equal experiment 707's rows. With a world from `e00710_inject`, only the three lag value
columns of the non-held-out rows change.

**The standardiser and the target statistics are fitted on the unshifted rows** of the last 730
days, so the `i` and `n` states share one input scale, and the standardised lags of the held-out
rows are the same in both states. The `Fold` and `Rows` records carry a `shifted` mask: whether the
row has a lag value that the world changed.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

import e00707_arms as arms707  # ty: ignore[unresolved-import]
import e00707_common as common  # ty: ignore[unresolved-import]
import e00707_folds as folds707  # ty: ignore[unresolved-import]
import e00707_prep as prep  # ty: ignore[unresolved-import]
import e00710_arms as arms  # ty: ignore[unresolved-import]
import e00710_inject as inject  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00707_models import Layout  # ty: ignore[unresolved-import]

_us = folds707._us
CHUNK = folds707.CHUNK


@dataclass
class Fold:
    """The standardised fine-tuning rows of one fold, with the mask of shifted-lag rows.

    The fields are those of `e00707_folds.Fold`, plus `shifted`.
    """

    x: np.ndarray
    y: np.ndarray
    primary: np.ndarray
    weight: np.ndarray
    is_val: np.ndarray
    lead: np.ndarray
    origin: np.ndarray
    valid: np.ndarray
    standardiser: prep.Standardiser
    target_mean: np.ndarray
    target_std: np.ndarray
    layout: Layout
    dropped: folds707.DroppedRows = field(default_factory=folds707.DroppedRows)
    legacy_flags: bool = False
    shifted: np.ndarray | None = None


@dataclass
class Rows:
    """Standardised rows of a training stage, with the mask of shifted-lag rows.

    The fields are those of `e00707_folds.Rows`, plus `shifted`.
    """

    x: np.ndarray
    y: np.ndarray
    primary: np.ndarray
    is_val: np.ndarray
    lead: np.ndarray
    origin: np.ndarray
    valid: np.ndarray
    shifted: np.ndarray | None = None


def _concat(*, parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Return the parts' arrays joined, key by key."""
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def build_fold(
    *,
    refit: datetime,
    window: arms707.Window,
    power: dict[str, np.ndarray] | None = None,
    cut: bool = True,
) -> Fold:
    """Return a fold's fine-tuning rows for a window, with a power world applied to the lags.

    This is `e00707_folds.build_fold` with one change: the lag values of the non-held-out rows are
    rebuilt from `power`, and the standardiser and target statistics are fitted on the unshifted
    rows.

    Args:
        refit: The fold's refit time.
        window: The window whose rows are kept.
        power: Each Primary's shifted power by site label, or `None` for no injection.
        cut: Whether to apply the standing cuts.

    Returns:
        The fold's standardised rows, the standardiser, and the target statistics.

    Raises:
        ValueError: If a row is not known at the cut, or the window keeps no row.
    """
    _, mask, layout = prep.feature_names()
    legacy = window == arms707.W24_STORED
    flagged = folds707.flagged_times(legacy=legacy) if cut else {}
    dropped = folds707.DroppedRows()
    parts = []
    for row in prep.primaries().iter_rows(named=True):
        z = prep.load_site(site=row["site"])
        keep = folds707.window_mask(z=z, refit=refit, window=window)
        if cut:
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
        held, _ = prep.held_out_weeks(origin_day=prep.origin_days(origin=z["origin"][keep]))
        x_shift, shifted = inject.shift_rows(
            x=z["x"][keep],
            valid_us=z["valid"][keep],
            origin_us=z["origin"][keep],
            held=held,
            power=None if power is None else power[row["site"]],
            observed=None if power is None else inject.observed_power()[row["site"]],
        )
        parts.append(
            {k: z[k][keep] for k in ("x", "y", "origin", "valid", "lead")}
            | ({} if power is None else {"x_shift": x_shift})
            | {
                "shifted": shifted,
                "primary": np.full(int(keep.sum()), row["primary"], dtype=np.int64),
            }
        )
    if not parts:
        msg = f"window {window.name} keeps no row at {refit:%Y-%m-%d}"
        raise ValueError(msg)
    cat = _concat(parts=parts)
    del parts
    if power is None:
        cat["x_shift"] = cat["x"]
    prep.assert_known_at_cut(valid=cat["valid"], origin=cat["origin"], refit=refit)
    held, purged = prep.held_out_weeks(origin_day=prep.origin_days(origin=cat["origin"]))
    train = ~held & ~purged
    standardiser = prep.Standardiser.fit(x=cat["x"][train], mask=mask)
    age_days = (_us(t=refit) - cat["valid"]) / prep.US_PER_DAY
    weight = 0.5 ** (age_days / folds707.HALF_LIFE_DAYS)
    n_primaries = len(prep.primaries())
    target_mean, target_std = np.zeros(n_primaries), np.ones(n_primaries)
    for p in range(n_primaries):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep = train | held
    return Fold(
        x=standardiser.apply(x=cat["x_shift"][keep]),
        y=cat["y"][keep],
        primary=cat["primary"][keep],
        weight=weight[keep].astype(np.float32),
        is_val=held[keep],
        lead=cat["lead"][keep],
        origin=cat["origin"][keep],
        valid=cat["valid"][keep],
        standardiser=standardiser,
        target_mean=target_mean,
        target_std=target_std,
        layout=layout,
        dropped=dropped,
        legacy_flags=legacy,
        shifted=cat["shifted"][keep],
    )


def ens_rows(
    *,
    standardiser: prep.Standardiser,
    refit: datetime,
    window: arms707.Window,
    power: dict[str, np.ndarray] | None = None,
    legacy_flags: bool = False,
    cut: bool = True,
) -> tuple[Rows, folds707.DroppedRows]:
    """Return the unweighted `S_ens` rows of a window on the fold's scale, with a power world.

    This is `e00707_folds.ens_rows` with one change: the lag values of the non-held-out rows are
    rebuilt from `power`.

    Args:
        standardiser: The fold's standardiser, fitted on the unshifted rows.
        refit: The fold's refit time.
        window: The window whose rows are kept.
        power: Each Primary's shifted power by site label, or `None` for no injection.
        legacy_flags: Whether to read experiment 398's list of flagged half-hours.
        cut: Whether to apply the standing cuts.

    Returns:
        The window's rows (purged weeks removed, held-out weeks flagged) and the cut counts.

    Raises:
        ValueError: If the window keeps no row.
    """
    flagged = folds707.flagged_times(legacy=legacy_flags) if cut else {}
    dropped = folds707.DroppedRows()
    parts: list[dict[str, np.ndarray]] = []
    for row in prep.primaries().iter_rows(named=True):
        z = prep.load_site(site=row["site"])
        keep = folds707.window_mask(z=z, refit=refit, window=window)
        held, purged = prep.held_out_weeks(origin_day=prep.origin_days(origin=z["origin"]))
        keep &= ~purged
        if cut:
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
        n = int(keep.sum())
        folds707.assert_no_covid(x=z["x"][keep], window=window, site=row["site"])
        prep.assert_known_at_cut(valid=z["valid"][keep], origin=z["origin"][keep], refit=refit)
        raw, shifted = inject.shift_rows(
            x=z["x"][keep],
            valid_us=z["valid"][keep],
            origin_us=z["origin"][keep],
            held=held[keep],
            power=None if power is None else power[row["site"]],
            observed=None if power is None else inject.observed_power()[row["site"]],
        )
        parts.append(
            {
                "x": standardiser.apply(x=raw),
                "y": z["y"][keep],
                "primary": np.full(n, row["primary"], dtype=np.int64),
                "is_val": held[keep],
                "lead": z["lead"][keep],
                "origin": z["origin"][keep],
                "valid": z["valid"][keep],
                "shifted": shifted,
            }
        )
        del z, raw
    if not parts:
        msg = f"window {window.name} keeps no S_ens row at {refit:%Y-%m-%d}"
        raise ValueError(msg)
    return Rows(**_concat(parts=parts)), dropped


def fold_pretrain(
    *,
    standardiser: prep.Standardiser,
    spread: np.ndarray,
    refit: datetime,
    power: dict[str, np.ndarray] | None = None,
    legacy_flags: bool = False,
    cut: bool = True,
) -> tuple[Rows, folds707.DroppedRows]:
    """Return a fold's ERA5-and-CAMS rows on the fold's scale, with a power world.

    This is `e00707_folds.fold_pretrain` (without the placebo's permuter) with one change: the lag
    values of the non-held-out rows are rebuilt from `power`.

    Args:
        standardiser: The fold's standardiser, fitted on the unshifted rows.
        spread: The fold's median ENS spread by lead band (`e00707_folds.spread_table`).
        refit: The fold's cut.
        power: Each Primary's shifted power by site label, or `None` for no injection.
        legacy_flags: Whether to read experiment 398's list of flagged half-hours.
        cut: Whether the standing cuts apply.

    Returns:
        The rows and the cut counts.
    """
    _, std_cols = folds707.column_indices()
    cutoff_valid, cutoff_origin = _us(t=refit - common.REFIT_CUTOFF_AGE), _us(t=refit)
    flagged = folds707.flagged_times(legacy=legacy_flags) if cut else {}
    dropped = folds707.DroppedRows()
    parts: list[dict[str, np.ndarray]] = []
    for row in prep.primaries().iter_rows(named=True):
        z = np.load(prep.ERA_RAW_DIR / f"{row['site']}.npz")
        origin, valid = z["origin"], z["valid"]
        keep = (valid <= cutoff_valid) & (origin <= cutoff_origin)
        held, purged = prep.held_out_weeks(origin_day=prep.origin_days(origin=origin))
        keep &= ~purged
        if cut:
            keep, counts = folds707.apply_cuts(
                keep=keep,
                valid=valid,
                lead=z["lead"],
                site=row["site"],
                series_id=int(row["time_series_id"]),
                flagged=flagged,
            )
            dropped += counts
        if not keep.any():
            continue
        prep.assert_known_at_cut(valid=valid[keep], origin=origin[keep], refit=refit)
        lead = z["lead"][keep]
        raw, shifted = inject.shift_rows(
            x=z["x"][keep],
            valid_us=valid[keep],
            origin_us=origin[keep],
            held=held[keep],
            power=None if power is None else power[row["site"]],
            observed=None if power is None else inject.observed_power()[row["site"]],
        )
        band = np.minimum((lead // folds707.SPREAD_BAND_HOURS).astype(int), folds707.N_BANDS - 1)
        x = np.empty_like(raw, dtype=np.float32)
        for a in range(0, len(raw), CHUNK):
            b = min(a + CHUNK, len(raw))
            chunk = standardiser.apply(x=raw[a:b])
            chunk[:, std_cols] = spread[band[a:b]]
            x[a:b] = chunk
        parts.append(
            {
                "x": x,
                "y": z["y"][keep],
                "primary": np.full(len(x), row["primary"], dtype=np.int64),
                "is_val": held[keep],
                "lead": lead,
                "origin": origin[keep],
                "valid": valid[keep],
                "shifted": shifted,
            }
        )
    return Rows(**_concat(parts=parts)), dropped


def run_inputs(
    *,
    origin: datetime,
    standardiser: prep.Standardiser,
    power: dict[str, np.ndarray] | None = None,
) -> list[tuple[dict, pl.DataFrame, np.ndarray]]:
    """Return, for one scored run, each Primary's keys, summary rows, and standardised inputs.

    This is `e00707_prep.run_inputs` with one change: the lags are built from `power` (each
    Primary's power on the grid by site label) instead of the observed power. With `power=None` the
    result equals experiment 707's.

    Args:
        origin: The run's `power_fcst_init_time`.
        standardiser: The fold's standardiser.
        power: Each Primary's power by site label, such as one pass of experiment 581's bank or the
            corrected power of the verified events; `None` for the observed power.

    Returns:
        One item per Primary: (series row, keys frame, standardised inputs), one row per valid time.
    """
    _series, arrays = common.series_and_arrays()
    frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
    out = []
    for row in prep.primaries().iter_rows(named=True):
        site_arrays = arrays[row["site"]]
        if power is not None:
            site_arrays = replace(site_arrays, power=power[row["site"]])
        prepared = common.prepared_run(
            run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
            row=row,
            arrays=site_arrays,
        )
        summary = pl.read_parquet(
            prep.OLD_SUMMARY_DIR / f"init={origin - common.PUBLICATION_DELAY:%Y-%m-%d}.parquet"
        ).filter(pl.col("time_series_id") == row["time_series_id"])
        first = prepared.filter(pl.col("ensemble_member") == pl.col("ensemble_member").min())
        prepared = first.join(
            summary.drop("time_series_id"),
            on=["power_fcst_init_time", "valid_time"],
            how="inner",
            validate="1:1",
            maintain_order="left",
        )
        prep.assert_origin_after_run(
            nwp_init=prepared["nwp_init_time"].dt.epoch("us").to_numpy(),
            origin=prepared["power_fcst_init_time"].dt.epoch("us").to_numpy(),
        )
        keys = prepared.select(*common.KEYS, "lead_hours").unique(
            subset=list(common.KEYS), maintain_order=True
        )
        out.append((row, keys, standardiser.apply(x=prep.feature_matrix(frame=prepared))))
    return out


def tuning_rows(
    *, standardiser: prep.Standardiser, month: str
) -> list[tuple[dict, pl.DataFrame, np.ndarray, np.ndarray]]:
    """Return a tuning fold's scored rows: the cached ENS rows whose origin falls in the month.

    Args:
        standardiser: The fold's standardiser.
        month: The tuning month, such as `2025-01`.

    Returns:
        One item per Primary: (series row, keys frame with `lead_hours`, standardised inputs, the
        target as a fraction of capacity). Only rows with a finite target and a lead of at most 240
        hours, and a valid time before the first scored fold's refit, are kept; the flagged exact
        zeros are dropped by `folds707.apply_cuts`'s list.
    """
    first = prep.refit_of(month=month)
    start, stop = _us(t=first), _us(t=(first + timedelta(days=32)).replace(day=1))
    first_scored = _us(t=prep.refit_of(month=arms.SCORED_MONTHS[0]))
    flagged = folds707.flagged_times()
    out = []
    for row in prep.primaries().iter_rows(named=True):
        z = prep.load_site(site=row["site"])
        keep = (z["origin"] >= start) & (z["origin"] < stop) & (z["lead"] <= 240.0)
        keep &= z["valid"] < first_scored
        bad = flagged.get(int(row["time_series_id"]))
        if bad is not None:
            keep &= ~np.isin(z["valid"], bad)
        if not keep.any():
            continue
        keys = pl.DataFrame(
            {
                "time_series_id": np.full(int(keep.sum()), int(row["time_series_id"])),
                "power_fcst_init_time": pl.Series(z["origin"][keep]).cast(pl.Datetime("us", "UTC")),
                "valid_time": pl.Series(z["valid"][keep]).cast(pl.Datetime("us", "UTC")),
                "lead_hours": z["lead"][keep],
            }
        )
        out.append((row, keys, standardiser.apply(x=z["x"][keep]), z["y"][keep]))
    return out
