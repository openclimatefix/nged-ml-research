"""Fold rows for experiment 707: the fine-tuning, `S_ens`, and `S_era` rows of one fold.

The fine-tuning rows follow experiment 425's `build_ens_fold` (experiment 397's summary-scheme
rows with experiment 407's cuts), and the ERA5-and-CAMS rows follow experiment 425's
`fold_pretrain` (experiment 418's rows with the same cuts). Three changes serve this experiment:

- the ENS rows are the back-filled rows from 2020-10-15 followed by experiment 397's rows from
  2024-04-01, selected by a `e00707_arms.Window` (trailing days, a floor, and a floor on lag times);
- `ens_rows` returns the unweighted rows of the `S_ens` stage, standardised on the fold's scale;
- `fold_pretrain` accepts a function that replaces the ERA5 weather of each row, for the placebo.

Three cuts apply to every set of rows. As in experiment 425, a D15 or D18 row is dropped if its
valid time or its weekly lag is before 2025-03-01, and a row is dropped if its own half-hour or any
of its three origin lags is a flagged exact zero. In addition, a row of a Primary in
`prep.FIRST_USABLE` is dropped if its valid time or its weekly lag is before the Primary's first
usable date (experiment 418 applied it to the valid time only). Every set of rows asserts that its
valid times are at least 3.5 hours before the fold's cut. The fold's standardiser and per-Primary
target statistics are fitted on the fine-tuning rows, and every stage uses them.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00707_arms as arms  # ty: ignore[unresolved-import]
import e00707_common as common  # ty: ignore[unresolved-import]
import e00707_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00707_models import Layout  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SHORT_HISTORY: Final[tuple[str, ...]] = ("D15", "D18")
CUT: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
"""A D15 or D18 row whose valid time or weekly lag is before this date is dropped."""
FLAGGED_KEYS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00398-load-shift-event-rescore-without-flagged-zero-actuals"
    / "flagged_keys.parquet"
)
"""The half-hours that the cleaned power table flags as `substation_zero` (experiment 398)."""
SPREAD_BAND_HOURS: Final[float] = 24.0
N_BANDS: Final[int] = 15
CHUNK: Final[int] = 1_000_000
HALF_LIFE_DAYS: Final[float] = common.SHORT_HALF_LIFE_DAYS
Permuter = Callable[..., np.ndarray]
"""A function `(site=, x=, valid=, refit=) -> x` that replaces the weather means of ERA5 rows."""


def _us(*, t: datetime) -> int:
    """Return a UTC datetime as microseconds since the epoch."""
    return int(t.timestamp() * 1_000_000)


@dataclass(frozen=True)
class DroppedRows:
    """How many rows (otherwise kept by the window) each cut removed."""

    short_history_start: int = 0
    before_first_usable: int = 0
    flagged_zero: int = 0

    def __add__(self, other: DroppedRows) -> DroppedRows:
        """Return the sum of two counts."""
        return DroppedRows(
            short_history_start=self.short_history_start + other.short_history_start,
            before_first_usable=self.before_first_usable + other.before_first_usable,
            flagged_zero=self.flagged_zero + other.flagged_zero,
        )

    def as_dict(self) -> dict[str, int]:
        """Return the counts as a JSON-ready mapping."""
        return {
            "short_history_start": self.short_history_start,
            "before_first_usable": self.before_first_usable,
            "flagged_zero": self.flagged_zero,
        }


@dataclass
class Fold:
    """The standardised fine-tuning rows of one fold, ready for a network.

    `x` has the 46 columns of `prep.feature_names` and no extra flag columns.
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
    dropped: DroppedRows = field(default_factory=DroppedRows)


@dataclass
class Rows:
    """Standardised rows of a training stage (no weights: the stage is unweighted)."""

    x: np.ndarray
    y: np.ndarray
    primary: np.ndarray
    is_val: np.ndarray
    lead: np.ndarray
    origin: np.ndarray
    valid: np.ndarray


def flagged_times() -> dict[int, np.ndarray]:
    """Return each series' flagged exact-zero half-hours, as microseconds since the epoch."""
    flagged = pl.read_parquet(FLAGGED_KEYS)
    return {
        int(i): np.sort(g["valid_time"].dt.epoch("us").to_numpy())
        for (i,), g in flagged.group_by("time_series_id")
    }


def window_mask(*, z: dict[str, np.ndarray], refit: datetime, window: arms.Window) -> np.ndarray:
    """Return the rows that a window keeps at a fold's cut.

    A row is known at the cut if its valid time is at least 3.5 hours before the refit and its run
    began at or before it. The window then keeps a row if its valid time is later than the refit
    minus `window.days`, and, with a floor, if its valid time and origin are at or after the floor
    and (with `floor_lags`) so is the time of its oldest lag, the weekly one.

    Args:
        z: The Primary's rows from `prep.load_site`.
        refit: The fold's cut.
        window: The start rule.

    Returns:
        A boolean mask over the rows.
    """
    keep = (z["valid"] <= _us(t=refit - common.REFIT_CUTOFF_AGE)) & (z["origin"] <= _us(t=refit))
    if window.days is not None:
        keep &= z["valid"] > _us(t=refit) - window.days * prep.US_PER_DAY
    if window.floor is not None:
        floor = _us(t=window.floor)
        keep &= (z["valid"] >= floor) & (z["origin"] >= floor)
        if window.floor_lags:
            week_age = prep.lag_ages_hours(lead_hours=z["lead"])[:, 2] * prep.US_PER_HOUR
            keep &= (z["valid"] - week_age.astype(np.int64)) >= floor
    return keep


def apply_cuts(
    *,
    keep: np.ndarray,
    valid: np.ndarray,
    lead: np.ndarray,
    site: str,
    series_id: int,
    flagged: dict[int, np.ndarray],
) -> tuple[np.ndarray, DroppedRows]:
    """Return `keep` with the start cuts and the flagged-zero drop applied.

    Three cuts apply in this order. A D15 or D18 row is dropped if its valid time or weekly-lag
    time is before 2025-03-01. A row of a Primary in `prep.FIRST_USABLE` is dropped if its valid
    time or its weekly-lag time is before that Primary's first usable date. A row is dropped if its
    own half-hour or one of its three origin lags is a flagged exact zero.

    Args:
        keep: Rows the window already keeps (not modified).
        valid: Valid time of every row, microseconds since the epoch.
        lead: Lead of every row in hours; the lag ages follow from it.
        site: The Primary's label (`D15` and `D18` get the start cut).
        series_id: The Primary's `time_series_id`, the key into `flagged`.
        flagged: Each series' flagged exact-zero half-hours, from `flagged_times`.

    Returns:
        The reduced mask, and how many of the rows in `keep` each cut removed (each cut among the
        rows that the earlier cuts left).
    """
    keep = keep.copy()
    lag_us = (prep.lag_ages_hours(lead_hours=lead) * prep.US_PER_HOUR).astype(np.int64)
    start_cut = 0
    if site in SHORT_HISTORY:
        too_early = (valid - lag_us[:, 2]) < _us(t=CUT)
        start_cut = int((keep & too_early).sum())
        keep &= ~too_early
    first_usable = 0
    usable_from = prep.FIRST_USABLE.get(site)
    if usable_from is not None:
        before = _us(t=usable_from)
        unusable = (valid < before) | ((valid - lag_us[:, 2]) < before)
        first_usable = int((keep & unusable).sum())
        keep &= ~unusable
    flagged_zero = 0
    bad = flagged.get(series_id)
    if bad is not None:
        hit = np.isin(valid, bad)
        for j in range(lag_us.shape[1]):
            hit |= np.isin(valid - lag_us[:, j], bad)
        flagged_zero = int((keep & hit).sum())
        keep &= ~hit
    return keep, DroppedRows(
        short_history_start=start_cut,
        before_first_usable=first_usable,
        flagged_zero=flagged_zero,
    )


def build_fold(*, refit: datetime, window: arms.Window, cut: bool = True) -> Fold:
    """Return a fold's fine-tuning rows for a window, with the standing cuts applied.

    With the window `W24_STORED` the rows equal experiment 425's `build_ens_fold`. The standardiser
    and the per-Primary target statistics are fitted on the rows that remain, excluding the held-out
    weeks and their purge. Row weights halve every half-year of age at the refit.

    Raises:
        ValueError: If a row is not known at the cut, or the window keeps no row.
    """
    _, mask, layout = prep.feature_names()
    flagged = flagged_times() if cut else {}
    dropped = DroppedRows()
    parts = []
    for row in prep.primaries().iter_rows(named=True):
        z = prep.load_site(site=row["site"])
        keep = window_mask(z=z, refit=refit, window=window)
        if cut:
            keep, counts = apply_cuts(
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
        parts.append(
            {k: z[k][keep] for k in ("x", "y", "origin", "valid", "lead")}
            | {"primary": np.full(int(keep.sum()), row["primary"], dtype=np.int64)}
        )
    if not parts:
        msg = f"window {window.name} keeps no row at {refit:%Y-%m-%d}"
        raise ValueError(msg)
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    prep.assert_known_at_cut(valid=cat["valid"], origin=cat["origin"], refit=refit)
    held, purged = prep.held_out_weeks(origin_day=prep.origin_days(origin=cat["origin"]))
    train = ~held & ~purged
    standardiser = prep.Standardiser.fit(x=cat["x"][train], mask=mask)
    age_days = (_us(t=refit) - cat["valid"]) / prep.US_PER_DAY
    weight = 0.5 ** (age_days / HALF_LIFE_DAYS)
    n_primaries = len(prep.primaries())
    target_mean, target_std = np.zeros(n_primaries), np.ones(n_primaries)
    for p in range(n_primaries):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep = train | held
    return Fold(
        x=standardiser.apply(x=cat["x"][keep]),
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
    )


def ens_rows(
    *, fold: Fold, refit: datetime, window: arms.Window, cut: bool = True
) -> tuple[Rows, DroppedRows]:
    """Return the unweighted `S_ens` rows of a window on the fold's input scale, and the cut counts.

    The rows are the window's rows without the purged weeks; the held-out weeks are the early-
    stopping rows, as in the fold. The standardiser and target statistics are the fold's.

    Raises:
        ValueError: If the window keeps no row.
    """
    flagged = flagged_times() if cut else {}
    dropped = DroppedRows()
    parts: list[Rows] = []
    for row in prep.primaries().iter_rows(named=True):
        z = prep.load_site(site=row["site"])
        keep = window_mask(z=z, refit=refit, window=window)
        held, purged = prep.held_out_weeks(origin_day=prep.origin_days(origin=z["origin"]))
        keep &= ~purged
        if cut:
            keep, counts = apply_cuts(
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
        prep.assert_known_at_cut(valid=z["valid"][keep], origin=z["origin"][keep], refit=refit)
        parts.append(
            Rows(
                x=fold.standardiser.apply(x=z["x"][keep]),
                y=z["y"][keep],
                primary=np.full(n, row["primary"], dtype=np.int64),
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
    out = Rows(**{k: np.concatenate([getattr(p, k) for p in parts]) for k in Rows.__annotations__})
    return out, dropped


# --- ERA5 and CAMS rows (S_era) -----------------------------------------------------------------


def column_indices() -> tuple[np.ndarray, np.ndarray]:
    """Return the column indices of the `mean_*` and the `std_*` inputs, in `WEATHER` order."""
    names, _, _ = prep.feature_names()
    return (
        np.array([names.index(f"mean_{w}") for w in common.WEATHER]),
        np.array([names.index(f"std_{w}") for w in common.WEATHER]),
    )


def spread_table(*, fold: Fold) -> np.ndarray:
    """Return the median standardised ENS spread by lead band of the fold's training rows.

    The array has shape (bands, 7): the `std_*` inputs that `S_era` rows take, because ERA5 has no
    ensemble spread.
    """
    _, std_cols = column_indices()
    train = ~fold.is_val
    band = np.minimum((fold.lead[train] // SPREAD_BAND_HOURS).astype(int), N_BANDS - 1)
    x = fold.x[train]
    table = np.zeros((N_BANDS, len(common.WEATHER)), dtype=np.float32)
    for b in range(N_BANDS):
        rows = band == b
        if rows.any():
            table[b] = np.median(x[rows][:, std_cols], axis=0)
    return table


def fold_pretrain(
    *,
    fold: Fold,
    refit: datetime,
    cut: bool = True,
    permute: Permuter | None = None,
) -> tuple[Rows, DroppedRows]:
    """Return a fold's ERA5-and-CAMS rows on the fold's input scale, and the cut counts.

    As experiment 425's `fold_pretrain`: a D15 or D18 row is dropped if its target or its weekly
    lag is before 2025-03-01, and any row whose target or lag half-hour is a flagged exact zero is
    dropped. The `std_*` columns take the fold's median ENS spread by lead band. A row is kept only
    if its valid time is at least 3.5 hours before the refit and its origin is at or before it; the
    held-out weeks and their purge are the same absolute weeks as the fold's.

    Args:
        fold: The fold, whose standardiser and spread table the rows use.
        refit: The fold's cut.
        cut: Whether the two cuts apply.
        permute: If given, a function `(site=, x=, valid=, refit=) -> x` that returns the raw
            (unstandardised) rows with the weather means replaced (the placebo).

    Returns:
        The rows and the cut counts.
    """
    table = spread_table(fold=fold)
    _, std_cols = column_indices()
    cutoff_valid, cutoff_origin = _us(t=refit - common.REFIT_CUTOFF_AGE), _us(t=refit)
    flagged = flagged_times() if cut else {}
    dropped = DroppedRows()
    parts: list[Rows] = []
    for row in prep.primaries().iter_rows(named=True):
        z = np.load(prep.ERA_RAW_DIR / f"{row['site']}.npz")
        origin, valid = z["origin"], z["valid"]
        keep = (valid <= cutoff_valid) & (origin <= cutoff_origin)
        held, purged = prep.held_out_weeks(origin_day=prep.origin_days(origin=origin))
        keep &= ~purged
        if cut:
            keep, counts = apply_cuts(
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
        raw, lead = z["x"][keep], z["lead"][keep]
        if permute is not None:
            raw = permute(site=row["site"], x=raw, valid=valid[keep], refit=refit)
        band = np.minimum((lead // SPREAD_BAND_HOURS).astype(int), N_BANDS - 1)
        x = np.empty_like(raw, dtype=np.float32)
        for a in range(0, len(raw), CHUNK):
            b = min(a + CHUNK, len(raw))
            chunk = fold.standardiser.apply(x=raw[a:b])
            chunk[:, std_cols] = table[band[a:b]]
            x[a:b] = chunk
        parts.append(
            Rows(
                x=x,
                y=z["y"][keep],
                primary=np.full(len(x), row["primary"], dtype=np.int64),
                is_val=held[keep],
                lead=lead,
                origin=origin[keep],
                valid=valid[keep],
            )
        )
    out = Rows(**{k: np.concatenate([getattr(p, k) for p in parts]) for k in Rows.__annotations__})
    return out, dropped


# --- the two extra inputs of R_F+flags ------------------------------------------------------


def flag_columns(*, origin: np.ndarray) -> np.ndarray:
    """Return the two indicators of `R_F+flags`, shape (rows, 2), float32.

    An indicator is 1 for a row whose ENS run was initialised on or after the date (the run is the
    origin minus 9 hours). The dates are the resolution upgrade and the source change.
    """
    nwp_init = origin - int(common.PUBLICATION_DELAY.total_seconds() * 1_000_000)
    return np.stack(
        [(nwp_init >= _us(t=date)).astype(np.float32) for date in arms.FLAG_DATES],
        axis=1,
    )
