"""Fold rows and inputs for the library network of experiment 413.

The rows are experiment 397's `summary` raw matrices (experiment 87's cleaned control-member rows
with the ensemble mean and spread of each weather column), with three differences: the D15 and D18
short-history cut of experiment 405 (training rows, and the lags that fall before the cut), four
yearly harmonic columns appended, and a memory-lean build that standardises one Primary at a time.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00413_models as models  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SHORT_HISTORY_SITES: Final[tuple[str, ...]] = ("D15", "D18")
SHORT_HISTORY_CUT: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
"""D15 and D18 drop every row, lag input, and scoring actual before this instant (`common.md`)."""
US_PER_HOUR: Final[int] = 3_600_000_000
REFERENCE_TEMPERATURE_C: Final[float] = 15.5
NAMES, MASK, LAYOUT = base.feature_names(scheme="summary")
CALENDAR_HOLIDAYS: Final[tuple[str, ...]] = tuple(
    n
    for g in ("calendar", "holidays")
    for n in NAMES[LAYOUT.slices[g].start : LAYOUT.slices[g].stop]
)


def col(*, name: str) -> int:
    """Return an input column index."""
    return NAMES.index(name)


def yearly_columns(*, valid_us: np.ndarray) -> np.ndarray:
    """Return the sine and cosine of the day of the year at one and two cycles, float32."""
    doy = pl.Series(valid_us).cast(pl.Datetime("us")).dt.ordinal_day().to_numpy().astype(np.float64)
    angle = 2.0 * np.pi * doy / 365.25
    return np.stack(
        [np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle)], axis=1
    ).astype(np.float32)


def mask_short_history_lags(*, x: np.ndarray, valid_us: np.ndarray, site: str) -> None:
    """Set the lags that fall before the short-history cut to missing, in place (D15 and D18)."""
    if site not in SHORT_HISTORY_SITES:
        return
    cut_us = int(SHORT_HISTORY_CUT.timestamp() * 1_000_000)
    for tag in ("origin", "day", "week"):
        age = x[:, col(name=f"lag_{tag}_age")].astype(np.float64) * base.LEAD_SCALE_HOURS
        lag_time = valid_us - np.rint(age * 2.0).astype(np.int64) * (US_PER_HOUR // 2)
        before = lag_time < cut_us
        x[before, col(name=f"lag_{tag}")] = np.nan
        x[before, col(name=f"lag_{tag}_missing")] = 1.0


@dataclass
class Fold:
    """The standardised training rows of one fold, with the yearly columns appended.

    Attributes:
        x: Standardised inputs, float32, `base.feature_names` columns then four yearly columns.
        y: The target, the power over the effective capacity.
        primary: The Primary index of each row.
        weight: The recency weight of each row.
        is_val: Whether the row is in an early-stopping week.
        standardiser: The mean, standard deviation and mask of the standardised columns.
        target_mean: Each Primary's mean target on the training rows.
        target_std: Each Primary's standard deviation of the target on the training rows.
        dropped: Rows dropped per site because the target is missing (flagged or unobserved).
        columns: The column indices the physics network reads.
    """

    x: np.ndarray
    y: np.ndarray
    primary: np.ndarray
    weight: np.ndarray
    is_val: np.ndarray
    standardiser: base.Standardiser
    target_mean: np.ndarray
    target_std: np.ndarray
    dropped: dict[str, int]
    columns: models.Columns


def _us(*, t: datetime) -> int:
    return int(t.timestamp() * 1_000_000)


def columns_of(*, standardiser: base.Standardiser) -> models.Columns:
    """Return the column indices and the reference values, in standardised units."""
    n_base = len(NAMES)
    t = col(name="mean_temperature_2m")
    s = col(name="mean_shortwave")
    reference = (REFERENCE_TEMPERATURE_C - standardiser.mean[t]) / standardiser.std[t]
    if not -5.0 < reference < 5.0:
        msg = f"reference temperature {reference:.2f} sd is outside the clip range"
        raise ValueError(msg)
    return models.Columns(
        calendar_holidays=[col(name=n) for n in CALENDAR_HOLIDAYS],
        lags_lead_spread=[
            *[col(name=n) for n in NAMES[LAYOUT.slices["lags"]]],
            *[col(name=n) for n in NAMES[LAYOUT.slices["lead"]]],
            *[col(name=n) for n in NAMES if n.startswith("std_")],
        ],
        temperature=t,
        shortwave=s,
        cos_zenith=col(name="cos_zenith"),
        yearly=list(range(n_base, n_base + models.N_YEARLY)),
        n_base=n_base,
        temperature_reference=float(
            (REFERENCE_TEMPERATURE_C - standardiser.mean[t]) / standardiser.std[t]
        ),
        shortwave_zero=float((0.0 - standardiser.mean[s]) / standardiser.std[s]),
        temperature_per_std=float(standardiser.std[t]),
    )


def _standardise(*, xs: list[np.ndarray], train: np.ndarray) -> base.Standardiser:
    """Fit the standardiser on the training rows, then standardise every Primary's rows in place.

    Statistics are streamed Primary by Primary, so the fold never holds a second copy of the matrix.
    The yearly columns are left as they are.
    """
    n_base = len(NAMES)
    sums, squares, counts = np.zeros(n_base), np.zeros(n_base), np.zeros(n_base)
    offset = 0
    for x in xs:
        t = train[offset : offset + len(x)]
        offset += len(x)
        block = x[t, :n_base].astype(np.float64)
        finite = np.isfinite(block)
        block = np.where(finite, block, 0.0)
        sums += block.sum(0)
        squares += (block**2).sum(0)
        counts += finite.sum(0)
    mean = sums / np.maximum(counts, 1)
    std = np.sqrt(np.maximum(squares / np.maximum(counts, 1) - mean**2, 0.0))
    standardiser = base.Standardiser(
        mean=np.where(MASK, mean, 0.0),
        std=np.maximum(np.where(MASK, std, 1.0), 1e-6),
        mask=MASK,
    )
    mean32, std32 = standardiser.mean.astype(np.float32), standardiser.std.astype(np.float32)
    for x in xs:
        z = (x[:, :n_base] - mean32) / std32
        np.clip(z, -base.INPUT_CLIP_SD, base.INPUT_CLIP_SD, out=z)
        x[:, :n_base] = np.nan_to_num(z, nan=0.0)
    return standardiser


def build_fold(*, refit: datetime) -> Fold:
    """Return a fold's rows under experiment 397's rule, with the short-history cut applied.

    A fold sees a row only if its valid time is at least 3.5 hours before the refit and its run
    began at or before the refit, within the 24 months before the refit, and one week in 12 is held
    out (with purging) for early stopping. Rows whose target is missing (the flagged
    `substation_zero` half-hours and unobserved ones) are dropped and counted.
    """
    table = base.primaries()
    parts: list[dict[str, np.ndarray]] = []
    dropped: dict[str, int] = {}
    for row in table.iter_rows(named=True):
        z = np.load(base.RAW_DIR / "summary" / f"{row['site']}.npz")
        valid, origin = z["valid"], z["origin"]
        keep = (
            (valid <= _us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (origin <= _us(t=refit))
            & (valid > _us(t=refit - timedelta(days=base.WINDOW_DAYS)))
        )
        if row["site"] in SHORT_HISTORY_SITES:
            keep &= valid >= _us(t=SHORT_HISTORY_CUT)
        if not keep.any():
            continue
        x = z["x"][keep].copy()
        y = z["y"][keep].astype(np.float32)
        mask_short_history_lags(x=x, valid_us=valid[keep], site=row["site"])
        usable = np.isfinite(y)
        dropped[row["site"]] = int((~usable).sum())
        parts.append(
            {
                "x": np.concatenate([x[usable], yearly_columns(valid_us=valid[keep][usable])], 1),
                "y": y[usable],
                "origin": origin[keep][usable],
                "valid": valid[keep][usable],
                "primary": np.full(int(usable.sum()), row["primary"], dtype=np.int64),
            }
        )
    cat_origin = np.concatenate([p["origin"] for p in parts])
    cat_valid = np.concatenate([p["valid"] for p in parts])
    cat_primary = np.concatenate([p["primary"] for p in parts])
    cat_y = np.concatenate([p["y"] for p in parts])
    origin_day = ((cat_origin - _us(t=base.FIRST_RUN)) // (86_400 * 1_000_000)).astype(np.int64)
    held, purged = base.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    xs = [p.pop("x") for p in parts]
    standardiser = _standardise(xs=xs, train=train)
    age_days = (_us(t=refit) - cat_valid) / (86_400 * 1_000_000)
    weight = (0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)).astype(np.float32)
    n_primaries = len(table)
    target_mean, target_std = np.zeros(n_primaries), np.ones(n_primaries)
    for p in range(n_primaries):
        sel = train & (cat_primary == p)
        if sel.any():
            target_mean[p] = cat_y[sel].mean()
            target_std[p] = max(float(cat_y[sel].std()), 1e-3)
    keep = train | held
    x_all = np.concatenate(xs)[keep]
    del xs
    return Fold(
        x=x_all,
        y=cat_y[keep],
        primary=cat_primary[keep],
        weight=weight[keep],
        is_val=held[keep],
        standardiser=standardiser,
        target_mean=target_mean,
        target_std=target_std,
        dropped=dropped,
        columns=columns_of(standardiser=standardiser),
    )


def test_inputs(
    *, origin: datetime, standardiser: base.Standardiser
) -> list[tuple[dict, pl.DataFrame, np.ndarray]]:
    """Return, for one forecast run, each Primary's keys and inputs with the yearly columns."""
    out = []
    for row, keys, x in base.run_inputs(origin=origin, scheme="summary", standardiser=standardiser):
        valid_us = keys["valid_time"].dt.epoch("us").to_numpy()
        out.append((row, keys, np.concatenate([x, yearly_columns(valid_us=valid_us)], axis=1)))
    return out
