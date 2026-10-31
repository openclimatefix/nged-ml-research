"""Training rows with a varied forecast origin, reading delay, and weather run (experiment 422).

Experiment 397's rows fix the forecast origin at 09:00 UTC (the 00:00 UTC run plus the 9-hour
publication delay). Each arm here redraws, per row and per seed, the origin offset `h` after the run
start, the delay `d` of the newest power reading, and whether the previous run supplies the weather.
The lead, the three origin lags, their ages, and their missing flags are recomputed from the drawn
origin; the weather, solar, calendar, and holiday columns are experiment 397's. Arms T2 to T4 add
the age of the weather forecast and the age of the newest reading as two extra inputs.

The functions are imported by `e00422_fit.py` and `e00422_predict.py`.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00410_stale_predict as stale  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00397_models import Layout  # ty: ignore[unresolved-import]

AGE_COLUMNS: Final[tuple[str, ...]] = ("weather_age", "power_age")
AGE_SCALE_HOURS: Final[float] = 48.0
US_PER_HOUR: Final[int] = 3_600_000_000
STEP_US: Final[int] = 1_800_000_000
PUBLICATION_HOURS: Final[float] = common.PUBLICATION_DELAY.total_seconds() / 3600.0
NEWEST_AGE_HOURS: Final[float] = 3.5
GRID_START_US: Final[int] = int(common.GRID_START.timestamp() * 1e6)
OFFSETS_VARIED: Final[tuple[float, ...]] = (9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0)
DELAYS_VARIED: Final[tuple[float, ...]] = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0)
PREVIOUS_RUN_PROBABILITY: Final[float] = 0.25
LAG_DROP_PROBABILITY: Final[float] = 0.10
CYCLE_DAYS: Final[int] = 7 * prep.HELD_WEEK_PERIOD
PURGE_BEFORE_DAYS: Final[int] = 15
PURGE_AFTER_DAYS: Final[int] = 9
"""Experiment 397 purges 7 days after a held-out week; a delay of up to 24 hours adds 2 days."""


@dataclass(frozen=True)
class TrainArm:
    """How an arm redraws the origin, the reading delay, the weather run, and the lags."""

    name: str
    offsets: tuple[float, ...]
    delays: tuple[float, ...]
    previous_run_probability: float
    lag_drop_probability: float
    age_inputs: bool


TRAIN_ARMS: Final[tuple[TrainArm, ...]] = (
    TrainArm("T0", (9.0,), (0.0,), 0.0, 0.0, False),
    TrainArm("T1", OFFSETS_VARIED, (0.0,), 0.0, 0.0, False),
    TrainArm("T2", OFFSETS_VARIED, DELAYS_VARIED, 0.0, 0.0, True),
    TrainArm("T2n", OFFSETS_VARIED, DELAYS_VARIED, 0.0, 0.0, False),
    TrainArm("T3", OFFSETS_VARIED, DELAYS_VARIED, PREVIOUS_RUN_PROBABILITY, 0.0, True),
    TrainArm(
        "T4", OFFSETS_VARIED, DELAYS_VARIED, PREVIOUS_RUN_PROBABILITY, LAG_DROP_PROBABILITY, True
    ),
)


def arm_by_name(*, name: str) -> TrainArm:
    """Return a training arm by name."""
    return next(a for a in TRAIN_ARMS if a.name == name)


def feature_names(*, age_inputs: bool) -> tuple[list[str], np.ndarray, Layout]:
    """Return experiment 397's summary-scheme names, standardise mask, and layout, plus the ages."""
    names, mask, layout = prep.feature_names(scheme="summary")
    if not age_inputs:
        return names, mask, layout
    names = [*names, *AGE_COLUMNS]
    mask = np.concatenate([mask, np.zeros(len(AGE_COLUMNS), dtype=bool)])
    return names, mask, Layout(slices=layout.slices, n_inputs=len(names))


def cleaned_arrays() -> dict[str, np.ndarray]:
    """Return each Primary's power over capacity, NaN at flagged half-hours and before the cut."""
    series, arrays = fd._series_and_arrays()
    clean = stale.cleaned_power(
        arrays=arrays, series=series.filter(pl.col("substation_type") == "Primary")
    )
    cut = int(common.grid_index(times=stale.SHORT_HISTORY_CUT))
    for label in stale.SHORT_HISTORY_LABELS:
        clean[label][:cut] = np.nan
    return clean


def purge_masks(*, origin_day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return held-out (early stopping) and purged masks by run day, with a 9-day purge after."""
    cycle = origin_day % CYCLE_DAYS
    held_start = CYCLE_DAYS - 7
    held = cycle >= held_start
    purged = ((cycle >= held_start - PURGE_BEFORE_DAYS) & ~held) | (cycle < PURGE_AFTER_DAYS)
    return held, purged & ~held


def _keys(*, origin_us: np.ndarray, valid_us: np.ndarray) -> np.ndarray:
    """Return one int64 key per (run origin, valid time) pair."""
    return ((origin_us - GRID_START_US) // STEP_US) * (1 << 22) + (
        valid_us - GRID_START_US
    ) // STEP_US


def swap_previous_weather(
    *, x: np.ndarray, names: list[str], origin_us: np.ndarray, valid_us: np.ndarray, use: np.ndarray
) -> np.ndarray:
    """Overwrite the weather columns of the rows in `use` with the previous run's, in place.

    A row takes the previous run's values at its valid time only if that run's row is in `x`.

    Returns:
        A boolean mask of the rows that were swapped.
    """
    n = len(valid_us)
    keys = _keys(origin_us=origin_us, valid_us=valid_us)
    order = np.argsort(keys)
    wanted = _keys(origin_us=origin_us - 24 * US_PER_HOUR, valid_us=valid_us)
    position = np.clip(np.searchsorted(keys[order], wanted), 0, n - 1)
    source = order[position]
    found = use & (keys[source] == wanted)
    weather = [i for i, c in enumerate(names) if c.startswith(("mean_", "std_"))]
    x[np.ix_(found, weather)] = x[np.ix_(source[found], weather)]
    return found


def augmented_primary(  # noqa: PLR0915
    *,
    z: dict[str, np.ndarray],
    keep: np.ndarray,
    power: np.ndarray,
    arm: TrainArm,
    names: list[str],
    rng: np.random.Generator,
) -> dict[str, np.ndarray] | None:
    """Return one Primary's raw (unstandardised) training rows for an arm, or `None` if empty.

    Args:
        z: The Primary's experiment 397 raw summary arrays (`x`, `valid`, `origin`).
        keep: The fold's row mask (valid time window and the 3.5-hour refit gap).
        power: The Primary's cleaned power over capacity on the grid.
        arm: The arm.
        names: The input column names of the arm.
        rng: The generator that draws the origin offset, delay, run, and lag dropout.

    Returns:
        `x`, `y`, `valid`, `origin` (the experiment 397 origin, in microseconds), `lead`, or `None`.
    """
    valid_us = z["valid"][keep]
    base_origin_us = z["origin"][keep]
    x = z["x"][keep]
    if x.shape[1] < len(names):
        x = np.concatenate([x, np.zeros((len(x), len(names) - x.shape[1]), np.float32)], axis=1)
    n = len(valid_us)
    run_us = base_origin_us - int(PUBLICATION_HOURS * US_PER_HOUR)
    offset = rng.choice(np.array(arm.offsets), size=n)
    delay = rng.choice(np.array(arm.delays), size=n)
    new_origin_us = run_us + (offset * US_PER_HOUR).astype(np.int64)
    usable = valid_us > new_origin_us
    weather_age = offset.copy()
    if arm.previous_run_probability > 0:
        use_previous = rng.random(n) < arm.previous_run_probability
        found = swap_previous_weather(
            x=x, names=names, origin_us=base_origin_us, valid_us=valid_us, use=use_previous
        )
        weather_age = np.where(found, weather_age + 24.0, weather_age)
    target_index = ((valid_us - GRID_START_US) // STEP_US).astype(np.int64)
    origin_index = ((new_origin_us - GRID_START_US) // STEP_US).astype(np.int64)
    effective_origin = origin_index - np.rint(2.0 * delay).astype(np.int64)
    if offset.min() < PUBLICATION_HOURS:
        msg = "a drawn origin precedes the run's publication"
        raise ValueError(msg)
    y = common.take(values=power, index=target_index)
    flagged_or_missing = int((usable & ~np.isfinite(y)).sum())
    usable &= np.isfinite(y)
    if not usable.any():
        return None
    sel = np.nonzero(usable)[0]
    x, y = x[sel], y[sel]
    valid_us, base_origin_us = valid_us[sel], base_origin_us[sel]
    target_index, effective_origin = target_index[sel], effective_origin[sel]
    delay, weather_age, origin_index = delay[sel], weather_age[sel], origin_index[sel]
    if not (target_index > origin_index).all():
        msg = "a valid time is not after its drawn origin"
        raise ValueError(msg)
    lags = common.origin_lags(power=power, target=target_index, origin=effective_origin)
    if arm.lag_drop_probability > 0:
        lags[rng.random(len(lags)) < arm.lag_drop_probability] = np.nan
    lead = (target_index - origin_index) / 2.0
    ages = prep.lag_ages_hours(lead_hours=lead + delay)
    missing = ~np.isfinite(lags)
    column = {name: j for j, name in enumerate(names)}
    for i, tag in enumerate(("origin", "day", "week")):
        x[:, column[f"lag_{tag}"]] = lags[:, i]
        x[:, column[f"lag_{tag}_missing"]] = missing[:, i]
        x[:, column[f"lag_{tag}_age"]] = ages[:, i] / prep.LEAD_SCALE_HOURS
    scaled = lead / prep.LEAD_SCALE_HOURS
    x[:, column["lead"]] = scaled
    x[:, column["lead_sqrt"]] = np.sqrt(np.clip(scaled, 0, None))
    if arm.age_inputs:
        x[:, column[AGE_COLUMNS[0]]] = weather_age / AGE_SCALE_HOURS
        x[:, column[AGE_COLUMNS[1]]] = (NEWEST_AGE_HOURS + delay) / AGE_SCALE_HOURS
    return {
        "x": x,
        "y": y.astype(np.float32),
        "valid": valid_us,
        "origin": base_origin_us,
        "lead": lead.astype(np.float32),
        "delay": delay.astype(np.float32),
        "dropped": np.array([flagged_or_missing]),
    }


def build_train(
    *, arm: TrainArm, refit: datetime, seed: int, clean: dict[str, np.ndarray]
) -> prep.FoldTrain:
    """Return the standardised training rows of one fold, arm, and seed.

    A fold sees a row only if its valid time is at least 3.5 hours before the refit and within the
    24 months before it. Because a row's valid time is after its drawn origin, no lag, weather run,
    or target comes from after the refit.
    """
    names, mask, layout = feature_names(age_inputs=arm.age_inputs)
    table = prep.primaries()
    rng = np.random.default_rng(seed + 7919 * int(refit.timestamp() // 86_400))
    parts = []
    dropped = 0
    for row in table.iter_rows(named=True):
        z = np.load(prep.RAW_DIR / "summary" / f"{row['site']}.npz")
        keep = (
            (z["valid"] <= prep._us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (z["origin"] <= prep._us(t=refit))
            & (z["valid"] > prep._us(t=refit - timedelta(days=prep.WINDOW_DAYS)))
        )
        if not keep.any():
            continue
        built = augmented_primary(
            z={k: z[k] for k in ("x", "valid", "origin")},
            keep=keep,
            power=clean[row["site"]],
            arm=arm,
            names=names,
            rng=rng,
        )
        if built is None:
            continue
        dropped += int(built.pop("dropped")[0])
        built["primary"] = np.full(len(built["y"]), row["primary"], dtype=np.int64)
        parts.append(built)
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    print(
        f"{arm.name} {refit:%Y-%m}: {dropped:,} half-hours dropped (flagged, start cut, unobserved)"
    )
    origin_day = ((cat["origin"] - prep._us(t=prep.FIRST_RUN)) // (86_400 * 1_000_000)).astype(
        np.int64
    )
    held, purged = purge_masks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = prep.Standardiser.fit(x=cat["x"][train], mask=mask)
    age_days = (prep._us(t=refit) - cat["valid"]) / (86_400 * 1_000_000)
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n_primaries = len(table)
    target_mean, target_std = np.zeros(n_primaries), np.ones(n_primaries)
    for p in range(n_primaries):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    kept = np.nonzero(train | held)[0]
    x_kept = np.empty((len(kept), cat["x"].shape[1]), dtype=np.float32)
    for start in range(0, len(kept), 1_000_000):
        chunk = kept[start : start + 1_000_000]
        x_kept[start : start + len(chunk)] = standardiser.apply(x=cat["x"][chunk])
    del cat["x"]
    return prep.FoldTrain(
        x=x_kept,
        y=cat["y"][kept],
        primary=cat["primary"][kept],
        weight=weight[kept].astype(np.float32),
        is_val=held[kept],
        lead=cat["lead"][kept],
        origin_day=origin_day[kept],
        standardiser=standardiser,
        target_mean=target_mean,
        target_std=target_std,
        layout=layout,
    )
