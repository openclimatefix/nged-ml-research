"""The arms of experiment 396 and the input columns each one adds to experiment 87's recipe.

An `Arm` says which columns it adds or drops, which XGBoost parameters it changes, how many rounds
it fits and scores, and how it forecasts. The column builders take a frame with `valid_time` (and,
for the winter columns, `power_fcst_init_time` and `ensemble_member`) and return it with the new
columns, so the same function builds the training rows and the prediction rows.

Nothing here fits a model.
"""

import itertools
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import cache, lru_cache
from pathlib import Path
from typing import Final

import holidays
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
ENS_STORE_FOLDERS: Final[dict[str, Path]] = {
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
"""Folders holding each Primary's `<site>.npz`: `origins`, `temperature`, `shortwave`."""
HALF_HOUR: Final[np.timedelta64] = np.timedelta64(30, "m")
STEPS_PER_RUN_DAY: Final[int] = 48
MAX_LEAD_STEPS: Final[int] = 672
"""336 hours: the longest lead any experiment 396 forecast needs."""
TRAILING_HOURS: Final[tuple[int, ...]] = (6, 24, 72, 168)
HEATING_BASE_C: Final[float] = 15.5
BAND_EDGES_HOURS: Final[tuple[float, ...]] = (0.0, 24.0, 72.0, 240.0, 336.0)
HOLIDAY_TYPES: Final[dict[str, int]] = {
    "bank_holiday": 1,
    "easter": 2,
    "christmas_period": 3,
    "new_year": 4,
    "other": 5,
}
"""Integer codes of `holiday_type`; a non-holiday day is missing (NaN), never 0."""
HOLIDAY_NAME_WORDS: Final[tuple[tuple[tuple[str, ...], str], ...]] = (
    (("good friday", "easter"), "easter"),
    (("christmas", "boxing"), "christmas_period"),
    (("new year",), "new_year"),
    (("bank holiday", "may day"), "bank_holiday"),
)
"""Words of a `holidays` name that give the type, in the order they are tried."""
BANK_HOLIDAY_CLIP_DAYS: Final[int] = 7
CHRISTMAS_PERIOD_START: Final[tuple[int, int]] = (12, 24)
CHRISTMAS_PERIOD_END: Final[tuple[int, int]] = (12, 31)


@dataclass(frozen=True)
class Arm:
    """One arm of the screen.

    Attributes:
        name: Folder name under `arm_preds/` and `boosters/`.
        tier: A, B, C, or D (the run order of the plan).
        add: Groups of columns the arm adds: `bank_holiday`, `distance`, `weekend`, `holiday_type`,
            `trailing_temperature`, `trailing_shortwave`, `day_of_year`,
            `summary_temperature_shortwave`, `summary7` (the mean and standard deviation of all
            seven weather variables replace them), `mean7` (the seven are replaced by their
            ensemble mean, in training and prediction rows), and `mean7_train` (replaced by the
            ensemble mean in training rows only).
        drop: Input columns the arm removes (the no-calendar control).
        params: XGBoost parameters that replace the control's.
        rounds: Boosting rounds fitted.
        scored_rounds: The numbers of rounds a fit is scored at (a prefix of the fitted trees); the
            arm's name gets `_r<rounds>` appended for each. Empty means score at `rounds` only.
        half_life_years: Recency half-life of the row weights, or `None` for experiment 87's 0.5.
        monotone: Columns constrained so the quantiles are non-increasing in them (-1) or
            non-decreasing (+1), as `(column, sign)`.
        seed: XGBoost seed.
        bands: Lead bands in hours, one booster each; empty means one booster for every lead.
        once: One forecast per run from the ensemble summary instead of the mixture of 51 members.
    """

    name: str
    tier: str
    add: tuple[str, ...] = ()
    drop: tuple[str, ...] = ()
    params: dict[str, float | int] = field(default_factory=dict)
    rounds: int = 500
    scored_rounds: tuple[int, ...] = ()
    half_life_years: float | None = None
    monotone: tuple[tuple[str, int], ...] = ()
    seed: int = 0
    bands: tuple[tuple[float, float], ...] = ()
    once: bool = False


def _bands(*, edges: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    return tuple(itertools.pairwise(edges))


ARMS: Final[dict[str, Arm]] = {
    arm.name: arm
    for arm in (
        Arm("C0", "A"),
        Arm("C1", "A", seed=1),
        Arm("E1_es2", "A", add=("summary_temperature_shortwave",), once=True),
        Arm("E2_es7", "B", add=("summary7",), once=True),
        Arm("E3_em_em", "B", add=("mean7",), once=True),
        Arm("E4_em_pool", "B", add=("mean7_train",)),
        Arm("H0_nocal", "A", drop=("hour_local", "day_of_week_local")),
        Arm("H1", "A", add=("bank_holiday", "distance")),
        Arm("H1d", "A", add=("distance",)),
        Arm("HW", "A", add=("weekend",)),
        Arm("HT", "A", add=("holiday_type",)),
        Arm("HWT", "A", add=("weekend", "holiday_type")),
        Arm("W1", "A", add=("trailing_temperature",)),
        Arm("W2", "A", monotone=(("temperature_2m", -1),)),
        Arm(
            "W3",
            "A",
            add=("trailing_temperature",),
            monotone=(("temperature_2m", -1), ("ttemp", -1), ("heating", 1)),
        ),
        Arm("W4", "A", add=("day_of_year",)),
        Arm("W7", "A2", add=("trailing_shortwave",)),
        Arm("P_rounds", "C", rounds=1000, scored_rounds=(200, 500, 1000)),
        Arm("P_eta003", "C", params={"eta": 0.03}),
        Arm("P_eta010", "C", params={"eta": 0.1}),
        Arm("P_depth4", "C", params={"max_depth": 4}),
        Arm("P_depth8", "C", params={"max_depth": 8}),
        Arm("P_mcw10", "C", params={"min_child_weight": 10}),
        Arm("P_mcw50", "C", params={"min_child_weight": 50}),
        Arm("P_sub06", "C", params={"subsample": 0.6}),
        Arm("P_sub10", "C", params={"subsample": 1.0}),
        Arm("P_lambda10", "C", params={"lambda": 10}),
        Arm("P_hl025", "C", half_life_years=0.25),
        Arm("P_hl1", "C", half_life_years=1.0),
        Arm("P_hl2", "C", half_life_years=2.0),
        Arm("B4", "D", bands=_bands(edges=BAND_EDGES_HOURS)),
        Arm("B2", "D", bands=_bands(edges=(0.0, 72.0, 336.0))),
    )
}
"""Every arm that needs only data already on disk. E2 to E5, W5, and W6 need the five-variable
ensemble extraction and are added with it."""


@cache
def bank_holiday_table() -> pl.DataFrame:
    """Return one row per calendar day 2018 to 2028: holiday flag, type code, and signed distance.

    `holiday_type` is missing (null) on a day that is neither a bank holiday nor in the Christmas
    period. `days_to_bank_holiday` is positive before the nearest bank holiday, negative after it,
    and clipped at +-7, matching the sign convention of experiment 87's `days_to_nearest_holiday`.
    The calendar is the `holidays` package's England calendar (MIT licence), the one experiment 87
    uses.
    """
    years = range(2018, 2029)
    calendar = holidays.country_holidays("GB", subdiv="ENG", years=years)
    days = pl.date_range(date(2018, 1, 1), date(2028, 12, 31), eager=True)
    names = {d: n.lower() for d, n in calendar.items()}

    def code(day: date) -> int | None:
        name = names.get(day)
        if name is None:
            in_period = CHRISTMAS_PERIOD_START <= (day.month, day.day) <= CHRISTMAS_PERIOD_END
            return HOLIDAY_TYPES["christmas_period"] if in_period else None
        for words, kind in HOLIDAY_NAME_WORDS:
            if any(w in name for w in words):
                return HOLIDAY_TYPES[kind]
        return HOLIDAY_TYPES["other"]

    holiday_dates = np.array(sorted(calendar), dtype="datetime64[D]")
    day_array = days.to_numpy()
    gaps = (holiday_dates[None, :] - day_array[:, None]).astype("timedelta64[D]").astype(int)
    nearest = gaps[np.arange(days.len()), np.abs(gaps).argmin(axis=1)]
    return pl.DataFrame(
        {
            "date": days,
            "is_bank_holiday": [float(d in names) for d in days],
            "day_before_bank_holiday": [float((d + timedelta(days=1)) in names) for d in days],
            "day_after_bank_holiday": [float((d - timedelta(days=1)) in names) for d in days],
            "days_to_bank_holiday": np.clip(
                nearest, -BANK_HOLIDAY_CLIP_DAYS, BANK_HOLIDAY_CLIP_DAYS
            ).astype(float),
            "holiday_type": [None if (c := code(d)) is None else float(c) for d in days],
        }
    )


def add_calendar(*, frame: pl.DataFrame, arm: Arm) -> pl.DataFrame:
    """Add the arm's calendar columns, computed from the row's local date only."""
    wanted = {"bank_holiday", "distance", "weekend", "holiday_type", "day_of_year"} & set(arm.add)
    if not wanted:
        return frame
    local = pl.col("valid_time").dt.offset_by("-30m").dt.convert_time_zone("Europe/London")
    out = frame.with_columns(_date=local.dt.date())
    table = bank_holiday_table().rename({"date": "_date"})
    out = out.join(table, on="_date", how="left", validate="m:1", maintain_order="left")
    new: list[pl.Expr] = []
    if "weekend" in arm.add:
        new.append((local.dt.weekday() >= 6).cast(pl.Float32).alias("is_weekend"))
    if "day_of_year" in arm.add:
        angle = 2 * np.pi * (local.dt.ordinal_day() - 1) / 365.25
        new += [
            angle.sin().cast(pl.Float32).alias("doy_sin"),
            angle.cos().cast(pl.Float32).alias("doy_cos"),
        ]
    return _finish_calendar(out=out.with_columns(*new), arm=arm)


def _finish_calendar(*, out: pl.DataFrame, arm: Arm) -> pl.DataFrame:
    """Keep only the calendar columns the arm adds, as float32 (a missing type stays missing)."""
    keep: list[str] = []
    if "bank_holiday" in arm.add:
        keep += ["is_bank_holiday", "day_before_bank_holiday", "day_after_bank_holiday"]
    if "distance" in arm.add:
        keep.append("days_to_bank_holiday")
    if "holiday_type" in arm.add:
        keep.append("holiday_type")
    table_columns = bank_holiday_table().columns
    drop = [c for c in out.columns if (c in table_columns or c == "_date") and c not in keep]
    out = out.drop(drop)
    return out.with_columns(pl.col(*keep).cast(pl.Float32).fill_null(float("nan"))) if keep else out


def calendar_columns(*, arm: Arm) -> list[str]:
    """Return the calendar columns the arm adds, in matrix order."""
    out: list[str] = []
    if "bank_holiday" in arm.add:
        out += ["is_bank_holiday", "day_before_bank_holiday", "day_after_bank_holiday"]
    if "distance" in arm.add:
        out.append("days_to_bank_holiday")
    if "weekend" in arm.add:
        out.append("is_weekend")
    if "holiday_type" in arm.add:
        out.append("holiday_type")
    if "day_of_year" in arm.add:
        out += ["doy_sin", "doy_cos"]
    return out


@lru_cache(maxsize=1)
def _ensemble_variable(*, site: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a Primary's run origins and one stored variable, shape (runs, members, 702)."""
    with np.load(ENS_STORE_FOLDERS[site] / f"{site}.npz") as stored:
        return stored["origins"], stored[key].astype(np.float32)


def _trailing_tables(
    *, site: str, key: str = "temperature", windows: tuple[int, ...] = TRAILING_HOURS
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Return the origins and, per window, the trailing mean of a variable at every run and step.

    For run r with origin t0 and step s (valid time t0 + (s + 1) half-hours), the window of w hours
    holds the 2w half-hours ending at the valid time. A half-hour after t0 is read from run r's
    own member trajectory. A half-hour at or before t0 is read from the same member's first day of
    the latest earlier run that covers it, which is the forecast the service holds at the origin;
    no analysis or ERA5 value enters. A window that reaches before the first run is NaN.

    Returns:
        The origins and `{hours: array (runs, members, MAX_LEAD_STEPS)}` in degrees Celsius.
    """
    origins, temperature = _ensemble_variable(site=site, key=key)
    if np.isnan(temperature).any():
        msg = f"{site}: the stored {key} holds missing values, which the cumulative sum spreads"
        raise ValueError(msg)
    runs, members, _ = temperature.shape
    gaps = np.diff(origins) // np.timedelta64(1, "h")
    if not (gaps == 24).all():
        msg = f"{site}: runs are not daily, so the past series cannot be stitched"
        raise ValueError(msg)
    # The past half of a window is the ensemble mean of the earlier runs' first day, the same for
    # every member, so the control member's training rows and the 51 members' prediction rows see
    # one past series. A member-by-member stitch would join unrelated perturbations at each seam.
    past_mean = temperature[:, :, :STEPS_PER_RUN_DAY].mean(axis=1).reshape(-1)
    past = np.broadcast_to(past_mean, (members, past_mean.size))
    longest = 2 * max(windows)
    tables = {h: np.full((runs, members, MAX_LEAD_STEPS), np.nan, np.float32) for h in windows}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for r in range(longest // STEPS_PER_RUN_DAY, runs):
            before = past[:, STEPS_PER_RUN_DAY * r - longest : STEPS_PER_RUN_DAY * r]
            combined = np.concatenate([before, temperature[r, :, :MAX_LEAD_STEPS]], axis=1)
            cumulative = np.concatenate(
                [np.zeros((members, 1)), np.cumsum(combined.astype(np.float64), axis=1)], axis=1
            )
            positions = longest + np.arange(MAX_LEAD_STEPS) + 1
            for hours in windows:
                width = 2 * hours
                tables[hours][r] = (
                    (cumulative[:, positions] - cumulative[:, positions - width]) / width
                ).astype(np.float32)
    return origins, tables


_TRAILING: dict[str, tuple[np.ndarray, dict[int, np.ndarray]]] = {}


def trailing_columns() -> list[str]:
    """Return the trailing-temperature column names in matrix order."""
    return [
        *(f"ttemp_{h}h" for h in TRAILING_HOURS),
        *(f"heating_{h}h" for h in TRAILING_HOURS),
    ]


def add_trailing_temperature(*, frame: pl.DataFrame, site: str) -> pl.DataFrame:
    """Add the trailing means of temperature and their degrees below 15.5 C (W1).

    Args:
        frame: Rows with `power_fcst_init_time`, `valid_time`, and `ensemble_member`.
        site: The Primary label, which picks the ensemble store.

    Returns:
        The frame with `trailing_columns()`; rows the store does not cover hold NaN.
    """
    if site not in _TRAILING:
        _TRAILING.clear()
        _TRAILING[site] = _trailing_tables(site=site)
    origins, tables = _TRAILING[site]
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    run = np.searchsorted(origins, origin)
    run_ok = (run < len(origins)) & (origins[np.minimum(run, len(origins) - 1)] == origin)
    step = ((valid - origin) // HALF_HOUR).astype(np.int64) - 1
    member = frame["ensemble_member"].to_numpy().astype(np.int64)
    ok = run_ok & (step >= 0) & (step < MAX_LEAD_STEPS)
    columns: dict[str, np.ndarray] = {}
    for hours in TRAILING_HOURS:
        values = np.full(frame.height, np.nan, np.float32)
        values[ok] = tables[hours][run[ok], member[ok], step[ok]]
        columns[f"ttemp_{hours}h"] = values
        columns[f"heating_{hours}h"] = np.maximum(HEATING_BASE_C - values, 0.0).astype(np.float32)
    return frame.with_columns(*[pl.Series(n, v) for n, v in columns.items()])


def band_of(*, lead_hours: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Return the mask of leads in `(low, high]` hours (the first band includes 0.5 hours)."""
    return (lead_hours > band[0]) & (lead_hours <= band[1])


SEVEN_VARIABLES: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)
"""The recipe's weather columns, named as in experiment 397's `ens_summary` files."""
MEAN7_COLUMNS: Final[tuple[str, ...]] = tuple(f"mean_{v}" for v in SEVEN_VARIABLES)
STD7_COLUMNS: Final[tuple[str, ...]] = tuple(f"std_{v}" for v in SEVEN_VARIABLES)
SUMMARY7_FOLDER: Final[Path] = (
    DATA / "00397-plain-and-attention-mixture-density-network-for-primary-demand"
    / "work" / "ens_summary"
)  # fmt: skip
"""Experiment 397's per-run ensemble mean and standard deviation of the seven variables."""

SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "ens_temp_mean",
    "ens_temp_std",
    "ens_sw_mean",
    "ens_sw_std",
)
SUMMARY_REPLACES: Final[tuple[str, ...]] = ("temperature_2m", "shortwave")
"""Arm E1 replaces these two control-member columns by the four summary columns."""


@lru_cache(maxsize=1)
def _summary_tables(*, site: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return a Primary's origins and the mean and standard deviation over the 51 members.

    The tables hold temperature and shortwave, shape (runs, 702), computed over the member axis
    ignoring missing values, as experiment 394's `store_of` does.
    """
    with np.load(ENS_STORE_FOLDERS[site] / f"{site}.npz") as stored:
        origins = stored["origins"]
        tables = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for name, key in (("temp", "temperature"), ("sw", "shortwave")):
                values = stored[key].astype(np.float64)
                tables[f"ens_{name}_mean"] = np.nanmean(values, axis=1).astype(np.float32)
                tables[f"ens_{name}_std"] = np.nanstd(values, axis=1).astype(np.float32)
    return origins, tables


def add_summary(*, frame: pl.DataFrame, site: str) -> pl.DataFrame:
    """Add the four ensemble summary columns of arm E1; rows the store does not cover hold NaN."""
    origins, tables = _summary_tables(site=site)
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    run = np.searchsorted(origins, origin)
    run_ok = (run < len(origins)) & (origins[np.minimum(run, len(origins) - 1)] == origin)
    step = ((valid - origin) // HALF_HOUR).astype(np.int64) - 1
    ok = run_ok & (step >= 0) & (step < tables["ens_temp_mean"].shape[1])
    columns = {}
    for name in SUMMARY_COLUMNS:
        values = np.full(frame.height, np.nan, np.float32)
        values[ok] = tables[name][run[ok], step[ok]]
        columns[name] = values
    return frame.with_columns(*[pl.Series(n, v) for n, v in columns.items()])


@lru_cache(maxsize=1)
def _summary7_table(*, time_series_id: int) -> pl.DataFrame:
    """Return one Primary's 51-member mean and standard deviation of the seven variables."""
    return (
        pl.scan_parquet(SUMMARY7_FOLDER / "init=*.parquet")
        .filter(pl.col("time_series_id") == time_series_id)
        .select("power_fcst_init_time", "valid_time", *MEAN7_COLUMNS, *STD7_COLUMNS)
        .collect()
    )


def add_summary7(*, frame: pl.DataFrame, time_series_id: int, replace: bool) -> pl.DataFrame:
    """Add experiment 397's ensemble summary of the seven variables to a frame.

    With `replace`, each weather column holds its ensemble mean and the summary columns are not
    kept. Otherwise the frame gains `mean_<v>` and `std_<v>` for each variable. Rows the summaries
    do not cover hold NaN.
    """
    joined = frame.join(
        _summary7_table(time_series_id=time_series_id),
        on=["power_fcst_init_time", "valid_time"],
        how="left",
        maintain_order="left",
    ).with_columns(pl.col(*MEAN7_COLUMNS, *STD7_COLUMNS).fill_null(float("nan")))
    if not replace:
        return joined
    return joined.with_columns(*[pl.col(f"mean_{v}").alias(v) for v in SEVEN_VARIABLES]).drop(
        *MEAN7_COLUMNS, *STD7_COLUMNS
    )


SHORTWAVE_HOURS: Final[tuple[int, ...]] = (3, 6, 24)
"""The trailing windows of arm W7's shortwave means; the 6-hour mean gives the earlier 3 hours."""
SHORTWAVE_COLUMNS: Final[tuple[str, ...]] = (
    *(f"sw_mean_{h}h" for h in SHORTWAVE_HOURS),
    "csi_change_3h",
)
MIN_CLEAR_SKY_W_M2: Final[float] = 20.0
"""A 3-hour window with a mean clear-sky shortwave below this (dawn, dusk, night) has no clear-sky
index, and the change in clear-sky index is then 0."""
_SHORTWAVE: dict[str, tuple[np.ndarray, dict[int, np.ndarray]]] = {}


def _clear_sky_grid(*, site: str, first: np.datetime64, size: int) -> np.ndarray:
    """Return Haurwitz clear-sky shortwave (W m-2) on `size` half-hours starting at `first`."""
    import e00087_common as common  # ty: ignore[unresolved-import]

    series = common.series_table().filter(pl.col("site") == site)
    stamps = pl.Series(first + np.arange(1, size + 1) * HALF_HOUR).dt.replace_time_zone("UTC")
    zenith_deg = common.derived_columns(
        valid_time=stamps,
        latitude=float(series["latitude"].item()),
        longitude=float(series["longitude"].item()),
    )["solar_zenith"].to_numpy()
    cosine = np.cos(np.radians(zenith_deg))
    with np.errstate(divide="ignore", over="ignore"):
        clear = 1098.0 * cosine * np.exp(-0.057 / np.where(cosine > 0, cosine, 1.0))
    return np.where(cosine > 0, clear, 0.0)


def add_trailing_shortwave(*, frame: pl.DataFrame, site: str) -> pl.DataFrame:
    """Add arm W7's trailing shortwave means and the 3-hour change in clear-sky index.

    The windows end at the valid time and are built as for W1, from the stored ensemble shortwave:
    a half-hour after the origin comes from the row's own member, an earlier half-hour from the
    ensemble mean of the earlier runs' first day. `csi_change_3h` is the clear-sky index (shortwave
    over Haurwitz clear-sky shortwave) of the 3 hours ending at the valid time, minus that of the 3
    hours before them; it is 0 where either window's mean clear-sky shortwave is below
    `MIN_CLEAR_SKY_W_M2`. Rows the store does not cover hold NaN.
    """
    if site not in _SHORTWAVE:
        _SHORTWAVE.clear()
        _SHORTWAVE[site] = _trailing_tables(site=site, key="shortwave", windows=SHORTWAVE_HOURS)
    origins, tables = _SHORTWAVE[site]
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    run = np.searchsorted(origins, origin)
    run_ok = (run < len(origins)) & (origins[np.minimum(run, len(origins) - 1)] == origin)
    step = ((valid - origin) // HALF_HOUR).astype(np.int64) - 1
    member = frame["ensemble_member"].to_numpy().astype(np.int64)
    ok = run_ok & (step >= 0) & (step < MAX_LEAD_STEPS)
    means = {}
    for hours in SHORTWAVE_HOURS:
        values = np.full(frame.height, np.nan, np.float32)
        values[ok] = tables[hours][run[ok], member[ok], step[ok]]
        means[hours] = values
    first = origins[0] - np.timedelta64(2, "D")
    last = origins[-1] + np.timedelta64(MAX_LEAD_STEPS * 30, "m")
    size = int((last - first) // HALF_HOUR) + 1
    clear = _clear_sky_grid(site=site, first=first, size=size)
    cumulative = np.concatenate([[0.0], np.cumsum(clear)])
    index = np.clip(((valid - first) // HALF_HOUR).astype(np.int64), 12, size - 1)
    now_clear = (cumulative[index] - cumulative[index - 6]) / 6.0
    earlier_clear = (cumulative[index - 6] - cumulative[index - 12]) / 6.0
    earlier_sw = 2.0 * means[6] - means[3]
    with np.errstate(divide="ignore", invalid="ignore"):
        change = means[3] / now_clear - earlier_sw / earlier_clear
    daylight = (now_clear >= MIN_CLEAR_SKY_W_M2) & (earlier_clear >= MIN_CLEAR_SKY_W_M2)
    column = np.where(daylight, change, 0.0)
    column = np.where(np.isnan(means[3]) | np.isnan(means[6]), np.nan, column).astype(np.float32)
    return frame.with_columns(
        *[pl.Series(f"sw_mean_{h}h", means[h]) for h in SHORTWAVE_HOURS],
        pl.Series("csi_change_3h", column),
    )
