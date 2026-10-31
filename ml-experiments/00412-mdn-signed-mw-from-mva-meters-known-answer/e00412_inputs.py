# ruff: noqa: E501
"""CAMS solar inputs, the injected farm, and the fold rows for experiment 412.

- **CAMS arrays.** `cams_arrays` gives each Primary's unit PV shape `u` (the fleet plant's AC power
  from CAMS hourly irradiance, as `studies.dp_features.unit_pv_shape` computes it) and clear-sky
  index `k` on experiment 87's dense half-hour grid. `u` is divided by one constant, the 99th
  percentile of the daylight values pooled over the 16 Primaries, so a capacity of C points means
  that the shape reaches C points at its 99th percentile.
- **Folded Primaries.** The MVA meter reads `sqrt(P^2 + Q^2)`. `site_arrays` returns, for one
  Primary, the reading matrix and target as the meter would give them (`fold=True`) or as the signed
  meter gives them (`fold=False`), with an optional metered farm's output subtracted from the signed
  power first (the injection of experiments 129 and 405, at the level of experiment 408).
- **The fold.** `build_fold` returns experiment 397's training rows for one refit with the
  irradiance and solar-geometry columns removed (the C arms of experiment 405), the D15 and D18
  short-history cut, and per-row CAMS `u`.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_prep as prep405  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.baselines import haurwitz_w_m2
from studies.dp_features import FLEET
from studies.pv_physics import half_hour_sun_and_sky, plant_power_mw
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

SLUG: Final[str] = "mdn-signed-mw-from-mva-meters-known-answer"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00412-{SLUG}" / "work"
CAMS_PATH: Final[Path] = (
    REPO_DATA_DIR / "studies/downloads/reanalysis/CAMS/demand_points/cams_demand_points.parquet"
)
MVA_ONLY: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
SIGNED_SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
SIGNED_REPORT_ONLY: Final[tuple[str, ...]] = ("D15", "D18")
ROTATIONS: Final[dict[str, tuple[str, str]]] = {
    "r1": ("D5", "D10"),
    "r2": ("D9", "D14"),
    "r3": ("D11", "D16"),
}
"""The pairs of signed-MW Primaries whose sign is hidden, one pair per rotation."""
FARM_OF_SITE: Final[dict[str, str]] = {
    "D5": "A",
    "D9": "B",
    "D10": "C",
    "D11": "A",
    "D14": "B",
    "D16": "C",
}
"""Three farms rotate across the six signed-MW Primaries (the anonymised labels of experiment 39's
frame), as in experiment 408, so each farm is injected at two Primaries."""
SHORT_HISTORY_SITES: Final[tuple[str, ...]] = prep405.SHORT_HISTORY_SITES
K_CLIP: Final[float] = 1.5
DAYLIGHT_U: Final[float] = 0.01
U99_PATH: Final[Path] = WORK_DIR / "u99.json"
LEVEL_WINDOW_END: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
"""The scale of an injection is set on readings before this instant, as in experiment 408."""
SUNNY_K: Final[float] = 0.7
MIDDAY_HOURS: Final[tuple[int, int]] = (10, 14)
SUMMER_MONTHS: Final[tuple[int, int]] = (4, 9)
NAMES = prep405.NAMES
NAMES_MASK: Final[np.ndarray] = base.feature_names(scheme="summary")[1]
US_PER_DAY: Final[int] = 86_400 * 1_000_000
US_PER_HALF_HOUR: Final[int] = 1_800_000_000


# --- CAMS arrays --------------------------------------------------------------------------------


def _cams_raw(*, row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the unscaled unit shape and the clear-sky index on the dense grid."""
    hourly = (
        pl.read_parquet(CAMS_PATH, columns=["site", "time", "ghi_w_m2"])
        .filter(pl.col("site") == row["site"])
        .select("time", pl.col("ghi_w_m2").cast(pl.Float64))
        .drop_nulls()
        .sort("time")
    )
    sky = half_hour_sun_and_sky(hourly=hourly, latitude=row["latitude"], longitude=row["longitude"])
    clear = haurwitz_w_m2(apparent_zenith_deg=sky.zenith_deg)
    index = np.divide(sky.ghi_w_m2, clear, out=np.zeros_like(clear), where=clear > 1.0)
    frame = pl.DataFrame(
        {
            "valid_time": pl.Series(sky.half_hour_end_time).dt.replace_time_zone("UTC"),
            "u": plant_power_mw(sky=sky, parameters=FLEET),
            "k": np.clip(index, 0.0, K_CLIP),
        }
    )
    joined = pl.DataFrame({"valid_time": common.grid_times()}).join(
        frame, on="valid_time", how="left", validate="1:1", maintain_order="left"
    )
    return joined["u"].to_numpy().astype(np.float64), joined["k"].to_numpy().astype(np.float64)


def write_cams() -> str:
    """Write each Primary's unscaled `u` and `k` on the dense grid, and the pooled `u99` constant."""
    series = common.series_table()
    out_dir = WORK_DIR / "cams"
    out_dir.mkdir(parents=True, exist_ok=True)
    daylight = []
    for row in base.primaries().iter_rows(named=True):
        site_row = next(r for r in series.iter_rows(named=True) if r["site"] == row["site"])
        u, k = _cams_raw(row=site_row)
        np.savez(out_dir / f"{row['site']}.npz", u=u, k=k)
        daylight.append(u[np.nan_to_num(u, nan=0.0) > DAYLIGHT_U])
    value = float(np.percentile(np.concatenate(daylight), 99))
    U99_PATH.write_text(f'{{"u99": {value}}}')
    return f"u99 = {value:.5f} over {sum(len(d) for d in daylight):,} daylight half-hours"


def u99() -> float:
    """Return the stored scaling constant (run the `cams` command first)."""
    import json

    return float(json.loads(U99_PATH.read_text())["u99"])


_GRID_US: np.ndarray | None = None


def grid_index(*, valid_us: np.ndarray) -> np.ndarray:
    """Return the dense-grid index of each valid time (microseconds since the epoch)."""
    global _GRID_US  # noqa: PLW0603
    if _GRID_US is None:
        _GRID_US = common.grid_times().dt.epoch("us").to_numpy()
    pos = np.searchsorted(_GRID_US, valid_us)
    if not np.array_equal(_GRID_US[np.clip(pos, 0, len(_GRID_US) - 1)], valid_us):
        msg = "a valid time is not on the dense half-hour grid"
        raise ValueError(msg)
    return pos


def cams_for_rows(*, site: str, valid_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return scaled `u` and `k` for the rows' valid times (NaN where CAMS is missing)."""
    z = np.load(WORK_DIR / "cams" / f"{site}.npz")
    pos = grid_index(valid_us=valid_us)
    return (z["u"][pos] / u99()).astype(np.float32), z["k"][pos].astype(np.float32)


def shuffled_u(*, site: str, valid_us: np.ndarray, seed: int = 412) -> np.ndarray:
    """Return `u` with its values permuted across days inside each (year, month, half-hour) group.

    The permutation is taken on the dense grid, so every row at one valid time gets one value, and
    the monthly daylight profile is kept while the day's sky is replaced by another day's.
    """
    z = np.load(WORK_DIR / "cams" / f"{site}.npz")
    u = z["u"] / u99()
    stamps = common.grid_times()
    start = stamps.dt.offset_by("-30m")
    key = (
        (start.dt.year().to_numpy().astype(np.int64) * 100 + start.dt.month().to_numpy()) * 48
        + start.dt.hour().to_numpy() * 2
        + (start.dt.minute().to_numpy() // 30)
    )
    rng = np.random.default_rng(seed)
    out = u.copy()
    order = np.argsort(key, kind="stable")
    bounds = np.flatnonzero(np.diff(key[order])) + 1
    for group in np.split(order, bounds):
        out[group] = u[rng.permutation(group)]
    return out[grid_index(valid_us=valid_us)].astype(np.float32)


def hour_and_month(*, valid_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the UTC hour and month of each half-hour's start (the valid time stamps its end)."""
    start = pl.Series(valid_us - US_PER_HALF_HOUR).cast(pl.Datetime("us"))
    return start.dt.hour().to_numpy(), start.dt.month().to_numpy()


def is_midday(*, valid_us: np.ndarray) -> np.ndarray:
    """Return whether each half-hour starts in a UTC summer-midday hour (April to September)."""
    hour, month = hour_and_month(valid_us=valid_us)
    return (
        (hour >= MIDDAY_HOURS[0])
        & (hour < MIDDAY_HOURS[1])
        & (month >= SUMMER_MONTHS[0])
        & (month <= SUMMER_MONTHS[1])
    )


# --- one Primary's reading matrix ---------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    """How the folded pair's meter reading is made from the signed power.

    Attributes:
        level: The injected farm's size as a multiple of the Primary's median sunny-midday reading
            before 2025-07-01 (0 means no injection, the placebo).
        q_points: The reactive floor `Q` of the reading `sqrt(P^2 + Q^2)`, in points of capacity.
    """

    level: float
    q_points: float

    @property
    def tag(self) -> str:
        """Return a file-name tag."""
        return f"L{self.level:g}_Q{self.q_points:g}"


def level_scale(*, site: str, y_signed: np.ndarray, valid_us: np.ndarray, level: float) -> float:
    """Return the injected farm's size as a fraction of effective capacity.

    It is `level` times the median absolute signed reading on sunny (CAMS `k` at least 0.7)
    summer-midday half-hours before 2025-07-01 (the window of experiment 408's injection).
    """
    _u, k = cams_for_rows(site=site, valid_us=valid_us)
    mask = (
        is_midday(valid_us=valid_us)
        & (np.nan_to_num(k) >= SUNNY_K)
        & (valid_us < int(LEVEL_WINDOW_END.timestamp() * 1_000_000))
        & np.isfinite(y_signed)
    )
    if mask.sum() < 200:
        msg = f"{site}: only {int(mask.sum())} sunny summer-midday rows before 2025-07-01"
        raise ValueError(msg)
    return level * float(np.median(np.abs(y_signed[mask])))


@dataclass
class SiteArrays:
    """One Primary's rows as the meter gives them, and the signed truth."""

    x: np.ndarray
    y: np.ndarray
    y_truth: np.ndarray
    valid: np.ndarray
    origin: np.ndarray
    lead: np.ndarray


def site_arrays(*, site: str, fold: bool, setting: Setting | None) -> SiteArrays:
    """Return a Primary's raw rows with the reading and the lags as the meter gives them.

    Args:
        site: The Primary.
        fold: Whether to fold the reading and the lags as an MVA meter does. A Primary that is
            already MVA-only is never folded here (its raw reading is the magnitude).
        setting: The injection level and reactive floor, applied only when `fold` is true and the
            Primary is signed. `None` applies no change.

    Returns:
        The rows. `y_truth` is the signed power with the farm subtracted (NaN where the injected
        farm has a gap in any rotation farm, so every setting trains on the same rows).
    """
    z = np.load(base.RAW_DIR / "summary" / f"{site}.npz")
    x = z["x"].copy()
    y = z["y"].astype(np.float64).copy()
    valid, origin = z["valid"], z["origin"]
    y_truth = y.copy()
    if fold and site in SIGNED_SIX and setting is not None:
        increment = (
            level_scale(site=site, y_signed=y, valid_us=valid, level=setting.level)
            if setting.level > 0
            else 0.0
        )
        q = setting.q_points / 100.0
        farm = prep405.farm_norm_series(farm=FARM_OF_SITE[site])
        others = [prep405.farm_norm_series(farm=f) for f in sorted(set(FARM_OF_SITE.values()))]

        def injected(*, times: np.ndarray) -> np.ndarray:
            values = prep405.farm_at(times_us=times, series=farm, placebo=False)
            gap = np.zeros(len(times), dtype=bool)
            for other in others:
                gap |= np.isnan(prep405.farm_at(times_us=times, series=other, placebo=False))
            return np.where(gap, np.nan, values)

        unique, inverse = np.unique(valid, return_inverse=True)
        y_truth = y - increment * injected(times=unique)[inverse]
        y = np.sqrt(y_truth**2 + q**2)
        for tag in ("origin", "day", "week"):
            raw = x[:, prep405.col(name=f"lag_{tag}")].astype(np.float64)
            age = (
                x[:, prep405.col(name=f"lag_{tag}_age")].astype(np.float64) * base.LEAD_SCALE_HOURS
            )
            lag_time = valid - np.rint(age * 2.0).astype(np.int64) * (prep405.US_PER_HOUR // 2)
            uniq, inv = np.unique(lag_time, return_inverse=True)
            signed_lag = raw - increment * injected(times=uniq)[inv]
            x[:, prep405.col(name=f"lag_{tag}")] = np.sqrt(signed_lag**2 + q**2)
            gone = np.isnan(x[:, prep405.col(name=f"lag_{tag}")])
            x[gone, prep405.col(name=f"lag_{tag}_missing")] = 1.0
    prep405.mask_short_history_lags(x=x, valid_us=valid, site=site)
    return SiteArrays(
        x=x,
        y=y.astype(np.float32),
        y_truth=y_truth.astype(np.float32),
        valid=valid,
        origin=origin,
        lead=z["lead"],
    )


# --- the fold -----------------------------------------------------------------------------------


@dataclass
class Fold:
    """The standardised training rows of one refit, with CAMS `u` and the Primaries' fold flags.

    `u` is the network's input to the solar term (permuted in the shuffle arm) and `gate_u` is the
    real scaled unit shape, which sets the folded likelihood's gate in every arm.
    """

    x: np.ndarray
    y: np.ndarray
    primary: np.ndarray
    weight: np.ndarray
    is_val: np.ndarray
    u: np.ndarray
    gate_u: np.ndarray
    standardiser: base.Standardiser
    keep: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    folded: np.ndarray
    setting: Setting
    pair: tuple[str, ...]
    n_dropped: dict[str, int]


def _us(*, t: datetime) -> int:
    return int(t.timestamp() * 1_000_000)


def folded_flags(*, pair: tuple[str, ...]) -> np.ndarray:
    """Return, per Primary index, whether its target is folded (the MVA-only eight plus the pair)."""
    table = base.primaries()
    return np.array([s in MVA_ONLY or s in pair for s in table["site"]], dtype=bool)


def build_fold(
    *, refit: datetime, pair: tuple[str, ...], setting: Setting, shuffle_u: bool = False
) -> Fold:
    """Return a refit's training rows: experiment 397's rule, the short-history cut, the fold.

    The Primaries in `pair` are signed-MW Primaries whose meter is replaced by an MVA reading with
    the injected farm; every other signed-MW Primary keeps its signed target. Half-hours where the
    injection union has a gap leave training (counted in `n_dropped`).
    """
    table = base.primaries()
    parts = []
    dropped: dict[str, int] = {}
    for row in table.iter_rows(named=True):
        site = row["site"]
        arr = site_arrays(site=site, fold=site in pair, setting=setting)
        keep = (
            (arr.valid <= _us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (arr.origin <= _us(t=refit))
            & (arr.valid > _us(t=refit - timedelta(days=base.WINDOW_DAYS)))
        )
        if site in SHORT_HISTORY_SITES:
            keep &= arr.valid >= _us(t=prep405.SHORT_HISTORY_CUT)
        usable = keep & np.isfinite(arr.y)
        dropped[site] = int(keep.sum() - usable.sum())
        if not usable.any():
            continue
        u_real = cams_for_rows(site=site, valid_us=arr.valid[usable])[0]
        u = shuffled_u(site=site, valid_us=arr.valid[usable]) if shuffle_u else u_real
        parts.append(
            {
                "x": arr.x[usable],
                "y": arr.y[usable],
                "origin": arr.origin[usable],
                "valid": arr.valid[usable],
                "primary": np.full(int(usable.sum()), row["primary"], dtype=np.int64),
                "u": np.nan_to_num(u, nan=0.0).astype(np.float32),
                "g": np.nan_to_num(u_real, nan=0.0).astype(np.float32),
            }
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    origin_day = ((cat["origin"] - _us(t=base.FIRST_RUN)) // US_PER_DAY).astype(np.int64)
    held, purged = base.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = base.Standardiser.fit(x=cat["x"][train], mask=NAMES_MASK)
    age_days = (_us(t=refit) - cat["valid"]) / US_PER_DAY
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n = len(table)
    target_mean, target_std = np.zeros(n), np.ones(n)
    for p in range(n):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep_rows = train | held
    cols = prep405.kept_columns(irradiance_free=True)
    return Fold(
        x=standardiser.apply(x=cat["x"][keep_rows])[:, cols],
        y=cat["y"][keep_rows],
        primary=cat["primary"][keep_rows],
        weight=weight[keep_rows].astype(np.float32),
        is_val=held[keep_rows],
        u=cat["u"][keep_rows],
        gate_u=cat["g"][keep_rows],
        standardiser=standardiser,
        keep=cols,
        target_mean=target_mean,
        target_std=target_std,
        folded=folded_flags(pair=pair),
        setting=setting,
        pair=pair,
        n_dropped=dropped,
    )
