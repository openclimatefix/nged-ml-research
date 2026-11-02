"""Shared half-hour layer for experiment 508: windows by sun position, injection, day tables.

Reuses experiment 506's data layer (`e00506_data`, folder `ml-experiments/00506-*`), which sits on
experiment 87's dense half-hour grid. No model is fitted.
"""

from typing import Final

import e00506_data as data  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SLUG: Final[str] = "mva-shoulder-hours-and-d8-d22-dips-solar-or-load-shift"
WORK = data.ML_EXPERIMENTS_DATA_DIR / f"00508-{SLUG}" / "work"
LEVELS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0)
ALL_LEVELS: Final[tuple[float, ...]] = (0.0, *LEVELS)
WINDOWS: Final[tuple[str, ...]] = ("midday", "morning", "evening", "day")
SHOULDERS: Final[tuple[str, ...]] = ("morning", "evening")
ARMS: Final[tuple[str, ...]] = ("Q0", "PF097")
MIN_HALF_HOURS: Final[dict[str, int]] = {"midday": 6, "morning": 3, "evening": 3, "day": 40}
MIDDAY_HOURS: Final[tuple[float, float]] = (10.0, 15.0)
SHOULDER_UCS: Final[tuple[float, float]] = (0.05, 0.3)
K_FLOOR: Final[float] = 0.1
CLOUDY_K: Final[float] = 0.3
TAN_PHI_097: Final[float] = 0.2506
RATIO_MONTHS: Final[tuple[int, ...]] = (5, 6, 7)
MIN_DARK_HALF_HOURS: Final[int] = 10


def base_frame(*, site: str) -> pl.DataFrame:
    """Return the site's half-hours on the grid with date, hour, clear-sky shape, and power."""
    stamps = data.grid()
    p = data.power_points(site=site)
    u, k = data.cams(site=site)
    hour = data.local_hour_start(stamps=stamps)
    local = stamps.dt.offset_by("-30m").dt.convert_time_zone("Europe/London")
    month = local.dt.month().to_numpy()
    weekend = (local.dt.weekday().to_numpy() >= 6).astype(int)
    k_clean = np.nan_to_num(k, nan=0.0)
    ucs = np.where(k_clean >= K_FLOOR, np.nan_to_num(u) / np.maximum(k_clean, K_FLOOR), np.nan)
    dark = (np.nan_to_num(u, nan=1.0) < 0.01) & np.isfinite(p)
    key = month * 2 + weekend
    level_by_key = {
        int(c): float(np.median(p[dark & (key == c)]))
        for c in np.unique(key)
        if (dark & (key == c)).sum() > 50
    }
    demand = np.array([level_by_key.get(int(c), np.nan) for c in key])
    return pl.DataFrame(
        {
            "date": (
                local.dt.year().to_numpy().astype(np.int64) * 10000
                + month.astype(np.int64) * 100
                + local.dt.day().to_numpy().astype(np.int64)
            ),
            "year": local.dt.year().to_numpy(),
            "month": month,
            "hour": hour,
            "ucs": ucs,
            "u": np.nan_to_num(u),
            "k": k_clean,
            "p": p,
            "dark": dark,
            "demand": demand,
            "before_cut": (stamps < data.LEVEL_WINDOW_END).to_numpy(),
        }
    )


def window_masks(*, frame: pl.DataFrame, finite: np.ndarray) -> dict[str, np.ndarray]:
    """Return the boolean half-hour mask of each window (and the cloudy shoulder windows)."""
    hour, ucs, k = frame["hour"].to_numpy(), frame["ucs"].to_numpy(), frame["k"].to_numpy()
    sunny = k >= data.SUNNY_K
    shoulder = (ucs >= SHOULDER_UCS[0]) & (ucs <= SHOULDER_UCS[1])
    cloudy = (k >= K_FLOOR) & (k < CLOUDY_K)
    mid = (hour >= MIDDAY_HOURS[0]) & (hour < MIDDAY_HOURS[1])
    return {
        "midday": mid & sunny & finite,
        "midday_cloudy": mid & cloudy & finite,
        "morning": shoulder & (hour < 12) & sunny & finite,
        "evening": shoulder & (hour >= 12) & sunny & finite,
        "day": finite,
        "morning_cloudy": shoulder & (hour < 12) & cloudy & finite,
        "evening_cloudy": shoulder & (hour >= 12) & cloudy & finite,
    }


def signed_power(*, site: str, frame: pl.DataFrame, level: float) -> np.ndarray:
    """Return the signed power with a solar farm injected (signed = p - level x median|p| x farm).

    The median is taken over the sunny midday half-hours of April to September before the level
    window ends, as in experiment 506. The eight apparent-power Primaries are never injected.
    """
    p = frame["p"].to_numpy()
    if level == 0 or site not in data.FARM_OF_SITE:
        return p.copy()
    masks = window_masks(frame=frame, finite=np.isfinite(p))
    farm = data.farm_norm(farm=data.FARM_OF_SITE[site])
    month = frame["month"].to_numpy()
    before = frame["before_cut"].to_numpy() & masks["midday"] & np.isin(month, range(4, 10))
    return p - level * float(np.median(np.abs(p[before]))) * farm


def readings(*, site: str, frame: pl.DataFrame, signed: np.ndarray) -> dict[str, np.ndarray]:
    """Return the two arms' readings (floor 0 and power factor 0.97).

    The eight apparent-power Primaries report raw `|p|` in both arms.
    """
    if site in data.MVA_ONLY:
        raw = np.abs(signed)
        return {"Q0": raw, "PF097": raw}
    demand = frame["demand"].to_numpy()
    return {
        "Q0": np.abs(signed),
        "PF097": np.sqrt(signed**2 + (TAN_PHI_097 * np.abs(demand)) ** 2),
    }


def day_tables(*, site: str, level: float) -> tuple[pl.DataFrame, dict[str, float]]:
    """Return one row per (UK day, window) and the observed export-to-import ratios.

    Columns: window, date, year, month, n, signed_min, signed_max, Q0, PF097 (the day's minimum
    reading in the window), dark_min (the day's minimum floor-0 reading over dark half-hours).
    """
    frame = base_frame(site=site)
    signed = signed_power(site=site, frame=frame, level=level)
    finite = np.isfinite(signed)
    read = readings(site=site, frame=frame, signed=signed)
    masks = window_masks(frame=frame, finite=finite)
    month = frame["month"].to_numpy()
    dark = frame["dark"].to_numpy() & finite
    tables = []
    for name, mask in masks.items():
        window = name.removesuffix("_cloudy")
        sky = "cloudy" if name.endswith("_cloudy") else "sunny"
        if name == "day":
            sky = "any"
        sub = pl.DataFrame(
            {
                "date": frame["date"].to_numpy()[mask],
                "year": frame["year"].to_numpy()[mask],
                "month": month[mask],
                "signed": signed[mask],
                "Q0": read["Q0"][mask],
                "PF097": read["PF097"][mask],
            }
        )
        tables.append(
            sub.group_by("date")
            .agg(
                pl.col("year").first(),
                pl.col("month").first(),
                n=pl.len(),
                signed_min=pl.col("signed").min(),
                signed_max=pl.col("signed").max(),
                Q0=pl.col("Q0").min(),
                PF097=pl.col("PF097").min(),
            )
            .filter(pl.col("n") >= MIN_HALF_HOURS[window])
            .with_columns(window=pl.lit(window), sky=pl.lit(sky))
        )
    dark_min = (
        pl.DataFrame({"date": frame["date"].to_numpy()[dark], "dark_min": np.abs(signed[dark])})
        .group_by("date")
        .agg(dark_min=pl.col("dark_min").min(), n_dark=pl.len())
        .filter(pl.col("n_dark") >= MIN_DARK_HALF_HOURS)
        .drop("n_dark")
    )
    table = pl.concat(tables).join(dark_min, on="date", how="left")
    summer = np.isin(month, RATIO_MONTHS)
    ratios = {}
    for arm in ARMS:
        mid = masks["midday"] & summer
        night = dark & summer
        ratios[arm] = float(np.median(read[arm][mid]) / np.median(read[arm][night]))
    return table.with_columns(site=pl.lit(site), level=pl.lit(level)), ratios
