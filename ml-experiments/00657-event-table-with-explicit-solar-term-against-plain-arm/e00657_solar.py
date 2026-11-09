"""Solar term of experiment 657: unit curve, capacity, pooled gain, and the three arm residuals.

    DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 \
        PYTHONPATH=ml-experiments/00653-edge-state-event-table-penalised-segmentation:. \
        uv run --no-sync python e00657_solar.py

Running this file prints the gain, its Huber fit statistics, the recorded-weather check, and the
sunny-midday normalised-MAE change (P5). The other scripts import `build`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Final

import numpy as np
import polars as pl
from e00653_common import DAY, E580, GRID_LENGTH, GRID_START, SCORED_FROM, SERIES_PATH, Panel  # ty: ignore[unresolved-import]
from e00653_core import Frozen  # ty: ignore[unresolved-import]
from e00653_inject import OFFSET_2019  # ty: ignore[unresolved-import]
from e00653_run import setup  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
"""The six Primaries metered in megawatts."""
ARM_NAMES: Final[tuple[str, ...]] = ("plain", "solar", "placebo")
FEATURES: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00006-stage1a-demand-input-feature-sweep/inputs/features.parquet"
)
CAPACITY: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00649-pv-capacity-table-season-year-from-stored-probes/work/capacity_table.parquet"
)
WORK: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00657-event-table-with-explicit-solar-term-against-plain-arm"
)
N_DAYS: Final[int] = GRID_LENGTH // DAY
GAIN_MAX: Final[float] = 1.5
HUBER_K: Final[float] = 1.345
SUNNY_SHARE: Final[float] = 0.7
SUNNY_CLEARNESS: Final[float] = 0.7
MIDDAY: Final[slice] = slice(20, 30)
"""Half-hours 10:00 to 14:30 UTC within a day (the half-hours ending 10:30 to 15:00)."""
FIT_YEARS_END: Final[int] = SCORED_FROM
"""Half-hour index of 2024-01-01: the gain is fitted before this."""


@dataclass
class SolarInputs:
    """Everything the later stages need from the solar term."""

    panel: Panel
    res: dict[str, np.ndarray]
    frozen: Frozen
    delta: dict[str, dict[str, np.ndarray]]
    """Per arm name, the half-hourly term added to the plain residual (zero outside the six)."""
    gain: float
    capacity: dict[str, float]
    sunny_day: dict[str, np.ndarray]
    """Per site, a boolean per day: 70% or more of 10:00-15:00 UTC half-hours are clear."""
    spans: list[tuple[str, int, int]]
    """Labelled and verified event spans (site, start, end) on the grid."""
    unit: dict[str, np.ndarray]
    stats: dict[str, float]


def label_spans() -> list[tuple[str, int, int]]:
    """Labelled and verified event spans (site, start, end), on the half-hour grid."""
    labelled = pl.read_parquet(E580 / "events_labelled_status.parquet")
    spans = [
        (r["label"], r["label_onset_idx"] - OFFSET_2019, r["label_end_idx"] - OFFSET_2019)
        for r in labelled.iter_rows(named=True)
    ]
    for r in pl.read_parquet(E580 / "events_verified.parquet").iter_rows(named=True):
        spans += [
            (r["primary_a"], r["onset_a_idx"] - OFFSET_2019, r["end_a_idx"] - OFFSET_2019),
            (r["primary_b"], r["onset_b_idx"] - OFFSET_2019, r["end_b_idx"] - OFFSET_2019),
        ]
    return spans


def busy_mask(*, spans: list[tuple[str, int, int]], site: str, pad_days: int) -> np.ndarray:
    """Half-hours within `pad_days` of a labelled or verified event at `site`."""
    busy = np.zeros(GRID_LENGTH, bool)
    for lab_site, a, b in spans:
        if lab_site == site:
            busy[max(0, a - pad_days * DAY) : max(0, b + pad_days * DAY)] = True
    return busy


def haurwitz(*, zenith_deg: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky global horizontal irradiance (W/m2); zero when the sun is down."""
    cosz = np.cos(np.radians(zenith_deg))
    safe = np.where(cosz > 0.02, cosz, np.nan)
    return np.nan_to_num(1098.0 * safe * np.exp(-0.059 / safe))


def load_irradiance(
    *, site_ids: dict[str, int]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Shortwave and solar zenith on the grid for each site (NaN where the feature row is absent).

    Recorded-weather check (printed by `check_recorded`): the `shortwave` column is ERA5
    surface shortwave at the valid time, one row per (series, valid_time), with no forecast-origin,
    lead, or ensemble column, so it is a recorded (analysis) product and not a forecast.
    """
    cols = ["time_series_id", "valid_time", "shortwave", "solar_zenith"]
    df = (
        pl.scan_parquet(FEATURES)
        .filter(pl.col("time_series_id").is_in(list(site_ids.values())))
        .select(cols)
        .collect()
    )
    idx = (
        (
            df["valid_time"].dt.replace_time_zone(None) - GRID_START.replace(tzinfo=None)
        ).dt.total_minutes()
        // 30
    ).to_numpy()
    sw: dict[str, np.ndarray] = {}
    zen: dict[str, np.ndarray] = {}
    for site, sid in site_ids.items():
        m = (df["time_series_id"] == sid).to_numpy() & (idx >= 0) & (idx < GRID_LENGTH)
        a = np.full(GRID_LENGTH, np.nan)
        z = np.full(GRID_LENGTH, np.nan)
        a[idx[m]] = df["shortwave"].to_numpy()[m]
        z[idx[m]] = df["solar_zenith"].to_numpy()[m]
        sw[site], zen[site] = a, z
    return sw, zen


def check_recorded(*, sw: dict[str, np.ndarray], zen: dict[str, np.ndarray]) -> None:
    """Print the evidence that `shortwave` is recorded weather, not a forecast."""
    names = pl.scan_parquet(FEATURES).collect_schema().names()
    origin_cols = [
        c
        for c in names
        if any(k in c.lower() for k in ("init", "lead", "issue", "ensemble", "member"))
    ]
    full = pl.scan_parquet(FEATURES).select(
        (pl.len() - pl.struct("time_series_id", "valid_time").n_unique()).alias("duplicate_rows")
    )
    stack = np.stack([sw[s] for s in sw])
    spread = np.nanmax(np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0))
    ci = sw[SIX[0]] / np.where(
        haurwitz(zenith_deg=zen[SIX[0]]) > 50, haurwitz(zenith_deg=zen[SIX[0]]), np.nan
    )
    night = np.nanmax(np.abs(sw[SIX[0]][zen[SIX[0]] > 95]))
    print("== Recorded-weather check on the `shortwave` column of experiment 6's feature table")
    print(f"forecast-origin, lead, or ensemble columns: {origin_cols or 'none'}")
    print(f"duplicate (series, valid_time) rows: {full.collect()['duplicate_rows'][0]}")
    print(
        f"largest spread of shortwave across the six Primaries at one valid time: {spread:.3g} W/m2"
    )
    print(f"largest |shortwave| with the sun 5 degrees below the horizon: {night:.3g} W/m2")
    print(
        f"clearness index 99.9th percentile {np.nanpercentile(ci, 99.9):.2f}, "
        f"median daylight {np.nanmedian(ci):.2f}"
    )
    print(f"shortwave maximum {np.nanmax(stack):.0f} W/m2")


def huber_slope(*, x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Slope through the origin of `y` on `x` by Huber iteratively reweighted least squares."""
    w = np.ones_like(x)
    beta = float((x * y).sum() / (x * x).sum())
    scale = 1.0
    for _ in range(50):
        r = y - beta * x
        scale = float(1.4826 * np.median(np.abs(r - np.median(r)))) or 1e-9
        z = np.abs(r) / scale
        w = np.where(z <= HUBER_K, 1.0, HUBER_K / np.maximum(z, 1e-9))
        new = float((w * x * y).sum() / (w * x * x).sum())
        if abs(new - beta) < 1e-9:
            beta = new
            break
        beta = new
    return beta, scale, len(x)


def sunny_days(*, sw: np.ndarray, zen: np.ndarray) -> np.ndarray:
    """Boolean per day: at least 70% of the 10:00-15:00 UTC half-hours have clearness >= 0.7."""
    clear = haurwitz(zenith_deg=zen)
    with np.errstate(invalid="ignore", divide="ignore"):
        ci = np.where(clear > 20, sw / clear, np.nan)
    mid = (ci.reshape(N_DAYS, DAY)[:, MIDDAY] >= SUNNY_CLEARNESS).mean(axis=1)
    return mid >= SUNNY_SHARE


def build() -> SolarInputs:
    """Build the unit curves, capacity, gain, and the plain, solar, and placebo residual terms."""
    panel, res, frozen = setup()
    series = pl.read_parquet(SERIES_PATH)
    ids = {r["site"]: int(r["time_series_id"]) for r in series.iter_rows(named=True)}
    site_ids = {s: ids[s] for s in SIX}
    sw, zen = load_irradiance(site_ids=site_ids)
    cap_table = pl.read_parquet(CAPACITY).filter(
        (pl.col("probe_year") == 2025) & (pl.col("cutoff") == "2026-06")
    )
    capacity = {r["site"]: float(r["cal_p50"]) for r in cap_table.iter_rows(named=True)}
    assert set(capacity) >= set(SIX), capacity
    spans = label_spans()
    unit: dict[str, np.ndarray] = {}
    sunny: dict[str, np.ndarray] = {}
    for s in SIX:
        day = np.isfinite(zen[s]) & (zen[s] < 90) & (sw[s] > 0)
        fit_years = np.arange(GRID_LENGTH) < FIT_YEARS_END
        p99 = float(np.percentile(sw[s][day & fit_years], 99))
        unit[s] = np.clip(sw[s] / p99, 0, None)
        sunny[s] = sunny_days(sw=sw[s], zen=zen[s])
    check_recorded(sw=sw, zen=zen)
    # Gain: Huber slope of -r on C*u over daylight April-September half-hours of 2022-2023,
    # outside the labelled and verified event spans, on the residual the decoder sees.
    month = np.array(
        [(GRID_START + timedelta(minutes=30 * int(i))).month for i in range(0, GRID_LENGTH, 48)]
    ).repeat(DAY)
    xs, ys = [], []
    for s in SIX:
        ok = (
            (np.arange(GRID_LENGTH) < FIT_YEARS_END)
            & (month >= 4)
            & (month <= 9)
            & (unit[s] > 0)
            & np.isfinite(unit[s])
            & np.isfinite(res[s])
            & ~busy_mask(spans=spans, site=s, pad_days=0)
        )
        xs.append(capacity[s] * unit[s][ok])
        ys.append(-res[s][ok])
    x, y = np.concatenate(xs), np.concatenate(ys)
    beta, scale, n = huber_slope(x=x, y=y)
    ols = float((x * y).sum() / (x * x).sum())
    gain = float(np.clip(beta, 0.0, GAIN_MAX))
    stats = {
        "g_huber": beta,
        "g": gain,
        "g_ols": ols,
        "n": float(n),
        "huber_scale_mw": scale,
        "mad_y_mw": float(1.4826 * np.median(np.abs(y - np.median(y)))),
    }
    solar_term = {s: np.nan_to_num(gain * capacity[s] * unit[s]) for s in SIX}
    placebo_term = {
        s: np.nan_to_num(gain * capacity[SIX[(i + 1) % len(SIX)]] * unit[s])
        for i, s in enumerate(SIX)
    }
    zero = np.zeros(GRID_LENGTH)
    delta = {
        "plain": dict.fromkeys(panel.sites, zero),
        "solar": {s: solar_term.get(s, zero) for s in panel.sites},
        "placebo": {s: placebo_term.get(s, zero) for s in panel.sites},
    }
    return SolarInputs(panel, res, frozen, delta, gain, capacity, sunny, spans, unit, stats)


def arm_residual(
    *, inp: SolarInputs, arm: str, base: dict[str, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    """The residual of an arm: `base` (default the plain residual) plus the arm's term."""
    src = inp.res if base is None else base
    return {s: src[s] + inp.delta[arm][s] for s in inp.panel.sites}


def midday_nmae(*, inp: SolarInputs, residual: dict[str, np.ndarray], lo: int, hi: int) -> float:
    """Normalised MAE (per cent of effective capacity) in sunny-midday half-hours at the six."""
    tot, cnt = 0.0, 0
    for s in SIX:
        day = np.repeat(inp.sunny_day[s], DAY)
        mid = np.tile(np.isin(np.arange(DAY), np.arange(DAY)[MIDDAY]), N_DAYS)
        m = (
            day
            & mid
            & (np.arange(GRID_LENGTH) >= lo)
            & (np.arange(GRID_LENGTH) < hi)
            & np.isfinite(residual[s])
        )
        tot += float(np.abs(residual[s][m]).sum()) / inp.panel.capacity[s]
        cnt += int(m.sum())
    return 100 * tot / cnt


def main() -> None:
    """Print the gain, the fit statistics, the sunny-day counts, and P5."""
    inp = build()
    print("\n== Gain (six Primaries pooled, April-September 2022-2023, daylight, outside events)")
    print({k: round(v, 4) for k, v in inp.stats.items()})
    print(
        "capacity C (MW, calibrated 50th percentile, probe 2025, cutoff 2026-06):",
        {k: round(inp.capacity[k], 2) for k in SIX},
    )
    gate = inp.gain < 0.05
    print(f"gain gate: g={inp.gain:.3f} {'UNDER 0.05' if gate else 'at or above 0.05'}")
    print("\n== Sunny-midday days (share of days)")
    for s in SIX:
        print(
            s,
            f"all {inp.sunny_day[s].mean():.3f}",
            f"scored years {inp.sunny_day[s][SCORED_FROM // DAY :].mean():.3f}",
        )
    print("\n== P5: normalised MAE (% of capacity) in sunny-midday half-hours, six Primaries")
    for name, lo, hi in (
        ("tuning 2022-2023", 0, SCORED_FROM),
        ("scored 2024-01..2026-06", SCORED_FROM, GRID_LENGTH),
    ):
        v = {
            a: midday_nmae(inp=inp, residual=arm_residual(inp=inp, arm=a), lo=lo, hi=hi)
            for a in ARM_NAMES
        }
        print(
            f"{name}: plain {v['plain']:.3f}  solar {v['solar']:.3f} "
            f"({100 * (v['solar'] / v['plain'] - 1):+.1f}%)  "
            f"placebo {v['placebo']:.3f} ({100 * (v['placebo'] / v['plain'] - 1):+.1f}%)"
        )


if __name__ == "__main__":
    main()
