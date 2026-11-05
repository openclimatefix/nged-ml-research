# ruff: noqa: E501
"""Experiment 571: known-answer synthetic worlds, injected into the cleaned power over capacity.

A world adds a known component to the target series of six Primaries over the whole history. The
injection is made at one place, experiment 422's `cleaned_arrays`, which every later reader of the
target (the fine-tuning rows, the pre-training rows, the test rows, and the scoring actuals) calls, so
training, pre-training, and scoring all see the same series. The component added is saved per
Primary and half-hour (`worlds/<world>/truth.npz`, `truth` is what was really added after the
non-negativity clip of the MVA Primaries; `steps` and `sens` are the unclipped parts).

Worlds (`WORLDS`): `W0` clean control; `Wa` abrupt level steps; `Wa-in` the same plus one extra step
inside each scored month (report-only); `Wb1` a growing temperature sensitivity; `Wb2` a changed
daily shape (an evening bump); `Wc` Wa and Wb1 together; `real` is the real series, untouched.
"""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00571-drift-offset-and-embedding-hybrid-known-answer-worlds"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
WORLDS: Final[tuple[str, ...]] = (
    "real",
    "W0",
    "Wa",
    "Wa-in",
    "Wb1",
    "Wb2",
    "Wc",
    "Wa-big",
    "Wb1-big",
    "Wc-big",
)
BIG: Final[float] = 3.0
"""Factor on every injected amplitude in a `-big` world (made readable after the first world proved unreadable)."""
INJECTED: Final[tuple[int, ...]] = (0, 2, 8, 5, 7, 6)
"""Primary indices injected: D2, D4, D12 (the three MVA Primaries with the fewest labelled events) and D9, D11, D10 (the three MW Primaries with the fewest). D8 and D14 are excluded, and D15 and D18 are report-only."""
MVA_INJECTED: Final[frozenset[int]] = frozenset({0, 2, 8})
"""The MVA-only Primaries among `INJECTED`; their reading is clipped at zero after the injection."""
FOLD_MONTHS: Final[tuple[str, ...]] = ("2025-12", "2026-06")
STEP_SIZE: Final[tuple[float, float]] = (0.03, 0.06)
"""Step size range, as a fraction of effective capacity (the cleaned power is already over capacity)."""
MIN_GAP_DAYS: Final[int] = 60
HDD_BASE_C: Final[float] = 15.0
SENS_START: Final[datetime] = datetime(2023, 1, 1, tzinfo=UTC)
SENS_FULL: Final[datetime] = datetime(2025, 6, 1, tzinfo=UTC)
SENS_MAX: Final[float] = 0.004
"""Extra load per heating degree, as a fraction of capacity, from `SENS_FULL` on."""
BUMP_MAX: Final[float] = 0.03
STEP_PER_DAY: Final[int] = 48


def world_dir(*, world: str) -> Path:
    """Return the world's private folder."""
    return WORK / "worlds" / world


def _idx(*, t: datetime) -> int:
    """Return the grid index of a time."""
    return int((t.timestamp() * 1e6 - prep422.GRID_START_US) // prep422.STEP_US)


def month_bounds(*, month: str) -> tuple[datetime, datetime]:
    """Return the first instant of a month and of the next month, UTC."""
    year, mon = (int(p) for p in month.split("-"))
    start = datetime(year, mon, 1, tzinfo=UTC)
    stop = datetime(year + (mon == 12), mon % 12 + 1, 1, tzinfo=UTC)
    return start, stop


def temperature_grid() -> np.ndarray:
    """Return each Primary's ERA5 mean 2 m temperature (degrees Celsius) on the half-hour grid.

    Read from the raw pre-training matrices of experiment 418 (the column `mean_temperature_2m` at the
    row of shortest lead of each valid time). A gap (a Primary has pre-training rows only where its
    power exists) is filled from the Primary whose temperature differs least on the overlap, then
    interpolated. This is the realised weather
    at the target time, used only to define the injected component (the world's truth), never as a
    model input.
    """
    n = data.grid_length()
    keep, columns, _ = data.non_lag_layout()
    col = keep[columns.index("mean_temperature_2m")]
    out = np.full((len(prep.primaries()), n), np.nan, dtype=np.float32)
    for row in prep.primaries().iter_rows(named=True):
        z = np.load(prep418.RAW_DIR / f"{row['site']}.npz")
        idx = ((z["valid"] - prep422.GRID_START_US) // prep422.STEP_US).astype(np.int64)
        ok = (idx >= 0) & (idx < n)
        order = np.argsort(z["lead"][ok], kind="stable")[::-1]  # shortest lead written last
        series = np.full(n, np.nan, dtype=np.float32)
        series[idx[ok][order]] = z["x"][ok][order, col]
        out[row["primary"]] = series
    for q in range(len(out)):
        missing = ~np.isfinite(out[q])
        if not missing.any():
            continue
        both = [(float(np.nanmean((out[q] - out[d]) ** 2)), d) for d in range(len(out)) if d != q]
        for _, donor in sorted(d for d in both if np.isfinite(d[0])):
            fill = missing & np.isfinite(out[donor])
            out[q, fill] = out[donor, fill]
            missing &= ~fill
            if not missing.any():
                break
        known = np.flatnonzero(np.isfinite(out[q]))
        out[q] = np.interp(np.arange(n), known, out[q, known])
    return out


def draw_steps(*, primary: int, world: str) -> list[tuple[datetime, float]]:
    """Return the dates and signed sizes of a Primary's abrupt steps (fraction of capacity).

    Per injected Primary there are three steps: one 20 to 40 days before the 2025-12 fold, one 20 to
    40 days before the 2026-06 fold (each is also inside the 15 months before its fold, or before the
    later fold), and one anywhere from 2022-01 to 2025-03. They are at least `MIN_GAP_DAYS` days
    apart, and none is inside a scored month. `Wa-in` adds one inside each scored month.
    """
    rng = np.random.default_rng([571, primary])
    dates = []
    for month in FOLD_MONTHS:
        start, _ = month_bounds(month=month)
        dates.append(start - timedelta(days=int(rng.integers(20, 41))))
    while True:
        early = datetime(2022, 1, 1, tzinfo=UTC) + timedelta(days=int(rng.integers(0, 1180)))
        if all(abs((early - d).days) >= MIN_GAP_DAYS for d in dates):
            dates.append(early)
            break
    sizes = rng.uniform(*STEP_SIZE, size=len(dates)) * rng.choice([-1.0, 1.0], size=len(dates))
    steps = list(zip(dates, sizes.tolist(), strict=True))
    if world == "Wa-in":
        for month in FOLD_MONTHS:
            start, _ = month_bounds(month=month)
            steps.append(
                (
                    start + timedelta(days=int(rng.integers(10, 20))),
                    float(rng.uniform(*STEP_SIZE) * rng.choice([-1.0, 1.0])),
                )
            )
    return steps


def step_path(*, steps: list[tuple[datetime, float]], n: int) -> np.ndarray:
    """Return the cumulative step component on the grid."""
    path = np.zeros(n, dtype=np.float64)
    for when, size in steps:
        path[_idx(t=when) :] += size
    return path


def sensitivity_path(*, temperature: np.ndarray) -> np.ndarray:
    """Return s(t) times the heating degrees below 15 degrees Celsius, as a fraction of capacity."""
    n = len(temperature)
    t0, t1 = _idx(t=SENS_START), _idx(t=SENS_FULL)
    s = SENS_MAX * np.clip((np.arange(n) - t0) / (t1 - t0), 0.0, 1.0)
    return s * np.clip(HDD_BASE_C - temperature.astype(np.float64), 0.0, None)


def bump_path(*, n: int) -> np.ndarray:
    """Return a changed daily shape: a raised-cosine bump 17:00 to 21:00 local time, amplitude 0 to 3%.

    The amplitude rises linearly over the same span as the sensitivity ramp, then stays constant.
    """
    zone = ZoneInfo("Europe/London")
    start = datetime.fromtimestamp(prep422.GRID_START_US / 1e6, UTC)
    step = timedelta(microseconds=int(prep422.STEP_US))
    local = np.empty(n, dtype=np.float64)
    for i in range(n):
        t = (start + i * step).astimezone(zone)
        local[i] = t.hour + t.minute / 60.0
    shape = np.where(
        (local >= 17.0) & (local < 21.0),
        0.5 * (1.0 - np.cos(2 * np.pi * (local - 17.0) / 4.0)),
        0.0,
    )
    t0, t1 = _idx(t=SENS_START), _idx(t=SENS_FULL)
    return BUMP_MAX * np.clip((np.arange(n) - t0) / (t1 - t0), 0.0, 1.0) * shape


def build_truth(*, world: str, n: int, temperature: np.ndarray | None) -> dict[str, np.ndarray]:
    """Return the unclipped components `steps`, `sens`, and `bump`, each (injected Primaries, grid)."""
    scale = BIG if world.endswith("-big") else 1.0
    world = world.removesuffix("-big")
    k = len(INJECTED)
    parts = {name: np.zeros((k, n), dtype=np.float32) for name in ("steps", "sens", "bump")}
    for i, primary in enumerate(INJECTED):
        if world in {"Wa", "Wa-in", "Wc"}:
            parts["steps"][i] = step_path(steps=draw_steps(primary=primary, world=world), n=n)
        if world in {"Wb1", "Wc"}:
            assert temperature is not None
            parts["sens"][i] = sensitivity_path(temperature=temperature[primary])
        if world == "Wb2":
            parts["bump"][i] = bump_path(n=n)
    return {name: (scale * v).astype(np.float32) for name, v in parts.items()}


def install(world: str) -> Path | None:
    """Patch experiment 422's `cleaned_arrays` so the cleaned power carries the world's injection.

    Returns the truth file's path (None for `real`). Call once, before any row is built.
    """
    assert world in WORLDS, world
    if world == "real":
        return None
    assert not getattr(prep422.cleaned_arrays, "injected", False), "a world is already installed"
    original = prep422.cleaned_arrays
    sites = prep.primaries()["site"].to_list()
    out = world_dir(world=world) / "truth.npz"

    memo: dict[str, np.ndarray] = {}

    def injected_arrays() -> dict[str, np.ndarray]:
        clean = original()
        n = len(clean[sites[0]])
        if not memo:
            parts = build_truth(
                world=world, n=n, temperature=None if world == "W0" else temperature_grid()
            )
            memo.update(parts)
        parts = memo
        truth = np.zeros((len(INJECTED), n), dtype=np.float32)
        for i, primary in enumerate(INJECTED):
            site = sites[primary]
            added = (parts["steps"][i] + parts["sens"][i] + parts["bump"][i]).astype(np.float64)
            before = clean[site].astype(np.float64)
            after = before + added
            if primary in MVA_INJECTED:
                after = np.where(np.isfinite(after), np.maximum(after, 0.0), after)
            clean[site] = after.astype(clean[site].dtype)
            truth[i] = np.where(np.isfinite(before), after - before, added).astype(np.float32)
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp.npz")
            np.savez(tmp, truth=truth, primaries=np.array(INJECTED), **parts)  # ty: ignore[invalid-argument-type]
            tmp.rename(out)
        return clean

    injected_arrays.injected = True  # ty: ignore[unresolved-attribute]
    prep422.cleaned_arrays = injected_arrays
    return out


def truth_full(*, world: str, n_primaries: int = 16) -> np.ndarray:
    """Return the saved truth as (all Primaries, grid): zeros for the untouched Primaries."""
    z = np.load(world_dir(world=world) / "truth.npz")
    full = np.zeros((n_primaries, z["truth"].shape[1]), dtype=np.float32)
    full[list(INJECTED)] = z["truth"]
    return full


def world_from_env() -> str:
    """Return the world named by the `WORLD` environment variable (default: the real series)."""
    return os.environ.get("WORLD", "real")
