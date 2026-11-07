# ruff: noqa: E501
"""Shared pieces of experiment 639: paths, the arms, and the capacity timeline of an explicit-PV network.

An explicit-PV network trains on the gross target `y + C(t) * inc * u(t)` and subtracts the same
term from its forecast. `C(t)` is a capacity timeline in megawatts: experiment 474's table (rung R4)
rescaled to the probe's level, plus a step increment measured by the probe, plus a trainable offset.
Experiment 634's rows, injector, frames, probe, and control networks, and experiment 638's step
networks, are reused unchanged.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00634_common as k634  # ty: ignore[unresolved-import]
import e00638_common as k638  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00639-pv-explicit-hindcast-probe-set-capacity-step-world"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
CUTOFFS: Final[tuple[str, ...]] = ("2025-07", "2026-03", "2026-06")
"""Pretraining cutoffs. Each is a test window start; the windows cover the sunniest half of the year."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
TEST_DAYS: Final = 60
DEV_END_US: Final[int] = k638.DEV_END_US
WORLDS: Final[tuple[str, ...]] = ("real", "step")
FREE_CONDITION: Final[dict[str, str]] = {"real": "control", "step": "step_spring23_mw4"}
"""The free network of each world: experiment 634's control, and experiment 638's `spring23` step."""
STEP_KEY: Final = "spring23"
STEP_MW: Final = 4.0
EXPLICIT_ARMS: Final[tuple[str, ...]] = ("cal", "raw", "r4", "oracle", "perm", "shuf")
ARMS_BY_WORLD: Final[dict[str, tuple[str, ...]]] = {
    "real": ("free", "cal", "raw", "r4", "perm", "shuf"),
    "step": ("free", "cal", "raw", "r4", "oracle", "perm", "shuf"),
}
PERM_SHIFT: Final = 2
"""The `perm` placebo gives each of the six Primaries the timeline of the Primary two places on."""
DELTA_PRIOR_MW: Final = 0.25
DELTA_PENALTY_ROWS: Final = 1000.0
"""The prior on a Primary's offset is a penalty `delta^2 / (2 * prior^2) / DELTA_PENALTY_ROWS` on the per-row mean loss."""
FACTOR_BETWEEN_FARMS_RELATIVE: Final = 0.04
SUNNY_KT: Final = 0.7
SUNNY_HOURS: Final[tuple[int, int]] = (10, 15)
"""Sunny half-hours: UTC hour from 10 to 15 inclusive, with a clear-sky index of at least 0.7."""
SHUFFLE_BLOCK_DAYS: Final = 21
SEASON_YEARS: Final[tuple[int, int]] = (2019, 2025)
K_CLIP: Final[tuple[float, float]] = (0.5, 1.5)
LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99)

P634: Final[Path] = k634.WORK
P638: Final[Path] = k638.WORK


def free_weights_path(*, cutoff: str, world: str, seed: int) -> Path:
    """Return the free network of a world: experiments 634 and 638 hold seeds 0 to 2, this one seeds 3 and 4."""
    condition = FREE_CONDITION[world]
    old = (
        k634.weights_path(cutoff=cutoff, condition=condition, seed=seed)
        if world == "real"
        else k638.weights_path(cutoff=cutoff, condition=condition, seed=seed)
    )
    return old if old.exists() else WORK / "pretrained" / cutoff / f"{world}_free_s{seed}.pt"


def explicit_weights_path(*, cutoff: str, world: str, arm: str, seed: int) -> Path:
    """Return where an explicit network (or its saved statistics, beside it as `.json`) is stored."""
    return WORK / "pretrained" / cutoff / f"{world}_{arm}_s{seed}.pt"


def prediction_path(*, cutoff: str, world: str, arm: str, seed: int) -> Path:
    """Return where an arm's test-window quantiles are stored."""
    return WORK / "predictions" / cutoff / f"{world}_{arm}_s{seed}.npz"


def test_rows_path(*, cutoff: str, world: str) -> Path:
    """Return where a cutoff's test-window row keys, actuals, and masks are stored."""
    return WORK / "predictions" / cutoff / f"{world}_rows.npz"


def probe_path(*, cutoff: str, world: str) -> Path:
    """Return where the probe reads of a world's free networks are stored."""
    return WORK / "probes" / cutoff / f"{world}_free.parquet"


def last_complete_season_year(*, cutoff: str) -> int:
    """Return the last season-year (July to June, named by July's year) that ended before the cutoff."""
    year, month = int(cutoff[:4]), int(cutoff[5:7])
    return min((year - 1 if month >= 7 else year - 2), SEASON_YEARS[1])


def season_year(*, t_us: np.ndarray, cutoff: str) -> np.ndarray:
    """Return each time's season-year (July to June, named by July's year), held at the last complete one before the cutoff."""
    stamp = np.asarray(t_us, dtype="datetime64[us]")
    year = stamp.astype("datetime64[Y]").astype(int) + 1970
    month = stamp.astype("datetime64[M]").astype(int) % 12 + 1
    sy = np.where(month >= 7, year, year - 1)
    return np.clip(sy, SEASON_YEARS[0], last_complete_season_year(cutoff=cutoff))


def r4_table() -> dict[tuple[str, int], float]:
    """Return experiment 474's R4 table: the median AC capacity (MW) by site and season-year."""
    path = next(iter((ML_EXPERIMENTS_DATA_DIR).glob("00474-*/work/nged_table.parquet")))
    table = pl.read_parquet(path)
    return {
        (r["site"], int(r["season_year"])): float(r["ac_p50"]) for r in table.iter_rows(named=True)
    }


@dataclass(frozen=True)
class SiteCapacity:
    """One Primary's capacity parameters for one cutoff and arm. All capacities are in megawatts."""

    k: float
    d: float
    rel_sd_k: float
    rel_sd_d: float


def _read_probes(*, cutoff: str, world: str) -> pl.DataFrame:
    """Return the `plain` probe reads of a world's free networks (all five seeds), one row per seed and site."""
    return pl.read_parquet(probe_path(cutoff=cutoff, world=world)).filter(
        pl.col("model") == "plain"
    )


def _factor(*, cutoff: str) -> tuple[float, float]:
    """Return the mean leave-one-farm-out recovery factor and its mean standard deviation at a cutoff (experiment 638)."""
    reads = pl.read_csv(P638 / "step_reads_shift0.csv").filter(pl.col("cutoff") == cutoff)
    return float(reads["factor"].to_numpy().mean()), float(reads["factor_sd"].to_numpy().mean())


def capacity_parameters(*, cutoff: str, arm: str) -> dict[str, SiteCapacity]:
    """Return each of the six Primaries' `k`, `d`, and relative spreads for an explicit arm.

    `k` rescales the R4 table to the probe's level: the median over seeds of the real-world free
    network's probed capacity, divided by the factor (arm `cal`), divided by R4 at the reference
    season-year, and clipped. `d` is the median over seeds of the step-world free network's probed
    capacity minus the real-world free network's (paired by seed and site), divided by the factor
    (arm `cal`), floored at zero. Arms `r4` and `oracle` use the table unchanged. Relative spreads add
    in quadrature the seed spread, the bootstrap spread, the factor spread, and the spread of the factor
    between farms.
    """
    table = r4_table()
    factor, factor_sd = _factor(cutoff=cutoff)
    real, step = (_read_probes(cutoff=cutoff, world=w) for w in WORLDS)
    out: dict[str, SiteCapacity] = {}
    for site in k634.SIX:
        if arm in {"r4", "oracle"}:
            d = STEP_MW if arm == "oracle" else 0.0
            out[site] = SiteCapacity(k=1.0, d=d, rel_sd_k=0.0, rel_sd_d=0.0)
            continue
        divisor = factor if arm in {"cal", "perm", "shuf"} else 1.0
        r_real = real.filter(pl.col("site") == site).sort("seed")
        r_step = step.filter(pl.col("site") == site).sort("seed")
        base = float(np.median(r_real["capacity_mw"].to_numpy()))
        diff = (r_step["capacity_mw"] - r_real["capacity_mw"]).to_numpy()
        reference = table[(site, last_complete_season_year(cutoff=cutoff))]
        k = float(np.clip(base / divisor / reference, *K_CLIP))
        d = max(float(np.median(diff)) / divisor, 0.0)
        boot_real = float(np.mean([np.std(b) for b in r_real["boot"].to_list()]))
        boot_step = float(np.mean([np.std(b) for b in r_step["boot"].to_list()]))
        seed_real = float(np.std(r_real["capacity_mw"].to_numpy(), ddof=1))
        seed_diff = float(np.std(diff, ddof=1))
        factor_rel = factor_sd / factor if divisor != 1.0 else 0.0
        between = FACTOR_BETWEEN_FARMS_RELATIVE if divisor != 1.0 else 0.0
        rel_k = float(
            np.sqrt((seed_real / base) ** 2 + (boot_real / base) ** 2 + factor_rel**2 + between**2)
        )
        denom = max(abs(float(np.median(diff))), 1e-6)
        rel_d = float(
            np.sqrt(
                (seed_diff / denom) ** 2
                + (np.hypot(boot_real, boot_step) / denom) ** 2
                + factor_rel**2
                + between**2
            )
        )
        out[site] = SiteCapacity(k=k, d=d, rel_sd_k=min(rel_k, 1.0), rel_sd_d=min(rel_d, 1.0))
    if arm == "perm":
        sites = list(k634.SIX)
        out = {s: out[sites[(i + PERM_SHIFT) % len(sites)]] for i, s in enumerate(sites)}
    return out


def parameters_json(*, cutoff: str, arm: str) -> dict:
    """Return `capacity_parameters` in a form that can be saved in a table."""
    return {s: vars(v) for s, v in capacity_parameters(cutoff=cutoff, arm=arm).items()}


def window_us(*, cutoff: str) -> tuple[int, int]:
    """Return the test window `[start, stop)` in microseconds: the cutoff to `TEST_DAYS` later, never past 2026-07-01."""
    start = datetime.fromisoformat(f"{cutoff}-01").replace(tzinfo=UTC)
    start_us = int(start.timestamp() * 1e6)
    stop_us = min(start_us + TEST_DAYS * k634.DAY_US, DEV_END_US)
    return start_us, stop_us


def save_json(*, path: Path, value: object) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=1))
    tmp.rename(path)
