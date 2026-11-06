# ruff: noqa: E501
"""Experiment 587: cases, worlds, and paths shared by the fit wrapper and the scorer.

A *case* is one load-shift: the verified D5/D16 transfer of experiment 580, or one design of its
injected bank. A *world* is a power stack (Primaries by half-hour grid, share of capacity): `obs` is
the cleaned observed power, `inj<spec_id>` is `obs` plus one injected design, and `cor` is `obs`
minus the verified transfer. For each case the event world `E` carries the event and the no-event
world `N` does not (verified: E = obs, N = cor; injected: E = inj, N = obs).
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00587-absorption-of-verified-and-injected-load-shifts-x5"
OUT = ML_EXPERIMENTS_DATA_DIR / NAME
D580 = (
    ML_EXPERIMENTS_DATA_DIR / "00580-verified-load-shift-event-set-and-label-audit" / "work"
)
SITE_MAP = (
    ML_EXPERIMENTS_DATA_DIR / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
)
SEED: Final[int] = 20261106
"""Fixed seed of the design draw."""
GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
PER_DAY: Final[int] = 48
VERIFIED_TRANSFER: Final[int] = 23
VERIFIED_MONTHS: Final[tuple[str, ...]] = ("2025-12", "2026-01")
SEEDS: Final[tuple[int, ...]] = (0, 1)
ARMS: Final[dict[str, tuple[str, str]]] = {
    "M0": ("565", "C-M0"),
    "D5": ("565", "D5"),
    "P5": ("574", "P5"),
}
"""Arm name to (harness experiment, that experiment's arm tag)."""


@dataclass(frozen=True)
class Event:
    """One load-shift at one Primary: signed depth as a share of capacity, onset and end grid indices."""

    case: str
    label: str
    primary: int
    time_series_id: int
    depth: float
    onset_idx: int
    end_idx: int

    @property
    def sign(self) -> int:
        """Return +1 where the Primary carries more power during the event."""
        return 1 if self.depth > 0 else -1


def site_map() -> pl.DataFrame:
    """Return the label, primary index, time series id, and capacity of the 16 Primaries."""
    return pl.read_csv(SITE_MAP)


def month_of_idx(idx: int) -> str:
    """Return the `YYYY-MM` of a grid index."""
    t = GRID_START + timedelta(minutes=30 * idx)
    return f"{t.year}-{t.month:02d}"


def idx_to_time(idx: np.ndarray | int) -> np.ndarray:
    """Return grid indices as numpy datetimes (UTC, microsecond)."""
    base = np.datetime64(GRID_START.replace(tzinfo=None), "us")
    return base + np.asarray(idx).astype("int64") * np.timedelta64(30 * 60 * 1_000_000, "us")


def designs() -> pl.DataFrame:
    """Return the 12 injected designs: the 9 cells depth x length (7, 15, 30 days) at start day 3, and 3 15-day designs at day 12."""
    bank = pl.read_parquet(D580 / "injected_event_bank.parquet").sort("spec_id")
    rng = np.random.default_rng(SEED)
    picks = []
    cells = [(d, n, 3) for d in (0.15, 0.30, 0.50) for n in (7, 15, 30)]
    cells += [(d, 15, 12) for d in (0.15, 0.30, 0.50)]
    for depth, length, start in cells:
        pool = bank.filter(
            (pl.col("depth_share_of_smaller_capacity") == depth)
            & (pl.col("length_days") == length)
            & (pl.col("start_day") == start)
            & pl.col("month").is_in(["2025-10", "2025-12", "2026-01", "2026-03", "2026-06"])
        )
        picks.append(pool.row(int(rng.integers(pool.height)), named=True))
    return pl.DataFrame(picks)


def injected_events(*, spec: dict) -> list[Event]:
    """Return the two per-Primary events (donor and recipient) of one design."""
    sm = site_map()
    out = []
    for lab, mw in ((spec["donor"], spec["donor_depth_mw"]), (spec["recipient"], spec["recipient_depth_mw"])):
        r = sm.filter(pl.col("label") == lab).row(0, named=True)
        out.append(
            Event(
                f"inj{spec['spec_id']}", lab, r["primary"], r["time_series_id"],
                mw / r["capacity_mw"], spec["onset_idx"], spec["end_idx"],
            )
        )  # fmt: skip
    return out


def verified_events() -> list[Event]:
    """Return the two per-Primary events of the verified D5/D16 transfer of experiment 580."""
    sm = site_map()
    q = (
        pl.read_parquet(D580 / "events_verified.parquet")
        .filter(pl.col("transfer_id") == VERIFIED_TRANSFER)
        .row(0, named=True)
    )
    out = []
    for side in ("a", "b"):
        r = sm.filter(pl.col("label") == q[f"primary_{side}"]).row(0, named=True)
        out.append(
            Event(
                "verified", r["label"], r["primary"], r["time_series_id"],
                q[f"depth_{side}_mw"] / r["capacity_mw"], q[f"onset_{side}_idx"], q[f"end_{side}_idx"],
            )
        )  # fmt: skip
    return out


def add_events(*, clean: np.ndarray, events: list[Event], factor: float) -> np.ndarray:
    """Return a copy of `clean` with `factor` x each event's depth added over its span (finite readings only)."""
    out = clean.copy()
    for e in events:
        seg = out[e.primary, e.onset_idx : e.end_idx]
        out[e.primary, e.onset_idx : e.end_idx] = np.where(np.isfinite(seg), seg + factor * e.depth, seg)
    return out


def case_events(*, case: str) -> list[Event]:
    """Return the events of a case name (`verified` or `inj<spec_id>`)."""
    if case == "verified":
        return verified_events()
    spec = designs().filter(pl.col("spec_id") == int(case.removeprefix("inj"))).row(0, named=True)
    return injected_events(spec=spec)


def case_months(*, case: str) -> list[str]:
    """Return the fold months a case is fitted in."""
    if case == "verified":
        return list(VERIFIED_MONTHS)
    return [designs().filter(pl.col("spec_id") == int(case.removeprefix("inj")))["month"][0]]


def case_worlds(*, case: str) -> tuple[str, str]:
    """Return the (event world, no-event world) names of a case."""
    return ("obs", "cor") if case == "verified" else (case, "obs")


def world_clean(*, world: str, obs: np.ndarray) -> np.ndarray:
    """Return the power stack of a world."""
    if world == "obs":
        return obs
    if world == "cor":
        return add_events(clean=obs, events=verified_events(), factor=-1.0)
    return add_events(clean=obs, events=case_events(case=world), factor=1.0)


def world_dir(*, arm: str, world: str, on: str | None = None):
    """Return the folder of an arm's fits in a world; `on` names the world whose inputs score the fits."""
    return OUT / "work" / arm / (world if on is None else f"{world}_on_{on}")
