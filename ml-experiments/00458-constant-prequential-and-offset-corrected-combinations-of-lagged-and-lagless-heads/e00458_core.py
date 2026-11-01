"""Experiment 458: combinations of the stored lagged and lagless heads (no fit, stored forecasts).

Each head is a cumulative distribution function that is linear between its 11 stored quantiles, as
in experiment 402's `e00402_mixture.py`. A mixture with lagged weight `w` has exact CRPS
`w^2 A + 2 w B + C` per row, where `A`, `B`, and `C` are the integrals of `(F1-F2)^2`,
`(F1-F2)(F2-H)`, and `(F2-H)^2` (`F1` lagged, `F2` lagless, `H` the step at the actual). The
integrals are computed once per row, so every weight rule is scored without rebuilding the mixture.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00402_mixture as mixture  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA_404: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00404-mdn-lagless-and-calibration-module-trust-weights" / "work"
)
DATA_445: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00445-flag-oracle-o-rule-ceiling-for-lag-event-detectors" / "work"
)
NRA_SHARES: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails"
    / "work"
    / "nra_correction_share.npz"
)
OUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00458-constant-prequential-and-offset-corrected-combinations-of-lagged-and-lagless-heads"
)
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
WARM_UP: Final[str] = "2025-10"
LEVELS: Final[np.ndarray] = mixture.LEVELS
LEVEL_NAMES: Final[tuple[str, ...]] = tuple(f"q{round(x * 100):02d}" for x in LEVELS)
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
TAILS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
BAND_EDGES: Final[np.ndarray] = np.array([12.0, 36.0, 72.0, 120.0])
BAND_NAMES: Final[tuple[str, ...]] = ("0-12h", "12-36h", "36-72h", "days4-5", "days6-10")
GROUPS: Final[tuple[str, ...]] = ("signed_six", "mva_eight", "report_only")
MVA_ONLY: Final[frozenset[str]] = frozenset({"D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"})
REPORT_ONLY: Final[frozenset[str]] = frozenset({"D15", "D18"})
STEPS_PER_DAY: Final[int] = 48
DELIVERY_STEPS: Final[int] = 12
"""An actual is usable at an origin when its target is at least 6 hours (12 steps) earlier."""
WINDOW_STEPS: Final[int] = 336
OFFSET_STEPS: Final[int] = 48
MIN_USABLE: Final[int] = 48
WEIGHT_FLOOR: Final[float] = 0.2
GRID_SIZE: Final[int] = 121248
CHUNK: Final[int] = 60_000
KEY: Final[list[str]] = ["primary", "origin", "target"]


def seed_of(*, month: str, cell_seed: int) -> int:
    """Return the fit seed whose files hold `month` (only 2026-06 has seeds 1 and 2)."""
    return cell_seed if month == "2026-06" else 0


def month_start_index(*, month: str) -> int:
    """Return the grid index of the first half-hour of `month`."""
    year, mon = (int(x) for x in month.split("-"))
    return (datetime(year, mon, 1, tzinfo=UTC) - common.GRID_START).days * STEPS_PER_DAY


def group_index() -> np.ndarray:
    """Return the group index (see `GROUPS`) of each Primary."""
    sites = common.series_table().filter(pl.col("substation_type") == "Primary")["site"].to_list()
    return np.array(
        [2 if s in REPORT_ONLY else 1 if s in MVA_ONLY else 0 for s in sites], dtype=np.int64
    )


def pinball_points(*, q: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return experiment 404's score in points of capacity: 200 x mass-weighted pinball loss."""
    diff = actual[:, None] - q
    loss = np.maximum(LEVELS * diff, (LEVELS - 1.0) * diff)
    return 200.0 * (loss * ATOM_MASSES).sum(axis=1)


def band_of(*, lead: np.ndarray) -> np.ndarray:
    """Return the table-P lead band index (0 to 4) of each lead in hours."""
    return np.searchsorted(BAND_EDGES, lead, side="right")


def _path(*, head: str, month: str, seed: int) -> Path:
    name = f"{month}_s{seed}_w-1.parquet"
    if head == "lagless":
        return DATA_445 / "predictions" / "O-lagless" / name
    return DATA_404 / "predictions" / head / name


@dataclass
class Cell:
    """One scored month and fit seed: keys, flags, actuals, and the four heads' quantiles."""

    month: str
    seed: int
    keys: pl.DataFrame
    actual: np.ndarray
    lead: np.ndarray
    flags: dict[str, np.ndarray]
    q: dict[str, np.ndarray]
    group: np.ndarray
    band: np.ndarray

    @property
    def primary(self) -> np.ndarray:
        """Return the Primary index of each row."""
        return self.keys["primary"].to_numpy().astype(np.int64)

    @property
    def origin(self) -> np.ndarray:
        """Return the origin grid index of each row."""
        return self.keys["origin"].to_numpy().astype(np.int64)

    @property
    def target(self) -> np.ndarray:
        """Return the target grid index of each row."""
        return self.keys["target"].to_numpy().astype(np.int64)

    def rowset(self, *, name: str) -> np.ndarray:
        """Return the mask of a row set: nolabel, harm, event, post, or all."""
        f = self.flags
        event = f["tgt_real"] | f["tgt_post"]
        if name == "event":
            return event
        if name == "post":
            return f["tgt_post"]
        if name == "harm":
            return f["win_real"] & ~event
        if name == "nolabel":
            return ~(f["win_real"] | f["lag_real"] | event)
        return np.ones(len(self.actual), dtype=bool)


def load_cell(*, month: str, seed: int) -> Cell:
    """Read the four heads of one cell on identical rows with leads of 0 to 240 hours."""
    heads = {"B-mix": "B-mix", "A0": "A0-inj", "S1": "S1", "Z": "lagless"}
    cols = [*KEY, "lead_hours", "actual", "win_real", "lag_real", "tgt_real", "tgt_post", *LEVEL_NAMES]
    frames = {
        k: pl.read_parquet(_path(head=v, month=month, seed=seed), columns=cols)
        for k, v in heads.items()
    }
    base = frames["B-mix"]
    keep = (base["lead_hours"] <= 240).to_numpy() & np.isfinite(base["actual"].to_numpy())
    for k, frame in frames.items():
        if not frame.select(KEY).equals(base.select(KEY)):
            msg = f"{k} rows differ from B-mix in {month} seed {seed}"
            raise ValueError(msg)
    base = base.filter(pl.Series(keep))
    q = {
        k: frame.filter(pl.Series(keep)).select(LEVEL_NAMES).to_numpy().astype(np.float64)
        for k, frame in frames.items()
    }
    groups = group_index()
    lead = base["lead_hours"].to_numpy().astype(np.float64)
    return Cell(
        month=month,
        seed=seed,
        keys=base.select(KEY),
        actual=base["actual"].to_numpy().astype(np.float64),
        lead=lead,
        flags={f: base[f].to_numpy() for f in ("win_real", "lag_real", "tgt_real", "tgt_post")},
        q=q,
        group=groups[base["primary"].to_numpy().astype(np.int64)],
        band=band_of(lead=lead),
    )


def integrals(*, lagged: np.ndarray, lagless: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the (rows, 3) array of A, B, C per row for a lagged and lagless pair."""
    out = np.empty((len(actual), 3))
    for lo in range(0, len(actual), CHUNK):
        sl = slice(lo, lo + CHUNK)
        pair = mixture.Pair(q1=lagged[sl], q2=lagless[sl], actual=actual[sl])
        x = pair.x
        h = (x >= actual[sl, None]).astype(np.float64)[:, :-1]
        dx = x[:, 1:] - x[:, :-1]
        u0, u1 = (pair.f1 - pair.f2)[:, :-1], (pair.f1 - pair.f2)[:, 1:]
        v0, v1 = pair.f2[:, :-1] - h, pair.f2[:, 1:] - h

        def seg(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
            return (dx * (2 * a0 * b0 + a0 * b1 + a1 * b0 + 2 * a1 * b1) / 6.0).sum(axis=1)

        out[sl, 0] = seg(u0, u1, u0, u1)
        out[sl, 1] = seg(u0, u1, v0, v1)
        out[sl, 2] = seg(v0, v1, v0, v1)
    return out


def mixture_quantiles(
    *, lagged: np.ndarray, lagless: np.ndarray, weights: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return the 11 quantiles of the mixture of two heads, for each named per-row weight."""
    out = {k: np.empty((len(lagged), len(LEVELS))) for k in weights}
    for lo in range(0, len(lagged), CHUNK):
        sl = slice(lo, lo + CHUNK)
        pair = mixture.Pair(q1=lagged[sl], q2=lagless[sl])
        for name, w in weights.items():
            out[name][sl] = pair.quantiles(w=w[sl])
    return out


def short_lead_series(*, seed: int) -> dict[str, np.ndarray]:
    """Return per-head pinball loss and the lagless residual on a (Primary, grid) array.

    Each target is scored by the forecast of the latest origin at or before it (lead up to 24
    hours), against the metered actual. Cells are NaN where no stored forecast covers the target.
    """
    out = {k: np.full((16, GRID_SIZE), np.nan) for k in ("A0", "S1", "Z", "resid")}
    for month in MONTHS:
        cell = load_cell(month=month, seed=seed_of(month=month, cell_seed=seed))
        sel = (cell.target - cell.origin) <= STEPS_PER_DAY
        p, t, a = cell.primary[sel], cell.target[sel], cell.actual[sel]
        if len(np.unique(p * GRID_SIZE + t)) != len(p):
            msg = f"two short-lead forecasts cover one target in {month}"
            raise ValueError(msg)
        for head in ("A0", "S1", "Z"):
            out[head][p, t] = pinball_points(q=cell.q[head][sel], actual=a)
        out["resid"][p, t] = a - cell.q["Z"][sel][:, LEVEL_NAMES.index("q50")]
    return out


def window_features(*, cell: Cell, series: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return per-row recent loss difference (lagged minus lagless) and lagless median residual.

    Both use only targets at least 6 hours before the origin. A row whose Primary-origin has fewer
    than `MIN_USABLE` usable targets gets difference 0 (weight 0.5) and offset 0, and
    `short_window` is True for it.
    """
    pairs = np.unique(np.stack([cell.primary, cell.origin], axis=1), axis=0)
    diff = {h: np.zeros(len(pairs)) for h in ("A0", "S1")}
    offset = np.zeros(len(pairs))
    short = np.zeros(len(pairs), dtype=bool)
    for i, (p, o) in enumerate(pairs):
        hi = o - DELIVERY_STEPS
        w_lo = hi - WINDOW_STEPS
        if w_lo < 0:
            msg = "window starts before the grid"
            raise ValueError(msg)
        zl = series["Z"][p, w_lo + 1 : hi + 1]
        ok = np.isfinite(zl)
        if ok.sum() < MIN_USABLE:
            short[i] = True
            continue
        for h in diff:
            diff[h][i] = np.nanmean(series[h][p, w_lo + 1 : hi + 1][ok]) - zl[ok].mean()
        r = series["resid"][p, hi - OFFSET_STEPS + 1 : hi + 1]
        r = r[np.isfinite(r)]
        if len(r) >= OFFSET_STEPS // 2:
            offset[i] = np.median(r)
    index = {(int(p), int(o)): i for i, (p, o) in enumerate(pairs)}
    row = np.array([index[(int(p), int(o))] for p, o in zip(cell.primary, cell.origin, strict=True)])
    return {
        "d_A0": diff["A0"][row],
        "d_S1": diff["S1"][row],
        "offset": offset[row],
        "short_window": short[row],
    }


def prequential_weight(*, d: np.ndarray, temperature: float) -> np.ndarray:
    """Return the lagged weight: softmax of negative recent loss, floored at 0.2 on each side."""
    w = 1.0 / (1.0 + np.exp(np.clip(d / temperature, -50, 50)))
    return np.clip(w, WEIGHT_FLOOR, 1.0 - WEIGHT_FLOOR)


def exact_crps(*, abc: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return the exact mixture CRPS per row (capacity fractions) for lagged weight `w`."""
    return w * w * abc[:, 0] + 2.0 * w * abc[:, 1] + abc[:, 2]
