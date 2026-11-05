# ruff: noqa: E501, D107, D205
"""Experiment 583: the shared row table of X3-syn's stored forecasts, and the after-snap-back rule.

Every arm is built from stored quantile forecasts; nothing is refitted. Two forecasts are read, both
the seed mean of experiment 581's X3-syn network (X3-med fine-tuned with lag-only synthetic
contamination), at leads 0 to 120 h:

- `V`: X3-syn with its lags (`fits/V3D5`).
- `M`: the same weights with every lag masked (`fits/V3D5-removed`), the "lags-masked twin" and the
  only lag-free forecast X3-syn has stored.

Time is the half-hour grid index of experiment 578 (`GRID_START`); power is a share of effective
capacity. Phases are by the valid time of the row, as in experiment 582.
"""

from typing import Final

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00582_arms as a  # ty: ignore[unresolved-import]
import e00582_common as k582  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00583-after-snap-back-event-band-on-x3-syn"
WORK: Final = ML_EXPERIMENTS_DATA_DIR / NAME
X3SYN: Final = ML_EXPERIMENTS_DATA_DIR / "00581-robust-multiweek-lag-encoder-x3/arms/x3syn/work"
INJECT: Final = X3SYN / "inject"
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
"""The five fold months for which X3-syn forecasts are stored (2026-07 stays unused)."""
IQ: Final = a.IQ
QCOLS: Final = a.QCOLS
KEYS: Final = a.KEYS
HOLD_DAYS: Final[int] = 7
"""Days the event-mode band stays on after a snap-back."""
GATE_M: Final[int] = 3
"""Days of the sustained lag-versus-expectation gap (the brief's m = 3)."""
LAG_DAYS: Final[int] = 2
"""The newest day the gate reads is `origin day - 2`, so every reading is at least 24 h old at any origin hour (the v3 lag rule)."""
GATE_K_SD: Final[float] = a.GATE_K_SD
GATE_THETA: Final[float] = a.GATE_THETA
N_PLACEBO: Final[int] = 20


def _seed_mean(*, tag: str) -> pl.DataFrame:
    """Return the seed-mean quantiles of X3-syn's stored `fits/<tag>` forecasts, keyed by KEYS."""
    parts = [
        pl.read_parquet(X3SYN / "fits" / tag / f"{m}_{d}_s{s}.parquet", columns=[*KEYS, *QCOLS])
        for m in MONTHS
        for d in ("d0", "d14")
        for s in a.SEEDS
    ]
    return pl.concat(parts).group_by(KEYS).agg([pl.col(q).mean() for q in QCOLS])


class X4Table(a.Table):
    """Rows of the five fold months with X3-syn's lag network (`arms["V"]`) and its lags-masked
    twin (`arms["M"]`). The methods that read only `label`, `clean`, `prim`, `idx`, `n` are
    inherited from experiment 582's table.
    """

    def __init__(self) -> None:
        sm = k578.sites()
        self.sites = sm
        prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
        self.group = np.array(sm["group"].to_list())
        self.label = np.array(sm["label"].to_list())
        self.capacity = sm["capacity_mw"].to_numpy()
        base = _seed_mean(tag="V3D5").join(
            _seed_mean(tag="V3D5-removed").rename({q: f"M_{q}" for q in QCOLS}),
            on=KEYS,
            how="inner",
        )
        base = base.with_columns(
            lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0,
            prim=pl.col("time_series_id").replace_strict(prim_of, return_dtype=pl.Int64),
            idx=(
                (pl.col("valid_time").dt.epoch("us") - k578.prep.GRID_START_US) // k578.prep.STEP_US
            ).cast(pl.Int64),
            origin_idx=(
                (pl.col("power_fcst_init_time").dt.epoch("us") - k578.prep.GRID_START_US)
                // k578.prep.STEP_US
            ).cast(pl.Int64),
        ).filter(pl.col("lead") < a.MAX_LEAD_H)
        self.clean = k578.clean()
        self.shift = k582.verified_shift(shape=self.clean.shape)
        p, i = base["prim"].to_numpy(), base["idx"].to_numpy()
        obs = self.clean[p, i]
        base = base.with_columns(obs=pl.Series(obs), nra=pl.Series(obs + self.shift[p, i])).filter(
            pl.col("obs").is_finite()
        )
        base = base.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
        self.base = base
        self.prim = base["prim"].to_numpy()
        self.idx = base["idx"].to_numpy()
        self.origin_idx = base["origin_idx"].to_numpy()
        self.lead = base["lead"].to_numpy()
        self.obs = base["obs"].to_numpy().astype(np.float32)
        self.nra = base["nra"].to_numpy().astype(np.float32)
        self.month = base["month"].to_numpy()
        self.arms = {
            "V": np.sort(base.select(QCOLS).to_numpy().astype(np.float32), axis=1),
            "M": np.sort(
                base.select([f"M_{q}" for q in QCOLS]).to_numpy().astype(np.float32), axis=1
            ),
        }
        self.n = len(self.obs)
        self.band = np.where(self.lead < 24, 0, 1)
        self.group_of_row = self.group[self.prim]
        self.origin_day = self.origin_idx // 48
        self.free = self._event_free()
        del self.base


# ---------------------------------------------------------------- the gate and the snap-back rule


def gate_inputs(*, t: X4Table) -> dict:
    """Return the daily lag-versus-expectation gap (16, days), the per-Primary threshold, the
    event-near mask, and the origin days. Experiment 582's `run_gates` computes the gap (observed
    minus the lags-masked median at 24-48 h lead, a day-level reading of the lags) and the
    threshold `max(3 sd, 5% of capacity)` from event-free days.
    """
    g = a.run_gates(t=t)
    return {k: g[k] for k in ("gap", "thr", "sd", "near", "origin_days")}


def gate_state(
    *,
    gap: np.ndarray,
    thr: np.ndarray,
    origin_days: list[int],
    release_days: int = 1,
    lag_days: int = LAG_DAYS,
    m: int = GATE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (on, snap), each (16, days) booleans by origin day.

    `on[p, d]`: the lag-versus-expectation gap of Primary `p` was sustained for `m` days (same sign,
    each day above the threshold), the newest of those days being `d - lag_days`, and has not yet
    collapsed. `snap[p, d]`: the gate turned off at origin day `d` (the gap fell below half the
    threshold on each of the newest `release_days` days). A gap in the origin days (between fold
    months) resets the state.
    """
    n_days = gap.shape[1]
    on = np.zeros((16, n_days), bool)
    snap = np.zeros((16, n_days), bool)
    for p in range(gap.shape[0]):
        state = False
        prev = None
        for d in origin_days:
            if prev is not None and d - prev > 1:
                state = False
            prev = d
            w = gap[p, d - lag_days - m + 1 : d - lag_days + 1]
            sustained = bool(np.isfinite(w).all()) and bool(
                (np.sign(w) == np.sign(w[0])).all() and (np.abs(w) > thr[p]).all()
            )
            if not state and sustained:
                state = True
            elif state:
                r = gap[p, d - lag_days - release_days + 1 : d - lag_days + 1]
                if np.isfinite(r).all() and (np.abs(r) < 0.5 * thr[p]).all():
                    state = False
                    snap[p, d] = True
            on[p, d] = state
    return on, snap


def hold_from(*, snap: np.ndarray, days: int = HOLD_DAYS) -> np.ndarray:
    """Return (16, days) True on the `days` origin days from each snap-back day on."""
    hold = np.zeros_like(snap)
    for p in range(snap.shape[0]):
        for d in np.flatnonzero(snap[p]):
            hold[p, d : d + days] = True
    return hold


def placebo_snap(
    *, snap: np.ndarray, origin_days: list[int], rng: np.random.Generator
) -> np.ndarray:
    """Return a snap-back array with the same number of snap-backs per Primary, on origin days
    drawn at random (without replacement) from that Primary's origin days.
    """
    out = np.zeros_like(snap)
    od = np.array(origin_days)
    for p in range(snap.shape[0]):
        n = int(snap[p].sum())
        if n:
            out[p, rng.choice(od, size=min(n, len(od)), replace=False)] = True
    return out


ARMS: Final[tuple[str, ...]] = ("V", "M", "Union", "A1", "A2", "A3", "A4")
"""`V`: X3-syn with lags. `M`: the lags-masked twin. `Union`: the widest of the two bands at every
level with the lag median, on every row (experiment 582's union band, the control). `A1`: `Union`
on origin days held after a snap-back, `V` elsewhere. `A2`: `Union` on gate-on and held origin
days, `V` elsewhere. `A3`: as `A1` with the lags-masked median on held days. `A4`: the placebo,
`A1` with the snap-backs moved to random origin days (the same number per Primary), averaged over
`N_PLACEBO` draws."""


def union_masked_median(*, union: np.ndarray, masked: np.ndarray) -> np.ndarray:
    """Return the union band with the lags-masked twin's median (arm A3's event-mode band)."""
    out = union.copy()
    out[:, IQ[0.5]] = masked[:, IQ[0.5]]
    return out


def combine(
    *, base: dict[str, np.ndarray], hold: np.ndarray, on: np.ndarray, frac: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Return every arm from per-row arrays of its four building blocks.

    `base` holds `V`, `M`, `Union`, and `UnionM` (the union band with the masked median) as arrays
    whose first axis is the row (quantiles or row-level metrics, alike). `hold` and `on` are
    per-row booleans: the row's origin day is held after a snap-back, or the gate is on. `frac` is
    the per-row share of placebo draws on which the row's origin day was held.
    """
    sel = (slice(None),) + (None,) * (base["V"].ndim - 1)
    h, g = hold[sel], (hold | on)[sel]
    out = {
        "V": base["V"],
        "M": base["M"],
        "Union": base["Union"],
        "A1": np.where(h, base["Union"], base["V"]),
        "A2": np.where(g, base["Union"], base["V"]),
        "A3": np.where(h, base["UnionM"], base["V"]),
    }
    if frac is not None:
        f = frac[sel]
        out["A4"] = base["V"] + f * (base["Union"] - base["V"])
    return out


def building_blocks(*, v: np.ndarray, m: np.ndarray) -> dict[str, np.ndarray]:
    """Return the quantiles of the four building blocks `V`, `M`, `Union`, and `UnionM`."""
    u = a.union_band(lag=v, masked=m)
    return {"V": v, "M": m, "Union": u, "UnionM": union_masked_median(union=u, masked=m)}


# ---------------------------------------------------------------- row-level metrics

METRICS: Final[tuple[str, ...]] = (
    "cover90",
    "cover98",
    "ge05",
    "le95",
    "ge01",
    "le99",
    "pin01",
    "pin05",
    "pin95",
    "pin99",
    "width90",
    "width98",
    "ae50",
)


def row_metrics(*, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return (n_rows, len(METRICS)) per-row values: coverage indicators, one-sided indicators,
    pinball losses at 0.01, 0.05, 0.95, 0.99, interval widths, and the absolute error of the
    median. Every reported metric is the mean of a column over a set of rows, so a placebo's
    average over draws is the average of its row-level columns.
    """
    return np.stack(
        [
            ((q[:, IQ[0.05]] <= y) & (y <= q[:, IQ[0.95]])).astype(np.float32),
            ((q[:, IQ[0.01]] <= y) & (y <= q[:, IQ[0.99]])).astype(np.float32),
            (y >= q[:, IQ[0.05]]).astype(np.float32),
            (y <= q[:, IQ[0.95]]).astype(np.float32),
            (y >= q[:, IQ[0.01]]).astype(np.float32),
            (y <= q[:, IQ[0.99]]).astype(np.float32),
            a.pinball(y=y, q=q[:, IQ[0.01]], tau=0.01),
            a.pinball(y=y, q=q[:, IQ[0.05]], tau=0.05),
            a.pinball(y=y, q=q[:, IQ[0.95]], tau=0.95),
            a.pinball(y=y, q=q[:, IQ[0.99]], tau=0.99),
            q[:, IQ[0.95]] - q[:, IQ[0.05]],
            q[:, IQ[0.99]] - q[:, IQ[0.01]],
            np.abs(y - q[:, IQ[0.5]]),
        ],
        axis=1,
    ).astype(np.float32)
