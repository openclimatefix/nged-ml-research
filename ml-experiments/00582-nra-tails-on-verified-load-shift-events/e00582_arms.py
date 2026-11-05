# ruff: noqa: E501, C416, ANN001, ANN201, D107
"""Experiment 582: load the stored forecasts and build every arm from their quantiles.

Every arm is an (rows, 11) array of quantiles at `TAUS`, in fractions of effective capacity, on one
shared table of (Primary, origin, valid time) rows with the forecast issued 0 to 120 h before the
valid time. Quantiles are averaged across the two seeds. Arms read only stored forecasts, so the
weather of every row is the run issued at or before the forecast origin by construction (experiment
572's code asserts it row by row when it writes the forecasts).

- `V`: v3+D5, the lag network (experiments 572 and 578; this experiment for 2026-05).
- `M`: the same weights with every lag masked (the lags-masked twin).
- `W`: the same weights with only the week lag masked.
- `C`: the corrected-lag twin, lags rebuilt from the verified counterfactual (this experiment; equal to `V` in months with no verified event).
- `E`: the lag-free D5 network (experiment 565; this experiment for 2026-05; absent in 2025-07 and 2025-08).
"""

from typing import Final

import e00400_blend as b400  # ty: ignore[unresolved-import]
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00582_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

TAUS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
QCOLS: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
IQ: Final[dict[float, int]] = {round(float(t), 2): i for i, t in enumerate(TAUS)}
SEEDS: Final[tuple[int, ...]] = (0, 1)
MAX_LEAD_H: Final[int] = 120
K578_W: Final = k578.WORK
K565_W: Final = k578.d6.W
GROUPS: Final[tuple[str, ...]] = ("MVA-only eight", "signed-MW six", "D15 and D18")
PRIMARY_NAMES: Final[dict[str, str]] = {
    "MVA-only eight": "the eight MVA Primaries",
    "signed-MW six": "the six MW Primaries",
    "D15 and D18": "D15 and D18 (report-only)",
}


def _load(*, root, tag: str, months: tuple[str, ...]) -> pl.DataFrame:
    """Return the seed-mean quantiles of one stored arm, 0-120 h leads, keyed by KEYS."""
    parts = []
    for m in months:
        for d in ("d0", "d14"):
            seeds = [
                pl.read_parquet(
                    root / "fits" / tag / f"{m}_{d}_s{s}.parquet", columns=[*KEYS, *QCOLS]
                )
                for s in SEEDS
            ]
            parts.extend(seeds)
    return pl.concat(parts).group_by(KEYS).agg([pl.col(q).mean() for q in QCOLS])


def _months_of(*, arm: str) -> list[tuple]:
    """Return (root, tag, months) triples of an arm."""
    new = ("2026-05",)
    old = tuple(m for m in k.FOLD_MONTHS if m not in new)
    if arm == "E":
        return [(K565_W, "D5", tuple(m for m in k578.MONTHS)), (k.WORK, "D5", new)]
    if arm == "C":
        return [
            (k.WORK, "V3D5-corrected", k.EVENT_MONTHS)
        ]  # 578's corrected twin uses the labelled corrections and is not read
    tag = {"V": "V3D5", "M": "V3D5-removed"}[arm]
    return [(K578_W, tag, old), (k.WORK, tag, new)]


def sibling_root(*, tag: str):
    """Return the work folder of the week-masked twin."""
    return k.WORK.parent / "work_weekmask"


class Table:
    """The shared row table and the arms' quantile arrays."""

    def __init__(self, *, with_weekmask: bool = True) -> None:
        sm = k578.sites()
        self.sites = sm
        prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
        self.group = np.array(sm["group"].to_list())
        self.label = np.array(sm["label"].to_list())
        self.capacity = sm["capacity_mw"].to_numpy()
        frames = {}
        for arm in ("V", "M", "C", "E"):
            fr = pl.concat([_load(root=r, tag=t, months=ms) for r, t, ms in _months_of(arm=arm)])
            frames[arm] = fr
        if with_weekmask:
            frames["W"] = _load(
                root=k.WORK.parent / "work_weekmask", tag="V3D5-weekmasked", months=k.FOLD_MONTHS
            )
        base = frames["V"]
        for arm, fr in frames.items():
            if arm != "V":
                base = base.join(fr.rename({q: f"{arm}_{q}" for q in QCOLS}), on=KEYS, how="left")
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
        ).filter(pl.col("lead") < MAX_LEAD_H)
        clean = k578.clean()
        self.clean = clean
        self.shift = k.verified_shift(shape=clean.shape)
        p, i = base["prim"].to_numpy(), base["idx"].to_numpy()
        obs = clean[p, i]
        base = base.with_columns(obs=pl.Series(obs), nra=pl.Series(obs + self.shift[p, i])).filter(
            pl.col("obs").is_finite()
        )
        # fold month of the origin: months are named by the origin's month, as in the fits
        base = base.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
        self.df = base.select(
            "prim",
            "idx",
            "origin_idx",
            "lead",
            "obs",
            "nra",
            "month",
            "power_fcst_init_time",
            "valid_time",
        )
        self.prim = self.df["prim"].to_numpy()
        self.idx = self.df["idx"].to_numpy()
        self.origin_idx = self.df["origin_idx"].to_numpy()
        self.lead = self.df["lead"].to_numpy()
        self.obs = self.df["obs"].to_numpy().astype(np.float32)
        self.nra = self.df["nra"].to_numpy().astype(np.float32)
        self.month = self.df["month"].to_numpy()
        self.arms: dict[str, np.ndarray] = {}
        for arm in frames:
            cols = [("" if arm == "V" else f"{arm}_") + q for q in QCOLS]
            q = base.select(cols).to_numpy().astype(np.float32)
            self.arms[arm] = np.sort(q, axis=1)  # no quantile crossing
        # corrected-lag twin equals the lag network where no corrected forecast exists (no verified event in the month)
        ok = np.isfinite(self.arms["C"]).all(axis=1)
        self.arms["C"] = np.where(ok[:, None], self.arms["C"], self.arms["V"])
        self.n = len(self.obs)
        self.band = np.where(self.lead < 24, 0, 1)
        self.group_of_row = self.group[self.prim]
        self.free = self._event_free()

    def _event_free(self) -> np.ndarray:
        """Return rows with no labelled event and no verified transfer within 7 days of the valid time at the same Primary."""
        ev = pl.read_parquet(k578.EVENTS)
        lab2p = dict(zip(self.label.tolist(), range(len(self.label)), strict=True))
        spans: list[tuple[int, int, int]] = [
            (lab2p[r["label"]], r["onset_idx"], r["end_idx"])
            for r in ev.iter_rows(named=True)
            if r["label"] in lab2p
        ]
        spans += [
            (r["primary"], r["onset_idx"], r["end_idx"])
            for r in k.verified_events().iter_rows(named=True)
        ]
        near = np.zeros((len(self.label), self.clean.shape[1]), bool)
        pad = 7 * 48
        for p, a, b in spans:
            near[p, max(a - pad, 0) : b + pad] = True
        return ~near[self.prim, self.idx]

    def neighbours(self) -> dict[int, list[int]]:
        """Return each Primary's metered neighbours (NGED's interconnection list)."""
        lab2p = dict(zip(self.label.tolist(), range(len(self.label)), strict=True))
        return {
            lab2p[a]: [lab2p[x] for x in v if x in lab2p]
            for a, v in b400.nged_neighbours().items()
            if a in lab2p
        }


def pinball(*, y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """Return the pinball loss of each row at level `tau` (same units as the inputs)."""
    d = y - q
    return np.maximum(tau * d, (tau - 1.0) * d)


def crps_from_quantiles(*, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return an approximation of each row's CRPS: twice the mean pinball over the 11 quantile levels."""
    return 2.0 * np.mean(
        [pinball(y=y, q=q[:, j], tau=float(t)) for j, t in enumerate(TAUS)], axis=0
    )


# ---------------------------------------------------------------- recalibration


def scale_quantiles(
    *, q: np.ndarray, s: np.ndarray, rows_band: np.ndarray | None = None
) -> np.ndarray:
    """Return quantiles with each level's distance from the median multiplied by `s[level]`."""
    med = q[:, [IQ[0.5]]]
    return med + (q - med) * s[None, :]


def fit_scales(*, q: np.ndarray, y: np.ndarray, lo: float = 0.2, hi: float = 6.0) -> np.ndarray:
    """Return, for each level, the factor on the distance from the median that makes the share of `y` at or below the quantile equal the level.

    The share is monotone in the factor on each side of the median, so a bisection finds it.
    """
    s = np.ones(len(TAUS))
    med = q[:, IQ[0.5]]
    for j, tau in enumerate(TAUS):
        if abs(tau - 0.5) < 1e-9:
            continue
        d = q[:, j] - med
        a, b = lo, hi
        for _ in range(40):
            mid = 0.5 * (a + b)
            share = np.mean(y <= med + mid * d)
            # a wider band lowers a lower-level quantile and raises an upper-level one: coverage at or below is falling for lower levels
            if (share < tau) == (tau > 0.5):
                a = mid
            else:
                b = mid
        s[j] = 0.5 * (a + b)
    return s


def recalibrate(*, t: Table, arm: np.ndarray, usable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the arm recalibrated per group and lead band on earlier months, and a mask of rows that have a fit.

    For a fold month, the factors are fitted on the event-free rows (`t.free`) of every earlier month in
    `usable` (rows where the arm exists), in the same group and lead band. The first month has no fit.
    """
    out = arm.copy()
    has = np.zeros(t.n, bool)
    months = sorted({m for m in t.month[usable]})
    for mi, m in enumerate(months):
        if mi == 0:
            continue
        here = t.month == m
        first_origin = t.origin_idx[here].min()
        # purge: a training row's valid time must precede the fold month's first origin
        earlier = np.isin(t.month, months[:mi]) & usable & t.free & (t.idx < first_origin)
        for g in GROUPS:
            for b in (0, 1):
                tr = earlier & (t.group_of_row == g) & (t.band == b)
                te = here & usable & (t.group_of_row == g) & (t.band == b)
                if tr.sum() < 500 or te.sum() == 0:
                    continue
                s = fit_scales(q=arm[tr], y=t.obs[tr])
                out[te] = scale_quantiles(q=arm[te], s=s)
                has[te] = True
    return np.sort(out, axis=1), has


# ---------------------------------------------------------------- mixtures and unions

LEVELS_EXT: Final[np.ndarray] = np.concatenate([[0.0], TAUS, [1.0]])


def _knots(q: np.ndarray) -> np.ndarray:
    """Return the 13 abscissae of a quantile function: the 11 quantiles plus an extension at levels 0 and 1."""
    lo = q[:, :1] - 0.25 * (q[:, 1:2] - q[:, :1])
    hi = q[:, -1:] + 0.25 * (q[:, -1:] - q[:, -2:-1])
    k_ = np.concatenate([lo, q, hi], axis=1)
    # strictly increasing knots, so the CDF is well defined
    return np.maximum.accumulate(k_ + np.arange(13)[None, :] * 1e-7, axis=1)


def _cdf(*, x: np.ndarray, kn: np.ndarray) -> np.ndarray:
    """Return the piecewise-linear CDF of knots `kn` (rows, 13) at points `x` (rows, m)."""
    pos = (x[:, :, None] >= kn[:, None, :]).sum(-1) - 1
    pos = np.clip(pos, 0, 11)
    x0 = np.take_along_axis(kn, pos, 1)
    x1 = np.take_along_axis(kn, pos + 1, 1)
    l0, l1 = LEVELS_EXT[pos], LEVELS_EXT[pos + 1]
    f = l0 + (l1 - l0) * np.clip((x - x0) / np.maximum(x1 - x0, 1e-9), 0.0, 1.0)
    return np.where(x < kn[:, :1], 0.0, np.where(x >= kn[:, -1:], 1.0, f))


def mixture(*, a: np.ndarray, b: np.ndarray, w: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    """Return the quantiles of the mixture `w * a + (1 - w) * b` of two distributions given by their 11 quantiles (piecewise-linear CDFs)."""
    out = np.empty_like(a)
    for s in range(0, len(a), chunk):
        sl = slice(s, s + chunk)
        ka, kb = _knots(a[sl]), _knots(b[sl])
        wt = w[sl][:, None]
        bp = np.sort(np.concatenate([ka, kb], axis=1), axis=1)
        f = wt * _cdf(x=bp, kn=ka) + (1.0 - wt) * _cdf(x=bp, kn=kb)
        for j, tau in enumerate(TAUS):
            pos = np.clip((f < tau).sum(1) - 1, 0, bp.shape[1] - 2)
            f0 = np.take_along_axis(f, pos[:, None], 1)[:, 0]
            f1 = np.take_along_axis(f, pos[:, None] + 1, 1)[:, 0]
            x0 = np.take_along_axis(bp, pos[:, None], 1)[:, 0]
            x1 = np.take_along_axis(bp, pos[:, None] + 1, 1)[:, 0]
            out[sl, j] = x0 + (x1 - x0) * np.clip((tau - f0) / np.maximum(f1 - f0, 1e-12), 0.0, 1.0)
    return out


def union_band(*, lag: np.ndarray, masked: np.ndarray) -> np.ndarray:
    """Return the widest of the two bands at every level, with the lag network's median."""
    out = np.where(TAUS[None, :] < 0.5, np.minimum(lag, masked), np.maximum(lag, masked))
    out[:, IQ[0.5]] = lag[:, IQ[0.5]]
    return out


def per_lead_weights(*, t: Table, a: np.ndarray, b: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Return a per-row weight on `a`, per lead band, fitted on earlier months' event-free rows (0.5 where none exist)."""
    w = np.full(t.n, 0.5, np.float32)
    months = sorted({m for m in t.month[usable]})
    grid = np.linspace(0, 1, 11)
    for mi, m in enumerate(months):
        if mi == 0:
            continue
        for band in (0, 1):
            first_origin = t.origin_idx[t.month == m].min()
            tr = (
                np.isin(t.month, months[:mi])
                & usable
                & t.free
                & (t.band == band)
                & (t.idx < first_origin)
            )
            te = (t.month == m) & usable & (t.band == band)
            if tr.sum() < 500 or te.sum() == 0:
                continue
            sub = np.flatnonzero(tr)
            sub = sub[:: max(1, len(sub) // 20000)]
            best, best_w = np.inf, 0.5
            for wv in grid:
                mix = mixture(a=a[sub], b=b[sub], w=np.full(len(sub), wv, np.float32))
                loss = float(
                    np.mean(
                        [
                            pinball(y=t.obs[sub], q=mix[:, j], tau=float(tau)).mean()
                            for j, tau in enumerate(TAUS)
                        ]
                    )
                )
                if loss < best:
                    best, best_w = loss, float(wv)
            w[te] = best_w
    return w


# ---------------------------------------------------------------- the gate (experiment 578's G1 and G2, descriptive thresholds)

GATE_M, GATE_K_SD, GATE_THETA, GATE_RELEASE_DAYS, HOLD_DAYS = 3, 3.0, 0.05, 2, 7


def daily_gap(*, t: Table) -> np.ndarray:
    """Return (16, days) daily mean of (observed - lags-masked forecast) at 24-48 h lead, NaN with under 40 half-hours."""
    sel = (t.lead >= 24) & (t.lead < 48)
    n_days = t.clean.shape[1] // 48
    s, c = np.zeros((16, n_days)), np.zeros((16, n_days))
    d = t.idx[sel] // 48
    np.add.at(s, (t.prim[sel], d), t.obs[sel] - t.arms["M"][sel, IQ[0.5]])
    np.add.at(c, (t.prim[sel], d), 1)
    return np.where(c >= 40, s / np.maximum(c, 1), np.nan)


def run_gates(*, t: Table) -> dict:
    """Return per-variant gate state (16, days) for origin days, and the thresholds used."""
    gap = daily_gap(t=t)
    n_days = gap.shape[1]
    ev = pl.read_parquet(k578.EVENTS)
    lab2p = dict(zip(t.label.tolist(), range(len(t.label)), strict=True))
    near = np.zeros((16, n_days), bool)
    spans = [
        (lab2p[r["label"]], r["onset_idx"], r["end_idx"])
        for r in ev.iter_rows(named=True)
        if r["label"] in lab2p
    ]
    spans += [
        (r["primary"], r["onset_idx"], r["end_idx"])
        for r in k.verified_events().iter_rows(named=True)
    ]
    for p, a, b in spans:
        near[p, max(a // 48 - 3, 0) : b // 48 + 4] = True
    sd = np.array([np.nanstd(gap[p][~near[p]]) for p in range(16)])
    sd = np.where(np.isfinite(sd), sd, 0.0)
    thr = np.maximum(GATE_K_SD * sd, GATE_THETA)
    nb = t.neighbours()
    origin_days = sorted({int(o) // 48 for o in np.unique(t.origin_idx)})

    def sustained(p: int, d: int, sign: float | None) -> float:
        g = gap[p, d - GATE_M : d]
        if not np.isfinite(g).all():
            return 0.0
        s = np.sign(g)
        if (s != s[0]).any() or np.abs(g).min() <= thr[p]:
            return 0.0
        return float(s[0]) if sign is None or s[0] == sign else 0.0

    state = {v: np.zeros((16, n_days), bool) for v in ("G1", "G2")}
    for variant in ("G1", "G2"):
        for p in range(16):
            on = False
            prev = None
            for d in origin_days:
                if prev is not None and d - prev > 1:
                    on = False  # a gap between fold months: the gate state does not carry over
                prev = d
                s = sustained(p, d, None)
                if variant == "G2" and s != 0:
                    s = s if any(sustained(q, d, -s) != 0 for q in nb.get(p, [])) else 0.0
                if not on and s != 0:
                    on = True
                elif on:
                    g = gap[p, d - GATE_RELEASE_DAYS : d]
                    if np.isfinite(g).all() and (np.abs(g) < 0.5 * thr[p]).all():
                        on = False
                state[variant][p, d] = on
    return {
        "state": state,
        "thr": thr,
        "sd": sd,
        "gap": gap,
        "near": near,
        "origin_days": origin_days,
    }


def hold_after_release(
    *, on: np.ndarray, days: int = HOLD_DAYS, valid_days: np.ndarray | None = None
) -> np.ndarray:
    """Return (16, days) True on the days after a gate turned off, for `days` days (the after-snap-back rule)."""
    hold = np.zeros_like(on)
    for p in range(on.shape[0]):
        d = 1
        while d < on.shape[1]:
            if on[p, d - 1] and not on[p, d] and (valid_days is None or valid_days[d]):
                hold[p, d : d + days] = True
            d += 1
    return hold & ~on


def gated_arm(*, t: Table, on: np.ndarray, with_rule: bool) -> np.ndarray:
    """Return the gated forecast: the masked twin while the gate is on; with the rule, the week-masked twin for 7 days after release."""
    od = t.origin_idx // 48
    g = on[t.prim, od]
    out = np.where(g[:, None], t.arms["M"], t.arms["V"])
    if with_rule:
        h = hold_after_release(on=on, valid_days=_origin_day_mask(t=t, n=on.shape[1]))[t.prim, od]
        out = np.where(h[:, None], t.arms["W"], out)
    return out


def _origin_day_mask(*, t: Table, n: int) -> np.ndarray:
    """Return True on days that are forecast origin days (a release is only seen on an origin day)."""
    m = np.zeros(n, bool)
    m[np.unique(t.origin_idx) // 48] = True
    return m
