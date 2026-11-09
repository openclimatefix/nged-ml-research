# ruff: noqa: ANN001, ANN201, D103, E501, PLR0912, PLR0915, C901
"""Experiment 663: the K1 residual panel and the end detector's inputs.

The residual of half-hour `t` at a Primary is the observed cleaned power (plus the injected shift in
a bank world) minus K1's median forecast of `t` issued at the latest daily origin at least 24 hours
before `t` (the day-ahead row of experiment 647's real-row head data, which holds K1's unshifted
output). Every detector input at an origin `o` uses only half-hours before `o`, so nothing after
the origin enters.

Per fold the noise unit (1.4826 times the median absolute deviation) and the constant (median) of
each Primary come from event-free, unflagged, unverified residuals in the four training months
only. The median across Primaries at each half-hour is removed (common-mode weather error).
"""

from dataclasses import dataclass

import e00650_common as c650  # ty: ignore[unresolved-import]
import e00663_common as c663
import numpy as np

H3, DAY, BLK = 6, 48, 6
N_BLOCKS_FRESH = 16  # 3-hour blocks over the last 48 hours
N_DAYS_BEFORE = 8  # daily means over the 8 days before that
PAD = 12 * DAY  # grid padding before a month, for the 10-day context
ROUTINE_DAYS = 60
FEATURE_NAMES = (
    [f"fresh_{i}" for i in range(N_BLOCKS_FRESH)]
    + [f"day_{i}" for i in range(N_DAYS_BEFORE)]
    + ["avail", "miss_0", "miss_1", "miss_2", "miss_3"]
    + ["nb_last24", "nb_prev48", "step_own", "opp_step", "nb_step_mean", "routine_rate"]
)
N_FEATURES = len(FEATURE_NAMES)
NB_COLUMNS = tuple(FEATURE_NAMES.index(n) for n in ("nb_last24", "nb_prev48", "opp_step", "nb_step_mean"))
"""Columns that read neighbours (zeroed in the no-neighbour arm)."""


@dataclass
class MonthPanel:
    """K1's day-ahead median and the observed power of one month on a local grid."""

    month: str
    lo: int  # grid index of local column 0
    obs: np.ndarray  # (16, T) observed cleaned power, capacity fraction; NaN where missing or flagged
    k1: np.ndarray  # (16, T) K1 day-ahead median; NaN where absent
    free: np.ndarray  # (16, T) True where the half-hour is event-free, unflagged, and unverified


def load_month_panel(*, month: str) -> MonthPanel:
    z = np.load(c663.E647 / "head_data" / f"real_{month}.npz")
    ms = c650.month_start_idx(month=month)
    return panel_from_rows(name=month, z=z, lo=ms - PAD, hi=ms + 31 * DAY + 3 * DAY)


def load_window_panel() -> MonthPanel:
    """The final-test window's panel, from `window/real_window.npz` (K1's rows of experiment 620)."""
    z = np.load(c663.WORK / "window" / "real_window.npz")
    return panel_from_rows(name="window", z=z, lo=int(z["oi"].min()) - PAD, hi=int(z["vi"].max()) + 1)


def panel_from_rows(*, name: str, z, lo: int, hi: int) -> MonthPanel:
    obs = np.full((16, hi - lo), np.nan, np.float32)
    k1 = np.full_like(obs, np.nan)
    free = np.zeros(obs.shape, bool)
    day_ahead = (z["lead"] >= 24) & (z["lead"] < 48)
    p, v = z["prim"][day_ahead].astype(int), z["vi"][day_ahead].astype(int) - lo
    inside = (v >= 0) & (v < hi - lo)
    ok = ~z["flagged"][day_ahead] & inside
    obs[p[ok], v[ok]] = z["y"][day_ahead][ok]
    k1[p[inside], v[inside]] = z["q"][day_ahead][inside][:, 5]
    free[p[inside], v[inside]] = (z["event_free"] & ~z["verified"] & ~z["flagged"])[day_ahead][inside]
    return MonthPanel(month=name, lo=lo, obs=obs, k1=k1, free=free)


def bank_k1(*, month: str, panel: MonthPanel) -> dict[int, np.ndarray]:
    """K1's day-ahead median on each bank pass's own (shifted-lag) rows, (16, T) per pass."""
    b = np.load(c650.WORK / "head_data" / f"bank_{month}.npz")
    day_ahead = (b["lead"] >= 24) & (b["lead"] < 48)
    prim, vi, pas, q50 = b["prim"][day_ahead].astype(int), b["vi"][day_ahead].astype(int) - panel.lo, b["pass_"][day_ahead], b["q"][day_ahead][:, 5]
    out = {}
    for r in np.unique(pas):
        k1 = np.full_like(panel.k1, np.nan)
        m = pas == r
        k1[prim[m], vi[m]] = q50[m]
        out[int(r)] = k1
    return out


@dataclass
class Fold:
    """Noise unit and constant per Primary, from the training months of one fold."""

    sigma: np.ndarray
    const: np.ndarray


def fit_fold(*, panels: dict[str, MonthPanel], train_months: list[str]) -> Fold:
    res = [
        np.where(panels[m].free, panels[m].obs - panels[m].k1, np.nan)
        for m in train_months
    ]
    res = np.concatenate(res, axis=1)
    res = res - np.nan_to_num(np.nanmedian(res, axis=0))[None, :]  # as `world_z`: after the common-mode median
    const = np.nanmedian(res, axis=1)
    mad = np.nanmedian(np.abs(res - const[:, None]), axis=1)
    return Fold(sigma=np.maximum(1.4826 * mad, 1e-4), const=const)


def world_z(*, panel: MonthPanel, fold: Fold, own: np.ndarray | None, mirror: np.ndarray | None = None,
            k1: np.ndarray | None = None, extra: np.ndarray | None = None, extra_common: np.ndarray | None = None):  # fmt: skip
    """Return the standardised residual (16, T) of a world.

    `own` and `mirror` are the injected shifts on the full grid (the Primary's own shift and the
    receivers' mirrored shift, shares of capacity); `k1` is the world's K1 day-ahead median (the
    real-row one when None); `extra` is added per Primary before, and `extra_common` after, the
    removal of the median across Primaries.
    """
    obs = panel.obs.copy()
    for shift in (own, mirror):
        if shift is not None:
            obs = obs + shift[:, panel.lo : panel.lo + obs.shape[1]]
    z = (obs - (panel.k1 if k1 is None else k1) - fold.const[:, None]) / fold.sigma[:, None]
    if extra is not None:
        z = z + extra
    med = np.nanmedian(z, axis=0)
    z = z - np.nan_to_num(med)[None, :]
    if extra_common is not None:
        z = z + extra_common
    return z.astype(np.float32)


def hard_negatives(*, shape: tuple[int, int], nb_of: dict[int, list[int]], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-Primary, common-mode) additive perturbations (noise units) that are not load shifts."""
    rng = np.random.default_rng(seed)
    n, t = shape
    out = np.zeros((n, t + 12 * DAY), np.float32)
    common = np.zeros(t + 12 * DAY, np.float32)
    for p in range(n):
        for _ in range(rng.poisson(t / DAY / 10.0)):
            kind = rng.integers(5)
            a = int(rng.integers(0, t - 6 * DAY))
            amp = float(rng.choice([-1, 1]) * rng.uniform(2, 6))
            if kind == 0:  # pulse under 3 hours
                out[p, a : a + int(rng.integers(1, 6))] += amp
            elif kind == 1:  # ramp of 2 to 12 hours, held 6 hours to 2 days, then instant return
                w, h = int(rng.integers(4, 24)), int(rng.integers(12, 2 * DAY))
                out[p, a : a + w] += amp * np.linspace(0, 1, w)
                out[p, a + w : a + w + h] += amp
            elif kind == 2:  # routine step: 4 to 10 hours repeated for 5 days
                w = int(rng.integers(8, 20))
                for d in range(5):
                    out[p, a + d * DAY : a + d * DAY + w] += amp / 2
            elif kind == 3:  # opposite steps at this Primary and an unconnected one, 1 to 3 days
                far = [q for q in range(n) if q != p and q not in nb_of.get(p, [])]
                q = int(rng.choice(far))
                h = int(rng.integers(DAY, 3 * DAY))
                out[p, a : a + h] += amp
                out[q, a : a + h] -= amp
            else:  # smooth common mode, 3 to 10 days
                h = int(rng.integers(3 * DAY, 10 * DAY))
                common[a : a + h] += 0.5 * amp / 3 * np.sin(np.linspace(0, np.pi, h))
    return out[:, :t], common[:t]


def rewire(*, nb_of: dict[int, list[int]], seed: int) -> dict[int, list[int]]:
    """Degree-preserving edge swaps that keep as few original edges as possible; return the new sets."""
    edges = sorted({tuple(sorted((a, b))) for a, v in nb_of.items() for b in v})
    orig = set(edges)
    rng = np.random.default_rng(seed)
    cur = list(edges)
    best, best_kept = list(cur), len(cur)
    for _ in range(20000):
        i, j = rng.choice(len(cur), 2, replace=False)
        (a, b), (c, d) = cur[i], cur[j]
        if rng.random() < 0.5:
            c, d = d, c
        e1, e2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        if a == d or c == b or e1 == e2 or e1 in cur or e2 in cur:
            continue
        cur[i], cur[j] = e1, e2
        kept = sum(e in orig for e in cur)
        if kept < best_kept:
            best, best_kept = list(cur), kept
        if best_kept == 0:
            break
    out: dict[int, list[int]] = {p: [] for p in range(16)}
    for a, b in best:
        out[a].append(b)
        out[b].append(a)
    rewire.kept = best_kept  # ty: ignore[unresolved-attribute]
    return out


def _mean(cs: np.ndarray, cn: np.ndarray, p: int, a: int, b: int) -> tuple[float, int]:
    top = cn.shape[1] - 1
    a, b = min(max(a, 0), top), min(max(b, 0), top)
    n = cn[p, b] - cn[p, a]
    return (float((cs[p, b] - cs[p, a]) / n) if n > 0 else 0.0), int(n)


def features(*, z: np.ndarray, panel: MonthPanel, origins: np.ndarray, prims: np.ndarray,
             nb_of: dict[int, list[int]]) -> np.ndarray:  # fmt: skip
    """Return the detector inputs (n, N_FEATURES) for rows (`prims`, grid-index `origins`)."""
    fin = np.isfinite(z)
    cs = np.concatenate([np.zeros((16, 1)), np.cumsum(np.where(fin, z, 0.0), axis=1, dtype=np.float64)], axis=1)
    cn = np.concatenate([np.zeros((16, 1), int), np.cumsum(fin, axis=1)], axis=1)
    out = np.zeros((len(origins), N_FEATURES), np.float32)
    for r, (p, o) in enumerate(zip(prims.tolist(), (origins - panel.lo).tolist(), strict=True)):
        avail = 0
        for i in range(N_BLOCKS_FRESH):
            b = o - i * H3
            m, n = _mean(cs, cn, p, b - H3, b)
            out[r, i] = m
            avail += n > 0
        for i in range(N_DAYS_BEFORE):
            b = o - 2 * DAY - i * DAY
            m, n = _mean(cs, cn, p, b - DAY, b)
            out[r, N_BLOCKS_FRESH + i] = m
            avail += n > 0
        out[r, N_BLOCKS_FRESH + N_DAYS_BEFORE] = avail / (N_BLOCKS_FRESH + N_DAYS_BEFORE)
        for i in range(4):
            out[r, N_BLOCKS_FRESH + N_DAYS_BEFORE + 1 + i] = float(_mean(cs, cn, p, o - (i + 1) * H3, o - i * H3)[1] == 0)
        base = N_BLOCKS_FRESH + N_DAYS_BEFORE + 5
        l24, _ = _mean(cs, cn, p, o - DAY, o)
        p48, _ = _mean(cs, cn, p, o - 3 * DAY, o - DAY)
        own = l24 - p48
        nbs = nb_of.get(p, [])
        steps, lasts, prevs = [], [], []
        for q in nbs:
            a, _ = _mean(cs, cn, q, o - DAY, o)
            b, _ = _mean(cs, cn, q, o - 3 * DAY, o - DAY)
            lasts.append(a)
            prevs.append(b)
            steps.append(a - b)
        out[r, base : base + 2] = (np.mean(lasts) if lasts else 0.0, np.mean(prevs) if prevs else 0.0)
        out[r, base + 2] = own
        sgn = np.sign(own)
        out[r, base + 3] = max(0.0, max((-sgn * s for s in steps), default=0.0))
        out[r, base + 4] = np.mean(steps) if steps else 0.0
        days = [_mean(cs, cn, p, o - (d + 1) * DAY, o - d * DAY) for d in range(2, ROUTINE_DAYS)]  # ends 2 days before the origin
        means = np.array([m for m, n in days if n > 0])
        out[r, base + 5] = float((np.abs(np.diff(means)) > 3.0).sum() / max(len(means), 1)) if len(means) > 1 else 0.0
    return np.clip(out, -20.0, 20.0)
