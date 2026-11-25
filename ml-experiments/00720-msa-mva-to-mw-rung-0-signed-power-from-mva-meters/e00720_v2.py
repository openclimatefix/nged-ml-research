# ruff: noqa: E501, PLR0915, PT018, PT017
"""Experiment 720, rung 0e: V2, K1, and V3 against experiment 694's tracker `M2`, on the same series.

The hindsight model (`e00720_hmm` with `K = 3`) and experiment 694's `M2` (`grid_track`, imported and
rerun, never edited) see the same injected reading and are scored on the same half-hours with the
same flag rule, `P(n >= 1) > 0.5` smoothed. The contrast is paired by segment and read by a cluster
bootstrap over segments, as the amended addendum requires.

Settings of the hindsight model on a 694 segment, all stated departures: no weather is read (the 694
panel holds none), so the demand mean is calendar only; no PV estimate enters (`p = 0`); the demand
mean, the autoregression, and the residual scale are fitted on the 28-day pre-roll alone, which holds
no engine output; the unit size is profiled on a coarse grid and then on a fine grid around the best
value; `K` is fixed at 3 unless more candidates are given. The inferred unit counts and flags are kept
in memory and are written only under the private folder, never printed.

`python e00720_v2.py --selftest` uses a synthetic host and engine. `--run` needs the stored panel of
experiment 196 and the engine record, and waits for the code review.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any, Final

import e00720_bars as bars
import e00720_fit as fit
import e00720_hmm as hmm
import numpy as np

Q_FLOOR_MVA: Final[float] = 0.14
PRE_ROLL: Final[int] = 28 * 48
STAY: Final[float] = 0.97
COARSE_G: Final[tuple[float, ...]] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
FINE_OFFSETS: Final[tuple[float, ...]] = (-0.25, 0.25)
N_BINS_V2: Final[int] = 21
EM_ROUNDS_V2: Final[int] = 2
MIN_UNIT_MVA: Final[float] = 1.0
"""The unit size is not searched below this value, so a step of demand smaller than a unit is absorbed by the
demand mean and never read as an engine. A step of 1 MVA or more cannot be told from an engine that stays on."""
TREND_WINDOW_DAYS: Final[int] = 21
TREND_QUANTILE: Final[float] = 0.8
"""The slow trend is the upper quantile of the daily-mean residual over a centred window, less the same
statistic on the pre-roll: an engine lowers the net reading, so the idle days sit at the upper edge."""
TREND_REFITS: Final[int] = 1
"""After the first scan of the unit size, the trend is refitted from the half-hours the model believes are
free of engine output, and the whole coarse grid of the unit size is scanned again, with the fine offsets around its winner."""
IDLE_P0: Final[float] = 0.9
"""A half-hour feeds the refitted trend only when its smoothed `P(n = 0)` is above this value."""
MIN_IDLE_HALF_HOURS: Final[int] = 12
"""A day feeds the refitted trend only with at least this many such half-hours."""
MAX_TREND_SLOPE: Final[float] = 0.03
"""The first trend moves at most this many MVA per day (3.6 MVA over 120 days), so a spell of engine output
that lasts days, which looks like a step, cannot pull it down."""
START_DOW_GRID: Final[int] = 5
"""2022-01-01, the start of experiment 694's grid, was a Saturday (Monday is 0)."""


def day_types(*, n: int, first_index: int) -> np.ndarray:
    """Return the day type (0 weekday, 1 Saturday, 2 Sunday) of each of `n` half-hours from grid index `first_index`."""
    dow = (START_DOW_GRID + (first_index + np.arange(n)) // 48) % 7
    return np.where(dow == 5, 1, np.where(dow == 6, 2, 0))


def to_days(*, a: np.ndarray, first_index: int) -> tuple[np.ndarray, int]:
    """Pad a half-hourly series to whole days; return `(n_days, 48)` and the number of leading pad steps."""
    lead = first_index % 48
    n = len(a) + lead
    n_days = -(-n // 48)
    out = np.full(n_days * 48, np.nan)
    out[lead : lead + len(a)] = a
    return out.reshape(n_days, 48), lead


def slow_trend(*, ymat: np.ndarray, f_mat: np.ndarray, calib_rows: np.ndarray) -> np.ndarray:
    """Return a per-day offset that follows a slow drift of demand without following the engine.

    The daily mean of the residual `ymat - f_mat` is summarised by its upper quantile over a centred
    window, smoothed over 7 days, and the same statistic on the days of `calib_rows` (the pre-roll,
    which holds no engine and no trend by construction) is subtracted.

    Args:
        ymat: `(days, 48)` magnitude of the reading.
        f_mat: `(days, 48)` the calendar mean fitted on the pre-roll.
        calib_rows: `(days,)` boolean, the days of the pre-roll.

    Returns:
        The offset per day, `(days,)`, with the pre-roll mean removed.
    """
    with np.errstate(all="ignore"):
        day = np.nanmean(ymat - f_mat, axis=1)
    n = len(day)
    half = TREND_WINDOW_DAYS // 2
    stat = np.full(n, np.nan)
    for d in range(n):
        w = day[max(0, d - half) : min(n, d + half + 1)]
        if np.isfinite(w).sum() >= 5:
            stat[d] = np.nanquantile(w, TREND_QUANTILE)
    ok = np.isfinite(stat)
    if not ok.any():
        return np.zeros(n)
    stat = np.where(ok, stat, np.nanmedian(stat))
    smooth = np.convolve(np.pad(stat, 3, mode="edge"), np.ones(7) / 7.0, mode="valid")
    smooth = rate_limited(a=smooth, max_step=MAX_TREND_SLOPE)
    base = float(np.nanmean(smooth[calib_rows])) if calib_rows.any() else 0.0
    return smooth - base


def rate_limited(*, a: np.ndarray, max_step: float) -> np.ndarray:
    """Return `a` followed by a path that moves at most `max_step` per step, the mean of a forward and a backward pass."""

    def one(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = out[i - 1] + np.clip(x[i] - out[i - 1], -max_step, max_step)
        return out

    return 0.5 * (one(a) + one(a[::-1])[::-1])


def idle_trend(
    *, ymat: np.ndarray, f_mat: np.ndarray, p0_mat: np.ndarray, calib_rows: np.ndarray
) -> np.ndarray:
    """Return a per-day offset fitted only on half-hours with a high probability of no engine output.

    The daily mean of `ymat - f_mat` over the half-hours with `p0_mat > IDLE_P0` is taken on days that
    hold at least `MIN_IDLE_HALF_HOURS` of them, smoothed by the median of the days in a centred window,
    interpolated linearly across the days that hold too few (a long engine spell), smoothed over 7 days,
    and calibrated on the pre-roll as `slow_trend` is. A spell therefore cannot lower the trend, because
    the days inside it contribute nothing and the trend bridges them.
    """
    resid = ymat - f_mat
    use = np.isfinite(resid) & (p0_mat > IDLE_P0)
    cnt = use.sum(axis=1)
    day = np.where(
        cnt >= MIN_IDLE_HALF_HOURS,
        np.where(use, resid, 0.0).sum(axis=1) / np.maximum(cnt, 1),
        np.nan,
    )
    n = len(day)
    half = TREND_WINDOW_DAYS // 2
    med = np.full(n, np.nan)
    for d in range(n):
        w = day[max(0, d - half) : min(n, d + half + 1)]
        if np.isfinite(w).sum() >= 3:
            med[d] = np.nanmedian(w)
    ok = np.isfinite(med)
    if not ok.any():
        return np.zeros(n)
    med = np.interp(np.arange(n), np.flatnonzero(ok), med[ok])
    smooth = np.convolve(np.pad(med, 3, mode="edge"), np.ones(7) / 7.0, mode="valid")
    base = float(np.nanmean(smooth[calib_rows])) if calib_rows.any() else 0.0
    return smooth - base


@dataclass
class HindsightFit:
    """The result of one hindsight fit; `flag` and `filt` are `P(n >= 1) > 0.5` smoothed and filtered."""

    flag: np.ndarray
    filt: np.ndarray
    unit_size: float
    n_max: int
    log_likelihood: float
    seconds: float
    passes: int
    n_hat: np.ndarray | None = None
    p_any: np.ndarray | None = None
    """Smoothed `P(n >= 1)` per half-hour after the pre-roll."""
    e_n: np.ndarray | None = None
    """Smoothed expected unit count per half-hour after the pre-roll."""
    n_obs: int = 0
    p_max: np.ndarray | None = None
    """Largest smoothed unit-count posterior per half-hour after the pre-roll."""
    """Most probable unit count per half-hour after the pre-roll (smoothed)."""


def hindsight_fit(
    *,
    y: np.ndarray,
    first_index: int,
    pre_roll: int = PRE_ROLL,
    n_max: int = 3,
    q: float = Q_FLOOR_MVA,
    coarse: tuple[float, ...] = COARSE_G,
    n_bins: int = N_BINS_V2,
    gate: hmm.CostGate | None = None,
    projected_fits: float = 1.0,
) -> HindsightFit:
    """Fit the switching model to one reading and return the flags of the part after the pre-roll.

    Args:
        y: Reading (MVA), pre-roll first. NaN half-hours are treated as missing.
        first_index: Grid index of `y[0]` (sets the day type).
        pre_roll: Half-hours at the start that are assumed to hold no engine output.
        n_max: `K`.
        q: Reactive floor in the units of `y`.
        coarse: Coarse grid of the unit size.
        n_bins: Residual grid size.
        gate: The cost gate; the first fit measures it and later fits are checked against it.
        projected_fits: How many fits the whole run will make (for the projection).

    Returns:
        A `HindsightFit` holding the flags, the filtered probabilities, the unit size, the log-
        likelihood, and the timing of the fit.
    """
    t0 = time.perf_counter()
    mag = np.sqrt(np.maximum(y**2 - q**2, 0.0))
    ymat, lead = to_days(a=mag, first_index=first_index)
    dtype_days, _ = to_days(
        a=day_types(n=len(y), first_index=first_index).astype(float), first_index=first_index
    )
    dtype_day = np.nan_to_num(np.nanmedian(dtype_days, axis=1)).astype(int)
    anchors = np.zeros(ymat.shape, dtype=bool)
    flat_anchor = np.zeros(ymat.size, dtype=bool)
    flat_anchor[lead : lead + pre_roll] = True
    anchors = flat_anchor.reshape(ymat.shape) & np.isfinite(ymat)
    temp0 = np.zeros(len(dtype_day))
    f_mat = fit.fit_demand_mean(g=ymat, rows=anchors, day_type=dtype_day, temperature=temp0)
    phi, sigma_r = fit.fit_residual(g=ymat, f=f_mat, rows=anchors)
    calib_days = anchors.any(axis=1)
    trend = slow_trend(ymat=ymat, f_mat=f_mat, calib_rows=calib_days)
    f = (f_mat + trend[:, None]).reshape(-1)[lead : lead + len(y)]
    p = np.zeros(len(y))
    passes = 0
    last_post: hmm.Posterior | None = None

    def spec_for(g: float, trans: np.ndarray | None = None) -> hmm.HmmSpec:
        k = n_max + 1
        base = np.full((k, k), (1.0 - STAY) / max(k - 1, 1))
        np.fill_diagonal(base, STAY if k > 1 else 1.0)
        return hmm.HmmSpec(
            grid=hmm.residual_grid(phi=min(phi, 0.97), sigma_r=sigma_r, n_bins=n_bins),
            phi=min(phi, 0.97), sigma_r=sigma_r, unit_size=g, n_max=n_max,
            trans_n=base if trans is None else trans, q=q, sigma_e=0.08,
        )  # fmt: skip

    def pass_at(
        g: float, *, want_xi: bool = False, trans: np.ndarray | None = None
    ) -> tuple[hmm.Posterior, hmm.HmmSpec]:
        nonlocal passes
        if gate is not None:
            gate.check_elapsed()
        spec = spec_for(g, trans)
        post = hmm.forward_backward(spec=spec, m=y, f=f, p=p, want_xi=want_xi)
        passes += 1
        return post, spec

    if n_max == 0:
        post0, _ = pass_at(1.0)
        n0 = len(y) - pre_roll
        return HindsightFit(
            flag=np.zeros(n0, dtype=bool), filt=np.zeros(n0, dtype=bool), unit_size=0.0, n_max=0,
            log_likelihood=post0.log_likelihood, seconds=time.perf_counter() - t0, passes=passes,
            n_hat=np.zeros(n0, dtype=int), p_any=np.zeros(n0), e_n=np.zeros(n0),
            n_obs=int(np.isfinite(y[pre_roll:]).sum()),
        )  # fmt: skip
    scores: dict[float, float] = {}
    for i, g in enumerate(coarse):
        if i == 0 and gate is not None and not np.isfinite(gate.seconds_per_pass):
            gate.measure(run_one_pass=lambda g=g: pass_at(g))
            n_passes = projected_fits * (
                len(coarse)
                + len(FINE_OFFSETS)
                + TREND_REFITS * (1 + len(coarse) + len(FINE_OFFSETS))
                + EM_ROUNDS_V2
                + 1
            )
            gate.check(n_passes_total=round(n_passes))
            scores[g] = float("nan")  # replaced below
        post, _ = pass_at(g)
        scores[g] = post.log_likelihood
    best = max(scores, key=lambda g: scores[g])
    for off in FINE_OFFSETS:
        g = best + off
        if g >= MIN_UNIT_MVA and g not in scores:
            scores[g] = pass_at(g)[0].log_likelihood
    best = max(scores, key=lambda g: scores[g])
    for _ in range(TREND_REFITS):
        sm0 = hmm.summarise(post=pass_at(best)[0], spec=spec_for(best))["p_units"][:, 0]
        p0_mat, _ = to_days(a=sm0, first_index=first_index)
        trend = idle_trend(ymat=ymat, f_mat=f_mat, p0_mat=p0_mat, calib_rows=calib_days)
        f = (f_mat + trend[:, None]).reshape(-1)[lead : lead + len(y)]
        scores = {g: pass_at(g)[0].log_likelihood for g in coarse}
        top = max(scores, key=lambda g: scores[g])
        for off in FINE_OFFSETS:
            g = top + off
            if g >= MIN_UNIT_MVA and g not in scores:
                scores[g] = pass_at(g)[0].log_likelihood
        best = max(scores, key=lambda g: scores[g])
    trans = None
    for _ in range(EM_ROUNDS_V2):
        post, spec = pass_at(best, want_xi=True, trans=trans)
        trans = hmm.update_trans_n(xi_n=post.xi_n)
    last_post, spec = pass_at(best, trans=trans)
    sm = hmm.summarise(post=last_post, spec=spec)["p_units"]
    n_hat = np.argmax(sm, axis=1)[pre_roll:]
    e_n = (sm * np.arange(sm.shape[1])[None, :]).sum(axis=1)[pre_roll:]
    p_max = sm.max(axis=1)[pre_roll:]
    fl = hmm.summarise(post=last_post, spec=spec, version="filtered")["p_units"]
    any_s = 1.0 - sm[:, 0]
    any_f = 1.0 - fl[:, 0]
    return HindsightFit(
        flag=(any_s > 0.5)[pre_roll:], filt=(any_f > 0.5)[pre_roll:], unit_size=best, n_max=n_max,
        log_likelihood=last_post.log_likelihood, seconds=time.perf_counter() - t0, passes=passes, n_hat=n_hat, p_any=any_s[pre_roll:], e_n=e_n, p_max=p_max,
        n_obs=int(np.isfinite(y[pre_roll:]).sum()),
    )  # fmt: skip


def f1_counts(*, truth: np.ndarray, flag: np.ndarray) -> tuple[int, int, int]:
    """Return `(tp, fp, fn)` of a boolean flag against a boolean truth."""
    return int((truth & flag).sum()), int((~truth & flag).sum()), int((truth & ~flag).sum())


def f1(tp: float, fp: float, fn: float) -> float:
    """Half-hour F1 from counts (0 when nothing is flagged or true)."""
    d = 2 * tp + fp + fn
    return 2 * tp / d if d > 0 else 0.0


N_WEEKS: Final[int] = 7
N_BOOT: Final[int] = 10_000
EXPECTED_SEGMENTS: Final[int] = 44
EXPECTED_SCHEDULES: Final[int] = 9


def weekly_counts(*, truth: np.ndarray, flag: np.ndarray) -> np.ndarray:
    """Return `(7, 3)` counts of `(tp, fp, fn)` per week of 336 half-hours, as experiment 694 counts them."""
    out = np.zeros((N_WEEKS, 3), dtype=int)
    for w in range(N_WEEKS):
        sl = slice(w * 336, min((w + 1) * 336, len(truth)))
        out[w] = f1_counts(truth=truth[sl], flag=flag[sl])
    return out


def build_tensor(
    *,
    counts: dict[tuple[str, str, str, int], np.ndarray],
    scheds: tuple[str, ...],
    segs: list[tuple[str, str, int]],
) -> np.ndarray:
    """Return `(schedule-weeks, segments, 3)` counts: the clusters are schedule-weeks, as in experiment 694."""
    x = np.zeros((len(scheds) * N_WEEKS, len(segs), 3))
    for si, sched in enumerate(scheds):
        for gi, (host, period, index) in enumerate(segs):
            x[si * N_WEEKS : (si + 1) * N_WEEKS, gi, :] = counts[(sched, host, period, index)]
    return x


def cluster_bootstrap_paired(
    *, x_a: np.ndarray, x_b: np.ndarray, n_boot: int = N_BOOT, seed: int = 720
) -> dict[str, float]:
    """Return the pooled F1 of `x_a` minus that of `x_b`, with a 95% interval from experiment 694's two-stage bootstrap.

    Each resample draws schedule-weeks (clusters) with replacement and, inside every drawn cluster,
    segments with replacement (`e00694_report.cluster_bootstrap`), and the same draw is applied to both
    arms, so the contrast is paired.

    Args:
        x_a: `(clusters, segments, 3)` counts of the first method (the hindsight model).
        x_b: The same cells for the second method (experiment 694's `M2`).
        n_boot: Number of resamples.
        seed: Random seed.

    Returns:
        A dictionary with `point` (the pooled F1 difference), `lower` and `upper` (the 95%
        interval), `f1_a`, `f1_b`, `n_clusters`, and `n_segments`.

    Raises:
        ValueError: When `x_a` and `x_b` differ in shape.
    """
    if x_a.shape != x_b.shape:
        msg = "the two methods must cover the same clusters and segments"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    n_c, n_s = x_a.shape[:2]
    both = np.stack([x_a, x_b], axis=2)  # (clusters, segments, 2, 3)
    diffs = np.empty(n_boot)
    chunk = 250
    for lo in range(0, n_boot, chunk):
        m = min(chunk, n_boot - lo)
        ci = rng.integers(n_c, size=(m, n_c))
        si = rng.integers(n_s, size=(m, n_c, n_s))
        s = both[ci[:, :, None], si].sum(axis=(1, 2))  # (m, 2, 3)
        diffs[lo : lo + m] = np.array([f1(*r[0]) - f1(*r[1]) for r in s])
    tot = both.sum(axis=(0, 1))
    return {
        "point": float(f1(*tot[0]) - f1(*tot[1])),
        "lower": float(np.percentile(diffs, 2.5)),
        "upper": float(np.percentile(diffs, 97.5)),
        "f1_a": float(f1(*tot[0])),
        "f1_b": float(f1(*tot[1])),
        "n_clusters": float(n_c),
        "n_segments": float(n_s),
    }


def false_alarm(*, flag: np.ndarray) -> dict[str, float]:
    """Return the flagged share and the active days (at least 4 flags) per 30 days of an engine-free span."""
    n_days = len(flag) // 48
    blocks = flag[: n_days * 48].reshape(n_days, 48).sum(axis=1) if n_days else np.zeros(0)
    active = float((blocks >= 4).sum())
    return {
        "flagged_share": float(flag.mean()) if len(flag) else float("nan"),
        "active_days_per_month": active / max(n_days, 1) * 30.0,
    }


def per_host_false_alarm(*, runs: list[tuple[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Pool un-injected flags per host, as experiment 694's P3 does, and return each host's share and active days a month."""
    out: dict[str, dict[str, float]] = {}
    for host in sorted({h for h, _ in runs}):
        flags = [f for h, f in runs if h == host]
        flagged, n = sum(int(f.sum()) for f in flags), sum(len(f) for f in flags)
        act = sum(
            int((f[: len(f) // 48 * 48].reshape(-1, 48).sum(axis=1) >= 4).sum()) for f in flags
        )
        days = sum(len(f) // 48 for f in flags)
        out[host] = {
            "flagged_share": flagged / max(n, 1),
            "active_days_per_month": act / max(days, 1) * 30.0,
        }
    return out


def worst_host(*, fa: dict[str, dict[str, float]]) -> dict[str, float]:
    """Return the largest flagged share and the largest active days a month over the hosts (the bar must hold at every host)."""
    return {
        "flagged_share": max(v["flagged_share"] for v in fa.values()),
        "active_days_per_month": max(v["active_days_per_month"] for v in fa.values()),
    }


def v2_verdict(
    *, x_hind: np.ndarray, x_m2: np.ndarray, fa_hindsight: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Return the V2 bar from the schedule-week tensors and the per-host un-injected false alarms of the hindsight model."""
    c = cluster_bootstrap_paired(x_a=x_hind, x_b=x_m2)
    worst = worst_host(fa=fa_hindsight)
    return {"contrast": c, "bar": bars.v2_bar(lower=c["lower"], point=c["point"], **worst)}


def consistency_gate(*, compared: int, mismatches: int, expected: int, label: str) -> None:
    """Raise `RuntimeError` unless exactly `expected` stored results were compared and every one matched.

    Zero compared results is a failure, not a pass: it means the stored keys did not line up with the rerun.
    """
    if compared != expected:
        msg = f"{label}: compared {compared} stored results, expected {expected}; the rerun is not on the stored segments"
        raise RuntimeError(msg)
    if mismatches:
        msg = f"{label}: {mismatches} of {compared} reruns of M2 differ from the stored results; stop and report"
        raise RuntimeError(msg)


def m2_counts(
    *, yy: np.ndarray, lvl: np.ndarray, units: np.ndarray, offset: int
) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover - needs experiment 694's code
    """Rerun experiment 694's `M2` on one injected segment; return its flags and its weekly counts `(7, 4)`.

    The weekly counts follow `e00694_run.inject_task` exactly, so they can be compared with the stored
    `inject.pkl`. This calls `grid_track` directly and never one of the pool-based stage functions.
    """
    import e00694_trackers as t694  # ty: ignore[unresolved-import]
    from e00694_common import f1_counts as f1_counts_694  # ty: ignore[unresolved-import]

    flag = t694.grid_track(y=yy, lvl=lvl).p_any_smooth > 0.5
    wf = flag[offset : offset + len(units)]
    wt = units > 0
    counts = np.zeros((7, 4), dtype=int)
    for w in range(7):
        sl = slice(w * 336, min((w + 1) * 336, len(units)))
        counts[w] = f1_counts_694(truth=wt[sl], flag=wf[sl])
    return flag, counts


def unit_recall(*, n_hat: np.ndarray, units: np.ndarray, k: int) -> tuple[int, int]:
    """Return `(hits, total)` of half-hours that truly hold `k` units and are given `k` units."""
    t = units == k
    return int((n_hat[t] == k).sum()), int(t.sum())


def select_k(
    *, y: np.ndarray, first_index: int, **kwargs: Any
) -> tuple[HindsightFit, dict[int, HindsightFit]]:
    """Fit `K = 0` to 3 and return the fit with the lowest Bayesian information criterion, and all four fits.

    The criterion is `-2 log L + p ln n` with `p = K (K + 1) + 1` parameters for `K > 0` and 0 for `K = 0`,
    which is experiment 694's count. `K = 3` is the forced control of V3.
    """
    fits = {k: hindsight_fit(y=y, first_index=first_index, n_max=k, **kwargs) for k in (0, 1, 2, 3)}

    def bic(k: int) -> float:
        n_par = k * (k + 1) + (1 if k > 0 else 0)
        return -2.0 * fits[k].log_likelihood + n_par * np.log(max(fits[k].n_obs, 2))

    best = min(fits, key=bic)
    return fits[best], fits


def run_v2(
    *,
    limit: int | None = None,
    cap_hours: float = 15.0,
    fit_kwargs: dict[str, Any] | None = None,
    expected_segments: int = EXPECTED_SEGMENTS,
    expected_schedules: int = EXPECTED_SCHEDULES,
) -> int:  # pragma: no cover - needs the stored panel
    """Run V2, K1, and V3 against experiment 694's stored assets (real data; waits for the code review).

    Phase 1 reruns experiment 694's `M2` on every segment (injected with every schedule, and un-injected),
    on K1, and compares each result with the stored `inject.pkl`, `fa.pkl`, and `k1.pkl`. The run stops
    before any hindsight fit when the number of compared results is not the expected one or any differs.
    Phase 2 fits the hindsight model on the same series. All outputs go under the private folder.
    The cost gate times the first hindsight fit and stops the run when the projection exceeds the cap.
    """
    import pickle

    import e00694_common as c694  # ty: ignore[unresolved-import]
    import e00694_run as r694  # ty: ignore[unresolved-import]
    import e00694_trackers as t694  # ty: ignore[unresolved-import]
    import e00720_guard as guard
    import polars as pl

    fk = fit_kwargs or {}
    r694.init()
    segs = r694.segments()
    if limit:
        segs = segs[:limit]
    names = list(r694.SCHEDULES)
    prim = tuple(n for n in names if r694.SCHEDULES[n]["primary"])
    if len(segs) != expected_segments or len(names) != expected_schedules:
        msg = f"expected {expected_segments} segments and {expected_schedules} schedules, found {len(segs)} and {len(names)}"
        raise RuntimeError(msg)
    seg_keys = [(s.host, s.period, s.index) for s in segs]
    stored = {
        (d["host"], d["period"], d["index"], d["sched"], d["rule"]): d["arms"]["M2"]["counts"]
        for d in pickle.loads(next(c694.DATA_DIR.glob("inject.pkl")).read_bytes())
    }
    stored_fa = {
        (d["host"], d["period"], d["index"]): d["arms"]["M2"]
        for d in pickle.loads(next(c694.DATA_DIR.glob("fa.pkl")).read_bytes())
    }
    # ---- phase 1: rerun M2 everywhere and compare with the stored results
    t_phase1 = time.perf_counter()
    m2_flag: dict[tuple[str, str, str, int], np.ndarray] = {}
    wk_m2: dict[tuple[str, str, str, int], np.ndarray] = {}
    inj_series: dict[tuple[str, str, str, int], np.ndarray] = {}
    compared = mismatches = 0
    for seg in segs:
        y0 = r694._segment_series(seg=seg)
        for name in names:
            sched = r694.SCHEDULES[name]
            g, units = sched["g"], sched["units"]
            drive = np.zeros(r694.TOTAL_LEN)
            lo = r694.PRE_ROLL + r694.INJECT_OFFSET
            drive[lo : lo + len(g)] = g
            yi = c694.inject_folded(y=y0, g=drive)
            lvl = c694.quantise_spread(spread=c694.causal_spread(y=yi))[r694.PRE_ROLL :]
            flag, cnt = m2_counts(
                yy=yi[r694.PRE_ROLL :], lvl=lvl, units=units, offset=r694.INJECT_OFFSET
            )
            key = (name, seg.host, seg.period, seg.index)
            m2_flag[key], wk_m2[key] = flag, cnt[:, :3]
            inj_series[key] = yi
            k_s = (seg.host, seg.period, seg.index, name, "folded")
            if k_s in stored:
                compared += 1
                mismatches += int(not np.array_equal(stored[k_s], cnt))
    consistency_gate(
        compared=compared,
        mismatches=mismatches,
        expected=len(segs) * len(names),
        label="injected runs",
    )
    fa_compared = fa_mismatch = 0
    unin_flag_m2: dict[tuple[str, str, int], np.ndarray] = {}
    for seg in segs:
        y0 = r694._segment_series(seg=seg)
        lvl0 = c694.quantise_spread(spread=c694.causal_spread(y=y0))[r694.PRE_ROLL :]
        fl = t694.grid_track(y=y0[r694.PRE_ROLL :], lvl=lvl0).p_any_smooth > 0.5
        unin_flag_m2[(seg.host, seg.period, seg.index)] = fl
        st = stored_fa.get((seg.host, seg.period, seg.index))
        if st is not None:
            fa_compared += 1
            fa_mismatch += int(int(fl.sum()) != int(st["flagged"]))
    consistency_gate(
        compared=fa_compared, mismatches=fa_mismatch, expected=len(segs), label="un-injected runs"
    )
    k1_stored = {
        d["rule"]: d["arms"]["M2"]
        for d in pickle.loads(next(c694.DATA_DIR.glob("k1.pkl")).read_bytes())
    }
    k1 = k1_series()
    k1_flag = t694.grid_track(y=k1["yy"], lvl=k1["lvl"]).p_any_smooth > 0.5
    tp, fp, fn, _ = c694.f1_counts(truth=(k1["units"] > 0)[k1["valid"]], flag=k1_flag[k1["valid"]])
    consistency_gate(
        compared=1,
        mismatches=int((tp, fp, fn) != (k1_stored["folded"]["tp"], k1_stored["folded"]["fp"], k1_stored["folded"]["fn"])),
        expected=1, label="K1",
    )  # fmt: skip
    m2_hours = (time.perf_counter() - t_phase1) / 3600.0
    # ---- phase 2: the hindsight model on the same series
    n_fits = len(segs) * len(names) + len(segs) + 1 + 9
    gate = hmm.CostGate(cap_hours=max(cap_hours - m2_hours, 0.0))
    wk_h: dict[tuple[str, str, str, int], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for seg in segs:
        for name in names:
            sched = r694.SCHEDULES[name]
            units = sched["units"]
            key = (name, seg.host, seg.period, seg.index)
            hf = hindsight_fit(
                y=inj_series[key],
                first_index=seg.start,
                pre_roll=r694.PRE_ROLL,
                gate=gate,
                projected_fits=n_fits,
                **fk,
            )
            win = slice(r694.INJECT_OFFSET, r694.INJECT_OFFSET + len(units))
            truth = units > 0
            wk_h[key] = weekly_counts(truth=truth, flag=hf.flag[win])
            ur = (
                [unit_recall(n_hat=hf.n_hat[win], units=units, k=k) for k in range(4)]
                if hf.n_hat is not None
                else []
            )
            low_conf = float("nan")
            if hf.p_max is not None and truth.any():
                low_conf = float((hf.p_max[win][truth] < 0.7).mean())
            rows.append(
                {"cluster": f"{seg.host}|{seg.period}|{seg.index}", "schedule": name, "primary": bool(sched["primary"]), "unit_size_hat": hf.unit_size, "seconds": hf.seconds,
                 "f1_hindsight": f1(*f1_counts(truth=truth, flag=hf.flag[win])), "f1_m2": f1(*f1_counts(truth=truth, flag=m2_flag[key][win])),
                 "low_confidence_running_share": low_conf,
                 **{f"recall_units_{k}": (h / t if t else float("nan")) for k, (h, t) in enumerate(ur)}}
            )  # fmt: skip
    unin_h: list[tuple[str, np.ndarray]] = []
    unin_m: list[tuple[str, np.ndarray]] = []
    for seg in segs:
        hf = hindsight_fit(
            y=r694._segment_series(seg=seg),
            first_index=seg.start,
            pre_roll=r694.PRE_ROLL,
            gate=gate,
            projected_fits=n_fits,
            **fk,
        )
        unin_h.append((seg.host, hf.flag))
        unin_m.append((seg.host, unin_flag_m2[(seg.host, seg.period, seg.index)]))
    fa_h = per_host_false_alarm(runs=unin_h)
    fa_m = per_host_false_alarm(runs=unin_m)
    x_h = build_tensor(counts=wk_h, scheds=prim, segs=seg_keys)
    x_m = build_tensor(counts=wk_m2, scheds=prim, segs=seg_keys)
    verdict = v2_verdict(x_hind=x_h, x_m2=x_m, fa_hindsight=fa_h)
    pl.DataFrame(rows).write_csv(guard.private_path(name="v2_fits.csv"))
    pl.DataFrame([{"host": h, **v} for h, v in fa_h.items()]).write_csv(
        guard.private_path(name="v2_false_alarm_hindsight_by_host.csv")
    )
    pl.DataFrame([{"host": h, **v} for h, v in fa_m.items()]).write_csv(
        guard.private_path(name="v2_false_alarm_m2_by_host.csv")
    )
    pl.DataFrame(
        [
            {
                **verdict["contrast"],
                **{f"bar_{k}": v for k, v in verdict["bar"].items()},
                "m2_reruns_compared": compared,
                "m2_reruns_mismatched": mismatches,
                "m2_phase_hours": m2_hours,
            }
        ]
    ).write_csv(guard.private_path(name="v2_bar.csv"))
    k1v3 = run_k1_v3(
        gate=gate, n_fits=n_fits, fit_kwargs=fk, k1=k1, k1_m2_f1=c694.f1_of(tp=tp, fp=fp, fn=fn)
    )
    pl.DataFrame([k1v3]).write_csv(guard.private_path(name="k1_v3.csv"))
    print(
        {k: verdict["contrast"][k] for k in ("point", "lower", "upper", "f1_a", "f1_b")},
        "bar passed:",
        verdict["bar"]["passed"],
    )
    return 0


def added_back_end_index() -> int:
    """Return the grid index where V3's second control period ends, read from the private era file.

    The date of the period's end is an era date, so it is not written in any committed file. Create
    `era_dates.json` under the private folder with the key `added_back_end` (an ISO date).
    """
    import json
    from datetime import UTC, datetime

    import e00694_common as c694  # ty: ignore[unresolved-import]
    import e00720_guard as guard

    path = guard.private_path(name="era_dates.json")
    if not path.exists():
        msg = f"create {path} with the key added_back_end (an ISO date); era dates stay out of committed files"
        raise SystemExit(msg)
    end = datetime.fromisoformat(json.loads(path.read_text())["added_back_end"]).replace(tzinfo=UTC)
    return c694.stamp(when=end)


def k1_series() -> dict[str, Any]:  # pragma: no cover - needs the stored panel
    """Return K1's injected reading `yy` (after the pre-roll), its spread level, the record's units, the valid mask, and the offsets.

    The record is placed on D8's half-hour grid exactly as `e00694_run.k1_task` does; the half-hours
    where the record has a gap hold no injection and are not scored.
    """
    import e00694_common as c694  # ty: ignore[unresolved-import]
    import e00694_run as r694  # ty: ignore[unresolved-import]

    frame = c694.load_record().sort("time")
    idx = (
        (frame["time"].dt.epoch("s").to_numpy() - int(c694.GRID_START.timestamp())) // 1800
    ).astype(int)
    t0 = int(idx[0])
    n = int(idx[-1] - t0 + 1)
    g_rec, u_rec = c694.record_arrays()
    g = np.zeros(n)
    units = np.zeros(n, dtype=int)
    valid = np.zeros(n, dtype=bool)
    g[idx - t0], units[idx - t0], valid[idx - t0] = g_rec, u_rec, True
    y = r694.SERIES["D8"][t0 - r694.PRE_ROLL : t0 + n].copy()
    drive = np.zeros(len(y))
    drive[r694.PRE_ROLL :] = g
    yi = c694.inject_folded(y=y, g=drive)
    lvl = c694.quantise_spread(spread=c694.causal_spread(y=yi))[r694.PRE_ROLL :]
    return {
        "y": yi,
        "yy": yi[r694.PRE_ROLL :],
        "lvl": lvl,
        "units": units,
        "valid": valid,
        "first_index": t0 - r694.PRE_ROLL,
    }


def run_k1_v3(
    *,
    gate: hmm.CostGate,
    n_fits: int,
    fit_kwargs: dict[str, Any] | None = None,
    k1: dict[str, Any],
    k1_m2_f1: float,
) -> dict[str, Any]:  # pragma: no cover - needs the stored panel
    """Score K1 (hindsight F1 against the rerun M2) and V3 (each of D8's two engine-free periods, scored on its own).

    V3 fits `K = 0` to 3 on each control period, picks `K` by the information criterion, and also reports
    the forced `K = 3` fit. The bar needs every period to pass for both fits.
    """
    import e00694_common as c694  # ty: ignore[unresolved-import]
    import e00694_run as r694  # ty: ignore[unresolved-import]

    kw = fit_kwargs or {}
    hf = hindsight_fit(
        y=k1["y"],
        first_index=k1["first_index"],
        pre_roll=r694.PRE_ROLL,
        gate=gate,
        projected_fits=n_fits,
        **kw,
    )
    truth = k1["units"] > 0
    tp, fp, fn = f1_counts(truth=truth[k1["valid"]], flag=hf.flag[k1["valid"]])
    k1_f1 = f1(tp, fp, fn)
    out: dict[str, Any] = {
        "k1_f1_hindsight": k1_f1,
        "k1_f1_m2_rerun": k1_m2_f1,
        **{f"k1_{k}": v for k, v in bars.k1_bar(f1=k1_f1).items()},
    }
    connected = c694.stamp(when=c694.ENGINE_CONNECTED)
    periods = {
        "before_connection": (r694.PRE_ROLL, connected),
        "added_back": (connected, added_back_end_index()),
    }
    per: dict[str, dict[str, float]] = {}
    for name, (a, b) in periods.items():
        yy = r694.SERIES["D8"][a - r694.PRE_ROLL : b]
        chosen, fits = select_k(
            y=yy,
            first_index=a - r694.PRE_ROLL,
            pre_roll=r694.PRE_ROLL,
            gate=gate,
            projected_fits=n_fits,
            **kw,
        )
        fa_sel = false_alarm(flag=chosen.flag)
        fa_forced = false_alarm(flag=fits[3].flag)
        per[name] = {
            "flagged_share": fa_sel["flagged_share"],
            "active_days_per_month": max(
                fa_sel["active_days_per_month"], fa_forced["active_days_per_month"]
            ),
            "forced_k3_flagged_share": fa_forced["flagged_share"],
        }
        out[f"v3_{name}_k_selected"] = chosen.n_max
        out[f"v3_{name}_flagged_share"] = fa_sel["flagged_share"]
        out[f"v3_{name}_forced_k3_flagged_share"] = fa_forced["flagged_share"]
        out[f"v3_{name}_active_days_per_month_selected"] = fa_sel["active_days_per_month"]
        out[f"v3_{name}_active_days_per_month_forced"] = fa_forced["active_days_per_month"]
    out["v3_passed"] = bars.v3_all_periods(periods=per)["passed"]
    return out


def _synthetic_host(
    *,
    seed: int,
    n_days: int,
    drift: float,
    engine: bool,
    pre_days: int = 28,
    spell: bool = False,
    step: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a synthetic reading, the true unit count, and the grid index of its first half-hour.

    The demand has a daily and a weekly shape, an AR(1) residual, a linear drift of `drift` MVA over the
    segment (after the pre-roll), and, with `engine`, a run of units of 1.6 MVA after the pre-roll.
    """
    rng = np.random.default_rng(seed)
    first = 7 * 48 * 3
    n = (pre_days + n_days) * 48
    idx = np.arange(n)
    h = idx % 48
    dtype = day_types(n=n, first_index=first)
    base = 6.0 + 1.5 * np.sin(2 * np.pi * (h - 14) / 48.0) * (1.0 - 0.2 * (dtype > 0))
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.9 * ar[t - 1] + rng.normal(0, 0.08)
    trend = drift * np.clip((idx - pre_days * 48) / (n_days * 48), 0.0, 1.0)
    trend = trend + step * (idx >= (pre_days + 60) * 48)
    units = np.zeros(n, dtype=int)
    if spell:
        units[(pre_days + 30) * 48 : (pre_days + 51) * 48] = 2  # two units for 21 days
    elif engine:
        state = 0
        for t in range(pre_days * 48 + 48 * 10, n):
            if rng.random() > 0.97:
                state = int(rng.integers(0, 3))
            units[t] = state
    y = (
        np.sqrt(np.maximum(base + ar + trend - 1.6 * units, 0.0) ** 2 + Q_FLOOR_MVA**2)
        + rng.standard_t(4, n) * 0.05
    )
    return y, units, first


def selftest() -> int:
    """Check the machinery on synthetic hosts with known answers, including seasonal drift of demand."""
    small: dict[str, Any] = {"coarse": (1.0, 1.6, 2.0), "n_bins": 9}
    y, units, first = _synthetic_host(seed=1, n_days=60, drift=0.0, engine=True)
    gate = hmm.CostGate(cap_hours=1.0)
    hf = hindsight_fit(y=y, first_index=first, gate=gate, projected_fits=1, n_max=2, **small)
    score = f1(*f1_counts(truth=(units > 0)[PRE_ROLL:], flag=hf.flag))
    assert score > 0.8, score
    assert abs(hf.unit_size - 1.6) <= 0.5, hf.unit_size
    assert hf.unit_size >= MIN_UNIT_MVA
    # drift of demand: engine-free segments must stay unflagged
    drift_shares = {}
    for drift in (-0.5, -1.0):
        yd, _, fd = _synthetic_host(seed=2, n_days=120, drift=drift, engine=False)
        h = hindsight_fit(y=yd, first_index=fd, **small)
        drift_shares[drift] = false_alarm(flag=h.flag)
        assert drift_shares[drift]["flagged_share"] < 0.05, (drift, drift_shares[drift])
        assert drift_shares[drift]["active_days_per_month"] <= 1.0, (drift, drift_shares[drift])
    # drift together with an engine: still found
    ye, ue, fe = _synthetic_host(seed=3, n_days=120, drift=-1.0, engine=True)
    he = hindsight_fit(y=ye, first_index=fe, **small)
    drift_f1 = f1(*f1_counts(truth=(ue > 0)[PRE_ROLL:], flag=he.flag))
    assert drift_f1 > 0.7, drift_f1
    # the trend must not absorb the engine: no drift with an engine, and a long spell of two units
    out_err = lambda h, u, off: float(np.mean(np.abs(h.unit_size * h.n_hat - 1.6 * u[off:])))  # noqa: E731
    assert out_err(hf, units, PRE_ROLL) < 0.35, out_err(hf, units, PRE_ROLL)
    spell_f1, spell_err = {}, {}
    for drift_s in (0.0, -1.0):
        ys, us, fs = _synthetic_host(seed=6, n_days=100, drift=drift_s, engine=True, spell=True)
        hs = hindsight_fit(y=ys, first_index=fs, n_max=2, **small)
        spell_f1[drift_s] = f1(*f1_counts(truth=(us > 0)[PRE_ROLL:], flag=hs.flag))
        spell_err[drift_s] = out_err(hs, us, PRE_ROLL)
        assert spell_f1[drift_s] > 0.95, (drift_s, spell_f1[drift_s])
        assert spell_err[drift_s] < 0.35, (drift_s, spell_err[drift_s])
    # a high-duty random schedule (about 66%) on the full coarse grid: the unit size must come back, not a smaller alias
    yh, uh, fh = _synthetic_host(seed=7, n_days=148, drift=-1.0, engine=True)
    hh = hindsight_fit(y=yh, first_index=fh, n_max=2, n_bins=9)
    assert 0.6 < (uh > 0)[PRE_ROLL:].mean() < 0.8, (uh > 0)[PRE_ROLL:].mean()
    assert abs(hh.unit_size - 1.6) <= 0.3, hh.unit_size
    assert out_err(hh, uh, PRE_ROLL) < 0.35, out_err(hh, uh, PRE_ROLL)
    # a step of demand smaller than a unit must not trip the V3 bars
    yst, _, fst = _synthetic_host(seed=7, n_days=120, drift=0.0, engine=False, step=-0.7)
    hst = hindsight_fit(y=yst, first_index=fst, **small)
    step_fa = false_alarm(flag=hst.flag)
    assert step_fa["flagged_share"] <= 0.03 and step_fa["active_days_per_month"] <= 1.0, step_fa
    # the rate limiter follows a slow path and refuses a fast one
    ramp = np.linspace(0.0, -1.0, 120)
    assert np.allclose(rate_limited(a=ramp, max_step=MAX_TREND_SLOPE), ramp, atol=1e-6)
    spell_like = np.r_[np.zeros(40), np.full(21, -3.0), np.zeros(59)]
    assert rate_limited(a=spell_like, max_step=MAX_TREND_SLOPE).min() > -0.7
    # K by the information criterion: none on an engine-free series, at least one with an engine
    yq, _, fq = _synthetic_host(seed=4, n_days=40, drift=0.0, engine=False)
    assert select_k(y=yq, first_index=fq, **small)[0].n_max == 0
    chosen, _ = select_k(y=y, first_index=first, **small)
    assert chosen.n_max >= 1, chosen.n_max
    # schedule-week bootstrap: better counts have a lower bound above 0, identical counts span 0
    rng = np.random.default_rng(0)
    xa = np.stack(
        [[rng.integers(55, 65, 3) * np.array([1, 0.1, 0.15]) for _ in range(8)] for _ in range(35)]
    )
    xb = np.stack(
        [[rng.integers(45, 55, 3) * np.array([1, 0.2, 0.4]) for _ in range(8)] for _ in range(35)]
    )
    c = cluster_bootstrap_paired(x_a=xa, x_b=xb, n_boot=1000)
    assert c["lower"] > 0.0 and c["point"] > 0.05, c
    same = cluster_bootstrap_paired(x_a=xa, x_b=xa, n_boot=1000)
    assert same["lower"] <= 0.0 <= same["upper"]
    assert bars.v2_bar(
        lower=c["lower"], point=c["point"], flagged_share=0.01, active_days_per_month=0.5
    )["passed"]
    assert not bars.v2_bar(lower=-0.01, point=0.2, flagged_share=0.01, active_days_per_month=0.5)[
        "passed"
    ]
    # the stored-result gate: exact match passes; a mismatch or zero compared results stops the run
    consistency_gate(compared=396, mismatches=0, expected=396, label="t")
    for kwargs in ({"compared": 396, "mismatches": 1}, {"compared": 0, "mismatches": 0}):
        try:
            consistency_gate(expected=396, label="t", **kwargs)
        except RuntimeError:
            continue
        msg = f"the gate let {kwargs} through"
        raise AssertionError(msg)
    fa = per_host_false_alarm(
        runs=[("A", np.zeros(480, dtype=bool)), ("B", np.ones(480, dtype=bool))]
    )
    assert fa["A"]["flagged_share"] == 0.0
    assert fa["B"]["active_days_per_month"] == 30.0
    assert worst_host(fa=fa)["flagged_share"] == 1.0
    # the cost gate refuses an over-cap projection
    tiny = hmm.CostGate(cap_hours=1e-9)
    try:
        hindsight_fit(
            y=y[:600],
            first_index=first,
            pre_roll=300,
            gate=tiny,
            projected_fits=100,
            n_max=1,
            coarse=(1.5,),
            n_bins=9,
        )
    except RuntimeError:
        pass
    else:
        msg = "the cost gate did not stop the fit"
        raise AssertionError(msg)
    print(
        f"PASS V2 machinery: F1 {score:.3f} with an engine; drift of -0.5 and -1.0 MVA flags {drift_shares[-0.5]['flagged_share']:.3f} and {drift_shares[-1.0]['flagged_share']:.3f} of engine-free half-hours, F1 {drift_f1:.3f} with an engine and drift; spell of two units F1 {spell_f1[0.0]:.3f} and {spell_f1[-1.0]:.3f} (output error {spell_err[0.0]:.2f} and {spell_err[-1.0]:.2f} MVA) without and with drift; a 0.7 MVA step flags {step_fa['flagged_share']:.3f}; K by BIC; schedule-week bootstrap; consistency gate; cost gate"
    )
    return 0


def selftest_694() -> int:
    """Run `run_v2` end to end against stubbed experiment 694 assets (tiny synthetic series, no real data).

    The stub replaces the panel, the schedules, the engine record, the stored pickles, and the private
    folder, so `run_v2` and `run_k1_v3` execute every line on a series of about 70 days with a reduced
    residual grid. A first run with a deliberately wrong stored count must stop before any hindsight fit.
    A second run with the correct stored results must write every output.
    """
    import pickle
    import tempfile
    from datetime import timedelta
    from pathlib import Path

    import e00694_common as c694  # ty: ignore[unresolved-import]
    import e00694_run as r694  # ty: ignore[unresolved-import]
    import e00694_trackers as t694  # ty: ignore[unresolved-import]
    import e00720_guard as guard
    import polars as pl

    tmp = Path(tempfile.mkdtemp(prefix="e720_stub_", dir=str(Path.home() / ".cache")))
    guard.PRIVATE_DIR = tmp / "private"
    c694.DATA_DIR = tmp / "d694"
    c694.DATA_DIR.mkdir()
    pre, seg_len, offset, n_inj = 8 * 48, 24 * 48, 3 * 48, 7 * 48
    r694.PRE_ROLL, r694.SEGMENT_LEN, r694.TOTAL_LEN, r694.INJECT_OFFSET = (
        pre,
        seg_len,
        pre + seg_len,
        offset,
    )
    rng = np.random.default_rng(11)
    n_total = 70 * 48
    hh = np.arange(n_total) % 48
    host = 6.0 + 1.2 * np.sin(2 * np.pi * (hh - 14) / 48.0) + rng.normal(0, 0.08, n_total)
    r694.SERIES.clear()
    r694.SERIES.update(
        {"D3": np.sqrt(host**2 + Q_FLOOR_MVA**2), "D8": np.sqrt((host + 3.0) ** 2 + Q_FLOOR_MVA**2)}
    )
    real_spread = c694.causal_spread
    c694.causal_spread = lambda *, y: (
        np.full(len(y), 0.08) if len(y) < 28 * 48 else real_spread(y=y)
    )  # a 28-day window is longer than K1's stub series
    r694.init = lambda: None
    seg = r694.Segment(host="D3", period="pre-solar", index=0, start=0)
    r694.segments = lambda: [seg]
    units = np.zeros(n_inj, dtype=int)
    units[100:160], units[200:260], units[260:300] = 1, 2, 3
    synth = np.roll(units, 40)
    r694.SCHEDULES.clear()
    r694.SCHEDULES.update(
        {
            "record": {"units": units, "g": units * 1.6, "primary": True, "k": 3, "u": 1.6},
            "synth0": {"units": synth, "g": synth * 1.6, "primary": False, "k": 3, "u": 1.6},
        }
    )
    y0 = r694._segment_series(seg=seg)
    stored: list[dict[str, Any]] = []
    for name in r694.SCHEDULES:
        sch = r694.SCHEDULES[name]
        drive = np.zeros(r694.TOTAL_LEN)
        drive[pre + offset : pre + offset + n_inj] = sch["g"]
        yi = c694.inject_folded(y=y0, g=drive)
        lvl = c694.quantise_spread(spread=c694.causal_spread(y=yi))[pre:]
        _, counts = m2_counts(yy=yi[pre:], lvl=lvl, units=sch["units"], offset=offset)
        stored.append(
            {
                "host": "D3",
                "period": "pre-solar",
                "index": 0,
                "sched": name,
                "rule": "folded",
                "arms": {"M2": {"counts": counts}},
            }
        )
    lvl0 = c694.quantise_spread(spread=c694.causal_spread(y=y0))[pre:]
    fl0 = t694.grid_track(y=y0[pre:], lvl=lvl0).p_any_smooth > 0.5
    fa_stored = [
        {
            "host": "D3",
            "period": "pre-solar",
            "index": 0,
            "arms": {"M2": {"flagged": int(fl0.sum())}},
        }
    ]
    start = c694.GRID_START
    rec = pl.DataFrame(
        {
            "time": [start + timedelta(minutes=30 * (20 * 48 + i)) for i in range(n_inj)],
            "power": units * 1.6,
            "units": units,
        }
    )
    c694.load_record = lambda: rec
    c694.record_arrays = lambda: (units * 1.6, units)
    c694.ENGINE_CONNECTED = start + timedelta(days=30)
    guard.PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    (guard.PRIVATE_DIR / "era_dates.json").write_text(
        '{"added_back_end": "' + (start + timedelta(days=45)).date().isoformat() + '"}'
    )
    k1 = k1_series()
    k1_fl = t694.grid_track(y=k1["yy"], lvl=k1["lvl"]).p_any_smooth > 0.5
    tp, fp, fn, tn = c694.f1_counts(truth=(k1["units"] > 0)[k1["valid"]], flag=k1_fl[k1["valid"]])
    k1_stored = [{"rule": "folded", "arms": {"M2": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}}]

    def write_stored(*, wrong: bool) -> None:
        bad = [
            dict(
                d,
                arms={
                    "M2": {"counts": d["arms"]["M2"]["counts"] + (1 if (wrong and i == 1) else 0)}
                },
            )
            for i, d in enumerate(stored)
        ]
        (c694.DATA_DIR / "inject.pkl").write_bytes(pickle.dumps(bad))
        (c694.DATA_DIR / "fa.pkl").write_bytes(pickle.dumps(fa_stored))
        (c694.DATA_DIR / "k1.pkl").write_bytes(pickle.dumps(k1_stored))

    kw: dict[str, Any] = {"coarse": (1.2, 1.6, 2.0), "n_bins": 9}
    write_stored(wrong=True)
    try:
        run_v2(cap_hours=1.0, fit_kwargs=kw, expected_segments=1, expected_schedules=2)
    except RuntimeError as err:
        assert "differ from the stored" in str(err), err
        assert not (guard.PRIVATE_DIR / "v2_bar.csv").exists(), (
            "the bar was written after a mismatch"
        )
    else:
        msg = "a wrong stored count did not stop run_v2"
        raise AssertionError(msg)
    write_stored(wrong=False)
    code = run_v2(cap_hours=1.0, fit_kwargs=kw, expected_segments=1, expected_schedules=2)
    out = {p.name for p in guard.PRIVATE_DIR.iterdir()}
    assert {
        "v2_fits.csv",
        "v2_bar.csv",
        "k1_v3.csv",
        "v2_false_alarm_hindsight_by_host.csv",
    } <= out, out
    bar = pl.read_csv(guard.PRIVATE_DIR / "v2_bar.csv")
    assert bar["m2_reruns_compared"][0] == 2 and bar["m2_reruns_mismatched"][0] == 0
    k = pl.read_csv(guard.PRIVATE_DIR / "k1_v3.csv")
    assert {
        "v3_passed",
        "k1_f1_hindsight",
        "k1_f1_m2_rerun",
        "v3_before_connection_k_selected",
    } <= set(k.columns)
    print(
        f"PASS V2, K1, and V3 harness against stubbed 694 assets: a wrong stored count stops the run before any fit; the correct run writes {sorted(out)}"
    )
    return code


def main() -> int:
    """Parse arguments and run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest-694", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.selftest_694:
        return selftest_694()
    if args.run:
        return run_v2(limit=args.limit)
    ap.error("give --selftest or --run")
    return 2


if __name__ == "__main__":
    sys.exit(main())
