# ruff: noqa: E501
"""Experiment 514: the four-signal candidate scan and its features (pure numpy, no fitting).

`scan_series` finds candidate pulses in one scaled residual series: closed pulses (a step in, a
plateau, a step back toward the earlier level) at 10 durations from 1 hour to 14 days, and open
steps (no return observed yet) that started within the last 14 days. `candidate_features` adds the
partner signals (synchronised opposite-sign edges at the start and end, and the symmetry of the
sizes) and the shape correlation. Everything reads only values below `end`.
"""

from dataclasses import dataclass
from typing import Final

import e00514_data as d514  # ty: ignore[unresolved-import]
import numpy as np

DURATIONS: Final[tuple[int, ...]] = (2, 4, 8, 16, 32, 64, 128, 256, 512, 672)
"""Pulse durations in half-hours: 1 hour to 14 days."""
FLANK_MIN: Final[int] = 12
FLANK_MAX: Final[int] = 336
ADMIT_SCORE: Final[float] = 2.5
"""Candidates need a score of at least this many noise units."""
SUPPRESS_OVERLAP: Final[float] = 0.5
"""A candidate is dropped if more than this share of its span is covered by a kept candidate."""
MAX_CANDIDATES: Final[int] = 4_000
OPEN_MAX: Final[int] = 672
OPEN_MIN: Final[int] = 8
EDGE_TOLERANCE: Final[int] = 2
"""Half-hours either side of a pulse edge in which a partner edge counts as synchronised."""
THRESHOLDS: Final[tuple[float, float]] = (3.0, 5.0)
SYMMETRY_TOLERANCE: Final[float] = 2.5
SHAPE_MIN_STEPS: Final[int] = 24
EXCLUDED_PARTNERS: Final[tuple[int, ...]] = (10, 12)
"""D15 and D18 (Primary indices 10 and 12) are never partners: their history is too short in most worlds."""
FEATURES: Final[tuple[str, ...]] = (
    "log_score", "log_duration", "open", "return_ratio", "edge_asymmetry", "amplitude",
    "after_fraction", "start_opp3", "start_opp5", "end_opp3", "both_opp3", "start_same3",
    "end_same3", "has_pair", "symmetry", "shape_corr",
)  # fmt: skip


@dataclass
class Scales:
    """Noise scales of one fold, fixed from the history before the fold's refit."""

    sigma: np.ndarray
    """(20, len(DURATIONS)) robust sd of the in-minus-before step, per series and duration."""
    spread: np.ndarray
    """(20,) robust sd of the series (scaled units)."""
    edge_sd: np.ndarray
    """(20,) robust sd of the partner edge statistic."""
    mw_per_z: np.ndarray
    """(20,) MW per unit of each series."""


def _prefix(*, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return prefix sums of the finite values and of their count."""
    ok = np.isfinite(x)
    return np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))]), np.concatenate(
        [[0], np.cumsum(ok)]
    )


def _mean(
    *, cs: np.ndarray, cn: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and the count of finite values in [lo, hi) with indices clipped to the series."""
    n = len(cs) - 1
    lo = np.clip(lo, 0, n)
    hi = np.clip(hi, 0, n)
    count = cn[hi] - cn[lo]
    total = cs[hi] - cs[lo]
    return np.where(count > 0, total / np.maximum(count, 1), np.nan), count


def flank_of(*, d: int) -> int:
    """Return the flank length in half-hours for a duration."""
    return int(np.clip(d, FLANK_MIN, FLANK_MAX))


def window_stats(*, x: np.ndarray, d: int) -> dict[str, np.ndarray]:
    """Return, for every start `a`, the plateau mean, the two flank means, and the flank counts."""
    cs, cn = _prefix(x=x)
    n = len(x)
    f = flank_of(d=d)
    a = np.arange(n)
    m_in, c_in = _mean(cs=cs, cn=cn, lo=a, hi=a + d)
    m_b, c_b = _mean(cs=cs, cn=cn, lo=a - f, hi=a)
    m_a, c_a = _mean(cs=cs, cn=cn, lo=a + d, hi=a + d + f)
    ok_in = (c_in >= 0.5 * d) & (a + d <= n)
    ok_b = (c_b >= 0.5 * f) & (a - f >= 0)
    min_after = max(4, min(f, 48) // 2)
    ok_a = (c_a >= min_after) & (a + d < n)
    return {
        "m_in": m_in, "m_b": m_b, "m_a": m_a, "ok_in": ok_in, "ok_b": ok_b, "ok_a": ok_a,
        "after_fraction": np.minimum(c_a / f, 1.0),
    }  # fmt: skip


def fit_sigma(*, x: np.ndarray, hist_end: int) -> np.ndarray:
    """Return the robust sd of the in-minus-before step at each duration, from rows below `hist_end`."""
    sub = x[:hist_end]
    out = np.full(len(DURATIONS), np.nan)
    for k, dur in enumerate(DURATIONS):
        s = window_stats(x=sub, d=dur)
        step = (s["m_in"] - s["m_b"])[s["ok_in"] & s["ok_b"]]
        out[k] = d514.robust_sd(x=step)
    return out


def fit_scales(*, matrix: np.ndarray, mw_per_z: np.ndarray, hist_end: int) -> Scales:
    """Fit the noise scales of all 20 series from rows below `hist_end`."""
    sigma = np.stack([fit_sigma(x=matrix[i], hist_end=hist_end) for i in range(20)])
    spread = np.array([d514.robust_sd(x=matrix[i, :hist_end]) for i in range(20)])
    edge_sd = np.array(
        [d514.robust_sd(x=d514.edge_statistic(x=matrix[i, :hist_end])) for i in range(20)]
    )
    return Scales(sigma=sigma, spread=spread, edge_sd=edge_sd, mw_per_z=mw_per_z)


@dataclass
class Candidates:
    """Candidate pulses of one series. Every array has one entry per candidate."""

    start: np.ndarray
    duration: np.ndarray
    is_open: np.ndarray
    score: np.ndarray
    delta_in: np.ndarray
    delta_out: np.ndarray
    after_fraction: np.ndarray

    def __len__(self) -> int:
        """Return the number of candidates."""
        return len(self.start)


def scan_series(*, x: np.ndarray, sigma: np.ndarray, end: int, lo: int = 0) -> Candidates:
    """Return the thinned candidate pulses of one series.

    Args:
        x: Scaled residual, NaN where unavailable and from `end` on.
        sigma: This series' step noise per duration.
        end: First index that is not observed.
        lo: First index a candidate may start at.

    Returns:
        The candidates, thinned by greedy suppression.
    """
    starts, durs, opens, scores, d1s, d2s, afs = [], [], [], [], [], [], []
    for k, dur in enumerate(DURATIONS):
        s = window_stats(x=x, d=dur)
        d1 = s["m_in"] - s["m_b"]
        d2 = s["m_in"] - s["m_a"]
        closed = s["ok_in"] & s["ok_b"] & s["ok_a"] & (np.sign(d1) == np.sign(d2))
        score = np.where(closed, np.minimum(np.abs(d1), np.abs(d2)) / sigma[k], 0.0)
        score[:lo] = 0.0
        idx = np.flatnonzero(np.nan_to_num(score) >= ADMIT_SCORE)
        starts.append(idx)
        durs.append(np.full(len(idx), dur))
        opens.append(np.zeros(len(idx), dtype=bool))
        scores.append(score[idx])
        d1s.append(d1[idx])
        d2s.append(d2[idx])
        afs.append(s["after_fraction"][idx])
    # Open steps: plateau from `a` to `end`, flank before only.
    cs, cn = _prefix(x=x)
    a = np.arange(max(lo, end - OPEN_MAX), end - OPEN_MIN)
    if len(a):
        dur = end - a
        f = np.clip(dur, FLANK_MIN, FLANK_MAX)
        m_in, c_in = _mean(cs=cs, cn=cn, lo=a, hi=np.full(len(a), end))
        m_b, c_b = _mean(cs=cs, cn=cn, lo=a - f, hi=a)
        ok = (c_in >= 0.5 * dur) & (c_b >= 0.5 * f) & (a - f >= 0)
        sig = np.exp(np.interp(np.log(dur), np.log(DURATIONS), np.log(np.maximum(sigma, 1e-9))))
        score = np.where(ok, np.abs(m_in - m_b) / sig, 0.0)
        idx = np.flatnonzero(np.nan_to_num(score) >= ADMIT_SCORE)
        starts.append(a[idx])
        durs.append(dur[idx])
        opens.append(np.ones(len(idx), dtype=bool))
        scores.append(score[idx])
        d1s.append((m_in - m_b)[idx])
        d2s.append(np.zeros(len(idx)))
        afs.append(np.zeros(len(idx)))
    cat = lambda parts: np.concatenate(parts) if parts else np.array([])  # noqa: E731
    start, dur_a, is_open, score_a = (
        cat(starts).astype(np.int64),
        cat(durs).astype(np.int64),
        cat(opens).astype(bool),
        cat(scores),
    )
    d1_a, d2_a, af_a = cat(d1s), cat(d2s), cat(afs)
    order = np.argsort(-score_a, kind="stable")[: MAX_CANDIDATES * 3]
    covered = np.zeros(len(x) + 1, dtype=bool)
    keep: list[int] = []
    for j in order:
        a0, a1 = int(start[j]), int(min(start[j] + dur_a[j], len(x)))
        if a1 <= a0:
            continue
        if covered[a0:a1].mean() > SUPPRESS_OVERLAP:
            continue
        keep.append(int(j))
        covered[a0:a1] = True
        if len(keep) >= MAX_CANDIDATES:
            break
    sel = np.array(keep, dtype=np.int64)
    return Candidates(
        start=start[sel], duration=dur_a[sel], is_open=is_open[sel], score=score_a[sel],
        delta_in=d1_a[sel], delta_out=d2_a[sel], after_fraction=af_a[sel],
    )  # fmt: skip


def shape_profile(*, power: np.ndarray) -> np.ndarray:
    """Return the median of the same half-hour in the 4 earlier weeks (causal demand shape)."""
    n = len(power)
    past = np.stack(
        [
            np.concatenate([np.full(k * d514.WEEK, np.nan), power[: n - k * d514.WEEK]])
            for k in range(1, 5)
        ]
    )
    with np.errstate(all="ignore"):
        return np.nanmedian(past, axis=0)


def edge_matrix(*, matrix: np.ndarray, scales: Scales) -> np.ndarray:
    """Return the (20, n) partner edge statistic in units of each series' own edge noise."""
    out = np.stack(
        [d514.edge_statistic(x=matrix[i]) / scales.edge_sd[i] for i in range(matrix.shape[0])]
    )
    out[list(EXCLUDED_PARTNERS)] = np.nan
    return out


def candidate_features(
    *, cands: Candidates, series: int, matrix: np.ndarray, edges: np.ndarray, scales: Scales,
    shape: np.ndarray | None,
) -> np.ndarray:  # fmt: skip
    """Return the (n_candidates, len(FEATURES)) feature matrix of one series' candidates."""
    n_c = len(cands)
    out = np.zeros((n_c, len(FEATURES)))
    if n_c == 0:
        return out
    others = np.array([q for q in range(matrix.shape[0]) if q != series])
    cs = [_prefix(x=matrix[q]) for q in range(matrix.shape[0])]
    for j in range(n_c):
        a, dur = int(cands.start[j]), int(cands.duration[j])
        is_open = bool(cands.is_open[j])
        d1, d2 = float(cands.delta_in[j]), float(cands.delta_out[j])
        sign = 1.0 if d1 >= 0 else -1.0
        end_t = a + dur
        # partner edges: opposite-sign at the start, and at the end for a closed pulse
        lo_s, hi_s = max(a - EDGE_TOLERANCE, 0), a + EDGE_TOLERANCE + 1
        e_start = edges[others, lo_s:hi_s]
        with np.errstate(all="ignore"):
            opp_s = np.nanmax(np.where(np.isfinite(e_start), -sign * e_start, -np.inf), axis=1)
            same_s = np.nanmax(np.where(np.isfinite(e_start), sign * e_start, -np.inf), axis=1)
        if is_open:
            opp_e = np.full(len(others), -np.inf)
            same_e = np.full(len(others), -np.inf)
        else:
            e_end = edges[others, max(end_t - EDGE_TOLERANCE, 0) : end_t + EDGE_TOLERANCE + 1]
            if e_end.shape[1] == 0:
                opp_e = np.full(len(others), -np.inf)
                same_e = np.full(len(others), -np.inf)
            else:
                with np.errstate(all="ignore"):
                    opp_e = np.nanmax(np.where(np.isfinite(e_end), sign * e_end, -np.inf), axis=1)
                    same_e = np.nanmax(np.where(np.isfinite(e_end), -sign * e_end, -np.inf), axis=1)
        t3, t5 = THRESHOLDS
        both = (opp_s >= t3) & (opp_e >= t3)
        # symmetry of sizes in MW over the pulse, for partners with both edges
        symmetry, has_pair = 0.0, 0.0
        if both.any():
            f = flank_of(d=dur)
            m_d = (
                _mean(cs=cs[series][0], cn=cs[series][1], lo=np.array([a]), hi=np.array([end_t]))[0]
                - _mean(cs=cs[series][0], cn=cs[series][1], lo=np.array([a - f]), hi=np.array([a]))[
                    0
                ]
            )
            donor_mw = float(m_d[0]) * scales.mw_per_z[series]
            partner_mw = 0.0
            for q in others[both]:
                m_q = (
                    _mean(cs=cs[q][0], cn=cs[q][1], lo=np.array([a]), hi=np.array([end_t]))[0]
                    - _mean(cs=cs[q][0], cn=cs[q][1], lo=np.array([a - f]), hi=np.array([a]))[0]
                )
                partner_mw += float(np.nan_to_num(m_q[0])) * scales.mw_per_z[q]
            if np.isfinite(donor_mw) and abs(donor_mw) > 1e-9:
                ratio = float(np.clip(-partner_mw / donor_mw, 0.05, 20.0))
                symmetry, has_pair = abs(np.log(ratio)), 1.0
        shape_corr = 0.0
        if shape is not None and dur >= SHAPE_MIN_STEPS and not is_open:
            seg = matrix[series, a:end_t]
            prof = shape[a:end_t]
            ok = np.isfinite(seg) & np.isfinite(prof)
            if ok.sum() >= SHAPE_MIN_STEPS // 2 and np.std(seg[ok]) > 0 and np.std(prof[ok]) > 0:
                shape_corr = float(np.corrcoef(seg[ok], prof[ok])[0, 1])
        amp = (abs(d1) if is_open else 0.5 * (abs(d1) + abs(d2))) / scales.spread[series]
        out[j] = [
            np.log1p(cands.score[j]), np.log(dur), float(is_open),
            _return_ratio(d1=d1, d2=d2, is_open=is_open),
            0.0 if is_open else abs(np.log(max(abs(d1), 1e-9) / max(abs(d2), 1e-9))),
            amp, float(cands.after_fraction[j]),
            float((opp_s >= t3).sum()), float((opp_s >= t5).sum()), float((opp_e >= t3).sum()),
            float(both.sum()), float((same_s >= t3).sum()), float((same_e >= t3).sum()),
            has_pair, symmetry, shape_corr,
        ]  # fmt: skip
    return out


def _return_ratio(*, d1: float, d2: float, is_open: bool) -> float:
    """Return |after - before| / min(|d1|, |d2|); 1 for an open step (no return observed)."""
    if is_open:
        return 1.0
    return float(abs(d1 - d2) / max(min(abs(d1), abs(d2)), 1e-9))


def row_probability(*, cands: Candidates, p_cand: np.ndarray, n: int) -> np.ndarray:
    """Return the largest candidate probability that covers each of the `n` rows (-1 if none)."""
    p = np.full(n, -1.0)
    for j in np.argsort(p_cand):
        a, b = int(cands.start[j]), int(min(cands.start[j] + cands.duration[j], n))
        p[a:b] = np.maximum(p[a:b], p_cand[j])
    return p
