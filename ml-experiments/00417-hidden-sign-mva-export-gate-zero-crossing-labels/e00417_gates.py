# ruff: noqa: E501
"""The Primary-level gate (design A), the high-irradiance row gate (H), and the labels (design C).

Everything is computed from a fold's training-window rows only (rows up to the refit minus the
cutoff age, inside the 730-day window), so no test-month data enters a gate or a label.

- **Primary gate** (experiment 366's rule as experiment 408 fits it): a Primary's gate is open when
  the Spearman correlation of the reading with the scaled CAMS unit shape `u`, on summer-midday
  (UTC 10 to 14, April to September) rows with clear-sky index `k >= 0.7`, exceeds +0.3, or the 1st
  percentile of those readings is below 3 points of capacity (1 point above the reactive floor,
  because every folded reading is at least 2 points). A shut gate closes the folded density at every
  row of the Primary.
- **High-irradiance gate**: `u_hi(h) = D_ref(h) / C_max`, where `D_ref(h)` is the median reading at
  UTC hour `h` on April-to-September training rows with `k < 0.3`, and `C_max` is the largest
  studies-page capacity, in points of capacity, among the signed-MW Primaries that are not hidden.
  The folded density applies only where `u >= max(0.3, u_hi(h))`.
- **Labels**: the W-shape labeller of `e00417_diagnostics`, with its threshold set to the 99th
  percentile of the W-shape score over the no-injection clear days of the visible signed Primaries.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Final

import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_pv as pv  # ty: ignore[unresolved-import]
import e00417_day_flip as dayflip
import e00417_diagnostics as diag
import e00417_inputs as inputs
import numpy as np
from scipy.stats import spearmanr

GATE_CORRELATION: Final[float] = 0.3
GATE_PERCENTILE_POINTS: Final[float] = 3.0
GATE_K_MIN: Final[float] = 0.7
OVERCAST_K_MAX: Final[float] = 0.3
MIN_GATE_ROWS: Final[int] = 200
MIN_REFERENCE_ROWS: Final[int] = 50
GATE_U: Final[float] = 0.3
LABEL_THRESHOLD_PERCENTILE: Final[float] = 99.0


@dataclass
class PrimaryGate:
    """One Primary's gate decision with the two statistics behind it."""

    site: str
    open: bool
    spearman: float
    p1_points: float
    n_rows: int


def _unique_rows(*, fold: inputs.Fold, primary: int) -> tuple[np.ndarray, ...]:
    """Return the Primary's valid time, reading (points), `u`, and `k`, one row per valid time."""
    sel = fold.primary == primary
    valid, first = np.unique(fold.valid[sel], return_index=True)
    idx = np.flatnonzero(sel)[first]
    return valid, fold.y[idx].astype(np.float64) * 100.0, fold.gate_u[idx], fold.k[idx]


def _summer(*, valid_us: np.ndarray) -> np.ndarray:
    _, month = inputs.hour_and_month(valid_us=valid_us)
    return (month >= 4) & (month <= 9)


def primary_gates(*, fold: inputs.Fold) -> dict[str, PrimaryGate]:
    """Return the gate decision of every folded Primary from the fold's training rows."""
    table = base.primaries()
    out = {}
    for row in table.iter_rows(named=True):
        site = row["site"]
        if not fold.folded[row["primary"]]:
            continue
        valid, r, u, k = _unique_rows(fold=fold, primary=row["primary"])
        sunny = inputs.is_midday(valid_us=valid) & (k >= GATE_K_MIN)
        n = int(sunny.sum())
        if n < MIN_GATE_ROWS:
            out[site] = PrimaryGate(site, False, float("nan"), float("nan"), n)
            continue
        rho = float(spearmanr(r[sunny], u[sunny]).statistic)
        p1 = float(np.percentile(r[sunny], 1.0))
        out[site] = PrimaryGate(
            site, bool(rho > GATE_CORRELATION or p1 < GATE_PERCENTILE_POINTS), rho, p1, n
        )
    return out


def c_max_points(*, refit, pair: tuple[str, ...]) -> float:  # noqa: ANN001
    """Return the largest studies-page capacity, in points of effective capacity, among visible signed Primaries."""
    cell = int(np.clip(pv.latest_complete_label(refit=refit) - 2023, 0, 2))
    best = 0.0
    for row in base.primaries().iter_rows(named=True):
        if row["site"] in inputs.SIGNED_SIX and row["site"] not in pair:
            centre, _sd = pv.studies_prior(site=row["site"], n_cells=3, refit=refit)
            best = max(best, 100.0 * float(centre[cell]) / float(row["effective_capacity_mw"]))
    return best


def high_irradiance_thresholds(*, fold: inputs.Fold, refit, pair: tuple[str, ...]) -> np.ndarray:  # noqa: ANN001
    """Return (n_primaries, 24): the minimum `u` at which each Primary's folded density applies."""
    c_max = c_max_points(refit=refit, pair=pair)
    n = len(fold.target_mean)
    out = np.full((n, 24), GATE_U)
    for p in range(n):
        if not fold.folded[p]:
            continue
        valid, r, _u, k = _unique_rows(fold=fold, primary=p)
        hour, _ = inputs.hour_and_month(valid_us=valid)
        overcast = (k < OVERCAST_K_MAX) & _summer(valid_us=valid)
        for h in range(24):
            sel = overcast & (hour == h)
            if sel.sum() >= MIN_REFERENCE_ROWS:
                out[p, h] = max(GATE_U, float(np.median(r[sel])) / c_max)
    return out


def open_mask(
    *,
    fold: inputs.Fold,
    primary: np.ndarray,
    gate_u: np.ndarray,
    valid_us: np.ndarray,
    mode: str,
    gates: dict[str, PrimaryGate] | None,
    thresholds: np.ndarray | None,
) -> np.ndarray:
    """Return where the folded density applies, for rows of any Primaries.

    Args:
        fold: The fold (for the Primary names and folded flags).
        primary: Primary index of each row.
        gate_u: The real scaled unit shape of each row.
        valid_us: The valid time of each row.
        mode: `none` (experiment 412's rule), `primary` (design A), or `primary+hi` (A with H).
        gates: The Primary gates (for `primary` and `primary+hi`).
        thresholds: The high-irradiance thresholds (for `primary+hi`).
    """
    open_ = gate_u >= GATE_U
    if mode == "none":
        return open_
    assert gates is not None
    names = base.primaries()["site"].to_list()
    primary_open = np.array(
        [gates[names[p]].open if names[p] in gates else True for p in range(len(names))]
    )
    open_ &= primary_open[primary]
    if mode == "primary+hi":
        assert thresholds is not None
        hour, _ = inputs.hour_and_month(valid_us=valid_us)
        open_ &= gate_u >= thresholds[primary, hour]
    return open_


def labels_for_fold(*, fold: inputs.Fold, refit) -> tuple[np.ndarray, dict]:  # noqa: ANN001
    """Return the zero-crossing label of each fold row (-1, 0, +1) and the threshold and counts."""
    window_lo = int((refit - timedelta(days=base.WINDOW_DAYS)).timestamp() * 1_000_000)
    window_hi = int((refit - regimes.REFIT_CUTOFF_AGE).timestamp() * 1_000_000)
    placebo = inputs.Setting(0.0, fold.setting.q_points, fold.setting.power_factor)
    visible = [s for s in inputs.SIGNED_SIX if s not in fold.pair]
    scores = np.concatenate(
        [
            diag.clear_days(
                t=diag.load_table(site=s, setting=placebo, start_us=window_lo, end_us=window_hi + 1)
            ).score
            for s in visible
        ]
    )
    threshold = float(np.percentile(scores, LABEL_THRESHOLD_PERCENTILE))
    label = np.zeros(len(fold.y), dtype=np.int8)
    info: dict = {"threshold_points": threshold, "n_visible_days": len(scores), "sites": {}}
    names = base.primaries()
    for row in names.iter_rows(named=True):
        if not fold.folded[row["primary"]]:
            continue
        site = row["site"]
        t = diag.load_table(
            site=site, setting=fold.setting, start_us=window_lo, end_us=window_hi + 1
        )
        sc = diag.clear_days(t=t)
        flagged = sc.score > threshold
        sel = np.flatnonzero(fold.primary == row["primary"])
        start = fold.valid[sel] - inputs.US_PER_HALF_HOUR
        day = start // inputs.US_PER_DAY
        labelled_days = set(sc.day[flagged].tolist())
        in_day = np.isin(day, list(labelled_days))
        lab = np.where(in_day, 1, 0).astype(np.int8)
        for d, left, right in zip(
            sc.day[flagged], sc.left[flagged], sc.right[flagged], strict=True
        ):
            between = (start >= t.start_us[left]) & (start <= t.start_us[right]) & (day == d)
            lab[between] = -1
        label[sel] = lab
        info["sites"][site] = {
            "clear_days": len(sc.score),
            "labelled_days": int(flagged.sum()),
            "reverse_rows": int((lab < 0).sum()),
        }
    return label, info


# --- the day-level gate (designs I and C merged) ----------------------------------------------

BREAKPOINT_MARGIN: Final[float] = 0.05


@dataclass
class DayGate:
    """The whole-day gate: a threshold fitted on the visible signed Primaries, and profiles."""

    threshold_smooth: float
    threshold_broken: float
    profiles: dict[str, dayflip.Profile]
    window_lo: int
    window_hi: int
    setting: inputs.Setting
    n_visible_days: int


def fit_day_gate(*, fold: inputs.Fold, refit) -> DayGate:  # noqa: ANN001
    """Fit the day gate's threshold on the visible signed Primaries' no-injection training days.

    The threshold is the 98th percentile of the day score over those days, so no hidden Primary
    enters it. Each folded Primary's demand profile comes from its own training window.
    """
    window_lo = int((refit - timedelta(days=base.WINDOW_DAYS)).timestamp() * 1_000_000)
    window_hi = int((refit - regimes.REFIT_CUTOFF_AGE).timestamp() * 1_000_000) + 1
    placebo = inputs.Setting(0.0, fold.setting.q_points, fold.setting.power_factor)
    scores, broken = [], []
    for s in (s for s in inputs.SIGNED_SIX if s not in fold.pair):
        h = dayflip.hourly(
            t=diag.load_table(site=s, setting=placebo, start_us=window_lo, end_us=window_hi)
        )
        res = dayflip.score_days(h=h, profile=dayflip.build_profile(h=h))
        scores.append(res.score)
        broken.append(res.broken)
    all_scores, all_broken = np.concatenate(scores), np.concatenate(broken)
    profiles = {}
    for row in base.primaries().iter_rows(named=True):
        if fold.folded[row["primary"]]:
            t = diag.load_table(
                site=row["site"], setting=fold.setting, start_us=window_lo, end_us=window_hi
            )
            profiles[row["site"]] = dayflip.build_profile(h=dayflip.hourly(t=t))
    return DayGate(
        threshold_smooth=float(
            np.percentile(all_scores[~all_broken], dayflip.THRESHOLD_PERCENTILE)
        ),
        threshold_broken=float(np.percentile(all_scores[all_broken], dayflip.THRESHOLD_PERCENTILE)),
        profiles=profiles,
        window_lo=window_lo,
        window_hi=window_hi,
        setting=fold.setting,
        n_visible_days=len(all_scores),
    )


def day_open_rows(
    *,
    gate: DayGate,
    site: str,
    valid_us: np.ndarray,
    start_us: int | None = None,
    end_us: int | None = None,
) -> np.ndarray:
    """Return, per row of one Primary, whether its day is flagged and `u` is near or above the breakpoint.

    Training rows pass the training window as `start_us` and `end_us`, so a day at the window's edge is
    scored on the same truncated readings the threshold used. Test rows leave them unset, so the table
    covers the rows' days whole. The gate reads the observed folded readings and never the signed truth.
    A day outside April to September, or an ineligible day, is closed.
    """
    if len(valid_us) == 0:
        return np.zeros(0, dtype=bool)
    lo = start_us if start_us is not None else int(valid_us.min()) - inputs.US_PER_DAY
    hi = end_us if end_us is not None else int(valid_us.max()) + inputs.US_PER_DAY
    h = dayflip.hourly(t=diag.load_table(site=site, setting=gate.setting, start_us=lo, end_us=hi))
    result = dayflip.score_days(h=h, profile=gate.profiles[site])
    flagged = {
        int(d): float(bp)
        for d, sc, bp, brk in zip(
            result.day, result.score, result.breakpoint, result.broken, strict=True
        )
        if sc > (gate.threshold_broken if brk else gate.threshold_smooth) and np.isfinite(bp)
    }
    u_rows, _k = inputs.cams_for_rows(site=site, valid_us=valid_us)
    day = (valid_us - inputs.US_PER_HALF_HOUR) // inputs.US_PER_DAY
    breakpoint_u = np.array([flagged.get(int(d), np.inf) for d in day])
    return np.nan_to_num(u_rows) >= breakpoint_u - BREAKPOINT_MARGIN
