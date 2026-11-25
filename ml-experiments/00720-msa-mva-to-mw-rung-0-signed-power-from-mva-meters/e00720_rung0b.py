# ruff: noqa: E501, E731, PT018, PERF401
"""Experiment 720, rungs 0b and 0e (V1): the sign read-outs side by side on one injected world.

A `ProbeWorld` is one Primary at one injection level and one fold: day matrices of the reading with
the floor removed `v`, the PV estimate `p`, the gate `u`, the day types, the temperatures, the signed
truth, the reference (training) days, and the scored days. `evaluate_world` scores four read-outs,
each with the real sky and with the sky shifted by 28 days:

- `anchored`: gross demand from dull-sky rows of the reference days (the read-out of the design);
- `unanchored_day` and `unanchored_sky`: the leave-one-out control with two weightings, three passes;
- `hindsight`: the hidden Markov model of `e00720_hmm` with `K = 0`, parameters fitted on anchors.

`build_probe_world` builds a world from the stored panel and a farm's recorded output. The real run
(`--run`) waits for the code review. `--selftest` runs on synthetic worlds.

The injected capacity is the oracle one (the injected increment) times 1.0, with 0.7 and 1.3 as the
sensitivity of the design. That departs from the design, which estimates the injected capacity from
experiment 662's probe, and the page says so.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00720_fit as fit
import e00720_fold as fold
import e00720_hmm as hmm
import e00720_real as real
import e00720_sign as sign
import e00720_worlds as worlds
import numpy as np

Q_POINTS: Final[float] = 2.0
SHIFT_DAYS: Final[int] = 28
SUNNY_K: Final[float] = 0.7
LEVEL_WINDOW_END_US: Final[int] = int(np.datetime64("2025-07-01", "us").astype(np.int64))


@dataclass
class ProbeWorld:
    """One Primary, one level, one fold, as day matrices (points of capacity)."""

    name: str
    v: np.ndarray
    p: np.ndarray
    u: np.ndarray
    day_type: np.ndarray
    temperature: np.ndarray
    x_true: np.ndarray
    reference_days: np.ndarray
    scored_days: np.ndarray
    days_us: np.ndarray | None = None
    k: np.ndarray | None = None
    """CAMS clear-sky index (days by 48); anchors are rows with `k < 0.3`, as the design says."""


def _safe(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0)


def _hindsight_post(
    *, pw: ProbeWorld, p: np.ndarray
) -> tuple[hmm.Posterior, hmm.HmmSpec, np.ndarray]:
    """Fit the `K = 0` model on anchor rows and run it over 7 days before the scored days and the scored days."""
    ref = pw.reference_days
    g = p + pw.v
    anchors = np.zeros_like(pw.v, dtype=bool)
    anchors[ref] = (pw.u if pw.k is None else np.nan_to_num(pw.k))[ref] < sign.U_MIN
    f_all = fit.fit_demand_mean(g=g, rows=anchors, day_type=pw.day_type, temperature=pw.temperature)
    phi, sigma_r = fit.fit_residual(g=g, f=f_all, rows=anchors)
    spec = fit.k0_spec(phi=phi, sigma_r=sigma_r, q=Q_POINTS)
    lo = int(pw.scored_days.min()) - 7
    hi = int(pw.scored_days.max()) + 1
    days = np.arange(lo, hi)
    reading = fold.magnitude_to_reading(y=pw.v[days], q=Q_POINTS).reshape(-1)
    post = hmm.forward_backward(
        spec=spec, m=reading, f=f_all[days].reshape(-1), p=p[days].reshape(-1), want_xi=False
    )
    return post, spec, days


def hindsight_rho(*, pw: ProbeWorld, p: np.ndarray) -> np.ndarray:
    """Return `P(X < 0)` for the scored days from the `K = 0` hidden Markov model.

    The demand mean and residual are fitted on anchor rows of the reference days, and the pass runs
    over the 7 days before the first scored day and the scored days.
    """
    post, spec, days = _hindsight_post(pw=pw, p=p)
    pneg = hmm.summarise(post=post, spec=spec)["p_negative"].reshape(len(days), -1)
    out = np.full(pw.v.shape, np.nan)
    out[days] = pneg
    return out[pw.scored_days]


def hindsight_bars(
    *,
    pw: ProbeWorld,
    p: np.ndarray,
    anchored_rho: np.ndarray | None = None,
    seed_noise: float | None = None,
) -> dict[str, float | bool]:
    """Return the sign-probability and signed-CRPS bars of the `K = 0` model, smoothed and filtered, flattened.

    Scored on the gated half-hours of the scored days outside the 2-point dead band.
    """
    import e00720_bars as bars

    post, spec, days = _hindsight_post(pw=pw, p=p)
    x_full = np.full(pw.v.shape, np.nan)
    x_full[pw.scored_days] = np.where(
        pw.u[pw.scored_days] >= sign.U_MIN, pw.x_true[pw.scored_days], np.nan
    )
    mag = np.nan_to_num(pw.v[days]).reshape(-1)  # the observed magnitude, never the truth
    anch = None
    if anchored_rho is not None:
        full = np.full(pw.v.shape, np.nan)
        full[pw.scored_days] = anchored_rho
        anch = full[days].reshape(-1)
    res = bars.both_versions(
        post=post,
        spec=spec,
        x_true=x_full[days].reshape(-1),
        magnitude=mag,
        anchored_rho=anch,
        seed_noise=seed_noise,
    )
    flat: dict[str, float | bool] = {}
    for version in ("smoothed", "filtered"):
        r = res[version]
        flat[f"{version}_reliability"] = float(r["sign"]["reliability"])
        flat[f"{version}_max_gap_bin_200"] = float(r["sign"]["max_gap_bin_200"])
        flat[f"{version}_sign_passed"] = bool(r["sign"]["passed"])
        flat[f"{version}_crps_soft"] = float(r["crps"]["soft"])
        flat[f"{version}_crps_passed"] = bool(r["crps"]["passed"])
        flat[f"{version}_reliability_mixture_p"] = float(r["sign_mixture"]["reliability"])
        flat[f"{version}_second_half_pending"] = bool(r["second_half_pending"])
    flat["gap_reliability"] = float(res["gap"]["reliability"])
    flat["gap_crps"] = float(res["gap"]["crps"])
    return flat


MIDDAY_SLOTS: Final[tuple[int, int]] = (20, 28)
"""Half-hour slots 10:00 to 14:00 UTC, where the area is also reported alone."""
REFERENCES: Final[tuple[str, ...]] = ("reference_reading_blind", "reference_hour_of_day")
"""Pre-registered: the anchored read-out must beat both on the pooled area and on the midday area."""


def reading_blind_rho(*, p: np.ndarray) -> np.ndarray:
    """Score each half-hour by the PV estimate alone (never reads the meter); a rank score in `[0, 1]`."""
    top = float(np.nanmax(p)) if np.isfinite(p).any() else 1.0
    return np.clip(p / max(top, 1e-9), 0.0, 1.0)


def hour_of_day_rho(*, u: np.ndarray, reference_days: np.ndarray, n_scored: int) -> np.ndarray:
    """Score each half-hour by the reference days' mean yield at that half-hour of the day ("midday means export")."""
    shape = np.nanmean(u[reference_days], axis=0)
    return np.tile(shape / max(float(shape.max()), 1e-9), (n_scored, 1))


def evaluate_world(
    *, pw: ProbeWorld, keep: dict[str, np.ndarray] | None = None
) -> list[dict[str, Any]]:
    """Score every read-out with the real and the shifted sky; return one row per read-out and sky.

    With `keep`, the real-sky `rho` arrays (scored days by 48) are stored in it under the read-out's name.
    """
    rows: list[dict[str, Any]] = []
    sd = pw.scored_days
    p_real = _safe(pw.p)
    p_shift = _safe(worlds.shift_days(a=pw.p, days=SHIFT_DAYS))
    for sky, p in (("real", p_real), ("shifted", p_shift)):
        anchored, fallback = sign.anchored_posterior(
            v=pw.v, p=p, u=pw.u, day_type=pw.day_type, temperature=pw.temperature,
            reference_days=pw.reference_days, scored_days=sd, anchor_value=pw.k,
        )  # fmt: skip
        results: dict[str, np.ndarray] = {
            "anchored": anchored,
            "hindsight": hindsight_rho(pw=pw, p=p),
        }
        lo, hi = max(int(sd.min()) - 7, 0), min(int(sd.max()) + 8, len(pw.v))
        for weights in ("day", "sky"):
            passes = sign.unanchored_posterior(
                v=pw.v[lo:hi], p=p[lo:hi], u=pw.u[lo:hi], day_type=pw.day_type[lo:hi], weights=weights,  # ty: ignore[invalid-argument-type]
            )  # fmt: skip
            for k, rho in enumerate(passes, start=1):
                full = np.full(pw.v.shape, np.nan)
                full[lo:hi] = rho
                results[f"unanchored_{weights}_pass{k}"] = full[sd]
        results["reference_reading_blind"] = reading_blind_rho(p=p[sd])
        results["reference_hour_of_day"] = hour_of_day_rho(
            u=pw.u, reference_days=pw.reference_days, n_scored=len(sd)
        )
        if keep is not None and sky == "real":
            keep.update(results)
        midday = np.zeros(pw.u[sd].shape, dtype=bool)
        midday[:, MIDDAY_SLOTS[0] : MIDDAY_SLOTS[1]] = True
        for name, rho in results.items():
            metrics = sign.evaluate_sign(rho=rho, x_true=pw.x_true[sd], u=pw.u[sd])
            mid = sign.evaluate_sign(rho=rho, x_true=pw.x_true[sd], u=pw.u[sd], row_mask=midday)
            metrics["auc_evaluable"] = (
                min(metrics["n_import"], metrics["n_reverse"]) >= MIN_CLASS_ROWS
            )
            metrics["n_midday_import"] = mid["n_import"]
            metrics["n_midday_reverse"] = mid["n_reverse"]
            midday_ok = min(mid["n_import"], mid["n_reverse"]) >= MIN_CLASS_ROWS
            metrics["auc_midday"] = mid["auc"] if midday_ok else float("nan")
            rows.append(
                {
                    "world": pw.name,
                    "readout": name,
                    "sky": sky,
                    **metrics,
                    "fallback_share": float(fallback.mean()),
                }
            )
    return rows


MIN_CLASS_ROWS: Final[int] = 30
"""A pooled or midday area counts only when each class holds at least this many scored half-hours."""
N_BOOT_REFERENCES: Final[int] = 500
MIN_EVALUABLE_WORLDS: Final[int] = 6
"""A level's midday contrast is judged only when at least this many worlds have both classes at midday."""


def paired_auc_difference(
    *,
    rho_a: np.ndarray,
    rho_b: np.ndarray,
    x_true: np.ndarray,
    u: np.ndarray,
    midday_only: bool = False,
    n_boot: int = N_BOOT_REFERENCES,
    seed: int = 720,
) -> dict[str, float]:
    """Return the area difference `a - b` on one world with a day-block bootstrap (scored days resampled).

    Rows are the gated half-hours outside the dead band, as in `sign.evaluate_sign`. Returns the point
    difference, the 2.5% and 97.5% bootstrap quantiles, and the counts of each class; the difference is
    NaN when either class holds fewer than `MIN_CLASS_ROWS` rows.
    """
    n_days = x_true.shape[0]
    ok = np.isfinite(rho_a) & np.isfinite(rho_b) & np.isfinite(x_true) & (u >= sign.U_MIN)
    ok &= np.abs(np.nan_to_num(x_true)) >= sign.DEAD_BAND_POINTS
    if midday_only:
        ok[:, : MIDDAY_SLOTS[0]] = False
        ok[:, MIDDAY_SLOTS[1] :] = False
    pos = x_true < 0.0
    n_rev, n_imp = int((ok & pos).sum()), int((ok & ~pos).sum())
    nan = float("nan")
    if min(n_rev, n_imp) < MIN_CLASS_ROWS:
        return {"point": nan, "lower": nan, "upper": nan, "n_reverse": n_rev, "n_import": n_imp}

    def diff(days: np.ndarray) -> float:
        m = ok[days]
        p = pos[days][m]
        return sign.auc(score=rho_a[days][m], positive=p) - sign.auc(
            score=rho_b[days][m], positive=p
        )

    point = diff(np.arange(n_days))
    rng = np.random.default_rng(seed)
    draws = np.array([diff(rng.integers(0, n_days, n_days)) for _ in range(n_boot)])
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    return {
        "point": point,
        "lower": float(lo),
        "upper": float(hi),
        "n_reverse": n_rev,
        "n_import": n_imp,
    }


def reference_bar(
    *, store: dict[str, np.ndarray], n_boot: int = N_BOOT_REFERENCES
) -> list[dict[str, Any]]:
    """Return the pre-registered references bar, one row per reference, area, and injection level.

    The worlds are the injected worlds (levels 1 and 2) at the oracle estimate scale and host scale
    1.0, read from `rung0b_rho.npz`'s keys `name|readout`. For each world the paired area difference
    (anchored minus the reference) has a day-block bootstrap interval. The row's point is the mean
    over its evaluable worlds, and its interval is the mean of the worlds' interval ends (a
    conservative stand-in for a joint bootstrap, which the independent worlds would only narrow).
    The outcome is `beats` when the lower end is above 0, `loses` when the upper end is below 0,
    `inconclusive` otherwise, and `not_evaluable` when fewer than `MIN_EVALUABLE_WORLDS` worlds hold
    both classes (midday) or none does (pooled). A world with both classes counts for the pooled
    area.
    """
    names = sorted({k.rsplit("|", 1)[0] for k in store if k.endswith("|x_true")})
    out: list[dict[str, Any]] = []
    for level in (1.0, 2.0):
        sel = [n for n in names if n.endswith(f"|L{level:g}|s1|h1")]
        for ref in REFERENCES:
            for area in ("pooled", "midday"):
                parts = [
                    paired_auc_difference(
                        rho_a=store[f"{n}|anchored"], rho_b=store[f"{n}|{ref}"], x_true=store[f"{n}|x_true"],
                        u=store[f"{n}|u"], midday_only=area == "midday", n_boot=n_boot,
                    )
                    for n in sel
                ]  # fmt: skip
                good = [q for q in parts if np.isfinite(q["point"])]
                need = MIN_EVALUABLE_WORLDS if area == "midday" else 1
                row: dict[str, Any] = {
                    "reference": ref, "area": area, "level": level, "n_worlds": len(sel), "n_evaluable": len(good),
                    "n_reverse": sum(q["n_reverse"] for q in parts), "n_import": sum(q["n_import"] for q in parts),
                }  # fmt: skip
                if len(good) < need:
                    row |= {
                        "point": float("nan"),
                        "lower": float("nan"),
                        "upper": float("nan"),
                        "outcome": "not_evaluable",
                    }
                else:
                    lo = float(np.mean([q["lower"] for q in good]))
                    hi = float(np.mean([q["upper"] for q in good]))
                    row |= {
                        "point": float(np.mean([q["point"] for q in good])),
                        "lower": lo,
                        "upper": hi,
                    }
                    row["outcome"] = "beats" if lo > 0 else "loses" if hi < 0 else "inconclusive"
                out.append(row)
    return out


def references_outcome(*, bar: list[dict[str, Any]]) -> str:
    """Return `passed`, `failed`, or `inconclusive`.

    Any pooled row that is `not_evaluable` gives `inconclusive`, and otherwise the result is the
    bar's pass or fail.
    """
    if any(r["area"] == "pooled" and r["outcome"] == "not_evaluable" for r in bar) or not bar:
        return "inconclusive"
    return "passed" if references_bar_passed(bar=bar) else "failed"


def references_bar_passed(*, bar: list[dict[str, Any]]) -> bool:
    """Return whether every pooled row `beats` and no evaluable midday row `loses` (pre-registered rule).

    A pooled row that is `not_evaluable` makes the bar `inconclusive` rather than failed; see `references_outcome`.
    """
    pooled = [r for r in bar if r["area"] == "pooled"]
    mid = [r for r in bar if r["area"] == "midday" and r["outcome"] != "not_evaluable"]
    return (
        bool(pooled)
        and all(r["outcome"] == "beats" for r in pooled)
        and all(r["outcome"] != "loses" for r in mid)
    )


def false_reverse_no_injection(*, pw: ProbeWorld) -> dict[str, float]:
    """Return the false-reverse rate of the anchored read-out on the world's sunny middays (gate forced open)."""
    sd = pw.scored_days
    rho, _ = sign.anchored_posterior(
        v=pw.v, p=_safe(pw.p), u=pw.u, day_type=pw.day_type, temperature=pw.temperature,
        reference_days=pw.reference_days, scored_days=sd, anchor_value=pw.k,
    )  # fmt: skip
    mid = (pw.u[sd] >= 0.5) & (np.abs(np.arange(48) - 24)[None, :] <= 4)
    rate, n = sign.false_reverse_rate(rho=rho, mask=mid)
    return {"false_reverse_rate": rate, "n_rows": float(n)}


def build_probe_world(
    *,
    name: str,
    y_pct: np.ndarray,
    stamp_us: np.ndarray,
    u: np.ndarray,
    k: np.ndarray,
    temperature_c: np.ndarray,
    p_host_pct: np.ndarray,
    farm_at: Callable[[np.ndarray], np.ndarray] | None,
    level: float,
    fold_name: str,
    estimate_scale: float = 1.0,
    host_scale: float = 1.0,
    q: float = Q_POINTS,
    holidays_us: frozenset[int] = frozenset(),
) -> ProbeWorld:
    """Return a world: the host's signed power minus a farm's output, folded through the meter.

    Args:
        name: The label of the world, which is also its key in the stored arrays.
        y_pct: The host's signed power in points of capacity on the grid (NaN where flagged).
        stamp_us: Valid times (ends of half-hours).
        u: Scaled CAMS unit shape at the host.
        k: Clear-sky index at the host.
        temperature_c: Temperature at the host.
        p_host_pct: F7's PV estimate of the host's own unmetered PV, in points (0 where absent).
        farm_at: Maps times in microseconds to the farm's normalised output (NaN in a gap); `None` for
            the no-injection world (level 0).
        level: The injected size as a multiple of the median absolute signed reading on sunny
            summer-midday half-hours before 2025-07-01 (`0` is the no-injection placebo).
        fold_name: The scored month.
        estimate_scale: Multiplier of the injected capacity in `p` (1.0 oracle; 0.7 and 1.3 sensitivity).
        host_scale: Multiplier of the host's own PV estimate in `p` (1.0; 0.7 and 1.3 at level 0, where
            the D5 bar is read; 2.5 in the hard check world).
        q: The reactive floor in points, used to fold the signed power through the meter.
        holidays_us: Start-of-day instants (microseconds) of bank holidays, read as Sundays.

    Returns:
        The `ProbeWorld`. Half-hours where the farm has a gap are NaN in every array, so every
        level trains and scores on the same rows.
    """
    import e00720_census as census

    hour, month = census.hour_month(stamp_us=stamp_us)
    sunny = (
        (hour >= 10) & (hour < 14) & (month >= 4) & (month <= 9) & (np.nan_to_num(k) >= SUNNY_K)
        & (stamp_us < LEVEL_WINDOW_END_US) & np.isfinite(y_pct)
    )  # fmt: skip
    if level > 0.0:
        if int(sunny.sum()) < 200:
            msg = f"{name}: only {int(sunny.sum())} sunny summer-midday rows before 2025-07-01"
            raise ValueError(msg)
        increment = level * float(np.median(np.abs(y_pct[sunny])))
        assert farm_at is not None
        farm = farm_at(stamp_us)
        x = y_pct - increment * farm
    else:
        increment = 0.0
        x = y_pct.copy()
    p = host_scale * p_host_pct + estimate_scale * increment * np.nan_to_num(u)
    mat = lambda a: real.to_day_matrix(values=a, stamp_us=stamp_us)
    x_m, days_us = mat(x)
    reading = worlds.reading_from_signed(x=x_m, q=q)
    v = fold.reading_to_magnitude(m=reading, q=q)
    p_m, _ = mat(np.where(np.isfinite(x), p, np.nan))
    u_m, _ = mat(u)
    t_m, _ = mat(temperature_c)
    temp_day = np.nanmean(t_m, axis=1)
    weekday = ((days_us // real.US_PER_DAY) + 3) % 7  # 1970-01-01 was a Thursday; Monday is 0
    day_type = np.where(weekday == 5, 1, np.where(weekday == 6, 2, 0))
    if holidays_us:
        day_type = np.where(np.isin(days_us, list(holidays_us)), 2, day_type)
    w = real.fold_windows(fold=fold_name)
    ref = real.days_in(days_us=days_us, lo_us=w.train_start_us, hi_us=w.train_end_us)
    scored = real.days_in(days_us=days_us, lo_us=w.cutoff_us, hi_us=w.score_end_us)
    return ProbeWorld(
        name=name, v=v, p=np.where(np.isfinite(p_m), p_m, np.nan), u=np.nan_to_num(u_m), day_type=day_type,
        temperature=np.nan_to_num(temp_day, nan=10.0), x_true=x_m, reference_days=ref, scored_days=scored, days_us=days_us, k=np.nan_to_num(mat(k)[0]),
    )  # fmt: skip


LEVELS: Final[tuple[float, ...]] = (0.0, 1.0, 2.0)
SENSITIVITY: Final[tuple[float, ...]] = (0.7, 1.3)
"""Multipliers of the injected capacity in `p` at level 2 (1.0 is the oracle)."""
RUNG0B_CAP_HOURS: Final[float] = 3.0


def _yyyymmdd(*, days_us: np.ndarray) -> np.ndarray:
    d = (days_us // real.US_PER_DAY).astype("datetime64[D]")
    return (
        (d.astype("datetime64[Y]").astype(int) + 1970) * 10000
        + (d.astype("datetime64[M]").astype(int) % 12 + 1) * 100
        + (d - d.astype("datetime64[M]")).astype(int)
        + 1
    )


def run_worlds(
    *,
    panel: real.Panel,
    sites: tuple[str, ...],
    folds: tuple[str, ...],
    farm_for_site: dict[str, Callable[[np.ndarray], np.ndarray]],
    host_p: Callable[[str, str], np.ndarray],
    labels_674: Any = None,
    limit_worlds: int | None = None,
    gate: hmm.CostGate | None = None,
    rho_store: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score every world and return `(readout_rows, reference_rows)`; nothing is written here.

    A world is a site, a fold, a level, and an estimate scale. Level 0 has no injection; levels 1 and 2
    use the oracle scale 1.0, and level 2 also uses 0.7 and 1.3. Every world of a site and fold reads
    the same rows, because the union gap of the farm series is masked in the host's power as well.
    The first world is timed, and the cost gate stops the run when the projection exceeds its cap.
    """
    import e00720_references as refs

    gate = gate or hmm.CostGate(cap_hours=RUNG0B_CAP_HOURS)
    plan = [
        (site, fold_name, level, scale, hscale)
        for fold_name in folds
        for site in sites
        for level, scale, hscale in [
            (0.0, 1.0, 1.0),
            *[(0.0, 1.0, hs) for hs in SENSITIVITY],
            (1.0, 1.0, 1.0),
            (2.0, 1.0, 1.0),
            *[(2.0, sc, 1.0) for sc in SENSITIVITY],
        ]
    ]
    if limit_worlds:
        plan = plan[:limit_worlds]
    rows: list[dict[str, Any]] = []
    ref_rows: list[dict[str, Any]] = []
    for i, (site, fold_name, level, scale, hscale) in enumerate(plan):
        t0 = time.perf_counter()
        farm_at = farm_for_site[site]
        y = panel.power_pct[panel.sites.index(site)]
        y = np.where(np.isfinite(farm_at(panel.stamp_us)), y, np.nan)
        j = panel.sites.index(site)
        pw = build_probe_world(
            name=f"{site}|{fold_name}|L{level:g}|s{scale:g}|h{hscale:g}", y_pct=y, stamp_us=panel.stamp_us, u=panel.u[j], k=panel.k[j],
            temperature_c=panel.temperature_c[j], p_host_pct=host_p(site, fold_name),
            farm_at=farm_at if level > 0 else None, level=level, fold_name=fold_name, estimate_scale=scale, host_scale=hscale,
        )  # fmt: skip
        keep: dict[str, np.ndarray] = {}
        for r in evaluate_world(pw=pw, keep=keep):
            rows.append(
                {
                    "site": site,
                    "fold": fold_name,
                    "level": level,
                    "scale": scale,
                    "host_scale": hscale,
                    **r,
                }
            )
        sd = pw.scored_days
        if rho_store is not None:
            for key, arr in keep.items():
                rho_store[f"{pw.name}|{key}"] = arr
            rho_store[f"{pw.name}|x_true"] = pw.x_true[sd]
            rho_store[f"{pw.name}|u"] = pw.u[sd]
        assert pw.days_us is not None
        month = ((pw.days_us.astype("datetime64[us]").astype("datetime64[M]").astype(int)) % 12) + 1
        weekend = (pw.day_type > 0).astype(int)
        demand = refs.dark_demand_levels(
            v=pw.v, u=pw.u, month=month, weekend=weekend, rows=pw.reference_days
        )
        called = refs.gated_rule_called(p=_safe(pw.p), u=pw.u, demand=demand)
        rm = refs.rule_metrics(called=called[sd], x_true=pw.x_true[sd], u=pw.u[sd])
        ref_rows.append(
            {
                "site": site,
                "fold": fold_name,
                "level": level,
                "scale": scale,
                "host_scale": hscale,
                "reference": "506_gated_rule",
                **rm,
            }
        )
        truth = refs.day_truth(x_true=pw.x_true[sd], u=pw.u[sd])
        for name in ("anchored", "hindsight"):
            ds = refs.day_scores(
                call=refs.day_call_from_rho(rho=keep[name], u=pw.u[sd]), truth=truth
            )
            ref_rows.append(
                {
                    "site": site,
                    "fold": fold_name,
                    "level": level,
                    "scale": scale,
                    "host_scale": hscale,
                    "reference": f"720_{name}_day",
                    **ds,
                }
            )
        if labels_674 is not None and scale == 1.0 and hscale == 1.0:
            dates = _yyyymmdd(days_us=pw.days_us[sd])
            lab = labels_674.filter((labels_674["site"] == site) & (labels_674["level"] == level))
            cls = dict(zip(lab["date"].to_list(), lab["cls"].to_list(), strict=True))
            neg = dict(zip(lab["date"].to_list(), lab["any_negative"].to_list(), strict=True))
            label = np.array([cls.get(int(d), "not_scorable") for d in dates], dtype=object)
            truth_674 = np.array(
                [bool(neg.get(int(d), False)) for d in dates]
            )  # 674's own world, not 720's
            ds = refs.day_scores(call=truth_674, truth=truth_674, label=label)
            ref_rows.append(
                {
                    "site": site,
                    "fold": fold_name,
                    "level": level,
                    "scale": scale,
                    "host_scale": hscale,
                    "reference": "674_day_labeller",
                    **ds,
                }
            )
        ref_rows.append(
            {
                "site": site,
                "fold": fold_name,
                "level": level,
                "scale": scale,
                "host_scale": hscale,
                "reference": "720_hindsight_bars",
                **hindsight_bars(pw=pw, p=_safe(pw.p), anchored_rho=keep.get("anchored")),
            }
        )
        if level == 0.0:
            ref_rows.append(
                {
                    "site": site,
                    "fold": fold_name,
                    "level": level,
                    "scale": scale,
                    "host_scale": hscale,
                    "reference": "720_anchored_false_reverse_gate_open",
                    **false_reverse_no_injection(pw=pw),
                }
            )
        if i == 0:
            gate.seconds_per_pass = time.perf_counter() - t0
            gate.check(n_passes_total=len(plan))
    return rows, ref_rows


def run_real(
    *, dense_raw: Path | None, limit_worlds: int | None
) -> int:  # pragma: no cover - needs the stored panel
    """Run rung 0b on the stored panel (waits for the Opus code review). Writes only under the private folder.

    Reads the farms' recorded output (`e00506_data.farm_norm`), experiment 716's PV estimate and as-of
    capacity (put its folder on `EXTRA_PYTHONPATH`), and experiment 674's stored day labels. Estimated
    cost: 120 worlds at about 1 to 3 minutes each, under the 3 core-hour cap of the gate.
    """
    import e00506_data as data  # ty: ignore[unresolved-import]
    import e00716_pv as pv  # ty: ignore[unresolved-import]
    import e00720_guard as guard
    import e00720_references as refs
    import polars as pl

    if dense_raw is None:
        msg = "give --dense-raw (experiment 713's dense_raw.npy)"
        raise SystemExit(msg)
    panel = real.load_panel(dense_raw_path=dense_raw)
    n = len(panel.stamp_us)
    raw = np.load(dense_raw, mmap_mode="r")[:, :n]
    farms = {f: data.farm_norm(farm=f) for f in "ABC"}
    union_gap = ~np.isfinite(sum(farms.values()))
    step = real.US_PER_HALF_HOUR

    def make_farm(letter: str) -> Callable[[np.ndarray], np.ndarray]:
        series = np.where(union_gap, np.nan, farms[letter])
        return lambda times: series[((times - panel.stamp_us[0]) // step).astype(int)]

    farm_for_site = {s: make_farm(data.FARM_OF_SITE[s]) for s in real.SIGNED_SIX}
    commit = real.check_f7_commit(module_file=Path(pv.__file__))
    guard.private_path(name="rung0b_provenance.json").write_text(json.dumps({"f7_commit": commit}))
    t649, t662 = pv.read_tables()
    series = data.series_table()
    cap_mw = np.array(
        [float(series.filter(pl.col("site") == s)["effective_capacity_mw"][0]) for s in panel.sites]
    )
    doy = pv.day_of_year(grid_start_us=int(panel.stamp_us[0]), n_grid=n)
    cache: dict[str, np.ndarray] = {}

    def host_p(site: str, fold: str) -> np.ndarray:
        if fold not in cache:
            caps = np.array(
                [
                    pv.as_of_capacity(site=s, fold=fold, table_649=t649, table_662=t662).mw
                    for s in panel.sites
                ]
            )
            est = pv.dense_share(
                raw=raw, doy=doy, capacities_mw=caps, power_capacity_mw=cap_mw
            )  # F7's own u99 default
            cache[fold] = 100.0 * np.asarray(est.share, dtype=np.float64)
        return cache[fold][panel.sites.index(site)]

    store: dict[str, np.ndarray] = {}
    rows, ref_rows = run_worlds(
        panel=panel, sites=real.SIGNED_SIX, folds=real.RUNG0_FOLDS, farm_for_site=farm_for_site, host_p=host_p,
        labels_674=refs.read_674_labels(), limit_worlds=limit_worlds, rho_store=store,
    )  # fmt: skip
    np.savez_compressed(str(guard.private_path(name="rung0b_rho.npz")), **store)  # ty: ignore[invalid-argument-type]
    pl.DataFrame(rows).write_csv(guard.private_path(name="rung0b_readouts.csv"))
    pl.DataFrame(ref_rows).write_csv(guard.private_path(name="rung0b_references.csv"))
    bar = reference_bar(store=store)
    pl.DataFrame(bar).write_csv(guard.private_path(name="rung0b_reference_bar.csv"))
    print(
        f"wrote {len(rows)} read-out rows and {len(ref_rows)} reference rows under the private folder; references bar: {references_outcome(bar=bar)}"
    )
    return 0


def check_references_machinery() -> dict[float, tuple[float, float, float]]:
    """Check the references bar on exporter worlds with a realistic import share and on a constructed store."""
    # an exporter world with a realistic import share at gated half-hours: both classes at midday, a reportable bar
    shares: dict[float, tuple[float, float, float]] = {}
    for seed, pvf in ((41, 0.4), (42, 0.5)):
        we = worlds.synthetic_world(n_days=900, capacity=100.0, seed=seed, pv_fraction=pvf)
        pwe = ProbeWorld(
            f"synthetic exporter {pvf}", fold.reading_to_magnitude(m=we.reading, q=Q_POINTS), we.p, we.u,
            we.day_type, we.temperature, we.x, np.arange(0, 540), np.arange(560, 740),
        )  # fmt: skip
        keep_e: dict[str, np.ndarray] = {}
        rows_e = evaluate_world(pw=pwe, keep=keep_e)
        real_rows = {r["readout"]: r for r in rows_e if r["sky"] == "real"}
        a = real_rows["anchored"]
        assert min(a["n_import"], a["n_reverse"]) >= MIN_CLASS_ROWS, (
            pvf,
            a["n_import"],
            a["n_reverse"],
        )
        assert min(a["n_midday_import"], a["n_midday_reverse"]) >= MIN_CLASS_ROWS, pvf
        assert np.isfinite(a["auc_midday"])
        d = paired_auc_difference(
            rho_a=keep_e["anchored"], rho_b=keep_e["reference_reading_blind"], x_true=pwe.x_true[pwe.scored_days], u=pwe.u[pwe.scored_days], n_boot=100,
        )  # fmt: skip
        assert d["lower"] <= d["point"] <= d["upper"], d
        shares[pvf] = (
            round(a["auc"], 3),
            round(real_rows["reference_reading_blind"]["auc"], 3),
            round(d["lower"], 3),
        )
    # the bar's machinery: a world whose anchored score is the truth beats a coin-flip reference, and a one-class world is not evaluable
    rng_b = np.random.default_rng(1)
    xt = rng_b.normal(0, 5, (60, 48))
    uu = np.full((60, 48), 0.8)
    good = (xt < 0).astype(float) * 0.8 + 0.1 + rng_b.normal(0, 0.05, xt.shape)
    junk = rng_b.uniform(0, 1, xt.shape)
    st = {}
    for ref in REFERENCES:
        st["S|f|L1|s1|h1|anchored"] = good
        st[f"S|f|L1|s1|h1|{ref}"] = junk
    st["S|f|L1|s1|h1|x_true"] = xt
    st["S|f|L1|s1|h1|u"] = uu
    bar = reference_bar(store=st, n_boot=50)
    assert all(
        r["outcome"] == "beats" for r in bar if r["level"] == 1.0 and r["area"] == "pooled"
    ), bar
    assert all(r["outcome"] == "not_evaluable" for r in bar if r["level"] == 2.0), bar
    st["S|f|L1|s1|h1|x_true"] = -np.abs(xt) - 3.0  # every row reverse
    assert all(r["outcome"] == "not_evaluable" for r in reference_bar(store=st, n_boot=50)), (
        "one class must not be evaluable"
    )
    assert not references_bar_passed(bar=reference_bar(store=st, n_boot=50))
    assert references_outcome(bar=reference_bar(store=st, n_boot=50)) == "inconclusive"
    return shares


def selftest() -> int:  # noqa: PLR0915
    """Check `evaluate_world` and `build_probe_world` on synthetic data with known answers."""
    w = worlds.synthetic_world(n_days=900, capacity=100.0, seed=31, pv_fraction=1.0)
    v = fold.reading_to_magnitude(m=w.reading, q=Q_POINTS)
    pw = ProbeWorld(
        "synthetic exporter",
        v,
        w.p,
        w.u,
        w.day_type,
        w.temperature,
        w.x,
        np.arange(0, 540),
        np.arange(560, 600),
    )
    rows = evaluate_world(pw=pw)
    by = {(r["readout"], r["sky"]): r for r in rows}
    for r in rows:
        if r["readout"] in {"anchored", "hindsight", "unanchored_sky_pass3"}:
            print(
                f"  {r['readout']:22s} {r['sky']:8s} area {r['auc']:.3f} recall {r['recall_at_5pct_false']:.3f} rows {int(r['n_rows'])} reverse {int(r['n_reverse'])}"
            )
    assert by[("anchored", "real")]["auc"] - by[("anchored", "shifted")]["auc"] >= 0.1
    assert by[("hindsight", "real")]["auc"] > 0.9
    # a no-injection world has no reverse flow, and the false-reverse rate is under 1%
    w0 = worlds.synthetic_world(n_days=900, capacity=100.0, seed=32, pv_fraction=0.12)
    pw0 = ProbeWorld(
        "synthetic placebo",
        fold.reading_to_magnitude(m=w0.reading, q=Q_POINTS),
        w0.p,
        w0.u,
        w0.day_type,
        w0.temperature,
        w0.x,
        np.arange(0, 540),
        np.arange(560, 600),
    )
    fr = false_reverse_no_injection(pw=pw0)
    # a hard check world: the PV estimate is 2.5 times too high at a non-exporter, so the check must be able to fail
    pw_hard = ProbeWorld(
        "hard",
        pw0.v,
        2.5 * pw0.p,
        pw0.u,
        pw0.day_type,
        pw0.temperature,
        pw0.x_true,
        pw0.reference_days,
        pw0.scored_days,
    )
    fr_hard = false_reverse_no_injection(pw=pw_hard)
    assert fr_hard["false_reverse_rate"] > 5 * max(fr["false_reverse_rate"], 0.01), (fr, fr_hard)
    # measured, and stated on the page: no false reverse flow at 1.0 and 1.3 times, a clear rise at 2.5 times
    fr13 = false_reverse_no_injection(
        pw=ProbeWorld(
            "x13",
            pw0.v,
            1.3 * pw0.p,
            pw0.u,
            pw0.day_type,
            pw0.temperature,
            pw0.x_true,
            pw0.reference_days,
            pw0.scored_days,
        )
    )
    assert fr13["false_reverse_rate"] < 0.01 and fr_hard["false_reverse_rate"] > 0.05, (
        fr13,
        fr_hard,
    )
    shares = check_references_machinery()
    assert {"reference_reading_blind", "reference_hour_of_day"} <= {r["readout"] for r in rows}
    assert fr["false_reverse_rate"] < 0.01 and fr["n_rows"] > 100, fr
    # plumbing of the injection on a fake panel
    start = int(np.datetime64("2023-01-01T00:30:00", "us").astype(np.int64))
    n = 1300 * 48
    stamp = start + np.arange(n, dtype=np.int64) * real.US_PER_HALF_HOUR
    rng = np.random.default_rng(3)
    day = np.arange(n) // 48
    hh = np.arange(n) % 48
    u_fake = (
        np.clip(np.sin(np.pi * (hh - 10) / 28.0), 0, 1)
        * np.where((day % 365 > 90) & (day % 365 < 270), 1.0, 0.4)
        * rng.uniform(0.2, 1.0, 1300)[day]
    )
    y_fake = 40.0 + 5.0 * rng.normal(size=n) - 5.0 * u_fake
    farm = lambda times: np.clip(
        np.sin(np.pi * (((times // real.US_PER_HALF_HOUR) % 48) - 10) / 28.0), 0, 1
    )
    built = build_probe_world(
        name="fake", y_pct=y_fake, stamp_us=stamp, u=u_fake, k=np.clip(u_fake * 1.2, 0, 1.5), temperature_c=np.full(n, 12.0),
        p_host_pct=5.0 * u_fake, farm_at=farm, level=2.0, fold_name="2026-04",
    )  # fmt: skip
    assert built.scored_days.size == 30 and built.reference_days.size > 600, (
        built.scored_days.size,
        built.reference_days.size,
    )
    assert not np.intersect1d(built.scored_days, built.reference_days).size
    assert (built.x_true[built.scored_days] < 0).any(), "the injection must create reverse flow"
    # the full world loop on a fake panel, with a fake farm, a fake host estimate, and fake 674 labels
    import polars as pl

    fake_panel = real.Panel(
        sites=["D5"], stamp_us=stamp, power_pct=y_fake[None, :], u=u_fake[None, :], k=np.clip(u_fake * 1.2, 0, 1.5)[None, :],
        temperature_c=np.full((1, n), 12.0),
    )  # fmt: skip
    labels = pl.DataFrame(
        {
            "site": ["D5"],
            "level": [2.0],
            "date": [20260415],
            "cls": ["reverse"],
            "any_negative": [True],
        }
    )
    rows, ref_rows = run_worlds(
        panel=fake_panel, sites=("D5",), folds=("2026-04",), farm_for_site={"D5": farm},
        host_p=lambda site, fold: 5.0 * u_fake, labels_674=labels, limit_worlds=5,
    )  # fmt: skip
    assert {r["host_scale"] for r in rows} == {1.0, 0.7, 1.3}
    kinds = {r["reference"] for r in ref_rows}
    assert {
        "506_gated_rule",
        "720_anchored_day",
        "674_day_labeller",
        "720_anchored_false_reverse_gate_open",
    } <= kinds, kinds
    tiny = hmm.CostGate(cap_hours=1e-12)
    try:
        run_worlds(
            panel=fake_panel,
            sites=("D5",),
            folds=("2026-04",),
            farm_for_site={"D5": farm},
            host_p=lambda s, f: 5.0 * u_fake,
            limit_worlds=2,
            gate=tiny,
        )
    except RuntimeError:
        pass
    else:
        msg = "the cost gate did not stop the world loop"
        raise AssertionError(msg)
    assert (
        int(
            _yyyymmdd(days_us=np.array([int(np.datetime64("2026-04-05", "us").astype(np.int64))]))[
                0
            ]
        )
        == 20260405
    )
    print(
        f"  exporter worlds, by PV share: anchored area, reading-blind area, lower bound of the difference: {shares}"
    )
    print(
        f"PASS rung 0b driver: anchored beats shifted sky by {by[('anchored', 'real')]['auc'] - by[('anchored', 'shifted')]['auc']:.3f}; hindsight area {by[('hindsight', 'real')]['auc']:.3f}; placebo false reverse {fr['false_reverse_rate']:.4f}; injection plumbing builds {built.scored_days.size} scored days"
    )
    return 0


def main() -> int:
    """Parse arguments and run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dense-raw", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.run:
        return run_real(dense_raw=args.dense_raw, limit_worlds=args.limit)
    ap.error("give --selftest or --run")
    return 2


if __name__ == "__main__":
    sys.exit(main())
