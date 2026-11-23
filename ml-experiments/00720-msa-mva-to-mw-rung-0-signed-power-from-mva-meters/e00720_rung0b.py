# ruff: noqa: E501, D417, E731, PT018
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
import sys
from collections.abc import Callable
from dataclasses import dataclass
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


def _safe(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0)


def hindsight_rho(*, pw: ProbeWorld, p: np.ndarray) -> np.ndarray:
    """Return `P(X < 0)` for the scored days from the `K = 0` hidden Markov model.

    The demand mean and residual are fitted on anchor rows of the reference days, and the pass runs
    over the 7 days before the first scored day and the scored days.
    """
    ref = pw.reference_days
    g = p + pw.v
    anchors = np.zeros_like(pw.v, dtype=bool)
    anchors[ref] = pw.u[ref] < sign.U_MIN
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
    pneg = hmm.summarise(post=post, spec=spec)["p_negative"].reshape(len(days), -1)
    out = np.full(pw.v.shape, np.nan)
    out[days] = pneg
    return out[pw.scored_days]


def evaluate_world(*, pw: ProbeWorld) -> list[dict[str, Any]]:
    """Score every read-out with the real and the shifted sky; return one row per read-out and sky."""
    rows: list[dict[str, Any]] = []
    sd = pw.scored_days
    p_real = _safe(pw.p)
    p_shift = _safe(worlds.shift_days(a=pw.p, days=SHIFT_DAYS))
    for sky, p in (("real", p_real), ("shifted", p_shift)):
        anchored, fallback = sign.anchored_posterior(
            v=pw.v, p=p, u=pw.u, day_type=pw.day_type, temperature=pw.temperature,
            reference_days=pw.reference_days, scored_days=sd,
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
        for name, rho in results.items():
            metrics = sign.evaluate_sign(rho=rho, x_true=pw.x_true[sd], u=pw.u[sd])
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


def false_reverse_no_injection(*, pw: ProbeWorld) -> dict[str, float]:
    """Return the false-reverse rate of the anchored read-out on the world's sunny middays (gate forced open)."""
    sd = pw.scored_days
    rho, _ = sign.anchored_posterior(
        v=pw.v, p=_safe(pw.p), u=pw.u, day_type=pw.day_type, temperature=pw.temperature,
        reference_days=pw.reference_days, scored_days=sd,
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
    q: float = Q_POINTS,
    holidays_us: frozenset[int] = frozenset(),
) -> ProbeWorld:
    """Return a world: the host's signed power minus a farm's output, folded through the meter.

    Args:
        y_pct: The host's signed power in points of capacity on the grid (NaN where flagged).
        stamp_us: Valid times (ends of half-hours).
        u: Scaled CAMS unit shape at the host; `k`: clear-sky index.
        temperature_c: Temperature at the host.
        p_host_pct: F7's PV estimate of the host's own unmetered PV, in points (0 where absent).
        farm_at: Maps times in microseconds to the farm's normalised output (NaN in a gap); `None` for
            the no-injection world (level 0).
        level: The injected size as a multiple of the median absolute signed reading on sunny
            summer-midday half-hours before 2025-07-01 (`0` is the no-injection placebo).
        fold_name: The scored month.
        estimate_scale: Multiplier of the injected capacity in `p` (1.0 oracle; 0.7 and 1.3 sensitivity).
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
    p = p_host_pct + estimate_scale * increment * np.nan_to_num(u)
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
        temperature=np.nan_to_num(temp_day, nan=10.0), x_true=x_m, reference_days=ref, scored_days=scored,
    )  # fmt: skip


def selftest() -> int:
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
    print(
        f"PASS rung 0b driver: anchored beats shifted sky by {by[('anchored', 'real')]['auc'] - by[('anchored', 'shifted')]['auc']:.3f}; hindsight area {by[('hindsight', 'real')]['auc']:.3f}; placebo false reverse {fr['false_reverse_rate']:.4f}; injection plumbing builds {built.scored_days.size} scored days"
    )
    return 0


def main() -> int:
    """Parse arguments and run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.run:
        msg = "the real-data run of rung 0b is not wired yet: it waits for the code review (see the page)"
        raise SystemExit(msg)
    ap.error("give --selftest")
    return 2


if __name__ == "__main__":
    sys.exit(main())
