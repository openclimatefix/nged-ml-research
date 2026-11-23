# ruff: noqa: E501
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
COARSE_G: Final[tuple[float, ...]] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
FINE_OFFSETS: Final[tuple[float, ...]] = (-0.25, 0.25)
N_BINS_V2: Final[int] = 21
EM_ROUNDS_V2: Final[int] = 2
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
    projected_fits: int = 1,
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
    f = f_mat.reshape(-1)[lead : lead + len(y)]
    p = np.zeros(len(y))
    passes = 0
    last_post: hmm.Posterior | None = None

    def spec_for(g: float, trans: np.ndarray | None = None) -> hmm.HmmSpec:
        k = n_max + 1
        base = np.full((k, k), (1.0 - STAY) / max(k - 1, 1))
        np.fill_diagonal(base, STAY)
        return hmm.HmmSpec(
            grid=hmm.residual_grid(phi=min(phi, 0.97), sigma_r=sigma_r, n_bins=n_bins),
            phi=min(phi, 0.97), sigma_r=sigma_r, unit_size=g, n_max=n_max,
            trans_n=base if trans is None else trans, q=q, sigma_e=0.08,
        )  # fmt: skip

    def pass_at(
        g: float, *, want_xi: bool = False, trans: np.ndarray | None = None
    ) -> tuple[hmm.Posterior, hmm.HmmSpec]:
        nonlocal passes
        spec = spec_for(g, trans)
        post = hmm.forward_backward(spec=spec, m=y, f=f, p=p, want_xi=want_xi)
        passes += 1
        return post, spec

    scores: dict[float, float] = {}
    for i, g in enumerate(coarse):
        if i == 0 and gate is not None and not np.isfinite(gate.seconds_per_pass):
            gate.measure(run_one_pass=lambda g=g: pass_at(g))
            n_passes = projected_fits * (len(coarse) + len(FINE_OFFSETS) + EM_ROUNDS_V2 + 1)
            gate.check(n_passes_total=n_passes)
            scores[g] = float("nan")  # replaced below
        post, _ = pass_at(g)
        scores[g] = post.log_likelihood
    best = max(scores, key=lambda g: scores[g])
    for off in FINE_OFFSETS:
        g = best + off
        if g > 0.0 and g not in scores:
            scores[g] = pass_at(g)[0].log_likelihood
    best = max(scores, key=lambda g: scores[g])
    trans = None
    for _ in range(EM_ROUNDS_V2):
        post, spec = pass_at(best, want_xi=True, trans=trans)
        trans = hmm.update_trans_n(xi_n=post.xi_n)
    last_post, spec = pass_at(best, trans=trans)
    sm = hmm.summarise(post=last_post, spec=spec)["p_units"]
    fl = hmm.summarise(post=last_post, spec=spec, version="filtered")["p_units"]
    any_s = 1.0 - sm[:, 0]
    any_f = 1.0 - fl[:, 0]
    return HindsightFit(
        flag=(any_s > 0.5)[pre_roll:], filt=(any_f > 0.5)[pre_roll:], unit_size=best, n_max=n_max,
        log_likelihood=last_post.log_likelihood, seconds=time.perf_counter() - t0, passes=passes,
    )  # fmt: skip


def f1_counts(*, truth: np.ndarray, flag: np.ndarray) -> tuple[int, int, int]:
    """Return `(tp, fp, fn)` of a boolean flag against a boolean truth."""
    return int((truth & flag).sum()), int((~truth & flag).sum()), int((truth & ~flag).sum())


def f1(tp: float, fp: float, fn: float) -> float:
    """Half-hour F1 from counts (0 when nothing is flagged or true)."""
    d = 2 * tp + fp + fn
    return 2 * tp / d if d > 0 else 0.0


def cluster_bootstrap_f1_difference(
    *,
    a: dict[str, tuple[int, int, int]],
    b: dict[str, tuple[int, int, int]],
    n_boot: int = 4000,
    seed: int = 720,
) -> dict[str, float]:
    """Return the pooled F1 of `a` minus that of `b` over clusters, with a 95% percentile interval.

    Args:
        a: Cluster name to summed `(tp, fp, fn)` of the first method (the hindsight model).
        b: The same clusters for the second method (experiment 694's `M2`).
        n_boot: Resamples of clusters, with replacement.
        seed: Random seed.
    """
    names = sorted(a)
    if names != sorted(b):
        msg = "the two methods must cover the same clusters"
        raise ValueError(msg)
    ca = np.array([a[k] for k in names], dtype=float)
    cb = np.array([b[k] for k in names], dtype=float)
    point = f1(*ca.sum(0)) - f1(*cb.sum(0))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(names), len(names))
        diffs[i] = f1(*ca[idx].sum(0)) - f1(*cb[idx].sum(0))
    return {
        "point": float(point),
        "lower": float(np.percentile(diffs, 2.5)),
        "upper": float(np.percentile(diffs, 97.5)),
        "f1_a": f1(*ca.sum(0)),
        "f1_b": f1(*cb.sum(0)),
        "n_clusters": float(len(names)),
    }


def false_alarm(*, flag: np.ndarray, first_index_of_flag: int) -> dict[str, float]:
    """Return the flagged share and the active days (at least 4 flags) per 30 days of an engine-free span."""
    n_days = len(flag) // 48
    blocks = flag[: n_days * 48].reshape(n_days, 48).sum(axis=1) if n_days else np.zeros(0)
    active = float((blocks >= 4).sum())
    return {
        "flagged_share": float(flag.mean()) if len(flag) else float("nan"),
        "active_days_per_month": active / max(n_days, 1) * 30.0,
    }


def v2_verdict(
    *,
    hind: dict[str, tuple[int, int, int]],
    m2: dict[str, tuple[int, int, int]],
    fa: dict[str, float],
) -> dict[str, Any]:
    """Return the V2 bar from per-segment counts and the false-alarm statistics of the un-injected segments."""
    c = cluster_bootstrap_f1_difference(a=hind, b=m2)
    return {
        "contrast": c,
        "bar": bars.v2_bar(
            lower=c["lower"],
            point=c["point"],
            flagged_share=fa["flagged_share"],
            active_days_per_month=fa["active_days_per_month"],
        ),
    }


def run_v2(
    *, limit: int | None = None, cap_hours: float = 15.0
) -> int:  # pragma: no cover - needs the stored panel
    """Run V2 and K1 on the segments of experiment 694 (real data; waits for the code review).

    Writes per-segment counts and the bar table under the private folder. Nothing is printed per
    segment. The projected cost is checked after the first fit and the run stops above the cap.
    """
    import e00694_run as r694  # ty: ignore[unresolved-import]
    import e00694_trackers as t694  # ty: ignore[unresolved-import]
    import e00720_guard as guard
    import polars as pl
    from e00694_common import PRE_ROLL as PR694  # ty: ignore[unresolved-import]

    r694.init()
    segs = r694.segments()[: limit or None]
    names = [n for n, s in r694.SCHEDULES.items() if s["primary"]]
    gate = hmm.CostGate(cap_hours=cap_hours)
    hind: dict[str, tuple[int, int, int]] = {}
    m2: dict[str, tuple[int, int, int]] = {}
    rows = []
    n_fits = len(segs) * len(names)
    for seg in segs:
        for name in names:
            sched = r694.SCHEDULES[name]
            g, units = sched["g"], sched["units"]
            y = r694._segment_series(seg=seg)
            drive = np.zeros(r694.TOTAL_LEN)
            lo = PR694 + r694.INJECT_OFFSET
            drive[lo : lo + len(g)] = g
            yi = r694.inject_folded(y=y, g=drive)
            lvl = r694.quantise_spread(spread=r694.causal_spread(y=yi))[PR694:]
            m2_flag = t694.grid_track(y=yi[PR694:], lvl=lvl).p_any_smooth > 0.5
            hf = hindsight_fit(
                y=yi, first_index=seg.start, pre_roll=PR694, gate=gate, projected_fits=n_fits
            )
            win = slice(r694.INJECT_OFFSET, r694.INJECT_OFFSET + len(g))
            truth = units > 0
            key = f"{seg.host}|{seg.period}|{seg.index}"
            for store, flag in ((hind, hf.flag), (m2, m2_flag)):
                t = np.array(store.get(key, (0, 0, 0))) + np.array(
                    f1_counts(truth=truth, flag=flag[win])
                )
                store[key] = (int(t[0]), int(t[1]), int(t[2]))
            rows.append({"cluster": key, "schedule": name, "seconds": hf.seconds})
    out = v2_verdict(
        hind=hind, m2=m2, fa={"flagged_share": float("nan"), "active_days_per_month": float("nan")}
    )
    pl.DataFrame(rows).write_csv(guard.private_path(name="v2_fits.csv"))
    print({k: out["contrast"][k] for k in ("point", "lower", "upper", "f1_a", "f1_b")})
    return 0


def selftest() -> int:
    """Check the machinery on a synthetic host and engine with a known answer."""
    rng = np.random.default_rng(1)
    first = 7 * 48 * 3
    n = PRE_ROLL + 60 * 48
    idx = np.arange(n)
    h = idx % 48
    dtype = day_types(n=n, first_index=first)
    base = 6.0 + 1.5 * np.sin(2 * np.pi * (h - 14) / 48.0) * (1.0 - 0.2 * (dtype > 0))
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.9 * ar[t - 1] + rng.normal(0, 0.08)
    units = np.zeros(n, dtype=int)
    state = 0
    for t in range(PRE_ROLL + 48 * 10, n):
        if rng.random() > 0.97:
            state = int(rng.integers(0, 3))
        units[t] = state
    g = 1.6 * units
    y = np.sqrt(np.maximum(base + ar - g, 0.0) ** 2 + Q_FLOOR_MVA**2) + rng.standard_t(4, n) * 0.05
    gate = hmm.CostGate(cap_hours=1.0)
    hf = hindsight_fit(
        y=y,
        first_index=first,
        gate=gate,
        projected_fits=1,
        n_max=2,
        coarse=(1.0, 1.5, 2.0),
        n_bins=15,
    )
    truth = (units > 0)[PRE_ROLL:]
    tp, fp, fn = f1_counts(truth=truth, flag=hf.flag)
    score = f1(tp, fp, fn)
    assert score > 0.8, score
    assert abs(hf.unit_size - 1.6) <= 0.5, hf.unit_size
    # the paired contrast machinery with constructed counts
    a = {f"s{i}": (60, 5, 10) for i in range(30)}
    b = {f"s{i}": (50, 10, 20) for i in range(30)}
    c = cluster_bootstrap_f1_difference(a=a, b=b)
    assert c["lower"] > 0.0
    assert abs(c["point"] - (f1(60, 5, 10) - f1(50, 10, 20))) < 1e-9
    same = cluster_bootstrap_f1_difference(a=a, b=a)
    assert same["lower"] <= 0.0 <= same["upper"]
    assert bars.v2_bar(
        lower=c["lower"], point=c["point"], flagged_share=0.01, active_days_per_month=0.5
    )["passed"]
    assert not bars.v2_bar(lower=-0.01, point=0.2, flagged_share=0.01, active_days_per_month=0.5)[
        "passed"
    ]
    fa = false_alarm(flag=np.zeros(480, dtype=bool), first_index_of_flag=0)
    assert fa["flagged_share"] == 0.0
    assert fa["active_days_per_month"] == 0.0
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
        f"PASS V2 machinery: hindsight F1 {score:.3f}, unit size {hf.unit_size:.2f} (true 1.60) in {hf.seconds:.1f} s and {hf.passes} passes; paired contrast and bars checked; cost gate stops an over-cap run"
    )
    return 0


def main() -> int:
    """Parse arguments and run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.run:
        return run_v2(limit=args.limit)
    ap.error("give --selftest or --run")
    return 2


if __name__ == "__main__":
    sys.exit(main())
