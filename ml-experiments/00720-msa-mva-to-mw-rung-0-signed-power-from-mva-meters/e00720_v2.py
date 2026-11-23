# ruff: noqa: E501, PLR0915, PT018
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
    n_hat: np.ndarray | None = None
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
            gate.check(n_passes_total=round(n_passes))
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
    n_hat = np.argmax(sm, axis=1)[pre_roll:]
    fl = hmm.summarise(post=last_post, spec=spec, version="filtered")["p_units"]
    any_s = 1.0 - sm[:, 0]
    any_f = 1.0 - fl[:, 0]
    return HindsightFit(
        flag=(any_s > 0.5)[pre_roll:], filt=(any_f > 0.5)[pre_roll:], unit_size=best, n_max=n_max,
        log_likelihood=last_post.log_likelihood, seconds=time.perf_counter() - t0, passes=passes, n_hat=n_hat,
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


def m2_counts(
    *, yy: np.ndarray, lvl: np.ndarray, units: np.ndarray, offset: int
) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover - needs experiment 694's code
    """Rerun experiment 694's `M2` on one injected segment; return its flags and its weekly counts.

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


def run_v2(
    *, limit: int | None = None, cap_hours: float = 15.0, fit_kwargs: dict[str, Any] | None = None
) -> int:  # pragma: no cover - needs the stored panel
    """Run V2, K1, and V3 against experiment 694's stored assets (real data; waits for the code review).

    Reads experiment 694's panel and engine record through its own loaders, runs `grid_track` (M2) on the
    same segments and injected schedules, fits the hindsight model, compares the rerun of M2 with the
    stored `inject.pkl` and `k1.pkl`, and writes per-segment counts and the bar tables under the private
    folder only. Nothing is printed per segment. The cost gate times the first fit and stops the run when
    the projected total exceeds `cap_hours`.
    """
    import pickle

    import e00694_common as c694  # ty: ignore[unresolved-import]
    import e00694_run as r694  # ty: ignore[unresolved-import]
    import e00720_guard as guard
    import polars as pl

    fk = fit_kwargs or {}
    r694.init()
    segs = r694.segments()
    if limit:
        segs = segs[:limit]
    names = list(r694.SCHEDULES)
    stored = {
        (d["host"], d["period"], d["index"], d["sched"], d["rule"]): d["arms"]["M2"]["counts"]
        for d in pickle.loads(next(c694.DATA_DIR.glob("inject.pkl")).read_bytes())
    }
    gate = hmm.CostGate(cap_hours=cap_hours)
    n_fits = (
        len(segs) * len(names) + 2 + 8
    )  # injected runs, K1 twice, and V3's two eras (about 5 segment-lengths of steps)
    hind: dict[str, tuple[int, int, int]] = {}
    m2: dict[str, tuple[int, int, int]] = {}
    fa_h = [0, 0, 0, 0]  # flagged outside the window, half-hours outside, active days, days
    fa_m = [0, 0, 0, 0]
    rows: list[dict[str, Any]] = []
    mismatches = compared = 0
    run_gate = hmm.CostGate(cap_hours=cap_hours)
    for seg in segs:
        y0 = r694._segment_series(seg=seg)
        for name in names:
            t_iter = time.perf_counter()
            sched = r694.SCHEDULES[name]
            g, units = sched["g"], sched["units"]
            drive = np.zeros(r694.TOTAL_LEN)
            lo = r694.PRE_ROLL + r694.INJECT_OFFSET
            drive[lo : lo + len(g)] = g
            yi = c694.inject_folded(y=y0, g=drive)
            lvl = c694.quantise_spread(spread=c694.causal_spread(y=yi))[r694.PRE_ROLL :]
            yy = yi[r694.PRE_ROLL :]
            m2_flag, m2_c = m2_counts(yy=yy, lvl=lvl, units=units, offset=r694.INJECT_OFFSET)
            key_s = (seg.host, seg.period, seg.index, name, "folded")
            if key_s in stored:
                compared += 1
                mismatches += int(not np.array_equal(stored[key_s], m2_c))
            hf = hindsight_fit(
                y=yi,
                first_index=seg.start,
                pre_roll=r694.PRE_ROLL,
                gate=gate,
                projected_fits=n_fits,
                **fk,
            )
            win = slice(r694.INJECT_OFFSET, r694.INJECT_OFFSET + len(g))
            outside = np.ones(len(hf.flag), dtype=bool)
            outside[win] = False
            truth = units > 0
            cluster = f"{seg.host}|{seg.period}|{seg.index}"
            for store, flag, fa in ((hind, hf.flag, fa_h), (m2, m2_flag, fa_m)):
                if sched["primary"]:
                    t = np.array(store.get(cluster, (0, 0, 0))) + np.array(
                        f1_counts(truth=truth, flag=flag[win])
                    )
                    store[cluster] = (int(t[0]), int(t[1]), int(t[2]))
                fa[0] += int(flag[outside].sum())
                fa[1] += int(outside.sum())
                after_start = -(-(r694.INJECT_OFFSET + len(g)) // 48) * 48
                for part in (flag[: r694.INJECT_OFFSET], flag[after_start:]):
                    act, days = r694.active_days(flag=part)
                    fa[2] += act
                    fa[3] += days
            ur = (
                [unit_recall(n_hat=hf.n_hat[win], units=units, k=k) for k in range(4)]
                if hf.n_hat is not None
                else []
            )
            rows.append(
                {"cluster": cluster, "schedule": name, "primary": bool(sched["primary"]), "unit_size_hat": hf.unit_size, "seconds": hf.seconds,
                 "f1_hindsight": f1(*f1_counts(truth=truth, flag=hf.flag[win])), "f1_m2": f1(*f1_counts(truth=truth, flag=m2_flag[win])),
                 **{f"recall_units_{k}": (h / t if t else float("nan")) for k, (h, t) in enumerate(ur)}}
            )  # fmt: skip
            if np.isnan(run_gate.seconds_per_pass):
                run_gate.seconds_per_pass = (
                    time.perf_counter() - t_iter
                )  # one M2 rerun plus one hindsight fit
                run_gate.check(n_passes_total=len(segs) * len(names) + 10)
    mismatch_share = mismatches / max(compared, 1)
    fa = {
        "flagged_share": fa_h[0] / max(fa_h[1], 1),
        "active_days_per_month": fa_h[2] / max(fa_h[3], 1) * 30.0,
    }
    fa_m2 = {
        "flagged_share": fa_m[0] / max(fa_m[1], 1),
        "active_days_per_month": fa_m[2] / max(fa_m[3], 1) * 30.0,
    }
    verdict = v2_verdict(hind=hind, m2=m2, fa=fa)
    pl.DataFrame(rows).write_csv(guard.private_path(name="v2_fits.csv"))
    pl.DataFrame(
        {
            "cluster": list(hind),
            "tp": [v[0] for v in hind.values()],
            "fp": [v[1] for v in hind.values()],
            "fn": [v[2] for v in hind.values()],
        }
    ).write_csv(guard.private_path(name="v2_hindsight_counts.csv"))
    pl.DataFrame(
        {
            "cluster": list(m2),
            "tp": [v[0] for v in m2.values()],
            "fp": [v[1] for v in m2.values()],
            "fn": [v[2] for v in m2.values()],
        }
    ).write_csv(guard.private_path(name="v2_m2_counts.csv"))
    pl.DataFrame(
        [
            {
                **verdict["contrast"],
                **{f"bar_{k}": v for k, v in verdict["bar"].items()},
                "m2_rerun_mismatch_share": mismatch_share,
                "fa_flagged_share_hindsight": fa["flagged_share"],
                "fa_active_days_hindsight": fa["active_days_per_month"],
                "fa_flagged_share_m2": fa_m2["flagged_share"],
                "fa_active_days_m2": fa_m2["active_days_per_month"],
            }
        ]
    ).write_csv(guard.private_path(name="v2_bar.csv"))
    k1v3 = run_k1_v3(gate=gate, n_fits=n_fits, fit_kwargs=fk)
    pl.DataFrame([k1v3]).write_csv(guard.private_path(name="k1_v3.csv"))
    print(
        {k: verdict["contrast"][k] for k in ("point", "lower", "upper", "f1_a", "f1_b")},
        f"M2 rerun differs from the stored counts in {mismatch_share:.3f} of runs",
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


def run_k1_v3(
    *, gate: hmm.CostGate, n_fits: int, fit_kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:  # pragma: no cover - needs the stored panel
    """Run K1 (D8's own 44 days with the engine record subtracted) and V3 (D8's two engine-free periods).

    `r694.init()` must have run. K1 is placed on D8's grid exactly as `e00694_run.k1_task` does, scored
    on the half-hours where the record is valid, and compared with the stored M2 score. V3 fits the
    hindsight model with `K` forced to 3 on each control period (the strictest form of the control) and
    reports the flagged share and the active days a month, pooled over the two periods and per period.
    """
    import pickle

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
    hf = hindsight_fit(
        y=yi,
        first_index=t0 - r694.PRE_ROLL,
        pre_roll=r694.PRE_ROLL,
        gate=gate,
        projected_fits=n_fits,
        **(fit_kwargs or {}),
    )
    truth = units > 0
    tp, fp, fn = f1_counts(truth=truth[valid], flag=hf.flag[valid])
    k1_f1 = f1(tp, fp, fn)
    stored_k1 = next(
        d
        for d in pickle.loads(next(c694.DATA_DIR.glob("k1.pkl")).read_bytes())
        if d["rule"] == "folded"
    )["arms"]["M2"]
    m2_stored_f1 = c694.f1_of(tp=stored_k1["tp"], fp=stored_k1["fp"], fn=stored_k1["fn"])
    out: dict[str, Any] = {
        "k1_f1_hindsight": k1_f1,
        "k1_f1_m2_stored": m2_stored_f1,
        **{f"k1_{k}": v for k, v in bars.k1_bar(f1=k1_f1).items()},
    }
    periods = {
        "before_connection": (r694.PRE_ROLL, c694.stamp(when=c694.ENGINE_CONNECTED)),
        "added_back": (
            c694.stamp(when=c694.ENGINE_CONNECTED),
            c694.stamp(
                when=__import__("datetime").datetime(2023, 8, 15, tzinfo=__import__("datetime").UTC)
            ),
        ),
    }
    flagged = total = active = days = 0
    for name, (a, b) in periods.items():
        yy = r694.SERIES["D8"][a - r694.PRE_ROLL : b]
        h = hindsight_fit(
            y=yy,
            first_index=a - r694.PRE_ROLL,
            pre_roll=r694.PRE_ROLL,
            gate=gate,
            projected_fits=n_fits,
            **{"n_max": 3, **(fit_kwargs or {})},
        )
        fa = false_alarm(flag=h.flag, first_index_of_flag=a)
        out[f"v3_{name}_flagged_share"] = fa["flagged_share"]
        out[f"v3_{name}_active_days_per_month"] = fa["active_days_per_month"]
        flagged += int(h.flag.sum())
        total += len(h.flag)
        act, d = r694.active_days(flag=h.flag)
        active += act
        days += d
    pooled = {
        "flagged_share": flagged / max(total, 1),
        "active_days_per_month": active / max(days, 1) * 30.0,
    }
    bar = bars.v3_bar(
        flagged_share=pooled["flagged_share"],
        active_days_per_month=pooled["active_days_per_month"],
        forced_k3_flagged_share=pooled["flagged_share"],
    )
    out.update({f"v3_{k}": v for k, v in bar.items()})
    return out


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


def selftest_694() -> int:
    """Run `run_v2` end to end against stubbed experiment 694 assets (tiny synthetic series, no real data).

    The stub replaces the panel, the schedules, the engine record, the stored pickles, and the private
    folder, so `run_v2` and `run_k1_v3` execute every line on a series of about 70 days with a reduced
    residual grid. It checks that the stored M2 counts are compared, that a deliberately altered stored
    count is detected, and that the output files appear under the private folder stub.
    """
    import pickle
    import tempfile
    from datetime import timedelta
    from pathlib import Path

    import e00694_common as c694  # ty: ignore[unresolved-import]
    import e00694_run as r694  # ty: ignore[unresolved-import]
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
    r694.init = lambda: None
    seg = r694.Segment(host="D3", period="pre-solar", index=0, start=0)
    r694.segments = lambda: [seg]
    units = np.zeros(n_inj, dtype=int)
    units[100:160], units[200:260] = 1, 2
    units[260:300] = 3
    synth = np.roll(units, 40)
    r694.SCHEDULES.clear()
    r694.SCHEDULES.update(
        {
            "record": {"units": units, "g": units * 1.6, "primary": True, "k": 3, "u": 1.6},
            "synth0": {"units": synth, "g": synth * 1.6, "primary": False, "k": 3, "u": 1.6},
        }
    )
    # the stored M2 counts: right for the first schedule, wrong for the second
    y0 = r694._segment_series(seg=seg)
    stored = []
    for name, wrong in (("record", False), ("synth0", True)):
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
                "arms": {"M2": {"counts": counts + (1 if wrong else 0)}},
            }
        )
    (c694.DATA_DIR / "inject.pkl").write_bytes(pickle.dumps(stored))
    (c694.DATA_DIR / "k1.pkl").write_bytes(
        pickle.dumps([{"rule": "folded", "arms": {"M2": {"tp": 40, "fp": 5, "fn": 10, "tn": 100}}}])
    )
    start = c694.GRID_START
    rec_t = [start + timedelta(minutes=30 * (20 * 48 + i)) for i in range(n_inj)]
    rec = pl.DataFrame({"time": rec_t, "power": units * 1.6, "units": units})
    c694.load_record = lambda: rec
    c694.record_arrays = lambda: (units * 1.6, units)
    c694.ENGINE_CONNECTED = start + timedelta(days=30)
    (guard.PRIVATE_DIR).mkdir(parents=True, exist_ok=True)
    (guard.PRIVATE_DIR / "era_dates.json").write_text(
        '{"added_back_end": "' + (start + timedelta(days=45)).date().isoformat() + '"}'
    )
    code = run_v2(cap_hours=1.0, fit_kwargs={"coarse": (1.2, 1.6, 2.0), "n_bins": 9})
    out = {p.name for p in guard.PRIVATE_DIR.iterdir()}
    assert {
        "v2_fits.csv",
        "v2_bar.csv",
        "k1_v3.csv",
        "v2_m2_counts.csv",
        "v2_hindsight_counts.csv",
    } <= out, out
    bar = pl.read_csv(guard.PRIVATE_DIR / "v2_bar.csv")
    assert abs(bar["m2_rerun_mismatch_share"][0] - 0.5) < 1e-9, bar["m2_rerun_mismatch_share"][0]
    fits = pl.read_csv(guard.PRIVATE_DIR / "v2_fits.csv")
    assert fits.height == 2
    k = pl.read_csv(guard.PRIVATE_DIR / "k1_v3.csv")
    assert "v3_passed" in k.columns and "k1_f1_hindsight" in k.columns
    print(
        f"PASS V2 and K1 and V3 harness against stubbed 694 assets: {fits.height} injected runs, stored-count mismatch detected (0.5), outputs {sorted(out)}"
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
