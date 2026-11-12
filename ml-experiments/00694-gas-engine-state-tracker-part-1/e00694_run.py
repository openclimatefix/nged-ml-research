r"""Run the trackers: false-alarm pass, known-answer injections, and real D8.

    OMP_NUM_THREADS=1 PYTHONPATH=src:ml-experiments/00694-gas-engine-state-tracker-part-1 \\
        uv run --no-sync python ml-experiments/00694-*/e00694_run.py {fa|inject|d8}

Each stage writes one pickle under the experiment's data folder, atomically.
"""

import pickle
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import get_context
from typing import Any, Final

import numpy as np
from e00694_common import (
    DATA_DIR,
    ENGINE_CONNECTED,
    FADE_START,
    GRID_START,
    HOSTS,
    INJECT_OFFSET,
    NET_END,
    NET_START,
    POST_SOLAR_START,
    PRE_ROLL,
    SEGMENT_LEN,
    SOLAR_START,
    causal_spread,
    f1_counts,
    inject_folded,
    inject_signed,
    load_panel,
    load_record,
    n_true_switches,
    panel_power,
    quantise_spread,
    record_arrays,
    stamp,
    switch_errors,
    synthetic_units,
)
from e00694_trackers import (
    EM_ROUNDS,
    TrackResult,
    _grid_em,
    count_track,
    grid_evidence,
    grid_track,
    pairing_track,
    select_grid,
)

ARMS: Final[tuple[str, ...]] = ("M0", "M1", "M2", "M2-dwell")
N_WEEKS: Final[int] = 7
TOTAL_LEN: Final[int] = PRE_ROLL + SEGMENT_LEN
UNIT_MW: Final[float] = 1.97
"""One engine unit, from the record's output levels (experiment 196)."""
N_WORKERS: Final[int] = 24

PANEL: dict[int, dict] = {}
SCHEDULES: dict[str, dict[str, Any]] = {}
SERIES: dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class Segment:
    """A host segment: 28 days of pre-roll and 120 days of tracked history."""

    host: str
    period: str
    index: int
    start: int


def segments() -> list[Segment]:
    """Three segments per host in each period, except D8, which has only the pre-engine period."""
    out = []
    first_a = 0
    first_b = stamp(when=POST_SOLAR_START)
    for host in (*HOSTS, "D8"):
        periods = [("pre-solar", first_a)] + ([] if host == "D8" else [("solar", first_b)])
        for period, first in periods:
            for i in range(3):
                start = first + i * TOTAL_LEN
                if host == "D8" and start + TOTAL_LEN > stamp(when=ENGINE_CONNECTED):
                    continue
                if start + TOTAL_LEN > len(SERIES[host]):
                    continue
                if np.isfinite(SERIES[host][start : start + TOTAL_LEN]).mean() < 0.9:
                    print(f"skipped {host} {period} {i}: under 90% of half-hours present")
                    continue
                out.append(Segment(host=host, period=period, index=i, start=start))
    return out


def build_schedules() -> dict[str, dict[str, Any]]:
    """The engine record and the synthetic schedules built from its runs."""
    g, units = record_arrays()
    out: dict[str, dict[str, Any]] = {
        "record": {"units": units, "g": g, "primary": True, "k": 3, "u": UNIT_MW},
    }
    for seed in range(4):
        u = synthetic_units(units=units, seed=seed, k_cap=3)
        out[f"synth{seed}"] = {"units": u, "g": u * 2.0, "primary": True, "k": 3, "u": 2.0}
    extra = ((4, 1.0, 3), (5, 3.0, 3), (6, 2.0, 1), (7, 2.0, 2))
    for seed, unit, k_cap in extra:
        u = synthetic_units(units=units, seed=seed, k_cap=k_cap)
        out[f"synth{seed}"] = {"units": u, "g": u * unit, "primary": False, "k": k_cap, "u": unit}
    return out


def run_arms(*, yy: np.ndarray, lvl: np.ndarray) -> dict[str, dict[str, Any]]:
    """Run the four arms on one tracked reading and return the flags and the fits."""
    out: dict[str, dict[str, Any]] = {}
    out["M0"] = {"flag": pairing_track(y=yy, lvl=lvl), "k": None, "u": None}
    m1 = count_track(y=yy, lvl=lvl)
    out["M1"] = _pack(m1)
    m2 = grid_track(y=yy, lvl=lvl)
    out["M2"] = _pack(m2)
    m2d = grid_track(y=yy, lvl=lvl, dwell=True, fixed=(m2.k_max, m2.unit))
    out["M2-dwell"] = _pack(m2d)
    return out


def _pack(r: TrackResult) -> dict[str, Any]:
    return {"flag": r.p_any_smooth > 0.5, "k": r.k_max, "u": r.unit, "filt": r.p_any_filt > 0.5}


def active_days(*, flag: np.ndarray) -> tuple[int, int]:
    """Count whole UTC days in `flag` (which starts on a day boundary) with at least 4 flags."""
    n = len(flag) // 48
    if n == 0:
        return 0, 0
    blocks = flag[: n * 48].reshape(n, 48).sum(axis=1)
    return int((blocks >= 4).sum()), n


def _segment_series(*, seg: Segment) -> np.ndarray:
    return SERIES[seg.host][seg.start : seg.start + TOTAL_LEN].copy()


def fa_task(seg: Segment) -> dict[str, Any]:
    """Un-injected pass: share of half-hours flagged and active days, per arm."""
    y = _segment_series(seg=seg)
    lvl = quantise_spread(spread=causal_spread(y=y))[PRE_ROLL:]
    yy = y[PRE_ROLL:]
    arms = run_arms(yy=yy, lvl=lvl)
    out = {}
    for arm, r in arms.items():
        flag = r["flag"]
        days = active_days(flag=flag)
        out[arm] = {
            "flagged": int(flag.sum()),
            "n": len(flag),
            "active": days[0],
            "days": days[1],
            "k": r["k"],
            "u": r["u"],
        }
    return {"host": seg.host, "period": seg.period, "index": seg.index, "arms": out}


def inject_task(args: tuple[Segment, str, str]) -> dict[str, Any]:
    """Inject one schedule into one segment under one rule and score every arm."""
    seg, sched_name, rule = args
    sched = SCHEDULES[sched_name]
    g, units = sched["g"], sched["units"]
    y = _segment_series(seg=seg)
    drive = np.zeros(TOTAL_LEN)
    lo = PRE_ROLL + INJECT_OFFSET
    drive[lo : lo + len(g)] = g
    y = inject_folded(y=y, g=drive) if rule == "folded" else inject_signed(y=y, g=drive)
    lvl = quantise_spread(spread=causal_spread(y=y))[PRE_ROLL:]
    yy = y[PRE_ROLL:]
    arms = run_arms(yy=yy, lvl=lvl)
    out = {}
    win = slice(INJECT_OFFSET, INJECT_OFFSET + len(g))
    for arm, r in arms.items():
        flag = r["flag"]
        counts = np.zeros((N_WEEKS, 4), dtype=int)
        wt = units > 0
        wf = flag[win]
        for w in range(N_WEEKS):
            sl = slice(w * 336, min((w + 1) * 336, len(g)))
            counts[w] = f1_counts(truth=wt[sl], flag=wf[sl])
        unit_hit = np.array([int((wf & (units == k)).sum()) for k in range(4)])
        unit_tot = np.array([int((units == k).sum()) for k in range(4)])
        outside = np.ones(SEGMENT_LEN, dtype=bool)
        outside[win] = False
        before = flag[:INJECT_OFFSET]
        after_start = -(-(INJECT_OFFSET + len(g)) // 48) * 48
        after = flag[after_start:]
        days_b = active_days(flag=before)
        days_a = active_days(flag=after)
        out[arm] = {
            "counts": counts,
            "unit_hit": unit_hit,
            "unit_tot": unit_tot,
            "sw_err": switch_errors(truth=wt, flag=wf),
            "n_sw": n_true_switches(truth=wt),
            "fa_flagged": int(flag[outside].sum()),
            "fa_n": int(outside.sum()),
            "fa_active": days_b[0] + days_a[0],
            "fa_days": days_b[1] + days_a[1],
            "k": r["k"],
            "u": r["u"],
        }
    return {
        "host": seg.host,
        "period": seg.period,
        "index": seg.index,
        "sched": sched_name,
        "rule": rule,
        "arms": out,
    }


def example_task() -> dict[str, Any]:
    """One injected segment kept whole, for the example figure: D3, first segment, the record."""
    seg = next(s for s in segments() if s.host == "D3" and s.period == "pre-solar")
    sched = SCHEDULES["record"]
    g, units = sched["g"], sched["units"]
    y = _segment_series(seg=seg)
    drive = np.zeros(TOTAL_LEN)
    lo = PRE_ROLL + INJECT_OFFSET
    drive[lo : lo + len(g)] = g
    clean = y[PRE_ROLL:].copy()
    y = inject_folded(y=y, g=drive)
    lvl = quantise_spread(spread=causal_spread(y=y))[PRE_ROLL:]
    yy = y[PRE_ROLL:]
    arms = run_arms(yy=yy, lvl=lvl)
    truth = np.zeros(SEGMENT_LEN, dtype=int)
    truth[INJECT_OFFSET : INJECT_OFFSET + len(g)] = units
    return {
        "y": yy,
        "clean": clean,
        "truth": truth,
        "arms": {a: {"flag": r["flag"], "k": r["k"], "u": r["u"]} for a, r in arms.items()},
    }


def save(*, name: str, obj: object) -> None:
    """Write a pickle atomically."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DATA_DIR / f"{name}.tmp"
    tmp.write_bytes(pickle.dumps(obj))
    tmp.replace(DATA_DIR / f"{name}.pkl")


def init() -> None:
    """Load the panel and the schedules into module globals (inherited by forked workers)."""
    PANEL.update(load_panel())
    for label in (*HOSTS, "D8"):
        SERIES[label] = panel_power(panel=PANEL, label=label)
    SCHEDULES.update(build_schedules())


def stage_fa() -> None:
    """Un-injected false-alarm pass over every segment."""
    segs = segments()
    t0 = time.time()
    with get_context("fork").Pool(N_WORKERS) as pool:
        res = pool.map(fa_task, segs, chunksize=1)
    save(name="fa", obj=res)
    print(f"fa: {len(res)} segments in {time.time() - t0:.0f} s")


def stage_inject() -> None:
    """Known-answer injections: every schedule, segment, and rule."""
    segs = segments()
    tasks = [(s, name, rule) for s in segs for name in SCHEDULES for rule in ("folded", "signed")]
    t0 = time.time()
    with get_context("fork").Pool(N_WORKERS) as pool:
        res = pool.map(inject_task, tasks, chunksize=1)
    save(name="inject", obj=res)
    print(f"inject: {len(res)} runs in {time.time() - t0:.0f} s")


def k1_task(rule: str) -> dict[str, Any]:
    """K1: D8's own reading in the record's 44 days, with the record subtracted.

    The stored record has one gap of 12 half-hours, so it is placed on D8's half-hour grid by its
    timestamps, the gap holds no injection, and the gap half-hours are not scored.
    """
    frame = load_record().sort("time")
    idx = ((frame["time"].dt.epoch("s").to_numpy() - int(GRID_START.timestamp())) // 1800).astype(
        int
    )
    t0 = int(idx[0])
    n = int(idx[-1] - t0 + 1)
    g_rec, u_rec = record_arrays()
    g = np.zeros(n)
    units = np.zeros(n, dtype=int)
    valid = np.zeros(n, dtype=bool)
    g[idx - t0], units[idx - t0], valid[idx - t0] = g_rec, u_rec, True
    y = SERIES["D8"][t0 - PRE_ROLL : t0 + n].copy()
    drive = np.zeros(len(y))
    drive[PRE_ROLL:] = g
    y = inject_folded(y=y, g=drive) if rule == "folded" else inject_signed(y=y, g=drive)
    lvl = quantise_spread(spread=causal_spread(y=y))[PRE_ROLL:]
    yy = y[PRE_ROLL:]
    arms = run_arms(yy=yy, lvl=lvl)
    truth = units > 0
    out = {}
    for arm, r in arms.items():
        tp, fp, fn, tn = f1_counts(truth=truth[valid], flag=r["flag"][valid])
        out[arm] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "k": r["k"],
            "u": r["u"],
            "n_gap": int((~valid).sum()),
        }
    return {"rule": rule, "arms": out}


def d8_era(args: tuple[str, datetime, datetime]) -> dict[str, Any]:
    """Run M1, M2, and M2 without the fold on one era of real D8 (spread from all of D8)."""
    label, lo, hi = args
    y_all = SERIES["D8"]
    lvl_all = quantise_spread(spread=causal_spread(y=y_all))
    a, b = stamp(when=lo), min(stamp(when=hi), len(y_all))
    yy, lvl = y_all[a:b], lvl_all[a:b]
    m1 = count_track(y=yy, lvl=lvl)
    m2 = grid_track(y=yy, lvl=lvl)
    m2s = grid_track(y=yy, lvl=lvl, fold=False)
    base = np.datetime64(GRID_START.replace(tzinfo=None), "s")
    month = (base + np.arange(a, b) * np.timedelta64(1800, "s")).astype("datetime64[M]")
    return {
        "label": label,
        "start": a,
        "month": month.astype(str),
        "M1": {"k": m1.k_max, "u": m1.unit, "flag": m1.p_any_smooth > 0.5, "k_path": m1.k_smooth},
        "M2": {
            "k": m2.k_max,
            "u": m2.unit,
            "flag": m2.p_any_smooth > 0.5,
            "filt": m2.p_any_filt > 0.5,
            "k_path": m2.k_smooth,
            "transition": m2.transition,
        },
        "M2s": {"k": m2s.k_max, "u": m2s.unit, "flag": m2s.p_any_smooth > 0.5},
        "y": yy,
    }


def p5_task(unused: int) -> dict[str, Any]:
    """P5: fit on odd months of D8's net era, score even months with and without the fold."""
    y_all = SERIES["D8"]
    spread = causal_spread(y=y_all)
    lvl_all = quantise_spread(spread=spread)
    a, b = stamp(when=NET_START), stamp(when=NET_END)
    base = np.datetime64(GRID_START.replace(tzinfo=None), "s")
    idx = np.arange(a, b)
    month = (base + idx * np.timedelta64(1800, "s")).astype("datetime64[M]")
    labels = np.unique(month)
    odd, even = labels[::2], labels[1::2]
    fit_mask = np.isin(month, odd)
    y_fit, lvl_fit = y_all[a:b][fit_mask], lvl_all[a:b][fit_mask]
    out: dict[str, Any] = {}
    for name, fold in (("fold", True), ("signed", False)):
        k_sel, u = select_grid(y=y_fit, lvl=lvl_fit, fold=fold)
        k = max(k_sel, 1)
        a_k, _ = _grid_em(
            y=y_fit, lvl=lvl_fit, k_max=k, unit=u, fold=fold, dwell=False, rounds=EM_ROUNDS
        )
        per_month = {}
        for m in even:
            sel = month == m
            ev = grid_evidence(
                use_occupancy=False,
                y=y_all[a:b][sel],
                lvl=lvl_all[a:b][sel],
                k_max=k,
                unit=u,
                fold=fold,
                a_k=a_k,
            )
            per_month[str(m)] = (float(ev), int(sel.sum()))
        out[name] = {"k": k, "u": u, "per_month": per_month}
    return out


def main() -> None:
    """Dispatch on the stage named on the command line."""
    stage = sys.argv[1]
    init()
    if stage == "fa":
        stage_fa()
    elif stage == "inject":
        stage_inject()
    elif stage == "example":
        save(name="example", obj=example_task())
    elif stage == "k1":
        res = [k1_task(rule) for rule in ("folded", "signed")]
        save(name="k1", obj=res)
    elif stage == "d8":
        eras = [
            ("net", NET_START, NET_END),
            ("solar", SOLAR_START, FADE_START),
            ("fade", FADE_START, datetime(2026, 7, 1, tzinfo=UTC)),
        ]
        with get_context("fork").Pool(3) as pool:
            res = pool.map(d8_era, eras)
        save(name="d8", obj=res)
        save(name="p5", obj=p5_task(0))
    else:
        raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
