# ruff: noqa: E501, D205, D107, PLR0915
"""Experiment 592: score the six gates (`e00592_common`) on experiment 584's C1 forecast.

No fit and no GPU. Writes, under `data/ml-experiments/00592-*/`:

- `bank_event_phase.csv`, `bank_summary.csv`: the injected bank (experiment 581's events, C1 forecasts
  from experiment 584), per event, phase, lead band, and gate; and summarised with event bootstraps.
- `trigger_times.csv`: per injected or verified event and gate, the days from onset to the first gate-on
  origin day and from the end to the first release, and the gate's on-state around the event.
- `real_event_phase.csv`: the verified December pair per phase, lead band, and gate.
- `event_free.csv`, `false_triggers.csv`: the false-alarm cost (pinball loss at q05 and q95 against G0)
  and the false triggers per Primary-year on event-free rows.

The band under every gate is experiment 584's C2: the widest of C1's band and its lags-masked twin's
band while the gate is on and for 7 days after it releases, C1 elsewhere.
"""

import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00582_arms as a  # ty: ignore[unresolved-import]
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00583_score as s  # ty: ignore[unresolved-import]
import e00584_score as k584  # ty: ignore[unresolved-import]
import e00592_common as g  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

GROUPS_SCORED = s.GROUPS_SCORED
BANDS = s.BANDS
PHASES = ("before", "days0-3", "days4-end", "after0-7", "after7-14")
s.N_COUNTED = 10  # five gates against G0 x q05 and q95


def blocks(*, v: np.ndarray, m: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Return the row-level metrics of C1 (`V`) and the union band (`Union`) against `y`."""
    return {
        "V": c.row_metrics(y=y, q=v),
        "Union": c.row_metrics(y=y, q=a.union_band(lag=v, masked=m)),
    }


def gate_metrics(*, pre: dict[str, np.ndarray], mode: np.ndarray) -> np.ndarray:
    """Return the row metrics under one gate: the union band's on rows in event mode, C1's elsewhere."""
    return np.where(mode[:, None], pre["Union"], pre["V"])


class Inputs:
    """The table, the three real gap arrays, and the gate thresholds."""

    def __init__(self) -> None:
        k584.setup("c1")
        self.t = c.X4Table()
        gi = c.gate_inputs(t=self.t)
        self.gi = gi
        self.origin_days = gi["origin_days"]
        self.n_days = gi["gap"].shape[1]
        e2 = g.lag_free_median(t=self.t)
        self.e2_share = float(np.isfinite(e2).mean())
        self.fresh = g.FreshGap(t=self.t, e2=e2, n_days=self.n_days)
        self.fresh_gap = self.fresh.gap()
        self.fresh_thr = g.threshold(gap=self.fresh_gap, near=gi["near"])
        self.nb = self.t.neighbours()
        self.states = g.all_gates(
            old_gap=gi["gap"], old_thr=gi["thr"], fresh_gap=self.fresh_gap,
            fresh_thr=self.fresh_thr, origin_days=self.origin_days, neighbours=self.nb,
        )  # fmt: skip
        self.masks = g.arm_masks(states=self.states)


def day_index(*, t: c.X4Table, rows: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """Return `arr[Primary, origin day]` for each of `rows`."""
    return arr[t.prim[rows], t.origin_day[rows]]


# ---------------------------------------------------------------- event-free rows


def false_triggers(*, inp: Inputs) -> pl.DataFrame:
    """Return, per gate and group, false triggers per Primary-year: gate turn-ons on event-free
    (Primary, origin day) pairs whose gate input is present, and the share of those days in event mode.
    """
    near = s.dilated_near(near=inp.gi["near"])
    rows = []
    for gate, (on, _snap) in inp.states.items():
        gap = inp.gi["gap"] if gate in {"G0", "G1", "G2"} else inp.fresh_gap
        m = {"G0": 3, "G1": 2, "G2": 1}.get(gate, 1)
        lag = c.LAG_DAYS if gate in {"G0", "G1", "G2"} else 0
        mode = inp.masks[gate][0]
        turn_on = on & ~np.roll(on, 1, axis=1)
        for grp in (*GROUPS_SCORED, "D15 and D18", "all 16", "all with a neighbour"):
            ps = [
                p
                for p in range(16)
                if grp == "all 16"
                or (grp == "all with a neighbour" and inp.nb.get(p))
                or inp.t.group[p] == grp
            ]
            pairs = [
                (p, d)
                for p in ps
                for d in inp.origin_days
                if not near[p, d] and np.isfinite(gap[p, d - lag - m + 1 : d - lag + 1]).all()
            ]
            if not pairs:
                continue
            pp = np.array(pairs)
            rows.append(
                {
                    "gate": gate, "group": grp, "event_free_origin_days": len(pairs),
                    "false_triggers": int(turn_on[pp[:, 0], pp[:, 1]].sum()),
                    "false_triggers_per_primary_year": float(
                        turn_on[pp[:, 0], pp[:, 1]].sum() / g.years_of(pairs=len(pairs))
                    ),
                    "share_days_event_mode": float(mode[pp[:, 0], pp[:, 1]].mean()),
                }
            )  # fmt: skip
    return pl.DataFrame(rows)


def event_free(*, inp: Inputs) -> pl.DataFrame:
    """Return pinball loss at q05 and q95 (points of capacity) on event-free rows at 24-120 h per gate
    and group, with the difference to G0 and its month-block bootstrap interval at the Bonferroni
    level of `s.N_COUNTED` counted contrasts.
    """
    t = inp.t
    near = s.dilated_near(near=inp.gi["near"])
    pre = blocks(v=t.arms["V"], m=t.arms["M"], y=t.obs)
    free = t.free & ~near[t.prim, t.origin_day]
    months = sorted(set(t.month))
    out = []
    cols = {k: c.METRICS.index(k) for k in ("pin05", "pin95", "width90", "cover90")}
    per_gate = {
        gate: gate_metrics(
            pre=pre, mode=day_index(t=t, rows=np.arange(t.n), arr=inp.masks[gate][0])
        )
        for gate in g.GATES
    }
    for grp in (*GROUPS_SCORED, "both groups", "D15 and D18"):
        gsel = (
            np.isin(t.group_of_row, GROUPS_SCORED)
            if grp == "both groups"
            else t.group_of_row == grp
        )
        sel = free & gsel & (t.band == 1)
        if sel.sum() < 500:
            continue
        for gate in g.GATES:
            r = {
                "gate": gate,
                "group": grp,
                "n_rows": int(sel.sum()),
                "counted": grp == "both groups",
            }
            for k, j in cols.items():
                scale = 1.0 if k == "cover90" else 100.0
                r[k] = float(per_gate[gate][sel, j].mean() * scale)
            for k in ("pin05", "pin95"):
                j = cols[k]
                diffs = np.array(
                    [
                        (
                            per_gate[gate][sel & (t.month == m), j]
                            - per_gate["G0"][sel & (t.month == m), j]
                        ).mean()
                        * 100.0
                        for m in months
                        if (sel & (t.month == m)).any()
                    ]
                )
                lo, hi = s.bonferroni_interval(diffs=diffs)
                r[f"{k}_minus_G0"] = float(diffs.mean())
                r[f"{k}_minus_G0_lo"], r[f"{k}_minus_G0_hi"], r[f"{k}_mde"] = lo, hi, (hi - lo) / 2
            out.append(r)
    return pl.DataFrame(out)


# ---------------------------------------------------------------- the verified pair


def real_events(*, inp: Inputs) -> pl.DataFrame:
    """Return the verified events' per-phase, per-band metrics under each gate (descriptive)."""
    t = inp.t
    pre = blocks(v=t.arms["V"], m=t.arms["M"], y=t.nra)
    ev = sc.attach_depth(ev=sc.merged_events(), t=t)
    out = []
    for e in ev.iter_rows(named=True):
        sgn = 1.0 if e["depth_share"] > 0 else -1.0
        for phase in PHASES:
            lo, hi = sc.window(onset=e["onset_idx"], end=e["end_idx"], phase=phase)
            for band in (0, 1):
                sel = np.flatnonzero(
                    (t.prim == e["primary"]) & (t.idx >= lo) & (t.idx < hi) & (t.band == band)
                )
                if len(sel) < 100:
                    continue
                for gate in g.GATES:
                    mode = day_index(t=t, rows=sel, arr=inp.masks[gate][0])
                    mm = gate_metrics(pre={k: v[sel] for k, v in pre.items()}, mode=mode)
                    out.append(
                        {
                            "event": e["name"], "group": e["group"], "label": e["label"],
                            "length_days": e["length_days"], "phase": phase, "band": BANDS[band],
                            "gate": gate, "n_rows": len(sel),
                            "share_origin_days_event_mode": float(mode.mean()),
                            **s.metric_means(rows_metrics=mm, y_sign=sgn),
                        }
                    )  # fmt: skip
    return pl.DataFrame(out)


def trigger_rows(
    *, ev: pl.DataFrame, states: dict, source: str, extra: dict, rel: tuple[int, int] = (-7, 14)
) -> list[dict]:
    """Return, per event and gate, the delay from onset to the first gate-on origin day, the delay
    from the end to the first release, and the gate's on-state from `rel[0]` days before the onset to
    `rel[1]` days after the end (a string of 0 and 1).
    """
    out = []
    for gate, (on, snap) in states.items():
        for r in s.trigger_times(ev=ev, on=on, snap=snap, label=source):
            p, d0, d1 = r["primary"], r["onset_day"], r["end_day"]
            days = range(d0 + rel[0], d1 + rel[1] + 1)
            trace = "".join("1" if 0 <= d < on.shape[1] and on[p, d] else "0" for d in days)
            out.append({**r, "gate": gate, **extra, "on_trace": trace, "trace_start_rel": rel[0]})
    return out


# ---------------------------------------------------------------- the injected bank


def score_bank(*, inp: Inputs) -> tuple[pl.DataFrame, list[dict]]:
    """Return the per-event bank table and the trigger rows (the gates see the contaminated lag).

    Every injected event adds `sign * depth` to the day-old-lag gap on its days (experiment 583's
    rule) and to the fresh-power readings on exactly the half-hours it covers. For G4 the same
    load leaves a neighbour (the `(pass + Primary)`-th of the Primary's metered neighbours), so the
    neighbour's fresh readings get `-sign * depth * capacity(Primary) / capacity(neighbour)`.
    """
    t, gi = inp.t, inp.gi
    ev_all = pl.read_parquet(s.BANK_PARQUET)
    kk_all = s.key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(kk_all)
    sorted_keys = kk_all[order]
    months = [m for m in c.MONTHS if m in set(ev_all["month"].unique().to_list())]
    rows_out: list[dict] = []
    trig_rows: list[dict] = []
    for m in months:
        in_m = t.month == m
        days_m = np.unique(t.idx[in_m] // 48)
        origin_days_m = sorted({int(d) for d in np.unique(t.origin_day[in_m])})
        sl = slice(days_m.min(), days_m.max() + 1)
        for r in range(bank.N_PASSES):
            ev = ev_all.filter((pl.col("month") == m) & (pl.col("pass") == r)).filter(
                pl.col("primary").is_in([p for p in range(16) if t.group[p] in GROUPS_SCORED])
            )
            if ev.height == 0:
                continue
            lo_p = {e["primary"]: e["onset_idx"] - 7 * 48 for e in ev.iter_rows(named=True)}
            hi_p = {e["primary"]: e["end_idx"] + 14 * 48 for e in ev.iter_rows(named=True)}
            lo = np.array([lo_p.get(p, 1 << 40) for p in range(16)])[t.prim]
            hi = np.array([hi_p.get(p, -1) for p in range(16)])[t.prim]
            rows = np.flatnonzero(in_m & t.free & (t.idx % 2 == 0) & (t.idx >= lo) & (t.idx < hi))
            vin = s.load_injected(
                t=t, month=m, r=r, rows=rows, order=order, sorted_keys=sorted_keys
            )
            ok = np.isfinite(vin).all(axis=1)
            rows, vin = rows[ok], vin[ok]
            if len(rows) == 0:
                continue
            old = np.full_like(gi["gap"], np.nan)
            old[:, sl] = gi["gap"][:, sl]
            shifts: list[tuple[int, int, int, float]] = []
            for e in ev.iter_rows(named=True):
                p, shift = e["primary"], e["sign"] * e["depth"]
                old[p, e["onset_idx"] // 48 : e["end_idx"] // 48] += shift
                shifts.append((p, e["onset_idx"], e["end_idx"], float(shift)))
                nbs = inp.nb.get(p, [])
                if nbs:
                    n = nbs[(r + p) % len(nbs)]
                    shifts.append(
                        (
                            n,
                            e["onset_idx"],
                            e["end_idx"],
                            float(-shift * t.capacity[p] / t.capacity[n]),
                        )
                    )
            fresh = inp.fresh.gap(shifts=shifts)
            # a reading on or before the month's first origin day would use the previous month's last forecast, stored for some months only
            fresh[:, : origin_days_m[0] + 1] = np.nan
            fresh[:, origin_days_m[-1] + 1 :] = np.nan
            states = g.all_gates(
                old_gap=old, old_thr=gi["thr"], fresh_gap=fresh, fresh_thr=inp.fresh_thr,
                origin_days=origin_days_m, neighbours=inp.nb,
            )  # fmt: skip
            masks = g.arm_masks(states=states)
            pre = blocks(v=vin, m=t.arms["M"][rows], y=t.obs[rows])
            for e in ev.iter_rows(named=True):
                p, sgn, depth = e["primary"], float(e["sign"]), float(e["depth"])
                evf = pl.DataFrame(
                    [{**e, "length_days": float(e["length_days"]), "first_day": int(days_m.min())}]
                )
                trig_rows.extend(
                    trigger_rows(
                        ev=evf, states=states, source=f"bank {m} r{r}",
                        extra={"month": m, "pass": r, "has_neighbour": bool(inp.nb.get(p)),
                               "depth": depth, "sign": int(sgn)},
                    )
                )  # fmt: skip
                for phase in PHASES:
                    wlo, whi = sc.window(onset=e["onset_idx"], end=e["end_idx"], phase=phase)
                    for band in (0, 1):
                        sel = np.flatnonzero(
                            (t.prim[rows] == p)
                            & (t.idx[rows] >= wlo)
                            & (t.idx[rows] < whi)
                            & (t.band[rows] == band)
                        )
                        if len(sel) < 60:
                            continue
                        for gate in g.GATES:
                            mode = masks[gate][0][t.prim[rows][sel], t.origin_day[rows][sel]]
                            mm = gate_metrics(pre={k: v[sel] for k, v in pre.items()}, mode=mode)
                            rows_out.append(
                                {
                                    "month": m, "pass": r, "primary": p, "group": str(t.group[p]),
                                    "sign": int(sgn), "depth": depth,
                                    "length_days": int(e["length_days"]), "phase": phase,
                                    "band": BANDS[band], "arm": gate, "n_rows": len(sel),
                                    "share_origin_days_event_mode": float(mode.mean()),
                                    **s.metric_means(rows_metrics=mm, y_sign=sgn),
                                }
                            )  # fmt: skip
            print("bank", m, r, flush=True)
    return pl.DataFrame(rows_out), trig_rows


def main() -> None:
    """Score every gate and write the tables."""
    g.WORK.mkdir(parents=True, exist_ok=True)
    inp = Inputs()
    print("rows", inp.t.n, "E2 share of rows", inp.e2_share, flush=True)
    print("thresholds old", np.round(inp.gi["thr"], 3), "fresh", np.round(inp.fresh_thr, 3))
    ft = false_triggers(inp=inp)
    ft.write_csv(g.WORK / "false_triggers.csv")
    print(ft.filter(pl.col("group") == "all 16"))
    event_free(inp=inp).write_csv(g.WORK / "event_free.csv")
    real_events(inp=inp).write_csv(g.WORK / "real_event_phase.csv")
    ev = sc.attach_depth(ev=sc.merged_events(), t=inp.t)
    scored = {int(d) for d in np.unique(inp.t.idx // 48)}
    ev = ev.filter(
        pl.col("onset_idx").map_elements(lambda i: int(i) // 48 in scored, return_dtype=pl.Boolean)
    )
    trig = trigger_rows(ev=ev, states=inp.states, source="verified", extra={})
    tab, trig_bank = score_bank(inp=inp)
    tab.write_csv(g.WORK / "bank_event_phase.csv")
    s.summarise_bank(tab=tab).write_csv(g.WORK / "bank_summary.csv")
    pl.DataFrame([*trig, *trig_bank], infer_schema_length=None).write_csv(
        g.WORK / "trigger_times.csv"
    )
    np.savez(
        g.WORK / "gate_state.npz",
        fresh_gap=inp.fresh_gap, old_gap=inp.gi["gap"], old_thr=inp.gi["thr"],
        fresh_thr=inp.fresh_thr, **{f"{k}_on": v[0] for k, v in inp.states.items()},
        **{f"{k}_snap": v[1] for k, v in inp.states.items()},
    )  # fmt: skip
    print("done", tab.height, "bank rows")


if __name__ == "__main__":
    main()
