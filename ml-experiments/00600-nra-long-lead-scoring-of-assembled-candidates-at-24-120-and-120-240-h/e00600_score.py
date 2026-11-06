# ruff: noqa: E501, D205, PLR0915
"""Experiment 600: score C0, C1, A1 and their G4 assembled forecasts, and E2, at 0-24, 24-120, and 120-240 h.

No fit and no GPU. This is experiment 597's scoring with the stored forecasts read to 240 h and the
lead split into three bands (`e00600_common`). For one arm (`c0`, `c1`, or `a1`) it writes, under
`data/ml-experiments/00600-*/<arm>/`: `gate_check.json`, `bank_event_phase.csv`, `bank_summary.csv`,
`trigger_times.csv`, `false_triggers.csv`, `real_event_phase.csv`, `event_free.csv`, and
`event_free_rows.npz`, as in experiment 597. The system `E2` (the lag-free network of experiment 565)
is scored alongside, written by the `c1` run only.

Usage: python e00600_score.py ARM   (ARM: c0, c1, or a1)
"""

import json
import sys

import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00582_arms as a  # ty: ignore[unresolved-import]
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00583_score as s  # ty: ignore[unresolved-import]
import e00592_score as k592  # ty: ignore[unresolved-import]
import e00597_common as k  # ty: ignore[unresolved-import]
import e00600_common as k600  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

BANDS = k600.BANDS
BAND_IDS = (0, 1, 2)
s.N_COUNTED = k600.N_COUNTED
SYSTEMS = ("plain", "G4")
REFERENCE = "E2"
"""The lag-free network of experiment 565, scored as a reference when `Inputs.e2` is present."""
TAIL_METRICS = ("pin01", "pin05", "pin95", "pin99", "cover98", "nra_side99")
SIDE = {1: c.IQ[0.05], -1: c.IQ[0.95]}
"""The NRA-side quantile index by event sign: a load-raising shift (+1) lifts the forecast, so the NRA lies below it (q05 side)."""


def systems(
    *, v: np.ndarray, m: np.ndarray, wide: np.ndarray, e2: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Return the quantiles of the plain network, the assembled forecast, and (if given) E2."""
    out = {"plain": v, "G4": k.assemble(v=v, m=m, wide=wide)}
    if e2 is not None:
        out[REFERENCE] = np.nan_to_num(e2, nan=0.0)
    return out


class Inputs(k600.Inputs):
    """Experiment 600's inputs plus E2's quantiles aligned to the table (only for the `c1` run)."""

    def __init__(self, *, arm: str) -> None:
        """Build the inputs of `arm` and, for `c1`, align E2's quantiles to the rows."""
        super().__init__(arm=arm)
        self.e2 = k600.lag_free_quantiles(t=self.t) if arm == "c1" else None
        self.has_e2 = None if self.e2 is None else np.isfinite(self.e2).all(axis=1)
        if self.e2 is not None:
            print("E2 rows present:", float(self.has_e2.mean()), flush=True)

    def usable(self, *, name: str) -> np.ndarray | bool:
        """Return the mask of table rows on which system `name` exists."""
        return self.has_e2 if name == REFERENCE else True


def system_names(*, inp: Inputs) -> tuple[str, ...]:
    """Return the systems scored for this arm."""
    return (*SYSTEMS, REFERENCE) if inp.e2 is not None else SYSTEMS


def row_stats(*, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return (rows, len(METRICS) + 1) row metrics with the CRPS (share of capacity) appended."""
    return np.concatenate(
        [c.row_metrics(y=y, q=q), a.crps_from_quantiles(y=y, q=q)[:, None].astype(np.float32)],
        axis=1,
    )


STAT_NAMES = (*c.METRICS, "crps")


def means(*, rows_metrics: np.ndarray, sgn: float | None) -> dict[str, float]:
    """Return the mean of every row metric; with `sgn`, also the NRA-side shares of the event."""
    m = dict(zip(STAT_NAMES, rows_metrics.mean(axis=0).tolist(), strict=True))
    if sgn is not None:
        m["nra_side95"] = m["ge05"] if sgn > 0 else m["le95"]
        m["nra_side99"] = m["ge01"] if sgn > 0 else m["le99"]
    return m


def follow(*, q: np.ndarray, twin: np.ndarray, sgn: float, depth: float) -> dict[str, float]:
    """Return the median follow ratio and the NRA-side tail follow ratio: the mean movement of the
    quantile away from the clean-lag twin, in the direction of the event, as a share of its depth.
    """
    j = SIDE[int(sgn)]
    return {
        "follow_q50": float(((q[:, c.IQ[0.5]] - twin[:, c.IQ[0.5]]) * sgn).mean() / depth),
        "follow_safety": float(((q[:, j] - twin[:, j]) * sgn).mean() / depth),
    }


# ---------------------------------------------------------------- event-free rows


def event_free(*, inp: Inputs) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Return CRPS, coverage, and pinball loss on event-free rows (no labelled or verified event
    within 7 days, 24-120 h and 0-24 h), per group and system, with the G4 minus plain contrast.
    """
    t = inp.t
    wide = inp.mode[t.prim, t.origin_day]
    sy = systems(v=t.arms["V"], m=t.arms["M"], wide=wide, e2=inp.e2)
    st = {n: row_stats(y=t.obs, q=q) for n, q in sy.items()}
    months = sorted(set(t.month))
    j = {n: STAT_NAMES.index(n) for n in ("cover90", "pin05", "pin95", "width90", "crps")}
    out = []
    for grp in (*k.GROUPS_SCORED, "both groups", "D15 and D18"):
        gsel = (
            np.isin(t.group_of_row, k.GROUPS_SCORED)
            if grp == "both groups"
            else t.group_of_row == grp
        )
        for band in BAND_IDS:
            sel = t.free & gsel & (t.band == band)
            if sel.sum() < 500:
                continue
            for name in system_names(inp=inp):
                sel_n = sel & inp.usable(name=name)
                r = {
                    "system": name, "group": grp, "band": BANDS[band], "n_rows": int(sel_n.sum()),
                    "share_rows_wide": float(wide[sel].mean()) if name == "G4" else 0.0,
                    "counted": grp == "both groups" and band == 2 and name == "G4",
                }  # fmt: skip
                for key, col in j.items():
                    r[key] = float(st[name][sel_n, col].mean() * 100.0)
                r["cover90"] /= 100.0  # ty: ignore[unsupported-operator]
                if name == "G4":
                    for key in ("pin05", "pin95", "crps"):
                        d = np.array(
                            [
                                (st["G4"][sel & (t.month == m), j[key]] - st["plain"][sel & (t.month == m), j[key]]).mean() * 100.0
                                for m in months
                                if (sel & (t.month == m)).any()
                            ]
                        )  # fmt: skip
                        lo, hi = s.bonferroni_interval(diffs=d)
                        r[f"{key}_minus_plain"], r[f"{key}_lo"], r[f"{key}_hi"] = (
                            float(d.mean()),
                            lo,
                            hi,
                        )
                        r[f"{key}_mde"] = (hi - lo) / 2
                out.append(r)
    rows = {
        "prim": t.prim, "origin_idx": t.origin_idx, "idx": t.idx, "band": t.band,
        "free": t.free, "group": t.group_of_row.astype(str), "month": t.month.astype(str),
        "crps_g4": st["G4"][:, j["crps"]], "crps_plain": st["plain"][:, j["crps"]],
        "pin05_g4": st["G4"][:, j["pin05"]], "pin95_g4": st["G4"][:, j["pin95"]],
    }  # fmt: skip
    return pl.DataFrame(out, infer_schema_length=None), rows


# ---------------------------------------------------------------- the verified pair


def real_events(*, inp: Inputs) -> pl.DataFrame:
    """Return the verified events' per-phase, per-band metrics for the plain network and G4."""
    t = inp.t
    wide = inp.mode[t.prim, t.origin_day]
    sy = systems(v=t.arms["V"], m=t.arms["M"], wide=wide, e2=inp.e2)
    st = {n: row_stats(y=t.nra, q=q) for n, q in sy.items()}
    ev = sc.attach_depth(ev=sc.merged_events(), t=t)
    out = []
    for e in ev.iter_rows(named=True):
        sgn = 1.0 if e["depth_share"] > 0 else -1.0
        for phase in k.PHASES:
            lo, hi = sc.window(onset=e["onset_idx"], end=e["end_idx"], phase=phase)
            for band in BAND_IDS:
                sel = np.flatnonzero(
                    (t.prim == e["primary"]) & (t.idx >= lo) & (t.idx < hi) & (t.band == band)
                )
                if len(sel) < 100:
                    continue
                for name in system_names(inp=inp):
                    sel_n = sel[inp.has_e2[sel]] if name == REFERENCE else sel
                    if len(sel_n) < 100:
                        continue
                    out.append(
                        {
                            "event": e["name"], "group": e["group"], "length_days": e["length_days"],
                            "phase": phase, "band": BANDS[band], "system": name, "n_rows": len(sel_n),
                            "share_origin_days_wide": float(wide[sel_n].mean()),
                            **means(rows_metrics=st[name][sel_n], sgn=sgn),
                        }
                    )  # fmt: skip
    return pl.DataFrame(out)


# ---------------------------------------------------------------- the injected bank


def score_bank(*, inp: Inputs) -> tuple[pl.DataFrame, list[dict]]:
    """Return the per-event bank table and the G4 trigger rows (experiment 592's injection: the gate
    sees the contaminated lag gap and the shifted fresh readings, with the mirrored neighbour).
    """
    t, gi = inp.t, inp.gi
    ev_all = pl.read_parquet(s.BANK_PARQUET)
    kk_all = s.key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(kk_all)
    sorted_keys = kk_all[order]
    months = [m for m in c.MONTHS if m in set(ev_all["month"].unique().to_list())]
    rows_out: list[dict] = []
    trig_rows: list[dict] = []
    real_wide_all = inp.mode[t.prim, t.origin_day]
    for m in months:
        in_m = t.month == m
        days_m = np.unique(t.idx[in_m] // 48)
        origin_days_m = sorted({int(d) for d in np.unique(t.origin_day[in_m])})
        sl = slice(days_m.min(), days_m.max() + 1)
        for r in range(bank.N_PASSES):
            ev = ev_all.filter((pl.col("month") == m) & (pl.col("pass") == r)).filter(
                pl.col("primary").is_in([p for p in range(16) if t.group[p] in k.GROUPS_SCORED])
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
                        (n, e["onset_idx"], e["end_idx"], float(-shift * t.capacity[p] / t.capacity[n]))
                    )  # fmt: skip
            fresh = inp.fresh.gap(shifts=shifts)
            fresh[:, : origin_days_m[0] + 1] = np.nan
            fresh[:, origin_days_m[-1] + 1 :] = np.nan
            states = k592.g.all_gates(
                old_gap=old, old_thr=gi["thr"], fresh_gap=fresh, fresh_thr=inp.fresh_thr,
                origin_days=origin_days_m, neighbours=inp.nb,
            )  # fmt: skip
            wide_inj = k592.g.arm_masks(states=states)["G4"][0]
            y = t.obs[rows]
            v_clean, m_q = t.arms["V"][rows], t.arms["M"][rows]
            e2_rows = None if inp.e2 is None else inp.e2[rows]
            q_inj = systems(
                v=vin, m=m_q, wide=wide_inj[t.prim[rows], t.origin_day[rows]], e2=e2_rows
            )
            q_twin = systems(v=v_clean, m=m_q, wide=real_wide_all[rows], e2=e2_rows)
            st = {n: row_stats(y=y, q=q) for n, q in q_inj.items()}
            for e in ev.iter_rows(named=True):
                p, sgn, depth = e["primary"], float(e["sign"]), float(e["depth"])
                evf = pl.DataFrame(
                    [{**e, "length_days": float(e["length_days"]), "first_day": int(days_m.min())}]
                )
                trig_rows.extend(
                    k592.trigger_rows(
                        ev=evf, states={"G4": states["G4"]}, source=f"bank {m} r{r}",
                        extra={"month": m, "pass": r, "has_neighbour": bool(inp.nb.get(p)),
                               "depth": depth, "sign": int(sgn)},
                    )
                )  # fmt: skip
                for phase in k.PHASES:
                    wlo, whi = sc.window(onset=e["onset_idx"], end=e["end_idx"], phase=phase)
                    for band in BAND_IDS:
                        sel = np.flatnonzero(
                            (t.prim[rows] == p)
                            & (t.idx[rows] >= wlo)
                            & (t.idx[rows] < whi)
                            & (t.band[rows] == band)
                        )
                        if len(sel) < 60:
                            continue
                        for name in system_names(inp=inp):
                            sel_n = sel[inp.has_e2[rows][sel]] if name == REFERENCE else sel
                            if len(sel_n) < 60:
                                continue
                            wide_sel = (
                                wide_inj[t.prim[rows][sel_n], t.origin_day[rows][sel_n]]
                                if name == "G4"
                                else np.zeros(len(sel_n), bool)
                            )
                            rows_out.append(
                                {
                                    "month": m, "pass": r, "primary": p, "group": str(t.group[p]),
                                    "sign": int(sgn), "depth": depth, "length_days": int(e["length_days"]),
                                    "phase": phase, "band": BANDS[band], "system": name,
                                    "n_rows": len(sel_n), "share_origin_days_wide": float(wide_sel.mean()),
                                    **means(rows_metrics=st[name][sel_n], sgn=sgn),
                                    **follow(q=q_inj[name][sel_n], twin=q_twin[name][sel_n], sgn=sgn, depth=depth),
                                }
                            )  # fmt: skip
            print("bank", m, r, flush=True)
    return pl.DataFrame(rows_out), trig_rows


def summarise_bank(*, tab: pl.DataFrame) -> pl.DataFrame:
    """Return per (group, class, phase, band, system) means over events with 95% event-bootstrap
    intervals. Classes: event length, and (for 15 and 30 day events) depth.
    """
    metrics = [*STAT_NAMES, "nra_side95", "nra_side99", "follow_q50", "follow_safety"]
    classes = {
        "all lengths": pl.lit(True),
        "7 days": pl.col("length_days") == 7,
        "15 days": pl.col("length_days") == 15,
        "30 days": pl.col("length_days") == 30,
        "15 and 30 days": pl.col("length_days") >= 15,
        "15 and 30 days, depth 0.15": (pl.col("length_days") >= 15) & (pl.col("depth") == 0.15),
        "15 and 30 days, depth 0.30": (pl.col("length_days") >= 15) & (pl.col("depth") == 0.30),
        "15 and 30 days, depth 0.50": (pl.col("length_days") >= 15) & (pl.col("depth") == 0.50),
    }
    out = []
    for gname, gexpr in (
        ("MVA-only eight", pl.col("group") == "MVA-only eight"),
        ("signed-MW six", pl.col("group") == "signed-MW six"),
        ("both groups", pl.lit(True)),
    ):
        for cname, cexpr in classes.items():
            sub = tab.filter(gexpr & cexpr)
            for (phase, band, system), grp in sub.group_by("phase", "band", "system"):
                rec = {"group": gname, "length_class": cname, "phase": phase, "band": band,
                       "system": system, "n_events": grp.height}  # fmt: skip
                for mname in metrics:
                    mean, lo, hi = sc._boot_mean(vals=grp[mname].to_numpy())
                    rec[mname], rec[f"{mname}_lo"], rec[f"{mname}_hi"] = mean, lo, hi
                out.append(rec)
    return pl.DataFrame(out)


def main(arm: str) -> None:
    """Score one arm and write the tables."""
    out = k600.WORK / arm
    out.mkdir(parents=True, exist_ok=True)
    inp = Inputs(arm=arm)
    chk = inp.check_gate_state()
    (out / "gate_check.json").write_text(json.dumps(chk))
    print("rows", inp.t.n, "gate state equals experiment 592's:", chk, flush=True)
    ft = k592.false_triggers(inp=inp).filter(pl.col("gate").is_in(["G0", "G4"]))
    ft.write_csv(out / "false_triggers.csv")
    ef, rows = event_free(inp=inp)
    ef.write_csv(out / "event_free.csv")
    np.savez(out / "event_free_rows.npz", **rows)  # ty: ignore[invalid-argument-type]
    real_events(inp=inp).write_csv(out / "real_event_phase.csv")
    tab, trig = score_bank(inp=inp)
    tab.write_csv(out / "bank_event_phase.csv")
    summarise_bank(tab=tab).write_csv(out / "bank_summary.csv")
    sc_ev = sc.attach_depth(ev=sc.merged_events(), t=inp.t)
    scored = {int(d) for d in np.unique(inp.t.idx // 48)}
    sc_ev = sc_ev.filter(
        pl.col("onset_idx").map_elements(lambda i: int(i) // 48 in scored, return_dtype=pl.Boolean)
    )
    real_trig = k592.trigger_rows(
        ev=sc_ev, states={"G4": inp.states["G4"]}, source="verified", extra={}
    )
    pl.DataFrame([*real_trig, *trig], infer_schema_length=None).write_csv(out / "trigger_times.csv")
    np.save(out / "mode_g4.npy", inp.mode)
    print("done", tab.height, "bank rows", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
