# ruff: noqa: E501, C416, RUF046, D103, D205
"""Experiment 582: build the arms, score them by phase of each verified event and on rows with no
labelled event.

Writes, to this experiment's work folder: `arms_summary.csv` (calibration, widths),
`event_phase.csv` (one row per merged Primary-event, phase, lead band, arm), `phase_summary.csv`
(group means with event-block bootstrap intervals), `event_free.csv` (pinball and CRPS against R
with month-block intervals), `gates.csv` (gate state and false triggers), and `day_ahead.parquet`
(the plotted series).

Event unit: one row per Primary and span, merging verified transfers that overlap at a Primary
(the three May 2026 transfers overlap at D12). The NRA reference is the observed power minus the
fitted depth inside the span (`Table.nra`), and the observed power elsewhere. Phases are by the
valid time: before = the 7 days before onset; days 0-3; days 4 to the end; after = 0-7 days after
the end; after2 = 7-14 days after the end.
"""

import sys
from typing import Final

import e00582_arms as a  # ty: ignore[unresolved-import]
import e00582_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

IQ = a.IQ
PHASES: Final[tuple[str, ...]] = ("before", "days0-3", "days4-end", "after0-7", "after7-14")
BANDS: Final[dict[int, str]] = {0: "0-24 h", 1: "24-120 h"}
RNG = np.random.default_rng(582)
N_BOOT = 4000


def merged_events() -> pl.DataFrame:
    """Return one row per (Primary, merged span): onset, end, mean signed depth share over the
    span, and transfer ids.
    """
    ev = k.verified_events().sort("primary", "onset_idx")
    out: list[dict] = []
    for p, g in ev.group_by("primary", maintain_order=True):
        cur = None
        for r in g.iter_rows(named=True):
            if cur is not None and r["onset_idx"] <= cur["end_idx"]:
                cur["end_idx"] = max(cur["end_idx"], r["end_idx"])
                cur["transfers"].append(r["transfer_id"])
            else:
                if cur is not None:
                    out.append(cur)
                cur = {
                    "primary": p[0],
                    "label": r["label"],
                    "onset_idx": r["onset_idx"],
                    "end_idx": r["end_idx"],
                    "transfers": [r["transfer_id"]],
                }
        if cur is not None:
            out.append(cur)
    return pl.DataFrame(out)


def attach_depth(*, ev: pl.DataFrame, t: a.Table) -> pl.DataFrame:
    """Add the span's mean signed depth (capacity share; positive = the Primary carried more
    during the span than the NRA).
    """
    depth = [
        float(-t.shift[r["primary"], r["onset_idx"] : r["end_idx"]].mean())
        for r in ev.iter_rows(named=True)
    ]
    cap = t.capacity
    return ev.with_columns(
        depth_share=pl.Series(depth),
        depth_mw=pl.Series(
            [d * float(cap[p]) for d, p in zip(depth, ev["primary"].to_list(), strict=True)]
        ),
        length_days=(pl.col("end_idx") - pl.col("onset_idx")) / 48.0,
        group=pl.Series([str(t.group[p]) for p in ev["primary"].to_list()]),
    ).with_columns(
        name=pl.col("label")
        + pl.lit(" ")
        + pl.Series([k.date_of(i) for i in ev["onset_idx"].to_list()])
    )


def window(*, onset: int, end: int, phase: str) -> tuple[int, int]:
    """Return the half-hour grid window [lo, hi) of a phase."""
    d = 48
    return {
        "before": (onset - 7 * d, onset),
        "days0-3": (onset, min(onset + 3 * d, end)),
        "days4-end": (onset + 3 * d, end),
        "after0-7": (end, end + 7 * d),
        "after7-14": (end + 7 * d, end + 14 * d),
    }[phase]


STORES: dict[str, dict] = {}
"""Recalibration factors and mixture weights per arm, keyed by (month, group, band) or (month, band);
filled by `build_arms`.
"""


def build_arms(
    *, t: a.Table, gates: dict
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Return every arm's quantile array, the mask of rows on which every arm exists (`common`),
    and the has-fit masks.
    """
    V, M, E = t.arms["V"], t.arms["M"], t.arms["E"]
    have_e = np.isfinite(E).all(axis=1)
    arms: dict[str, np.ndarray] = {
        "v3+D5": V,
        "Masked": M,
        "Week-masked": t.arms["W"],
        "Lag-free D5": np.where(have_e[:, None], E, np.nan),
        "Corrected lags": t.arms["C"],
    }
    allrows = np.ones(t.n, bool)
    STORES["R"] = {}
    arms["R"], has_r = a.recalibrate(t=t, arm=V, usable=allrows, store=STORES["R"])
    w50 = np.full(t.n, 0.5, np.float32)
    w70 = np.full(t.n, 0.7, np.float32)
    STORES["MixPL_w"] = {}
    wpl = a.per_lead_weights(
        t=t, a=V, b=np.where(have_e[:, None], E, V), usable=have_e, store=STORES["MixPL_w"]
    )
    has: dict[str, np.ndarray] = {"R": has_r}
    for name, w in (("Mix50", w50), ("Mix70", w70), ("MixPL", wpl)):
        mix = np.full_like(V, np.nan)
        mix[have_e] = a.mixture(a=V[have_e], b=E[have_e], w=w[have_e])
        arms[name] = mix
        rc = np.full_like(V, np.nan)
        STORES[f"{name}-R"] = {}
        r_, h_ = a.recalibrate(
            t=t, arm=np.where(have_e[:, None], mix, V), usable=have_e, store=STORES[f"{name}-R"]
        )
        rc[have_e] = r_[have_e]
        arms[f"{name}-R"] = rc
        has[f"{name}-R"] = h_ & have_e
    arms["Union"] = a.union_band(lag=V, masked=M)
    for gname in ("G1", "G2"):
        on = gates["state"][gname]
        arms[gname] = a.gated_arm(t=t, on=on, with_rule=False)
        arms[f"{gname}+SB"] = a.gated_arm(t=t, on=on, with_rule=True)
    common = has_r & have_e
    for h in has.values():
        common &= h
    return arms, common, has


def _boot_mean(*, vals: np.ndarray, groups: np.ndarray | None = None) -> tuple[float, float, float]:
    """Return the mean and a 95% bootstrap interval over the entries of `vals` (events or months
    as blocks).
    """
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(v), size=(N_BOOT, len(v)))
    m = v[idx].mean(axis=1)
    return (float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def event_phase_table(
    *, t: a.Table, arms: dict[str, np.ndarray], common: np.ndarray, ev: pl.DataFrame
) -> pl.DataFrame:
    """Return the per-event, per-phase, per-band metrics of every arm (section 4 of the brief)."""
    rows: list[dict] = []
    C, E, R = arms["Corrected lags"], arms["Lag-free D5"], arms["R"]
    for e in ev.iter_rows(named=True):
        sgn = (
            1.0 if e["depth_share"] > 0 else -1.0
        )  # +1: the event raised the Primary's load, so "towards the event" is upward
        depth = abs(e["depth_share"])
        for phase in PHASES:
            lo, hi = window(onset=e["onset_idx"], end=e["end_idx"], phase=phase)
            for band in (0, 1):
                sel = (
                    common
                    & (t.prim == e["primary"])
                    & (t.idx >= lo)
                    & (t.idx < hi)
                    & (t.band == band)
                )
                n = int(sel.sum())
                if n < 100:
                    continue
                y = t.nra[sel]
                for name, q in arms.items():
                    qs = q[sel]
                    r = {
                        "event": e["name"],
                        "group": e["group"],
                        "label": e["label"],
                        "phase": phase,
                        "band": BANDS[band],
                        "arm": name,
                        "n_rows": n,
                        "cover90": float(np.mean((qs[:, IQ[0.05]] <= y) & (y <= qs[:, IQ[0.95]]))),
                        "cover98": float(np.mean((qs[:, IQ[0.01]] <= y) & (y <= qs[:, IQ[0.99]]))),
                        "median_in_band_corrected": float(
                            np.mean(
                                (qs[:, IQ[0.05]] <= C[sel, IQ[0.5]])
                                & (C[sel, IQ[0.5]] <= qs[:, IQ[0.95]])
                            )
                        ),
                        "width90": float(np.mean(qs[:, IQ[0.95]] - qs[:, IQ[0.05]])),
                        "width98": float(np.mean(qs[:, IQ[0.99]] - qs[:, IQ[0.01]])),
                        "width90_over_R": float(
                            np.mean(qs[:, IQ[0.95]] - qs[:, IQ[0.05]])
                            / np.mean(R[sel, IQ[0.95]] - R[sel, IQ[0.05]])
                        ),
                        "width98_over_R": float(
                            np.mean(qs[:, IQ[0.99]] - qs[:, IQ[0.01]])
                            / np.mean(R[sel, IQ[0.99]] - R[sel, IQ[0.01]])
                        ),
                    }
                    if (
                        sgn < 0
                    ):  # load lowered: the NRA lies above the following forecast, so the upper tail is the NRA side
                        r["nra_side_cover95"] = float(np.mean(y <= qs[:, IQ[0.95]]))
                        r["nra_side_cover99"] = float(np.mean(y <= qs[:, IQ[0.99]]))
                        r["nra_side_tau"] = "q95/q99"
                    else:
                        r["nra_side_cover95"] = float(np.mean(y >= qs[:, IQ[0.05]]))
                        r["nra_side_cover99"] = float(np.mean(y >= qs[:, IQ[0.01]]))
                        r["nra_side_tau"] = "q05/q01"
                    for tau in (0.01, 0.05, 0.95, 0.99):
                        r[f"pinball_q{int(round(tau * 100)):02d}"] = float(
                            a.pinball(y=y, q=qs[:, IQ[tau]], tau=tau).mean()
                        )
                    for tau in (0.05, 0.5, 0.95, 0.99):
                        j = IQ[tau]
                        r[f"follow_q{int(round(tau * 100)):02d}"] = float(
                            np.mean((qs[:, j] - C[sel, j]) * sgn) / depth
                        )
                        r[f"follow_vs_lagfree_q{int(round(tau * 100)):02d}"] = float(
                            np.mean((qs[:, j] - E[sel, j]) * sgn) / depth
                        )
                    # the safety follow ratio: the quantile on the NRA side (q95 when the load was lowered, q05 when it was raised)
                    side = IQ[0.95] if sgn < 0 else IQ[0.05]
                    r["follow_safety"] = float(np.mean((qs[:, side] - C[sel, side]) * sgn) / depth)
                    rows.append(r)
    return pl.DataFrame(rows)


def phase_summary(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return per (group, phase, band, arm) means over events with a 95% event-block bootstrap
    interval, and the leave-one-Primary-out range.
    """
    metrics = [
        c
        for c in table.columns
        if c not in {"event", "group", "label", "phase", "band", "arm", "n_rows", "nra_side_tau"}
    ]
    out: list[dict] = []
    for groups, key in (
        (("MVA-only eight",), "MVA-only eight"),
        (("signed-MW six",), "signed-MW six"),
        (("MVA-only eight", "signed-MW six"), "both groups"),
    ):
        sub = table.filter(pl.col("group").is_in(groups))
        for (phase, band, arm), g in sub.group_by("phase", "band", "arm"):
            r = {"group": key, "phase": phase, "band": band, "arm": arm, "n_events": g.height}
            for m in metrics:
                v = g[m].to_numpy()
                mean, lo, hi = _boot_mean(vals=v)
                r[m], r[f"{m}_lo"], r[f"{m}_hi"] = mean, lo, hi
                # leave-one-Primary-out range
                loo = []
                for lab in g["label"].unique().to_list():
                    vv = g.filter(pl.col("label") != lab)[m].to_numpy()
                    if len(vv):
                        loo.append(float(np.nanmean(vv)))
                r[f"{m}_loo_min"], r[f"{m}_loo_max"] = (
                    (min(loo), max(loo)) if loo else (np.nan, np.nan)
                )
            out.append(r)
    return pl.DataFrame(out)


def free_table(*, t: a.Table, arms: dict[str, np.ndarray], common: np.ndarray) -> pl.DataFrame:
    """Return the false-alarm table on rows with no labelled event: absolute pinball and CRPS,
    the difference against R, calibration, and width.
    """
    rows: list[dict] = []
    R = arms["R"]
    months = sorted({m for m in t.month[common]})
    mn = t.month
    for g in ("MVA-only eight", "signed-MW six", "D15 and D18"):
        for band in (0, 1):
            sel0 = common & t.free & (t.group_of_row == g) & (t.band == band)
            if sel0.sum() < 500:
                continue
            for name, q in arms.items():
                losses = {
                    f"pinball_q{int(round(tau * 100)):02d}": a.pinball(
                        y=t.obs, q=q[:, IQ[tau]], tau=tau
                    )
                    for tau in (0.01, 0.05, 0.95, 0.99)
                }
                losses["crps"] = a.crps_from_quantiles(y=t.obs, q=np.where(np.isfinite(q), q, 0.0))
                ref = {
                    f"pinball_q{int(round(tau * 100)):02d}": a.pinball(
                        y=t.obs, q=R[:, IQ[tau]], tau=tau
                    )
                    for tau in (0.01, 0.05, 0.95, 0.99)
                }
                ref["crps"] = a.crps_from_quantiles(y=t.obs, q=R)
                r = {
                    "group": g,
                    "band": BANDS[band],
                    "arm": name,
                    "n_rows": int(sel0.sum()),
                    "n_months": len(months),
                }
                for key, loss in losses.items():
                    per_month = (
                        np.array(
                            [
                                loss[sel0 & (mn == m)].mean()
                                for m in months
                                if (sel0 & (mn == m)).any()
                            ]
                        )
                        * 100
                    )
                    per_month_ref = (
                        np.array(
                            [
                                ref[key][sel0 & (mn == m)].mean()
                                for m in months
                                if (sel0 & (mn == m)).any()
                            ]
                        )
                        * 100
                    )
                    diff = per_month - per_month_ref
                    mean, lo, hi = _boot_mean(vals=diff)
                    r[key], r[f"{key}_minus_R"], r[f"{key}_minus_R_lo"], r[f"{key}_minus_R_hi"] = (
                        float(loss[sel0].mean() * 100),
                        mean,
                        lo,
                        hi,
                    )
                    r[f"{key}_minus_R_signs"] = "".join(
                        "+" if d > 0 else "-" if d < 0 else "0" for d in diff
                    )
                for tau in TAUS_REPORT:
                    r[f"below_q{int(round(tau * 100)):02d}"] = float(
                        np.mean(t.obs[sel0] <= q[sel0, IQ[tau]])
                    )
                r["width90_points"] = float(np.mean(q[sel0, IQ[0.95]] - q[sel0, IQ[0.05]]) * 100)
                r["width98_points"] = float(np.mean(q[sel0, IQ[0.99]] - q[sel0, IQ[0.01]]) * 100)
                rows.append(r)
    return pl.DataFrame(rows)


TAUS_REPORT = (0.01, 0.05, 0.50, 0.95, 0.99)


def gate_table(*, t: a.Table, gates: dict) -> pl.DataFrame:
    """Return false triggers per Primary-year on origin days with no labelled event, per gate
    variant, and event-day trigger days.
    """
    rows = []
    near = gates["near"]
    gap = gates["gap"]
    for gname, st in gates["state"].items():
        free = [
            (p, d)
            for p in range(16)
            for d in gates["origin_days"]
            if not near[p, d] and np.isfinite(gap[p, d - a.GATE_M : d]).all()
        ]
        trig = sum(1 for p, d in free if st[p, d])
        hold = a.hold_after_release(on=st, valid_days=a._origin_day_mask(t=t, n=st.shape[1]))
        held = sum(1 for p, d in free if hold[p, d])
        yrs = len(free) / 365.25
        rows.append(
            {
                "variant": gname,
                "event_free_origin_days": len(free),
                "gated_origin_days": trig,
                "held_origin_days": held,
                "false_trigger_days_per_primary_year": trig / yrs,
                "held_days_per_primary_year": held / yrs,
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    t = a.Table()
    gates = a.run_gates(t=t)
    arms, common, _ = build_arms(t=t, gates=gates)
    ev = attach_depth(ev=merged_events(), t=t)
    print(ev)
    print(
        "rows",
        t.n,
        "common",
        int(common.sum()),
        "months in common",
        sorted({m for m in t.month[common]}),
    )
    table = event_phase_table(t=t, arms=arms, common=common, ev=ev)
    table.write_csv(k.WORK / "event_phase.csv")
    summ = phase_summary(table=table)
    summ.write_csv(k.WORK / "phase_summary.csv")
    free = free_table(t=t, arms=arms, common=common)
    free.write_csv(k.WORK / "event_free.csv")
    g = gate_table(t=t, gates=gates)
    g.write_csv(k.WORK / "gates.csv")
    ev.with_columns(
        transfers=pl.col("transfers").list.eval(pl.element().cast(pl.String)).list.join("+")
    ).write_csv(k.WORK / "events.csv")
    print(g)


if __name__ == "__main__":
    sys.exit(main())
