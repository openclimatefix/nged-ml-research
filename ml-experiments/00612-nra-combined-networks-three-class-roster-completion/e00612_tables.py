# ruff: noqa: D103, E501, PLR0915, C901
"""Experiment 612: the result tables, from the score folders written by `e00612_score.py` (CPU).

Every contrast is a paired difference of two systems on the same cells, with a Bonferroni-adjusted
bootstrap interval over cells (a cell is one injected event: month, pass, Primary; or one event-free
Primary-month-band cell). `N_COUNTED` is the count of contrasts whose `counted` flag is set.

Writes `tables/*.csv` under the experiment's private folder.

Usage: python e00612_tables.py
"""

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00612-nra-combined-networks-three-class-roster-completion"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
SCORE = ROOT / "score"
TABLES = ROOT / "tables_out"
N_BOOT = 4000
RNG = np.random.default_rng(612)
EIGHT, SIX = "MVA-only eight", "signed-MW six"
GROUPS = (EIGHT, SIX)
TOL = {"cover": 0.01, "follow": 0.01, "absorb": 0.03, "crps": 0.0006}
"""Tolerances. CRPS is a share of capacity; 0.0006 is the printed minimum detectable effect of experiment 591 (0.06 percentage points)."""
FIRST = "2025-10"
"""The first month has no tail constants, so contrasts that involve `+t` systems skip it. Contrasts of plain systems use all months, unless stated."""


def load(*, arm: str, name: str) -> pl.DataFrame | None:
    f = SCORE / arm / name
    return pl.read_csv(f, infer_schema_length=None) if f.exists() else None


def ci(*, diffs: np.ndarray, weights: np.ndarray | None = None, alpha: float) -> tuple[float, float, float]:
    """Return the (weighted) mean difference and its percentile bootstrap interval at level 1 - alpha."""
    d = np.asarray(diffs, float)
    w = np.ones(len(d)) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(d) & np.isfinite(w)
    d, w = d[ok], w[ok]
    if len(d) < 2:
        return (float("nan"),) * 3
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    means = (d[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    return (
        float((d * w).sum() / w.sum()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def verdict(*, lo: float, hi: float, tol: float, higher_is_better: bool) -> str:
    """Return 'better', 'worse', 'equivalent', or 'inconclusive' for an interval against a tolerance."""
    if (higher_is_better and lo > tol) or (not higher_is_better and hi < -tol):
        return "better"
    worse_lo, worse_hi = (-np.inf, -tol) if higher_is_better else (tol, np.inf)
    if worse_lo <= lo and hi <= worse_hi:
        return "worse"
    if -tol <= lo and hi <= tol:
        return "equivalent"
    return "inconclusive"


# ---------------------------------------------------------------- bank contrasts


def bank_cells(*, arm: str, system: str, band: str, phase: str, months: list[str] | None = None) -> pl.DataFrame | None:
    t = load(arm=arm, name="bank_event_phase.csv")
    if t is None:
        return None
    t = t.filter(
        (pl.col("system") == system) & (pl.col("band") == band) & (pl.col("phase") == phase) & (pl.col("length_days") >= 15)
    )
    if months:
        t = t.filter(pl.col("month").is_in(months))
    return t.select("month", "pass", "primary", "length_days", "depth", "sign", "cover90", "follow_q50", "width90", "crps")


def bank_contrast(*, a: tuple[str, str], b: tuple[str, str], band: str, phase: str, metric: str, alpha: float, months: list[str] | None = None) -> dict | None:
    """Return mean of (arm a, system a) minus (arm b, system b) over matched injected events."""
    ca = bank_cells(arm=a[0], system=a[1], band=band, phase=phase, months=months)
    cb = bank_cells(arm=b[0], system=b[1], band=band, phase=phase, months=months)
    if ca is None or cb is None:
        return None
    j = ca.join(cb, on=["month", "pass", "primary", "length_days", "depth", "sign"], suffix="_b")
    if j.height < 5:
        return None
    mean, lo, hi = ci(diffs=(j[metric] - j[f"{metric}_b"]).to_numpy(), alpha=alpha)
    return {"n_events": j.height, "diff": mean, "lo": lo, "hi": hi, "mde": (hi - lo) / 2,
            "level_a": float(j[metric].mean()), "level_b": float(j[f"{metric}_b"].mean())}  # fmt: skip


# ---------------------------------------------------------------- event-free contrasts


def cell_frame(*, arm: str, system: str, band: str, group: str, months: list[str] | None = None) -> pl.DataFrame | None:
    t = load(arm=arm, name="event_free_cells.csv")
    if t is None:
        return None
    t = t.filter((pl.col("system") == system) & (pl.col("band") == band) & (pl.col("group") == group))
    if months:
        t = t.filter(pl.col("month").is_in(months))
    return t


def free_contrast(*, a: tuple[str, str], b: tuple[str, str], band: str, group: str, metric: str, alpha: float, months: list[str] | None = None) -> dict | None:
    ca = cell_frame(arm=a[0], system=a[1], band=band, group=group, months=months)
    cb = cell_frame(arm=b[0], system=b[1], band=band, group=group, months=months)
    if ca is None or cb is None:
        return None
    j = ca.join(cb, on=["prim", "month"], suffix="_b")
    if j.height < 5:
        return None
    mean, lo, hi = ci(diffs=(j[metric] - j[f"{metric}_b"]).to_numpy(), weights=j["n_rows"].to_numpy(), alpha=alpha)
    w = j["n_rows"].to_numpy().astype(float)
    return {"n_cells": j.height, "diff": mean, "lo": lo, "hi": hi, "mde": (hi - lo) / 2,
            "level_a": float((j[metric].to_numpy() * w).sum() / w.sum()),
            "level_b": float((j[f"{metric}_b"].to_numpy() * w).sum() / w.sum())}  # fmt: skip


def free_level(*, arm: str, system: str, band: str, group: str, metric: str, months: list[str] | None = None) -> float | None:
    c = cell_frame(arm=arm, system=system, band=band, group=group, months=months)
    if c is None or c.height == 0:
        return None
    w = c["n_rows"].to_numpy().astype(float)
    return float((c[metric].to_numpy() * w).sum() / w.sum())


def cost_level(*, arm: str, system: str, months: list[str] | None = None) -> dict | None:
    c = load(arm=arm, name="cost_cells.csv")
    if c is None:
        return None
    c = c.filter(pl.col("system") == system)
    if months:
        c = c.filter(pl.col("month").is_in(months))
    if c.height == 0:
        return None
    return {f: float(c[f"cost_{f}"].mean()) for f in (0.9, 0.75, 0.5)}


# ---------------------------------------------------------------- tables


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    tail_months = ["2025-12", "2026-01", "2026-03", "2026-06"]
    nets = {"K1 (D5)": ("k1_d5", "K1"), "K2 (D5)": ("k1_d5", "K2"), "K3 (D5)": ("k3_d5", "K2"),
            "K1 (O2)": ("k1_o2", "K1"), "K2 (O2)": ("k1_o2", "K2"), "K3 (O2)": ("k3_o2", "K2"),
            "Class 2 (own band width)": ("c2_e3w", "K2"), "Class 2 (gate band width)": ("c2_e3w", "K2w")}  # fmt: skip

    # --- T1: level table, with and without tail constants, on the four tail months
    rows = []
    for name, (arm, sysname) in nets.items():
        for tails in (False, True):
            system = sysname + ("+t" if tails else "")
            for band in ("24-120 h", "120-240 h"):
                for group in GROUPS:
                    rec = {"network": name, "tail_constants": tails, "band": band, "group": group}
                    for metric in ("cover90", "below_q05", "above_q95", "crps", "width90"):
                        rec[f"free_{metric}"] = free_level(arm=arm, system=system, band=band, group=group, metric=metric, months=tail_months)
                    rows.append(rec)
                bank = load(arm=arm, name="bank_summary.csv")
                if bank is not None:
                    for phase in ("days4-end", "after0-7"):
                        b = bank.filter((pl.col("month_set") == "tail months") & (pl.col("group") == "both groups") & (pl.col("length_class") == "15 and 30 days")
                                        & (pl.col("band") == band) & (pl.col("phase") == phase) & (pl.col("system") == system))  # fmt: skip
                        if b.height:
                            rows.append({"network": name, "tail_constants": tails, "band": band, "group": "bank both groups",
                                         "phase": phase, "bank_cover90": float(b["cover90"][0]), "bank_lo": float(b["cover90_lo"][0]),
                                         "bank_hi": float(b["cover90_hi"][0]), "follow_q50": float(b["follow_q50"][0])})  # fmt: skip
    pl.DataFrame(rows, infer_schema_length=None).write_csv(TABLES / "t1_levels.csv")

    # --- T2: O2 minus D5, tail constants on both, same four months; counted tests
    contr = []

    def add(label: str, rec: dict | None, *, tol: float, hib: bool, family: str) -> None:
        if rec is None:
            return
        contr.append({"contrast": label, "family": family, **rec, "tolerance": tol,
                      "verdict": verdict(lo=rec["lo"], hi=rec["hi"], tol=tol, higher_is_better=hib)})  # fmt: skip

    families = []
    for net in ("K1", "K2", "K3"):
        arm_d, arm_o = {"K1": "k1_d5", "K2": "k1_d5", "K3": "k3_d5"}[net], {"K1": "k1_o2", "K2": "k1_o2", "K3": "k3_o2"}[net]
        system = "K1+t" if net == "K1" else "K2+t"
        families.append((net, (arm_o, system), (arm_d, system)))
    counted = []
    for net, a, b in families:
        for band in ("24-120 h", "120-240 h"):
            for group in GROUPS:
                counted.append(("free", net, a, b, band, group, "cover90", TOL["cover"], True))
            for group in GROUPS:
                counted.append(("free", net, a, b, band, group, "crps", TOL["crps"], False))
        if net != "K3":
            for phase in ("days4-end", "after0-7"):
                counted.append(("bank", net, a, b, "24-120 h", phase, "cover90", TOL["cover"], True))
            counted.append(("bank", net, a, b, "24-120 h", "days4-end", "follow_q50", TOL["follow"], False))
    n_counted = len(counted)
    alpha = 0.05 / n_counted
    print("counted tests for O2 against D5:", n_counted)
    for kind, net, a, b, band, x, metric, tol, hib in counted:
        if kind == "free":
            rec = free_contrast(a=a, b=b, band=band, group=x, metric=metric, alpha=alpha, months=tail_months)
            label = f"{net}: O2 minus D5, event-free {metric}, {band}, {'eight MVA' if x == EIGHT else 'six MW'}"
            add(label, rec, tol=tol, hib=hib, family="O2 vs D5")
        else:
            rec = bank_contrast(a=a, b=b, band=band, phase=x, metric=metric, alpha=alpha, months=tail_months)
            add(f"{net}: O2 minus D5, injected {metric}, {band}, {x}", rec, tol=tol, hib=hib, family="O2 vs D5")
    c2 = pl.DataFrame(contr, infer_schema_length=None)
    c2 = c2.with_columns(pl.lit(n_counted).alias("n_counted"))
    c2.write_csv(TABLES / "t2_o2_vs_d5.csv")
    with pl.Config(tbl_rows=80, tbl_width_chars=220, fmt_str_lengths=90):
        print(c2.select("contrast", "diff", "lo", "hi", "mde", "verdict"))


def ablations() -> None:
    """Pair-month ablations (no E3, no offset) against K1 on the same two months, with tail constants."""
    pair = ["2025-12"]  # the only pair month that has tail constants
    rows = []
    ref = ("k1_d5", "K1+t")
    arms = {"no E3": "noe3_d5", "no offset": "nooff_d5"}
    n_counted = 2 * 6
    alpha = 0.05 / n_counted
    for label, arm in arms.items():
        for band in ("24-120 h",):
            for phase in ("days4-end", "after0-7"):
                for months, tag in ((["2025-10", "2025-12"], "plain K1 (both pair months)"),):
                    rec = bank_contrast(a=(arm, "K1"), b=("k1_d5", "K1"), band=band, phase=phase, metric="cover90", alpha=alpha, months=months)
                    if rec:
                        rows.append({"removed": label, "measure": f"injected cover90 {phase} {band}", "set": tag, **rec,
                                     "verdict": verdict(lo=rec["lo"], hi=rec["hi"], tol=TOL["cover"], higher_is_better=True)})  # fmt: skip
            rec = bank_contrast(a=(arm, "K1"), b=("k1_d5", "K1"), band="24-120 h", phase="days4-end", metric="follow_q50", alpha=alpha, months=["2025-10", "2025-12"])
            if rec:
                rows.append({"removed": label, "measure": "injected follow_q50 days4-end", "set": "pair", **rec,
                             "verdict": verdict(lo=rec["lo"], hi=rec["hi"], tol=TOL["follow"], higher_is_better=False)})  # fmt: skip
            for group in GROUPS:
                for metric, tol, hib in (("cover90", TOL["cover"], True), ("crps", TOL["crps"], False)):
                    rec = free_contrast(a=(arm, "K2"), b=("k1_d5", "K2"), band=band, group=group, metric=metric, alpha=alpha, months=["2025-10", "2025-12"])
                    if rec:
                        rows.append({"removed": label, "measure": f"event-free {metric} {band} {'eight' if group == EIGHT else 'six'}", "set": "pair", **rec,
                                     "verdict": verdict(lo=rec["lo"], hi=rec["hi"], tol=tol, higher_is_better=hib)})  # fmt: skip
    out = pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.lit(n_counted).alias("n_counted"))
    out.write_csv(TABLES / "t3_ablations.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=200, fmt_str_lengths=60):
        print(out.select("removed", "measure", "n_events", "n_cells", "diff", "lo", "hi", "verdict"))
    _ = (pair, ref)


def class2() -> None:
    """Class 2 candidate (K2 + neighbour summaries + dropout, at the gate's band width) against K1 and K2, all five months, no tail constants."""
    rows = []
    cand = ("c2_e3w", "K2w")
    refs = {"K1": ("k1_d5", "K1"), "K2": ("k1_d5", "K2")}
    specs = []
    for ref in refs:
        for phase in ("days4-end", "after0-7"):
            specs.append(("bank", ref, "24-120 h", phase, "cover90", TOL["cover"], True))
        specs.append(("bank", ref, "24-120 h", "days4-end", "follow_q50", TOL["follow"], False))
        for band in ("24-120 h", "120-240 h"):
            for group in GROUPS:
                specs.append(("free", ref, band, group, "cover90", TOL["cover"], True))
                specs.append(("free", ref, band, group, "crps", TOL["crps"], False))
    n_counted = len(specs)
    alpha = 0.05 / n_counted
    for kind, ref, band, x, metric, tol, hib in specs:
        if kind == "bank":
            rec = bank_contrast(a=cand, b=refs[ref], band=band, phase=x, metric=metric, alpha=alpha)
            label = f"injected {metric}, {band}, {x}"
        else:
            rec = free_contrast(a=cand, b=refs[ref], band=band, group=x, metric=metric, alpha=alpha)
            label = f"event-free {metric}, {band}, {'eight MVA' if x == EIGHT else 'six MW'}"
        if rec:
            rows.append({"candidate_minus": ref, "measure": label, **rec, "tolerance": tol, "n_counted": n_counted,
                         "verdict": verdict(lo=rec["lo"], hi=rec["hi"], tol=tol, higher_is_better=hib)})  # fmt: skip
    out = pl.DataFrame(rows, infer_schema_length=None)
    out.write_csv(TABLES / "t4_class2.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=200, fmt_str_lengths=60):
        print(out.select("candidate_minus", "measure", "n_events", "n_cells", "level_a", "level_b", "diff", "lo", "hi", "verdict"))


def measures() -> None:
    """Levels at 120-240 h, curtailment cost, and event-free CRPS per group, for every network, plain and with tail constants (tail months for `+t`)."""
    tail_months = ["2025-12", "2026-01", "2026-03", "2026-06"]
    nets = {"K1 (D5)": ("k1_d5", "K1"), "K2 (D5)": ("k1_d5", "K2"), "K3 (D5)": ("k3_d5", "K2"),
            "K1 (O2)": ("k1_o2", "K1"), "K2 (O2)": ("k1_o2", "K2"), "K3 (O2)": ("k3_o2", "K2"),
            "Class 2 (gate band width)": ("c2_e3w", "K2w"), "Class 2 (own band width)": ("c2_e3w", "K2")}  # fmt: skip
    rows = []
    for name, (arm, system) in nets.items():
        for tails in (False, True):
            sysname = system + ("+t" if tails else "")
            months = tail_months if tails else None
            if tails and system == "K2w":
                continue
            c = cost_level(arm=arm, system=sysname, months=months)
            for band in ("24-120 h", "120-240 h"):
                for group in GROUPS:
                    rows.append({"network": name, "tail_constants": tails, "band": band, "group": group,
                                 "crps": free_level(arm=arm, system=sysname, band=band, group=group, metric="crps", months=months),
                                 "cover90": free_level(arm=arm, system=sysname, band=band, group=group, metric="cover90", months=months),
                                 **({f"cost_{k}": v for k, v in c.items()} if c and group == SIX and band == "24-120 h" else {})})  # fmt: skip
    out = pl.DataFrame(rows, infer_schema_length=None)
    out.write_csv(TABLES / "t5_measures.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=220, fmt_str_lengths=40):
        print(out)


if __name__ == "__main__":
    main()
    ablations()
    class2()
    measures()
