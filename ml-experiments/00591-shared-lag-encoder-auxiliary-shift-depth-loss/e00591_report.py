# ruff: noqa: B023, D103, E501, PT018, D205
"""Experiment 591: print every score table the write-up quotes (also written to `report.txt`).

Adapted from experiment 584's `e00584_report.py`. Reads the tables that `e00591_score.py` wrote for
each
arm (`score_<arm>/`), and `row_arrays.npz` for the paired CRPS contrasts. Arm names: A0 = experiment
584's
C1; the gated band (`A2` in experiment 583's tables) is shown as "+band".

Usage: python e00591_report.py [ARM ...]   (default: a0 a1 a2 a3 a4 a5, those that have been scored)
"""

import io
import sys
from contextlib import redirect_stdout

import e00582_arms as a  # ty: ignore[unresolved-import]
import e00583_score as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT = ML_EXPERIMENTS_DATA_DIR / "00591-shared-lag-encoder-auxiliary-shift-depth-loss"
ARMS = [
    x
    for x in (sys.argv[1:] or ["a0", "a1", "a2", "a3", "a4", "a5"])
    if (ROOT / f"score_{x}").exists()
]
PAIRS = [("a1", "a0"), ("a2", "a1"), ("a3", "a1"), ("a3", "a2"), ("a4", "a2"), ("a5", "a2")]
PH = ["before", "days0-3", "days4-end", "after0-7", "after7-14"]
GROUPS = ["both groups", "MVA-only eight", "signed-MW six"]
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)


def label(arm: str, block: str) -> str:
    return arm.upper() + ("+band" if block == "A2" else "")


SHOW = [(label(x, b), x, b) for x in ARMS for b in ("V", "A2")]


def quantile_crps(*, q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the per-row CRPS estimate 2 x mean pinball loss over the 11 stored quantile levels."""
    taus = np.asarray(a.TAUS)
    d = y[:, None] - q
    return 2.0 * np.maximum(taus * d, (taus - 1) * d).mean(axis=1)


def load(arm: str) -> dict:
    raw = np.load(ROOT / f"score_{arm}" / "row_arrays.npz", allow_pickle=True)
    key = (
        raw["prim"].astype(np.int64) * 10**12
        + raw["origin_idx"].astype(np.int64) * 10**6
        + raw["idx"]
    )
    order = np.argsort(key)
    return {
        "key": key[order],
        **{k: raw[k][order] for k in ("V", "nra", "free", "lead", "month", "group")},
    }


def contrasts() -> None:
    """New minus old on event-free rows at 24-120 h (points of capacity), month-block Bonferroni
    intervals (10 counted tests).
    """
    z = {x: load(x) for x in ARMS}
    for new, old in PAIRS:
        if new not in z or old not in z:
            continue
        common = np.intersect1d(z[new]["key"], z[old]["key"])
        m0, m1 = np.isin(z[old]["key"], common), np.isin(z[new]["key"], common)
        o, n = {k: v[m0] for k, v in z[old].items()}, {k: v[m1] for k, v in z[new].items()}
        assert np.array_equal(o["key"], n["key"]) and np.array_equal(o["nra"], n["nra"])
        sel0 = o["free"] & (o["lead"] >= 24)
        months = sorted(set(o["month"]))
        y = o["nra"]
        metrics = {
            "CRPS": lambda q: quantile_crps(q=q, y=y),
            "pinball q05+q95": lambda q: 0.5 * (a.pinball(y=y, q=q[:, a.IQ[0.05]], tau=0.05) + a.pinball(y=y, q=q[:, a.IQ[0.95]], tau=0.95)),
            "pinball q01+q99": lambda q: 0.5 * (a.pinball(y=y, q=q[:, a.IQ[0.01]], tau=0.01) + a.pinball(y=y, q=q[:, a.IQ[0.99]], tau=0.99)),
        }  # fmt: skip
        rows = []
        for gname in ("MVA-only eight", "signed-MW six"):
            sel = sel0 & (o["group"] == gname)
            for mname, fn in metrics.items():
                v0, v1 = fn(o["V"]) * 100, fn(n["V"]) * 100
                for fold_set, ms in (("all 5 folds", months), ("first pair", months[:2])):
                    d = np.array(
                        [
                            v1[sel & (o["month"] == m)].mean() - v0[sel & (o["month"] == m)].mean()
                            for m in ms
                        ]
                    )
                    lo, hi = s.bonferroni_interval(diffs=d)
                    rows.append({"group": gname, "metric": mname, "folds": fold_set, old.upper(): v0[sel].mean(), new.upper(): v1[sel].mean(),
                                 "diff": d.mean(), "lo": lo, "hi": hi, "mde": (hi - lo) / 2})  # fmt: skip
        print(
            f"\n== {new.upper()} minus {old.upper()}, event-free rows, 24-120 h, Bonferroni (10 counted tests; first pair = 2025-10 and 2025-12 only)"
        )
        print(pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(3)))


def main() -> None:
    b = pl.concat(
        [
            pl.read_csv(ROOT / f"score_{run}/bank_summary.csv")
            .filter(pl.col("arm") == blk)
            .with_columns(name=pl.lit(nm))
            for nm, run, blk in SHOW
        ]
    )
    long_ = b.filter((pl.col("length_class") == "15 and 30 days") & (pl.col("band") == "24-120 h"))
    for g in GROUPS:
        x = long_.filter(pl.col("group") == g)
        print(
            f"\n== Injected bank, {g}, 15 and 30 day events, 24-120 h: q05-q95 coverage [95% interval] / one-sided NRA share / width90 / tail pinball (x100)"
        )
        for ph in PH:
            y = x.filter(pl.col("phase") == ph).sort("name")
            print(ph, "n_events", y["n_events"][0] if y.height else 0)
            print(y.select("name", pl.col("cover90").round(3), pl.col("cover90_lo").round(3), pl.col("cover90_hi").round(3),
                           pl.col("cover98").round(3), pl.col("nra_side95").round(3), pl.col("nra_side99").round(3),
                           (pl.col("width90") * 100).round(2), (pl.col("pin01") * 100).round(3), (pl.col("pin05") * 100).round(3),
                           (pl.col("pin95") * 100).round(3), (pl.col("pin99") * 100).round(3)))  # fmt: skip
    print("\n== Verified D5 / D16 pair (December 2025), 24-120 h: q05-q95 coverage by phase")
    parts = []
    for nm, run, blk in SHOW:
        r = pl.read_csv(ROOT / f"score_{run}/real_event_phase.csv").filter(
            (pl.col("arm") == blk)
            & (pl.col("band") == "24-120 h")
            & pl.col("event").str.contains("2025-12-29")
        )
        parts.append(r.with_columns(name=pl.lit(nm)))
    r = pl.concat(parts)
    print(
        r.pivot(on="name", index=["event", "phase"], values="cover90")
        .select("event", "phase", *[pl.col(n).round(3) for n, _, _ in SHOW])
        .sort("event", "phase")
    )
    print("\n== Event-free rows, 24-120 h (points of capacity x100): width and pinball")
    ef = pl.concat([
        pl.read_csv(ROOT / f"score_{run}/event_free.csv").filter((pl.col("arm") == blk) & (pl.col("band") == "24-120 h")).with_columns(name=pl.lit(nm))
        for nm, run, blk in SHOW
    ])  # fmt: skip
    print(ef.select("group", "name", "n_rows", pl.col("width90").round(2), pl.col("pin01").round(3), pl.col("pin05").round(3), pl.col("pin95").round(3),
                    pl.col("pin99").round(3), pl.col("loss90_minus_V").round(4), pl.col("loss90_mde").round(4)).sort("group", "name"))  # fmt: skip
    contrasts()
    print("\n== Gates")
    for run in ARMS:
        print(run)
        print(pl.read_csv(ROOT / f"score_{run}/gates.csv"))


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    text = buf.getvalue()
    (ROOT / "report.txt").write_text(text)
    print(text)
