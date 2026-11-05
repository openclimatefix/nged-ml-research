# ruff: noqa: E501, D103
"""Experiment 584: print every table the write-up quotes (also written to `report.txt`).

Reads experiment 583's tables, produced by `e00584_score.py` for arms C0 and C1, and
`row_arrays.npz` for the paired contrasts and the quantile-based CRPS. Arm names: C0 = V of the
v3+D5 run; C1 = V of the P5 run; C2 = A2, C3 = Union, C4 and C4u = the width-matched controls.
"""

import io
from contextlib import redirect_stdout

import e00582_arms as a  # ty: ignore[unresolved-import]
import e00583_score as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT = ML_EXPERIMENTS_DATA_DIR / "00584-nra-candidate-pack-x6"
SHOW = [("C0", "c0", "V"), ("C0+A2", "c0", "A2"), ("C0+Union", "c0", "Union"), ("C1", "c1", "V"),
        ("C2", "c1", "A2"), ("C3", "c1", "Union"), ("C4", "c1", "C4"), ("C4u", "c1", "C4u"), ("C4w", "c1_w1.081", "C4u")]  # fmt: skip
PH = ["before", "days0-3", "days4-end", "after0-7", "after7-14"]
GROUPS = ["both groups", "MVA-only eight", "signed-MW six"]
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)


def bank() -> pl.DataFrame:
    parts = []
    for name, run, arm in SHOW:
        b = pl.read_csv(ROOT / f"score_{run}/bank_summary.csv").filter(pl.col("arm") == arm)
        parts.append(b.with_columns(name=pl.lit(name)))
    return pl.concat(parts)


def quantile_crps(*, q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the per-row CRPS estimate 2 x mean pinball loss over the 11 stored quantile levels."""
    taus = np.asarray(a.TAUS)
    d = y[:, None] - q
    pin = np.maximum(taus * d, (taus - 1) * d)
    return 2.0 * pin.mean(axis=1)


def contrasts() -> None:
    """C1 minus C0 on event-free rows at 24-120 h (points of capacity), month-block Bonferroni intervals."""
    raw = {r: np.load(ROOT / f"score_{r}/row_arrays.npz", allow_pickle=True) for r in ("c0", "c1")}
    key = {
        r: raw[r]["prim"].astype(np.int64) * 10**12
        + raw[r]["origin_idx"].astype(np.int64) * 10**6
        + raw[r]["idx"]
        for r in raw
    }
    assert all(len(np.unique(k)) == len(k) for k in key.values())
    common = np.intersect1d(key["c0"], key["c1"])
    print(f"rows: C0 {len(key['c0'])}, C1 {len(key['c1'])}, shared {len(common)}")
    take = {r: np.argsort(key[r])[np.isin(np.sort(key[r]), common)] for r in raw}
    z = {
        r: {k: raw[r][k][take[r]] for k in ("V", "nra", "free", "lead", "month", "group")}
        for r in raw
    }
    assert np.array_equal(z["c0"]["nra"], z["c1"]["nra"])
    z0, z1 = z["c0"], z["c1"]
    sel0 = z0["free"] & (z0["lead"] >= 24)
    months = sorted(set(z0["month"]))
    y = z0["nra"]
    metrics = {
        "CRPS (quantile approximation)": lambda q: quantile_crps(q=q, y=y),
        "pinball q05+q95 mean": lambda q: 0.5 * (a.pinball(y=y, q=q[:, a.IQ[0.05]], tau=0.05) + a.pinball(y=y, q=q[:, a.IQ[0.95]], tau=0.95)),
        "pinball q01+q99 mean": lambda q: 0.5 * (a.pinball(y=y, q=q[:, a.IQ[0.01]], tau=0.01) + a.pinball(y=y, q=q[:, a.IQ[0.99]], tau=0.99)),
    }  # fmt: skip
    rows = []
    for gname in ("MVA-only eight", "signed-MW six"):
        sel = sel0 & (z0["group"] == gname)
        for mname, fn in metrics.items():
            v0, v1 = fn(z0["V"]) * 100, fn(z1["V"]) * 100
            d = np.array(
                [
                    v1[sel & (z0["month"] == m)].mean() - v0[sel & (z0["month"] == m)].mean()
                    for m in months
                ]
            )
            lo, hi = s.bonferroni_interval(diffs=d)
            rows.append({"group": gname, "metric": mname, "C0": v0[sel].mean(), "C1": v1[sel].mean(),
                         "C1_minus_C0": d.mean(), "lo": lo, "hi": hi, "mde": (hi - lo) / 2})  # fmt: skip
    print(pl.DataFrame(rows))


def main() -> None:
    b = bank()
    long_ = b.filter((pl.col("length_class") == "15 and 30 days") & (pl.col("band") == "24-120 h"))
    for g in GROUPS:
        x = long_.filter(pl.col("group") == g)
        print(
            f"\n== Injected bank, {g}, 15 and 30 day events, 24-120 h: q05-q95 coverage [95% interval] / one-sided NRA share q05 or q95 / width90 (points of capacity x100)"
        )
        for ph in PH:
            y = x.filter(pl.col("phase") == ph).sort("name")
            print(ph, "n_events", y["n_events"][0] if y.height else 0)
            print(y.select("name", pl.col("cover90").round(3), pl.col("cover90_lo").round(3), pl.col("cover90_hi").round(3),
                           pl.col("cover98").round(3), pl.col("nra_side95").round(3), pl.col("nra_side99").round(3),
                           (pl.col("width90") * 100).round(2), (pl.col("pin01") * 100).round(3), (pl.col("pin05") * 100).round(3),
                           (pl.col("pin95") * 100).round(3), (pl.col("pin99") * 100).round(3)).sort("name"))  # fmt: skip
    print("\n== Verified D5 / D16 pair (December 2025), 24-120 h: q05-q95 coverage by phase")
    parts = []
    for name, run, arm in SHOW:
        r = pl.read_csv(ROOT / f"score_{run}/real_event_phase.csv").filter(
            (pl.col("arm") == arm)
            & (pl.col("band") == "24-120 h")
            & pl.col("event").str.contains("2025-12-29")
        )
        parts.append(r.with_columns(name=pl.lit(name)))
    r = pl.concat(parts)
    print(
        r.pivot(on="name", index=["event", "phase"], values="cover90")
        .select(pl.col("event"), pl.col("phase"), *[pl.col(n).round(3) for n, _, _ in SHOW])
        .sort("event", "phase")
    )
    print("\n== Event-free rows, 24-120 h (points of capacity x100): width and pinball")
    ef = []
    for name, run, arm in SHOW:
        f = pl.read_csv(ROOT / f"score_{run}/event_free.csv").filter(
            (pl.col("arm") == arm) & (pl.col("band") == "24-120 h")
        )
        ef.append(f.with_columns(name=pl.lit(name)))
    ef = pl.concat(ef)
    print(ef.select("group", "name", "n_rows", pl.col("width90").round(2), pl.col("pin01").round(3), pl.col("pin05").round(3),
                    pl.col("pin95").round(3), pl.col("pin99").round(3), pl.col("loss90_minus_V").round(4), pl.col("loss90_mde").round(4),
                    pl.col("loss98_minus_V").round(4), pl.col("loss98_mde").round(4)).sort("group", "name"))  # fmt: skip
    print("\n== C1 minus C0, event-free rows, 24-120 h, Bonferroni (16 counted tests)")
    contrasts()
    print("\n== Gates")
    for run in ("c0", "c1"):
        print(run)
        print(pl.read_csv(ROOT / f"score_{run}/gates.csv"))
    z = np.load(ROOT / "score_c1/row_arrays.npz", allow_pickle=True)
    print(
        "width factors C4, C4u (C1):",
        z["factors"],
        "C0:",
        np.load(ROOT / "score_c0/row_arrays.npz", allow_pickle=True)["factors"],
    )


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    text = buf.getvalue()
    (ROOT / "report.txt").write_text(text)
    print(text)
