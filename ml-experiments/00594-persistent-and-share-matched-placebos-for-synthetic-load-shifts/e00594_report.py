# ruff: noqa: E501, D103, PLC0206
"""Experiment 594: print every table the write-up quotes (also written to `report.txt`).

Arms: T0 = experiment 584's C1 (stored, the reference); PS (P-share) and PP (P-persist), this
experiment; T3 = experiment 590's variance-matched noise placebo and P5 = experiment 574's plain P5
(both stored, shown for reference and not counted). Reads each arm's score folder (`score_<arm>/`,
written by `e00594_score.py`), the per-event follow table (`follow_events.parquet`), and
`row_arrays.npz` for event-free CRPS contrasts. Counted contrasts: 8 = {PS, PP} minus T0 x {coverage
after the end, follow ratio after the end, event-free CRPS at the eight MVA Primaries, event-free
CRPS at the six MW Primaries}.
"""

import io
from contextlib import redirect_stdout

import e00582_arms as a  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT = (
    ML_EXPERIMENTS_DATA_DIR
    / "00594-persistent-and-share-matched-placebos-for-synthetic-load-shifts"
)
ROOT590 = ML_EXPERIMENTS_DATA_DIR / "00590-synthetic-load-shifts-in-the-training-targets"
ROOT584 = ML_EXPERIMENTS_DATA_DIR / "00584-nra-candidate-pack-x6"
SCORE = {
    "T0": ROOT584 / "score_c1",
    "PS": ROOT / "score_ps",
    "PP": ROOT / "score_pp",
    "T3": ROOT590 / "score_t3",
    "P5": ROOT590 / "score_p5",
}
PH = ["before", "days0-3", "days4-end", "after0-7", "after7-14"]
FOLLOW_PH = {
    "before": "before",
    "days0-3": "d0-3",
    "days4-end": "d4-end",
    "after0-7": "after0-7",
    "after7-14": "after7-14",
}
GROUPS = ["both groups", "MVA-only eight", "signed-MW six"]
N_COUNTED, N_BOOT = 8, 2000
COUNTED_ARMS = ("PS", "PP")
"""Arms whose contrast with T0 is counted; T3 and P5 are shown for reference and are not counted."""
RNG = np.random.default_rng(594)
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)


def interval(*, d: np.ndarray) -> tuple[float, float]:
    """Bootstrap interval of the mean of `d` (resampling its entries) at the Bonferroni level of 8."""
    alpha = 0.05 / N_COUNTED
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


def group_filter(df: pl.DataFrame, g: str) -> pl.DataFrame:
    return df if g == "both groups" else df.filter(pl.col("group") == g)


def bank() -> pl.DataFrame:
    parts = []
    for name, d in SCORE.items():
        b = pl.read_csv(d / "bank_summary.csv").filter(
            (pl.col("arm") == "V")
            & (pl.col("band") == "24-120 h")
            & (pl.col("length_class") == "15 and 30 days")
        )
        parts.append(b.with_columns(name=pl.lit(name)))
    return pl.concat(parts)


def follow_events() -> pl.DataFrame:
    f = pl.read_parquet(ROOT / "follow_events.parquet").filter(
        (pl.col("band") == "24-120h") & pl.col("length_days").cast(pl.Int64).is_in([15, 30])
    )
    sm = pl.read_csv(
        ML_EXPERIMENTS_DATA_DIR
        / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
    ).select("primary", "group")
    return f.join(sm, on="primary")


def coverage_and_follow_contrasts(fe: pl.DataFrame) -> None:
    """Print event-paired contrasts of each arm minus T0 after the end (event-level means)."""
    sub = fe.filter(pl.col("phase") == "after0-7")
    rows = []
    for g in GROUPS:
        x = group_filter(sub, g)
        piv = {
            m: x.pivot(on="arm", index=["month", "pass_r", "primary"], values=m).sort(
                "month", "pass_r", "primary"
            )
            for m in ("cov90", "follow_q50", "nra_side95")
        }
        for m, p in piv.items():
            for k in ("ps", "pp", "t3", "p5"):
                ok = p.filter(pl.col(k).is_not_null() & pl.col("t0").is_not_null())
                d = (ok[k] - ok["t0"]).to_numpy()
                lo, hi = interval(d=d)
                rows.append(
                    {
                        "group": g,
                        "measure": m,
                        "arm": k.upper(),
                        "counted": k.upper() in COUNTED_ARMS
                        and g == "both groups"
                        and m in ("cov90", "follow_q50"),
                        "n_events": len(d),
                        "T0": ok["t0"].mean(),
                        "arm_value": ok[k].mean(),
                        "diff": d.mean(),
                        "lo": lo,
                        "hi": hi,
                        "mde": (hi - lo) / 2,
                    }
                )
    print(pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(4)))


def follow_by_phase(fe: pl.DataFrame) -> None:
    print(
        "\n== Mean follow ratio at q50 and q95/q05 safety side by phase (24-120 h, 15 and 30 day injected events)"
    )
    for g in GROUPS:
        x = group_filter(fe, g)
        out = x.group_by("arm", "phase").agg(
            pl.len().alias("n_events"),
            pl.col("follow_q50").mean().round(3),
            pl.col("follow_q95").mean().round(3),
            pl.col("follow_q05").mean().round(3),
            pl.col("cov90").mean().round(3),
        )
        print(g)
        print(out.sort("phase", "arm"))


def quantile_crps(*, q: np.ndarray, y: np.ndarray) -> np.ndarray:
    taus = np.asarray(a.TAUS)
    d = y[:, None] - q
    return 2.0 * np.maximum(taus * d, (taus - 1) * d).mean(axis=1)


def event_free_contrasts() -> pl.DataFrame:
    raw = {k: np.load(d / "row_arrays.npz", allow_pickle=True) for k, d in SCORE.items()}
    key = {
        k: r["prim"].astype(np.int64) * 10**12 + r["origin_idx"].astype(np.int64) * 10**6 + r["idx"]
        for k, r in raw.items()
    }
    common = key["T0"]
    for k in key:
        common = np.intersect1d(common, key[k])
    print(f"rows: { ({k: len(v) for k, v in key.items()}) }, shared {len(common)}")
    take = {k: np.argsort(key[k])[np.isin(np.sort(key[k]), common)] for k in raw}
    z = {
        k: {f: raw[k][f][take[k]] for f in ("V", "nra", "free", "lead", "month", "group")}
        for k in raw
    }
    for k in z:
        assert np.array_equal(z["T0"]["nra"], z[k]["nra"])
    z0 = z["T0"]
    sel0 = z0["free"] & (z0["lead"] >= 24)
    months = sorted(set(z0["month"]))
    y = z0["nra"]
    metrics = {
        "CRPS": lambda q: quantile_crps(q=q, y=y),
        "pinball q05+q95 mean": lambda q: (
            0.5
            * (
                a.pinball(y=y, q=q[:, a.IQ[0.05]], tau=0.05)
                + a.pinball(y=y, q=q[:, a.IQ[0.95]], tau=0.95)
            )
        ),
        "pinball q01+q99 mean": lambda q: (
            0.5
            * (
                a.pinball(y=y, q=q[:, a.IQ[0.01]], tau=0.01)
                + a.pinball(y=y, q=q[:, a.IQ[0.99]], tau=0.99)
            )
        ),
        "coverage q05-q95": lambda q: ((y >= q[:, a.IQ[0.05]]) & (y <= q[:, a.IQ[0.95]])).astype(
            float
        ),
        "coverage q01-q99": lambda q: ((y >= q[:, a.IQ[0.01]]) & (y <= q[:, a.IQ[0.99]])).astype(
            float
        ),
    }
    scale = {
        "CRPS": 100,
        "pinball q05+q95 mean": 100,
        "pinball q01+q99 mean": 100,
        "coverage q05-q95": 1,
        "coverage q01-q99": 1,
    }
    rows = []
    for g in ("MVA-only eight", "signed-MW six"):
        sel = sel0 & (z0["group"] == g)
        for mname, fn in metrics.items():
            v0 = fn(z0["V"]) * scale[mname]
            for k in ("PS", "PP", "T3", "P5"):
                v1 = fn(z[k]["V"]) * scale[mname]
                d = np.array(
                    [
                        v1[sel & (z0["month"] == m)].mean() - v0[sel & (z0["month"] == m)].mean()
                        for m in months
                    ]
                )
                lo, hi = interval(d=d)
                rows.append(
                    {
                        "group": g,
                        "metric": mname,
                        "arm": k,
                        "T0": v0[sel].mean(),
                        "arm_value": v1[sel].mean(),
                        "diff": d.mean(),
                        "lo": lo,
                        "hi": hi,
                        "mde": (hi - lo) / 2,
                        "counted": k in COUNTED_ARMS,
                        "months_worse": int((d > 0).sum())
                        if "coverage" not in mname
                        else int((d < 0).sum()),
                    }
                )
    out = pl.DataFrame(rows)
    print(out.with_columns(pl.col(pl.Float64).round(4)))
    print(
        "(CRPS and pinball in points of capacity x100; coverage as a share; 'months_worse' of 5 fold months)"
    )
    return out


def main() -> None:
    b = bank()
    for g in GROUPS:
        x = b.filter(pl.col("group") == g)
        print(
            f"\n== Injected bank, {g}, 15 and 30 day events, 24-120 h: q05-q95 coverage [95% event interval], safe-side share, width and pinball (x100)"
        )
        for ph in PH:
            y = x.filter(pl.col("phase") == ph)
            print(ph, "n_events", y["n_events"][0] if y.height else 0)
            print(y.select("name", pl.col("cover90").round(3), pl.col("cover90_lo").round(3), pl.col("cover90_hi").round(3), pl.col("cover98").round(3), pl.col("nra_side95").round(3), pl.col("nra_side99").round(3), (pl.col("width90") * 100).round(2), (pl.col("pin01") * 100).round(3), (pl.col("pin05") * 100).round(3), (pl.col("pin95") * 100).round(3), (pl.col("pin99") * 100).round(3)).sort("name"))  # fmt: skip
    fe = follow_events()
    follow_by_phase(fe)
    print(
        "\n== Paired contrasts arm minus T0 after the end (0-7 days), 15 and 30 day injected events, Bonferroni level of 8"
    )
    coverage_and_follow_contrasts(fe)
    print("\n== Verified D5 / D16 pair (December 2025), 24-120 h: q05-q95 coverage by phase")
    parts = []
    for name, d in SCORE.items():
        r = pl.read_csv(d / "real_event_phase.csv").filter(
            (pl.col("arm") == "V")
            & (pl.col("band") == "24-120 h")
            & pl.col("event").str.contains("2025-12-29")
        )
        parts.append(r.with_columns(name=pl.lit(name)))
    r = pl.concat(parts)
    print(
        r.pivot(on="name", index=["event", "phase"], values="cover90")
        .select("event", "phase", *[pl.col(n).round(3) for n in SCORE])
        .sort("event", "phase")
    )
    print(
        "\n== Event-free rows, 24-120 h: contrasts arm minus T0 (month-block bootstrap, Bonferroni 8)"
    )
    event_free_contrasts()
    print("\n== Event-free rows: width and pinball (x100)")
    ef = pl.concat(
        [
            pl.read_csv(d / "event_free.csv")
            .filter((pl.col("arm") == "V") & (pl.col("band") == "24-120 h"))
            .with_columns(name=pl.lit(n))
            for n, d in SCORE.items()
        ]
    )
    print(
        ef.select(
            "group",
            "name",
            "n_rows",
            pl.col("width90").round(2),
            pl.col("pin01").round(3),
            pl.col("pin05").round(3),
            pl.col("pin95").round(3),
            pl.col("pin99").round(3),
        ).sort("group", "name")
    )


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (ROOT / "report.txt").write_text(buf.getvalue())
    print(buf.getvalue())
