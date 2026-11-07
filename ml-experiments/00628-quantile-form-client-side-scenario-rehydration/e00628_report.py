# ruff: noqa: E501, ANN001, ANN202, D103, PERF401, B018
"""Experiment 628: contrasts with month-block bootstrap intervals, printed as a report."""

import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00628-quantile-form-client-side-scenario-rehydration"
W: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
ALPHA: Final[float] = 0.0167
N_BOOT: Final[int] = 20000
RNG = np.random.default_rng(628)
TAILS: Final[tuple[str, ...]] = ("clamp", "linear", "gaussian")
COUPLINGS: Final[dict[str, str]] = {"IND": "J0", "COP": "J5", "SS": "J3"}


def month(col: str = "power_fcst_init_time") -> pl.Expr:
    return pl.col(col).dt.strftime("%Y-%m").alias("month")


def boot_ratio(num: np.ndarray, den: np.ndarray) -> tuple[float, float, float]:
    """Return mean(num)/mean(den) - 1 with a month-block interval."""
    n = len(num)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    r = num[idx].mean(1) / den[idx].mean(1) - 1
    lo, hi = np.quantile(r, [ALPHA / 2, 1 - ALPHA / 2])
    return float(num.mean() / den.mean() - 1), float(lo), float(hi)


def boot_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Return mean(a) - mean(b) with a month-block interval."""
    n = len(a)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    r = (a[idx] - b[idx]).mean(1)
    lo, hi = np.quantile(r, [ALPHA / 2, 1 - ALPHA / 2])
    return float((a - b).mean()), float(lo), float(hi)


def crps_report() -> None:
    c = pl.read_parquet(W / "crps.parquet").with_columns(month())
    print(
        "== CRPS (share of capacity points; contrast against J0, the in-pipeline independent paths at k = 51)"
    )
    for window, bands in {"days_1_10": None, "h12_36": ["h12_36"]}.items():
        sub = c if bands is None else c.filter(pl.col("band").is_in(bands))
        m = sub.group_by("month", "arm", "k", "family").agg(
            (pl.col("crps_sum").sum() / pl.col("n").sum()).alias("crps")
        )
        for fam in ("demand", "wind", "solar"):
            ref = m.filter((pl.col("arm") == "J0") & (pl.col("family") == fam)).sort("month")
            for k in (51, 1000):
                for coupling in ("COP",):
                    for tail in TAILS:
                        a = m.filter(
                            (pl.col("arm") == f"{coupling}-{tail}")
                            & (pl.col("k") == k)
                            & (pl.col("family") == fam)
                        ).sort("month")
                        est, lo, hi = boot_ratio(a["crps"].to_numpy(), ref["crps"].to_numpy())
                        print(
                            f"{window} {fam:6s} k={k:4d} {tail:8s}: {est * 100:+.2f}% ({lo * 100:+.2f}% to {hi * 100:+.2f}%), abs {a['crps'].mean():.3f} vs J0 {ref['crps'].mean():.3f}"
                        )
            # tail rules against each other at k = 1000
            a = m.filter(
                (pl.col("arm") == "COP-gaussian")
                & (pl.col("k") == 1000)
                & (pl.col("family") == fam)
            ).sort("month")
            b = m.filter(
                (pl.col("arm") == "COP-clamp") & (pl.col("k") == 1000) & (pl.col("family") == fam)
            ).sort("month")
            est, lo, hi = boot_ratio(a["crps"].to_numpy(), b["crps"].to_numpy())
            print(
                f"{window} {fam:6s} gaussian vs clamp at k=1000: {est * 100:+.2f}% ({lo * 100:+.2f}% to {hi * 100:+.2f}%)"
            )


def exceed_report() -> None:
    e0 = pl.read_parquet(W / "exceed.parquet")
    e1 = pl.read_parquet(W / "exceed_analytic.parquet")
    e = pl.concat([e0.filter(pl.col("arm").is_in(["N_knots", "J0", "J3", "J5"])), e1]).with_columns(
        month()
    )
    bands = ("h0_12", "h12_36", "h36_120", "h120_240")

    def rate(arm, fam, band, level):
        s = e.filter(
            (pl.col("arm") == arm)
            & (pl.col("family") == fam)
            & (pl.col("level") == level)
            & (pl.col("band") == band if band else True)
        )
        return s["hits"].sum() / max(s["n"].sum(), 1)  # ty: ignore[unsupported-operator]

    print(
        "== exceedance rate (%) at the four knot levels: stored knots, in-pipeline J5 (51-path empirical quantile), client quantile function (all rules equal the knots)"
    )
    for fam in ("demand", "wind", "solar"):
        for band in bands:
            parts = []
            for level in (0.01, 0.05, 0.95, 0.99):
                parts.append(
                    f"q{level}: knots {rate('N_knots', fam, band, level) * 100:5.2f} J5 {rate('J5', fam, band, level) * 100:5.2f} client {rate('Q11-gaussian', fam, band, level) * 100:5.2f}"
                )
            print(f"{fam:6s} {band:8s} " + " | ".join(parts))
    print(
        "== bar: max over families, bands and the four knot levels of |client - knots| and of |J5 - knots| (percentage points)"
    )
    for arm in ("Q11-clamp", "Q11-linear", "Q11-gaussian", "J5"):
        worst = max(
            abs(rate(arm, f, b, lv) - rate("N_knots", f, b, lv)) * 100
            for f in ("demand", "wind", "solar")
            for b in bands
            for lv in (0.01, 0.05, 0.95, 0.99)
        )
        print(f"{arm}: {worst:.2f} pp")
    print(
        "== past the knots: share of actuals below the 0.1%, 0.2%, 0.5% quantiles and above the 99.5%, 99.8%, 99.9% quantiles (nominal 0.1, 0.2, 0.5, 0.5, 0.2, 0.1)"
    )
    for fam in ("demand", "wind", "solar"):
        for tail in TAILS:
            arm = f"Q11-{tail}"
            vals = [
                rate(arm, fam, None, lv) * 100 for lv in (0.001, 0.002, 0.005, 0.995, 0.998, 0.999)
            ]
            print(f"{fam:6s} {tail:8s}: " + "  ".join(f"{v:5.2f}" for v in vals))
    print("== q99 sampling error: monthly spread of the q99 exceedance of the stored knots")
    for fam in ("demand", "wind", "solar"):
        s = (
            e.filter(
                (pl.col("arm") == "N_knots") & (pl.col("family") == fam) & (pl.col("level") == 0.99)
            )
            .group_by("month")
            .agg(pl.col("hits").sum(), pl.col("n").sum())
        )
        r = (s["hits"] / s["n"]).to_numpy()
        n = s["n"].sum()
        p = s["hits"].sum() / n  # ty: ignore[unsupported-operator]
        print(
            f"{fam:6s} rate {p * 100:.2f}%, naive binomial se {np.sqrt(p * (1 - p) / n) * 100:.3f} pp (n={n}), 10-month spread sd {r.std(ddof=1) * 100:.2f} pp"
        )


def coherence_report() -> None:
    c = pl.read_parquet(W / "coherence.parquet").with_columns(month())
    base = c.filter((pl.col("window") == "days_1_10") & (pl.col("slice") == "all"))
    print(
        "== variogram score at 51 paths (days 1 to 10): client arm vs in-pipeline arm of the same coupling"
    )
    groups = {
        "demand": base.filter((pl.col("block") == "spatial") & (pl.col("label") == "demand")),
        "wind": base.filter((pl.col("block") == "spatial") & (pl.col("label") == "wind")),
        "solar": base.filter((pl.col("block") == "spatial") & (pl.col("label") == "solar")),
        "temporal": base.filter(pl.col("block") == "temporal"),
        "joint": base.filter(pl.col("block") == "joint"),
    }
    for gname, df in groups.items():
        per = (
            df.group_by("month", "power_fcst_init_time", "arm")
            .agg(pl.col("variogram").mean())
            .group_by("month", "arm")
            .agg(pl.col("variogram").mean())
        )
        for coupling, ref in COUPLINGS.items():
            r = per.filter(pl.col("arm") == ref).sort("month")
            for tail in TAILS:
                a = per.filter(pl.col("arm") == f"{coupling}-{tail}").sort("month")
                est, lo, hi = boot_ratio(a["variogram"].to_numpy(), r["variogram"].to_numpy())
                print(
                    f"{gname:8s} {coupling}-{tail:8s} vs {ref}: {est * 100:+.1f}% ({lo * 100:+.1f}% to {hi * 100:+.1f}%), abs {a['variogram'].mean():.4f} vs {r['variogram'].mean():.4f}"
                )
    print("== coupling contrasts among client arms (gaussian tail): COP and SS against IND")
    for gname, df in groups.items():
        per = (
            df.group_by("month", "power_fcst_init_time", "arm")
            .agg(pl.col("variogram").mean())
            .group_by("month", "arm")
            .agg(pl.col("variogram").mean())
        )
        ind = per.filter(pl.col("arm") == "IND-gaussian").sort("month")
        for coupling in ("COP", "SS"):
            a = per.filter(pl.col("arm") == f"{coupling}-gaussian").sort("month")
            est, lo, hi = boot_ratio(a["variogram"].to_numpy(), ind["variogram"].to_numpy())
            print(
                f"{gname:8s} {coupling}-gaussian vs IND-gaussian: {est * 100:+.1f}% ({lo * 100:+.1f}% to {hi * 100:+.1f}%)"
            )


def query_report() -> None:
    q = pl.read_parquet(W / "query.parquet").with_columns(month())
    print(
        "== query picks (points of capacity): mean |set A - set B| (stability), mean |set A - N51| (distance to in-pipeline), by arm"
    )
    for pick in ("wind_min", "solar_p05"):
        for coupling, ref in COUPLINGS.items():
            n51 = q.filter(pl.col("arm") == ref).select(
                "power_fcst_init_time", pl.col(pick).alias("n51")
            )
            for tail in TAILS:
                arm = f"{coupling}-{tail}"
                a = q.filter((pl.col("arm") == arm) & (pl.col("set") == "A")).select(
                    "power_fcst_init_time", pl.col(pick).alias("a")
                )
                b = q.filter((pl.col("arm") == arm) & (pl.col("set") == "B")).select(
                    "power_fcst_init_time", pl.col(pick).alias("b")
                )
                a51 = q.filter((pl.col("arm") == arm) & (pl.col("set") == "A51")).select(
                    "power_fcst_init_time", pl.col(pick).alias("a51")
                )
                j = (
                    a.join(b, on="power_fcst_init_time")
                    .join(n51, on="power_fcst_init_time")
                    .join(a51, on="power_fcst_init_time")
                )
                print(
                    f"{pick:9s} {arm:13s}: mean pick {j['a'].mean():7.2f} (N51 {j['n51'].mean():7.2f}), |A-B| {(j['a'] - j['b']).abs().mean():.3f}, |A-N51| {(j['a'] - j['n51']).abs().mean():.3f}, |A51-N51| {(j['a51'] - j['n51']).abs().mean():.3f}"
                )


def main() -> None:
    which = sys.argv[1:] or ["crps", "exceed", "coherence", "query"]
    print("runs", pl.read_parquet(W / "crps.parquet")["power_fcst_init_time"].n_unique())
    {
        "crps": crps_report,
        "exceed": exceed_report,
        "coherence": coherence_report,
        "query": query_report,
    }
    for name in which:
        {
            "crps": crps_report,
            "exceed": exceed_report,
            "coherence": coherence_report,
            "query": query_report,
        }[name]()


if __name__ == "__main__":
    main()
