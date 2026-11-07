# ruff: noqa: E501, ANN001, ANN202, D103
"""Experiment 628, neural demand stage: contrasts with month-block intervals, printed as a report."""

import sys

import e00628_report as rep  # ty: ignore[unresolved-import]
import polars as pl

BANDS = ("h0_12", "h12_36", "h36_120", "h120_240")
TAILS = ("clamp", "linear", "gaussian")


def report(network: str) -> None:
    w = rep.W
    c = pl.read_parquet(w / f"neural_{network}_crps.parquet")
    e = pl.read_parquet(w / f"neural_{network}_exceed.parquet")
    q = pl.read_parquet(w / f"neural_{network}_query.parquet")
    print(
        f"##### {network}: origins {c['power_fcst_init_time'].n_unique()}, months {c['month'].n_unique()}"
    )
    m = c.group_by("month", "arm", "k").agg(
        (pl.col("crps_sum").sum() / pl.col("n").sum()).alias("crps")
    )
    ref = m.filter(pl.col("arm") == "N51").sort("month")
    for coupling in ("IND", "COP", "SS"):
        for tail in TAILS:
            a = m.filter((pl.col("arm") == f"{coupling}-{tail}") & (pl.col("k") == 1000)).sort(
                "month"
            )
            est, lo, hi = rep.boot_ratio(a["crps"].to_numpy(), ref["crps"].to_numpy())
            print(
                f"CRPS {coupling}-{tail:8s} (1000 draws) vs N51: {est * 100:+.2f}% ({lo * 100:+.2f}% to {hi * 100:+.2f}%), abs {a['crps'].mean():.3f} vs {ref['crps'].mean():.3f} % of capacity"
            )

    def rate(arm, level, band=None):
        s = e.filter((pl.col("arm") == arm) & (pl.col("level") == level))
        if band:
            s = s.filter(pl.col("band") == band)
        return s["hits"].sum() / max(s["n"].sum(), 1) * 100  # ty: ignore[unsupported-operator]

    print("exceedance at knot levels (%), stored knots vs client:")
    for band in BANDS:
        print(
            band,
            "  ".join(
                f"q{lv}: {rate('N_knots', lv, band):5.2f}/{rate('Q11-gaussian', lv, band):5.2f}"
                for lv in (0.01, 0.05, 0.95, 0.99)
            ),
        )
    print(
        "past the knots (%) at 0.1, 0.2, 0.5 and 99.5, 99.8, 99.9 (nominal 0.1 0.2 0.5 0.5 0.2 0.1):"
    )
    for tail in TAILS:
        print(
            f"{tail:8s}",
            "  ".join(
                f"{rate(f'Q11-{tail}', lv):5.2f}"
                for lv in (0.001, 0.002, 0.005, 0.995, 0.998, 0.999)
            ),
        )
    print(
        "q99 exceedance of the knots by month (%):",
        {
            mo: round(
                float(
                    e.filter(  # ty: ignore[unsupported-operator]
                        (pl.col("arm") == "N_knots")
                        & (pl.col("level") == 0.99)
                        & (pl.col("month") == mo)
                    )["hits"].sum()
                    / e.filter(
                        (pl.col("arm") == "N_knots")
                        & (pl.col("level") == 0.99)
                        & (pl.col("month") == mo)
                    )["n"].sum()
                    * 100
                ),
                2,
            )
            for mo in sorted(e["month"].unique())
        },
    )
    for pick in [col for col in q.columns if col.startswith("bsp_q95")]:
        for coupling in ("IND", "COP", "SS"):
            for tail in TAILS:
                arm = f"{coupling}-{tail}"
                a = q.filter((pl.col("arm") == arm) & (pl.col("set") == "A")).select(
                    "power_fcst_init_time", pl.col(pick).alias("a")
                )
                b = q.filter((pl.col("arm") == arm) & (pl.col("set") == "B")).select(
                    "power_fcst_init_time", pl.col(pick).alias("b")
                )
                n51 = q.filter((pl.col("arm") == arm) & (pl.col("set") == "A51")).select(
                    "power_fcst_init_time", pl.col(pick).alias("n51")
                )
                j = a.join(b, on="power_fcst_init_time").join(n51, on="power_fcst_init_time")
                print(
                    f"{pick} {arm:13s}: mean {j['a'].mean():6.2f} MW, |A-B| {(j['a'] - j['b']).abs().mean():.3f}, |A-N51| {(j['a'] - j['n51']).abs().mean():.3f}"
                )


if __name__ == "__main__":
    for net in sys.argv[1:] or ["K1"]:
        report(net)
