# ruff: noqa: E501
"""Aggregate experiment 626: exceedance tables, relative scores, the registered gate, and figures.

Reads `work/marginal_rows.parquet` and `work/path_rows.parquet` (stage 2) and prints the tables
that `result.md` quotes. Intervals are month-block bootstrap (10,000 resamples of the test months,
95%). Writes `work/report_tables.parquet` and the figures into the docs folder.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00626-*/env.sh
uv run --no-sync python ml-experiments/00626-*/e00626_report.py
```
"""

from collections.abc import Callable
from pathlib import Path
from typing import Final

import altair as alt
import e00626_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / c.SLUG
ARM_NAMES: Final[dict[str, str]] = {
    "A": "A unchanged",
    "B": "B one factor",
    "C": "C wider levels",
    "D": "D two factors",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "A": "#1A1A1A",
    "B": theme.DATA_BLUE,
    "C": theme.DATA_GREEN,
    "D": theme.BRAND_ORANGE,
}
N_BOOT: Final[int] = 10_000
RATE_TOLERANCE: Final[float] = 0.01
CRPS_LIMIT: Final[float] = 0.01
PATH_LIMIT: Final[float] = 0.02
RATES: Final[tuple[str, ...]] = ("below_q05", "below_q10", "above_q90", "above_q95")


def block_bootstrap(
    *, per_month: np.ndarray, stat: Callable[[np.ndarray], float], seed: int = 0
) -> tuple[float, float, float]:
    """Return a statistic of per-month rows (months, k) and its 95% month-block bootstrap interval."""
    rng = np.random.default_rng(seed)
    n = per_month.shape[0]
    draws = rng.integers(0, n, size=(N_BOOT, n))
    values = np.array([stat(per_month[d]) for d in draws])
    return (
        float(stat(per_month)),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def rate_table(*, marg: pl.DataFrame) -> pl.DataFrame:
    """Return pooled exceedance rates with intervals per arm, family, band, and slice."""
    out = []
    keys = (
        marg.select("arm", "family", "band", "slice")
        .unique()
        .sort("arm", "family", "band", "slice")
    )
    months = sorted(marg["month"].unique().to_list())
    by = marg.group_by("arm", "family", "band", "slice", "month").agg(
        pl.col("n", *RATES, "crps_sum").sum()
    )
    for arm, family, band, sl in keys.iter_rows():
        sub = by.filter(
            (pl.col("arm") == arm)
            & (pl.col("family") == family)
            & (pl.col("band") == band)
            & (pl.col("slice") == sl)
        )
        sub = (
            sub.join(pl.DataFrame({"month": months}), on="month", how="right")
            .sort("month")
            .fill_null(0)
        )
        mat = sub.select("n", *RATES, "crps_sum").to_numpy().astype(float)
        row = {"arm": arm, "family": family, "band": band, "slice": sl, "n": int(mat[:, 0].sum())}
        for j, name in enumerate(RATES, start=1):
            est, lo, hi = block_bootstrap(
                per_month=mat[:, [0, j]], stat=lambda m: m[:, 1].sum() / max(m[:, 0].sum(), 1)
            )
            row[name], row[name + "_lo"], row[name + "_hi"] = est, lo, hi
        row["crps"] = float(mat[:, 5].sum() / max(mat[:, 0].sum(), 1))
        out.append(row)
    return pl.DataFrame(out)


def crps_contrast(*, marg: pl.DataFrame) -> pl.DataFrame:
    """Return the CRPS of arms B to D relative to arm A, pooled over the bands, per family."""
    months = sorted(marg["month"].unique().to_list())
    out = []
    base = marg.filter(pl.col("slice") == "all")
    for family in c.FAMILIES:
        per = {}
        for arm in ("A", "B", "C", "D"):
            g = (
                base.filter((pl.col("family") == family) & (pl.col("arm") == arm))
                .group_by("month")
                .agg(pl.col("n", "crps_sum").sum())
                .sort("month")
            )
            per[arm] = g.select("n", "crps_sum").to_numpy().astype(float)
        for arm in ("B", "C", "D"):
            both = np.stack([per["A"][:, 1], per[arm][:, 1], per["A"][:, 0]], axis=1)
            est, lo, hi = block_bootstrap(
                per_month=both, stat=lambda m: m[:, 1].sum() / m[:, 0].sum() - 1
            )
            out.append(
                {
                    "family": family,
                    "arm": arm,
                    "crps_a": per["A"][:, 1].sum() / per["A"][:, 0].sum(),
                    "crps_arm": per[arm][:, 1].sum() / per["A"][:, 0].sum(),
                    "rel": est,
                    "lo": lo,
                    "hi": hi,
                    "months": len(months),
                }
            )
    return pl.DataFrame(out)


def path_contrast(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return arms B to D's energy and variogram scores relative to arm A, days 1 to 10."""
    d = rows.filter((pl.col("window") == "days_1_10") & (pl.col("slice") == "all")).with_columns(
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m")
    )
    out = []
    for block in ("joint", "spatial", "temporal"):
        labels = (
            ["all"]
            if block == "joint"
            else sorted(d.filter(pl.col("block") == block)["label"].unique().to_list())
        )
        groups = [(block, "pooled")] if block != "joint" else [(block, "all")]
        sets = {"pooled": labels} if block != "joint" else {"all": labels}
        for _, name in groups:
            sub = d.filter((pl.col("block") == block) & pl.col("label").is_in(sets[name]))
            for score in ("energy", "variogram"):
                per = {
                    a: sub.filter(pl.col("arm") == a)
                    .group_by("month")
                    .agg(pl.col(score).mean())
                    .sort("month")[score]
                    .to_numpy()
                    for a in ("A", "B", "C", "D")
                }
                for arm in ("B", "C", "D"):
                    both = np.stack([per["A"], per[arm]], axis=1)
                    est, lo, hi = block_bootstrap(
                        per_month=both, stat=lambda m: m[:, 1].mean() / m[:, 0].mean() - 1
                    )
                    out.append(
                        {
                            "block": block,
                            "label": name,
                            "score": score,
                            "arm": arm,
                            "score_a": float(per["A"].mean()),
                            "rel": est,
                            "lo": lo,
                            "hi": hi,
                        }
                    )
    return pl.DataFrame(out)


def gate(*, rates: pl.DataFrame, crps: pl.DataFrame, paths: pl.DataFrame) -> pl.DataFrame:
    """Return, per arm, whether each part of the registered `supported` rule holds."""
    out = []
    cells = rates.filter(pl.col("slice") == "all")
    for arm in ("A", "B", "C", "D"):
        sub = cells.filter(pl.col("arm") == arm)
        ok95 = (sub["above_q95"] - c.TARGET).abs() <= RATE_TOLERANCE
        ok05 = (sub["below_q05"] - c.TARGET).abs() <= RATE_TOLERANCE
        row = {
            "arm": arm,
            "cells_q95_ok": int(ok95.sum()),
            "cells_q05_ok": int(ok05.sum()),
            "cells": sub.height,
        }
        if arm != "A":
            row["crps_ok"] = bool((crps.filter(pl.col("arm") == arm)["rel"] <= CRPS_LIMIT).all())
            row["paths_ok"] = bool((paths.filter(pl.col("arm") == arm)["rel"] <= PATH_LIMIT).all())
        out.append(row)
    return pl.DataFrame(out)


def figure_rates(*, rates: pl.DataFrame) -> None:
    """Draw q95 and q05 exceedance by arm, family, and band with the 4% to 6% target band."""
    d = rates.filter(pl.col("slice") == "all").with_columns(
        arm_name=pl.col("arm").replace(ARM_NAMES)
    )
    long = pl.concat(
        [
            d.select(
                "arm_name",
                "family",
                "band",
                pl.lit("q95: share of actuals above").alias("side"),
                pl.col("above_q95").alias("rate"),
                pl.col("above_q95_lo").alias("lo"),
                pl.col("above_q95_hi").alias("hi"),
            ),
            d.select(
                "arm_name",
                "family",
                "band",
                pl.lit("q05: share of actuals below").alias("side"),
                pl.col("below_q05").alias("rate"),
                pl.col("below_q05_lo").alias("lo"),
                pl.col("below_q05_hi").alias("hi"),
            ),
        ]
    ).with_columns(rate=pl.col("rate") * 100, lo=pl.col("lo") * 100, hi=pl.col("hi") * 100)
    labels = {
        "h0_12": "0-12 h",
        "h12_36": "12-36 h",
        "h36_120": "36-120 h",
        "h120_240": "120-240 h",
    }
    long = long.with_columns(
        band_label=pl.col("band").replace(labels),
        target=pl.lit(5.0),
        low=pl.lit(4.0),
        high=pl.lit(6.0),
    )
    order = list(labels.values())
    scale = alt.Scale(domain=list(ARM_NAMES.values()), range=[ARM_COLOURS[a] for a in ARM_NAMES])
    base = alt.Chart(long.to_pandas())
    x = alt.X("band_label:N", sort=order, title="Lead band", axis=alt.Axis(labelAngle=-45))
    once = base.transform_aggregate(
        low="min(low)", high="max(high)", target="min(target)", groupby=["side", "family"]
    )  # ty: ignore[unresolved-attribute]
    target = once.mark_rect(opacity=0.15, color="#888888").encode(y="low:Q", y2="high:Q")
    line = once.mark_rule(strokeDash=[4, 3], color="#555555").encode(y="target:Q")
    bars = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        x=x,
        xOffset="arm_name:N",
        y="lo:Q",
        y2="hi:Q",
        color=alt.Color("arm_name:N", legend=None, scale=scale),
    )
    points = base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
        x=x,
        xOffset="arm_name:N",
        y=alt.Y("rate:Q", title="Exceedance rate (%)", scale=alt.Scale(domain=[0, 12])),
        color=alt.Color("arm_name:N", title="Arm (grey band: 4% to 6%)", scale=scale),
    )
    chart = (
        alt.layer(target, line, bars, points)
        .properties(width=190, height=170)
        .facet(
            row=alt.Row("side:N", title=None),
            column=alt.Column("family:N", title=None, sort=list(c.FAMILIES)),
        )
        .configure_legend(orient="right")
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / "figure_1_exceedance_by_arm.svg")


def main() -> None:
    """Print the tables and draw the figures."""
    marg = pl.read_parquet(c.WORK_626 / "marginal_rows.parquet")
    rows = pl.read_parquet(c.WORK_626 / "path_rows.parquet")
    pl.Config.set_tbl_rows(300)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_fmt_float("full")
    rates = rate_table(marg=marg)
    rates.write_parquet(c.WORK_626 / "report_rates.parquet")
    crps = crps_contrast(marg=marg)
    paths = path_contrast(rows=rows)
    paths.write_parquet(c.WORK_626 / "report_paths.parquet")
    print("== exceedance, all rows")
    print(
        rates.filter(pl.col("slice") == "all")
        .select("arm", "family", "band", "n", *RATES, "crps")
        .with_columns(pl.col(*RATES, "crps").round(4))
    )
    print("== exceedance, slices")
    print(
        rates.filter(pl.col("slice") != "all")
        .select("arm", "family", "band", "slice", "n", *RATES, "crps")
        .with_columns(pl.col(*RATES, "crps").round(4))
    )
    print("== exceedance pooled over families (rows weighted)")
    pooled = (
        marg.filter(pl.col("slice") == "all")
        .group_by("arm", "band")
        .agg(pl.col("n", *RATES).sum())
        .with_columns([(pl.col(r) / pl.col("n")).round(4) for r in RATES])
        .sort("arm", "band")
    )
    print(pooled)
    print("== CRPS relative to A (per family, bands pooled)")
    print(crps)
    print("== path scores relative to A, days 1 to 10")
    print(paths.with_columns(pl.col("score_a", "rel", "lo", "hi").round(4)))
    print("== gate")
    print(gate(rates=rates, crps=crps, paths=paths))
    figure_rates(rates=rates)


if __name__ == "__main__":
    main()
