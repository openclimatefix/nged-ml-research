# ruff: noqa: D103, E501, PERF401
"""Draw experiment 555's figures from the stored score tables, and print the tail-coverage table."""

import os
from pathlib import Path
from typing import Final

import altair as alt
import e00510_common as c  # ty: ignore[unresolved-import]
import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

OUT: Final[Path] = (
    Path(os.environ["DATA_PATH_INTERNAL"])
    / "ml-experiments/00555-issue-time-lags-at-06-12-18-utc-lag-light-demand-network"
)
DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00555-issue-time-lags-at-06-12-18-utc-lag-light-demand-network"
)
WIDTH: Final[int] = 560
ISSUES: Final[dict[int, str]] = {
    9: "09 (origin)",
    12: "12",
    18: "18",
    0: "00 (next day)",
    6: "06 (next day)",
}
SETS: Final[dict[str, str]] = {
    "eight": "MVA-only eight Primaries",
    "six": "Signed-MW six Primaries",
}
BANDS: Final[dict[str, str]] = {
    "0-6h": theme.BRAND_ORANGE,
    "6-12h": theme.DATA_BLUE,
    "12-24h": theme.DATA_GREEN,
    "24-48h": theme.DATA_PURPLE,
}
ARMS: Final[dict[str, str]] = {
    "R-DW": theme.DATA_GREEN,
    "R-DROP": theme.BRAND_ORANGE,
    "R9": theme.DATA_PURPLE,
}


def zero_rule() -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas())
        .mark_rule(color=theme.BLACK_2)
        .encode(x="y:Q")  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]


def figure_gap() -> None:
    t = pl.read_csv(OUT / "contrasts.csv").filter(
        (pl.col("contrast") == "R-DW - L-1d")
        & pl.col("band").is_in(list(BANDS))
        & pl.col("set").is_in(list(SETS))
    )
    panels = []
    for key, label in SETS.items():
        part = (
            t.filter(pl.col("set") == key)
            .with_columns(
                issue_label=pl.col("issue").replace_strict(ISSUES, return_dtype=pl.String)
            )
            .to_pandas()
        )
        base = alt.Chart(part).encode(
            y=alt.Y("issue_label:N", sort=list(ISSUES.values()), title="Issue time (UTC)"),
            yOffset="band:N",
        )  # ty: ignore[unresolved-attribute]
        colour = alt.Color(
            "band:N",
            title="Lead from the issue time",
            legend=alt.Legend(orient="top"),
            scale=alt.Scale(domain=list(BANDS), range=list(BANDS.values())),
        )
        bar = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low_month:Q",
                title="CRPS minus L-1d's (points of capacity; more negative is better)",
            ),
            x2="high_month:Q",
            color=colour,
        )  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=50).encode(x="difference:Q", color=colour)  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero_rule(), bar, dot).properties(
                width=WIDTH, height=250, title=alt.TitleParams(label, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "R-DW beats the lag-free L-1d by 0.2 to 0.35 points at every issue time at the eight.",
                "Dot: mean difference. Bar: 95% interval, 5 month blocks (near min to max). Line: same as L-1d.",
            ],
            anchor="start",
            fontSize=12,
        )
    ).save(DOCS / "figure_1_gap_to_l1d_by_issue_time.svg")


def figure_arms() -> None:
    t = (
        pl.read_csv(OUT / "contrasts.csv")
        .filter(
            pl.col("contrast").is_in([f"{a} - L-1d" for a in ARMS])
            & (pl.col("band") == "0-12h")
            & pl.col("set").is_in(list(SETS))
        )
        .with_columns(
            arm=pl.col("contrast").str.extract(r"^(\S+) - "),
            issue_label=pl.col("issue").replace_strict(ISSUES, return_dtype=pl.String),
        )
    )
    panels = []
    for key, label in SETS.items():
        part = t.filter(pl.col("set") == key).to_pandas()
        base = alt.Chart(part).encode(
            y=alt.Y("issue_label:N", sort=list(ISSUES.values()), title="Issue time (UTC)"),
            yOffset="arm:N",
        )  # ty: ignore[unresolved-attribute]
        colour = alt.Color(
            "arm:N",
            title=None,
            legend=alt.Legend(orient="top"),
            scale=alt.Scale(domain=list(ARMS), range=list(ARMS.values())),
        )
        bar = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low_month:Q",
                title="CRPS minus L-1d's, 0 to 12 h (points of capacity; more negative is better)",
            ),
            x2="high_month:Q",
            color=colour,
        )  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=50).encode(x="difference:Q", color=colour)  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero_rule(), bar, dot).properties(
                width=WIDTH, height=190, title=alt.TitleParams(label, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "R9, which also reads the newest reading, does no better than R-DW at 0 to 12 h.",
                "Dot: mean difference from L-1d. Bar: 95% interval from 5 month blocks. Line: same as L-1d.",
            ],
            anchor="start",
            fontSize=12,
        )
    ).save(DOCS / "figure_2_arms_at_0_to_12h.svg")


def figure_example() -> None:
    """Draw one 12 UTC forecast of one of the eight: first day and full 10 days, against the cleaned power."""
    month = "2026-03"
    clean = c.clean_stack()
    pr = {
        a: pl.read_parquet(OUT / f"work_i12/predictions/{a}/{month}_c0.parquet")
        for a in ("R-DW", "L-1d")
    }
    table = s.load().filter((pl.col("month") == month) & (pl.col("group") == "mva_eight"))
    primary = int(table["primary"].min())  # ty: ignore[invalid-argument-type]
    origin = int(table.filter(pl.col("primary") == primary)["origin"].unique().sort()[10])  # ty: ignore[invalid-argument-type]
    rows = []
    for arm, p in pr.items():
        sub = p.filter((pl.col("primary") == primary) & (pl.col("origin") == origin)).sort("target")
        for r in sub.iter_rows(named=True):
            rows.append(
                {
                    "arm": arm,
                    "hours": (r["target"] - origin - 6) / 2.0,
                    "q05": r["q05"],
                    "q50": r["q50"],
                    "q95": r["q95"],
                }
            )
    tgt = np.array(sorted({r["hours"] for r in rows}))
    idx = (tgt * 2 + origin + 6).astype(int)
    actual = pl.DataFrame({"hours": tgt, "actual": clean[primary, idx]})
    d = pl.DataFrame(rows).to_pandas()
    a = actual.to_pandas()
    colours = {"R-DW": theme.DATA_GREEN, "L-1d": theme.DATA_BLUE}
    panels = []
    for title, hi in (("First day after the 12 UTC issue time", 24.0), ("Full 10 days", 240.0)):
        dd, aa = d[(d["hours"] > 0) & (d["hours"] <= hi)], a[(a["hours"] >= 0) & (a["hours"] <= hi)]
        band = (
            alt.Chart(dd)
            .mark_area(opacity=0.2)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("hours:Q", title="Hours after the issue time"),
                y=alt.Y("q05:Q", title="Power over capacity"),
                y2="q95:Q",
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )
        )  # ty: ignore[unresolved-attribute]
        line = alt.Chart(dd).mark_line().encode(x="hours:Q", y="q50:Q", color="arm:N")  # ty: ignore[unresolved-attribute]
        act = (
            alt.Chart(aa)
            .mark_line(
                color=theme.BLACK_1 if hasattr(theme, "BLACK_1") else "#000000", strokeDash=[3, 2]
            )
            .encode(x="hours:Q", y="actual:Q")  # ty: ignore[unresolved-attribute]
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(band, line, act).properties(
                width=WIDTH, height=170, title=alt.TitleParams(title, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "One forecast issued at 12 UTC for one of the eight Primaries (March 2026): R-DW and L-1d.",
                "Line: median. Shade: 5% to 95% band. Dashed black: cleaned power. Both read the 00 UTC weather run.",
            ],
            anchor="start",
            fontSize=12,
        )
    ).save(DOCS / "figure_3_example_forecast.svg")


def tails() -> None:
    """Tail coverage and pinball loss at 0-12 h, R-DW against L-1d, per issue time."""
    base = s.load()
    out = []
    for tag, label in ((12, "12"), (18, "18"), (0, "00"), (6, "06")):
        tt = f"{tag:02d}"
        mine = pl.concat(
            [
                pl.read_parquet(
                    OUT / f"work_i{tt}/scores/rows_{m}.parquet",
                    columns=[
                        "primary",
                        "origin",
                        "target",
                        *[
                            f"R-DW__{k}_{q}"
                            for k in ("pin", "cov")
                            for q in ("q01", "q05", "q95", "q99")
                        ],
                        "R-DW__crps",
                    ],
                )
                for m in s.MONTHS
            ]
        )
        b = base.select(
            "primary",
            "origin",
            "target",
            "group",
            "cls",
            "lead_hours",
            "main16",
            *[f"L-1d__{k}_{q}" for k in ("pin", "cov") for q in ("q01", "q05", "q95", "q99")],
        ).join(mine, on=["primary", "origin", "target"], how="left")
        step = {12: 3.0, 18: 9.0, 0: 15.0, 6: 21.0}[tag]
        b = b.with_columns(lead_hours=pl.col("lead_hours") - step).filter(
            pl.col("main16")
            & pl.col("R-DW__crps").is_finite().fill_null(False)
            & (pl.col("lead_hours") > 0)
            & (pl.col("lead_hours") < 12)
            & (pl.col("group") == "mva_eight")
        )
        row = {"issue": label, "n": b.height}
        for q in ("q01", "q05", "q95", "q99"):
            row[f"cov_{q}_R-DW"] = round(float(b[f"R-DW__cov_{q}"].mean()), 3)  # ty: ignore[invalid-argument-type]
            row[f"cov_{q}_L-1d"] = round(float(b[f"L-1d__cov_{q}"].mean()), 3)  # ty: ignore[invalid-argument-type]
        out.append(row)
    t = pl.DataFrame(out)
    t.write_csv(OUT / "tails_0_12h_eight.csv")
    with pl.Config(tbl_cols=20, tbl_width_chars=250):
        print(t)


if __name__ == "__main__":
    figure_gap()
    figure_arms()
    figure_example()
    tails()
