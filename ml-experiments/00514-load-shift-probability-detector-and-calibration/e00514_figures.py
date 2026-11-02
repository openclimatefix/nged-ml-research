# ruff: noqa: E501
"""Experiment 514: figures (SVG, in the experiment's docs folder; optimise with svgo afterwards)."""

import json
import pickle
from pathlib import Path
from typing import Final

import altair as alt
import e00514_data as d514  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00514-load-shift-probability-detector-and-calibration"
)
W: Final[int] = 560
SIX: Final = ("D5", "D9", "D10", "D11", "D14", "D16")


def reliability() -> None:
    """Figure 1: reliability of the row probability on the four held-out axes at 7% prevalence."""
    ev = json.loads((d514.WORK_DIR / "eval_injected.json").read_text())["calibration"]
    rows = []
    for axis in ("held_seed", "held_month", "held_shape", "held_primary"):
        rows.extend(
            {"axis": axis, "mean_p": r["mean_p"], "observed": r["observed"], "weight": r["weight"]}
            for r in ev[f"{axis}@0.07"]["reliability"]
            if r["weight"] > 0
        )
    df = pl.DataFrame(rows).to_pandas()
    diag = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(color="#999", strokeDash=[4, 4])
        .encode(x="x:Q", y="y:Q")
    )  # ty: ignore[unresolved-attribute]
    pts = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "mean_p:Q", title="Mean predicted probability (bin)", scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y(
                "observed:Q",
                title="Share of rows inside an injected pulse",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("axis:N", title="Held-out axis"),
        )
    )
    (diag + pts).properties(
        width=W,
        height=300,
        title=alt.TitleParams(
            "Reliability of the row probability at 7% prevalence", anchor="start"
        ),
    ).save(DOCS / "figure_1_reliability.svg")


def recall() -> None:
    """Figure 2: recall of injected pulses by size and duration."""
    t = pl.read_parquet(d514.WORK_DIR / "recall_grid_rows.parquet")
    g = (
        t.group_by("size", "duration")
        .agg(pl.col("hit").mean().alias("recall"), pl.len().alias("n"))
        .with_columns((pl.col("duration") / 2).alias("hours"))
        .to_pandas()
    )
    base = alt.Chart(g).encode(
        x=alt.X("hours:O", title="Pulse duration (hours)"),
        y=alt.Y("size:O", title="Size (times the series' spread)"),
    )  # ty: ignore[unresolved-attribute]
    (
        base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
            color=alt.Color(
                "recall:Q", scale=alt.Scale(domain=[0, 1], scheme="blues"), title="Recall"
            )
        )
        + base.mark_text().encode(text=alt.Text("recall:Q", format=".2f"))  # ty: ignore[unresolved-attribute]
    ).properties(
        width=W,
        height=220,
        title=alt.TitleParams(
            "Recall of injected pulses at probability 0.5 or more", anchor="start"
        ),
    ).save(DOCS / "figure_2_recall.svg")


def shift() -> None:
    """Figure 3: share of the national same-sign shifted rows with p of at least 0.5, by block age."""
    ev = json.loads((d514.WORK_DIR / "eval_real.json").read_text())["shift"]
    rows = [
        {"fold": f, "age": int(a), "share": v["mean_share_p_ge_half"]}
        for f, d in ev.items()
        for a, v in d.items()
    ]
    alt.Chart(pl.DataFrame(rows).to_pandas()).mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("age:O", title="Days since the shifted 7-day block ended"),
        y=alt.Y(
            "share:Q",
            title="Share of shifted rows with p of 0.5 or more",
            scale=alt.Scale(domain=[0, 1]),
        ),
        color=alt.Color("fold:N", title="Fold"),
    ).properties(
        width=W,
        height=240,
        title=alt.TitleParams("The national same-sign shift of experiment 511", anchor="start"),
    ).save(DOCS / "figure_3_national_shift.svg")


def events() -> None:
    """Figures 4 to 9: p and the scaled residual around labelled events (three of the six, three of the eight)."""
    t = pickle.loads((d514.WORK_DIR / "p_2026-06.pkl").read_bytes())
    k = len(t["cutoffs"]) - 1
    c, origin = int(t["cutoffs"][k]), int(t["table_origin"][k])
    fold = d514.load_fold(month="2026-06")
    names = d514.sites()
    table = d514.labelled_event_table().filter(
        (pl.col("onset_idx") > origin + 30 * 48)
        & (pl.col("end_idx") < c - 14 * 48)
        & (pl.col("length_days") >= 0.5)
    )
    chosen = []
    for want in (True, False):
        pool = table.filter(pl.col("label").is_in(SIX) == want).sort("depth_share", descending=True)
        chosen += pool.head(3).to_dicts()
    for n, e in enumerate(chosen, start=1):
        i = names.index(e["label"])
        a, b = int(e["onset_idx"]), int(e["end_idx"])
        lo, hi = max(origin, a - 4 * 48), min(c - 8, b + 4 * 48)
        idx = np.arange(lo, hi)
        df = pl.DataFrame(
            {
                "hour": (idx - a) / 2.0,
                "p": t["p"][k][i, lo - origin : hi - origin].astype(float),
                "z": fold.z[i, lo:hi],
            }
        ).to_pandas()
        shade = (
            alt.Chart(pl.DataFrame({"s": [0.0], "e": [(b - a) / 2.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rect(opacity=0.15, color=theme.BRAND_ORANGE)
            .encode(x="s:Q", x2="e:Q")
        )  # ty: ignore[unresolved-attribute]
        zc = (
            alt.Chart(df)  # ty: ignore[unresolved-attribute]
            .mark_line(color=theme.DATA_SKY, opacity=0.8)
            .encode(
                x=alt.X("hour:Q", title="Hours since the labelled onset"),
                y=alt.Y("z:Q", title="Scaled residual"),
            )
        )  # ty: ignore[unresolved-attribute]
        pc = (
            alt.Chart(df)  # ty: ignore[unresolved-attribute]
            .mark_line(color=theme.DATA_BLUE)
            .encode(
                x="hour:Q", y=alt.Y("p:Q", title="Probability p", scale=alt.Scale(domain=[0, 1]))
            )
        )  # ty: ignore[unresolved-attribute]
        grp = "signed-MW" if e["label"] in SIX else "MVA-only"
        (
            alt.vconcat(
                (shade + zc).properties(width=W, height=140),
                (shade + pc).properties(width=W, height=110),
            )
            .properties(
                title=alt.TitleParams(
                    f"{e['label']} ({grp}), labelled event of {e['length_days']:.1f} days",
                    anchor="start",
                )
            )
            .save(DOCS / f"figure_{3 + n}_event_{n}.svg")
        )


def main() -> None:
    """Draw every figure."""
    reliability()
    recall()
    shift()
    events()


if __name__ == "__main__":
    main()
