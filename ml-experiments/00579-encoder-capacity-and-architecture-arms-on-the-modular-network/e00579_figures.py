# ruff: noqa: E501
"""Experiment 579: CRPS and tail pinball loss of each arm minus the base B at 24-120 h (two charts)."""

from pathlib import Path

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00579-encoder-capacity-and-architecture-arms-on-the-modular-network"
DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
GROUPS = {"eight": "The eight MVA Primaries", "six": "The six MW Primaries"}
KINDS = {
    "Encoder size": (theme.BRAND_ORANGE, ["WL", "DL", "WDL", "DT"]),
    "Primary embedding and inputs": (theme.DATA_PURPLE, ["EW", "P1", "P2", "F1", "F2"]),
    "Mixer and mixture": (theme.DATA_BLUE, ["ML", "K5", "K8", "A2"]),
    "Blocks and conditioning": (theme.DATA_GREEN, ["RB", "A1"]),
}
ORDER = [a for _, (_, arms) in KINDS.items() for a in arms]


def zero_rule() -> alt.Chart:
    """Return the dashed line at zero."""
    return (
        alt.Chart(pl.DataFrame({"z": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="grey", strokeDash=[4, 3])
        .encode(x="z:Q")
    )  # ty: ignore[unresolved-attribute]


def chart(*, measure: str, value: str, title: list[str], axis: str, fname: str) -> None:
    """Write one chart: dot estimate, line month-block 95% interval, one panel per Primary group."""
    t = pl.read_csv(ML_EXPERIMENTS_DATA_DIR / NAME / "work/scores/contrasts.csv").filter(
        (pl.col("slice") == "all rows")
        & (pl.col("band") == "24-120h")
        & (pl.col("measure") == measure)
        & pl.col("set").is_in(list(GROUPS))
        & pl.col("arm").is_in(ORDER)
    )
    scale = 100.0 if value == "relative" else 1.0
    rows = []
    for r in t.iter_rows(named=True):
        kind = next(k for k, (_, arms) in KINDS.items() if r["arm"] in arms)
        lo, hi = r["low_month"], r["high_month"]
        d = r["difference"]
        if value == "relative":
            lo, hi, d = lo / r["base_mean"], hi / r["base_mean"], d / r["base_mean"]
        rows.append(
            {
                "arm": r["arm"],
                "kind": kind,
                "group": GROUPS[r["set"]],
                "d": d * scale,
                "lo": lo * scale,
                "hi": hi * scale,
                "seeds": f"{r['seeds']} seed" + ("s" if r["seeds"] > 1 else ""),
            }
        )
    df = pl.DataFrame(rows)
    base = alt.Chart().encode(
        y=alt.Y("arm:N", sort=ORDER, title=None),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=list(KINDS), range=[c for c, _ in KINDS.values()]),
            legend=alt.Legend(orient="top", title=None, columns=4),
        ),
    )
    rule = base.mark_rule().encode(x=alt.X("lo:Q", title=axis), x2="hi:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
        x="d:Q", shape=alt.Shape("seeds:N", legend=alt.Legend(orient="top", title=None))
    )  # ty: ignore[unresolved-attribute]
    layers = alt.layer(zero_rule(), rule, dot).properties(width=330, height=210)
    out = layers.facet(data=df, row=alt.Row("group:N", title=None)).properties(
        title=alt.TitleParams(title, anchor="start", fontSize=11)
    )
    out.save(str(DOCS / fname))


def main() -> None:
    """Write both SVGs next to the page."""
    chart(measure="c", value="difference", axis="CRPS minus B's, points of capacity (negative is better)", fname="crps_vs_base.svg",
          title=["No arm beats the base B on CRPS at 24-120 h by more than seed noise; WL and RB are worse at one group", "Dot: estimate (seed 0 for the Stage 1 arms; seeds 0 and 1 for DL, ML, P1, and DT). Line: unadjusted 95% interval from resampling whole months (5 blocks). Dashed line: same as B."])  # fmt: skip
    chart(measure="t", value="relative", axis="Summed 5%, 95%, and 99% pinball loss relative to B's, % (negative is better)", fname="tail_vs_base.svg",
          title=["Tail pinball loss moves by a few per cent, mostly inside the interval", "Dot and line as on the CRPS chart; the pre-registered tail gain is 2%."])  # fmt: skip


if __name__ == "__main__":
    main()
