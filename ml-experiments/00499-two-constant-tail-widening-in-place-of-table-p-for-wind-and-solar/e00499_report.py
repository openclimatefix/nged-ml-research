"""Experiment 499: draw the two-constant-minus-table-P contrasts, absolute coverage, and examples.

Reads `contrasts.csv`, `scores.csv`, and `constants.csv` written by `e00499_two_constant.py` and
the stored rows of experiment 461. Writes SVG figures beside the write-up. Example forecasts are
drawn in fractions of capacity against hours since the run, with no calendar dates, because the
series are NGED metered generators.
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00499_two_constant as two  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

FIGURE_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / two.SLUG
WIDTH: Final[int] = 680
RULE_COLOURS: Final[dict[str, str]] = {
    "d": theme.DATA_BLUE,
    "e": theme.BRAND_ORANGE,
    "f": theme.DATA_PURPLE,
}
RULE_NAMES: Final[dict[str, str]] = {
    "d": "d: two per family",
    "e": "e: two per farm",
    "f": "f: two per farm and lead group",
}
TAIL_ORDER: Final[list[str]] = ["q01", "q05", "q95", "q99"]


def contrast_figure(*, criteria: list[str], limit: float, one_sided: bool, title: list[str],
                    name: str, site: str = "pooled", slice_name: str = "all") -> None:  # fmt: skip
    """Draw rule-minus-table-P contrasts with intervals and the equivalence band."""
    c = pl.read_csv(two.WORK / "contrasts.csv").filter(
        (pl.col("site") == site)
        & (pl.col("slice") == slice_name)
        & pl.col("rule").is_in(list(two.CANDIDATES))
        & pl.col("criterion").is_in(criteria)
    ).with_columns(
        tail=pl.col("criterion").str.slice(-3),
        rule_name=pl.col("rule").replace(RULE_NAMES),
        lo=pl.lit(0.0 if one_sided and False else -limit),
        hi=pl.lit(limit),
    )  # fmt: skip
    if one_sided:
        c = c.with_columns(lo=pl.lit(-limit * 5))
    band = alt.Chart(c).mark_rect(opacity=0.15, color=theme.DATA_GREEN).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(c).mark_rule(color=theme.GREY_2).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    y = alt.Y("family:N", title=None)
    bars = alt.Chart(c).mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title=None), x2="high:Q", y=y,
        color=alt.Color("rule_name:N", scale=alt.Scale(domain=list(RULE_NAMES.values()),
                        range=[RULE_COLOURS[r] for r in RULE_NAMES]), legend=alt.Legend(orient="top", title=None, labelLimit=400)),
        yOffset="rule_name:N",
    )  # fmt: skip
    pts = alt.Chart(c).mark_point(filled=True, size=40).encode(  # ty: ignore[unresolved-attribute]
        x="diff:Q", y=y, yOffset="rule_name:N",
        color=alt.Color("rule_name:N", legend=None,
                        scale=alt.Scale(domain=list(RULE_NAMES.values()),
                                        range=[RULE_COLOURS[r] for r in RULE_NAMES])),
    )  # fmt: skip
    chart = alt.layer(band, zero, bars, pts).properties(
        width=(WIDTH - 60) // len(criteria), height=70
    )
    chart = (
        chart.facet(column=alt.Column("tail:N", title=None, sort=TAIL_ORDER))
        if len(criteria) > 1
        else chart
    )
    chart.properties(
        title=alt.TitleParams(
            [*title[:-1], "Horizontal axis: " + title[-1].lower() + "."], anchor="start"
        )
    ).save(FIGURE_DIR / name)  # ty: ignore[unresolved-attribute]


def absolute_figure() -> None:
    """Draw each rule's absolute tail coverage against the nominal level."""
    s = pl.read_csv(two.WORK / "scores.csv").filter(
        (pl.col("slice") == "all") & (pl.col("by") == "family")
    )
    long = s.unpivot(
        index=["family", "rule"], on=["cov_q01", "cov_q05", "cov_q95", "cov_q99"],
        variable_name="criterion", value_name="coverage",
    ).with_columns(tail=pl.col("criterion").str.slice(-3))  # fmt: skip
    nominal = pl.DataFrame({"tail": TAIL_ORDER, "nominal": [0.01, 0.05, 0.95, 0.99]})
    long = long.join(nominal, on="tail")
    colours = {
        "raw": theme.GREY_1, "w1.15": theme.DATA_SKY, "P": theme.BLACK_1,
        **RULE_COLOURS,
    }  # fmt: skip
    points = alt.Chart(long).mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("rule:N", sort=list(colours), title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("coverage:Q", title=None, scale=alt.Scale(zero=False)),
        color=alt.Color("rule:N", scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                        legend=None),
    )  # fmt: skip
    rule = alt.Chart(long).mark_rule(strokeDash=[4, 3], color=theme.GREY_1).encode(y="nominal:Q")  # ty: ignore[unresolved-attribute]
    (
        alt.layer(points, rule)
        .properties(width=(WIDTH - 80) // 4, height=130)
        .facet(column=alt.Column("tail:N", title=None, sort=TAIL_ORDER), row=alt.Row("family:N", title=None))
        .resolve_scale(y="independent")
        .properties(title=alt.TitleParams(["Share of actuals at or below each tail quantile, by rule (dashed: nominal level).", "Rows: wind and solar, all scored leads. Raw: grey; 1.15 widening: sky; P: black; d, e, f: as in Figure 1."], anchor="start"))
        .save(FIGURE_DIR / "figure_4_absolute_coverage.svg")
    )  # fmt: skip


def constants_figure() -> None:
    """Draw the fitted upper and lower constants of rule e per farm and month."""
    k = pl.read_csv(two.WORK / "constants.csv").filter(pl.col("rule") == "e")
    long = k.unpivot(index=["family", "month", "key"], on=["lower", "upper"],
                     variable_name="tail", value_name="factor")  # fmt: skip
    (
        alt.Chart(long).mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month:N", title="Scored month (fitted on earlier months)",
                    axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("factor:Q", title="Constant (1 = raw spread)", scale=alt.Scale(zero=False)),
            color=alt.Color("key:N", title=None),
            strokeDash=alt.StrokeDash("tail:N", title=None),
        )
        .properties(width=WIDTH - 120, height=120)
        .facet(row=alt.Row("family:N", title=None))
        .resolve_scale(color="independent")
        .save(FIGURE_DIR / "figure_5_constants.svg")
    )  # fmt: skip


def example_figure() -> None:
    """Draw one wind and one solar run: raw, table-P-free two-constant bands, and the actual."""
    panels = []
    for family, site, month in (("wind", "Farm A", "2026-01"), ("solar", "Solar A", "2026-06")):
        frame = two.load(family=family).filter(
            (pl.col("site") == site) & (pl.col("month") == month)
        )
        init = frame.group_by("init").len().sort("len").tail(1)["init"].item()
        run = frame.filter(pl.col("init") == init).sort("lead_hours")
        k = pl.read_csv(two.WORK / "constants.csv").filter(
            (pl.col("family") == family) & (pl.col("month") == month) & (pl.col("rule") == "d")
        )
        lo, up = k["lower"].item(), k["upper"].item()
        cap = run["cap_mw"]
        run = run.with_columns(
            hours=pl.col("lead_hours"),
            actual=pl.col("actual") / cap,
            median=pl.col("q50") / cap,
            l90=(pl.col("q50") + lo * (pl.col("q05") - pl.col("q50"))) / cap,
            u90=(pl.col("q50") + up * (pl.col("q95") - pl.col("q50"))) / cap,
            l50=pl.col("q25") / cap,
            u50=pl.col("q75") / cap,
        )
        for label, limit in (("Day ahead (to 48 h)", 48.0), ("Whole horizon (to 240 h)", 241.0)):
            part = run.filter(pl.col("hours") <= limit)
            x = alt.X("hours:Q", title="Hours since the run started")
            band90 = (
                alt.Chart(part)
                .mark_area(opacity=0.2, color=theme.DATA_BLUE)
                .encode(  # ty: ignore[unresolved-attribute]
                    x=x, y=alt.Y("l90:Q", title="Fraction of capacity"), y2="u90:Q"
                )
            )
            band50 = (
                alt.Chart(part)
                .mark_area(opacity=0.3, color=theme.DATA_BLUE)
                .encode(  # ty: ignore[unresolved-attribute]
                    x=x, y="l50:Q", y2="u50:Q"
                )
            )
            med = alt.Chart(part).mark_line(color=theme.DATA_BLUE).encode(x=x, y="median:Q")  # ty: ignore[unresolved-attribute]
            act = (
                alt.Chart(part)
                .mark_line(color=theme.BRAND_ORANGE, aria=False)
                .encode(x=x, y="actual:Q")
            )  # ty: ignore[unresolved-attribute]
            panels.append(
                alt.layer(band90, band50, med, act).properties(
                    width=WIDTH - 90, height=110, title=f"{site}: {label}"
                )
            )  # fmt: skip
    alt.vconcat(*panels).save(FIGURE_DIR / "figure_6_examples.svg")


def main() -> None:
    """Draw every figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    contrast_figure(
        criteria=["cov_q01", "cov_q05", "cov_q95", "cov_q99"], limit=0.01, one_sided=False,
        title=["Coverage: two-constant rule minus table P",
               "Dot: estimate. Line: interval (Bonferroni level, 36 tests). Green band: limit.",
               "Difference in share of actuals at or below the quantile"],
        name="figure_1_coverage_difference.svg",
    )  # fmt: skip
    contrast_figure(
        criteria=["pinball_tails"], limit=0.05, one_sided=True,
        title=["Mean tail pinball loss: two-constant rule relative to table P",
               "Smaller is better; the green band is the 5% limit.",
               "Relative difference in pinball loss"],
        name="figure_2_pinball_difference.svg",
    )  # fmt: skip
    contrast_figure(
        criteria=["share_above_q99"], limit=0.002, one_sided=True,
        title=["Share of actuals above q99: two-constant rule minus table P",
               "Smaller is better; the green band is the 0.2 percentage point limit.",
               "Difference in share above q99"],
        name="figure_3_share_above_q99.svg",
    )  # fmt: skip
    absolute_figure()
    constants_figure()
    example_figure()


if __name__ == "__main__":
    main()
