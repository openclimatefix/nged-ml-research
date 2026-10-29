# ruff: noqa: E501
"""Draw the figures of experiment 379's write-up.

The numbers are those printed by `e00379_contrasts.py` (`work/contrasts_components.log`): pooled 99%
intervals with 20,000 resamples, per-Primary 95% intervals with 2,000 resamples, typed in so that
the charts need no re-score. Figure 3 reads the mean CRPS by lead band from the same log.

    uv run --no-sync python ml-experiments/00379-*/e00379_charts.py
"""

import re
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00379-bg2-components-all-16-primaries-to-14-days"
)
LOG: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00379-bg2-components-all-16-primaries-to-14-days"
    / "work"
    / "contrasts_components.log"
)
WIDTH: Final[int] = 680
WRAP: Final[int] = 110
GROUP_COLOURS: Final[dict[str, str]] = {
    "signed-MW six": ocf.DATA_BLUE,
    "MVA-only eight": ocf.BRAND_ORANGE,
    "short history (D15, D18)": ocf.DATA_PURPLE,
}
COMPONENTS: Final[list[tuple[str, float, float, float, float, float, float]]] = [
    # name, signed d, lo, hi, MVA d, lo, hi (99%, days 1 to 10; BG2 minus BG2 without the part)
    ("Stack member (ST)", -0.061, -0.094, -0.030, -0.010, -0.039, 0.025),
    ("Weekly XGBoost member (W)", -0.037, -0.067, -0.012, -0.031, -0.065, -0.003),
    ("Analogue member (AN)", -0.031, -0.052, -0.011, -0.023, -0.035, -0.013),
    ("Monthly XGBoost member (M)", -0.001, -0.004, 0.000, 0.000, 0.000, 0.002),
    ("Pooled-by-type member (PT)", 0.005, 0.000, 0.019, 0.002, 0.000, 0.007),
    ("Solar offset", -0.045, -0.081, -0.004, 0.000, 0.000, 0.000),
    ("Spread factor 1.15", -0.023, -0.053, 0.004, 0.008, -0.014, 0.028),
]
WEEKLY_ALONE: Final[list[tuple[str, str, float, float, float]]] = [
    ("D5", "signed-MW six", -0.074, -0.272, 0.040),
    ("D9", "signed-MW six", -0.011, -0.056, 0.024),
    ("D10", "signed-MW six", 0.081, 0.007, 0.177),
    ("D11", "signed-MW six", 0.010, -0.026, 0.055),
    ("D14", "signed-MW six", 0.131, 0.020, 0.273),
    ("D16", "signed-MW six", 0.128, 0.029, 0.292),
    ("D2", "MVA-only eight", 0.038, 0.005, 0.079),
    ("D3", "MVA-only eight", 0.014, 0.001, 0.026),
    ("D4", "MVA-only eight", -0.030, -0.108, 0.022),
    ("D8", "MVA-only eight", -0.026, -0.132, 0.062),
    ("D12", "MVA-only eight", 0.016, -0.037, 0.083),
    ("D19", "MVA-only eight", 0.025, -0.022, 0.074),
    ("D21", "MVA-only eight", 0.017, -0.004, 0.039),
    ("D22", "MVA-only eight", 0.095, 0.023, 0.178),
    ("D15", "short history (D15, D18)", 0.617, 0.219, 1.049),
    ("D18", "short history (D15, D18)", 0.075, -0.010, 0.174),
]
BAND_ORDER: Final[list[str]] = [
    "lead_day_1",
    "lead_days_5_7",
    "lead_days_8_10",
    "lead_days_11_14",
]
BAND_LABELS: Final[dict[str, str]] = {
    "lead_day_1": "Day 1",
    "lead_days_5_7": "Days 5 to 7",
    "lead_days_8_10": "Days 8 to 10",
    "lead_days_11_14": "Days 11 to 14",
}
LINE_ARMS: Final[dict[str, str]] = {
    "bg2": ocf.BRAND_ORANGE,
    "only_W": ocf.DATA_BLUE,
    "only_ST": ocf.DATA_GREEN,
    "manual_heuristic": ocf.DATA_PURPLE,
    "climatology": "#808080",
}
ARM_LABELS: Final[dict[str, str]] = {
    "bg2": "BG2",
    "only_W": "Weekly XGBoost alone",
    "only_ST": "Stack member alone",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}


def titled(*, title: str, subtitle: str) -> alt.TitleParams:
    """Return a title and subtitle wrapped to the plot width."""
    return alt.TitleParams(
        text=textwrap.wrap(title, 85),
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
    )


def zero_rule(*, domain_label: str) -> alt.Chart:
    """Return a dashed rule at zero."""
    return (
        alt.Chart(pl.DataFrame({"x": [0.0], "label": [domain_label]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )


def figure_components() -> None:
    """Write figure 1: what dropping each BG2 component does to the CRPS."""
    rows = []
    for name, sd, sl, sh, md, ml, mh in COMPONENTS:
        rows.append((name, "signed-MW six", sd, sl, sh))
        rows.append((name, "MVA-only eight", md, ml, mh))
    frame = pl.DataFrame(
        rows, schema=["component", "group", "difference", "lower", "upper"], orient="row"
    )
    order = [c[0] for c in COMPONENTS]
    base = alt.Chart(frame).encode(
        y=alt.Y("component:N", sort=order, title=None),
        yOffset="group:N",
        color=alt.Color(
            "group:N",
            scale=alt.Scale(domain=list(GROUP_COLOURS)[:2], range=list(GROUP_COLOURS.values())[:2]),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    bars = base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="BG2 minus BG2 without the component (points of capacity; negative: it helps)",
        ),
        x2="upper:Q",
    )
    dots = base.mark_point(filled=True, size=50).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    chart = (zero_rule(domain_label="same") + bars + dots).properties(
        width=WIDTH - 220,
        height=200,
        title=titled(
            title="Each part of BG2 adds under 0.07 points, and the monthly and pooled members add nothing",
            subtitle=(
                "Quantile-score CRPS, days 1 to 10, all 16 Primaries scored on the shared half-hours "
                "of 2025-07 to 2026-06. Dot: estimate. Line: 99% interval from resampling whole "
                "months. Dashed line: BG2 is unchanged by dropping the part. The solar offset exists "
                "at 5 of the signed-MW six only. MVA-only numbers assume reactive power is zero."
            ),
        ),
    )
    chart.save(DOCS_DIR / "figure_1_components.svg")


def figure_weekly_alone() -> None:
    """Write figure 2: the weekly XGBoost member alone minus BG2, per Primary."""
    frame = pl.DataFrame(
        WEEKLY_ALONE, schema=["row", "group", "difference", "lower", "upper"], orient="row"
    )
    order = [r[0] for r in WEEKLY_ALONE]
    base = alt.Chart(frame).encode(
        y=alt.Y("row:N", sort=order, title=None),
        color=alt.Color(
            "group:N",
            scale=alt.Scale(domain=list(GROUP_COLOURS), range=list(GROUP_COLOURS.values())),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    bars = base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="Weekly XGBoost alone minus BG2 (points of capacity; negative: alone is better)",
        ),
        x2="upper:Q",
    )
    dots = base.mark_point(filled=True, size=50).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    chart = (zero_rule(domain_label="same") + bars + dots).properties(
        width=WIDTH - 200,
        height=22 * len(WEEKLY_ALONE),
        title=titled(
            title="A plain weekly XGBoost model is within 0.14 points of BG2 at 15 of 16 Primaries",
            subtitle=(
                "Quantile-score CRPS, days 1 to 10, shared half-hours of 2025-07 to 2026-06. Dot: "
                "estimate. Line: 95% interval per Primary from resampling whole months. Dashed "
                "line: same as BG2. D15 and D18 are report only. MVA-only numbers assume reactive power is zero."
            ),
        ),
    )
    chart.save(DOCS_DIR / "figure_2_weekly_alone.svg")


def band_means() -> pl.DataFrame:
    """Parse the mean CRPS of each arm, group and lead band from the contrasts log."""
    out = []
    for line in LOG.read_text().splitlines():
        match = re.match(r"^(lead_\w+)\s+(signed-MW|MVA-only with D8)\s+n=\s*[\d,]+\s+(.*)$", line)
        if not match or match.group(1) not in BAND_ORDER:
            continue
        for arm, value in re.findall(r"(\w+)=([\d.]+)", match.group(3)):
            if arm in LINE_ARMS:
                out.append((match.group(1), match.group(2), arm, float(value)))
    return pl.DataFrame(out, schema=["band", "group", "arm", "crps"], orient="row")


def figure_by_lead() -> None:
    """Write figure 3: the CRPS of BG2, its two members, and the references by lead band."""
    frame = band_means().with_columns(
        band_label=pl.col("band").replace(BAND_LABELS),
        arm_label=pl.col("arm").replace(ARM_LABELS),
        group_label=pl.col("group").replace(
            {
                "signed-MW": "Signed-MW six",
                "MVA-only with D8": "MVA-only eight (with D8; reactive power assumed zero)",
            }
        ),
    )
    chart = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "band_label:N",
                sort=[BAND_LABELS[b] for b in BAND_ORDER],
                title=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "crps:Q",
                title="CRPS (points; smaller is better)",
                scale=alt.Scale(domain=[2, 5.2]),
            ),
            color=alt.Color(
                "arm_label:N",
                scale=alt.Scale(domain=list(ARM_LABELS.values()), range=list(LINE_ARMS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=WIDTH - 110, height=130)
        .facet(
            row=alt.Row(
                "group_label:N",
                title=None,
                header=alt.Header(labelOrient="top", labelAngle=0, labelAlign="left"),
            ),
            spacing=30,
        )
        .properties(
            title=titled(
                title="Skill falls with lead, and BG2 stays ahead of both references to day 14",
                subtitle=(
                    "Mean quantile-score CRPS by lead band, shared half-hours of 2025-07 to "
                    "2026-06. The MVA-only panel assumes reactive power is zero."
                ),
            )
        )
    )
    chart.save(DOCS_DIR / "figure_3_by_lead.svg")


if __name__ == "__main__":
    figure_components()
    figure_weekly_alone()
    figure_by_lead()
