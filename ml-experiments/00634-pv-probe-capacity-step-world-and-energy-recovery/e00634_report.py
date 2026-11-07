# ruff: noqa: E501
"""Experiment 634: the figures of the write-up, from `work/step_recoveries.csv` and `work/energy_recoveries.csv`.

Usage: python e00634_report.py
"""

from pathlib import Path

import altair as alt
import e00634_common as k
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00634-pv-probe-capacity-step-world-and-energy-recovery"
)
W = 640
CUTOFF_LABEL = {
    "2025-07": "Jul 2025",
    "2025-10": "Oct 2025",
    "2026-03": "Mar 2026",
    "2026-06": "Jun 2026",
}
ORDER = [CUTOFF_LABEL[c] for c in k.CUTOFFS]


def per_primary(rec: pl.DataFrame, *, conditions: list[str], column: str) -> pl.DataFrame:
    """Return the median over seeds per cutoff, condition, and Primary, then its median and range over the six Primaries."""
    sel = rec.filter(pl.col("condition").is_in(conditions))
    med = sel.group_by("cutoff", "condition", "site").agg(
        v=pl.col(column).median(), share=pl.col("share").first()
    )
    return (
        med.group_by("cutoff", "condition")
        .agg(
            med=pl.col("v").median(),
            lo=pl.col("v").min(),
            hi=pl.col("v").max(),
            share=pl.col("share").median(),
        )
        .with_columns(cutoff_label=pl.col("cutoff").replace_strict(CUTOFF_LABEL))
        .sort("cutoff", "condition")
    )


def line_panel(
    d: pl.DataFrame,
    *,
    colours: dict[str, str],
    title: str,
    domain: list[float],
    y_title: str,
    rule: float | None,
    rule_label: str | None,
    share: bool,
) -> alt.Chart:
    """Return one panel: median dots joined by a line, a range bar per cutoff, an optional reference rule and share line."""
    colour = alt.Color(
        "label:N",
        scale=alt.Scale(domain=list(colours), range=list(colours.values())),
        legend=alt.Legend(
            title=None, orient="top", direction="horizontal", labelLimit=400, symbolType="circle"
        ),
    )
    x = alt.X("cutoff_label:N", sort=ORDER, title=None, axis=alt.Axis(labelAngle=0))
    base = alt.Chart(d).encode(x=x, color=colour)
    layers = [
        base.mark_rule().encode(
            y=alt.Y("lo:Q", scale=alt.Scale(domain=domain), title=y_title), y2="hi:Q"
        ),  # ty: ignore[unresolved-attribute]
        base.mark_line(point=alt.OverlayMarkDef(filled=True, size=70)).encode(y="med:Q"),  # ty: ignore[unresolved-attribute]
    ]
    if rule is not None:
        layers.append(
            alt.Chart(pl.DataFrame({"y": [rule]}))
            .mark_rule(color="#444444", strokeDash=[5, 3], strokeWidth=1.5)
            .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=domain)))
        )
    if share:
        sh = d.select("cutoff_label", "share").unique().sort("cutoff_label")
        layers.append(
            alt.Chart(sh)
            .mark_line(
                color=theme.BRAND_ORANGE,
                strokeDash=[2, 2],
                point=alt.OverlayMarkDef(color=theme.BRAND_ORANGE, shape="diamond", size=60),
            )
            .encode(x=x, y=alt.Y("share:Q", scale=alt.Scale(domain=domain)))  # ty: ignore[unresolved-attribute]
        )
    return alt.layer(*layers).properties(
        width=W, height=190, title=alt.TitleParams(title, anchor="start", fontSize=12)
    )


def fig_step(rec: pl.DataFrame) -> None:
    """Calibrated recovery of the full step by cutoff, one panel per step month."""
    panels = []
    for month, label in (("spring", "Step from April 2025"), ("summer", "Step from July 2025")):
        conds = [f"step_{month}_mw2", f"step_{month}_mw4"]
        d = per_primary(rec, conditions=conds, column="recovery_cal").with_columns(
            label=pl.col("condition")
            .str.extract(r"mw(\d)", 1)
            .map_elements(lambda m: f"{m} MW step", return_dtype=pl.String)
        )
        panels.append(
            line_panel(
                d,
                colours={"2 MW step": theme.DATA_SKY, "4 MW step": theme.DATA_BLUE},
                title=label,
                domain=[-0.1, 1.0],
                y_title="Share of the step read",
                rule=0.7,
                rule_label=None,
                share=True,
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Share of an injected solar-capacity step that the probe reads, by training cutoff (1 is the whole step).",
                "Dot: median over the six MW Primaries (calibrated, 3 seeds). Line: range over Primaries.",
                "Orange diamonds: the share of donor rows dated after the step, the most a calendar-blind network could read.",
                "Dashed line: the 0.7 bar.",
            ],
            anchor="start",
            fontSize=11,
        )
    )
    chart.save(str(DOCS / "step_recovery_by_cutoff.svg"))


def fig_placebo(rec: pl.DataFrame) -> None:
    """Raw capacity difference of the real-farm step and the shuffled-shape placebo, per cutoff."""
    panels = []
    for month, label in (
        ("spring", "Step from April 2025, 4 MW"),
        ("summer", "Step from July 2025, 4 MW"),
    ):
        conds = [f"step_{month}_mw4", f"shufstep_{month}_mw4"]
        d = per_primary(rec, conditions=conds, column="delta_mw").with_columns(
            label=pl.col("condition").map_elements(
                lambda c: (
                    "Real farm shape" if c.startswith("step") else "Shuffled shape (no sky link)"
                ),
                return_dtype=pl.String,
            )
        )
        panels.append(
            line_panel(
                d,
                colours={
                    "Real farm shape": theme.DATA_GREEN,
                    "Shuffled shape (no sky link)": theme.DATA_PURPLE,
                },
                title=label,
                domain=[-0.5, 3.0],
                y_title="Capacity read (MW)",
                rule=0.3,
                rule_label=None,
                share=False,
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Capacity read by the probe for a 4 MW step of real-farm shape and of a shape with no sky link, by cutoff.",
                "Dot: median over the six MW Primaries (raw, 3 seeds, minus the network trained without the step). Line: range.",
                "Purple is a placebo: any capacity it reads is invented. Dashed line: the 0.3 MW bar for a placebo.",
            ],
            anchor="start",
            fontSize=11,
        )
    )
    chart.save(str(DOCS / "placebo_against_real_step.svg"))


def fig_energy(rec: pl.DataFrame) -> None:
    """Recovery of the real farm per Primary in injected megawatts and in injected energy (plain network)."""
    plain = rec.filter(pl.col("arm") == "plain")
    cols = {
        "recovery_mw": "Per injected MW",
        "recovery_energy_donors": "Per injected energy (donor rows)",
        "recovery_energy_rows": "Per injected energy (all rows)",
    }
    long = pl.concat(
        [
            plain.group_by("site").agg(v=pl.col(c).median()).with_columns(label=pl.lit(lab))
            for c, lab in cols.items()
        ]
    )
    colours = {
        "Per injected MW": theme.DATA_BLUE,
        "Per injected energy (donor rows)": theme.DATA_GREEN,
        "Per injected energy (all rows)": theme.DATA_SKY,
    }
    base = alt.Chart(long).encode(
        y=alt.Y("site:N", sort=list(k.SIX), title=None),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(
                title=None, orient="top", direction="horizontal", labelLimit=400, columns=1
            ),
        ),
    )
    dots = base.mark_point(filled=True, size=70).encode(
        x=alt.X(
            "v:Q",
            scale=alt.Scale(domain=[0.5, 1.1]),
            title="Share of the injected real farm that the probe reads",
        )
    )  # ty: ignore[unresolved-attribute]
    ref = (
        alt.Chart(pl.DataFrame({"x": [1.0]}))
        .mark_rule(color="#444444", strokeDash=[4, 3])
        .encode(x=alt.X("x:Q", scale=alt.Scale(domain=[0.5, 1.1])))
    )
    chart = alt.layer(ref, dots).properties(
        width=W,
        height=190,
        title=alt.TitleParams(
            [
                "Share of an injected real solar farm that the probe reads, per MW Primary, in megawatts and in energy.",
                "Dot: median over cutoffs 2026-03 and 2026-06, 3 seeds, and 4 farms (experiments 621, 623, and 624, plain network).",
                "Dashed line: the whole farm (1.0).",
            ],
            anchor="start",
            fontSize=11,
        ),
    )
    chart.save(str(DOCS / "energy_recovery_by_primary.svg"))


def main() -> None:
    """Write the three figures."""
    rec = pl.read_csv(k.WORK / "step_recoveries.csv")
    fig_step(rec)
    fig_placebo(rec)
    fig_energy(pl.read_csv(k.WORK / "energy_recoveries.csv"))


if __name__ == "__main__":
    main()
