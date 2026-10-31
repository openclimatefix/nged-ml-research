# ruff: noqa: E501
"""Draw the figures of experiment 424: gain by Primary, CRPS by arm, and example forecasts."""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00424_score as score
import numpy as np
import polars as pl

DOCS: Final[Path] = Path(
    "docs/ml-experiments/00424-neural-mva-export-gated-signed-ens-serving-test"
)
LAB_SCORES: Final[Path] = (
    score.OUT_DIR.parent.parent
    / "00420-neural-mva-export-gated-signed-final"
    / "work"
    / "scores_full.json"
)
WIDTH: Final[int] = 680
BLUE, ORANGE, SKY, GREY = "#306BFF", "#FF4901", "#10C5F7", "#7F7F7F"
TITLE_KW: Final[dict] = {"anchor": "start", "fontSize": 14, "subtitleFontSize": 11}


def gain_chart() -> alt.Chart:
    """Draw the gain over P0 on reverse-flow rows by hidden Primary, ENS inputs and laboratory inputs."""
    bars = json.loads((score.OUT_DIR / "scores_full.json").read_text())["bars"]
    lab = json.loads(LAB_SCORES.read_text())["bars"]["A"]
    rows = []
    for label, entry in (
        ("ENS inputs (this experiment)", bars),
        ("Recorded CAMS and ERA5 (experiment 420)", None),
    ):
        by = entry["s1_vs_p0_by_primary"] if entry else lab["b1_vs_p0_by_primary"]
        pooled = entry["s1_vs_p0_pooled"] if entry else lab["b1_vs_p0_pooled"]
        for site, v in {**by, "Pooled": pooled}.items():
            rows.append(
                {"inputs": label, "site": site, "gain": v["mean"], "lo": v["lo"], "hi": v["hi"]}
            )
    order = ["Pooled", "D5", "D9", "D10", "D11", "D14", "D16"]
    domain = ["ENS inputs (this experiment)", "Recorded CAMS and ERA5 (experiment 420)"]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("site:N", title=None, sort=order),
        yOffset="inputs:N",
        color=alt.Color(
            "inputs:N",
            scale=alt.Scale(domain=domain, range=[BLUE, SKY]),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    dots = base.mark_circle(size=70, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("gain:Q", title="CRPS gain of A over P0, points of capacity (larger is better)")
    )
    rules = base.mark_rule(aria=False).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    return (rules + dots).properties(
        width=WIDTH - 90,
        height=280,
        title=alt.TitleParams(
            "The gated network keeps most of its gain when it sees ensemble weather",
            subtitle=[
                "Signed CRPS on rows with reverse flow, level 2.0, 7 test months, 3 rotations.",
                "Dot: estimate. Line: 95% interval from resampling Primary-months.",
            ],
            **TITLE_KW,
        ),
    )


def crps_chart() -> alt.Chart:
    """Draw the mean CRPS on reverse-flow rows for each arm and input source."""
    d = json.loads((score.OUT_DIR / "scores_full.json").read_text())
    ro = d["bars"]["reported_only"]
    rows = [
        {"arm": "P0 (ENS)", "crps": d["bars"]["crps_reverse_rows"]["P0"], "src": "ENS"},
        {"arm": "A (ENS)", "crps": d["bars"]["crps_reverse_rows"]["A"], "src": "ENS"},
        {
            "arm": "A-applied (laboratory network, ENS inputs)",
            "crps": ro["crps_reverse_rows_applied"],
            "src": "Applied",
        },
        {"arm": "P0 (recorded)", "crps": ro["crps_reverse_rows_lab_P0"], "src": "Recorded"},
        {"arm": "A (recorded)", "crps": ro["crps_reverse_rows_lab_A"], "src": "Recorded"},
        {
            "arm": "ref (recorded, no gate)",
            "crps": ro["crps_reverse_rows_lab_ref"],
            "src": "Recorded",
        },
    ]
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_circle(size=80, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", title=None, sort=[r["arm"] for r in rows]),
            x=alt.X(
                "crps:Q",
                title="Mean signed CRPS on rows with reverse flow (points; smaller is better)",
            ),
            color=alt.Color(
                "src:N",
                scale=alt.Scale(domain=["ENS", "Applied", "Recorded"], range=[BLUE, ORANGE, SKY]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH - 200,
            height=190,
            title=alt.TitleParams(
                "Absolute scores on reverse-flow rows",
                subtitle=[
                    "Six hidden Primaries, 7 test months, 3 rotations. Point estimates, no interval."
                ],
                **TITLE_KW,
            ),
        )
    )


def example_chart() -> alt.VConcatChart:
    """Draw one forecast run at D5 (level 2.0): the day ahead and the full 10 days, for P0 and A."""
    arms = {"P0": ORANGE, "A": BLUE}
    frames = {
        arm: pl.read_parquet(score.FITS / arm / "L2_Q2_r1_2025-08_s0.parquet").filter(
            pl.col("site") == "D5"
        )
        for arm in arms
    }
    a = frames["A"]
    best = (
        a.filter(pl.col("truth_points") < 0)
        .group_by("origin")
        .len()
        .sort("len", descending=True)["origin"][0]
    )
    panels = []
    for name, limit in (("Day ahead (first 48 hours)", 48.0), ("Full 10-day horizon", 240.0)):
        layers = []
        for arm, colour in arms.items():
            f = (
                frames[arm]
                .filter((pl.col("origin") == best) & (pl.col("lead") <= limit))
                .sort("valid")
                .with_columns(pl.col("lead").alias("hours"))
            )
            band = (
                alt.Chart(f.select("hours", "q05", "q95"))
                .mark_area(opacity=0.25, color=colour, clip=True)
                .encode(x="hours:Q", y="q05:Q", y2="q95:Q")  # ty: ignore[unresolved-attribute]
            )
            median = (
                alt.Chart(f.select("hours", "q50"))
                .mark_line(color=colour, clip=True, aria=False)
                .encode(x="hours:Q", y="q50:Q")  # ty: ignore[unresolved-attribute]
            )
            layers += [band, median]
        truth = (
            alt.Chart(
                frames["A"]
                .filter((pl.col("origin") == best) & (pl.col("lead") <= limit))
                .select(pl.col("lead").alias("hours"), "truth_points")
            )
            .mark_line(color="black", clip=True, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "hours:Q",
                    title="Hours after the forecast origin",
                    axis=alt.Axis(values=list(np.arange(0, limit + 1, 12 if limit <= 48 else 48))),
                ),
                y=alt.Y("truth_points:Q", title="Signed power (points of capacity)"),
            )
        )
        panels.append(
            alt.layer(*layers, truth).properties(width=WIDTH - 90, height=140, title=name)
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "With ensemble weather the gated network still follows the reverse flow",
            subtitle=[
                "D5 with a farm injected (level 2.0), one forecast run in August 2025, ENS inputs. Black: signed truth.",
                "Orange: P0 median and 5% to 95% band. Blue: A median and 5% to 95% band.",
            ],
            **TITLE_KW,
        )
    )


def main() -> None:
    """Write the charts as SVG."""
    for name, chart in (
        ("gain_by_primary", gain_chart()),
        ("crps_by_arm", crps_chart()),
        ("example_forecast", example_chart()),
    ):
        chart.save(DOCS / f"{name}.svg")


if __name__ == "__main__":
    main()
