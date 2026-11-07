# ruff: noqa: E501
"""Experiment 635: the two figures of the write-up, from `work/recoveries.csv`."""

from pathlib import Path

import altair as alt
import e00635_common as k5
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00635-pv-probe-metered-neighbour-pv-as-sky-input"
)
ARM = {"a": "a: CAMS only", "b": "b: CAMS + metered farm", "c": "c: metered farm only"}
COLOURS = [theme.DATA_BLUE, theme.BRAND_ORANGE, theme.DATA_GREEN]
ORDER = list(ARM.values())


def colour() -> alt.Color:
    """Return the shared colour encoding (the key sits above the plot)."""
    return alt.Color(
        "arm:N",
        sort=ORDER,
        scale=alt.Scale(domain=ORDER, range=COLOURS),
        legend=alt.Legend(title=None, orient="top"),
    )


def main() -> None:
    """Write both figures."""
    rec = pl.read_csv(k5.WORK / "recoveries.csv").with_columns(arm=pl.col("arm").replace(ARM))
    real = (
        rec.filter((pl.col("world") == "farm") & (pl.col("mw") == 4.0))
        .group_by(["arm", "site"])
        .agg(pl.col("recovery").median())
    )
    pts = alt.Chart(real).mark_circle(size=90, opacity=0.9).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("recovery:Q", title="Probed capacity per injected AC MW (median over 3 seeds and 2 farm rotations)", scale=alt.Scale(domain=[0.5, 0.85])),
        y=alt.Y("site:N", title="MW Primary", sort=["D5", "D9", "D10", "D11", "D14", "D16"]),
        color=colour(), yOffset=alt.YOffset("arm:N", sort=ORDER),
    )  # fmt: skip
    pts = pts.properties(
        width=600,
        height=220,
        title=alt.TitleParams(
            [
                "A 4 MW real solar farm injected into net demand: the share the probe reads back.",
                "Larger is better; 1.0 is the injected capacity.",
            ],
            anchor="start",
            fontSize=12,
        ),
    )
    pts.save(str(DOCS / "real_farm_recovery.svg"))
    pl_ = rec.filter(
        pl.col("world").is_in(["profile", "shuffled", "phys"]) & (pl.col("mw") == 4.0)
    ).with_columns(
        world=pl.when(pl.col("mshuf"))
        .then(pl.lit("Profile placebo, metered column shuffled"))
        .otherwise(
            pl.col("world").replace(
                {
                    "profile": "Profile-only placebo",
                    "shuffled": "Shuffled-sky placebo",
                    "phys": "Physics-curve world",
                }
            )
        )
    )
    order = [
        "Physics-curve world",
        "Profile-only placebo",
        "Shuffled-sky placebo",
        "Profile placebo, metered column shuffled",
    ]
    box = alt.Chart(pl_.filter(pl.col("arm").is_in(ORDER))).mark_boxplot(extent="min-max", size=10).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("recovery:Q", title="Probed capacity per injected AC MW (every Primary, rotation, and seed)"),
        y=alt.Y("world:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)), color=colour(), yOffset=alt.YOffset("arm:N", sort=ORDER),
    )  # fmt: skip
    box = box.properties(
        width=420,
        height=260,
        title=alt.TitleParams(
            [
                "Placebos and the physics-curve world. Placebos should read near 0;",
                "the physics-curve world should read near 1.",
            ],
            anchor="start",
            fontSize=12,
        ),
    )
    box.save(str(DOCS / "placebos_and_physics_world.svg"))


if __name__ == "__main__":
    main()
