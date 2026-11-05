# ruff: noqa: E501
"""Experiment 574: the figures of the page.

1. `contrasts_vs_p0.svg`: each arm's CRPS minus P0's in three lead bands (dot: estimate over 2 seeds;
   line: 5-block month bootstrap interval, Bonferroni for the counted contrasts of 42 tests).
2. `crps_by_band.svg`: absolute CRPS of the references and the candidates in three lead bands.
3. `leak.svg`: injected 7-day shift (mean of five folds, range as a line) and the real-event absorbed
   share, per arm.
"""

from pathlib import Path

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms"
DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
SCORES = ML_EXPERIMENTS_DATA_DIR / NAME / "work/scores"
BANDS = ["0-24h", "24-120h", "120-240h"]
GROUPS = {"eight": "The eight MVA Primaries", "six": "The six MW Primaries"}
ARMS = [
    ("P1", "P1: spread columns in the mixer", theme.DATA_SKY),
    ("P3E0", "P3-E0: encoders unfrozen (E0)", theme.DATA_BLUE),
    ("P2", "P2: thermal-lag and solar-gain columns (B1)", theme.DATA_GREEN),
    ("P4", "P4: three-step schedule", theme.DATA_PURPLE),
    ("P4E", "P4-E: three-step schedule with 3-day embargo", theme.BRAND_ORANGE),
    ("P5", "P5: final candidate (P4-E plus B1)", theme.BLACK_1),
]


def zero_rule() -> alt.Chart:
    """Return the dashed line at zero."""
    return (
        alt.Chart(pl.DataFrame({"z": [0.0]}))
        .mark_rule(color="grey", strokeDash=[4, 3])
        .encode(x="z:Q")
    )  # ty: ignore[unresolved-attribute]


def contrasts() -> None:
    """Write figure 1."""
    t = pl.read_csv(SCORES / "contrasts.csv").filter(
        (pl.col("slice") == "all rows") & (pl.col("b") == "P0") & pl.col("set").is_in(list(GROUPS))
    )
    rows = [
        {"arm": label, "band": r["band"], "group": GROUPS[r["set"]], "d": r["difference"], "lo": r["low_bonferroni_month"] if r["counted"] else r["low_month"], "hi": r["high_bonferroni_month"] if r["counted"] else r["high_month"]}
        for arm, label, _ in ARMS
        for r in t.filter(pl.col("a") == arm).iter_rows(named=True)
        if r["band"] in BANDS
    ]  # fmt: skip
    order, colours = [a[1] for a in ARMS], [a[2] for a in ARMS]
    base = alt.Chart().encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=340)),
        color=alt.Color("arm:N", scale=alt.Scale(domain=order, range=colours), legend=None),
    )
    layers = alt.layer(
        zero_rule(),
        base.mark_rule().encode(x=alt.X("lo:Q", title=None), x2="hi:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=50).encode(x="d:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(width=170, height=130)
    chart = layers.facet(
        data=pl.DataFrame(rows),
        row=alt.Row("group:N", title=None),
        column=alt.Column("band:O", sort=BANDS, title="Lead band"),
    ).properties(
        title=alt.TitleParams(
            [
                "Only the embargoed three-step schedule lowers the error; the spread columns raise it",
                "Horizontal axis: CRPS minus P0's, in points of capacity (negative is better). Dot: estimate (2 seeds). Line: interval from resampling whole months (5 blocks); Bonferroni for 42 tests on counted contrasts (P1, P3-E0, P2, P4, P5), uncorrected 95% otherwise (P4-E).",
                "Dashed line: same as P0 (experiment 572's v3+D5).",
            ],
            anchor="start",
            fontSize=11,
        )
    )
    chart.save(str(DOCS / "contrasts_vs_p0.svg"))


def by_band() -> None:
    """Write figure 2."""
    names = {
        "M0": "M0 (no lags, no offset)",
        "E2": "E2 (offset, no lags)",
        "L1d": "L-1d (daily refit)",
        "Apool24": "A-pool-24 (daily refit)",
        "P0": "P0 (v3+D5)",
        "P5": "P5 (final candidate)",
    }
    colours = [
        theme.GREY_2,
        theme.DATA_SKY,
        theme.DATA_BLUE_LIGHT,
        theme.DATA_BLUE,
        theme.DATA_PURPLE,
        theme.BRAND_ORANGE,
    ]
    t = pl.read_csv(SCORES / "absolute.csv").filter(
        (pl.col("slice") == "all rows") & pl.col("arm").is_in(list(names)) & pl.col("set").is_in(list(GROUPS)) & pl.col("band").is_in(BANDS)
    ).with_columns(arm=pl.col("arm").replace(names), group=pl.col("set").replace(GROUPS))  # fmt: skip
    chart = (
        alt.Chart(t)
        .mark_bar()
        .encode(
            x=alt.X("band:O", sort=BANDS, title="Lead band", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("arm:N", sort=list(names.values())),
            y=alt.Y("mean_single_seed_crps:Q", title="CRPS (points of capacity, lower is better)"),
            color=alt.Color("arm:N", sort=list(names.values()), scale=alt.Scale(domain=list(names.values()), range=colours), legend=alt.Legend(orient="top", columns=2, title=None)),
        )  # ty: ignore[unresolved-attribute]
        .properties(width=520, height=150)
        .facet(row=alt.Row("group:N", title=None))
        .properties(title=alt.TitleParams(["The final candidate beats the daily-refit references at every lead band", "Mean single-seed CRPS over 5 folds and 2 seeds; M0, E2, and P0 are stored fits of experiments 565 and 572 (coarse refits), the L-1d and A-pool-24 references refit daily."], anchor="start", fontSize=11))
    )  # fmt: skip
    chart.save(str(DOCS / "crps_by_band.svg"))


def leak() -> None:
    """Write figure 3 (the injected shift; the real-event share joins when the harness has run)."""
    t = pl.read_csv(SCORES / "leak.csv").filter(
        pl.col("arm").is_in(
            ["M0", "E2", "P0", "P1", "P3E0", "P2", "P2plac", "P4", "P4E", "P5", "P7"]
        )
    )
    order = t["arm"].to_list()
    base = alt.Chart(t).encode(y=alt.Y("arm:N", sort=order, title=None))
    chart = alt.layer(
        base.mark_rule(color=theme.GREY_2).encode(x=alt.X("lo:Q", title="Injected 7-day shift followed by the forecast (points of capacity; lower is better)"), x2="hi:Q"),
        base.mark_point(filled=True, size=60, color=theme.BRAND_ORANGE).encode(x="leak:Q"),
        alt.Chart(pl.DataFrame({"z": [6.0]})).mark_rule(color="grey", strokeDash=[4, 3]).encode(x="z:Q"),
    ).properties(width=520, height=200, title=alt.TitleParams(["The embargoed three-step schedule roughly halves the shift the forecast follows", "Dot: mean of five folds (seed 0). Line: range over folds. Dashed line: the limit of 6.0 points, a flag and not a gate."], anchor="start", fontSize=11))  # fmt: skip
    chart.save(str(DOCS / "leak.svg"))


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    contrasts()
    by_band()
    leak()
