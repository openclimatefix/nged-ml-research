"""Draw the figures of experiment 371 from `work/results.json`."""

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf  # noqa: F401  (registers and enables the theme)
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00371-d8-monthly-xgboost-input-group-ablation"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
GAM_STACK_SCORE: Final[float] = 8.338
LABELS: Final[dict[str, str]] = {
    "control": "Monthly XGBoost as stored",
    "no_extra_weather": "Without 5 extra weather inputs",
    "no_lags": "Without 3 origin lags",
    "equal_weights": "Equal row weights",
    "ens_mean": "Ensemble-mean inputs",
    "gam_inputs_only": "GAM's inputs only",
}


def main() -> None:
    """Write the three figures."""
    results = json.loads((WORK / "results.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    order = list(LABELS.values())

    d8 = results["D8"]["arms"]
    table = pl.DataFrame(
        [
            {"arm": LABELS[n], **{k: d8[n]["score"][k] for k in ("mean", "low", "high")}}
            for n in LABELS
        ]
    )
    base = alt.Chart(table).encode(y=alt.Y("arm:N", sort=order, title=None))
    rule = (
        alt.Chart(pl.DataFrame({"x": [GAM_STACK_SCORE]}))
        .mark_rule(strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    label = (
        alt.Chart(pl.DataFrame({"x": [GAM_STACK_SCORE], "t": ["GAM stack, weekly (8.338)"]}))
        .mark_text(align="right", dx=-4, dy=-110, fontSize=11)
        .encode(x="x:Q", text="t:N")  # ty: ignore[unresolved-attribute]
    )
    chart = (
        base.mark_rule().encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "mean:Q",
                title="Quantile score at D8 (points; lower is better)",
                scale=alt.Scale(zero=False),
            )
        )
        + rule
        + label
    ).properties(
        width=420,
        height=260,
        title=alt.TitleParams(
            "Dropping the three origin lags takes monthly XGBoost from 7.70 to 8.25 at D8",
            subtitle="Points with 95% intervals (whole valid months resampled), 96,986 rows",
            anchor="start",
        ),
    )
    chart.save(DOCS_DIR / "figure_1_arms_at_d8.svg")

    rows = [
        {
            "site": site,
            "arm": LABELS[n],
            **{k: results[site]["arms"][n]["minus_control"][k] for k in ("mean", "low", "high")},
        }
        for site in ("D8", "D2", "D4")
        for n in LABELS
        if n != "control"
    ]
    diff = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("arm:N", sort=order[1:], title=None),
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="grey").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    layered = (
        diff.mark_rule().encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
        + diff.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("mean:Q", title="Score minus control (points; positive is worse)")
        )
        + zero
    ).properties(width=240, height=170)
    faceted = layered.facet(
        column=alt.Column("site:N", sort=["D8", "D2", "D4"], title=None)
    ).properties(
        title=alt.TitleParams(
            "Dropping the origin lags worsens the score at all three substations, most at D8",
            subtitle="D8 is the substation of interest; D2 and D4 are placebos",
            anchor="start",
        )
    )
    faceted.save(DOCS_DIR / "figure_2_arms_minus_control_by_site.svg")

    bands = []
    for name in ("no_lags", "gam_inputs_only", "equal_weights"):
        for band, b in results["D8"]["arms"][name]["bands"].items():
            d = b["minus_control"]
            bands.append(
                {
                    "arm": LABELS[name],
                    "band": band,
                    "mean": d["mean"],
                    "low": d["low"],
                    "high": d["high"],
                }
            )
    band_base = alt.Chart(pl.DataFrame(bands)).encode(
        y=alt.Y("band:N", sort=["0-24 h", "24-72 h", "72-120 h", "120-168 h"], title="Lead band"),
        color=alt.Color("arm:N", title=None),
        yOffset="arm:N",
    )
    band_chart = (
        band_base.mark_rule().encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
        + band_base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("mean:Q", title="Score minus control at D8 (points; positive is worse)")
        )
        + zero
    ).properties(
        width=420,
        height=220,
        title=alt.TitleParams(
            "Dropping the lags costs 0.49 to 0.61 points in every lead band at D8",
            subtitle="By lead band, paired with the control, 95% intervals",
            anchor="start",
        ),
    )
    band_chart.save(DOCS_DIR / "figure_3_d8_by_lead_band.svg")


if __name__ == "__main__":
    main()
