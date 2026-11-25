# ruff: noqa: E501
"""Experiment 715: plots 7 and 8 of the result page, drawn from stored files, with no fit.

Usage (through `run.sh`, never `uv run`):
    run.sh e00715_result_plots.py contrasts-by-fold   # plot 8, from `score/headline.json` only
    run.sh e00715_result_plots.py contrasts-by-lead   # plot 7, from the stored forecasts of M*, F4, and F4-R

**Read-only.** The script reads `score/headline.json` (plot 8), and, for plot 7, the stored forecasts
through the scorer's own `read_pair` (the same function and the same rows as `e00715_score.py
headline`, one call per lead band). It writes only SVG files under `plots/` in this experiment's data
folder, and no fit, no GPU, and no write to any other experiment's folder happen. Plot 8 needs under
1 GB of memory. Plot 7 reads the same forecast files as the headline scorer, so give it the scorer's
12G cap and allow about 10 minutes.

**Anonymisation.** Both plots are aggregates over a group of Primaries. No per-site series is drawn,
no generator is shown, and no axis carries a calendar date: the folds are numbered 1 to 8 in time
order, and the lead bands are hours after the origin. After running, optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline FILE` and look at the render.

**Colours** are Brand Orange and Data Blue from `plotting.ocf_theme`. Check them with the bundled
`dataviz` skill's `validate_palette.py` before the figures are committed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import altair as alt
import e00715_arms as arms
import e00715_score as score
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR) / "plots"
WIDTH_PX: Final = 680
GROUP_NAMES: Final = {
    "six": "The six signed-MW Primaries",
    "eight": "The eight MVA-only Primaries",
}
CONTRASTS: Final = {"N1": "N1: F4 minus M*", "N2": "N2: F4 minus F4-R"}
COLOURS: Final = {"N1: F4 minus M*": ocf.DATA_BLUE, "N2: F4 minus F4-R": ocf.BRAND_ORANGE}
DIRECTION: Final = "points of capacity; below zero means F4 forecasts better"


def zero_rule() -> alt.Chart:
    """Return the labelled reference rule at zero."""
    return (
        alt.Chart(pl.DataFrame({"y": [0.0]}))
        .mark_rule(color="#555555", strokeDash=[4, 3])
        .encode(y="y:Q")
    )


def by_fold() -> Path:
    """Plot 8: N1 and N2 for each of the 8 folds, from the monthly values in `score/headline.json`."""
    headline = json.loads(
        (arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR) / "score" / "headline.json").read_text()
    )
    rows = []
    for key, label in CONTRASTS.items():
        for group in GROUP_NAMES:
            for fold, value in enumerate(headline[key][group]["monthly"], start=1):
                rows.append(
                    {
                        "contrast": label,
                        "group": GROUP_NAMES[group],
                        "fold": fold,
                        "difference": float(value),
                    }
                )
    frame = pl.DataFrame(rows)
    assert frame.height == 2 * 2 * 8, frame.height
    base = alt.Chart(frame).encode(
        x=alt.X("fold:O", title="Fold, 1 to 8 in time order", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("difference:Q", title=f"CRPS difference ({DIRECTION})"),
        color=alt.Color(
            "contrast:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    layer = alt.layer(
        zero_rule(),
        base.mark_line(aria=False),
        base.mark_point(filled=True, size=70, aria=False),
    ).properties(width=WIDTH_PX - 120, height=170)
    chart = layer.facet(
        row=alt.Row("group:N", title=None, sort=list(GROUP_NAMES.values()))
    ).properties(
        title=alt.TitleParams(
            "Neighbour rows made no consistent difference to CRPS in any of the 8 folds",
            subtitle=[
                "Dot: difference for one fold (mean over 2 seeds). Dashed line: no difference.",
                "Contrasts over leads of 24 h up to but not including 240 h. Smaller is better.",
            ],
        )
    )
    path = OUT / "plot_08_contrasts_by_fold.svg"
    OUT.mkdir(parents=True, exist_ok=True)
    chart.save(path)
    return path


def by_lead() -> Path:
    """Plot 7: N1 and N2 by lead band, with the 98.75% interval, from the same stored forecasts as the headline."""
    groups = score.group_ids()
    m_star = score.star()
    rows = []
    for key, reference in (("N1", m_star), ("N2", "F4-R")):
        for band_name, band in score.BANDS.items():
            result = score.read_pair(
                treatment="F4",
                reference=reference,
                months=arms.ALL_FOLDS,
                level=score.LEVEL,
                groups=groups,
                band=band,
            )
            for group in GROUP_NAMES:
                c = result[group]
                rows.append({
                    "contrast": CONTRASTS[key], "group": GROUP_NAMES[group], "band": band_name,
                    "point": float(c["point"]), "lower": float(c["lower"]), "upper": float(c["upper"]),
                })  # fmt: skip
    frame = pl.DataFrame(rows)
    assert frame.height == 2 * 3 * 2, frame.height
    band_order = list(score.BANDS)
    base = alt.Chart(frame).encode(
        y=alt.Y("band:O", sort=band_order, title="Lead band (hours after the forecast origin)"),
        color=alt.Color(
            "contrast:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=alt.Legend(orient="top", title=None),
        ),
        yOffset="contrast:N",
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#555555", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    layer = alt.layer(
        rule,
        base.mark_rule(aria=False).encode(
            x=alt.X("lower:Q", title=f"CRPS difference ({DIRECTION})"), x2="upper:Q"
        ),
        base.mark_point(filled=True, size=70, aria=False).encode(x="point:Q"),
    ).properties(width=WIDTH_PX - 200, height=120)
    chart = layer.facet(
        row=alt.Row("group:N", title=None, sort=list(GROUP_NAMES.values()))
    ).properties(
        title=alt.TitleParams(
            "No lead band shows a gain from neighbour rows",
            subtitle=[
                "Dot: difference over 8 folds and 2 seeds. Line: 98.75% interval from resampling whole months.",
                "Dashed line: no difference. Smaller is better.",
            ],
        )
    )
    path = OUT / "plot_07_contrasts_by_lead_band.svg"
    OUT.mkdir(parents=True, exist_ok=True)
    chart.save(path)
    return path


def main() -> int:
    """Draw one plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("plot", choices=["contrasts-by-fold", "contrasts-by-lead"])
    args = parser.parse_args()
    path = by_fold() if args.plot == "contrasts-by-fold" else by_lead()
    print("wrote", path, "; now run: npx svgo@4 --multipass --precision=1 --final-newline", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
