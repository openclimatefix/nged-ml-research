"""Experiment 518: the two figures of the write-up, from the fit log and the score file.

```bash
ml-experiments/00518-*/run.sh ml-experiments/00518-*/e00518_report.py
```
"""

import json
from pathlib import Path

import altair as alt
import e00518_fit as fit  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl

DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / Path(__file__).parent.name


def figure_loss() -> None:
    """Save figure 1: the validation step loss by epoch, with the best epoch marked."""
    log = pl.read_csv(fit.EXPERIMENT_DIR / "fit_log_stage0_seed0.csv")
    summary = json.loads((fit.EXPERIMENT_DIR / "fit_summary_stage0_seed0.json").read_text())
    line = (
        alt.Chart(log)
        .mark_line(point=True, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("val_l_u:Q", title="Validation step loss (lower is better)"),
        )
    )
    best = (
        alt.Chart(pl.DataFrame({"epoch": [summary["best_epoch"]]}))
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 4])
        .encode(x="epoch:Q")  # ty: ignore[unresolved-attribute]
    )
    (line + best).properties(width=500, height=250).save(DOCS / "figure_1_validation_loss.svg")


def figure_found() -> None:
    """Save figure 2: the share of injected steps found within 2, 6, and 24 hours, with ceiling."""
    score = json.loads((fit.EXPERIMENT_DIR / "t1_score.json").read_text())["all"]
    rows = []
    for hours in (2, 6, 24):
        rows.append({"hours": f"within {hours} h", "who": "Encoder",
                     "share": float(score[f"found_{hours}h"].split()[0])})  # fmt: skip
        rows.append({"hours": f"within {hours} h", "who": "Ceiling (perfect predictor)",
                     "share": score[f"ceiling_found_{hours}h"]})  # fmt: skip
    chart = (
        alt.Chart(pl.DataFrame(rows))
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("hours:N", title="Onset error", sort=None),
            xOffset="who:N",
            y=alt.Y(
                "share:Q", title="Share of injected steps found", scale=alt.Scale(domain=[0, 1])
            ),
            color=alt.Color(
                "who:N",
                title=None,
                scale=alt.Scale(
                    domain=["Encoder", "Ceiling (perfect predictor)"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
            ),
        )
        .properties(width=400, height=250)
    )
    chart.save(DOCS / "figure_2_found_against_ceiling.svg")


if __name__ == "__main__":
    figure_loss()
    figure_found()
