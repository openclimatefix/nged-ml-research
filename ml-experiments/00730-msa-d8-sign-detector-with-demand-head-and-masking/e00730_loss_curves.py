"""Experiment 730: the validation-score curves of seed 0, one panel per arm.

Run this script after the fits, with `DATA_PATH_INTERNAL` set. It reads each seed-0 fit's stored
`history` and `best_epoch` from the experiment's `fits` folder and writes one SVG to its `figures`
folder. It is not a stage of `e00730_cli.py`, and it writes nothing else.
"""

import json
import os
from pathlib import Path
from typing import Final

import e00730_paths

# isort: split
import altair as alt
import e00729_figures as figures729
import polars as pl

SEED: Final = 0
"""The seed whose fits are drawn."""
PANELS: Final = (
    ("S-A", "S-A_unet", "Brier"),
    ("S", "S_unet", "Brier"),
    ("SD-0", "SD-0_unet", "Brier"),
    ("SD-K", "SD-K_unet", "Brier"),
    ("SD-W", "SD-W_unet", "Brier"),
    ("SD-R (is SD*)", "SD-R_unet", "Brier"),
    ("SDe-PT", "SDe-PT_unet", "Brier"),
    ("SDe-FR", "SDe-FR_unet", "Brier"),
    ("SD*-C", "SDstar-C_unet", "Brier"),
    ("SD*-λ", "SDstar-lambda_unet", "Brier"),
    ("A", "A_bilstm", "signed CRPS"),
    ("D", "D_bilstm", "signed CRPS"),
    ("D-sign", "D-sign_bilstm", "signed CRPS"),
)  # fmt: skip
"""Each panel's title, its fit's file stem, and the score that fit's early stop reads."""
PANEL_WIDTH_PX: Final = 125
PANEL_HEIGHT_PX: Final = 105
N_COLUMNS: Final = 3


def fits_folder() -> Path:
    """Return the experiment's `fits` folder under `DATA_PATH_INTERNAL`."""
    root = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments" / e00730_paths.FOLDER
    return root / "fits"


def curve_rows(*, fits: Path) -> pl.DataFrame:
    """Return the per-epoch scores of every panel, with the stored epoch flagged."""
    rows: list[dict] = []
    for title, stem, score in PANELS:
        fit = json.loads((fits / f"{stem}_s{SEED}.json").read_text())
        for epoch, value in enumerate(fit["history"], start=1):
            rows.append(
                {
                    "panel": f"{title} ({score})",
                    "epoch": epoch,
                    "score": value,
                    "stored": epoch == fit["best_epoch"],
                }
            )
    return pl.DataFrame(rows)


def loss_chart(*, curves: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the panels: validation score against epoch, with the stored epoch marked."""
    order = curves["panel"].unique(maintain_order=True).to_list()
    base = alt.Chart().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickCount=4))
    )
    line = base.mark_line(color=figures729.BLUE, strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "score:Q",
            title=None,
            scale=alt.Scale(zero=False),
            axis=alt.Axis(tickCount=4),
        )
    )
    rule = base.mark_rule(color=figures729.ORANGE, strokeDash=[3, 3]).transform_filter(  # ty: ignore[unresolved-attribute]
        "datum.stored"
    )
    dot = (
        base.mark_circle(color=figures729.ORANGE, size=45, opacity=1)
        .encode(y="score:Q")  # ty: ignore[unresolved-attribute]
        .transform_filter("datum.stored")
    )
    layered = alt.layer(line, rule, dot).properties(width=PANEL_WIDTH_PX, height=PANEL_HEIGHT_PX)
    return (
        layered.facet(
            facet=alt.Facet("panel:N", sort=order, title=None, header=alt.Header(labelLimit=260)),
            columns=N_COLUMNS,
            data=curves,
        )
        .resolve_scale(x="independent", y="independent")
        .properties(
            title=alt.TitleParams(
                [
                    "Validation score of each arm's seed-0 fit at every epoch",
                    "Blue line: the score the early stop reads (smaller is better).",
                    "Orange dot and dashed line: the stored epoch.",
                    "The training loss was not stored. Each panel has its own scales.",
                ],
                anchor="start",
            )
        )
    )


def main() -> None:
    """Draw the figure into the experiment's `figures` folder and print the `svgo` command."""
    curves = curve_rows(fits=fits_folder())
    path = fits_folder().parent / "figures" / "loss_curves.svg"
    figures729.save_svg(chart=loss_chart(curves=curves), path=path)
    print(path)
    print(figures729.optimise_command(path=path))


if __name__ == "__main__":
    main()
