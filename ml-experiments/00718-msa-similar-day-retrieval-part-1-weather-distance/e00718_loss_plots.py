# ruff: noqa: E501
"""Experiments 718 and 719: the validation loss of every fit, by pass and stage, from the stored fit logs (plots only, no fit).

Usage: e00718_loss_plots.py OUT_DIR PART

One chart per part: three panels (stages 1, 2, 3), one line per arm, each line the mean over the folds and seeds that
the arm was fitted on at that pass. Lower is better. A line shortens where fits stop early, and the mean at a late pass
then rests on fewer fits, so the chart prints the number of fits in each arm's legend label.
"""

import json
import sys
from pathlib import Path

import altair as alt
import e00713_figures as f713  # ty: ignore[unresolved-import]
import e00718_figures as fig
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARMS = {1: ("Z", "R", "R-A", "P90"), 2: ("Z", "U", "U-A")}
FOLDER = {
    1: "00718-msa-similar-day-retrieval-part-1-weather-distance",
    2: "00719-msa-similar-day-retrieval-part-2-union-of-measures",
}


def main() -> int:
    """Write the loss chart of one part."""
    out_dir, part = Path(sys.argv[1]), int(sys.argv[2])
    rows = []
    for arm in ARMS[part]:
        folder = (
            ML_EXPERIMENTS_DATA_DIR
            / FOLDER[1 if arm in {"Z", "R", "R-A", "P90"} else 2]
            / "work"
            / "logs"
            / arm
        )
        for f in sorted(folder.glob("*_fit_stage?.json")):
            d = json.loads(f.read_text())
            if d.get("smoke"):
                continue
            hist = (d.get("rerun") or d["first"]).get("val_history", [])
            rows += [
                {"arm": arm, "stage": f"stage {d['stage']}", "job": f.name, "pass": i + 1, "val": v}
                for i, v in enumerate(hist)
            ]
    df = pl.DataFrame(rows)
    n_fits = df.group_by("arm").agg(n=pl.col("job").n_unique())
    label = {r["arm"]: f"{r['arm']} ({r['n']} stage fits)" for r in n_fits.iter_rows(named=True)}
    agg = (
        df.group_by("arm", "stage", "pass")
        .agg(val=pl.col("val").mean())
        .with_columns(arm_label=pl.col("arm").replace_strict(label))
    )
    domain = [label[a] for a in ARMS[part] if a in label]
    scale = alt.Scale(domain=domain, range=[fig.COLOUR[a] for a in ARMS[part] if a in label])
    chart = (
        alt.Chart(agg)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.6)
        .encode(
            x=alt.X("pass:Q", title="Training pass"),
            y=alt.Y(
                "val:Q", title="Validation loss (lower is better)", scale=alt.Scale(zero=False)
            ),
            color=alt.Color("arm_label:N", scale=scale, title="Arm"),
        )
        .properties(width=210, height=180)
        .facet(column=alt.Column("stage:N", title=None))
        .properties(
            title=f"Experiment {717 + part}: mean validation loss by training pass, over all folds and seeds"
        )
    )
    f713.save(chart, path=out_dir / f"loss_part{part}.svg")
    print("wrote", out_dir / f"loss_part{part}.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
