"""Figures for experiment 487: contrasts against R0 and the CRPS change by fold.

Reads `work/scores/contrasts.csv` and writes two SVG files beside the write-up. Run
`npx svgo@4 --multipass --precision=1 --final-newline` on them before committing.

```bash
source .e_replay_env.sh   # sets PYTHONPATH; run from the repo root
uv run --no-sync python ml-experiments/00487-*/e00487_figures.py
```
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00487_fit as fit  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

WIDTH: Final[int] = 640
FIGURE_DIR: Final[Path] = Path("docs/ml-experiments") / fit.EXPERIMENT_DIR_NAME
SLICES: Final[dict[str, str]] = {
    "eight": "Eight, all rows",
    "six_clean": "Six, no labelled event",
    "six_nra": "Six, real events (NRA actuals)",
    "cold_eight": "Eight, cold nights",
    "cold_six": "Six, cold nights",
    "band_eight": "Eight, 12 to 36 h",
    "band_six": "Six, 12 to 36 h",
}
ARMS: Final[list[str]] = ["R25", "R50", "Rcold"]
ARM_COLOURS: Final[list[str]] = [theme.DATA_BLUE, theme.BRAND_ORANGE, theme.DATA_GREEN]


def main() -> None:
    """Write both figures."""
    contrasts = pl.read_csv(fit.WORK_DIR / "scores" / "contrasts.csv").filter(
        pl.col("contrast").str.ends_with("- R0") & (pl.col("metric") == "crps")
    )
    counted = contrasts.filter(pl.col("counted")).with_columns(
        arm=pl.col("contrast").str.replace(" - R0", ""),
        label=pl.col("slice").replace_strict(SLICES),
    )
    colour = alt.Color("arm:N", scale=alt.Scale(domain=ARMS, range=ARM_COLOURS), title=None)
    base = alt.Chart(counted).encode(
        y=alt.Y("label:N", sort=list(SLICES.values()), title=None), yOffset="arm:N", color=colour
    )
    points = base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="CRPS change against R0 (points; below 0 is better)")
    )
    bars = base.mark_rule().encode(x="low_bonferroni:Q", x2="high_bonferroni:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=theme.GREY_2).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    alt.layer(zero, bars, points).properties(width=WIDTH, height=260).save(
        FIGURE_DIR / "figure_1_crps_contrasts_against_r0.svg"
    )
    by_fold_rows = contrasts.filter(pl.col("slice").is_in(["six_clean", "eight"]))
    rows = [
        {"arm": r["contrast"].replace(" - R0", ""), "month": month, "difference": value,
         "group": SLICES[r["slice"]]}
        for r in by_fold_rows.iter_rows(named=True)
        for month, value in json.loads(r["by_month"]).items()
    ]  # fmt: skip
    (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("month:N", title=None, sort=list(fit.MONTHS)),
            xOffset="arm:N",
            y=alt.Y("difference:Q", title="CRPS change against R0 (points)"),
            color=colour,
        )
        .properties(width=WIDTH // 2, height=150)
        .facet(column=alt.Column("group:N", title=None))
        .save(FIGURE_DIR / "figure_2_crps_change_by_fold.svg")
    )


if __name__ == "__main__":
    main()
