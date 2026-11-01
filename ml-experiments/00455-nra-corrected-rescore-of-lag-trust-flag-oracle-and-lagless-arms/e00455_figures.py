"""Draw the contrast figures of experiment 455 from `pooled.csv` (no fit).

```bash
DATA_PATH_INTERNAL=<private data folder> \
    uv run --no-sync python ml-experiments/00455-*/e00455_figures.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00455-nra-corrected-rescore-of-lag-trust-flag-oracle-and-lagless-arms"
FIGURE_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
LABELS: Final[dict[str, str]] = {
    "A0-inj": "A0-inj: plain network",
    "S1": "S1: plain + lag surprises",
    "B3": "B3: attention over lags",
    "B6": "B6: learned windows",
    "lagless": "lagless head alone",
    "O-hard": "O-hard: exact flags, switch",
    "O-soft": "O-soft: exact flags, blend",
    "O-floor": "O-floor: exact flags, floor 0.5",
}
FAMILY: Final[dict[str, str]] = {
    "A0-inj": "forecasts without an oracle",
    "S1": "forecasts without an oracle",
    "B3": "forecasts without an oracle",
    "B6": "forecasts without an oracle",
    "lagless": "forecasts without an oracle",
    "O-hard": "exact event flags (oracle)",
    "O-soft": "exact event flags (oracle)",
    "O-floor": "exact event flags (oracle)",
}
COLOURS: Final[dict[str, str]] = {
    "forecasts without an oracle": theme.DATA_BLUE,
    "exact event flags (oracle)": theme.BRAND_ORANGE,
}
DEPTH_TITLES: Final[dict[str, str]] = {
    "metered": "Metered actual",
    "0.75": "NRA-corrected, depth factor 0.75",
    "1.0": "NRA-corrected, depth factor 1.0",
    "1.25": "NRA-corrected, depth factor 1.25",
}
WIDTH: Final[int] = 430


def figure(*, group: str, group_name: str, drop: bool, number: int, stem: str) -> None:
    """Draw one contrast figure: arm minus B-mix by depth factor, two intervals per arm."""
    prefix = "drop_" if drop else ""
    mean = "drop_largest_mean" if drop else "mean_of_months"
    depths = ["1.0", "0.75", "1.25"] if drop else list(DEPTH_TITLES)
    data = (
        pl.read_csv(
            ML_EXPERIMENTS_DATA_DIR / SLUG / "pooled.csv", schema_overrides={"depth": pl.String}
        )
        .filter((pl.col("group") == group) & pl.col("depth").is_in(depths))
        .with_columns(
            label=pl.col("arm").replace(LABELS),
            family=pl.col("arm").replace(FAMILY),
            depth_title=pl.col("depth").replace(DEPTH_TITLES),
            mean=pl.col(mean),
            low95=pl.col(f"{prefix}low95" if drop else "low95"),
            high95=pl.col(f"{prefix}high95" if drop else "high95"),
            low_bonf=pl.col(f"{prefix}low_bonf"),
            high_bonf=pl.col(f"{prefix}high_bonf"),
        )
    )
    y = alt.Y("label:N", sort=list(LABELS.values()), title=None)
    colour = alt.Color(
        "family:N",
        scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
        legend=alt.Legend(title=None, orient="top"),
    )
    base = alt.Chart()
    thin = base.mark_rule(strokeWidth=1, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=y, x=alt.X("low_bonf:Q", title="Arm minus B-mix (points of capacity)"), x2="high_bonf:Q",
        color=colour,
    )  # fmt: skip
    thick = base.mark_rule(strokeWidth=4, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="low95:Q", x2="high95:Q", color=colour
    )
    dots = base.mark_point(filled=True, size=60, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="mean:Q", color=colour
    )
    zero = base.mark_rule(color=theme.GREY_2, strokeDash=[4, 3]).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    note = "largest event of each month dropped" if drop else "all events"
    chart = (
        alt.layer(zero, thin, thick, dots, data=data.to_pandas())
        .properties(width=WIDTH, height=170)
        .facet(
            row=alt.Row(
                "depth_title:N",
                title=None,
                sort=[DEPTH_TITLES[d] for d in depths],
                header=alt.Header(labelAngle=0, labelAlign="left"),
            )
        )
        .properties(
            title=alt.TitleParams(
                [
                    f"Rows inside real events at the {group_name}: each arm minus B-mix ({note})",
                    "Left of the dashed line is better than B-mix. Dot: mean of month means.",
                    "Thick line: 95% interval. Thin line: Bonferroni interval for 48 contrasts.",
                    "Both resample the months (5 at the signed-MW six, 4 at the MVA-only eight).",
                ],
                anchor="start",
            )
        )
    )
    chart.save(FIGURE_DIR / f"figure_{number}_{stem}.svg")


def main() -> None:
    """Draw the three figures."""
    figure(group="signed_six", group_name="signed-MW six", drop=False, number=1, stem="signed_six")
    figure(group="mva_eight", group_name="MVA-only eight", drop=False, number=2, stem="mva_eight")
    figure(
        group="signed_six", group_name="signed-MW six", drop=True, number=3, stem="signed_six_drop"
    )


if __name__ == "__main__":
    main()
