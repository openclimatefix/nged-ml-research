# ruff: noqa: E501
"""Draw part 2's forecast examples and validation-loss curves from stored files only.

Usage: run.sh e00717_plots.py forecasts OUT_DIR   # 14-day and day-ahead forecasts of D11, D19, D21
       run.sh e00717_plots.py loss OUT_DIR        # validation loss per pass of the three fits of fold 2026-05, seed 0
Reads the stored forecasts of arms `HT` and `H` (fold 2026-05, seed 0) and the fit logs of `HT`,
`HT-S`, and `T`. It scores nothing and writes only the SVG files. The forecast hosts are D11, D19,
and D21 only. The day counted from the origin replaces the date on every axis.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00713_score as s713  # ty: ignore[unresolved-import]
import e00716_arms as arms  # ty: ignore[unresolved-import]
import e00716_figures as figs  # ty: ignore[unresolved-import]
import e00716_score as score  # ty: ignore[unresolved-import]
import e00717_guard as guard
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

guard.require_part1(arms, figs, score)

FOLD: Final = "2026-05"
SEED: Final = 0
HOSTS: Final = {"D11": 8, "D19": 9, "D21": 24}
"""The only Primaries that host a forecast example, with their `time_series_id`."""
ARM_COLOUR: Final = {
    "HT": theme.DATA_GREEN,
    "H": theme.DATA_BLUE,
    "HT-S": theme.DATA_SKY,
    "T": theme.BRAND_ORANGE,
}
WIDTH: Final = figs.WIDTH


def forecast_frame() -> tuple[pl.DataFrame, datetime]:
    """Return the host rows of arms `HT` and `H` for the middle origin of the fold, with the actual."""
    truth = s713.truth_stack()
    parts = []
    for arm in ("HT", "H"):
        raw = score.load_arm(arm=arm, seeds=(SEED,), months=(FOLD,))
        raw = raw.filter(pl.col("time_series_id").is_in(list(HOSTS.values())))
        parts.append(s713.add_losses(frame=raw, truth=truth).with_columns(arm=pl.lit(arm)))
    frame = pl.concat(parts)
    origins = sorted(frame["power_fcst_init_time"].unique().to_list())
    origin = origins[len(origins) // 2]
    return frame.filter(pl.col("power_fcst_init_time") == origin), origin


def tidy(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the host rows with the days from the origin, the host label, and the quantiles in % of capacity."""
    names = {v: k for k, v in HOSTS.items()}
    return frame.with_columns(
        day=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 1440.0,
        host=pl.col("time_series_id").replace_strict(names, return_dtype=pl.String),
        q05=pl.col("q05") * 100,
        q50=pl.col("q50") * 100,
        q95=pl.col("q95") * 100,
        actual=pl.col("actual") * 100,
    )


def forecast_chart(
    *, table: pl.DataFrame, lo: float, hi: float, title: list[str]
) -> alt.TopLevelMixin:
    """Return one panel per host: the 5% to 95% band and median of `HT`, the median of `H`, and the actual."""
    x = alt.X("day:Q", title="Days after the forecast origin", scale=alt.Scale(domain=[lo, hi]))
    band = (
        alt.Chart()
        .transform_filter(alt.datum.arm == "HT")
        .mark_area(opacity=0.25, color=theme.DATA_GREEN, clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x, y=alt.Y("q05:Q", title="Power, % of capacity"), y2="q95:Q"
        )
    )
    med = (
        alt.Chart()
        .mark_line(clip=True, strokeWidth=1.5)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y="q50:Q",
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=["HT", "H"], range=[theme.DATA_GREEN, theme.DATA_BLUE]),
                legend=alt.Legend(title="Median forecast", orient="top"),
            ),
        )
    )
    actual = (
        alt.Chart()
        .transform_filter(alt.datum.arm == "HT")
        .mark_line(clip=True, color="#222222", strokeWidth=1.5, strokeDash=[3, 2])
        .encode(  # ty: ignore[unresolved-attribute]
            x=x, y="actual:Q"
        )
    )
    return (
        alt.layer(band, med, actual, data=table)
        .properties(width=WIDTH, height=130)
        .facet(row=alt.Row("host:N", title=None, sort=list(HOSTS)))
        .properties(title=title)
    )


def cmd_forecasts(*, out_dir: Path) -> int:
    """Draw the 14-day and the day-ahead examples."""
    frame, origin = forecast_frame()
    print(
        "origin of the examples:", origin, "(the middle origin of the fold; hosts", list(HOSTS), ")"
    )
    table = tidy(frame=frame)
    note = (
        "Green band: 5% to 95% of arm HT. Dashed line: what happened. Days counted from the origin."
    )
    figs.save(
        forecast_chart(
            table=table,
            lo=0.0,
            hi=14.0,
            title=["Full 14-day forecast of three Primaries, May 2026 fold, seed 0", note],
        ),
        path=out_dir / "forecast_14day.svg",
    )
    figs.save(
        forecast_chart(
            table=table,
            lo=1.0,
            hi=2.0,
            title=["Day-ahead forecast (lead 24 h to 48 h), same origin", note],
        ),
        path=out_dir / "forecast_day_ahead.svg",
    )
    return 0


def cmd_loss(*, out_dir: Path) -> int:
    """Draw the validation loss of stages 2 and 3 for the three fits of one fold and seed."""
    rows = []
    for arm in ("HT", "HT-S", "T"):
        log = json.loads(
            (
                ML_EXPERIMENTS_DATA_DIR
                / "00717-msa-unmetered-pv-part-2-target-row-forecast"
                / "work"
                / "logs"
                / arm
                / f"{FOLD}_s{SEED}_fit.json"
            ).read_text()
        )
        for stage in log["stages"]:
            for i, v in enumerate(stage["first"]["val_history"], start=1):
                rows.append(
                    {
                        "arm": arm,
                        "stage": f"Stage {stage['stage']}",
                        "pass": i,
                        "validation_loss": v,
                    }
                )
    table = pl.DataFrame(rows)
    chart = (
        alt.Chart(table)
        .mark_line(strokeWidth=1.5)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("pass:Q", title="Training pass"),
            y=alt.Y(
                "validation_loss:Q",
                title="Validation loss (smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["HT", "HT-S", "T"],
                    range=[ARM_COLOUR["HT"], ARM_COLOUR["HT-S"], ARM_COLOUR["T"]],
                ),
                legend=alt.Legend(title="Arm", orient="top"),
            ),
        )
        .properties(width=WIDTH // 2, height=170)
        .facet(column=alt.Column("stage:N", title=None))
        .resolve_scale(y="independent")
        .properties(
            title=[
                "Validation loss per pass, fold 2026-05, seed 0",
                "Training loss was not stored. Each fit stops when validation loss has not improved for 8 passes.",
            ]
        )
    )
    figs.save(chart, path=out_dir / "validation_loss.svg")
    return 0


def main() -> int:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("forecasts", "loss"))
    parser.add_argument("out_dir", nargs="?", type=Path)
    args = parser.parse_args()
    assert args.out_dir is not None
    assert args.command is not None
    return (
        cmd_forecasts(out_dir=args.out_dir)
        if args.command == "forecasts"
        else cmd_loss(out_dir=args.out_dir)
    )


if __name__ == "__main__":
    sys.exit(main())
