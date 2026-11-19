# ruff: noqa: E501, PERF401
"""Experiment 702: print the tables of the write-up and draw its three charts from the score files.

    uv run python ml-experiments/00702-*/e00702_report.py [--out docs/ml-experiments/00702-*/]

Reads the CSV tables that `e00702_score.py` wrote to `scores/` in the experiment's private data
folder, and the phase logs under `arms/`. It fits nothing and scores nothing. It prints every table
that `result.md` quotes, and writes three SVG charts to `--out`:

- `contrasts_four_and_nine_months.svg`: the five planned contrasts with their 99% intervals;
- `per_primary_c1_c3.svg`: contrasts C1 and C3 for each of the 16 Primaries;
- `contrasts_by_month.svg`: the mean of each contrast in each of the nine scored months.

Run `npx svgo@4 --multipass --precision=1 --final-newline` on each chart before committing it.
"""

import argparse
import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00702-per-substation-versus-global-training-of-the-modular-demand-network"
ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
SCORES: Final[Path] = ROOT / "scores"
SETS: Final[dict[str, str]] = {
    "four_months": "The four test months of the plan",
    "nine_months": "Nine months (five added after the four were scored; post hoc)",
}
"""The two scored sets of months, keyed by the folder under `scores/`."""
SET_COLOURS: Final[list[str]] = [ocf.BRAND_ORANGE, ocf.DATA_BLUE]
TEST_MONTHS: Final[tuple[str, ...]] = ("2025-11", "2026-01", "2026-04", "2026-07")
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1 split-at-1 minus global": "C1: split-at-1 minus global",
    "C2 split-at-2 minus global": "C2: split-at-2 minus global",
    "C3 split-at-3 minus global": "C3: split-at-3 minus global",
    "C4 blend minus global pair": "C4: blend minus two global seeds",
    "C5 split-at-2-plus-encoders minus split-at-2": "C5: plus-encoders minus split-at-2",
}
PRIMARIES: Final[tuple[str, ...]] = (
    "D2", "D3", "D4", "D5", "D8", "D9", "D10", "D11", "D12", "D14", "D15", "D16", "D18", "D19", "D21", "D22",
)  # fmt: skip
WIDTH: Final[int] = 680
"""The width of the text column of a docs page, in pixels (`studies.charts.CONTENT_WIDTH_PX`)."""
LABEL_PX: Final[int] = 215
LANE: Final[float] = 0.16
"""Half the distance between the two sets' dots within a row, in row units."""
ARM_NOTE: Final[str] = (
    "Split-at-N: one neural network per Primary from step N of the three-step schedule."
)


def contrasts(*, name: str) -> pl.DataFrame:
    """Return one set's `contrasts.csv`."""
    return pl.read_csv(SCORES / name / "contrasts.csv")


def print_tables() -> None:
    """Print every table that the write-up quotes, from the score files and the phase logs."""
    with pl.Config(
        tbl_rows=200, tbl_cols=30, float_precision=3, tbl_width_chars=260, fmt_str_lengths=60
    ):
        for name in (*SETS, "extra_months"):
            print(f"\n== {name}")
            if name in SETS:
                print(pl.read_csv(SCORES / name / "arms.csv"))
                print(pl.read_csv(SCORES / name / "tails.csv"))
                print(pl.read_csv(SCORES / name / "row_counts.csv"))
                print(pl.read_csv(SCORES / name / "groups.csv"))
                print(
                    pl.read_csv(SCORES / name / "lead_bands.csv").pivot(
                        on="band", index="contrast", values="mean"
                    )
                )
                for file in ("per_primary", "leave_one_out"):
                    table = pl.read_csv(SCORES / name / f"{file}.csv").with_columns(
                        contrast=pl.col("contrast").str.slice(0, 2),
                        cell=pl.format(
                            "{} [{}, {}]",
                            pl.col("mean").round(2),
                            pl.col("lo").round(2),
                            pl.col("hi").round(2),
                        ),
                    )
                    print(file)
                    print(table.pivot(on="contrast", index=table.columns[0], values="cell"))
                    print(
                        table.group_by("contrast")
                        .agg(
                            negative=(pl.col("mean") < 0).sum(),
                            interval_below_zero=(pl.col("hi") < 0).sum(),
                            interval_above_zero=(pl.col("lo") > 0).sum(),
                        )
                        .sort("contrast")
                    )
            print(contrasts(name=name).transpose(include_header=True))
        print("\n== nine months, CRPS and normalised MAE of the median by arm and month")
        by_month = pl.read_csv(SCORES / "nine_months" / "arms_by_month.csv")
        print(by_month.pivot(on="month", index="arm", values="crps"))
        print(by_month.pivot(on="month", index="arm", values="nmae_median"))
        print("\n== copied share of the held-out injected shifts (four months, then nine)")
        print(pl.read_csv(SCORES / "bank_follow.csv"))
        print(pl.read_csv(SCORES / "nine_months" / "bank_follow.csv"))
        print("\n== match of the stored control, and the first run (interim, seed 0)")
        print(pl.read_csv(SCORES / "match_stored.csv"))
        print(pl.read_csv(SCORES / "pilot" / "pilot_per_primary.csv"))
        print("\n== training phases of the nine folds")
        phases = phase_table()
        print(
            phases.group_by("arm", "step")
            .agg(
                fits=pl.len(),
                fixed_epoch_fits=pl.col("fixed").sum(),
                epochs_median=pl.col("epochs").median(),
                epochs_min=pl.col("epochs").min(),
                epochs_max=pl.col("epochs").max(),
                seconds_median=pl.col("seconds").median(),
                gpu_hours=pl.col("seconds").sum() / 3600,
                gpu_peak_mb=pl.col("gpu_peak_mb").max(),
                host_peak_mb=pl.col("host_peak_rss_mb").max(),
            )
            .sort("arm", "step")
        )
        print("GPU hours of all training phases:", round(phases["seconds"].sum() / 3600, 2))
        print(
            phases.filter(
                (pl.col("arm") == "split1")
                & (pl.col("step") == "1")
                & pl.col("label").is_in(["D15", "D18", "D2"])
            )
            .group_by("label")
            .agg(
                train_rows_min=pl.col("train_rows").min(),
                train_rows_max=pl.col("train_rows").max(),
                epochs_median=pl.col("epochs").median(),
                epochs_max=pl.col("epochs").max(),
                held_weeks_min=pl.col("held_weeks").min(),
            )
            .sort("label")
        )


def phase_table() -> pl.DataFrame:
    """Return one row per training phase of the nine folds, from the `phases_*train*.json` logs."""
    rows = []
    for path in sorted(ROOT.glob("arms/*/20*/*/work/logs/phases_*train*.json")):
        arm, month = path.parts[-6], path.parts[-5]
        for phase in json.loads(path.read_text())["phases"]:
            rows.append(
                {
                    "arm": arm,
                    "month": month,
                    "label": phase["label"],
                    "step": phase["step"],
                    "epochs": phase["epochs"],
                    "fixed": phase["fixed_epochs"] is not None,
                    "seconds": phase["seconds"],
                    "train_rows": phase["train_rows"],
                    "held_weeks": phase["held_weeks"],
                    "gpu_peak_mb": phase["gpu_peak_mb"],
                    "host_peak_rss_mb": phase["host_peak_rss_mb"],
                }
            )
    return pl.DataFrame(rows)


def forest(
    *, data: pl.DataFrame, rows: list[str], x_title: str, row_px: int, title: list[str] | None
) -> alt.LayerChart | alt.FacetChart:
    """Return a dot-and-interval panel: one row per name in `rows`, one lane per set of months.

    Args:
        data: Columns `row_name`, `set` (a value of `SETS`), `mean`, `lo`, and `hi`.
        rows: The row names, top to bottom.
        x_title: The title of the horizontal axis.
        row_px: The height of a row in pixels.
        title: The panel's title lines, or None.

    Returns:
        The layered chart, with every second row shaded and a vertical rule at zero.
    """
    position = {name: i for i, name in enumerate(rows)}
    lane = {name: (i - 0.5) * 2 * LANE for i, name in enumerate(SETS.values())}
    frame = data.with_columns(
        y=pl.col("row_name").replace_strict(position, return_dtype=pl.Float64)
        + pl.col("set").replace_strict(lane, return_dtype=pl.Float64)
    )
    y_scale = alt.Scale(domain=[len(rows) - 0.5, -0.5])
    axis = alt.Axis(
        values=list(range(len(rows))),
        labelExpr="{" + ", ".join(f"{i}: '{n}'" for i, n in enumerate(rows)) + "}[datum.value]",
        ticks=False,
        domain=False,
        grid=False,
        title=None,
        labelLimit=LABEL_PX,
        minExtent=LABEL_PX,
        maxExtent=LABEL_PX,
    )
    colour = alt.Color(
        "set:N",
        scale=alt.Scale(domain=list(SETS.values()), range=SET_COLOURS),
        legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=600),
    )
    stripes = (
        alt.Chart(
            pl.DataFrame({"top": [i - 0.5 for i in range(1, len(rows), 2)]}).with_columns(
                bottom=pl.col("top") + 1
            )
        )
        .mark_rect(color="#EDEDED")
        .encode(y=alt.Y("top:Q", scale=y_scale, axis=axis), y2="bottom:Q")  # ty: ignore[unresolved-attribute]
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color=ocf.BLACK_2, strokeWidth=1)
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    base = alt.Chart(frame)
    interval = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lo:Q", title=x_title),
        x2="hi:Q",
        y=alt.Y("y:Q", scale=y_scale, axis=axis),
        color=colour,
    )
    dots = base.mark_point(filled=True, size=55, opacity=1).encode(  # ty: ignore[unresolved-attribute]
        x="mean:Q",
        y=alt.Y("y:Q", scale=y_scale, axis=axis),
        color=colour,
        shape=alt.Shape(
            "set:N",
            scale=alt.Scale(domain=list(SETS.values()), range=["circle", "diamond"]),
            legend=None,
        ),
    )
    chart = alt.layer(stripes, zero, interval, dots).properties(
        width=WIDTH - LABEL_PX - 25, height=row_px * len(rows)
    )
    return chart.properties(title=alt.Title(title, anchor="start", fontSize=12)) if title else chart


def both_sets(*, file: str, key: str) -> pl.DataFrame:
    """Return a per-row score file of both sets of months, with the set's display name in `set`."""
    return pl.concat(
        [
            pl.read_csv(SCORES / name / f"{file}.csv").with_columns(set=pl.lit(text))
            for name, text in SETS.items()
        ],
        how="diagonal",
    ).rename({key: "row_name"})


DIFFERENCE_TITLE: Final[str] = (
    "Difference in CRPS (points of capacity; below zero: the first-named arm is better)"
)
SCOPE: Final[str] = (
    "16 Primary substations, daily 12:00 UTC forecasts, leads of 0 to 240 hours, 3 seeds."
)
KEY: Final[str] = (
    "Dot: mean difference. Line: 99% interval from resampling 7-day blocks of forecast days and a seed."
)


def contrast_chart() -> alt.LayerChart | alt.FacetChart:
    """Return the chart of the five planned contrasts on the four test months and on nine months."""
    data = both_sets(file="contrasts", key="contrast").with_columns(
        row_name=pl.col("row_name").replace_strict(CONTRAST_LABELS)
    )
    return forest(
        data=data,
        rows=list(CONTRAST_LABELS.values()),
        x_title=DIFFERENCE_TITLE,
        row_px=40,
        title=None,
    ).properties(
        title=alt.Title(
            "No per-substation arm scores clearly below the global neural network",
            subtitle=[
                ARM_NOTE,
                "Blend: the mean of the global and the split-at-1 forecasts. Vertical line: no difference.",
                KEY,
                SCOPE,
            ],
            anchor="start",
        )
    )


def per_primary_chart() -> alt.VConcatChart:
    """Return the chart of contrasts C1 and C3 for each Primary, on both sets of months."""
    data = both_sets(file="per_primary", key="primary")
    panels = []
    for prefix, text in (
        ("C1", "C1: split-at-1 (every step per Primary) minus global"),
        ("C3", "C3: split-at-3 (only the last step per Primary) minus global"),
    ):
        panels.append(
            forest(
                data=data.filter(pl.col("contrast").str.starts_with(prefix)),
                rows=list(PRIMARIES),
                x_title=DIFFERENCE_TITLE,
                row_px=26,
                title=[text],
            )
        )
    return alt.vconcat(*panels).properties(
        title=alt.Title(
            "At no Primary does a per-substation arm score clearly below the global arm",
            subtitle=[
                "Each row is one Primary substation, scored alone. The two panels have different scales.",
                KEY,
                "Daily 12:00 UTC forecasts, leads of 0 to 240 hours, 3 seeds. Vertical line: no difference.",
            ],
            anchor="start",
        )
    )


def month_chart() -> alt.VConcatChart:
    """Return the chart of each contrast's mean in each of the nine scored months."""
    wide = contrasts(name="nine_months")
    months = [c.removeprefix("mean_") for c in wide.columns if c.startswith("mean_20")]
    long = (
        wide.select("contrast", *[f"mean_{m}" for m in months])
        .unpivot(index="contrast", variable_name="month", value_name="mean")
        .with_columns(month=pl.col("month").str.strip_prefix("mean_"))
        .with_columns(
            kind=pl.when(pl.col("month").is_in(list(TEST_MONTHS)))
            .then(pl.lit("Test month of the plan"))
            .otherwise(pl.lit("Month added after the four were scored (post hoc)")),
            zero=pl.lit(0.0),
        )
    )
    kinds = ["Test month of the plan", "Month added after the four were scored (post hoc)"]
    colour = alt.Color(
        "kind:N",
        scale=alt.Scale(domain=kinds, range=SET_COLOURS),
        legend=alt.Legend(title=None, orient="top", labelLimit=400),
    )
    y_scale = alt.Scale(domain=[-0.15, 0.4])
    y_axis = alt.Axis(
        values=[-0.1, 0, 0.2, 0.4],
        labelExpr="datum.value == 0 ? '0' : format(datum.value, '+.1f')",
        labelOverlap=False,
        labelFontSize=9,
    )
    panels = []
    for i, (name, text) in enumerate(CONTRAST_LABELS.items()):
        last = i == len(CONTRAST_LABELS) - 1
        base = alt.Chart(long.filter(pl.col("contrast") == name))
        x = alt.X(
            "month:O",
            title="Month of the forecast origin" if last else None,
            axis=alt.Axis(labels=last, ticks=last, labelAngle=0),
        )
        stems = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("zero:Q", scale=y_scale, title=None, axis=y_axis),
            y2="mean:Q",
            color=colour,
        )
        dots = base.mark_point(filled=True, size=60, opacity=1).encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("mean:Q", scale=y_scale, title=None, axis=y_axis),
            color=colour,
            shape=alt.Shape(
                "kind:N", scale=alt.Scale(domain=kinds, range=["circle", "diamond"]), legend=None
            ),
        )
        rule = (
            alt.Chart(pl.DataFrame({"y": [0.0]}))
            .mark_rule(color=ocf.BLACK_2, strokeWidth=1)
            .encode(y=alt.Y("y:Q", scale=y_scale, title=None, axis=y_axis))  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            alt.layer(rule, stems, dots).properties(
                width=WIDTH - 60, height=96, title=alt.Title(text, anchor="start", fontSize=12)
            )
        )
    return alt.vconcat(*panels, spacing=8).properties(
        title=alt.Title(
            "No contrast has the same sign in every month",
            subtitle=[
                "Mean difference in CRPS in each month, in points of capacity. Below the line at zero, the",
                "first-named arm is better. " + ARM_NOTE,
                "16 Primary substations, daily 12:00 UTC forecasts, leads of 0 to 240 hours, mean of 3 seeds.",
            ],
            anchor="start",
        )
    )


def main() -> None:
    """Print the tables, and write the charts when `--out` is given."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="folder for the SVG charts")
    args = parser.parse_args()
    print_tables()
    if args.out is None:
        return
    for name, chart in (
        ("contrasts_four_and_nine_months", contrast_chart()),
        ("per_primary_c1_c3", per_primary_chart()),
        ("contrasts_by_month", month_chart()),
    ):
        chart.save(args.out / f"{name}.svg")
        print("wrote", args.out / f"{name}.svg")


if __name__ == "__main__":
    main()
