"""Draw the figures of experiment 396's write-up from `results/screen_results.csv` only.

- `headline.svg`: CRPS difference from the control (C0) on all rows, per arm, for each fold, in
  two panels (one per metering group).
- `holiday_slices.svg`: the bank-holiday and Christmas slice differences of the holiday arms.
- `versus_e1.svg`: the Tier B and Tier C arms minus E1_es2, in the two main folds.

    uv run --no-sync python ml-experiments/00396-*/e00396_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

HERE: Final[Path] = Path(__file__).resolve().parent
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / HERE.name
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
MAIN_FOLDS: Final[tuple[str, ...]] = ("2026-01", "2026-06")
FOLD_COLOURS: Final[dict[str, str]] = {
    "2026-01": ocf.BRAND_ORANGE,
    "2026-06": ocf.DATA_BLUE,
    "2025-12": ocf.DATA_PURPLE,
    "2026-04": ocf.DATA_SKY,
}
GROUP_SHAPES: Final[dict[str, str]] = {"signed-MW": "circle", "MVA-only": "diamond"}
NOISE_FLOOR_MAX: Final[float] = 0.015
HOLIDAY_ARMS: Final[list[str]] = ["H1", "H1d", "HW", "HT", "HWT"]
VERSUS_ARMS: Final[list[str]] = [
    "E4_em_pool",
    "E2_es7",
    "E3_em_em",
    "P_hl1",
    "P_rounds_r1000",
    "P_rounds_r200",
]
EXCLUDED: Final[tuple[str, ...]] = ("C1", "P_rounds_r500")


def read_results() -> pl.DataFrame:
    """Return the screen table."""
    return pl.read_csv(HERE / "results" / "screen_results.csv")


def _key_values(*, names: dict[str, str], title: str) -> alt.Color:
    return alt.Color(
        "fold:N",
        scale=alt.Scale(domain=list(names), range=list(names.values())),
        legend=alt.Legend(title=title, orient="top"),
    )


def _shape() -> alt.Shape:
    return alt.Shape(
        "group:N",
        scale=alt.Scale(domain=list(GROUP_SHAPES), range=list(GROUP_SHAPES.values())),
        legend=alt.Legend(title="Metering group", orient="top"),
    )


def _zero_rule(*, label: str) -> alt.LayerChart:
    frame = pl.DataFrame({"x": [0.0], "label": [label]})
    rule = alt.Chart(frame).mark_rule(color="#808080", strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    text = (
        alt.Chart(frame)
        .mark_text(align="left", dx=3, dy=-6, baseline="bottom", color="#606060", fontSize=10)
        .encode(x="x:Q", y=alt.value(0), text="label:N")  # ty: ignore[unresolved-attribute]
    )
    return rule + text


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 60), subtitle=textwrap.wrap(subtitle, WRAP), anchor="start"
    )


def headline_chart(*, results: pl.DataFrame) -> None:
    """Write `headline.svg`."""
    data = results.filter(~pl.col("arm").is_in(EXCLUDED)).select("fold", "group", "arm", "all")
    order = (
        data.filter(pl.col("fold").is_in(MAIN_FOLDS))
        .group_by("arm")
        .agg(pl.col("all").mean())
        .sort("all")["arm"]
        .to_list()
    )
    band = pl.DataFrame({"low": [-NOISE_FLOOR_MAX], "high": [NOISE_FLOOR_MAX]})
    panels = []
    for group in GROUP_SHAPES:
        base = alt.Chart(data.filter(pl.col("group") == group))
        shade = (
            alt.Chart(band).mark_rect(color="#808080", opacity=0.15).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
        )
        dots = base.mark_point(filled=True, size=55, shape=GROUP_SHAPES[group]).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "all:Q",
                title="CRPS minus control's (points of effective capacity; negative is better)",
                scale=alt.Scale(domain=[-0.15, 0.45], clamp=True),
            ),
            y=alt.Y("arm:N", sort=order, title=None),
            color=_key_values(names=FOLD_COLOURS, title="Test month"),
        )
        panels.append(
            (shade + _zero_rule(label="control") + dots).properties(
                width=WIDTH - 140, height=15 * len(order), title=group
            )
        )
    chart = (
        alt.vconcat(*panels)
        .properties(
            title=_title(
                text="Ensemble-summary arms and W7 lower the error most consistently",
                subtitle=(
                    "Change in CRPS at days 1 to 10 against the control (C0), all rows, per test "
                    "month, in 16 Primaries split into the signed-MW and MVA-only metering groups. "
                    "Arms are ordered by their mean over 2026-01 and 2026-06. Grey band: plus or "
                    "minus 0.015, the largest gap between the two control seeds. Arms missing from "
                    "a month were not run there."
                ),
            )
        )
        .resolve_scale(color="shared")
    )
    chart.save(DOCS_DIR / "headline.svg")


def holiday_chart(*, results: pl.DataFrame) -> None:
    """Write `holiday_slices.svg`."""
    long = (
        results.filter(pl.col("arm").is_in(HOLIDAY_ARMS))
        .unpivot(
            on=["bank_holidays", "christmas"],
            index=["fold", "group", "arm"],
            variable_name="slice",
            value_name="difference",
        )
        .drop_nulls("difference")
        .with_columns(
            slice=pl.col("slice").replace(
                {"bank_holidays": "Bank-holiday days", "christmas": "Christmas to New Year"}
            )
        )
    )
    panels = []
    for slice_name in ("Bank-holiday days", "Christmas to New Year"):
        dots = (
            alt.Chart(long.filter(pl.col("slice") == slice_name))
            .mark_point(filled=True, size=55)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "difference:Q",
                    title="CRPS minus control's (points of effective capacity; negative is better)",
                ),
                y=alt.Y("arm:N", sort=HOLIDAY_ARMS, title=None),
                color=_key_values(names=FOLD_COLOURS, title="Test month"),
                shape=_shape(),
            )
        )
        panels.append(
            (_zero_rule(label="control") + dots).properties(
                width=WIDTH - 140, height=22 * len(HOLIDAY_ARMS), title=slice_name
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=_title(
            text="Holiday columns lower the error on the Easter bank holidays, not at Christmas",
            subtitle=(
                "Change in CRPS at days 1 to 10 against the control (C0), on the days of each "
                "slice only. Negative is better. Bank-holiday days fall in 2025-12, 2026-01, "
                "and 2026-04 (Easter). Christmas to New Year falls in 2025-12 and 2026-01. Dot "
                "colour: test month. Dot shape: metering group."
            ),
        )
    )
    chart.save(DOCS_DIR / "holiday_slices.svg")


def versus_chart(*, results: pl.DataFrame) -> None:
    """Write `versus_e1.svg`."""
    main = results.filter(pl.col("fold").is_in(MAIN_FOLDS))
    reference = main.filter(pl.col("arm") == "E1_es2").select(
        "fold", "group", reference=pl.col("all")
    )
    data = (
        main.filter(pl.col("arm").is_in(VERSUS_ARMS))
        .join(reference, on=["fold", "group"])
        .with_columns(difference=pl.col("all") - pl.col("reference"))
    )
    dots = (
        alt.Chart(data)
        .mark_point(filled=True, size=55)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="CRPS minus E1_es2's (points of effective capacity; negative is better)",
            ),
            y=alt.Y("arm:N", sort=VERSUS_ARMS, title=None),
            color=_key_values(names={f: FOLD_COLOURS[f] for f in MAIN_FOLDS}, title="Test month"),
            shape=_shape(),
        )
    )
    chart = (_zero_rule(label="E1_es2") + dots).properties(
        width=WIDTH - 140,
        height=24 * len(VERSUS_ARMS),
        title=_title(
            text="Training on the ensemble mean (E4_em_pool) beats E1_es2 in January, not in June",
            subtitle=(
                "Change in CRPS at days 1 to 10 of each Tier B and Tier C arm relative to "
                "E1_es2, the best Tier A arm, in the two main test months. Negative means "
                "better than E1_es2. Dot colour: test month. Dot shape: metering group."
            ),
        ),
    )
    chart.save(DOCS_DIR / "versus_e1.svg")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    table = read_results()
    headline_chart(results=table)
    holiday_chart(results=table)
    versus_chart(results=table)
