# ruff: noqa: E501
"""Example forecasts of experiment 556: an injected step and a real labelled event.

Each figure has a day-ahead and a 10-days-ahead version. Figure 5: a 7-day, 10%-of-capacity step
injected into the fine-tuning targets before the week-3 refit of the 2026-01 fold, seen at one
signed-MW Primary by L-Ww (F7d), F1d, and F3d-E3 (dashed: the same arm's uninjected fit). Figure 6:
a real labelled event of 26 points at the same Primary, fitted on metered and on NRA-corrected
targets by L-Ww (F7d) and F1d.
Axes count days from the forecast origin; no calendar date is shown.
"""

from datetime import timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00510_common as c  # ty: ignore[unresolved-import]
import e00556_fit as fit
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from plotting import ocf_theme as theme

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
WIDTH: Final[int] = 680
PRIMARY: Final[int] = 3
MONTH: Final[str] = "2026-01"
ARMS: Final[tuple[str, ...]] = ("F7d", "F1d", "F3d-E3")


def frame_of(
    *, cell: fit.Cell, model: torch.nn.Module, origin_us: int, primary: int
) -> pl.DataFrame:
    """Return quantile forecasts (points of capacity) of one origin at one Primary.

    The frame also holds the days from the origin and the actual load.
    """
    first = cell.test.origin[(cell.test.origin >= origin_us) & (cell.test.primary == primary)].min()
    sel = (cell.test.origin == first) & (cell.test.primary == primary)
    frame = c.forecast_frame(
        model=model,
        outlier=False,
        test=cell.test,
        select=sel,
        rows=cell.mr.base,
        widen=1.0,
        device=cell.device,
    )
    idx = np.nonzero(sel)[0]
    actual = cell.clean[primary, cell.test.target_idx[idx]] * 100.0
    levels = [x for x in frame.columns if x.startswith("q")]
    return frame.with_columns(
        *[pl.col(q) * 100.0 for q in levels],
        days=(pl.col("valid_time").dt.epoch("us") - int(first)) / c.DAY_US,
        actual=pl.Series(actual),
    )


def draw(
    *,
    parts: dict[str, pl.DataFrame],
    dashed: dict[str, pl.DataFrame],
    window: float,
    title: list[str],
    file: str,
    cf_label: str,
) -> None:
    """Draw one panel per arm: 50% and 90% bands, median, actual, and the counterfactual median."""
    panels = []
    colours = {"F7d": theme.DATA_GREEN, "F1d": theme.DATA_BLUE, "F3d-E3": theme.BRAND_ORANGE}
    for arm, frame in parts.items():
        f = frame.filter(pl.col("days") <= window).to_pandas()  # ty: ignore[unresolved-attribute]
        cf = (
            dashed[arm]
            .filter(pl.col("days") <= window)
            .select("days", cf=pl.col("q50"))
            .to_pandas()
        )  # ty: ignore[unresolved-attribute]
        base = alt.Chart(f).encode(
            x=alt.X(
                "days:Q",
                title="Days since the forecast origin",
                scale=alt.Scale(domain=[0, window]),
            )
        )
        b90 = base.mark_area(opacity=0.18, color=colours[arm]).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("q05:Q", title="Load (points of capacity)", scale=alt.Scale(zero=False)),
            y2="q95:Q",
        )
        b50 = base.mark_area(opacity=0.32, color=colours[arm]).encode(y="q25:Q", y2="q75:Q")  # ty: ignore[unresolved-attribute]
        med = base.mark_line(color=colours[arm], strokeWidth=2).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
        act = base.mark_line(color="black", strokeWidth=1.2).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
        dash = (
            alt.Chart(cf)  # ty: ignore[unresolved-attribute]
            .mark_line(color="#444444", strokeDash=[4, 3], strokeWidth=1.5)
            .encode(x="days:Q", y="cf:Q")
        )
        panels.append(
            alt.layer(b90, b50, med, dash, act).properties(
                width=WIDTH, height=150, title=alt.TitleParams(arm, anchor="start")
            )
        )
    note = f"Line: median. Dark band: 50% interval; light band: 90% interval. Black: actual. Dashed: {cf_label}."
    alt.vconcat(*panels).properties(title=alt.TitleParams([*title, note], anchor="start")).save(
        DOCS / file
    )


def main() -> None:
    """Fit the example cells and draw figures 5 and 6."""
    cell = fit.Cell(month=MONTH, device="cuda")
    # Injected step: week-3 origin (day 14), 7-day step of 10% of capacity at every Primary, positive sign.
    origin = fit.cut_of(month=MONTH, day=14)
    scen = fit.SCENARIOS["step7_p10"]
    stressed, clean = {}, {}
    for tag in ARMS:
        arm = fit.ARMS[tag]
        model = cell.fit(
            arm=arm, cut=origin, scen=scen, origin=origin, sign=1, ref=None, mad_model=None
        )
        stressed[tag] = frame_of(cell=cell, model=model, origin_us=c.us(t=origin), primary=PRIMARY)
        clean[tag] = frame_of(
            cell=cell,
            model=cell.control(arm=arm, cut=origin),
            origin_us=c.us(t=origin),
            primary=PRIMARY,
        )
    for window, name in ((1.0, "day_ahead"), (10.0, "ten_day")):
        draw(
            parts=stressed,
            dashed=clean,
            window=window,
            title=[
                "A 7-day step of +10% of capacity in the fine-tuning targets, seen at one signed-MW Primary.",
                "Forecast issued at the week-3 refit of the 2026-01 fold.",
            ],
            file=f"figure_5_injected_step_{name}.svg",
            cf_label="the same arm refitted without the step",
        )
    # Real event: metered versus NRA-corrected fine-tuning targets at the cut after the event.
    events = pl.read_parquet(fit.WORK / "scores" / "real_events.parquet").filter(
        (pl.col("arm") == "F1d")
        & (pl.col("primary") == PRIMARY)
        & (pl.col("month") == MONTH)
        & (pl.col("depth_points") >= 10)
    )
    row = events.sort("absorbed_share", descending=True).row(0, named=True)
    cut = fit.cut_of(month=MONTH, day=int(row["cut"][-2:]) - 1)
    metered, corrected = {}, {}
    for tag in ("F7d", "F1d"):
        arm = fit.ARMS[tag]
        model_m = cell.control(arm=arm, cut=cut)
        cell.nra_event = (PRIMARY, int(row["onset"]), int(row["end"]))
        model_c = cell.fit(
            arm=arm,
            cut=cut,
            scen=fit.Scenario("nra", chain=False),
            origin=cut,
            sign=1,
            ref=None,
            mad_model=None,
        )
        metered[tag] = frame_of(cell=cell, model=model_m, origin_us=c.us(t=cut), primary=PRIMARY)
        corrected[tag] = frame_of(cell=cell, model=model_c, origin_us=c.us(t=cut), primary=PRIMARY)
    for window, name in ((1.0, "day_ahead"), (10.0, "ten_day")):
        draw(
            parts=metered,
            dashed=corrected,
            window=window,
            title=[
                "A real labelled load-shift event ended about 3 days before the refit, at one signed-MW Primary.",
                f"Labelled depth {row['depth_points']:.0f} points of capacity; the refit trained on metered targets.",
            ],
            file=f"figure_6_real_event_{name}.svg",
            cf_label="the median after training on NRA-corrected targets",
        )
    print("event", row, "cut", cut)
    _ = timedelta


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    main()
