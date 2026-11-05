# ruff: noqa: E501, E402
"""Figure 3 of experiment 566: six real events, fits on metered and on corrected targets.

For each event the refit is the one `e00566_fit.py events` ran (one day after the event ends). Each
panel shows the 10 days after the forecast origin at one Primary: the actual load and, for arm C1
and for arm C4, the median forecast of the fit on metered targets and of the fit on corrected targets.
Axes count days from the forecast origin.
"""

import sys
from pathlib import Path
from typing import Final

_argv = sys.argv
sys.argv = [_argv[0], "events"]
import altair as alt
import e00510_common as c  # ty: ignore[unresolved-import]
import e00566_fit as fit
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

sys.argv = _argv
DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
SITES: Final[Path] = (
    fit.DD / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
)
WINDOW: Final[float] = 10.0
ARMS: Final[tuple[str, ...]] = ("C1", "C4")


def pick() -> pl.DataFrame:
    """Return six events: the three largest shares at the eight and the six, outside step 2."""
    sites = pl.read_csv(SITES).select("primary", "group")
    t = (
        pl.read_parquet(fit.W / "scores/real_events.parquet")
        .filter((pl.col("arm") == "C1") & (pl.col("min_age") == 1.0) & (pl.col("depth_points") >= 8) & ~pl.col("in_step2"))
        .join(sites, on="primary")
    )  # fmt: skip
    parts = [
        t.filter(pl.col("group") == g).sort("absorbed_share", descending=True).head(3)
        for g in ("MVA-only eight", "signed-MW six")
    ]
    return pl.concat(parts)


def frames(*, cell: fit.Cell, row: dict) -> pl.DataFrame:
    """Return the actual, and each arm's metered and corrected median, over the 10 days from the origin."""
    from datetime import datetime

    cut = datetime.fromisoformat(row["cut"] + "T00:00:00+00:00")
    p = row["primary"]
    first = cell.test.origin[(cell.test.origin >= c.us(t=cut)) & (cell.test.primary == p)].min()
    sel = np.nonzero((cell.test.origin == first) & (cell.test.primary == p))[0]
    days = (cell.test.valid[sel] - first) / c.DAY_US
    out = {"days": days, "actual": cell.clean[p, cell.test.target_idx[sel]] * 100.0}
    for tag in ARMS:
        arm = fit.ARMS[tag]
        refit = cell.select(arm=arm, cut=cut)
        prim, tgt = cell.mr.primary[refit.index], cell.mr.target_idx[refit.index]
        corr = -cell.shares[prim, np.clip(tgt, 0, cell.shares.shape[1] - 1)]
        corr = np.where((prim == p) & (tgt >= row["onset"]) & (tgt <= row["end"]), corr, 0.0)
        metered = cell.control(arm=arm, cut=cut, seed=0)
        corrected = cell.fit(arm=arm, cut=cut, seed=0, shift=corr.astype(np.float32))
        out[f"{tag} metered"] = cell.q50(model=metered, rows=sel) * 100.0
        out[f"{tag} corrected"] = cell.q50(model=corrected, rows=sel) * 100.0
    return pl.DataFrame(out).sort("days").filter(pl.col("days") <= WINDOW)


def main() -> None:
    """Draw the six panels (one row per event, one column per arm)."""
    events = pick().to_dicts()
    cells: dict[str, fit.Cell] = {}
    rows = []
    for i, ev in enumerate(events):
        if ev["month"] not in cells:
            cells.clear()
            cells[ev["month"]] = fit.Cell(month=ev["month"], device="cuda")
        f = frames(cell=cells[ev["month"]], row=ev)
        label = f"Event {i + 1}: {'eight' if ev['group'].startswith('MVA') else 'six'}, {ev['depth_points']:.0f} pts, {100 * ev['absorbed_share']:.0f}% absorbed"
        for tag in ARMS:
            for kind in ("metered", "corrected"):
                part = f.select("days", "actual", value=pl.col(f"{tag} {kind}")).with_columns(
                    event=pl.lit(label), arm=pl.lit(tag), fit=pl.lit(kind)
                )
                rows.append(part)
        print(label, flush=True)
    d = pl.concat(rows).to_pandas()
    base = alt.Chart(d).encode(x=alt.X("days:Q", title="Days since the forecast origin"))
    lines = base.mark_line(strokeWidth=1.8).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("value:Q", title="Load (points of capacity)", scale=alt.Scale(zero=False)),
        color=alt.Color("fit:N", scale=alt.Scale(domain=["metered", "corrected"], range=[theme.BRAND_ORANGE, theme.DATA_BLUE]), title="Fit trained on"),
    )  # fmt: skip
    actual = base.mark_line(color="black", strokeWidth=1).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(lines, actual)
        .properties(width=300, height=110)
        .facet(
            row=alt.Row(
                "event:N",
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left", labelLimit=220),
            ),
            column=alt.Column("arm:N", title=None),
        )
    )
    chart.properties(
        title=alt.TitleParams(
            [
                "Median forecast one day after a real load-shift event (black: actual).",
                "Fits on metered and on corrected targets; events outside step 2 with the largest absorbed shares.",
            ],
            anchor="start",
        )
    ).save(DOCS / "figure_3_real_event_examples.svg")


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    main()
