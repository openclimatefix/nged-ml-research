"""Experiment 468, Part B (real events): demand forecasts around labelled and scan-found events.

For each event, draws the actual readings, the day-ahead (12 to 36 hours ahead) median of the
lagless network (`LL`), S1, B-mix, and K2-S1 with K2-S1's 90% band, and a strip showing which
forecasts' lags fall inside a hand-labelled event. Forecasts are experiment 454's stored
`predictions/` frames (clean world, seed 0, one fit per month). Events are experiment 404's six
plotted labelled events plus scan-found events of experiment 456 that lie wholly inside a scored
month. NGED's event labels are incomplete, so a scan-found event is not proven to be a load shift.
Primaries are substations, so no anonymisation applies.

```bash
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    uv run --no-sync python ml-experiments/00468-*/e00468_part_b_real.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
PRED: Final[Path] = (
    DATA
    / "00454-lag-trust-mixture-and-ensemble-constructs-for-lagged-and-lagless-heads"
    / "work"
    / "predictions"
)
SCAN: Final[Path] = DATA / "00456-primary-demand-unmirrored-level-step-scan" / "work"
OUT: Final[Path] = (
    DATA / "00468-best-models-performance-analysis-with-time-series-graphs" / "part_b_real"
)
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
ARMS: Final[dict[str, str]] = {
    "LL": "Lagless network",
    "S1": "S1 (lagged, trust-weighted)",
    "B-mix": "B-mix",
    "K2-S1": "K2-S1",
}
COLOURS: Final[dict[str, str]] = {
    "Actual": ocf.BLACK_1,
    "Lagless network": ocf.DATA_SKY,
    "S1 (lagged, trust-weighted)": ocf.DATA_PURPLE,
    "B-mix": ocf.DATA_GREEN,
    "K2-S1": ocf.BRAND_ORANGE,
}
LABELLED_404: Final[tuple[tuple[str, int, int], ...]] = (
    ("D5", 112321, 112550),
    ("D5", 111746, 111922),
    ("D5", 107175, 107377),
    ("D14", 106721, 106748),
    ("D16", 112194, 112242),
    ("D14", 114748, 114772),
)
"""Primary, onset grid index, end grid index of experiment 404's six plotted events."""
MARGIN: Final[int] = 4 * 48
LEAD: Final[tuple[float, float]] = (12.0, 36.0)
WIDTH: Final[int] = 680
MAX_SCAN_EVENTS: Final[int] = 10


def month_bounds() -> dict[str, tuple[int, int]]:
    """Return the first and last target grid index each month's stored frames cover."""
    out = {}
    for m in MONTHS:
        d = pl.read_parquet(PRED / "K2-S1" / f"{m}_s0_w-1.parquet", columns=["target"])
        out[m] = (int(d["target"].min()), int(d["target"].max()))
    return out


def pick_events(*, sites: list[str], bounds: dict[str, tuple[int, int]]) -> pl.DataFrame:
    """Return the labelled and scan-found events that fit wholly inside one scored month."""
    rows = [
        dict(site=s, onset=a, end=b, source="labelled (experiment 404)", klass="labelled")
        for s, a, b in LABELLED_404
    ]
    scan = pl.read_csv(SCAN / "candidate_steps.csv").filter(
        ~pl.col("labelled")
        & (pl.col("end_day") - pl.col("start_day") >= 3)
        & (pl.col("points").abs() >= 4.0)
        & ~pl.col("near_block_boundary")
    )
    scan = scan.with_columns(onset=pl.col("start_day") * 48, end=(pl.col("end_day") + 1) * 48)
    covered = []
    for r in scan.iter_rows(named=True):
        for lo, hi in bounds.values():
            if lo + MARGIN <= r["onset"] and r["end"] + MARGIN <= hi:
                covered.append(r)
                break
    scan = pl.DataFrame(covered, schema=scan.schema).sort(pl.col("points").abs(), descending=True)
    for r in scan.head(MAX_SCAN_EVENTS).iter_rows(named=True):
        rows.append(
            dict(
                site=r["site"], onset=r["onset"], end=r["end"],
                source="scan-found, no NGED label (experiment 456)", klass=str(r["class"]),
            )
        )  # fmt: skip
    assert all(r["site"] in sites for r in rows)
    return pl.DataFrame(rows)


def month_for(*, onset: int, end: int, bounds: dict[str, tuple[int, int]]) -> str:
    """Return the first month whose frames cover the event and its margins."""
    for m, (lo, hi) in bounds.items():
        if lo + 48 <= onset - MARGIN and end + MARGIN <= hi:
            return m
    for m, (lo, hi) in bounds.items():
        if lo <= onset - 48 and end + 48 <= hi:
            return m
    for m, (lo, hi) in bounds.items():
        if lo <= onset and end <= hi:
            return m
    raise ValueError((onset, end))


def day_ahead(*, arm: str, month: str, primary: int, lo: int, hi: int) -> pl.DataFrame:
    """Return an arm's day-ahead rows for one Primary between two grid indices."""
    d = pl.read_parquet(PRED / arm / f"{month}_s0_w-1.parquet").filter(
        (pl.col("primary") == primary)
        & pl.col("lead_hours").is_between(*LEAD)
        & pl.col("target").is_between(lo, hi)
    )
    return d.unique(subset="target", keep="first").sort("target")


def figure(*, number: int, ev: dict, sites: list[str], bounds: dict[str, tuple[int, int]]) -> dict:
    """Draw one event's chart and return its summary numbers."""
    site, onset, end = ev["site"], ev["onset"], ev["end"]
    primary = sites.index(site)
    month = month_for(onset=onset, end=end, bounds=bounds)
    lo, hi = onset - MARGIN, end + MARGIN
    frames = {a: day_ahead(arm=a, month=month, primary=primary, lo=lo, hi=hi) for a in ARMS}
    base = frames["K2-S1"]
    lines = [
        base.select(
            "target", median=pl.col("actual") * 100, series=pl.lit("Actual")
        )
    ]
    for a, label in ARMS.items():
        lines.append(
            frames[a].select("target", median=pl.col("q50") * 100, series=pl.lit(label))
        )
    lines_df = pl.concat(lines).with_columns(day=(pl.col("target") - onset) / 48.0)
    band = base.select(
        "target", low=pl.col("q05") * 100, high=pl.col("q95") * 100, lag_real="lag_real"
    ).with_columns(day=(pl.col("target") - onset) / 48.0)
    x_domain = [float(band["day"].min()), float(band["day"].max())]
    x = alt.X(
        "day:Q",
        title="Days since the event (or step) began; shaded: the event",
        scale=alt.Scale(domain=x_domain),
    )
    shade = (
        alt.Chart(pl.DataFrame({"a": [0.0], "b": [min(end - onset, hi - onset) / 48.0]}))
        .mark_rect(color=ocf.GREY_3, opacity=0.35, clip=True)
        .encode(x="a:Q", x2="b:Q")
    )
    area = (
        alt.Chart(band)
        .mark_area(color=ocf.BRAND_ORANGE, opacity=0.18)
        .encode(x=x, y=alt.Y("low:Q", title="Power (% of capacity)"), y2="high:Q")
    )
    line = (
        alt.Chart(lines_df)
        .mark_line(aria=False)
        .encode(
            x=x,
            y="median:Q",
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=alt.Legend(title=None, orient="top", columns=3),
            ),
            strokeWidth=alt.condition(alt.datum.series == "Actual", alt.value(2.0), alt.value(1.2)),
        )
    )
    top = (shade + area + line).properties(width=WIDTH - 110, height=190)
    flagged = band.filter(pl.col("lag_real")).with_columns(day2=pl.col("day") + 1.0 / 48.0)
    strip = (
        alt.Chart(flagged if flagged.height else band.head(0).with_columns(day2=pl.col("day")))
        .mark_rect(color=ocf.GREY_1, clip=True)
        .encode(
            x=alt.X("day:Q", title=None, axis=None, scale=alt.Scale(domain=x_domain)),
            x2="day2:Q",
        )
        .properties(width=WIDTH - 110, height=12)
    )
    err = {}
    for a in ARMS:
        f = frames[a].filter((pl.col("target") >= onset) & (pl.col("target") <= end))
        err[a] = float((f["q50"] - f["actual"]).abs().mean() * 100) if f.height else float("nan")
    nonmirror = ev["source"].startswith("scan")
    title = alt.TitleParams(
        [
            f"Primary {site}, {ev['source']}: day-ahead forecasts (12 to 36 hours ahead), {month} fit.",
            f"Median error during the event, points of capacity: lagless {err['LL']:.1f}, S1 {err['S1']:.1f}, "
            f"B-mix {err['B-mix']:.1f}, K2-S1 {err['K2-S1']:.1f}.",
            "Orange band: K2-S1 90% interval. The event may be a load shift, a level change, or neither."
            if nonmirror
            else "Orange band: K2-S1 90% interval. Real readings.",
            "Grey strip under the plot: forecasts whose lags fall inside a labelled event.",
        ],
        anchor="start",
    )
    chart = alt.vconcat(top, strip, spacing=2).properties(title=title)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "png").mkdir(exist_ok=True)
    name = f"event_{number:02d}_{site}_{onset}"
    chart.save(OUT / f"{name}.svg")
    chart.save(OUT / "png" / f"{name}.png", scale_factor=1.5)
    return dict(number=number, site=site, onset=onset, end=end, month=month, source=ev["source"],
                klass=ev["klass"], n_halfhours=int(band.height),
                **{f"mae_during_{a}": v for a, v in err.items()})  # fmt: skip


def main() -> None:
    """Draw every real-event chart."""
    sites = common.series_table().filter(pl.col("substation_type") == "Primary")["site"].to_list()
    bounds = month_bounds()
    events = pick_events(sites=sites, bounds=bounds)
    out = [
        figure(number=i, ev=ev, sites=sites, bounds=bounds)
        for i, ev in enumerate(events.iter_rows(named=True), start=1)
    ]
    table = pl.DataFrame(out)
    table.write_csv(OUT / "events_summary.csv")
    print(table)


if __name__ == "__main__":
    main()
