# ruff: noqa: E501
"""Experiment 706: the four figures of the write-up, as SVG files under `docs/ml-experiments/00706-*/`.

Run it after `e00706_score.py` and `e00706_s2_score.py`, from the repository root with the
environment of `CLAUDE.md` set:

    uv run --no-sync python ml-experiments/00706-*/e00706_figures.py [--only fig1_lead_bands ...]

The script reads the score tables and the stored predictions under the experiment's private folder
and writes nothing there. Figure 2 needs the event spans of experiment 688, so the folders of
experiments 668, 683, and 688 are put on `sys.path` when that figure is drawn. The script prints
every number a figure shows, and the `svgo` command to run on each SVG before committing it. It never
runs `npx`.

Every figure covers the 14 Primary substations only, which are not NGED metered generators, so
substation names and calendar dates may appear.
"""

import argparse
import datetime as dt
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import altair as alt
import e00706_common as c
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import wrapped
from studies.experiment_store import read_predictions
from studies.sources import ML_EXPERIMENTS_DATA_DIR

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
FIGURE_DIR: Final[Path] = REPO_ROOT / "docs" / "ml-experiments" / f"00706-{c.SLUG}"
"""Where the SVG files go."""
TABLE_DIR: Final[Path] = c.OUT_DIR / "tables"
WIDTH_PX: Final[int] = 680
"""The width of a docs page's text column, as in `studies.charts.CONTENT_WIDTH_PX`."""
FIGURES: Final[tuple[str, ...]] = (
    "fig1_lead_bands",
    "fig2_events",
    "fig3_examples",
    "fig4_copied_share",
)

BANDS: Final[tuple[str, ...]] = ("day 1", "day 2", "days 3 to 4", "days 5 to 7", "days 8 to 10")
"""The lead bands of the score tables, in order."""
STAGE1_ARMS: Final[tuple[str, ...]] = ("G-lb", "D", "R0-168", "RR")
STAGE2_ARMS: Final[tuple[str, ...]] = ("N-rule", "N-flag real", "N-gate real")
ARM_COLOURS: Final[dict[str, str]] = {
    "G-lb": ocf.DATA_BLUE,
    "D": ocf.DATA_SKY,
    "R0-168": ocf.DATA_PURPLE,
    "RR": ocf.DATA_GREEN,
    "N-rule": ocf.DATA_PURPLE_LIGHT,
    "N-flag real": ocf.BRAND_ORANGE_LIGHT,
    "N-gate real": ocf.BRAND_ORANGE,
}
"""Main data colours and their light shades only, as the `dataviz` skill requires for a published chart."""
ARM_LABELS: Final[dict[str, str]] = {
    "G-lb": "G-lb",
    "D": "D",
    "R0-168": "R0-168",
    "RR": "RR (recipe R)",
    "N-rule": "N-rule (post hoc)",
    "N-flag real": "N-flag, real graph (post hoc)",
    "N-gate real": "N-gate, real graph (post hoc)",
}
POOLED: Final[str] = "pooled 14"

STAGE2_START: Final[dt.datetime] = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
"""The first valid time of the stage 2 row set (origins from 2026-01-01)."""
DAYS_BEFORE: Final[int] = 14
DAYS_AFTER: Final[int] = 7
MAX_EVENTS: Final[int] = 8
EXAMPLE_LABEL: Final[str] = "D5"
EXAMPLE_ORIGIN: Final[dt.datetime] = dt.datetime(2026, 3, 10, 12, tzinfo=dt.UTC)
"""The Primary and the 12:00 UTC origin that figure 3 draws, a signed-MW Primary in March 2026."""

FORECAST_COLOURS: Final[dict[str, str]] = {
    "observed (cleaned power)": ocf.BLACK_1,
    "stored normal-running median": ocf.BRAND_ORANGE,
    "G-lb corrected median": ocf.DATA_BLUE,
    "N-gate (real graph) corrected median (post hoc)": ocf.DATA_GREEN,
}
BAND_COLOUR: Final[str] = ocf.BRAND_ORANGE_LIGHT
THREE_ARM_SETS: Final[dict[str, str]] = {
    "stored normal-running median": "nra",
    "G-lb corrected median": "g",
    "N-gate (real graph) corrected median (post hoc)": "n-gate-real",
}
"""The forecast shown by each label of figure 2, and the prediction set that holds it (`nra` is the stored backtest file)."""

COPIED_VARIANTS: Final[tuple[str, ...]] = (
    "S-NRA re-run",
    "N-rule",
    "N-flag real",
    "N-gate real",
    "N-gate real zero neighbours",
)
BANK_COLUMN: Final[str] = "share: events of depth at least 3%"
BANK_EVENTS_COLUMN: Final[str] = "events: events of depth at least 3%"


# --------------------------------------------------------------------------- helpers


def read_table(*, name: str) -> pl.DataFrame:
    """Read a score table that the score scripts wrote, or stop with a clear message.

    Args:
        name: The file name without `.csv`.

    Returns:
        The table.

    Raises:
        SystemExit: If the file is missing, naming the file and the script that writes it.
    """
    path = TABLE_DIR / f"{name}.csv"
    if not path.exists():
        existing = sorted(p.name for p in TABLE_DIR.glob("*.csv")) if TABLE_DIR.exists() else []
        msg = (
            f"missing table {path}. Run e00706_score.py (stage 1 tables) or e00706_s2_score.py "
            f"(tables starting `s2_`) first. Tables found: {existing}"
        )
        raise SystemExit(msg)
    return pl.read_csv(path)


def require_columns(*, frame: pl.DataFrame, columns: Sequence[str], table: str) -> None:
    """Stop with a clear message when a table lacks a column that a figure reads."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        msg = f"table {table}.csv lacks columns {missing}; it has {frame.columns}"
        raise SystemExit(msg)


def title_params(*, takeaway: str, subtitle: str) -> alt.TitleParams:
    """Return a title whose first line group states the takeaway, wrapped to the plot width."""
    return alt.TitleParams(
        text=wrapped(text=takeaway, width=70),
        subtitle=wrapped(text=subtitle, width=95),
        anchor="start",
        offset=10,
        subtitleColor=ocf.BLACK_1,
        subtitlePadding=6,
        fontSize=ocf.font_size(style="Body Large", body_px=11),
        subtitleFontSize=ocf.font_size(style="Body", body_px=11),
    )


def finish(*, chart: alt.TopLevelMixin, name: str) -> Path:
    """Write the chart as SVG and print the `svgo` command to run on it."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.svg"
    chart.configure_view(stroke=None).configure_legend(  # ty: ignore[unresolved-attribute]
        orient="bottom", direction="horizontal"
    ).save(path)
    print(f"wrote {path}")
    print(f"  next: npx svgo@4 --multipass --precision=1 --final-newline {path}")
    return path


def print_frame(*, frame: pl.DataFrame, heading: str) -> None:
    """Print a table in full, so every plotted number is in the log."""
    print(f"\n== {heading}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=80, float_precision=4):
        print(frame)


# --------------------------------------------------------------------------- figure 1


def lead_band_differences(*, table: str, arms: Sequence[str], stage: str) -> pl.DataFrame:
    """Return the pooled-14 CRPS difference from S-NRA of each arm by lead band, long format."""
    frame = read_table(name=table)
    columns = [f"{band} minus S-NRA" for band in BANDS]
    require_columns(frame=frame, columns=["arm", "group", *BANDS, *columns], table=table)
    pooled = frame.filter(pl.col("group") == POOLED)
    missing = [arm for arm in (*arms, "S-NRA") if arm not in pooled["arm"].to_list()]
    if missing:
        msg = f"table {table}.csv has no pooled-14 row for {missing}"
        raise SystemExit(msg)
    levels = pooled.filter(pl.col("arm") == "S-NRA").select(*BANDS)
    print_frame(
        frame=levels, heading=f"{table}: S-NRA CRPS level on its own rows (points of capacity)"
    )
    return (
        pooled.filter(pl.col("arm").is_in(list(arms)))
        .select("arm", *columns)
        .unpivot(index="arm", variable_name="band_column", value_name="difference")
        .with_columns(
            band=pl.col("band_column").str.replace(" minus S-NRA", ""),
            stage=pl.lit(stage),
        )
        .drop("band_column")
    )


def figure_lead_bands() -> Path:
    """Draw figure 1: CRPS difference from the stored forecast by lead band, one line per arm."""
    stage1 = lead_band_differences(
        table="lead_bands_event_free", arms=STAGE1_ARMS, stage="stage 1: origins 2025-12 to 2026-06"
    )
    stage2 = lead_band_differences(
        table="s2_lead_bands_event_free",
        arms=STAGE2_ARMS,
        stage="stage 2, post hoc: origins 2026-01 to 2026-06",
    )
    frame = pl.concat([stage1, stage2]).with_columns(
        arm_label=pl.col("arm").replace_strict(ARM_LABELS, return_dtype=pl.String)
    )
    print_frame(
        frame=frame.sort("stage", "arm", "band"),
        heading="figure 1: CRPS minus S-NRA, points of capacity (negative is better), pooled 14, event-free real rows",
    )
    colour = alt.Color(
        "arm_label:N",
        title=None,
        scale=alt.Scale(
            domain=[ARM_LABELS[a] for a in (*STAGE1_ARMS, *STAGE2_ARMS)],
            range=[ARM_COLOURS[a] for a in (*STAGE1_ARMS, *STAGE2_ARMS)],
        ),
    )
    stage_domain = list(frame["stage"].unique(maintain_order=True))
    dash = alt.StrokeDash(
        "stage:N",
        title=None,
        scale=alt.Scale(domain=stage_domain, range=[[1, 0], [6, 4]]),
    )
    x = alt.X(
        "band:N",
        sort=list(BANDS),
        title="Lead time of the forecast",
        axis=alt.Axis(labelAngle=0),
    )
    y = alt.Y(
        "difference:Q",
        title="CRPS minus stored forecast's (points of capacity; below zero is better)",
    )
    base = alt.Chart(frame)
    lines = base.mark_line(strokeWidth=2).encode(x=x, y=y, color=colour, strokeDash=dash)  # ty: ignore[unresolved-attribute]
    points = base.mark_point(filled=True, size=45).encode(x=x, y=y, color=colour)  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"zero": [0.0]}))
        .mark_rule(color=ocf.BLACK_1, strokeWidth=1)
        .encode(y="zero:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = (zero + lines + points).properties(
        width=WIDTH_PX - 80,
        height=300,
        title=title_params(
            takeaway="Every arm lowers CRPS on day 1 and day 2, and every arm except R0-168 raises it by days 8 to 10.",
            subtitle=(
                "CRPS of each arm minus CRPS of the stored normal-running forecast (S-NRA) on the same rows, "
                "event-free real rows, 14 Primaries pooled. The stage 2 arms are post hoc. Stage 1 scores "
                "origins from 2025-12 and stage 2 from 2026-01, so the two sets of rows differ and each arm "
                "is differenced against S-NRA on its own rows."
            ),
        ),
    )
    return finish(chart=chart, name="fig1_lead_bands")


# --------------------------------------------------------------------------- figure 2


def import_event_spans() -> object:
    """Import `e00688_diagnose` with the folders of experiments 668, 683, and 688 on `sys.path`."""
    for number in ("00668", "00683", "00688"):
        for folder in sorted((REPO_ROOT / "ml-experiments").glob(f"{number}-*")):
            if str(folder) not in sys.path:
                sys.path.append(str(folder))
    import e00688_diagnose as diagnose  # ty: ignore[unresolved-import]

    return diagnose


def event_windows(*, count: int) -> pl.DataFrame:
    """Return up to `count` events across both metering groups, each at its own Primary.

    The candidates are the spans of `event_spans(kind="primary")`: the labelled events and the
    verified transfers. One span is kept per verified transfer and per Primary, spans whose window
    lies inside the stage 2 rows come first, and the groups are taken in turn so that both appear.
    """
    diagnose = import_event_spans()
    spans = diagnose.event_spans(kind="primary")  # ty: ignore[unresolved-attribute]
    grid_start: dt.datetime = diagnose.lib.GRID_START  # ty: ignore[unresolved-attribute]
    site = c.read_site_map().select("label", "time_series_id", "capacity_mw", "group")
    candidates = (
        spans.join(site, on="time_series_id", how="inner")
        .with_columns(
            start_time=pl.lit(grid_start) + pl.duration(minutes=30 * pl.col("start")),
            end_time=pl.lit(grid_start) + pl.duration(minutes=30 * pl.col("end")),
        )
        .with_columns(
            duration_days=(pl.col("end_time") - pl.col("start_time")).dt.total_hours() / 24,
            covers_stage2=pl.col("start_time") - pl.duration(days=DAYS_BEFORE) >= STAGE2_START,
        )
        .filter(
            pl.col("start_time") + pl.duration(days=DAYS_AFTER)
            <= pl.lit(dt.datetime(2026, 7, 1, tzinfo=dt.UTC))
        )
        .sort(["covers_stage2", "duration_days"], descending=True)
        .unique(subset=["source"], keep="first", maintain_order=True)
        .unique(subset=["time_series_id"], keep="first", maintain_order=True)
        .sort(["covers_stage2", "duration_days"], descending=True)
    )
    by_group = {
        group: candidates.filter(pl.col("group") == group)
        for group in candidates["group"].unique(maintain_order=True).to_list()
    }
    chosen: list[pl.DataFrame] = []
    position = 0
    while len(chosen) < count and any(position < len(frame) for frame in by_group.values()):
        for frame in by_group.values():
            if position < len(frame) and len(chosen) < count:
                chosen.append(frame.slice(position, 1))
        position += 1
    if not chosen:
        msg = "no event of event_spans(kind='primary') lies in the scored period"
        raise SystemExit(msg)
    return pl.concat(chosen)


def window_rows(
    *, series: int, start: dt.datetime, end: dt.datetime, capacity: float
) -> pl.DataFrame:
    """Return observed power and the three forecasts of one Primary over a window, in percent of capacity.

    Each forecast is stitched over daily origins: the row at each valid time comes from the latest
    origin before it, so its lead is 0 to 24 hours (day-ahead rows).
    """
    columns = ["valid_time", "power_fcst_init_time", "q50"]
    frames: list[pl.DataFrame] = []
    observed = (
        pl.scan_parquet(c.BACKTEST_DIR / "ground_truth_primaries.parquet")
        .filter(
            pl.col("time_series_id") == series,
            pl.col("valid_time") > start,
            pl.col("valid_time") <= end,
            pl.col("power").is_not_null(),
        )
        .select("valid_time", value=pl.col("power").cast(pl.Float64) * 100.0 / capacity)
        .with_columns(forecast=pl.lit("observed (cleaned power)"))
        .collect()
    )
    frames.append(observed)
    for label, source in THREE_ARM_SETS.items():
        if source == "nra":
            lazy = (
                pl.scan_parquet(c.BACKTEST_DIR / "primaries.parquet")
                .filter(pl.col("NRA_or_prevailing") == "NRA")
                .select("time_series_id", *columns)
            )
        else:
            lazy = stored_predictions(name=source).lazy().select("time_series_id", *columns)
        frames.append(
            lazy.filter(
                pl.col("time_series_id") == series,
                pl.col("valid_time") > start,
                pl.col("valid_time") <= end,
                pl.col("valid_time") > pl.col("power_fcst_init_time"),
                pl.col("valid_time") <= pl.col("power_fcst_init_time") + pl.duration(hours=24),
            )
            .select("valid_time", value=pl.col("q50").cast(pl.Float64) * 100.0 / capacity)
            .with_columns(forecast=pl.lit(label))
            .collect()
        )
    return pl.concat(frames)


_STORED: dict[str, pl.DataFrame] = {}


def stored_predictions(*, name: str) -> pl.DataFrame:
    """Read one stored prediction set of this experiment once, using the row set in its manifest."""
    if name not in _STORED:
        manifest = c.OUT_DIR / "predictions" / name / "manifest.json"
        if not manifest.exists():
            msg = f"missing prediction set {manifest.parent}; run the fit and score scripts first"
            raise SystemExit(msg)
        row_set_id = json.loads(manifest.read_text())["row_set_id"]
        _STORED[name] = read_predictions(
            experiment_id=c.EXPERIMENT_ID, name=name, row_set_id=row_set_id
        )
    return _STORED[name]


def figure_events() -> Path:
    """Draw figure 2: observed power and three forecasts around the start of real events."""
    events = event_windows(count=MAX_EVENTS)
    panels: list[alt.LayerChart] = []
    summary: list[dict[str, object]] = []
    for row in events.iter_rows(named=True):
        start: dt.datetime = row["start_time"]
        end: dt.datetime = row["end_time"]
        data = window_rows(
            series=row["time_series_id"],
            start=start - dt.timedelta(days=DAYS_BEFORE),
            end=start + dt.timedelta(days=DAYS_AFTER),
            capacity=row["capacity_mw"],
        ).with_columns(day=(pl.col("valid_time") - pl.lit(start)).dt.total_seconds() / 86400.0)
        title = f"{row['label']} ({row['group']}): {row['source']}, starts {start:%Y-%m-%d}"
        end_day = min((end - start).total_seconds() / 86400.0, float(DAYS_AFTER))
        wide = data.pivot(
            on="forecast", index="day", values="value", aggregate_function="first"
        ).drop_nulls("observed (cleaned power)")
        after = wide.filter(pl.col("day") >= 0)
        errors = {
            label: after.select(
                (pl.col(label) - pl.col("observed (cleaned power)")).abs().mean()
            ).item()
            for label in THREE_ARM_SETS
            if label in after.columns
        }
        summary.append(
            {
                "label": row["label"],
                "group": row["group"],
                "source": row["source"],
                "start": start,
                "end": end,
                "duration_days": row["duration_days"],
                **{f"mae after start: {k}": v for k, v in errors.items()},
            }
        )
        print_frame(frame=data.sort("forecast", "day"), heading=f"figure 2 series: {title}")
        shade = (
            alt.Chart(pl.DataFrame({"from": [0.0], "to": [end_day]}))
            .mark_rect(color=ocf.GREY_3, opacity=0.6)
            .encode(x="from:Q", x2="to:Q")  # ty: ignore[unresolved-attribute]
        )
        rules = (
            alt.Chart(pl.DataFrame({"day": [0.0, end_day] if end_day < DAYS_AFTER else [0.0]}))
            .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3])
            .encode(x="day:Q")  # ty: ignore[unresolved-attribute]
        )
        lines = (
            alt.Chart(data)
            .mark_line(clip=True, strokeWidth=1.3)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "day:Q",
                    title="Days from the start of the event",
                    scale=alt.Scale(domain=[-DAYS_BEFORE, DAYS_AFTER]),
                    axis=alt.Axis(values=[-14, -7, 0, 7]),
                ),
                y=alt.Y("value:Q", title="% of capacity", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "forecast:N",
                    title=None,
                    scale=alt.Scale(
                        domain=list(FORECAST_COLOURS), range=list(FORECAST_COLOURS.values())
                    ),
                ),
            )
        )
        panels.append(
            (shade + lines + rules).properties(
                width=WIDTH_PX - 80,
                height=110,
                title=alt.TitleParams(title, anchor="start", fontSize=11),
            )
        )
    print_frame(
        frame=pl.DataFrame(summary),
        heading="figure 2: events shown and mean absolute error after the start (% of capacity)",
    )
    chart = (
        alt.vconcat(*panels, spacing=18)
        .properties(
            title=title_params(
                takeaway="After a load shift starts, the stored forecast and both corrected forecasts keep forecasting the old level.",
                subtitle=(
                    "Observed cleaned power and the day-ahead median (leads 0 to 24 h, stitched over daily 12:00 UTC "
                    "origins) of the stored normal-running forecast, the G-lb corrected forecast, and the N-gate "
                    "corrected forecast (real graph, post hoc, from 2026-01 only), in percent of capacity, for "
                    "14 days before to 7 days after the start of a labelled event or verified transfer at a "
                    "Primary. The grey band runs from the event's start to its end, clipped at day 7."
                ),
            )
        )
        .resolve_scale(color="shared")
    )
    return finish(chart=chart, name="fig2_events")


# --------------------------------------------------------------------------- figure 3


def overlaps_event(*, series: int, start: dt.datetime, end: dt.datetime) -> bool:
    """Whether a labelled event or verified transfer of the Primary overlaps the window."""
    diagnose = import_event_spans()
    spans = diagnose.event_spans(kind="primary")  # ty: ignore[unresolved-attribute]
    grid_start: dt.datetime = diagnose.lib.GRID_START  # ty: ignore[unresolved-attribute]
    start_index = int((start - grid_start).total_seconds() // 1800)
    end_index = int((end - grid_start).total_seconds() // 1800)
    hits = spans.filter(
        (pl.col("time_series_id") == series)
        & (pl.col("start") < end_index)
        & (pl.col("end") > start_index)
    )
    return len(hits) > 0


def example_rows(*, series: int, origin: dt.datetime, capacity: float) -> pl.DataFrame:
    """Return the one forecast issued at `origin` of three kinds, and the observed power, in % of capacity."""
    end = origin + dt.timedelta(days=14)
    forecasts: list[pl.DataFrame] = []
    for label, columns in (
        ("stored normal-running q05", "q05"),
        ("stored normal-running q95", "q95"),
        ("stored normal-running median", "q50"),
    ):
        forecasts.append(
            pl.scan_parquet(c.BACKTEST_DIR / "primaries.parquet")
            .filter(
                pl.col("NRA_or_prevailing") == "NRA",
                pl.col("time_series_id") == series,
                pl.col("power_fcst_init_time") == origin,
                pl.col("valid_time") > origin,
            )
            .select("valid_time", value=pl.col(columns).cast(pl.Float64) * 100.0 / capacity)
            .with_columns(forecast=pl.lit(label))
            .collect()
        )
    forecasts.append(
        stored_predictions(name="g")
        .filter(
            pl.col("time_series_id") == series,
            pl.col("power_fcst_init_time") == origin,
            pl.col("valid_time") > origin,
        )
        .select("valid_time", value=pl.col("q50").cast(pl.Float64) * 100.0 / capacity)
        .with_columns(forecast=pl.lit("G-lb corrected median"))
    )
    forecasts.append(
        pl.scan_parquet(c.BACKTEST_DIR / "ground_truth_primaries.parquet")
        .filter(
            pl.col("time_series_id") == series,
            pl.col("valid_time") > origin,
            pl.col("valid_time") <= end,
            pl.col("power").is_not_null(),
        )
        .select("valid_time", value=pl.col("power").cast(pl.Float64) * 100.0 / capacity)
        .with_columns(forecast=pl.lit("observed (cleaned power)"))
        .collect()
    )
    return pl.concat(forecasts).with_columns(
        hours=(pl.col("valid_time") - pl.lit(origin)).dt.total_seconds() / 3600.0
    )


def example_panel(
    *, data: pl.DataFrame, hours: float, title: str, unit_days: bool
) -> alt.LayerChart:
    """Draw one panel of figure 3: the band, the two medians, and the observed power."""
    window = data.filter(pl.col("hours") <= hours)
    scale = 24.0 if unit_days else 1.0
    window = window.with_columns(x=pl.col("hours") / scale)
    band = window.filter(pl.col("forecast").str.contains("q05|q95")).pivot(
        on="forecast", index="x", values="value"
    )
    lines = window.filter(~pl.col("forecast").str.contains("q05|q95"))
    x = alt.X(
        "x:Q",
        title="Days after the forecast origin" if unit_days else "Hours after the forecast origin",
        scale=alt.Scale(domain=[0, hours / scale]),
        axis=alt.Axis(values=list(range(0, 15, 2)) if unit_days else [0, 6, 12, 18, 24]),
    )
    area = (
        alt.Chart(band)
        .mark_area(color=BAND_COLOUR, opacity=0.6)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y(
                "stored normal-running q05:Q", title="% of capacity", scale=alt.Scale(zero=False)
            ),
            y2="stored normal-running q95:Q",
        )
    )
    line = (
        alt.Chart(lines)
        .mark_line(strokeWidth=1.5)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("value:Q", title="% of capacity", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "forecast:N",
                title=None,
                scale=alt.Scale(
                    domain=[k for k in FORECAST_COLOURS if "N-gate" not in k],
                    range=[v for k, v in FORECAST_COLOURS.items() if "N-gate" not in k],
                ),
            ),
        )
    )
    return (area + line).properties(
        width=WIDTH_PX - 80, height=200, title=alt.TitleParams(title, anchor="start", fontSize=11)
    )


def figure_examples(*, label: str = EXAMPLE_LABEL, origin: dt.datetime = EXAMPLE_ORIGIN) -> Path:
    """Draw figure 3: one forecast of one Primary, over its first 24 hours and over all 14 days."""
    site = c.read_site_map().filter(pl.col("label") == label)
    if site.is_empty():
        msg = f"no Primary labelled {label!r} in the site map"
        raise SystemExit(msg)
    series, capacity, group = site["time_series_id"][0], site["capacity_mw"][0], site["group"][0]
    if overlaps_event(series=series, start=origin, end=origin + dt.timedelta(days=14)):
        print(
            f"WARNING: {label} has a labelled event or verified transfer within 14 days of {origin}; pick another origin with --origin"
        )
    data = example_rows(series=series, origin=origin, capacity=capacity)
    print_frame(
        frame=data.sort("forecast", "hours"),
        heading=f"figure 3: {label}, origin {origin:%Y-%m-%d %H:%M} UTC, % of capacity",
    )
    day_ahead = example_panel(
        data=data, hours=24.0, title="Day ahead (leads 0 to 24 hours)", unit_days=False
    )
    full = example_panel(data=data, hours=336.0, title="Full 14-day horizon", unit_days=True)
    chart = (
        alt.vconcat(day_ahead, full, spacing=24)
        .properties(
            title=title_params(
                takeaway="The G-lb correction moves the median by a fraction of the stored forecast's q05 to q95 band.",
                subtitle=(
                    f"One forecast of Primary {label} ({group}), issued at 12:00 UTC on {origin:%Y-%m-%d}, against "
                    "observed cleaned power, in percent of capacity. Band: q05 to q95 of the stored "
                    "normal-running forecast. Orange line: its median. Blue line: the median of the G-lb "
                    "corrected forecast. The origin is chosen as an event-free example, not as the best case."
                ),
            )
        )
        .resolve_scale(color="shared")
    )
    return finish(chart=chart, name="fig3_examples")


# --------------------------------------------------------------------------- figure 4


def figure_copied_share() -> Path:
    """Draw figure 4: the copied share of an injected shift, by arm."""
    frame = read_table(name="s2_bank_share_and_crps")
    require_columns(
        frame=frame,
        columns=["variant", BANK_COLUMN, BANK_EVENTS_COLUMN],
        table="s2_bank_share_and_crps",
    )
    shown = (
        frame.filter(pl.col("variant").is_in(COPIED_VARIANTS))
        .select("variant", events=BANK_EVENTS_COLUMN, share=pl.col(BANK_COLUMN) * 100.0)
        .with_columns(
            variant_label=pl.col("variant").replace(
                {
                    "S-NRA re-run": "S-NRA re-run (no look-back)",
                    "N-rule": "N-rule (no network)",
                    "N-flag real": "N-flag, real graph",
                    "N-gate real": "N-gate, real graph",
                    "N-gate real zero neighbours": "N-gate, zero neighbours",
                }
            )
        )
    )
    missing = set(COPIED_VARIANTS) - set(shown["variant"].to_list())
    if missing:
        msg = f"s2_bank_share_and_crps.csv has no row for {sorted(missing)}"
        raise SystemExit(msg)
    shown = shown.with_columns(
        order=pl.col("variant").replace_strict(
            {v: i for i, v in enumerate(COPIED_VARIANTS)}, return_dtype=pl.Int64
        )
    ).sort("order")
    print_frame(
        frame=shown,
        heading="figure 4: copied share of an injected shift (% of the shift), events of depth at least 3%",
    )
    labels = shown["variant_label"].to_list()
    colours = {
        "S-NRA re-run (no look-back)": ocf.DATA_SKY,
        "N-rule (no network)": ocf.DATA_PURPLE_LIGHT,
        "N-flag, real graph": ocf.BRAND_ORANGE_LIGHT,
        "N-gate, real graph": ocf.BRAND_ORANGE,
        "N-gate, zero neighbours": ocf.DATA_BLUE,
    }
    bars = (
        alt.Chart(shown)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("variant_label:N", sort=labels, title=None),
            x=alt.X(
                "share:Q",
                title="Share of the injected shift that the forecast copies (%; smaller is better)",
                scale=alt.Scale(domain=[0, 30]),
            ),
            color=alt.Color(
                "variant_label:N",
                legend=None,
                scale=alt.Scale(domain=labels, range=[colours[label] for label in labels]),
            ),
        )
    )
    text = (
        alt.Chart(shown)
        .mark_text(align="left", dx=4, color=ocf.BLACK_1)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("variant_label:N", sort=labels),
            x="share:Q",
            text=alt.Text("share:Q", format=".1f"),
        )
    )
    n_events = int(shown["events"][0])
    chart = (bars + text).properties(
        width=WIDTH_PX - 230,
        height=170,
        title=title_params(
            takeaway="The learned gate copies less of an injected shift than the rule, but only when it sees real neighbours.",
            subtitle=(
                f"Mechanism check, post hoc: {n_events} injected level shifts of depth at least 3% of capacity "
                "(bank events), each with the mirrored step put on the real-graph neighbours only. Bar: share "
                "of the shift that the forecast copies when it should not. N-gate with zero neighbours gets no "
                "neighbour information."
            ),
        ),
    )
    return finish(chart=chart, name="fig4_copied_share")


# --------------------------------------------------------------------------- main


def main() -> None:
    """Draw the figures named by `--only`, or all four."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--label", default=EXAMPLE_LABEL, help="Primary of figure 3, such as D5")
    parser.add_argument(
        "--origin",
        default=EXAMPLE_ORIGIN.strftime("%Y-%m-%d"),
        help="origin day of figure 3 (12:00 UTC), YYYY-MM-DD, within 2026-01-01 to 2026-06-16",
    )
    args = parser.parse_args()
    origin = dt.datetime.fromisoformat(args.origin).replace(hour=12, tzinfo=dt.UTC)
    if ML_EXPERIMENTS_DATA_DIR is None or not c.OUT_DIR.exists():
        msg = f"experiment folder {c.OUT_DIR} is missing; set DATA_PATH_INTERNAL"
        raise SystemExit(msg)
    if "fig1_lead_bands" in args.only:
        figure_lead_bands()
    if "fig2_events" in args.only:
        figure_events()
    if "fig3_examples" in args.only:
        figure_examples(label=args.label, origin=origin)
    if "fig4_copied_share" in args.only:
        figure_copied_share()


if __name__ == "__main__":
    main()
