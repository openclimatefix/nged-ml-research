# ruff: noqa: E501
"""Experiment 706: the four figures of the write-up, as SVG files under
`docs/ml-experiments/00706-*/`.

Run it after `e00706_score.py` and `e00706_s2_score.py`, from the repository root with the
environment of `CLAUDE.md` set:

    uv run --no-sync python ml-experiments/00706-*/e00706_figures.py [--only fig1_lead_bands ...]

The script reads the score tables and the stored predictions under the experiment's private folder
and writes nothing there. Figure 2 needs the event spans of experiment 688, so the folders of
experiments 668, 683, and 688 are put on `sys.path` when that figure is drawn. The script prints
every number a figure shows, and the `svgo` command to run on each SVG before committing it. It
never
runs `npx`.

Every figure covers the 14 Primary substations only, which are not NGED metered generators, so
substation names and calendar dates may appear. Each figure also gets a PNG preview in a scratch
folder whose path the script prints; the previews are not written under `docs/`.

The palette of figure 1 has not been run through `validate_palette.py`, because the bundled
`dataviz` skill in this checkout does not provide that script.
"""

import argparse
import datetime as dt
import functools
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Final

import altair as alt
import e00706_common as c
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import wrapped
from studies.experiment_store import HOLDOUT_START, PREDICTIONS_NAME

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
ARM_SHAPES: Final[dict[str, str]] = {
    "G-lb": "circle",
    "D": "square",
    "R0-168": "diamond",
    "RR": "triangle-up",
    "N-rule": "triangle-down",
    "N-flag real": "triangle-right",
    "N-gate real": "triangle-left",
}
"""A point shape per arm, as a backup to the colour."""
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

STAGE2_START: Final[dt.datetime] = dt.datetime(2026, 1, 1, 12, tzinfo=dt.UTC)
"""The first origin of the stage 2 row set; a day-ahead curve exists from this time on."""
STAGE1_START: Final[dt.datetime] = dt.datetime(2025, 12, 1, tzinfo=dt.UTC)
"""The first day of the stage 1 row set."""
HALF_HOUR: Final[str] = "30m"
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
SET_ROW_SET_IDS: Final[dict[str, str]] = {
    "g": c.ROW_SET_ID,
    "n-gate-real": c.ROW_SET_ID_FROM_JANUARY,
}
"""The row set that each stored prediction set of figures 2 and 3 must cover."""
SHADE_THIS: Final[str] = "event plotted in this panel"
SHADE_OTHER: Final[str] = "other events or transfers at this Primary"
OBSERVED_LABEL: Final[str] = "observed (cleaned power)"
STORED_MEDIAN: Final[str] = "stored normal-running median"
G_MEDIAN: Final[str] = "G-lb corrected median"

COPIED_VARIANTS: Final[tuple[str, ...]] = (
    "S-NRA re-run",
    "N-rule",
    "N-flag real",
    "N-gate real",
    "N-gate real zero neighbours",
    "N-gate rewired",
    "N-gate rewired zero neighbours",
)
OPTIONAL_VARIANTS: Final[frozenset[str]] = frozenset({"N-gate rewired zero neighbours"})
"""Variants that the figure draws when the table holds them and skips otherwise."""
VARIANT_LABELS: Final[dict[str, str]] = {
    "S-NRA re-run": "S-NRA re-run (no look-back)",
    "N-rule": "N-rule (no network)",
    "N-flag real": "N-flag, real graph",
    "N-gate real": "N-gate, real graph",
    "N-gate real zero neighbours": "N-gate, real graph, zero neighbours",
    "N-gate rewired": "N-gate, rewired graph",
    "N-gate rewired zero neighbours": "N-gate, rewired graph, zero neighbours",
}
VARIANT_FAMILIES: Final[dict[str, str]] = {
    "S-NRA re-run": "stored forecast",
    "N-rule": "rule",
    "N-flag real": "flag",
    "N-gate real": "gate",
    "N-gate real zero neighbours": "gate",
    "N-gate rewired": "gate",
    "N-gate rewired zero neighbours": "gate",
}
FAMILY_COLOURS: Final[dict[str, str]] = {
    "stored forecast": ocf.DATA_SKY,
    "rule": ocf.DATA_PURPLE,
    "flag": ocf.BRAND_ORANGE,
    "gate": ocf.DATA_GREEN,
}
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


@functools.cache
def preview_dir() -> Path:
    """Return a scratch folder for PNG previews, created once per run."""
    return Path(tempfile.mkdtemp(prefix="e00706_figure_previews_"))


def finish(*, chart: alt.TopLevelMixin, name: str) -> Path:
    """Write the chart as SVG, write a PNG preview to a scratch folder, and print the `svgo`
    command."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.svg"
    configured = chart.configure_view(stroke=None).configure_legend(  # ty: ignore[unresolved-attribute]
        orient="top", direction="horizontal"
    )
    configured.save(path)
    preview = preview_dir() / f"{name}.png"
    configured.save(preview, scale_factor=2.0)
    print(f"wrote {path}")
    print(f"  PNG preview: {preview}")
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


def join_names(*, names: Sequence[str]) -> str:
    """Join names with commas and a serial `and`."""
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def lead_band_takeaway(*, frame: pl.DataFrame) -> str:
    """Build figure 1's title from the plotted differences, so a re-score cannot leave it stale."""
    arms = frame["arm_label"].unique(maintain_order=True).to_list()
    early = (
        frame.filter(pl.col("band").is_in(["day 1", "day 2"]))
        .group_by("arm_label")
        .agg(below=(pl.col("difference") < 0).all())
    )
    late = frame.filter(pl.col("band") == "days 8 to 10").select(
        "arm_label", above=pl.col("difference") > 0
    )
    lowers = [a for a in arms if a in early.filter(pl.col("below"))["arm_label"].to_list()]
    raises = [a for a in arms if a in late.filter(pl.col("above"))["arm_label"].to_list()]
    early_text = (
        f"All {len(arms)} arms lower"
        if len(lowers) == len(arms)
        else f"{join_names(names=lowers)} lower"
        if lowers
        else "No arm lowers"
    )
    late_text = (
        f"all {len(arms)} raise it"
        if len(raises) == len(arms)
        else f"{join_names(names=raises)} raise it"
        if raises
        else "no arm raises it"
    )
    return f"{early_text} CRPS on day 1 and day 2, and {late_text} by days 8 to 10."


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
    all_arms = (*STAGE1_ARMS, *STAGE2_ARMS)
    colour = alt.Color(
        "arm_label:N",
        title=None,
        scale=alt.Scale(
            domain=[ARM_LABELS[a] for a in all_arms], range=[ARM_COLOURS[a] for a in all_arms]
        ),
    )
    shape = alt.Shape(
        "arm_label:N",
        title=None,
        scale=alt.Scale(
            domain=[ARM_LABELS[a] for a in all_arms], range=[ARM_SHAPES[a] for a in all_arms]
        ),
    )
    takeaway = lead_band_takeaway(frame=frame)
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
        title=["CRPS minus the stored forecast's", "(points of capacity; below zero is better)"],
    )
    base = alt.Chart(frame)
    lines = base.mark_line(strokeWidth=2).encode(x=x, y=y, color=colour, strokeDash=dash)  # ty: ignore[unresolved-attribute]
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=x, y=y, color=colour, shape=shape
    )
    zero_frame = pl.DataFrame({"zero": [0.0]})
    zero = (
        alt.Chart(zero_frame).mark_rule(color=ocf.BLACK_1, strokeWidth=1).encode(y="zero:Q")  # ty: ignore[unresolved-attribute]
    )
    zero_label = (
        alt.Chart(zero_frame)
        .mark_text(align="right", dy=-7, color=ocf.BLACK_1)
        .encode(  # ty: ignore[unresolved-attribute]
            y="zero:Q", x=alt.value(WIDTH_PX - 80), text=alt.value("same as stored forecast")
        )
    )
    chart = (zero + zero_label + lines + points).properties(
        width=WIDTH_PX - 80,
        height=300,
        title=title_params(
            takeaway=takeaway,
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


def import_event_spans() -> ModuleType:
    """Import `e00688_diagnose` with the folders of experiments 668, 683, and 688 on `sys.path`."""
    for number in ("00668", "00683", "00688"):
        for folder in sorted((REPO_ROOT / "ml-experiments").glob(f"{number}-*")):
            if str(folder) not in sys.path:
                sys.path.append(str(folder))
    import e00688_diagnose as diagnose  # ty: ignore[unresolved-import]

    return diagnose


def primary_spans() -> pl.DataFrame:
    """Return every labelled event and verified transfer at the 14 pooled Primaries, with times.

    The candidates are the spans of `event_spans(kind="primary")`. The join with the site map keeps
    only the signed-MW and MVA-only Primaries, so the two short-history Primaries D15 and D18 never
    appear.
    """
    diagnose = import_event_spans()
    spans = diagnose.event_spans(kind="primary")  # ty: ignore[unresolved-attribute]
    grid_start: dt.datetime = diagnose.lib.GRID_START  # ty: ignore[unresolved-attribute]
    site = (
        c.read_site_map()
        .filter(pl.col("group").is_in([c.GROUP_SIGNED, c.GROUP_MVA]))
        .select("label", "time_series_id", "capacity_mw", "group")
    )
    return (
        spans.join(site, on="time_series_id", how="inner")
        .with_columns(
            start_time=pl.lit(grid_start) + pl.duration(minutes=30 * pl.col("start")),
            end_time=pl.lit(grid_start) + pl.duration(minutes=30 * pl.col("end")),
        )
        .with_columns(
            duration_days=(pl.col("end_time") - pl.col("start_time")).dt.total_hours() / 24
        )
    )


def event_windows(*, spans: pl.DataFrame, count: int) -> pl.DataFrame:
    """Return up to `count` events across both metering groups, each at its own Primary.

    One span is kept per verified transfer and per Primary. Only spans whose whole window, from 14
    days before the start to 7 days after, lies between the first stage 2 origin and the holdout
    start are candidates, so every curve exists in every panel. The groups are taken in turn so that
    both appear.
    """
    candidates = (
        spans.filter(
            pl.col("start_time") - pl.duration(days=DAYS_BEFORE) >= STAGE2_START,
            pl.col("start_time") + pl.duration(days=DAYS_AFTER) <= HOLDOUT_START,
        )
        .sort("duration_days", descending=True)
        .unique(subset=["source"], keep="first", maintain_order=True)
        .unique(subset=["time_series_id"], keep="first", maintain_order=True)
        .sort("duration_days", descending=True)
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
        msg = "no event of event_spans(kind='primary') has its whole window in the stage 2 rows"
        raise SystemExit(msg)
    return pl.concat(chosen)


def half_hour_grid(*, after: dt.datetime, until: dt.datetime) -> pl.DataFrame:
    """Return every half-hour valid time in `(after, until]`, so a line breaks where telemetry is
    missing."""
    return pl.DataFrame(
        {
            "valid_time": pl.datetime_range(
                after + dt.timedelta(minutes=30),
                until,
                interval=HALF_HOUR,
                time_unit="us",
                time_zone="UTC",
                eager=True,
            )
        }
    )


def on_grid(*, grid: pl.DataFrame, rows: pl.DataFrame, label: str) -> pl.DataFrame:
    """Put `rows` (`valid_time`, `value`) on the full grid with nulls where a row is missing.

    Args:
        grid: The full valid-time grid.
        rows: The rows to place, with `valid_time` and `value`.
        label: The name of the series, used in the error message.

    Returns:
        The grid with `value` joined on, null where a row is missing.

    Raises:
        SystemExit: If `rows` holds two rows at one valid time, which would double a point.
    """
    rows = rows.select(pl.col("valid_time").dt.cast_time_unit("us"), "value")
    joined = grid.join(rows, on="valid_time", how="left").with_columns(forecast=pl.lit(label))
    if joined.height != grid.height:
        msg = f"{label}: more than one row at some valid time of the window"
        raise SystemExit(msg)
    return joined


def stored_lazy(*, name: str) -> pl.LazyFrame:
    """Scan one stored prediction set of this experiment, after checking its manifest's row set.

    The check compares the manifest with the row set this script expects for the set, so a set
    that was re-scored on other rows stops the script.
    """
    folder = c.OUT_DIR / "predictions" / name
    manifest = folder / "manifest.json"
    if not manifest.exists():
        msg = f"missing prediction set {folder}; run the fit and score scripts first"
        raise SystemExit(msg)
    expected = SET_ROW_SET_IDS[name]
    found = json.loads(manifest.read_text())["row_set_id"]
    if found != expected:
        msg = f"prediction set {name} covers row set {found!r}, not {expected!r}"
        raise SystemExit(msg)
    return pl.scan_parquet(folder / PREDICTIONS_NAME).filter(pl.col("valid_time") < HOLDOUT_START)


def window_rows(
    *, series: int, start: dt.datetime, end: dt.datetime, capacity: float
) -> pl.DataFrame:
    """Return observed power and the three forecasts of one Primary over a window, in percent of
    capacity.

    Each forecast is stitched over daily origins: the row at each valid time comes from the latest
    origin before it, so its lead is 0 to 24 hours (day-ahead rows). Every curve sits on the
    half-hour grid of the window with nulls at missing rows.
    """
    columns = ["valid_time", "power_fcst_init_time", "q50"]
    grid = half_hour_grid(after=start, until=end)
    observed = (
        pl.scan_parquet(c.BACKTEST_DIR / "ground_truth_primaries.parquet")
        .filter(
            pl.col("time_series_id") == series,
            pl.col("valid_time") > start,
            pl.col("valid_time") <= end,
            pl.col("valid_time") < HOLDOUT_START,
            pl.col("power").is_not_null(),
        )
        .select("valid_time", value=pl.col("power").cast(pl.Float64) * 100.0 / capacity)
        .collect()
    )
    frames = [on_grid(grid=grid, rows=observed, label=OBSERVED_LABEL)]
    for label, source in THREE_ARM_SETS.items():
        if source == "nra":
            lazy = (
                pl.scan_parquet(c.BACKTEST_DIR / "primaries.parquet")
                .filter(pl.col("NRA_or_prevailing") == "NRA", pl.col("valid_time") < HOLDOUT_START)
                .select("time_series_id", *columns)
            )
        else:
            lazy = stored_lazy(name=source).select("time_series_id", *columns)
        rows = (
            lazy.filter(
                pl.col("time_series_id") == series,
                pl.col("valid_time") > start,
                pl.col("valid_time") <= end,
                pl.col("valid_time") > pl.col("power_fcst_init_time"),
                pl.col("valid_time") <= pl.col("power_fcst_init_time") + pl.duration(hours=24),
            )
            .select("valid_time", value=pl.col("q50").cast(pl.Float64) * 100.0 / capacity)
            .collect()
        )
        if rows.is_empty():
            msg = (
                f"{label}: no rows for series {series} in the window; the panel would lack a curve"
            )
            raise SystemExit(msg)
        frames.append(on_grid(grid=grid, rows=rows, label=label))
    return pl.concat(frames)


def shading_rows(*, spans: pl.DataFrame, row: dict[str, object]) -> pl.DataFrame:
    """Return the day ranges of every span of the same Primary that overlaps the panel's window.

    The plotted event is marked `SHADE_THIS` and every other overlapping span `SHADE_OTHER`. Ranges
    are in days from the plotted event's start and clipped to the window.
    """
    start: dt.datetime = row["start_time"]  # ty: ignore[invalid-assignment]
    window_start = start - dt.timedelta(days=DAYS_BEFORE)
    window_end = start + dt.timedelta(days=DAYS_AFTER)
    return (
        spans.filter(
            pl.col("time_series_id") == row["time_series_id"],
            pl.col("end_time") > window_start,
            pl.col("start_time") < window_end,
        )
        .select(
            kind=pl.when(pl.col("source") == row["source"])
            .then(pl.lit(SHADE_THIS))
            .otherwise(pl.lit(SHADE_OTHER)),
            **{
                "from": (pl.col("start_time").clip(window_start, window_end) - pl.lit(start))
                .dt.total_seconds()
                .truediv(86400.0),
                "to": (pl.col("end_time").clip(window_start, window_end) - pl.lit(start))
                .dt.total_seconds()
                .truediv(86400.0),
            },
        )
        .unique(maintain_order=True)
    )


def figure_events() -> Path:
    """Draw figure 2: observed power and three forecasts around the start of real events."""
    spans = primary_spans()
    events = event_windows(spans=spans, count=MAX_EVENTS)
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
        wide = data.pivot(
            on="forecast", index="day", values="value", aggregate_function="first"
        ).drop_nulls(OBSERVED_LABEL)
        after = wide.filter(pl.col("day") >= 0)
        errors = {
            label: after.select((pl.col(label) - pl.col(OBSERVED_LABEL)).abs().mean()).item()
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
        shading = shading_rows(spans=spans, row=row)
        print_frame(frame=shading, heading=f"figure 2 shaded day ranges: {title}")
        shade = (
            alt.Chart(shading)
            .mark_rect(color=ocf.GREY_3)
            .encode(  # ty: ignore[unresolved-attribute]
                x="from:Q",
                x2="to:Q",
                opacity=alt.Opacity(
                    "kind:N",
                    legend=None,
                    scale=alt.Scale(domain=[SHADE_THIS, SHADE_OTHER], range=[0.85, 0.35]),
                ),
            )
        )
        rules = (
            alt.Chart(pl.DataFrame({"day": [0.0]}))
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
                takeaway=(
                    f"Observed power and three day-ahead medians around {len(panels)} labelled events "
                    "and verified transfers at the Primaries"
                ),
                subtitle=(
                    "Observed cleaned power and the day-ahead median (leads 0 to 24 h, stitched over daily 12:00 UTC "
                    "origins) of the stored normal-running forecast, the G-lb corrected forecast, and the N-gate "
                    "corrected forecast (real graph, post hoc), in percent of capacity, for 14 days before to "
                    "7 days after the start of the event, at a Primary with signed-MW or MVA-only metering. "
                    "Dark grey band: the event of the panel, from its start to its end, clipped at day 7. "
                    "Light grey bands: other events or transfers at the same Primary in the window. Lines "
                    "break where telemetry is missing."
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


def check_origin(*, origin: dt.datetime) -> None:
    """Stop unless the 14-day window of `origin` lies in the stage 1 rows and before the holdout."""
    if not (origin >= STAGE1_START and origin + dt.timedelta(days=14) <= HOLDOUT_START):
        msg = (
            f"origin {origin} must satisfy {STAGE1_START:%Y-%m-%d} <= origin and "
            f"origin + 14 days <= {HOLDOUT_START:%Y-%m-%d}"
        )
        raise SystemExit(msg)


def example_rows(*, series: int, origin: dt.datetime, capacity: float) -> pl.DataFrame:
    """Return the one forecast issued at `origin` of four kinds, and the observed power, in % of
    capacity.

    Every curve sits on the half-hour grid of the 14 days with nulls at missing rows.
    """
    end = origin + dt.timedelta(days=14)
    grid = half_hour_grid(after=origin, until=end)
    nra = (
        pl.scan_parquet(c.BACKTEST_DIR / "primaries.parquet")
        .filter(
            pl.col("NRA_or_prevailing") == "NRA",
            pl.col("time_series_id") == series,
            pl.col("power_fcst_init_time") == origin,
            pl.col("valid_time") > origin,
            pl.col("valid_time") <= end,
            pl.col("valid_time") < HOLDOUT_START,
        )
        .collect()
    )
    forecasts = [
        on_grid(
            grid=grid,
            rows=nra.select("valid_time", value=pl.col(column).cast(pl.Float64) * 100.0 / capacity),
            label=label,
        )
        for label, column in (
            ("stored normal-running q05", "q05"),
            ("stored normal-running q95", "q95"),
            (STORED_MEDIAN, "q50"),
        )
    ]
    g_rows = (
        stored_lazy(name="g")
        .filter(
            pl.col("time_series_id") == series,
            pl.col("power_fcst_init_time") == origin,
            pl.col("valid_time") > origin,
            pl.col("valid_time") <= end,
        )
        .select("valid_time", value=pl.col("q50").cast(pl.Float64) * 100.0 / capacity)
        .collect()
    )
    forecasts.append(on_grid(grid=grid, rows=g_rows, label=G_MEDIAN))
    observed = (
        pl.scan_parquet(c.BACKTEST_DIR / "ground_truth_primaries.parquet")
        .filter(
            pl.col("time_series_id") == series,
            pl.col("valid_time") > origin,
            pl.col("valid_time") <= end,
            pl.col("valid_time") < HOLDOUT_START,
            pl.col("power").is_not_null(),
        )
        .select("valid_time", value=pl.col("power").cast(pl.Float64) * 100.0 / capacity)
        .collect()
    )
    forecasts.append(on_grid(grid=grid, rows=observed, label=OBSERVED_LABEL))
    return pl.concat(forecasts).with_columns(
        hours=(pl.col("valid_time") - pl.lit(origin)).dt.total_seconds() / 3600.0
    )


def correction_share(*, data: pl.DataFrame, hours: float) -> float:
    """Return mean |G-lb q50 - stored q50| divided by mean (q95 - q05), over the rows up to
    `hours`."""
    wide = (
        data.filter(pl.col("hours") <= hours)
        .pivot(on="forecast", index="valid_time", values="value", aggregate_function="first")
        .drop_nulls(
            [G_MEDIAN, STORED_MEDIAN, "stored normal-running q05", "stored normal-running q95"]
        )
    )
    if wide.is_empty():
        msg = f"no half-hour has all four forecast curves within {hours} hours"
        raise SystemExit(msg)
    return wide.select(
        (pl.col(G_MEDIAN) - pl.col(STORED_MEDIAN)).abs().mean()
        / (pl.col("stored normal-running q95") - pl.col("stored normal-running q05")).mean()
    ).item()


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


def figure_examples(
    *, label: str = EXAMPLE_LABEL, origin: dt.datetime = EXAMPLE_ORIGIN, force: bool = False
) -> Path:
    """Draw figure 3: one forecast of one Primary, over its first 24 hours and over all 14 days.

    Args:
        label: The label of the Primary whose forecast is drawn.
        origin: The forecast origin of the drawn forecast.
        force: Draw even when an event overlaps the window.

    Returns:
        The path of the written SVG.

    Raises:
        SystemExit: If the origin is outside the scored rows, or an event overlaps the window and
            `force` is false.
    """
    check_origin(origin=origin)
    site = c.read_site_map().filter(pl.col("label") == label)
    if site.is_empty():
        msg = f"no Primary labelled {label!r} in the site map"
        raise SystemExit(msg)
    series, capacity, group = site["time_series_id"][0], site["capacity_mw"][0], site["group"][0]
    if (
        overlaps_event(series=series, start=origin, end=origin + dt.timedelta(days=14))
        and not force
    ):
        msg = (
            f"{label} has a labelled event or verified transfer within 14 days of {origin}; "
            "pick another origin with --origin, or pass --force to draw it anyway"
        )
        raise SystemExit(msg)
    data = example_rows(series=series, origin=origin, capacity=capacity)
    print_frame(
        frame=data.sort("forecast", "hours"),
        heading=f"figure 3: {label}, origin {origin:%Y-%m-%d %H:%M} UTC, % of capacity",
    )
    first_day = correction_share(data=data, hours=24.0)
    whole = correction_share(data=data, hours=336.0)
    print(
        f"\nfigure 3: mean |G-lb q50 - stored q50| / mean band width: {first_day:.4f} (24 h), {whole:.4f} (14 days)"
    )
    day_ahead = example_panel(
        data=data, hours=24.0, title="Day ahead (leads 0 to 24 hours)", unit_days=False
    )
    full = example_panel(data=data, hours=336.0, title="Full 14-day horizon", unit_days=True)
    chart = (
        alt.vconcat(day_ahead, full, spacing=24)
        .properties(
            title=title_params(
                takeaway=(
                    f"The G-lb correction moves the median by {first_day * 100:.0f}% of the stored "
                    f"q05 to q95 band's mean width over the first 24 hours, and by {whole * 100:.0f}% over the 14 days."
                ),
                subtitle=(
                    f"One forecast of Primary {label} ({group}), issued at 12:00 UTC on {origin:%Y-%m-%d}, against "
                    "observed cleaned power, in percent of capacity. Band: q05 to q95 of the stored "
                    "normal-running forecast. Orange line: its median. Blue line: the median of the G-lb "
                    "corrected forecast. The origin is chosen as an event-free example, not as the best case. "
                    "Lines break where telemetry is missing."
                ),
            )
        )
        .resolve_scale(color="shared")
    )
    return finish(chart=chart, name="fig3_examples")


# --------------------------------------------------------------------------- figure 4


def figure_copied_share() -> Path:
    """Draw figure 4: the copied share of an injected shift, by variant, coloured by family."""
    frame = read_table(name="s2_bank_share_and_crps")
    require_columns(
        frame=frame,
        columns=["variant", BANK_COLUMN, BANK_EVENTS_COLUMN],
        table="s2_bank_share_and_crps",
    )
    present = set(frame["variant"].to_list())
    missing = (set(COPIED_VARIANTS) - OPTIONAL_VARIANTS) - present
    if missing:
        msg = f"s2_bank_share_and_crps.csv has no row for {sorted(missing)}"
        raise SystemExit(msg)
    variants = [v for v in COPIED_VARIANTS if v in present]
    shown = (
        frame.filter(pl.col("variant").is_in(variants))
        .select("variant", events=BANK_EVENTS_COLUMN, share=pl.col(BANK_COLUMN) * 100.0)
        .with_columns(
            variant_label=pl.col("variant").replace_strict(VARIANT_LABELS, return_dtype=pl.String),
            family=pl.col("variant").replace_strict(VARIANT_FAMILIES, return_dtype=pl.String),
            order=pl.col("variant").replace_strict(
                {v: i for i, v in enumerate(variants)}, return_dtype=pl.Int64
            ),
        )
        .sort("order")
    )
    print_frame(
        frame=shown,
        heading="figure 4: copied share of an injected shift (% of the shift), events of depth at least 3%",
    )
    share = dict(zip(shown["variant"].to_list(), shown["share"].to_list(), strict=True))
    rule, real, rewired, zero = (
        share["N-rule"],
        share["N-gate real"],
        share["N-gate rewired"],
        share["N-gate real zero neighbours"],
    )
    if not (real < rule and rewired < rule):
        msg = (
            f"N-gate copies {real:.1f}% (real graph) and {rewired:.1f}% (rewired graph) against "
            f"N-rule's {rule:.1f}%: the comparative title no longer holds"
        )
        raise SystemExit(msg)
    takeaway = (
        f"With neighbours, real or rewired, N-gate copies {real:.1f}% or {rewired:.1f}% of an injected "
        f"shift against N-rule's {rule:.1f}%; with zero neighbours it copies {zero:.1f}%."
    )
    labels = shown["variant_label"].to_list()
    families = list(FAMILY_COLOURS)
    bars = (
        alt.Chart(shown)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("variant_label:N", sort=labels, title=None),
            x=alt.X(
                "share:Q",
                title="Share of the injected shift that the forecast copies (%; smaller is better)",
                scale=alt.Scale(domain=[0, max(max(shown["share"].to_list()) * 1.2, 1.0)]),
            ),
            color=alt.Color(
                "family:N",
                title=None,
                scale=alt.Scale(domain=families, range=list(FAMILY_COLOURS.values())),
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
        width=WIDTH_PX - 260,
        height=24 * len(labels),
        title=title_params(
            takeaway=takeaway,
            subtitle=(
                f"Mechanism check, post hoc: {n_events} injected level shifts of depth at least 3% of capacity "
                "(bank events), each with the mirrored step put on the real-graph neighbours only. Bar: share "
                "of the shift that the forecast copies when it should not. Colour: the family of the forecast "
                "(stored forecast, rule, flag, or gate). N-gate with zero neighbours gets no neighbour "
                "information."
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
        help="origin day of figure 3 (12:00 UTC), YYYY-MM-DD, from 2025-12-01 to 14 days before the holdout start",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="draw figure 3 even if an event overlaps its 14-day window",
    )
    args = parser.parse_args()
    origin = dt.datetime.fromisoformat(args.origin).replace(hour=12, tzinfo=dt.UTC)
    if not c.OUT_DIR.exists():
        msg = f"experiment folder {c.OUT_DIR} is missing; set DATA_PATH_INTERNAL"
        raise SystemExit(msg)
    if "fig1_lead_bands" in args.only:
        figure_lead_bands()
    if "fig2_events" in args.only:
        figure_events()
    if "fig3_examples" in args.only:
        figure_examples(label=args.label, origin=origin, force=args.force)
    if "fig4_copied_share" in args.only:
        figure_copied_share()
    print(f"\nPNG previews are in {preview_dir()}")


if __name__ == "__main__":
    main()
