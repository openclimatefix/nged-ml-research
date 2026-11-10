# ruff: noqa: E501
"""Private time-series figures for the BSPs, GSPs, wind farms, and solar farms of the backtest.

POLARS_MAX_THREADS=2 uv run --no-sync python ml-experiments/00668-*/e00668_private_plots.py OUT_DIR

Writes SVG figures, `index.html`, and `README.md` to OUT_DIR, which is private (the figures name
original ids and calendar dates of NGED's generators, so OUT_DIR is never committed or published).
Per series it draws (1) two weeks stitched day by day (demand: a cold week and a warm week; wind: a
windy week and a calm week; solar: a sunny week and a dull week), (2) two 14-day forecast fans
(one winter and one summer origin), and (3) the median at four lead times against the observed
power for the same two weeks. Every figure has at most two panels of 640 by 300 pixels.
"""

import html
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import altair as alt
import e00668_score_lib as lib
import numpy as np
import polars as pl

FOLDER = lib.FOLDER
WIDTH, HEIGHT = 640, 300
DAY = 48
BLACK, BLUE, PURPLE, ORANGE, TEAL = "#000000", "#306BFF", "#B701FF", "#FF4901", "#0E9F6E"
LEAD_COLOURS = {"0-1 d": BLUE, "2-3 d": TEAL, "6-7 d": PURPLE, "13-14 d": ORANGE}
LEAD_WINDOWS = {"0-1 d": (0, 1), "2-3 d": (2, 3), "6-7 d": (6, 7), "13-14 d": (13, 14)}
GROUPS = {
    "BSPs": ("bsps", "demand", [1, 2], "BSP"),
    "GSPs": ("gsps", "demand", [27, 28], "GSP"),
    "Wind farms": ("wind", "wind", [4, 5, 6], "Wind"),
    "Solar farms": ("solar", "solar", [21, 22, 23, 29, 30, 31], "Solar"),
}
WEEK_NAMES = {
    "demand": ("cold week", "warm week"),
    "wind": ("windy week", "calm week"),
    "solar": ("sunny week", "dull week"),
}
WEEK_RULES = {
    "demand": "the cold week is the Monday-to-Sunday week of highest mean power in December to February, and the warm week is the week of lowest mean power in June to August",
    "wind": "the windy week is the Monday-to-Sunday week of highest mean power and the calm week the week of lowest mean power",
    "solar": "the sunny week is the Monday-to-Sunday week of highest mean power in April to September, and the dull week the week of lowest mean power in the same months",
}


def to_time(idx: np.ndarray | int) -> object:
    """Return the UTC instant of a half-hour index."""
    base = np.datetime64(lib.GRID_START.replace(tzinfo=None), "us")
    return base + np.asarray(idx) * np.timedelta64(1800 * 10**6, "us")


def pick_weeks(*, obs: pl.DataFrame, kind: str) -> list[datetime]:
    """Return the Monday starts of the two example weeks (the first is the 'high' week)."""
    weeks = (
        obs.drop_nulls("power")
        .with_columns(
            week=pl.col("valid_time").dt.truncate("1w"), month=pl.col("valid_time").dt.month()
        )
        .group_by("week")
        .agg(mean=pl.col("power").mean(), n=pl.len(), month=pl.col("month").first())
        .filter(pl.col("n") >= 0.9 * 7 * DAY * (0.5 if kind == "solar" else 1.0))
        .filter(pl.col("week") >= datetime(2025, 11, 4, tzinfo=UTC))
        .filter(pl.col("week") <= datetime(2026, 10, 12, tzinfo=UTC))
    )
    if kind == "demand":
        hi = weeks.filter(pl.col("month").is_in([12, 1, 2])).sort("mean").tail(1)
        lo = weeks.filter(pl.col("month").is_in([6, 7, 8])).sort("mean").head(1)
        return [hi["week"][0], lo["week"][0]]
    if kind == "solar":
        pool = weeks.filter(pl.col("month").is_in([4, 5, 6, 7, 8, 9]))
        return [pool.sort("mean")["week"][-1], pool.sort("mean")["week"][0]]
    return [weeks.sort("mean")["week"][-1], weeks.sort("mean")["week"][0]]


def nearest_origin(*, origins: np.ndarray, when: datetime) -> int:
    """Return the available origin index nearest to `when`."""
    target = lib.grid_index(times=pl.Series([when]).dt.replace_time_zone("UTC"))[0]
    return int(origins[np.argmin(np.abs(origins - target))])


def observed_layer(frame: pl.DataFrame) -> alt.Chart:
    """Return the black observed-power line."""
    return (
        alt.Chart(frame.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(color=BLACK, strokeWidth=1)
        .encode(x="time:T", y="power:Q")
    )


def title(lines: list[str]) -> alt.TitleParams:
    """Return a left-aligned multi-line figure title."""
    return alt.TitleParams(lines, anchor="start", fontSize=13)


def weeks_figure(
    *, panels: list[tuple[str, pl.DataFrame, pl.DataFrame]], label: str, unit: str, has_heur: bool
) -> alt.VConcatChart:
    """Return up to two stacked panels of stitched day-ahead forecasts."""
    layers = []
    for name, stitched, obs in panels:
        base = alt.Chart(stitched.to_pandas()).encode(x=alt.X("time:T", title="Date (UTC)"))  # ty: ignore[unresolved-attribute]
        band = base.mark_area(opacity=0.25, color=BLUE).encode(  # ty: ignore
            y=alt.Y("q05:Q", title=f"{label} power ({unit})", scale=alt.Scale(zero=False)),
            y2="q95:Q",
        )
        med = base.mark_line(color=BLUE, strokeWidth=1.5).encode(y="q50:Q")  # ty: ignore
        layer = band + med
        if has_heur:
            layer = layer + base.mark_line(color=ORANGE, strokeDash=[4, 3], strokeWidth=1.2).encode(  # ty: ignore
                y="heur:Q"
            )
        layers.append(
            (layer + observed_layer(obs)).properties(width=WIDTH, height=HEIGHT, title=name)
        )
    return alt.vconcat(*layers).properties(
        title=title(
            [
                f"{label}: day-ahead forecasts, stitched day by day",
                "Black: observed. Blue: median with q05 to q95 band."
                + (" Orange dashes: manual heuristic." if has_heur else ""),
            ]
        )
    )


def fans_figure(
    *, panels: list[tuple[str, pl.DataFrame, pl.DataFrame, object]], label: str, unit: str
) -> alt.VConcatChart:
    """Return up to two stacked panels of 14-day forecasts from one origin each."""
    layers = []
    for name, fan, obs, origin in panels:
        base = alt.Chart(fan.to_pandas()).encode(x=alt.X("time:T", title="Date (UTC)"))  # ty: ignore[unresolved-attribute]
        outer = base.mark_area(opacity=0.2, color=BLUE).encode(  # ty: ignore
            y=alt.Y("q05:Q", title=f"{label} power ({unit})", scale=alt.Scale(zero=False)),
            y2="q95:Q",
        )
        inner = base.mark_area(opacity=0.3, color=BLUE).encode(y="q25:Q", y2="q75:Q")  # ty: ignore
        med = base.mark_line(color=BLUE, strokeWidth=1.5).encode(y="q50:Q")  # ty: ignore
        rule = (
            alt.Chart(pl.DataFrame({"time": pl.Series(np.array([origin]))}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color=BLACK)
            .encode(x="time:T")
        )
        layers.append(
            (outer + inner + med + observed_layer(obs) + rule).properties(
                width=WIDTH, height=HEIGHT, title=name
            )
        )
    return alt.vconcat(*layers).properties(
        title=title(
            [
                f"{label}: 14-day forecasts from two origins",
                "Black line: observed. Black vertical line: the forecast origin. Dark band q25 to q75, light band q05 to q95, blue line median.",
            ]
        )
    )


def leads_figure(
    *, panels: list[tuple[str, pl.DataFrame, pl.DataFrame]], label: str, unit: str
) -> alt.VConcatChart:
    """Return up to two stacked panels of the median at four lead times."""
    layers = []
    for name, long, obs in panels:
        line = (
            alt.Chart(long.to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1)
            .encode(
                x=alt.X("time:T", title="Date (UTC)"),
                y=alt.Y("q50:Q", title=f"{label} power ({unit})", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "lead:N",
                    title="Lead time",
                    scale=alt.Scale(domain=list(LEAD_COLOURS), range=list(LEAD_COLOURS.values())),
                ),
            )
        )
        layers.append(
            (line + observed_layer(obs)).properties(width=WIDTH, height=HEIGHT, title=name)
        )
    return alt.vconcat(*layers).properties(
        title=title(
            [
                f"{label}: the median at four lead times",
                "Black: observed. Colours: the median forecast for the same half-hours made 0-1, 2-3, 6-7, and 13-14 days ahead.",
            ]
        )
    )


def stitched_week(
    *, fc: pl.DataFrame, start: datetime, grid: np.ndarray | None, row: int
) -> pl.DataFrame:
    """Return the day-ahead (lead up to 24 hours) quantiles for one week, with the heuristic if a grid is given."""
    s0 = lib.grid_index(times=pl.Series([start]).dt.replace_time_zone("UTC"))[0]
    s = fc.filter(
        (pl.col("vidx") - pl.col("oidx") <= DAY)
        & (pl.col("vidx") > s0)
        & (pl.col("vidx") <= s0 + 7 * DAY)
    ).sort("vidx")
    out = s.select(time=pl.Series(to_time(s["vidx"].to_numpy())), q05="q05", q50="q50", q95="q95")  # ty: ignore[invalid-argument-type]
    if grid is not None:
        lead = (s["vidx"] - s["oidx"]).to_numpy() / 2.0
        heur = lib.manual_heuristic(
            grid=grid, row=np.full(s.height, row), vidx=s["vidx"].to_numpy(), lead_h=lead
        )[:, 5]
        out = out.with_columns(heur=pl.Series(heur))
    return out


def main(out: Path) -> None:  # noqa: PLR0915
    """Draw every figure and write the index page and the README."""
    out.mkdir(parents=True, exist_ok=True)
    cap = pl.read_parquet(FOLDER / "series_capacity.parquet")
    table_rows: list[tuple[str, str, int, str, float]] = []
    sections: list[tuple[str, list[tuple[str, str, str]]]] = []
    files: list[str] = []
    for group, (stem, kind, ids, prefix) in GROUPS.items():
        fc_all = pl.read_parquet(FOLDER / f"{stem}.parquet").filter(
            pl.col("time_series_id").is_in(ids)
        )
        truth = pl.read_parquet(FOLDER / f"ground_truth_{stem}.parquet").filter(
            pl.col("time_series_id").is_in(ids)
        )
        grid = lib.cleaned_grid(ids=ids) if kind == "demand" else None
        entries: list[tuple[str, str, str]] = []
        for n, sid in enumerate(ids):
            label = f"{prefix} {chr(65 + n)}" if kind != "demand" else f"{prefix} {sid}"
            unit = truth.filter(pl.col("time_series_id") == sid)["power_unit"][0]
            capacity = float(cap.filter(pl.col("time_series_id") == sid)["capacity"][0])
            table_rows.append((group, label, sid, unit, capacity))
            slug = label.lower().replace(" ", "_")
            obs_all = truth.filter(pl.col("time_series_id") == sid).select("valid_time", "power")
            fc = fc_all.filter(pl.col("time_series_id") == sid).with_columns(
                vidx=pl.Series(
                    lib.grid_index(
                        times=fc_all.filter(pl.col("time_series_id") == sid)["valid_time"]
                    )
                ),
                oidx=pl.Series(
                    lib.grid_index(
                        times=fc_all.filter(pl.col("time_series_id") == sid)["power_fcst_init_time"]
                    )
                ),
            )

            def obs_between(
                a: datetime, b: datetime, obs_all: pl.DataFrame = obs_all
            ) -> pl.DataFrame:
                return (
                    obs_all.filter((pl.col("valid_time") >= a) & (pl.col("valid_time") < b))
                    .drop_nulls("power")
                    .select(time=pl.col("valid_time").dt.replace_time_zone(None), power="power")
                )

            weeks = pick_weeks(obs=obs_all, kind=kind)
            names = WEEK_NAMES[kind]
            # (1) weeks
            panels = []
            for name, start in zip(names, weeks, strict=True):
                st = stitched_week(fc=fc, start=start, grid=grid, row=ids.index(sid))
                panels.append(
                    (
                        f"{name}: {start:%Y-%m-%d} to {start + timedelta(days=6):%Y-%m-%d}",
                        st,
                        obs_between(start, start + timedelta(days=7)),
                    )
                )
            f1 = f"{slug}_weeks.svg"
            weeks_figure(panels=panels, label=label, unit=unit, has_heur=grid is not None).save(
                out / f1
            )
            desc1 = (
                f"{label} (original id {sid}, {unit}). Two example weeks, {WEEK_RULES[kind]}: top {panels[0][0]}; bottom {panels[1][0]}. "
                "Each forecast point is the first 24 hours of the daily 12:00 UTC origin, so the blue line joins one day-long forecast to the next. "
                "Black is observed power; blue is the median with the q05 to q95 band"
                + ("; orange dashes are NGED's manual heuristic" if grid is not None else "")
                + ". Look for the band holding the black line about 90% of the time, jumps at the 12:00 joins, and the heating, solar, or wind pattern the median misses."
            )
            # (2) fans
            origins = np.sort(fc["oidx"].unique().to_numpy())
            fan_panels = []
            for name, when in (
                ("Winter origin", datetime(2025, 12, 15, 12, tzinfo=UTC)),
                ("Summer origin", datetime(2026, 6, 15, 12, tzinfo=UTC)),
            ):
                o = nearest_origin(origins=origins, when=when)
                s = fc.filter(pl.col("oidx") == o).sort("vidx")
                fan = s.select(
                    time=pl.Series(to_time(s["vidx"].to_numpy())),  # ty: ignore
                    q05="q05",
                    q25="q25",
                    q50="q50",
                    q75="q75",
                    q95="q95",
                )  # ty: ignore[invalid-argument-type]
                t0 = to_time(o)
                a = datetime.fromtimestamp(
                    float(t0.astype("datetime64[s]").astype(int)) - 3 * 86400,  # ty: ignore[unresolved-attribute]
                    tz=UTC,
                )
                fan_panels.append(
                    (
                        f"{name}: {np.datetime_as_string(t0, unit='m')} UTC",  # ty: ignore
                        fan,
                        obs_between(a, a + timedelta(days=17)),
                        t0,
                    )
                )
            f2 = f"{slug}_fans.svg"
            fans_figure(panels=fan_panels, label=label, unit=unit).save(out / f2)
            desc2 = (
                f"{label} (original id {sid}, {unit}). The full 14-day forecast from one run in winter (top) and one in summer (bottom). "
                "Black is observed power from 3 days before the origin to the end of the horizon; the vertical line is the origin. "
                "The dark band is q25 to q75 and the light band is q05 to q95. Look for the bands widening with lead time and for the median decaying towards the typical level."
            )
            # (3) lead times
            lp = []
            for name, start in zip(names, weeks, strict=True):
                s0 = lib.grid_index(times=pl.Series([start]).dt.replace_time_zone("UTC"))[0]
                parts = []
                for lead_name, (d0, d1) in LEAD_WINDOWS.items():
                    s = fc.filter(
                        (pl.col("vidx") - pl.col("oidx") >= d0 * DAY)
                        & (pl.col("vidx") - pl.col("oidx") < d1 * DAY)
                        & (pl.col("vidx") > s0)
                        & (pl.col("vidx") <= s0 + 7 * DAY)
                    ).sort("vidx")
                    parts.append(
                        s.select(
                            time=pl.Series(to_time(s["vidx"].to_numpy())),  # ty: ignore[invalid-argument-type]
                            q50="q50",
                        ).with_columns(lead=pl.lit(lead_name))
                    )  # ty: ignore[invalid-argument-type]
                lp.append(
                    (
                        f"{name}: {start:%Y-%m-%d} to {start + timedelta(days=6):%Y-%m-%d}",
                        pl.concat(parts),
                        obs_between(start, start + timedelta(days=7)),
                    )
                )
            f3 = f"{slug}_lead_times.svg"
            leads_figure(panels=lp, label=label, unit=unit).save(out / f3)
            desc3 = (
                f"{label} (original id {sid}, {unit}). The same two weeks as the first figure. Each coloured line is the median forecast for the same half-hours made 0-1 days, 2-3 days, 6-7 days, and 13-14 days ahead; black is observed power. "
                "Look for how quickly the longer-lead lines lose the daily shape and drift from the black line."
            )
            files += [f1, f2, f3]
            entries += [
                (f1, "Two example weeks", desc1),
                (f2, "14-day forecast fans", desc2),
                (f3, "Median at four lead times", desc3),
            ]
            print(label, "done", flush=True)
        sections.append((group, entries))
    write_index(out=out, sections=sections, table=table_rows)
    (out / "README.md").write_text(
        "# Private time-series plots for experiment 668 (private: not for publication)\n\n"
        "Built by `ml-experiments/00668-*/e00668_private_plots.py` in the research repository from the files in `backtest_year_2025-11_to_2026-10/`.\n\n"
        "- `index.html`: every figure inline with a description, and the table of series.\n"
        "- `event_D5_fan_*.svg`, `event_D16_fan_*.svg` (10 files): the load-shift fan figures removed from the public page.\n"
        "- `<series>_weeks.svg`, `<series>_fans.svg`, `<series>_lead_times.svg`: the new figures for each BSP, GSP, wind farm, and solar farm.\n"
        "- Files: " + ", ".join(f"`{f}`" for f in sorted(files)) + "\n"
    )


def write_index(*, out: Path, sections: list, table: list) -> None:
    """Write the single self-contained private page."""
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Private time-series plots, experiment 668</title>",
        "<style>body{font-family:sans-serif;max-width:900px;margin:2em auto;padding:0 1em}figure{margin:1.5em 0}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:2px 8px}.priv{background:#fde;border:1px solid #c33;padding:.5em;font-weight:bold}</style>",
        "<h1>Time-series plots of the one-year backtest</h1><p class='priv'>Private: not for publication. This page holds original generator ids and calendar dates.</p>",
        "<p>The forecast files are in <code>backtest_year_2025-11_to_2026-10/</code>; the page that documents them is experiment 668. Quantile bands are as stored (not the calibrated copies). The 10 load-shift fan figures at D5 and D16 removed from the public page are in this folder as <code>event_D*_fan_*.svg</code>.</p>",
        "<h2>Series</h2><table><tr><th>Group</th><th>Label</th><th>time_series_id</th><th>Unit</th><th>Capacity</th></tr>",
    ]
    for group, label, sid, unit, capacity in table:
        parts.append(
            f"<tr><td>{group}</td><td>{label}</td><td>{sid}</td><td>{unit}</td><td>{capacity:.2f}</td></tr>"
        )
    parts.append("</table>")
    for group, entries in sections:
        parts.append(f"<h2>{html.escape(group)}</h2>")
        for fname, caption, desc in entries:
            svg = (out / fname).read_text()
            svg = svg[svg.index("<svg") :]
            parts.append(
                f"<figure><h4>{html.escape(caption)}</h4>{svg}<figcaption>{html.escape(desc)}</figcaption></figure>"
            )
    parts.append("<h2>Load-shift fans at D5 and D16</h2>")
    for f in sorted(out.glob("event_D*_fan_*.svg")):
        svg = f.read_text()
        parts.append(
            f"<figure><h4>{f.name}</h4>{svg[svg.index('<svg') :]}<figcaption>Removed from the public page. Normal running above, prevailing below; 14-day forecast from the origin named in the file name.</figcaption></figure>"
        )
    (out / "index.html").write_text("\n".join(parts))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
