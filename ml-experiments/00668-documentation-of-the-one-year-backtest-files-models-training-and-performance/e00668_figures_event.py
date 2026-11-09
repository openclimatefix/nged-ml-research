# ruff: noqa: E501
"""Figures of the load shift at D5 and D16 (December 2025 to January 2026) in `primaries.parquet`.

uv run --no-sync python ml-experiments/00668-*/e00668_figures_event.py OUT_DIR

For each of D5 and D16 this draws (1) the day-ahead stitched forecast of the normal running and
the prevailing output beside the observed power, with the manual heuristic as a dashed third line,
and (2) 14-day forecasts from five origins (before the start, early inside, late inside, in the
last 24 hours before the end, and just after the end). It also writes `event_numbers.json`, which
holds the step start and end found in the observed power and the hours until the prevailing output
moved and returned. The step times are the largest change of the 12-hour mean in a window around
the dates of experiment 580. Substations may be named and dated; no generator appears.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import e00668_score_lib as lib
import numpy as np
import polars as pl

R = lib.FOLDER.parent
LABELS = ("D5", "D16")
COLOURS = {
    "Observed": "#000000",
    "Normal running": "#306BFF",
    "Prevailing": "#B701FF",
    "Manual heuristic": "#FF4901",
}
WIDTH, HEIGHT = 640, 300
DAY = 48
FITTED_STEP_MW = {"D5": -1.47, "D16": 1.37}
"""The transfer fitted to the telemetry by experiment 580 (transfer 23), in MW."""
START_GUESS = datetime(2025, 12, 29, tzinfo=UTC)
END_GUESS = datetime(2026, 1, 14, tzinfo=UTC)


def to_time(idx: np.ndarray | int) -> object:
    """Return the UTC instant of a half-hour index (a numpy datetime64 or a datetime)."""
    base = np.datetime64(lib.GRID_START.replace(tzinfo=None), "us")
    return base + np.asarray(idx) * np.timedelta64(1800 * 10**6, "us")


def find_step(*, y: np.ndarray, centre: int, span: int = 3 * DAY) -> int:
    """Return the half-hour index near `centre` with the largest change of the 12-hour mean."""
    best, where = -1.0, centre
    for i in range(centre - span, centre + span):
        a = np.nanmean(y[i - 24 : i])
        b = np.nanmean(y[i : i + 24])
        if np.isfinite(a) and np.isfinite(b) and abs(b - a) > best:
            best, where = abs(b - a), i
    return where


def series_figure(
    *, observed: pl.DataFrame, stitched: dict[str, pl.DataFrame], start: int, end: int, label: str
) -> alt.VConcatChart:
    """Return two stacked panels: normal running and prevailing, day-ahead stitched."""
    panels = []
    for name, frame in stitched.items():
        base = alt.Chart(frame.to_pandas()).encode(x=alt.X("time:T", title="Date (UTC)"))  # ty: ignore[unresolved-attribute]
        band = base.mark_area(opacity=0.25, color=COLOURS[name]).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("q05:Q", title=f"{label} power (MW)", scale=alt.Scale(zero=False)), y2="q95:Q"
        )
        med = base.mark_line(color=COLOURS[name], strokeWidth=1.5).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
        heur = base.mark_line(  # ty: ignore[unresolved-attribute]
            color=COLOURS["Manual heuristic"], strokeDash=[4, 3], strokeWidth=1.2
        ).encode(y="heur:Q")
        obs = (
            alt.Chart(observed.to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_line(color=COLOURS["Observed"], strokeWidth=1)
            .encode(x="time:T", y="power:Q")
        )
        rules = (
            alt.Chart(  # ty: ignore[unresolved-attribute]
                pl.DataFrame(
                    {"time": pl.Series(np.array([to_time(start), to_time(end)]))}
                ).to_pandas()
            )
            .mark_rule(color="#7F7F7F", strokeDash=[6, 4])
            .encode(x="time:T")
        )
        panels.append(
            (band + med + heur + obs + rules).properties(
                width=WIDTH,
                height=HEIGHT,
                title=f"{name}: forecast for the next 24 hours, stitched day by day",
            )
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                f"{label}: day-ahead forecasts through the load shift",
                "Black: observed. Colour: median with q05 to q95 band. Orange dashes: manual heuristic. Grey dashes: step start and end.",
            ],
            anchor="start",
            fontSize=13,
        )
    )


def fan_figure(
    *,
    observed: pl.DataFrame,
    fans: dict[str, pl.DataFrame],
    origin: int,
    start: int,
    end: int,
    label: str,
    caption: str,
) -> alt.VConcatChart:
    """Return two stacked panels of the 14-day forecast from one origin."""
    panels = []
    lo, hi = origin - 3 * DAY, origin + 14 * DAY + 1
    obs_frame = observed.filter((pl.col("idx") >= lo) & (pl.col("idx") < hi))
    for name, frame in fans.items():
        base = alt.Chart(frame.to_pandas()).encode(x=alt.X("time:T", title="Date (UTC)"))  # ty: ignore[unresolved-attribute]
        outer = base.mark_area(opacity=0.2, color=COLOURS[name]).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("q05:Q", title=f"{label} power (MW)", scale=alt.Scale(zero=False)), y2="q95:Q"
        )
        inner = base.mark_area(opacity=0.3, color=COLOURS[name]).encode(y="q25:Q", y2="q75:Q")  # ty: ignore[unresolved-attribute]
        med = base.mark_line(color=COLOURS[name], strokeWidth=1.5).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
        obs = (
            alt.Chart(obs_frame.to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_line(color=COLOURS["Observed"], strokeWidth=1)
            .encode(x="time:T", y="power:Q")
        )
        marks = [
            (to_time(origin), "#000000", []),
            (to_time(start), "#7F7F7F", [6, 4]),
            (to_time(end), "#7F7F7F", [6, 4]),
        ]
        rules = alt.layer(
            *[
                alt.Chart(pl.DataFrame({"time": pl.Series(np.array([t]))}).to_pandas())  # ty: ignore[unresolved-attribute]
                .mark_rule(color=c, strokeDash=d)
                .encode(x="time:T")
                for t, c, d in marks
                if to_time(lo) <= t <= to_time(hi)  # ty: ignore[unsupported-operator]
            ]
        )
        panels.append(
            (outer + inner + med + obs + rules).properties(width=WIDTH, height=HEIGHT, title=name)
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                f"{label}: 14-day forecast from {caption}",
                "Black solid line: the forecast origin. Grey dashes: step start and end. Dark band q25 to q75, light band q05 to q95.",
            ],
            anchor="start",
            fontSize=13,
        )
    )


def main(out: Path) -> None:
    """Draw the figures and write the numbers."""
    site = pl.read_csv(next(R.glob("00521-*/site_map.csv")))
    ids = {r["label"]: r["time_series_id"] for r in site.iter_rows(named=True)}
    file = pl.read_parquet(lib.FOLDER / "primaries.parquet").filter(
        pl.col("time_series_id").is_in(list(ids.values()))
    )
    grid_ids = sorted(ids.values())
    grid = lib.cleaned_grid(ids=grid_ids)
    out.mkdir(parents=True, exist_ok=True)
    numbers: dict[str, dict] = {}
    for label in LABELS:
        sid = ids[label]
        row = grid_ids.index(sid)
        y = np.full(grid.shape[1] + 2000, np.nan)
        y[: grid.shape[1]] = grid[row]
        start = find_step(
            y=y,
            centre=lib.grid_index(times=pl.Series([START_GUESS]).dt.replace_time_zone("UTC"))[
                0
            ].item(),
        )
        end = find_step(
            y=y,
            centre=lib.grid_index(times=pl.Series([END_GUESS]).dt.replace_time_zone("UTC"))[
                0
            ].item(),
        )
        observed = (
            pl.DataFrame({"idx": np.arange(start - 20 * DAY, end + 12 * DAY)})
            .with_columns(
                time=pl.Series(to_time(np.arange(start - 20 * DAY, end + 12 * DAY))),  # ty: ignore[invalid-argument-type]
                power=pl.Series(y[start - 20 * DAY : end + 12 * DAY]),
            )
            .drop_nulls("power")
        )
        kinds = {}
        for name, kind in (("Normal running", "NRA"), ("Prevailing", "prevailing")):
            f = file.filter(
                (pl.col("time_series_id") == sid) & (pl.col("NRA_or_prevailing") == kind)
            )
            f = f.with_columns(
                vidx=pl.Series(lib.grid_index(times=f["valid_time"])),
                oidx=pl.Series(lib.grid_index(times=f["power_fcst_init_time"])),
            )
            kinds[name] = f
        stitched = {}
        for name, f in kinds.items():
            s = f.filter(
                (pl.col("vidx") - pl.col("oidx") <= DAY)
                & (pl.col("vidx") >= start - 14 * DAY)
                & (pl.col("vidx") < end + 8 * DAY)
            ).sort("vidx")
            lead = (s["vidx"] - s["oidx"]).to_numpy() / 2.0
            heur = lib.manual_heuristic(
                grid=grid, row=np.full(s.height, row), vidx=s["vidx"].to_numpy(), lead_h=lead
            )[:, 5]
            stitched[name] = s.select(
                time=pl.Series(to_time(s["vidx"].to_numpy())),  # ty: ignore[invalid-argument-type]
                q05="q05",
                q50="q50",
                q95="q95",  # ty: ignore[invalid-argument-type]
            ).with_columns(heur=pl.Series(heur))
        series_figure(observed=observed, stitched=stitched, start=start, end=end, label=label).save(
            out / f"event_{label}_day_ahead.svg"
        )
        available = np.sort(kinds["Prevailing"]["oidx"].unique().to_numpy())
        before = available[available < start]
        inside = available[(available >= start) & (available < end)]
        origins = {
            "before_start": (int(before[-3]), "an origin 3 days before the step start"),
            "early_inside": (int(inside[2]), "an origin 3 days after the step start"),
            "late_inside": (int(inside[-4]), "an origin 4 days before the step end"),
            "last_day": (int(inside[-1]), "the last origin before the step end"),
            "after_end": (
                int(available[available >= end][0]),
                "the first origin after the step end",
            ),
        }
        for tag, (target, caption) in origins.items():
            o = target
            fans = {}
            for name, f in kinds.items():
                s = f.filter(pl.col("oidx") == o).sort("vidx")
                fans[name] = s.select(
                    time=pl.Series(to_time(s["vidx"].to_numpy())),  # ty: ignore[invalid-argument-type]
                    q05="q05",
                    q25="q25",
                    q50="q50",
                    q75="q75",
                    q95="q95",
                )
            fan_figure(
                observed=observed,
                fans=fans,
                origin=o,
                start=start,
                end=end,
                label=label,
                caption=caption,
            ).save(out / f"event_{label}_fan_{tag}.svg")
            numbers.setdefault(label, {}).setdefault("origins", {})[tag] = str(to_time(o))
        # hours until the prevailing output moved and returned
        daily = {}
        for name, f in kinds.items():
            d = (
                f.filter(pl.col("vidx") - pl.col("oidx") <= DAY)
                .group_by("oidx")
                .agg(m=pl.col("q50").mean())
                .sort("oidx")
            )
            daily[name] = d
        dev = (
            daily["Prevailing"]
            .join(daily["Normal running"], on="oidx", suffix="_n")
            .with_columns(dev=pl.col("m") - pl.col("m_n"))
        )
        step = FITTED_STEP_MW[label]
        half = 0.5 * abs(step)
        after_start = dev.filter(pl.col("oidx") >= start - DAY)
        moved = after_start.filter(pl.col("dev") * np.sign(step) >= half)
        returned = dev.filter(pl.col("oidx") >= end).filter(pl.col("dev").abs() < half)
        numbers[label].update(
            {
                "step_start": str(to_time(start)),
                "step_end": str(to_time(end)),
                "step_mw": step,
                "hours_until_prevailing_moved": (float(moved["oidx"].min() - start) / 2.0)  # ty: ignore[unsupported-operator]
                if moved.height
                else None,
                "hours_until_prevailing_returned": (float(returned["oidx"].min() - end) / 2.0)  # ty: ignore[unsupported-operator]
                if returned.height
                else None,
                "daily_deviation_mw": {
                    str(to_time(int(o))): float(v)
                    for o, v in dev.filter(
                        (pl.col("oidx") >= start - 5 * DAY) & (pl.col("oidx") < end + 8 * DAY)
                    )
                    .select("oidx", "dev")
                    .iter_rows()
                },
            }
        )
    (out / "event_numbers.json").write_text(json.dumps(numbers, indent=1))
    print(
        json.dumps(
            {
                k: {x: v for x, v in n.items() if x != "daily_deviation_mw"}
                for k, n in numbers.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
