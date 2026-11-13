r"""Draw experiment 697's published figures F1 to F4 and G1 to G6 and G8 as SVG.

    uv run --no-sync python $E/00697-*/e00697_figures.py

The figures hold Gas A and D8 only as shares over many half-hours: a weekday, a local half-hour, a
calendar month, a price bin, or a spell length. **No date, no year, and no series appears in any
SVG.** The script writes each SVG, runs `npx svgo@4 --multipass --precision=1 --final-newline` over
it, and stops if any SVG text holds a four-digit year. G7 (night share by month) is a dated series
of D8 and is drawn by `e00697_private_figures.py` into the private data folder only.

Every mark sets `aria=False`. Colours come from `plotting.ocf_theme` and are set explicitly. Every
hour axis is labelled 0, 6, 12, 18, and 24. Charts are `CONTENT_WIDTH_PX` wide.
"""

import re
import subprocess
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00697_analysis import (  # ty: ignore[unresolved-import]
    HEATMAP_MIN_DAYS,
    add_price_thirds,
    bin_shares,
    cell_bins,
    day_table,
    heat_map,
    overall_bins,
    record_edges,
    record_levels,
    spell_table,
    usable,
    with_state,
)
from e00697_common import (  # ty: ignore[unresolved-import]
    DOCS_DIR,
    EVENING_FIRST,
    EVENING_LAST,
    MAX_SPELL_LENGTH,
    MIN_DAYS_PER_BIN,
    UNIT_MW,
    hazard_columns,
    pooled_hazard,
)
from e00697_frame import load_frame  # ty: ignore[unresolved-import]

WIDTH: Final[int] = 680
"""`CONTENT_WIDTH_PX` of the docs pages."""
HOUR_VALUES: Final[tuple[int, ...]] = (0, 6, 12, 18, 24)
WEEKDAY_NAMES: Final[tuple[str, ...]] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
BLUE: Final[str] = ocf.DATA_BLUE
ORANGE: Final[str] = ocf.BRAND_ORANGE
SKY: Final[str] = ocf.DATA_SKY
GREEN: Final[str] = ocf.DATA_GREEN
TEXT_NODE: Final = re.compile(r">([^<>]+)<")
"""The text between SVG tags: axis labels, titles, and legend entries."""
YEAR_PATTERN: Final = re.compile(r"\b(?:19|20)\d\d\b")
"""A year in an SVG's text would mean a date reached a published figure."""
SVGO: Final[tuple[str, ...]] = (
    "npx",
    "svgo@4",
    "--multipass",
    "--precision=1",
    "--final-newline",
)
SERIES_COLOURS: Final[dict[str, str]] = {
    "D8, net era": BLUE,
    "D8, solar era": SKY,
    "Null control": ORANGE,
}
"""One colour per group in the charts that compare D8's eras with the null control."""
SERIES_NAMES: Final[dict[str, str]] = {"net": "D8, net era", "solar": "D8, solar era"}
ERA_TITLES: Final[dict[str, str]] = {"net": "Net era", "solar": "Solar era", "fade": "Fade era"}
HALF_TITLES: Final[dict[bool, str]] = {True: "summer half", False: "winter half"}


def series_colour(*, data: pl.DataFrame, column: str = "series", legend: bool = True) -> alt.Color:
    """Return the colour encoding of the series present in `data`, in the group colours."""
    present = list(dict.fromkeys(data[column].to_list()))
    return alt.Color(
        f"{column}:N",
        scale=alt.Scale(domain=present, range=[SERIES_COLOURS[p] for p in present]),
        legend=alt.Legend(title=None, labelLimit=300) if legend else None,
    )


def hour_axis(*, title: str = "Local time of day (hours)") -> alt.Axis:
    """Return an hour axis with ticks at 0, 6, 12, 18, and 24."""
    return alt.Axis(values=list(HOUR_VALUES), title=title, grid=False)


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write an SVG, optimise it with svgo, and refuse it if it holds a year."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / f"{name}.svg"
    chart.save(path)
    subprocess.run([*SVGO, str(path)], check=True, capture_output=True)
    years = YEAR_PATTERN.findall(" ".join(TEXT_NODE.findall(path.read_text())))
    if years:
        raise SystemExit(f"{path.name} holds a four-digit year, so a date may have leaked")
    width = int(re.search(r'width="(\d+)', path.read_text()).group(1))  # ty: ignore[unresolved-attribute]
    print(f"wrote {path.name} ({width} px wide)", flush=True)


def heat_chart(
    *, data: pl.DataFrame, title: str, top: float, colour: str = BLUE, legend: bool = True
) -> alt.Chart:
    """Return one weekday by local-half-hour heat map with a white to `colour` scale."""
    frame = data.with_columns(
        start=pl.col("local_half_hour") / 2,
        stop=(pl.col("local_half_hour") + 1) / 2,
        day=pl.col("weekday").replace_strict(
            dict(zip(range(1, 8), WEEKDAY_NAMES, strict=True)), return_dtype=pl.String
        ),
    )
    return (
        alt.Chart(frame, title=title)  # ty: ignore[unresolved-attribute]
        .mark_rect(aria=False)
        .encode(
            x=alt.X("start:Q", axis=hour_axis(), scale=alt.Scale(domain=[0, 24], nice=False)),
            x2="stop:Q",
            y=alt.Y("day:N", sort=list(WEEKDAY_NAMES), title=None),
            color=alt.Color(
                "share:Q",
                scale=alt.Scale(domain=[0, top], range=["#FFFFFF", colour], clamp=True),
                legend=alt.Legend(title="Running share", format=".0%") if legend else None,
            ),
        )
        .properties(width=150, height=120)
    )


def view_of(*, frame: pl.DataFrame, column: str, era: str | None = None) -> pl.DataFrame:
    """Return a view of a frame's rows (one era, if given) with `column` as the state."""
    rows = frame if era is None else frame.filter(pl.col("era") == era)
    return with_state(frame=rows, column=column)


def halves(*, view: pl.DataFrame) -> dict[bool, pl.DataFrame]:
    """Split a view into its British Summer Time half and its winter half."""
    return {half: view.filter(pl.col("summer_half") == half) for half in (True, False)}


def figure_1(*, frames: dict[str, pl.DataFrame]) -> None:
    """F1: heat maps of D8's flags and the null control's labels, and Gas A's weekday profiles."""
    rows = []
    for era in ("net", "solar"):
        for half, part in halves(view=view_of(frame=frames["d8"], column="ts", era=era)).items():
            rows.append(
                heat_map(view=part).with_columns(panel=pl.lit(f"D8 {era}, {HALF_TITLES[half]}"))
            )
    top = max(float(r["share"].max()) for r in rows if r.height)  # ty: ignore[invalid-argument-type]
    charts = [
        heat_chart(data=r.drop("panel"), title=r["panel"][0], top=top) for r in rows if r.height
    ]
    controls = []
    for half, part in halves(view=view_of(frame=frames["c1_null"], column="ts")).items():
        controls.append(
            heat_chart(
                data=heat_map(view=part),
                title=f"Null control, {HALF_TITLES[half]}",
                top=top,
                colour=ORANGE,
            )
        )
    record = view_of(frame=frames["record"], column="r1")
    record2 = view_of(frame=frames["record"], column="r2")
    profile = pl.concat(
        [
            profile_of(view=v, name=name)
            for v, name in ((record, "1 or more units"), (record2, "2 or more units"))
        ]
    )
    lines = (
        alt.Chart(profile, title="Gas A running share")  # ty: ignore[unresolved-attribute]
        .mark_line(aria=False, interpolate="step-after")
        .encode(
            x=alt.X("start:Q", axis=hour_axis(), scale=alt.Scale(domain=[0, 24], nice=False)),
            y=alt.Y("share:Q", title="Running share", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "name:N",
                scale=alt.Scale(
                    domain=["1 or more units", "2 or more units"], range=[BLUE, ORANGE]
                ),
                legend=alt.Legend(title=None),
            ),
            strokeDash=alt.StrokeDash("class:N", legend=alt.Legend(title=None)),
        )
        .properties(width=380, height=140)
    )
    chart = alt.vconcat(
        alt.hconcat(*charts[:2]),
        alt.hconcat(*charts[2:4]),
        alt.hconcat(*controls),
        lines,
    ).resolve_scale(color="shared")
    save(chart=chart, name="f1_calendar_heat_map")


def profile_of(*, view: pl.DataFrame, name: str) -> pl.DataFrame:
    """Return a view's mean state by local half-hour and weekday class, with no minimum days."""
    return (
        usable(frame=view)
        .group_by("weekend", "local_half_hour")
        .agg(share=pl.col("state").mean())
        .with_columns(
            name=pl.lit(name),
            start=pl.col("local_half_hour") / 2,
            **{
                "class": pl.when(pl.col("weekend"))
                .then(pl.lit("Weekend"))
                .otherwise(pl.lit("Weekday"))
            },
        )
        .sort("weekend", "local_half_hour")
    )


MIN_SPELLS_AT_RISK: Final[int] = 10
"""A hazard is drawn only where at least this many spells reach the length."""


def dwell_table(*, view: pl.DataFrame, name: str, kind: bool) -> pl.DataFrame:
    """Return the survival and the hazard of a view's complete spells for lengths 1 to 48."""
    lengths = spell_table(view=view, kind=kind)["length"].to_numpy()
    if len(lengths) == 0:
        return pl.DataFrame()
    sums = hazard_columns(lengths=lengths).sum(axis=0)
    rows = [
        {
            "series": name,
            "kind": "On-spells" if kind else "Off-spells",
            "length": n,
            "survival": float(np.mean(lengths >= n)),
            "hazard": (
                float(pooled_hazard(sums=sums, first=n, last=n))
                if int(np.sum(lengths >= n)) >= MIN_SPELLS_AT_RISK
                else None
            ),
            "n_spells": len(lengths),
        }
        for n in range(1, MAX_SPELL_LENGTH + 1)
    ]
    return pl.DataFrame(rows)


def figure_2(*, frames: dict[str, pl.DataFrame]) -> None:
    """F2: survival curves and hazards of on-spells and off-spells."""
    sets = [
        ("Gas A, 1 or more units", view_of(frame=frames["record"], column="r1")),
        ("D8 smoothed flags, net era", view_of(frame=frames["d8"], column="ts", era="net")),
        ("D8 filtered flags, net era", view_of(frame=frames["d8"], column="tf", era="net")),
        ("Weekday-class control, truth", view_of(frame=frames["c1_weekday"], column="truth1")),
        ("Weekday-class control, labels", view_of(frame=frames["c1_weekday"], column="ts")),
    ]
    tables = [dwell_table(view=v, name=n, kind=k) for n, v in sets for k in (True, False)]
    data = pl.concat([t for t in tables if t.height])
    names = [n for n, _ in sets]
    scale = alt.Scale(domain=names, range=[GREEN, BLUE, SKY, ORANGE, ORANGE])
    dash = alt.Scale(domain=names, range=[[1, 0], [1, 0], [1, 0], [1, 0], [4, 3]])
    panels = []
    for measure, title in (
        ("survival", "Share lasting n or more"),
        ("hazard", "Share ending at n"),
    ):
        panels.append(
            alt.Chart(data)  # ty: ignore[unresolved-attribute]
            .mark_line(aria=False, interpolate="step-after")
            .encode(
                x=alt.X(
                    "length:Q", title="Spell length n (half-hours)", scale=alt.Scale(domain=[1, 48])
                ),
                y=alt.Y(f"{measure}:Q", title=title, scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "series:N",
                    scale=scale,
                    legend=alt.Legend(title=None, columns=1, labelLimit=300),
                ),
                strokeDash=alt.StrokeDash("series:N", scale=dash, legend=alt.Legend(title=None)),
            )
            .properties(width=150, height=150)
            .facet(row=alt.Row("kind:N", title=None))
        )
    save(chart=alt.hconcat(*panels).resolve_scale(color="shared"), name="f2_dwell")


def price_rows(*, view: pl.DataFrame, name: str) -> list[dict]:
    """Return the running share by price third within cell and by overall price fifth."""
    min_days = 0 if name.startswith("Gas A") else MIN_DAYS_PER_BIN
    out: list[dict] = []
    priced = add_price_thirds(view=view)
    if priced.height:
        thirds = bin_shares(
            view=priced, bins=priced["third"].to_numpy(), label=name, min_days=min_days
        )
        out.extend(
            {"series": name, "panel": "Price third within cell", "bin": b.bin + 1, "share": b.share}
            for b in thirds
        )
    keep = usable(frame=view).filter(pl.col("system_price").is_not_null())
    if keep.height:
        bins = overall_bins(view=keep, column="system_price", n_bins=5)
        fifths = bin_shares(view=keep, bins=bins, label=name, min_days=min_days)
        out.extend(
            {"series": name, "panel": "Overall price fifth", "bin": b.bin + 1, "share": b.share}
            for b in fifths
        )
    return out


def figure_3(*, frames: dict[str, pl.DataFrame]) -> None:
    """F3: running share against system price, within cell and overall."""
    host = frames["host_null"]
    sets = [
        ("Gas A, 1 or more units", view_of(frame=frames["record"], column="r1"), GREEN, [1, 0]),
        ("Gas A, 2 or more units", view_of(frame=frames["record"], column="r2"), GREEN, [4, 3]),
        (
            "D8 flags, net era",
            view_of(frame=frames["d8"], column="ts", era="net"),
            BLUE,
            [1, 0],
        ),
        (
            "D8 flags, solar era",
            view_of(frame=frames["d8"], column="ts", era="solar"),
            SKY,
            [1, 0],
        ),
        (
            "D8 mean probability, net era",
            view_of(frame=frames["d8"], column="p_any_a", era="net"),
            BLUE,
            [4, 3],
        ),
        (
            "D8 mean probability, solar era",
            view_of(frame=frames["d8"], column="p_any_a", era="solar"),
            SKY,
            [4, 3],
        ),
        ("Null control on host series, labels", view_of(frame=host, column="ts"), ORANGE, [1, 0]),
    ]
    data = pl.DataFrame([r for name, view, _, _ in sets for r in price_rows(view=view, name=name)])
    names = [n for n, _, _, _ in sets]
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(aria=False, point=alt.OverlayMarkDef(aria=False, filled=True, size=40))
        .encode(
            x=alt.X("bin:O", title="Price bin (1 cheapest)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("share:Q", title="Running share", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=names, range=[c for _, _, c, _ in sets]),
                legend=alt.Legend(title=None, columns=1, labelLimit=300),
            ),
            strokeDash=alt.StrokeDash(
                "series:N",
                scale=alt.Scale(domain=names, range=[d for _, _, _, d in sets]),
                legend=alt.Legend(title=None),
            ),
        )
        .properties(width=170, height=180)
        .facet(column=alt.Column("panel:N", title=None))
        .resolve_scale(x="independent")
    )
    save(chart=chart, name="f3_price")


def figure_4(*, frames: dict[str, pl.DataFrame]) -> None:
    """F4: Gas A's output levels, ramp shape, and clock times of starts and stops."""
    record = frames["record"]
    power = np.clip(record["power"].to_numpy(), 0.0, None)
    running = power[np.rint(power / UNIT_MW) >= 1]
    hist = pl.DataFrame({"power": running}).with_columns(
        bin=(pl.col("power") / 0.25).floor() * 0.25
    )
    counts = hist.group_by("bin").agg(n=pl.len()).sort("bin")
    rules = pl.DataFrame({"level": [UNIT_MW * k for k in (1, 2, 3, 4)]})
    levels = (
        alt.Chart(counts, title="Gas A output while running")  # ty: ignore[unresolved-attribute]
        .mark_bar(aria=False, color=BLUE, size=6)
        .encode(
            x=alt.X("bin:Q", title="Output (MW), rules at multiples of 1.97 MW"),
            y=alt.Y("n:Q", title="Half-hours"),
        )
        + alt.Chart(rules).mark_rule(aria=False, color=ORANGE).encode(x="level:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=220, height=150)
    edges = record_edges(record=record)
    offsets = edges["offsets"]
    ramp = pl.DataFrame(
        {
            "offset": offsets * 2,  # ty: ignore[unsupported-operator]
            "share": list(edges["start_profile"]) + list(edges["stop_profile"]),  # ty: ignore[invalid-argument-type]
            "edge": ["Spell start"] * len(offsets) + ["Spell stop"] * len(offsets),  # ty: ignore[invalid-argument-type]
        }
    )
    ramp_chart = (
        alt.Chart(ramp, title="Output as a share of the level reached")  # ty: ignore[unresolved-attribute]
        .mark_line(aria=False, point=alt.OverlayMarkDef(aria=False, filled=True, size=40))
        .encode(
            x=alt.X(
                "offset:Q",
                title="Half-hours from the first running (start) or idle (stop) half-hour",
            ),
            y=alt.Y("share:Q", title="Share of level", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "edge:N", scale=alt.Scale(range=[BLUE, ORANGE]), legend=alt.Legend(title=None)
            ),
        )
        .properties(width=220, height=150)
    )
    clock = pl.DataFrame(
        {
            "start": np.tile(np.arange(48) / 2, 2),
            "stop": np.tile((np.arange(48) + 1) / 2, 2),
            "n": list(edges["start_half_hours"]) + list(edges["stop_half_hours"]),  # ty: ignore[invalid-argument-type]
            "edge": ["Spell start"] * 48 + ["Spell stop"] * 48,
        }
    )
    clock_chart = (
        alt.Chart(clock, title="Spell starts and stops by local time")  # ty: ignore[unresolved-attribute]
        .mark_bar(aria=False)
        .encode(
            x=alt.X("start:Q", axis=hour_axis(), scale=alt.Scale(domain=[0, 24], nice=False)),
            x2="stop:Q",
            y=alt.Y("n:Q", title="Spells"),
            color=alt.Color(
                "edge:N", scale=alt.Scale(range=[BLUE, ORANGE]), legend=alt.Legend(title=None)
            ),
        )
        .properties(width=WIDTH - 140, height=130)
    )
    print(f"F4 summary: {record_levels(record=record)}", flush=True)
    save(
        chart=alt.vconcat(alt.hconcat(levels, ramp_chart), clock_chart),
        name="f4_gas_a_levels_and_ramps",
    )


def bar(
    *,
    data: pl.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
    width: int = 170,
    sort: list[str] | None = None,
) -> alt.Chart:
    """Return a grouped line-with-points chart of `y` by `x`, coloured by `color`."""
    return (
        alt.Chart(data, title=title)  # ty: ignore[unresolved-attribute]
        .mark_line(aria=False, point=alt.OverlayMarkDef(aria=False, filled=True, size=40))
        .encode(
            x=alt.X(f"{x}:O", title=None, sort=sort, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{y}:Q", title="Running share", axis=alt.Axis(format=".0%")),
            color=series_colour(data=data, column=color),
        )
        .properties(width=width, height=150)
    )


def gallery_1(*, frames: dict[str, pl.DataFrame]) -> None:
    """G1: running days by weekday, persistence from one weekday to the next, and run lengths."""
    sets = {
        "D8, net era": view_of(frame=frames["d8"], column="ts", era="net"),
        "D8, solar era": view_of(frame=frames["d8"], column="ts", era="solar"),
        "Null control": view_of(frame=frames["c1_null"], column="ts"),
    }
    by_day, persistence = [], []
    for name, view in sets.items():
        days = day_table(view=view)
        by_day.append(
            days.group_by("weekday")
            .agg(share=pl.col("running_day").mean())
            .with_columns(
                series=pl.lit(name),
                day=pl.col("weekday").replace_strict(
                    dict(zip(range(1, 8), WEEKDAY_NAMES, strict=True)), return_dtype=pl.String
                ),
            )
        )
        keys = [c for c in ("seed", "era") if c in days.columns]
        pairs = days.with_columns(
            previous=pl.col("running_day").shift(1).over(*keys)
            if keys
            else pl.col("running_day").shift(1),
            gap=(pl.col("local_date") - pl.col("local_date").shift(1)).dt.total_days(),
        ).filter(
            pl.col("gap") == 1, pl.col("weekday").is_between(2, 5), pl.col("previous").is_not_null()
        )
        persistence.append(
            pairs.group_by("previous")
            .agg(share=pl.col("running_day").mean())
            .with_columns(
                series=pl.lit(name),
                previous_day=pl.when(pl.col("previous"))
                .then(pl.lit("Ran"))
                .otherwise(pl.lit("Idle")),
            )
        )
    left = bar(
        data=pl.concat(by_day),
        x="day",
        y="share",
        color="series",
        title="Days with 4+ running half-hours",
        sort=list(WEEKDAY_NAMES),
    )
    persistence_data = pl.concat(persistence)
    right = (
        alt.Chart(persistence_data, title="Next weekday, by previous")  # ty: ignore[unresolved-attribute]
        .mark_bar(aria=False)
        .encode(
            x=alt.X("previous_day:N", title="Previous weekday", axis=alt.Axis(labelAngle=0)),
            xOffset="series:N",
            y=alt.Y("share:Q", title="Share running", axis=alt.Axis(format=".0%")),
            color=series_colour(data=persistence_data, legend=False),
        )
        .properties(width=110, height=150)
    )
    save(chart=alt.hconcat(left, right), name="g1_running_days")


def gallery_2(*, frames: dict[str, pl.DataFrame]) -> None:
    """G2: start and stop clock times by era and half, for D8's smoothed flags."""
    rows = []
    for era in ("net", "solar"):
        for half, part in halves(view=view_of(frame=frames["d8"], column="ts", era=era)).items():
            spells = spell_table(view=part, kind=True)
            if spells.height == 0:
                continue
            stop = (spells["start_half_hour"] + spells["length"]) % 48
            for edge, hh in (("Start", spells["start_half_hour"]), ("Stop", stop)):
                counts = hh.value_counts()
                for value, n in zip(
                    counts[hh.name].to_list(), counts["count"].to_list(), strict=True
                ):
                    rows.append(
                        {
                            "panel": f"{ERA_TITLES[era]}, {HALF_TITLES[half]}",
                            "edge": edge,
                            "start": value / 2,
                            "stop": (value + 1) / 2,
                            "n": n,
                        }
                    )
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar(aria=False, opacity=0.8)
        .encode(
            x=alt.X("start:Q", axis=hour_axis(), scale=alt.Scale(domain=[0, 24], nice=False)),
            x2="stop:Q",
            y=alt.Y("n:Q", title="Spells"),
            color=alt.Color(
                "edge:N",
                scale=alt.Scale(domain=["Start", "Stop"], range=[BLUE, ORANGE]),
                legend=alt.Legend(title=None),
            ),
        )
        .properties(width=230, height=110)
        .facet(facet=alt.Facet("panel:N", title=None), columns=2)
    )
    save(chart=chart, name="g2_start_stop_times")


def gallery_3(*, frames: dict[str, pl.DataFrame]) -> None:
    """G3: share by price rank within the day, and the red band against the dearest six."""
    rows, band_rows = [], []
    for era in ("net", "solar"):
        view = usable(frame=view_of(frame=frames["d8"], column="ts", era=era)).filter(
            pl.col("price_rank_in_day").is_not_null()
        )
        by_rank = (
            view.with_columns(group=(pl.col("price_rank_in_day") // 6))
            .group_by("group")
            .agg(share=pl.col("state").mean())
            .with_columns(series=pl.lit(SERIES_NAMES[era]), rank=pl.col("group") * 6 + 1)
        )
        rows.append(by_rank)
        weekday = view.filter(~pl.col("weekend"))
        red = pl.col("local_half_hour").is_between(EVENING_FIRST, EVENING_LAST)
        groups = {
            "In the red band": weekday.filter(red),
            "Dearest 6, outside band": weekday.filter(pl.col("dearest_six"), ~red),
            "Other weekday": weekday.filter(~pl.col("dearest_six"), ~red),
            "Bank holiday, red band": weekday.filter(red, pl.col("bank_holiday")),
            "Ordinary day, red band": weekday.filter(
                red, ~pl.col("bank_holiday"), ~pl.col("christmas_week")
            ),
        }
        for name, group in groups.items():
            band_rows.append(
                {
                    "series": SERIES_NAMES[era],
                    "group": name,
                    "share": group["state"].mean() if group.height else None,
                    "days": group["local_date"].n_unique(),
                }
            )
    left = bar(
        data=pl.concat(rows),
        x="rank",
        y="share",
        color="series",
        title="By price rank in the day",
    )
    left = left.encode(
        x=alt.X(
            "rank:O", title="Price rank of the half-hour (1 cheapest)", axis=alt.Axis(labelAngle=0)
        )
    )
    band_data = pl.DataFrame(band_rows).filter(pl.col("share").is_not_null())
    right = (
        alt.Chart(  # ty: ignore[unresolved-attribute]
            band_data,
            title="Red band and dearest six",
        )  # ty: ignore[unresolved-attribute]
        .mark_bar(aria=False)
        .encode(
            y=alt.Y("group:N", title=None, sort=None),
            yOffset="series:N",
            x=alt.X("share:Q", title="Running share", axis=alt.Axis(format=".0%", tickCount=4)),
            color=series_colour(data=band_data, legend=False),
        )
        .properties(width=100, height=170)
    )
    save(chart=alt.hconcat(left, right), name="g3_price_rank_and_red_band")


def gallery_4(*, frames: dict[str, pl.DataFrame]) -> None:
    """G4: share by fifth of spark spread, fifth of gas price, and third of the day-ahead price."""
    rows = []
    for era in ("net", "solar"):
        view = view_of(frame=frames["d8"], column="ts", era=era)
        for column, n_bins, title in (
            ("spark_spread", 5, "Spark spread fifth"),
            ("gas_price_gbp_mwh", 5, "Gas price fifth"),
            ("day_ahead_price", 3, "Day-ahead price third"),
        ):
            keep = usable(frame=view).filter(pl.col(column).is_not_null())
            if keep.height == 0:
                continue
            binned = bin_shares(
                view=keep,
                bins=overall_bins(view=keep, column=column, n_bins=n_bins),
                label=column,
                min_days=MIN_DAYS_PER_BIN,
            )
            rows.extend(
                {"panel": title, "series": SERIES_NAMES[era], "bin": b.bin + 1, "share": b.share}
                for b in binned
            )
    data = pl.DataFrame(rows)
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(aria=False, point=alt.OverlayMarkDef(aria=False, filled=True, size=40))
        .encode(
            x=alt.X("bin:O", title="Bin (1 lowest)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("share:Q", title="Running share", axis=alt.Axis(format=".0%")),
            color=series_colour(data=data),
        )
        .properties(width=150, height=150)
        .facet(column=alt.Column("panel:N", title=None))
        .resolve_scale(x="independent")
    )
    save(chart=chart, name="g4_spark_spread_and_gas")


def gallery_5(*, frames: dict[str, pl.DataFrame]) -> None:
    """G5: D8's share minus the null control's, by third of weather within cell, net era."""
    rows = []
    d8 = view_of(frame=frames["d8"], column="ts", era="net")
    control = view_of(frame=frames["c1_null"], column="ts")
    for column, title in (
        ("temp", "Temperature third"),
        ("wind100", "100 m wind speed third"),
        ("ghi", "Irradiance third"),
    ):
        shares = {}
        for name, view in (("d8", d8), ("control", control)):
            keep = usable(frame=view).filter(pl.col(column).is_not_null())
            shares[name] = {
                b.bin: b.share
                for b in bin_shares(
                    view=keep,
                    bins=cell_bins(view=keep, column=column, n_bins=3),
                    label=column,
                    min_days=MIN_DAYS_PER_BIN,
                )
            }
        for bin_, share in shares["d8"].items():
            if bin_ in shares["control"]:
                rows.append(
                    {"panel": title, "bin": bin_ + 1, "difference": share - shares["control"][bin_]}
                )
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar(aria=False, color=BLUE)
        .encode(
            x=alt.X("bin:O", title="Third within cell (1 lowest)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "difference:Q", title="D8 share minus control share", axis=alt.Axis(format=".0%")
            ),
        )
        .properties(width=170, height=150)
        .facet(column=alt.Column("panel:N", title=None))
    )
    save(chart=chart, name="g5_weather")


def gallery_6(*, frames: dict[str, pl.DataFrame]) -> None:
    """G6: evening share on bank holidays, in Christmas week, and on clock-change days."""
    rows = []
    for era in ("net", "solar"):
        view = view_of(frame=frames["d8"], column="ts", era=era)
        evening = view.filter(
            pl.col("observed"), pl.col("local_half_hour").is_between(EVENING_FIRST, EVENING_LAST)
        )
        weekday = evening.filter(~pl.col("weekend"))
        groups = {
            "Ordinary weekdays": weekday.filter(
                ~pl.col("bank_holiday"), ~pl.col("christmas_week"), ~pl.col("clock_change_day")
            ),
            "Bank-holiday weekdays": weekday.filter(
                pl.col("bank_holiday"), ~pl.col("christmas_week")
            ),
            "Christmas week": evening.filter(pl.col("christmas_week")),
            "Clock-change days": evening.filter(pl.col("clock_change_day")),
        }
        for name, group in groups.items():
            rows.append(
                {
                    "series": SERIES_NAMES[era],
                    "group": name,
                    "share": group["state"].mean() if group.height else None,
                    "days": group["local_date"].n_unique(),
                }
            )
    data = pl.DataFrame(rows).filter(pl.col("share").is_not_null())
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_bar(aria=False)
        .encode(
            y=alt.Y("group:N", title=None, sort=None),
            yOffset="series:N",
            x=alt.X(
                "share:Q", title="Running share, 16:00 to 19:00 local", axis=alt.Axis(format=".0%")
            ),
            color=series_colour(data=data),
        )
        .properties(width=400, height=140)
    )
    save(chart=chart, name="g6_holidays_and_clock_changes")


def gallery_8(*, frames: dict[str, pl.DataFrame]) -> None:
    """G8: month by local half-hour heat maps of D8's weekday flags, and the mean block size."""
    charts = []
    for era in ("net", "solar"):
        view = usable(frame=view_of(frame=frames["d8"], column="ts", era=era)).filter(
            ~pl.col("weekend")
        )
        grid = (
            view.group_by("month", "local_half_hour")
            .agg(
                share=pl.col("state").mean(),
                days=pl.col("local_date").n_unique(),
                block=pl.col("fitted_mw").filter(pl.col("state") > 0.5).mean(),
            )
            .filter(pl.col("days") >= HEATMAP_MIN_DAYS)
            .with_columns(
                start=pl.col("local_half_hour") / 2, stop=(pl.col("local_half_hour") + 1) / 2
            )
        )
        for column, colour, title in (
            ("share", BLUE, "Running share"),
            ("block", GREEN, "Block size (MW)"),
        ):
            charts.append(
                alt.Chart(grid, title=f"{ERA_TITLES[era]}, weekdays")  # ty: ignore[unresolved-attribute]
                .mark_rect(aria=False)
                .encode(
                    x=alt.X(
                        "start:Q", axis=hour_axis(), scale=alt.Scale(domain=[0, 24], nice=False)
                    ),
                    x2="stop:Q",
                    y=alt.Y("month:O", title="Calendar month (1 is January)"),
                    color=alt.Color(
                        f"{column}:Q",
                        scale=alt.Scale(range=["#FFFFFF", colour]),
                        legend=alt.Legend(
                            title=title, format=".0%" if column == "share" else ".1f"
                        ),
                    ),
                )
                .properties(width=150, height=150)
            )
    save(
        chart=alt.vconcat(
            alt.hconcat(*charts[:2]).resolve_scale(color="independent"),
            alt.hconcat(*charts[2:]).resolve_scale(color="independent"),
        ).resolve_scale(color="independent"),
        name="g8_season",
    )


def host_null_frame() -> pl.DataFrame:
    """Return the stacked null-control frames of the C2 hosts, which cover the price period."""
    names = ("c2_d2_null", "c2_d3_null", "c2_d4_null")
    return pl.concat([load_frame(name=name) for name in names], how="diagonal_relaxed")


def main() -> None:
    """Load the frames the figures need and draw every published figure."""
    names = ("d8", "record", "c1_null", "c1_weekday")
    frames = {name: load_frame(name=name) for name in names}
    frames["host_null"] = host_null_frame()
    for step in (
        figure_1,
        figure_2,
        figure_3,
        figure_4,
        gallery_1,
        gallery_2,
        gallery_3,
        gallery_4,
        gallery_5,
        gallery_6,
        gallery_8,
    ):
        step(frames=frames)
    print("no dated figure was written")


if __name__ == "__main__":
    main()
