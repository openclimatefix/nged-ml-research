r"""Draw experiment 698's four published figures as SVG.

    uv run --no-sync python $E/00698-*/e00698_figures.py

1. D8's solar-era curve of median reading by fifth of CAMS solar output, beside each host's and
   capacity's curve for the three solar shapes and for no solar.
2. The level contrast and the weather contrasts for the three criteria, with 95% intervals.
3. Precision, recall, and F1 by local hour for the solar-free twin, the smooth twin, and the
   farm run.
4. The three label-only checks of D8's labels by era, with and without experiment 159's drops.

Figures show pooled scores and shares only. **No date, no year, and no Gas A or farm series appears
in any SVG.** The script writes each SVG, runs `npx svgo@4 --multipass --precision=1
--final-newline` over it, and stops if any SVG text holds a four-digit year. Every mark sets
`aria=False`. Colours are the five main colours of `plotting.ocf_theme`, set explicitly.
"""

import re
import subprocess
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00698_common import (  # ty: ignore[unresolved-import]
    AUDIT_PATH,
    CAPACITIES_MW,
    CRITERIA,
    DOCS_DIR,
    GATE_PATH,
    HOST_KEYS,
    PV_FREE,
    SCORES_PATH,
    THRESHOLDS,
    capacity_key,
    read_json,
)
from e00698_score import metrics  # ty: ignore[unresolved-import]

WIDTH: Final[int] = 680
"""`CONTENT_WIDTH_PX` of the docs pages."""
HOUR_VALUES: Final[tuple[int, ...]] = (0, 6, 12, 18, 24)
BLUE: Final[str] = ocf.DATA_BLUE
ORANGE: Final[str] = ocf.BRAND_ORANGE
SKY: Final[str] = ocf.DATA_SKY
PURPLE: Final[str] = ocf.DATA_PURPLE
GREEN: Final[str] = ocf.DATA_GREEN
TEXT_NODE: Final = re.compile(r">([^<>]+)<")
"""The text between SVG tags: axis labels, titles, and legend entries."""
ARIA_LABEL: Final = re.compile(r'aria-label="([^"]*)"')
"""Accessibility labels, which Vega writes for each mark unless the mark sets `aria=False`."""
YEAR_PATTERN: Final = re.compile(r"\b(?:19|20)\d\d\b")
"""A year in an SVG's text would mean a date reached a published figure."""
SVGO: Final[tuple[str, ...]] = ("npx", "svgo@4", "--multipass", "--precision=1", "--final-newline")
HOST_TITLES: Final[dict[str, str]] = {"d8pre": "D8 before the engine", "d3solar": "D3, solar era"}
SHAPE_NAMES: Final[dict[str, str]] = {
    "farm": "Farm output",
    "cams": "CAMS output",
    "smooth": "Smooth twin",
    PV_FREE: "No solar",
    "reference": "D8, solar era",
}
SHAPE_COLOURS: Final[dict[str, str]] = {
    "reference": BLUE,
    PV_FREE: GREEN,
    "farm": ORANGE,
    "cams": SKY,
    "smooth": PURPLE,
}
CONTRAST_NAMES: Final[dict[str, str]] = {
    "level": "Level (smooth twin minus no solar)",
    "weather_farm": "Weather, farm (farm minus smooth twin)",
    "weather_cams": "Weather, CAMS (CAMS minus smooth twin)",
}
CONTRAST_COLOURS: Final[dict[str, str]] = {
    "level": BLUE,
    "weather_farm": ORANGE,
    "weather_cams": SKY,
}
CRITERION_TITLES: Final[dict[str, str]] = {
    "f1": "F1 falls (points)",
    "false_flag": "False-flag share rises (points)",
    "bmd": "Bright-minus-dull flagged share rises (points)",
}
ERA_COLOURS: Final[dict[str, str]] = {"net": BLUE, "solar": ORANGE}
ERA_NAMES: Final[dict[str, str]] = {"net": "Net era", "solar": "Solar era"}
CHECK_TITLES: Final[dict[str, str]] = {
    "weekend_cheap": (
        "Flagged share, weekends and bank holidays, 09:00 to 15:00, negative spark spread"
    ),
    "night": "Flagged share, 00:00 to 05:00",
    "bmd": "Flagged share, brightest third minus dullest third of CAMS irradiance, 10:00 to 15:00",
}


def save(*, chart: alt.TopLevelMixin, name: str, directory: Path = DOCS_DIR) -> Path:
    """Write an SVG, optimise it with svgo, and refuse it if its text holds a year."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.svg"
    chart.save(path)
    subprocess.run([*SVGO, str(path)], check=True, capture_output=True)
    assert_no_year(text=path.read_text(), name=path.name)
    print(f"wrote {path.name}", flush=True)
    return path


def assert_no_year(*, text: str, name: str) -> None:
    """Stop if any text node or accessibility label of an SVG holds a four-digit year."""
    years = YEAR_PATTERN.findall(" ".join([*TEXT_NODE.findall(text), *ARIA_LABEL.findall(text)]))
    if years:
        msg = f"{name} holds a four-digit year ({years[:3]}), so a date may have leaked"
        raise SystemExit(msg)


def colour_scale(
    *, present: list[str], colours: dict[str, str], names: dict[str, str]
) -> alt.Scale:
    """Return a colour scale over the series present, in the group colours."""
    return alt.Scale(domain=[names[p] for p in present], range=[colours[p] for p in present])


def figure_1(*, gate: dict, directory: Path = DOCS_DIR) -> Path:
    """F1: the curve of median reading by fifth of CAMS solar output, D8's beside the hosts'."""
    reference = gate["reference"]["medians"]
    panels = []
    for host in HOST_KEYS:
        for cap in CAPACITIES_MW:
            key = capacity_key(capacity_mw=cap)
            curves = gate["curves"][host][key]
            rows = [("reference", i + 1, v) for i, v in enumerate(reference)]
            rows += [
                (PV_FREE, i + 1, v) for i, v in enumerate(gate["curves"][host]["none"]["free"])
            ]
            for shape, doc in curves.items():
                rows += [(shape, i + 1, v) for i, v in enumerate(doc["medians"])]
            data = pl.DataFrame(
                {
                    "shape": [SHAPE_NAMES[s] for s, _, _ in rows],
                    "fifth": [f for _, f, _ in rows],
                    "reading": [v for _, _, v in rows],
                }
            ).filter(pl.col("reading").is_not_nan())
            present = list(dict.fromkeys(s for s, _, _ in rows))
            panels.append(
                alt.Chart(data, title=f"{HOST_TITLES[host]}, {cap} MW")  # ty: ignore[unresolved-attribute]
                .mark_line(point=True, aria=False)
                .encode(
                    x=alt.X(
                        "fifth:O",
                        title="Fifth of CAMS solar output (1 lowest, 5 highest)",
                        axis=alt.Axis(labelAngle=0),
                    ),
                    y=alt.Y("reading:Q", title="Median reading (MVA)"),
                    color=alt.Color(
                        "shape:N",
                        scale=colour_scale(
                            present=present, colours=SHAPE_COLOURS, names=SHAPE_NAMES
                        ),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                )
                .properties(width=WIDTH - 90, height=130)
            )
    chart = alt.vconcat(*panels).resolve_scale(color="shared")
    chart = chart.properties(
        title=alt.TitleParams(
            "Median reading by fifth of solar output, May to September, 10:00 to 15:00 local time",
            subtitle=[
                "Lines: median reading in MVA of each solar curve at one capacity, with no engine.",
                "A run whose curve lies within 1 MVA of D8's in fifths 1 to 4 passes the gate.",
            ],
        )
    )
    return save(chart=chart, name="f1_curve_gate", directory=directory)


def figure_2(*, scores: dict, directory: Path = DOCS_DIR) -> Path:
    """F2: the level and weather contrasts for the three criteria, with intervals."""
    rows = []
    for host, doc in scores["hosts"].items():
        for key, table in doc["contrasts"].items():
            for name in CONTRAST_NAMES:
                labels = table.get(name)
                if labels is None:
                    continue
                for criterion in CRITERIA:
                    interval = labels["smooth"][criterion]
                    rows.append(
                        {
                            "criterion": criterion,
                            "run": f"{HOST_TITLES[host]}, {int(key[1:]) / 10:.1f} MW",
                            "contrast": CONTRAST_NAMES[name],
                            "point": interval["point"],
                            "lo": interval["lo"],
                            "hi": interval["hi"],
                        }
                    )
    data = pl.DataFrame(rows)
    panels = []
    present = [
        CONTRAST_NAMES[c] for c in CONTRAST_NAMES if CONTRAST_NAMES[c] in data["contrast"].to_list()
    ]
    scale = alt.Scale(
        domain=present,
        range=[CONTRAST_COLOURS[c] for c in CONTRAST_NAMES if CONTRAST_NAMES[c] in present],
    )
    for criterion in CRITERIA:
        part = data.filter(pl.col("criterion") == criterion)
        base = alt.Chart(part, title=CRITERION_TITLES[criterion])  # ty: ignore[unresolved-attribute]
        marks = base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("run:N", title=None),
            yOffset=alt.YOffset("contrast:N"),
            x=alt.X("lo:Q", title="Harm (positive is worse)"),
            x2="hi:Q",
            color=alt.Color("contrast:N", scale=scale, legend=alt.Legend(title=None, orient="top")),
        )
        points = base.mark_point(filled=True, aria=False, size=40).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("run:N"),
            yOffset=alt.YOffset("contrast:N"),
            x="point:Q",
            color=alt.Color("contrast:N", scale=scale, legend=None),
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(aria=False, color="#888888")
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        threshold = (
            alt.Chart(pl.DataFrame({"x": [THRESHOLDS[criterion]]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(aria=False, strokeDash=[4, 3], color="#888888")
            .encode(x="x:Q")
        )
        panels.append((marks + points + zero + threshold).properties(width=WIDTH - 220, height=110))
    chart = (
        alt.vconcat(*panels)
        .resolve_scale(color="shared")
        .properties(
            title=alt.TitleParams(
                "Harm to the tracker's labels from the level and from solar weather",
                subtitle=[
                    "Dot: estimate. Line: 95% interval, whole weeks resampled. Solid grey: zero.",
                    "Dashed grey: harm threshold. Scores at 10:00 to 15:00, 3 seeds pooled.",
                ],
            )
        )
    )
    return save(chart=chart, name="f2_contrasts", directory=directory)


def figure_3(*, scores: dict, directory: Path = DOCS_DIR) -> Path | None:
    """F3: precision, recall, and F1 by local hour for the free twin, smooth twin, and farm run."""
    doc = scores["hosts"].get("d8pre")
    if doc is None:
        return None
    key = next(
        (
            capacity_key(capacity_mw=c)
            for c in CAPACITIES_MW
            if f"{capacity_key(capacity_mw=c)}_farm" in doc["cells"]
        ),
        None,
    )
    if key is None:
        return None
    runs = {PV_FREE: f"none_{PV_FREE}", "smooth": f"{key}_smooth", "farm": f"{key}_farm"}
    rows = []
    for shape, cell in runs.items():
        groups = doc["cells"][cell]["smooth"]["groups"]["hour"]
        for hour, sums in groups.items():
            m = metrics(sums=sums)
            rows += [
                {
                    "shape": SHAPE_NAMES[shape],
                    "hour": int(hour) + 0.5,
                    "score": m[name],
                    "metric": title,
                }
                for name, title in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))
            ]
    data = pl.DataFrame(rows).filter(pl.col("score").is_not_nan())
    present = [SHAPE_NAMES[s] for s in runs]
    scale = alt.Scale(domain=present, range=[SHAPE_COLOURS[s] for s in runs])
    panels = [
        alt.Chart(data.filter(pl.col("metric") == metric), title=metric)  # ty: ignore[unresolved-attribute]
        .mark_line(aria=False)
        .encode(
            x=alt.X(
                "hour:Q",
                axis=alt.Axis(
                    values=list(HOUR_VALUES), title="Local time of day (hours)", grid=False
                ),
                scale=alt.Scale(domain=[0, 24], nice=False),
            ),
            y=alt.Y("score:Q", title=metric, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("shape:N", scale=scale, legend=alt.Legend(title=None, orient="top")),
        )
        .properties(width=WIDTH - 90, height=110)
        for metric in ("Precision", "Recall", "F1")
    ]
    chart = (
        alt.vconcat(*panels)
        .resolve_scale(color="shared")
        .properties(
            title=alt.TitleParams(
                "Tracker scores by local hour at D8 before the engine",
                subtitle=[
                    f"Smoothed labels, 3 seeds of the injected engine, {int(key[1:]) / 10:.1f} MW.",
                    "Higher is better. Lines: no solar, the smooth twin, and the farm's output.",
                ],
            )
        )
    )
    return save(chart=chart, name="f3_scores_by_hour", directory=directory)


def figure_4(*, audit: dict, directory: Path = DOCS_DIR) -> Path:
    """F4: the three label-only checks by era, with and without drops, with intervals."""
    result = audit["numbers"]["label_only"]
    rows = []
    for check, table in result.items():
        for era in ("net", "solar"):
            for drops, label in (("True", "With drops"), ("False", "Without drops")):
                est = table.get(era, {}).get(drops)
                if est is None:
                    continue
                rows.append(
                    {
                        "check": check,
                        "era": ERA_NAMES[era],
                        "drops": label,
                        "point": est["point"],
                        "lo": est["lo"],
                        "hi": est["hi"],
                    }
                )
    data = pl.DataFrame(rows)
    scale = alt.Scale(
        domain=[ERA_NAMES[e] for e in ("net", "solar")],
        range=[ERA_COLOURS[e] for e in ("net", "solar")],
    )
    panels = []
    for check, title in CHECK_TITLES.items():
        part = data.filter(pl.col("check") == check)
        base = alt.Chart(part, title=title)  # ty: ignore[unresolved-attribute]
        rules = base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("drops:N", title=None, sort=["With drops", "Without drops"]),
            yOffset=alt.YOffset("era:N"),
            x=alt.X("lo:Q", title="Flagged share (a share of half-hours)"),
            x2="hi:Q",
            color=alt.Color("era:N", scale=scale, legend=alt.Legend(title=None, orient="top")),
        )
        points = base.mark_point(filled=True, aria=False, size=40).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("drops:N", sort=["With drops", "Without drops"]),
            yOffset=alt.YOffset("era:N"),
            x="point:Q",
            color=alt.Color("era:N", scale=scale, legend=None),
        )
        panels.append((rules + points).properties(width=WIDTH - 120, height=70))
    chart = (
        alt.vconcat(*panels)
        .resolve_scale(color="shared", x="independent")
        .properties(
            title=alt.TitleParams(
                "D8's labels in the net era and in the solar era",
                subtitle=[
                    "Dot: estimate. Line: 95% interval, whole weeks resampled. Smoothed labels.",
                    "Drops are the multi-day demand drops of experiment 159.",
                ],
            )
        )
    )
    return save(chart=chart, name="f4_label_only_checks", directory=directory)


def main() -> None:
    """Draw the four figures from the stored gate, scores, and audit."""
    gate = read_json(path=GATE_PATH)
    scores = read_json(path=SCORES_PATH)
    audit = read_json(path=AUDIT_PATH)
    figure_1(gate=gate)
    figure_2(scores=scores)
    figure_3(scores=scores)
    figure_4(audit=audit)
    _ = np  # numpy is imported for the figures' callers


if __name__ == "__main__":
    main()
