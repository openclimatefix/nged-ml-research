# ruff: noqa: E501
"""Experiments 704 and 703: the figures of the page (SVG, optimised with svgo).

Usage: python e00704_figures.py [--experiment 704|703] [--months ...] [--fake-arm]

Reads the private tables of `e00704_score.py` and the per-seed forecasts, and writes into
`docs/ml-experiments/<experiment>/`:

- `crps_by_lead_band.svg`: CRPS of every system by lead band.
- `contrasts_by_lead_band.svg`: C1, C2, and C3 by lead band, CRPS and MAE, with intervals.
- `day_ahead.svg` and `fourteen_day.svg`: one typical forecast against the observed power.
- `event_<n>_<label>.svg`: at least six logged events across both metering groups, from 14 days
  before the start to 7 days after the end.

The Primaries are substations, so their labels are shown; no NGED metered generator appears.
`--fake-arm` aliases the missing arms to the control, to test the pipeline.
"""

import argparse
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00668_slices_primaries as s668  # ty: ignore[unresolved-import]
import e00704_score as sc
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX

REPO: Final = Path(__file__).resolve().parents[2]
COLOURS: Final = {
    "control": ocf.DATA_BLUE,
    "stored": ocf.DATA_SKY,
    "proxy": ocf.BRAND_ORANGE,
    "drop": ocf.DATA_PURPLE,
    "climatology": "#555555",
    "heuristic": "#999999",
}
LABELS: Final = {
    "control": "Control (with CAMS)",
    "stored": "Stored forecasts (with CAMS)",
    "proxy": "Arm A (analysis proxy)",
    "drop": "Arm B (columns dropped)",
    "climatology": "Climatology",
    "heuristic": "Manual heuristic",
}
CONTRAST_COLOURS: Final = {
    "C1 proxy minus drop": ocf.BRAND_ORANGE,
    "C2 proxy minus control": ocf.DATA_BLUE,
    "C3 drop minus control": ocf.DATA_PURPLE,
    "N1 replay minus stored (noise floor)": "#555555",
}
BAND_ORDER: Final = ["all", *lib.LEAD_BANDS]
BAND_LABEL: Final = {"all": "All leads"} | {b: b.capitalize() for b in lib.LEAD_BANDS}
WIDTH: Final = CONTENT_WIDTH_PX - 20
SCOPE_TEXT: Final = "16 Primaries, origins at 12:00 UTC"


def wrap(*, text: str, width: int = 110) -> list[str]:
    """Split a title or subtitle into lines of at most `width` characters."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return [*lines, line]


def save(*, chart: alt.TopLevelMixin, path: Path) -> None:
    """Write a chart as SVG and optimise it with svgo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(path)
    subprocess.run(
        ["npx", "--yes", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(path)],
        check=True,
        capture_output=True,
    )
    print("wrote", path, f"({path.stat().st_size / 1000:.0f} KB)")


def score_tables(*, spec: sc.Spec) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read the private score and contrast tables."""
    folder = spec.root / "score"
    return pl.read_csv(folder / "scores_long.csv"), pl.read_csv(folder / "contrasts.csv")


def crps_by_band(*, scores: pl.DataFrame, spec: sc.Spec, out: Path) -> None:
    """Draw the CRPS of every system by lead band."""
    systems = [s for s in COLOURS if s in scores["system"].unique().to_list()]
    data = scores.filter(
        (pl.col("scope_kind") == "band")
        & (pl.col("metric") == "crps")
        & (pl.col("tag") == "seed_average_of_scores")
        & pl.col("system").is_in(systems)
        & (pl.col("scope") != "all")
    ).with_columns(system=pl.col("system").replace(LABELS))
    domain = [LABELS[s] for s in systems]
    chart = (
        alt.Chart(data)
        .mark_line(point=True, strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "scope:N", sort=list(lib.LEAD_BANDS), title="Lead time", axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y(
                "value:Q",
                title="CRPS (points of capacity; smaller is better)",
                scale=alt.Scale(zero=True),
            ),
            color=alt.Color(
                "system:N",
                title=None,
                scale=alt.Scale(domain=domain, range=[COLOURS[s] for s in systems]),
                legend=alt.Legend(orient="top", columns=3),
            ),
            strokeDash=alt.StrokeDash(
                "system:N",
                legend=None,
                scale=alt.Scale(
                    domain=domain,
                    range=[
                        [1, 0] if s not in ("heuristic", "climatology") else [5, 3] for s in systems
                    ],
                ),
            ),
        )
        .properties(
            width=WIDTH - 80,
            height=280,
            title=alt.TitleParams(
                wrap(
                    text=f"CRPS by lead time, experiment {spec.key}: the control, both CAMS-free arms, and two baselines. "
                    f"Mean over seeds of each seed's CRPS; {SCOPE_TEXT}; {len(spec.seeds)} seeds; "
                    "origin months of this screen only (interim).",
                    width=100,
                ),
                anchor="start",
                fontSize=12,
            ),
        )
    )
    save(chart=chart, path=out / "crps_by_lead_band.svg")


def contrasts_by_band(*, contrasts: pl.DataFrame, spec: sc.Spec, out: Path) -> None:
    """Draw C1, C2, and C3 by lead band for CRPS and MAE, with the 98.33% interval."""
    data = contrasts.filter(pl.col("scope_kind").is_in(["all", "band"])).with_columns(
        band=pl.col("scope").replace(BAND_LABEL)
    )
    panels = []
    names = [c for c in CONTRAST_COLOURS if c in data["contrast"].unique().to_list()]
    for metric, title in (("crps", "CRPS"), ("mae", "MAE of the median")):
        part = data.filter(pl.col("metric") == metric)
        base = alt.Chart(part).encode(
            y=alt.Y("band:N", title=None, sort=[BAND_LABEL[b] for b in BAND_ORDER]),
            yOffset=alt.YOffset("contrast:N", sort=names),
            color=alt.Color(
                "contrast:N",
                title=None,
                scale=alt.Scale(domain=names, range=[CONTRAST_COLOURS[n] for n in names]),
                legend=alt.Legend(orient="top", columns=3),
            ),
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(color="#000000", strokeDash=[4, 3])
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        panel = (
            zero
            + base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "lo:Q",
                    title=f"{title}: first arm minus second (points of capacity; below 0 is better for the first arm)",
                ),
                x2="hi:Q",
            )
            + base.mark_point(filled=True, size=50, opacity=1).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
        ).properties(
            width=WIDTH - 110,
            height=190,
            title=alt.TitleParams(title, anchor="start", fontSize=12),
        )
        panels.append(panel)
    chart = (
        alt.vconcat(*panels)
        .resolve_scale(color="shared")
        .properties(
            title=alt.TitleParams(
                wrap(
                    text=f"Change in score between arms by lead time, experiment {spec.key}. Dot: pooled mean difference over rows and seeds. "
                    f"Line: 98.33% interval from resampling whole months and a seed (interim: few months). Dashed line: no change. {SCOPE_TEXT}.",
                    width=100,
                ),
                anchor="start",
                fontSize=12,
            )
        )
    )
    save(chart=chart, path=out / "contrasts_by_lead_band.svg")


# ----------------------------------------------------------------------------- forecasts


def seed_mean_frames(
    *, spec: sc.Spec, months: list[str], fake: bool, control: str
) -> dict[str, pl.DataFrame]:
    """Return each available arm's seed-mean contract rows (MW), with a `lead_h` column.

    `control` is `replay` or `stored`, as in `e00704_score.py`: the key `control` holds the refit of
    the unchanged chain (the files `replay`) or the stored forecasts (the files `control`).
    """
    out: dict[str, pl.DataFrame] = {}
    for key in ("control", "proxy", "drop"):
        arm = "replay" if key == "control" and control == "replay" else key
        have = sc.available(spec=spec, arm=arm, months=months)
        if have != months:
            if not fake:
                print(f"skip arm {arm} in the example figures: fits missing")
                continue
            local = replace(spec, fake=spec.fake | {arm})
        else:
            local = spec
        frame = sc.load_arm(spec=local, arm=arm, months=months)["mean"]
        out[key] = frame.with_columns(
            lead_h=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
        )
    return out


def observed(*, series: int, start: datetime, stop: datetime) -> pl.DataFrame:
    """Return the cleaned power of one Primary in `[start, stop)`."""
    return (
        pl.read_parquet(lib.FOLDER / "ground_truth_primaries.parquet")
        .filter(
            (pl.col("time_series_id") == series)
            & pl.col("valid_time").is_between(start, stop, closed="left")
            & pl.col("drop_reason").is_null()
        )
        .select("valid_time", power=pl.col("power"))
    )


def pick_typical(*, frames: dict[str, pl.DataFrame], rows_month: str) -> tuple[int, datetime]:
    """Pick the signed-MW Primary and origin whose control day-ahead CRPS is closest to the median.

    The choice uses the control only, so it cannot favour an arm. The day ahead is the first 24
    hours of the forecast.
    """
    site = sc.site_map()
    signed = site.filter(pl.col("group").str.starts_with("signed"))["time_series_id"].to_list()
    control = frames["control"].filter(
        (pl.col("lead_h") < 24)
        & pl.col("time_series_id").cast(pl.Int64).is_in(signed)
        & (pl.col("power_fcst_init_time").dt.strftime("%Y-%m") == rows_month)
    )
    truth = (
        pl.read_parquet(lib.FOLDER / "ground_truth_primaries.parquet")
        .filter(pl.col("drop_reason").is_null())
        .select(pl.col("time_series_id").cast(pl.Int64), "valid_time", "power")
    )
    cap = pl.read_parquet(lib.FOLDER / "series_capacity.parquet").select(
        pl.col("time_series_id").cast(pl.Int64), "capacity"
    )
    joined = (
        control.with_columns(pl.col("time_series_id").cast(pl.Int64))
        .join(truth, on=["time_series_id", "valid_time"])
        .join(cap, on="time_series_id")
        .drop_nulls("power")
    )
    y = joined["power"].to_numpy().astype(np.float64) * 100 / joined["capacity"].to_numpy()
    q = (
        joined.select(sc.x667.QCOLS).to_numpy().astype(np.float64)
        * 100
        / joined["capacity"].to_numpy()[:, None]
    )
    scored = joined.select("time_series_id", "power_fcst_init_time").with_columns(
        crps=pl.Series(2 * lib.pinball(y=y, q=q).mean(axis=1))
    )
    per = (
        scored.group_by("time_series_id", "power_fcst_init_time")
        .agg(crps=pl.col("crps").mean(), n=pl.len())
        .filter(pl.col("n") == 48)
    )
    target = float(per["crps"].median())  # ty: ignore[invalid-argument-type]
    best = per.with_columns(gap=(pl.col("crps") - target).abs()).sort("gap").row(0, named=True)
    print(
        f"typical day-ahead case: Primary id {best['time_series_id']}, origin {best['power_fcst_init_time']}, "
        f"CRPS {best['crps']:.2f} against the median {target:.2f} over signed-MW Primaries in {rows_month}"
    )
    return int(best["time_series_id"]), best["power_fcst_init_time"]


def forecast_panel(
    *,
    frames: dict[str, pl.DataFrame],
    series: int,
    origin: datetime,
    hours: float,
    title: str,
    height: int,
    date_format: str,
) -> alt.VConcatChart:
    """Return one stacked panel per arm: the q05 to q95 band, the median, and the observed power."""
    stop = origin + timedelta(hours=hours)
    truth = observed(series=series, start=origin, stop=stop)
    panels = []
    ymax = float(truth["power"].max()) if truth.height else 0.0  # ty: ignore[invalid-argument-type]
    for arm, frame in frames.items():
        part = frame.filter(
            (pl.col("time_series_id").cast(pl.Int64) == series)
            & (pl.col("power_fcst_init_time") == origin)
            & (pl.col("valid_time") < stop)
        )
        ymax = max(ymax, float(part["q95"].max()))  # ty: ignore[invalid-argument-type]
        x = alt.X("valid_time:T", title=None, axis=alt.Axis(format=date_format))
        band = (
            alt.Chart(part)
            .mark_area(opacity=0.3, color=COLOURS[arm])
            .encode(x=x, y=alt.Y("q05:Q", title="Power (MW or MVA)"), y2="q95:Q")  # ty: ignore[unresolved-attribute]
        )
        median = (
            alt.Chart(part).mark_line(color=COLOURS[arm], strokeWidth=1.6).encode(x=x, y="q50:Q")  # ty: ignore[unresolved-attribute]
        )
        obs = alt.Chart(truth).mark_line(color="#000000", strokeWidth=1.2).encode(x=x, y="power:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            (band + median + obs).properties(
                width=WIDTH - 90,
                height=height,
                title=alt.TitleParams(LABELS[arm], anchor="start", fontSize=11),
            )
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(wrap(text=title, width=100), anchor="start", fontSize=12)
    )


# ----------------------------------------------------------------------------- events


def pick_events(*, scored_rows: pl.DataFrame) -> list[dict]:
    """Return at least six events with scored rows, across both metering groups.

    Experiment 699's picks come first: the longest events of the "over 7 days, 2 units or more" set
    of experiment 653 that are not at D5 or D16. Only events with at least 3 days of scored day-ahead
    forecasts qualify, so the four origin months leave few of them. Further logged events fill the
    list to at least six, longest scored stretch first and at most one per Primary, until each
    metering group holds two. The verified transfer at D5 and D16 is added when scored rows cover it.
    """
    site = sc.site_map()
    group = dict(zip(site["label"].to_list(), site["group"].to_list(), strict=True))
    assert set(scored_rows["label"].unique().to_list()) == set(group), (
        "labels differ from the site map"
    )
    in_699 = s668.event_sets(site=site)["event table, over 7 days, 2 units or more"]
    in_699_keys = {(r["label"], r["start"]) for r in in_699.iter_rows(named=True)}
    counts = scored_rows.group_by("label", "vidx").len().select("label", "vidx")

    def inside_days(label: str, a: int, b: int) -> float:
        return (
            counts.filter(
                (pl.col("label") == label) & (pl.col("vidx") >= a) & (pl.col("vidx") < b)
            ).height
            / 48
        )

    candidates = []
    for e in sc.events_all(site=site).iter_rows(named=True):
        if e["label"] in {"D5", "D16"} and (e["label"], e["start"]) in in_699_keys:
            continue
        days = inside_days(e["label"], e["start"], e["end"])
        candidates.append(
            {
                **e,
                "days": (e["end"] - e["start"]) / 48,
                "scored_days": days,
                "group": group[e["label"]],
                "from_699": (e["label"], e["start"]) in in_699_keys,
            }
        )
    verified = {(k, s668.idx_of(a)) for k, (a, _) in s668.VERIFIED.items()}
    candidates = [c for c in candidates if (c["label"], c["start"]) not in verified]
    chosen = [c for c in candidates if c["from_699"] and c["scored_days"] >= 3]
    rest = sorted(
        (c for c in candidates if c not in chosen and c["scored_days"] >= 7),
        key=lambda c: -c["scored_days"],
    )

    def short_of_groups() -> bool:
        groups = [str(c["group"]).split(" ")[0] for c in chosen]
        return len(chosen) < 6 or groups.count("signed-MW") < 2 or groups.count("MVA-only") < 2

    for c in rest:
        if not short_of_groups():
            break
        if c["label"] not in {x["label"] for x in chosen}:
            chosen.append(c)
    for label, (a, b) in s668.VERIFIED.items():
        ia, ib = s668.idx_of(a), s668.idx_of(b)
        if inside_days(label, ia, ib) > 0:
            chosen.append(
                {
                    "label": label,
                    "start": ia,
                    "end": ib,
                    "days": (ib - ia) / 48,
                    "scored_days": inside_days(label, ia, ib),
                    "group": group[label],
                    "from_699": True,
                    "transfer": True,
                }
            )
        else:
            print(f"verified transfer at {label} has no scored rows in these months; not drawn")
    chosen.sort(key=lambda r: (r["label"], r["start"]))
    return chosen


def event_figure(
    *, frames: dict[str, pl.DataFrame], event: dict, ids: dict[str, int], n: int, out: Path
) -> None:
    """Draw one event: observed power and each arm's day-ahead median, with the event shaded."""
    start = lib.GRID_START + timedelta(minutes=30 * event["start"])
    end = lib.GRID_START + timedelta(minutes=30 * event["end"])
    lo, hi = start - timedelta(days=14), end + timedelta(days=7)
    series = ids[event["label"]]
    truth = observed(series=series, start=lo, stop=hi)
    parts = [
        truth.select(
            "valid_time", value=pl.col("power"), source=pl.lit("Observed"), segment=pl.lit("obs")
        )
    ]
    for arm, frame in frames.items():
        part = frame.filter(
            (pl.col("time_series_id").cast(pl.Int64) == series)
            & (pl.col("lead_h") < 24)
            & pl.col("valid_time").is_between(lo, hi, closed="left")
        )
        parts.append(
            part.select(
                "valid_time",
                value=pl.col("q50"),
                source=pl.lit(LABELS[arm]),
                segment=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
            )
        )
    data = pl.concat(parts)
    domain = ["Observed", *(LABELS[a] for a in frames)]
    colours = ["#000000", *(COLOURS[a] for a in frames)]
    shade = (
        alt.Chart(pl.DataFrame({"a": [max(start, lo)], "b": [min(end, hi)]}))
        .mark_rect(color="#D9D9D9", opacity=0.6)
        .encode(x="a:T", x2="b:T")  # ty: ignore[unresolved-attribute]
    )
    ends = (
        alt.Chart(pl.DataFrame({"t": [start, end]}))
        .mark_rule(color="#555555", strokeDash=[4, 3])
        .encode(x="t:T")  # ty: ignore[unresolved-attribute]
    )
    lines = (
        alt.Chart(data)
        .mark_line(strokeWidth=1.2)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("valid_time:T", title=None, axis=alt.Axis(format="%d %b %Y")),
            y=alt.Y("value:Q", title="Power (MW or MVA)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "source:N",
                title=None,
                scale=alt.Scale(domain=domain, range=colours),
                legend=alt.Legend(orient="top", columns=2),
            ),
            detail="segment:N",
        )
    )
    kind = "verified transfer" if event.get("transfer") else "logged event"
    group = "signed MW" if event["group"].startswith("signed") else "MVA only"
    chart = (shade + ends + lines).properties(
        width=WIDTH - 90,
        height=230,
        title=alt.TitleParams(
            wrap(
                text=f"{event['label']} ({group} metering), {kind} of {event['days']:.0f} days. Grey and dashed lines: the event's start and end. "
                "Lines: observed power, and the median of the forecast issued at 12:00 UTC for the next 24 hours (seed mean), from 14 days before the start to 7 days after the end. "
                "Forecasts exist only in the scored origin months, so lines break between them. NGED's logs may miss events.",
                width=100,
            ),
            anchor="start",
            fontSize=12,
        ),
    )
    save(chart=chart, path=out / f"event_{n}_{event['label']}.svg")


def main() -> None:
    """Draw every figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="704", choices=("704", "703"))
    parser.add_argument("--months", nargs="+", default=list(sc.MONTHS))
    parser.add_argument("--fake-arm", action="store_true")
    parser.add_argument("--control", choices=("replay", "stored"), default="replay")
    args = parser.parse_args()
    spec = sc.spec_of(key=args.experiment)
    months = sorted(args.months)
    out = REPO / "docs" / "ml-experiments" / spec.name
    scores, contrasts = score_tables(spec=spec)
    crps_by_band(scores=scores, spec=spec, out=out)
    contrasts_by_band(contrasts=contrasts, spec=spec, out=out)
    frames = seed_mean_frames(spec=spec, months=months, fake=args.fake_arm, control=args.control)
    typical_month = "2026-04" if "2026-04" in months else months[-1]
    series, origin = pick_typical(frames=frames, rows_month=typical_month)
    label = sc.site_map().filter(pl.col("time_series_id") == series)["label"][0]
    forecast_panel_args = {"frames": frames, "series": series, "origin": origin}
    save(
        chart=forecast_panel(
            **forecast_panel_args,
            hours=24,
            height=140,
            date_format="%H:%M",
            title=f"{label}, forecast issued at 12:00 UTC on {origin:%d %b %Y}: the first 24 hours (the day ahead). Black: observed power. "
            "Colour: median and q05 to q95 band of each arm (seed mean). Origin chosen as the typical case: the signed-MW Primary and origin whose "
            f"control CRPS over the first 24 hours is closest to the median of {typical_month}.",
        ),
        path=out / "day_ahead.svg",
    )
    save(
        chart=forecast_panel(
            **forecast_panel_args,
            hours=14 * 24,
            height=140,
            date_format="%d %b",
            title=f"{label}, the full 14-day forecast issued at 12:00 UTC on {origin:%d %b %Y}. Black: observed power. "
            "Colour: median and q05 to q95 band of each arm (seed mean). Power is in MW at a signed-MW Primary.",
        ),
        path=out / "fourteen_day.svg",
    )
    # scored rows with labels, for the event selection
    ctl = frames["control"].filter(pl.col("lead_h") < 24)
    site = sc.site_map()
    lab = dict(zip(site["time_series_id"].to_list(), site["label"].to_list(), strict=True))
    scored_rows = pl.DataFrame(
        {
            "label": [lab[int(t)] for t in ctl["time_series_id"].to_list()],
            "vidx": lib.grid_index(times=ctl["valid_time"]),
        }
    )
    events = pick_events(scored_rows=scored_rows)
    print(f"events drawn: {len(events)}")
    ids = dict(zip(site["label"].to_list(), site["time_series_id"].to_list(), strict=True))
    for n, event in enumerate(events, start=1):
        print(n, event)
        event_figure(frames=frames, event=event, ids=ids, n=n, out=out)
    print("done", datetime.now(UTC).isoformat())


if __name__ == "__main__":
    main()
