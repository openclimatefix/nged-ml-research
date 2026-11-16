# ruff: noqa: E501
"""Experiment 699: the figures of the page.

Usage: python e00699_figures.py OUTPUT_FOLDER

- `contrasts.svg`: the planned contrasts of families H, S2, and S3 and of the clean chain on MAE and CRPS,
  with both intervals.
- `copied_share.svg`: the copied-share contrast of the same families on the 581 bank (and on the 660 bank when run).
- `transfer_<D5|D16>.svg`: the verified transfer between D5 and D16, with a day-ahead and a 14-day-ahead panel.
- `event_<n>.svg`: the four longest logged events of the set "over 7 days, 2 units or more" outside D5 and D16.

The Primaries are substations, so their labels are shown; no NGED metered generator appears.
Run with the environment of `run699.sh`.
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import altair as alt
import e00521_stats as s521  # ty: ignore[unresolved-import]
import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00668_slices_primaries as s668  # ty: ignore[unresolved-import]
import pandas as pd  # noqa: TID251  (Altair takes pandas frames)
import polars as pl
from e00699_common import ARMS, FOLDS
from e00699_contrasts import families, load

WIDTH = 620
COLOUR = {
    "No injection": "#306BFF",
    "Stage 3 only": "#0B215F",
    "Stages 1 to 3": "#FF4901",
    "Stages 2 and 3": "#B701FF",
}
ARM_LABEL = {
    "none": "No injection",
    "s3": "Stage 3 only",
    "s123": "Stages 1 to 3",
    "s23": "Stages 2 and 3",
}
FAMILY_LABEL = {
    "H": "H: stages 1 to 3 minus stage 3 only",
    "S2": "S2: stages 2 and 3 minus stage 3 only",
    "S3": "S3: stages 1 to 3 minus stages 1 and 2",
    "clean": "Clean chain: stage 3 only minus no injection",
}


def contrast_rows(*, out: dict, keys: tuple[str, ...]) -> pd.DataFrame:  # ty: ignore[unresolved-attribute]
    """Return one row per family and quantity with the point and both intervals."""
    rows = []
    for family, label in FAMILY_LABEL.items():
        for key, name in (
            ("mae", "MAE"),
            ("crps", "CRPS"),
            ("share581", "Copied share, 581 bank"),
            ("share660", "Copied share, 660 bank"),
        ):
            iv = out.get(family, {}).get(key)
            if iv is None or key not in keys:
                continue
            rows.append(
                {
                    "family": label,
                    "quantity": name,
                    "point": iv.point,
                    "boot_lo": iv.boot_lo,
                    "boot_hi": iv.boot_hi,
                    "t_lo": iv.t_lo,
                    "t_hi": iv.t_hi,
                }
            )
    return pd.DataFrame(rows)  # ty: ignore[unresolved-attribute]


def interval_chart(
    *, frame: pd.DataFrame, title: list[str], x_title: str, colours: dict[str, str]
) -> alt.Chart:  # ty: ignore[unresolved-attribute]
    """Return the dots, the bootstrap interval (thick line), and the t interval (thin line)."""
    base = alt.Chart(frame).encode(
        y=alt.Y(
            "family:N", title=None, sort=list(FAMILY_LABEL.values()), axis=alt.Axis(labelLimit=330)
        ),
        yOffset="quantity:N",
        color=alt.Color(
            "quantity:N",
            title=None,
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(orient="top"),
        ),
    )
    zero = (
        alt.Chart(pd.DataFrame({"x": [0.0]}))
        .mark_rule(color="#555555", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = (
        zero
        + base.mark_rule(strokeWidth=1).encode(x=alt.X("t_lo:Q", title=x_title), x2="t_hi:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_rule(strokeWidth=4).encode(x="boot_lo:Q", x2="boot_hi:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=60, color="#000000").encode(x="point:Q")  # ty: ignore[unresolved-attribute]
    )
    return chart.properties(
        width=WIDTH - 340,
        height=210,
        title=alt.TitleParams(title, anchor="start", fontSize=12),
    )


def contrast_figures(*, out: dict, folds: str, path: Path) -> None:
    """Write the MAE and CRPS figure and the copied-share figure."""
    score = contrast_rows(out=out, keys=("mae", "crps"))
    interval_chart(
        frame=score,
        title=[
            f"Change in score from injecting shifts in more training stages, {folds}",
            "Score of the first arm minus the second, in points of capacity; below 0 is better for the first arm.",
            "Dot: estimate. Thick line: 98.33% bootstrap interval over months and seeds. Thin line: 98.33% t interval",
            "over months. Dashed line: no change. Eight seeds per arm; 16 Primaries; all rows.",
        ],
        x_title="Change in score (points of capacity; below 0 is better)",
        colours={"MAE": "#306BFF", "CRPS": "#FF4901"},
    ).save(path / "contrasts.svg")
    share = contrast_rows(out=out, keys=("share581", "share660"))
    interval_chart(
        frame=share,
        title=[
            f"Change in the copied share of an injected lag shift, {folds}",
            "Copied share: the slope of the median forecast's change on the shift, 1 means a full copy. Below 0 is better for the",
            "first arm. Dot: estimate. Thick line: 98.33% bootstrap interval. Thin line: 98.33% t interval. Dashed line: no change.",
        ],
        x_title="Change in copied share (below 0 is better)",
        colours={"Copied share, 581 bank": "#306BFF", "Copied share, 660 bank": "#10C5F7"},
    ).save(path / "copied_share.svg")


def event_rect(*, start: datetime, stop: datetime, label: str) -> alt.LayerChart:
    """Return a light grey rectangle over a logged event, with a short label."""
    frame = pd.DataFrame({"start": [start], "stop": [stop], "label": [label]})  # ty: ignore[unresolved-attribute]
    rect = alt.Chart(frame).mark_rect(color="#D9D9D9", opacity=0.6).encode(x="start:T", x2="stop:T")  # ty: ignore[unresolved-attribute]
    text = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_text(align="left", baseline="top", dx=4, dy=2, fontSize=11, color="#555555")
        .encode(x="start:T", y=alt.value(0), text="label:N")
    )
    return alt.layer(rect, text)  # ty: ignore[invalid-return-type]


def forecasts(*, arm: str, series: int, start: datetime, stop: datetime) -> pl.DataFrame:
    """Return the seed-mean median forecast of one arm for one series, with its origin."""
    path = ARMS[arm].folder / "backtests" / f"{ARMS[arm].file_stem}.parquet"
    return (
        pl.scan_parquet(path)
        .filter(
            (pl.col("time_series_id") == series)
            & pl.col("valid_time").is_between(start, stop, closed="left")
        )
        .group_by("valid_time", "power_fcst_init_time")
        .agg(pl.col("q50").mean())
        .collect()
    )


def panel(
    *,
    series: int,
    start: datetime,
    stop: datetime,
    event: tuple[datetime, datetime],
    lead_day: int | None,
    title: str,
) -> alt.LayerChart:
    """Return the observed power and the three arms' forecasts; `lead_day` None draws one full 14-day forecast."""
    truth = (
        pl.read_parquet(lib.FOLDER / "ground_truth_primaries.parquet")
        .filter(
            (pl.col("time_series_id") == series)
            & pl.col("valid_time").is_between(start, stop, closed="left")
        )
        .select("valid_time", pl.col("power").alias("value"))
        .with_columns(source=pl.lit("Observed"))
    )
    parts = [truth]
    middle = event[0] + (event[1] - event[0]) / 2
    for arm in ("none", "s3", "s123"):
        f = forecasts(arm=arm, series=series, start=start, stop=stop)
        if lead_day is None:
            origin = f.filter(pl.col("power_fcst_init_time") <= middle)[
                "power_fcst_init_time"
            ].max()
            f = f.filter(pl.col("power_fcst_init_time") == origin)
        else:
            f = f.with_columns(
                lead=((pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours() / 24)
                .floor()
                .cast(pl.Int32)
            ).filter(pl.col("lead") == lead_day)
        parts.append(
            f.select("valid_time", pl.col("q50").alias("value")).with_columns(
                source=pl.lit(ARM_LABEL[arm])
            )
        )
    frame = pl.concat(parts).to_pandas()
    domain = ["Observed", *(ARM_LABEL[a] for a in ("none", "s3", "s123"))]
    lines = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("valid_time:T", title=None),
            y=alt.Y("value:Q", title="Power (MW or MVA)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "source:N",
                title=None,
                scale=alt.Scale(
                    domain=domain,
                    range=[
                        "#000000",
                        COLOUR["No injection"],
                        COLOUR["Stage 3 only"],
                        COLOUR["Stages 1 to 3"],
                    ],
                ),
                legend=alt.Legend(orient="top", columns=4),
            ),
        )
    )
    return alt.layer(  # ty: ignore[invalid-return-type]
        event_rect(start=event[0], stop=event[1], label="logged event"), lines
    ).properties(width=WIDTH, height=240, title=alt.TitleParams(title, anchor="start", fontSize=12))


def event_figure(
    *, name: str, series: int, event: tuple[datetime, datetime], path: Path, note: str
) -> None:
    """Write a figure of one event with a day-ahead panel and a full 14-day-ahead forecast panel."""
    start, stop = event[0] - timedelta(days=14), event[1] + timedelta(days=14)
    top = panel(
        series=series, start=start, stop=stop, event=event, lead_day=1,
        title=f"{name}: median forecast 1 to 2 days ahead, seed mean, from 14 days before to 14 days after the event",
    )  # fmt: skip
    mid = event[0] + (event[1] - event[0]) / 2 - timedelta(days=7)
    bottom = panel(
        series=series, start=mid, stop=mid + timedelta(days=14), event=event, lead_day=None,
        title=f"{name}: one full 14-day median forecast, issued at 12:00 UTC in the middle of the event. {note}",
    )  # fmt: skip
    alt.vconcat(top, bottom).resolve_scale(color="shared").save(path)


def pick_events(*, site: pl.DataFrame) -> list[tuple[str, int, tuple[datetime, datetime]]]:
    """Return the four longest logged events that need no data beyond the scored folds, outside D5 and D16."""
    sets = s668.event_sets(site=site)
    table = sets["event table, over 7 days, 2 units or more"]
    first = datetime(2025, 11, 15, tzinfo=UTC)
    last = datetime(2026, 6, 15, tzinfo=UTC)
    group_of = dict(zip(site["label"].to_list(), site["group"].to_list(), strict=True))
    id_of = dict(zip(site["label"].to_list(), site["time_series_id"].to_list(), strict=True))
    rows = []
    for e in table.iter_rows(named=True):
        a = lib.GRID_START + timedelta(minutes=30 * e["start"])
        b = lib.GRID_START + timedelta(minutes=30 * e["end"])
        if e["label"] in {"D5", "D16"} or a < first or b > last:
            continue
        rows.append(
            {
                "label": e["label"],
                "group": group_of[e["label"]],
                "a": a,
                "b": b,
                "days": (b - a).days,
            }
        )
    rows.sort(key=lambda r: -r["days"])
    chosen: list[dict] = []
    for g in sorted(
        {r["group"] for r in rows}
    ):  # at least two from each metering group where the set allows
        chosen += [r for r in rows if r["group"] == g][:2]
    for r in rows:
        if len(chosen) >= 4:
            break
        if r not in chosen:
            chosen.append(r)
    chosen = sorted(chosen, key=lambda r: -r["days"])[:4]
    return [
        (f"{r['label']}, {r['days']} days", id_of[r["label"]], (r["a"], r["b"])) for r in chosen
    ]


def main() -> None:
    """Write every figure."""
    path = Path(sys.argv[1])
    path.mkdir(parents=True, exist_ok=True)
    arms = [k for k in ("none", "s3", "s23", "s12", "s123") if (ARMS[k].folder / "scores").exists()]
    frame = load(arms=arms)
    out = families(frame=frame)
    contrast_figures(
        out=out,
        folds=f"{len(frame.months)} folds ({frame.months[0]} to {frame.months[-1]})",
        path=path,
    )
    site = pl.read_csv(s521.DATA / "site_map.csv")
    id_of = dict(zip(site["label"].to_list(), site["time_series_id"].to_list(), strict=True))
    for name in ("D5", "D16"):
        event_figure(
            name=f"{name}, the verified transfer", series=id_of[name],
            event=s668.VERIFIED[name], path=path / f"transfer_{name}.svg",
            note="Grey: the confirmed transfer.",
        )  # fmt: skip
    for n, (name, series, event) in enumerate(pick_events(site=site), start=1):
        event_figure(
            name=name, series=series, event=event, path=path / f"event_{n}.svg",
            note="Grey: the logged event.",
        )  # fmt: skip
    print("figures written to", path, "for folds", FOLDS[0], "to", FOLDS[-1])


if __name__ == "__main__":
    main()
