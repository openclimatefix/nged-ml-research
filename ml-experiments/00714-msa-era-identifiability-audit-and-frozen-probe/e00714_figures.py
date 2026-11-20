"""Experiment 714: the three figures.

    .venv/bin/python ml-experiments/00714-*/e00714_figures.py [--smoke]

- `auroc_by_feature_set.svg`: AUROC by input set and horizon, with 99.17% intervals, for the second
  generator and for experiment 709's bank.
- `example_hosts.svg`: six example hosts from 14 days before to 7 days after the onset, three
  with a permanent
  change and three with a temporary move, anonymised (the label is a letter, the axis is days).
- `change_profile.svg`: the change by half-hour of the day (post-onset power minus the real power
  that the Primary would have had), for the temporary moves and each permanent kind, each profile
  divided by its own mean so that a constant change plots at 1.

Each SVG is written under `docs/ml-experiments/<id>-<slug>/` (or the private results folder with
`--smoke`) and then optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.
"""

import e00714_threads  # noqa: F401  (must run before numpy)

# isort: split

import argparse
import json
import subprocess
from pathlib import Path
from typing import Final

import altair as alt
import e00714_common as common
import e00714_generator2 as g2
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / (f"{common.EXPERIMENT_ID:05d}-{common.SLUG}")
)
WIDTH: Final[int] = 700
SETS: Final[dict[str, tuple[str, str]]] = {
    "host": ("Host rows", theme.DATA_BLUE),
    "host_old": ("Host rows plus six older rows", theme.DATA_GREEN),
    "host_nbr": ("Host rows plus neighbour rows", theme.BRAND_ORANGE),
}
KIND_ORDER: Final[tuple[str, ...]] = (
    "temporary move", "flat", "profile_prop", "daylight", "quantised", "intermittent"
)  # fmt: skip
KIND_LABEL: Final[dict[str, str]] = {
    "temporary move": "Temporary move",
    "flat": "Permanent, flat",
    "profile_prop": "Permanent, follows daily profile",
    "daylight": "Permanent, daylight only",
    "quantised": "Permanent, quantised",
    "intermittent": "Permanent, intermittent (secondary)",
}
KIND_COLOUR: Final[tuple[str, ...]] = (
    theme.BLACK_1, theme.DATA_BLUE, theme.DATA_GREEN, theme.BRAND_ORANGE, theme.DATA_PURPLE,
    theme.DATA_SKY,
)  # fmt: skip


def save(*, chart: alt.TopLevelMixin, path: Path) -> None:
    """Write a chart as SVG and optimise the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(path))
    subprocess.run(
        ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(path)],
        check=True, capture_output=True,
    )  # fmt: skip


def auroc_figure(*, results: Path, out: Path) -> None:
    """Draw AUROC by input set and horizon with intervals, for both worlds."""
    data = json.loads((results / "primary_contrasts.json").read_text())["auroc"]
    rows = []
    for key, value in data.items():
        world, k, name = key.split("/")
        if name in SETS:
            rows.append(
                {"world": "Second generator" if world == "g2" else "Experiment 709's bank",
                 "days": int(k[1:]), "set": SETS[name][0], **value}
            )  # fmt: skip
    frame = pl.DataFrame(rows)
    colour = alt.Color(
        "set:N", title=None,
        scale=alt.Scale(domain=[v[0] for v in SETS.values()], range=[v[1] for v in SETS.values()]),
        legend=alt.Legend(orient="top"),
    )  # fmt: skip
    panels = []
    for world in frame["world"].unique(maintain_order=True).to_list():
        sub = frame.filter(pl.col("world") == world)
        base = alt.Chart(sub).encode(
            x=alt.X("days:O", title="Days after the onset", axis=alt.Axis(labelAngle=0)),
            xOffset="set:N",
            color=colour,
        )  # ty: ignore[unresolved-attribute]
        bars = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("low:Q", title="AUROC", scale=alt.Scale(domain=[0.3, 1.0])), y2="high:Q"
        )
        points = base.mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
            y="auroc:Q"
        )
        rule = (
            alt.Chart(pl.DataFrame({"y": [0.7]}))
            .mark_rule(strokeDash=[4, 3], color="grey")
            .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            alt.layer(bars, points, rule).properties(width=WIDTH, height=200, title=world)
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Area under the ROC curve for telling a permanent change from a temporary move",
            subtitle=[
                "Dashed line: the threshold of 0.70.",
                "Bars: 99.17% intervals over passes and fold months.",
            ],
        )
    )
    save(chart=chart, path=out / "auroc_by_feature_set.svg")


def _example_events(*, ctx: common.Context) -> list[dict]:
    """Pick three permanent and three temporary events of the second generator, in order."""
    picks = [("era", "flat"), ("era", "profile_prop"), ("era", "daylight"),
             ("switch", "full"), ("switch", "partial"), ("switch", "hidden")]  # fmt: skip
    chosen = []
    month = "2026-01"
    for truth, what in picks:
        for index in g2.stored_indices(world="g2", month=month):
            built = g2.read_pass(world="g2", month=month, index=index)
            sel = built.truth.filter(
                (pl.col("truth") == truth)
                & ((pl.col("kind") == what) | (pl.col("mirror_state") == what))
                & (pl.col("size").abs() > 0.1)
                & (pl.col("duration_days").is_null() | (pl.col("duration_days") > 14))
            )
            if sel.height:
                chosen.append({"built": built, "event": sel.row(0, named=True), "what": what})
                break
    return chosen


def example_figure(*, ctx: common.Context, out: Path) -> None:
    """Draw the last 14 days before the origin of six example hosts."""
    rows = []
    for letter, item in zip("ABCDEF", _example_events(ctx=ctx), strict=False):
        event, built = item["event"], item["built"]
        onset = event["onset_hh"]
        lo, end = onset - 14 * common.DAY, onset + 7 * common.DAY
        power = built.power.get(site=event["host"], lo=lo, hi=end) * 100.0
        days = (np.arange(lo, end) - onset) / common.DAY
        label = f"Host {letter}: {'permanent' if event['truth'] == 'era' else 'temporary'}"
        rows += [
            {"host": label, "days": float(d), "power": float(p), "onset": 0.0}
            for d, p in zip(days, power, strict=True)
            if np.isfinite(p)
        ]
    frame = pl.DataFrame(rows)
    line = (
        alt.Chart(frame)
        .mark_line(color=theme.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "days:Q",
                title="Days relative to the onset",
                scale=alt.Scale(domain=[-14, 7]),
            ),
            y=alt.Y("power:Q", title="Power (% of capacity)", scale=alt.Scale(zero=False)),
        )
    )
    onset = (
        alt.Chart(frame)
        .mark_rule(color=theme.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x=alt.X("onset:Q", scale=alt.Scale(domain=[-14, 7])))  # ty: ignore[unresolved-attribute]
    )
    chart = (
        alt.layer(line, onset)
        .properties(width=WIDTH, height=110)
        .facet(row=alt.Row("host:N", title=None))
    )
    chart = chart.properties(
        title=alt.TitleParams(
            "From 14 days before to 7 days after the onset at six hosts, second generator",
            subtitle=["Dashed orange line: the onset of the injected event. Hosts are anonymised."],
        )
    )
    save(chart=chart, path=out / "example_hosts.svg")


def profile_figure(*, ctx: common.Context, out: Path) -> None:
    """Draw the change by half-hour of the day for the temporary moves and each permanent kind."""
    sums: dict[str, list[np.ndarray]] = {k: [] for k in KIND_ORDER}
    for world in ("g2", "gas"):
        for month in common.MONTHS:
            for index in g2.stored_indices(world=world, month=month):  # ty: ignore[invalid-argument-type]
                built = g2.read_pass(world=world, month=month, index=index)  # ty: ignore[invalid-argument-type]
                for ev in built.truth.iter_rows(named=True):
                    key = "temporary move" if ev["truth"] == "switch" else ev["kind"]
                    if ev["truth"] == "switch" and ev["duration_days"] < 14:
                        continue
                    on, host = ev["onset_hh"], ev["host"]
                    inj = built.power.get(site=host, lo=on, hi=on + 14 * common.DAY)
                    real = ctx.panel.power[host, on : on + 14 * common.DAY]
                    grid = (inj - real).reshape(14, common.DAY)
                    profile = np.nanmean(grid, axis=0)
                    mean = float(np.mean(profile))
                    if np.isfinite(mean) and abs(mean) > 1e-4:
                        sums[key].append(profile / mean)
    rows = []
    for key, profiles in sums.items():
        if not profiles:
            continue
        median = np.nanmedian(np.stack(profiles), axis=0)
        rows += [
            {"kind": KIND_LABEL[key], "hour": h / 2, "relative": float(v)}
            for h, v in enumerate(median)
        ]
    frame = pl.DataFrame(rows)
    labels = [KIND_LABEL[k] for k in KIND_ORDER if KIND_LABEL[k] in frame["kind"].to_list()]
    colours = [c for k, c in zip(KIND_ORDER, KIND_COLOUR, strict=True) if KIND_LABEL[k] in labels]
    chart = (
        alt.Chart(frame)
        .mark_line()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("hour:Q", title="Hour of day (UTC)", axis=alt.Axis(values=[0, 6, 12, 18, 24])),
            y=alt.Y(
                "relative:Q",
                title="Change relative to its daily mean",
                scale=alt.Scale(domain=[0.0, 2.5]),
            ),
            color=alt.Color(
                "kind:N",
                title=None,
                scale=alt.Scale(domain=labels, range=colours),
                legend=alt.Legend(orient="top", columns=2),
            ),
        )
        .properties(
            width=WIDTH,
            height=260,
            title=alt.TitleParams(
                "The change by half-hour of the day, by kind of event",
                subtitle=[
                    "Median over injected events of the post-onset power minus the real power,",
                    "each divided by its own daily mean.",
                ],
            ),
        )
    )
    save(chart=chart, path=out / "change_profile.svg")


def main() -> None:
    """Draw the figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        g2.use_folder(name="worlds_smoke")
    results = common.DATA_DIR / ("results_smoke" if args.smoke else "results")
    out = results / "figures" if args.smoke else DOCS
    ctx = common.load_context()
    auroc_figure(results=results, out=out)
    example_figure(ctx=ctx, out=out)
    profile_figure(ctx=ctx, out=out)


if __name__ == "__main__":
    main()
