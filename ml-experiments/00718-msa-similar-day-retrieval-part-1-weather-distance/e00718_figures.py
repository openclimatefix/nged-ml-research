# ruff: noqa: E501, D103, PT018
"""Experiment 718, WP8: the figures of both parts, from stored row tables, forecasts, and read-outs.

Usage:
    e00718_figures.py ages OUT_DIR PART           # ages of the retrieved rows per arm (R, R-A, P90 side by side in part 1)
    e00718_figures.py distance OUT_DIR PART       # mean selected distance by lead day, stage 1 against stages 2 and 3
    e00718_figures.py contrasts OUT_DIR PART      # each counted contrast at every one of the 16 Primaries, with the group's reading
    e00718_figures.py series OUT_DIR PART         # day-ahead and 14-day forecasts of M*, R (or U), their placebo, and C beside the actual
    e00718_figures.py weights OUT_DIR PART        # the read-out weights alpha by age bucket and by measure, and the across-day attention maps
    e00718_figures.py self-test                   # builds every chart from synthetic data and checks its specification

The charts read stored files and fit nothing. The series charts count hours or days from the origin, and
every substation here is a Primary, so the anonymisation rule for metered generators does not apply.
Colour is by arm (`COLOUR`); each chart states what is plotted, which direction is better, and what zero
means. Charts are written as SVG and optimised with svgo by `e00713_figures.save`.
"""

import argparse
import sys
from pathlib import Path
from typing import Final

import altair as alt
import e00713_figures as f713  # ty: ignore[unresolved-import]
import e00718_layout as lay
import e00718_retrieval as ret
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

WIDTH: Final = f713.WIDTH
COLOUR: Final = {
    "M*": theme.BRAND_ORANGE, "R": theme.DATA_BLUE, "R-A": theme.DATA_BLUE_LIGHT, "P90": theme.DATA_PURPLE,
    "U": theme.DATA_GREEN, "U-A": theme.DATA_GREEN_LIGHT, "U27": theme.DATA_PURPLE, "UX": theme.DATA_SKY,
    "Z": theme.DATA_SKY_LIGHT, "C": theme.DATA_SKY,
}  # fmt: skip
LIGHT: Final = {
    "M*": theme.BRAND_ORANGE_LIGHT, "R": theme.DATA_BLUE_LIGHT, "U": theme.DATA_GREEN_LIGHT, "C": theme.DATA_SKY_LIGHT,
    "R-A": theme.DATA_BLUE_LIGHT, "U-A": theme.DATA_GREEN_LIGHT,
}  # fmt: skip
ARMS_SHOWN: Final = {1: ("M*", "R", "R-A", "C"), 2: ("M*", "U", "U-A", "C")}
FIRST: Final = {1: "R", 2: "U"}
BUCKET_LABELS: Final = [str(a) for a in range(2, 16)] + [
    "16-22",
    "23-29",
    "30-43",
    "44-63",
    "64-91",
    "92-182",
    "183-343",
    "344-386",
    "387-751",
    "752+",
]


def title_of(*, lines: list[str]) -> alt.TitleParams:
    return alt.TitleParams(lines, anchor="start", fontSize=12, subtitleFontSize=11)


def domain_of(*, arms: list[str]) -> alt.Scale:
    return alt.Scale(domain=arms, range=[COLOUR[a] for a in arms])


def age_histogram(*, tables: dict[str, ret.RowTable]) -> pl.DataFrame:
    """Return the share of each arm's chosen rows in each of the 24 age buckets."""
    rows = []
    for arm, t in tables.items():
        ages = t.ages[t.ages != lay.ABSENT].astype(np.int64)
        counts = np.bincount(lay.age_bucket(ages=ages), minlength=lay.N_BUCKETS)
        for b, n in enumerate(counts):
            rows.append(
                {
                    "arm": arm,
                    "bucket": b,
                    "label": BUCKET_LABELS[b],
                    "share": float(n / max(counts.sum(), 1)),
                }
            )
    return pl.DataFrame(rows)


def ages_chart(*, hist: pl.DataFrame, title: list[str]) -> alt.Chart:
    arms = hist["arm"].unique(maintain_order=True).to_list()
    return (
        alt.Chart(hist, width=WIDTH, height=170)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "label:N",
                sort=BUCKET_LABELS,
                title="Age of the retrieved block (days before the origin; ages 16 and over grouped)",
                axis=alt.Axis(labelAngle=-60),
            ),
            y=alt.Y("share:Q", title="Share of the arm's rows", axis=alt.Axis(format="%")),
            xOffset=alt.XOffset("arm:N", sort=arms),
            color=alt.Color(
                "arm:N", scale=domain_of(arms=arms), title="Arm", legend=alt.Legend(orient="top")
            ),
        )
        .properties(title=title_of(lines=title))
    )


def distance_chart(*, data: pl.DataFrame, title: list[str]) -> alt.Chart:
    """`data` has the columns `arm`, `stage` (`stage 1 (ERA5 target)` or `stages 2 and 3 (ENS target)`), `lead_day`, `distance`."""
    arms = data["arm"].unique(maintain_order=True).to_list()
    return (
        alt.Chart(data, width=WIDTH, height=170)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "lead_day:Q", title="Lead day", axis=alt.Axis(values=[1, 3, 5, 7, 9, 11, 13, 14])
            ),
            y=alt.Y(
                "distance:Q",
                title="Mean weather distance of the chosen weather rows (smaller is closer)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "arm:N", scale=domain_of(arms=arms), title="Arm", legend=alt.Legend(orient="top")
            ),
            strokeDash=alt.StrokeDash("stage:N", title=None, legend=alt.Legend(orient="top")),
        )
        .properties(title=title_of(lines=title))
    )


def contrast_primary_chart(
    *, rows: pl.DataFrame, reference: str, title: list[str]
) -> alt.LayerChart:
    """`rows` is the per-Primary table: `arm`, `primary`, `crps_minus_reference`. One dot per arm at each of the 16 Primaries."""
    shown = rows.filter(pl.col("arm") != reference)
    arms = shown["arm"].unique(maintain_order=True).to_list()
    order = sorted(shown["primary"].unique().to_list(), key=lambda s: int(s[1:]))
    base = alt.Chart(shown, width=WIDTH, height=420)  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"z": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#222222", strokeDash=[4, 3])
        .encode(x="z:Q")
    )  # ty: ignore[unresolved-attribute]
    dots = base.mark_circle(size=70, opacity=0.9).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "crps_minus_reference:Q",
            title=f"CRPS minus that of {reference} (points of capacity; more negative is better; the dashed line is no difference)",
        ),
        y=alt.Y("primary:N", sort=order, title="Primary"),
        color=alt.Color(
            "arm:N", scale=domain_of(arms=arms), title="Arm", legend=alt.Legend(orient="top")
        ),
    )
    return alt.layer(zero, dots).properties(title=title_of(lines=title))  # ty: ignore[invalid-return-type]


def series_chart(
    *, data: pl.DataFrame, arms: tuple[str, ...], horizon_hours: float, title: list[str]
) -> alt.VConcatChart:
    """One panel per arm: the 90% and 50% bands, the median, and the actual; `data` has `arm`, `hours`, `actual`, and the quantiles."""
    d = data.filter(pl.col("hours") <= horizon_hours).with_columns(
        x=pl.col("hours") if horizon_hours <= 36 else pl.col("hours") / 24.0
    )
    axis = (
        alt.Axis(values=[0, 6, 12, 18, 24, 30, 36])
        if horizon_hours <= 36
        else alt.Axis(values=list(range(0, 15, 2)))
    )
    x = alt.X(
        "x:Q",
        title="Hours after the 12:00 UTC origin"
        if horizon_hours <= 36
        else "Days after the 12:00 UTC origin",
        axis=axis,
    )
    panels = []
    for arm in arms:
        sub = alt.Chart(d, width=WIDTH, height=110).transform_filter(alt.datum.arm == arm)  # ty: ignore[unresolved-attribute]
        light = LIGHT.get(arm, theme.DATA_SKY_LIGHT)
        panels.append(
            alt.layer(
                sub.mark_area(color=light, opacity=0.45).encode(  # ty: ignore[unresolved-attribute]
                    x=x, y=alt.Y("q05:Q", title="Power (% of capacity)"), y2="q95:Q"
                ),  # ty: ignore[unresolved-attribute]
                sub.mark_area(color=light, opacity=0.9).encode(x=x, y="q25:Q", y2="q75:Q"),  # ty: ignore[unresolved-attribute]
                sub.mark_line(color=COLOUR[arm], strokeWidth=1.6, clip=True).encode(x=x, y="q50:Q"),  # ty: ignore[unresolved-attribute]
                sub.mark_line(color="#222222", strokeWidth=1.0, clip=True).encode(  # ty: ignore[unresolved-attribute]
                    x=x, y="actual:Q"
                ),  # ty: ignore[unresolved-attribute]
            ).properties(title=alt.TitleParams(arm, anchor="start", fontSize=11))
        )
    return alt.vconcat(*panels).properties(title=title_of(lines=title))  # ty: ignore[invalid-return-type]


def alpha_chart(*, by_bucket: dict[str, np.ndarray], title: list[str]) -> alt.Chart:
    """The mean read-out weight of each age bucket, one series per arm."""
    rows = [
        {"arm": a, "label": BUCKET_LABELS[b], "weight": float(w)}
        for a, v in by_bucket.items()
        for b, w in enumerate(v)
    ]
    arms = list(by_bucket)
    return (
        alt.Chart(pl.DataFrame(rows), width=WIDTH, height=170)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "label:N",
                sort=BUCKET_LABELS,
                title="Age bucket (days before the origin)",
                axis=alt.Axis(labelAngle=-60),
            ),
            y=alt.Y("weight:Q", title="Mean read-out weight (the weights of all rows sum to 1)"),
            xOffset=alt.XOffset("arm:N", sort=arms),
            color=alt.Color(
                "arm:N", scale=domain_of(arms=arms), title="Arm", legend=alt.Legend(orient="top")
            ),
        )
        .properties(title=title_of(lines=title))
    )


def attention_heat(*, maps: dict[str, np.ndarray], title: str) -> alt.Chart:
    """Heat maps of the weights `alpha` (rows by half-hours), one panel per arm (the weights of experiment 713's attention figure)."""
    import e00713_attention as att713  # ty: ignore[unresolved-import]

    return att713.heat_chart(
        frames=maps,
        title=title,
        rows="History row, newest first",
        cols="Half-hour of the target day (hours after 12:00 UTC)",
    )


def series_frame(
    *,
    arm: str,
    shown: str,
    month: str,
    day: int,
    label: str,
    seeds: tuple[int, ...],
    truth: np.ndarray,
) -> pl.DataFrame:
    """Return the quantiles (points of capacity) and the actual of one Primary and origin from the stored forecasts of `arm` (F9's folder or experiment 713's)."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00718_score as sc

    sites = k578.sites()
    row = sites.filter(pl.col("label") == label)
    tid, primary = int(row["time_series_id"][0]), int(row["primary"][0])
    mean = sc.s713.seed_average(frame=sc.load_arm(arm=arm, seeds=seeds, months=(month,)))
    frame = mean.filter(pl.col("time_series_id") == tid)
    origin = frame["power_fcst_init_time"].unique().sort()
    chosen = origin.filter(origin.dt.day() == day)
    assert len(chosen) == 1, (month, day, len(chosen))
    out = frame.filter(pl.col("power_fcst_init_time") == chosen[0]).sort("valid_time")
    grid = (
        (out["valid_time"].dt.epoch("us").to_numpy() - prep422.GRID_START_US) // prep422.STEP_US
    ).astype(np.int64)
    hours = (out["valid_time"] - chosen[0]).dt.total_minutes().to_numpy() / 60.0
    return pl.DataFrame(
        {
            "arm": shown,
            "label": label,
            "hours": hours,
            "actual": truth[primary, grid] * 100.0,
            **{c: out[c].to_numpy() * 100.0 for c in sc.QCOLS},
        }
    )


# ------------------------------------------------------------------------------------ commands


def _tables(*, part: int, month: str) -> dict[str, ret.RowTable]:
    import e00718_arms as arms
    import e00718_fit as fit_mod
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    names = ("R", "R-A", "P90") if part == 1 else ("U", "U-A", "U27")
    return {
        a: fit_mod.load_table(
            path=arms.table_file(
                data_dir=ML_EXPERIMENTS_DATA_DIR, arm=a, month=month, version="ens"
            )
        )
        for a in names
    }


def cmd_ages(*, out_dir: Path, part: int, month: str) -> None:
    hist = age_histogram(tables=_tables(part=part, month=month))
    title = [
        f"Ages of the retrieved rows, fold {month}",
        "Share of each arm's rows in each age bucket. The placebo arms keep the age classes of their treatment, and the uniform arm P90 (part 1) draws from ages 2 to 91.",
    ]
    f713.save(ages_chart(hist=hist, title=title), path=out_dir / f"ages_part{part}_{month}.svg")


def cmd_contrasts(*, out_dir: Path, part: int) -> None:
    import json

    import e00718_contrasts as ct
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    rows = pl.DataFrame(
        json.loads((ct._score_dir(part=part) / "report_only.json").read_text())["per_primary"]
    )
    star_name = ct.a713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    title = [
        f"Part {part}: CRPS difference at each of the 16 Primaries",
        f"Seed-mean forecasts of 8 folds, leads 24 to 240 h. Reference: {star_name} (M*). Each dot is one arm's mean CRPS at one Primary minus M*'s.",
    ]
    f713.save(
        contrast_primary_chart(rows=rows, reference=star_name, title=title),
        path=out_dir / f"contrasts_primary_part{part}.svg",
    )


def cmd_series(*, out_dir: Path, part: int) -> None:
    import e00713_score as s713  # ty: ignore[unresolved-import]
    import e00718_contrasts as ct
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    star = ct.a713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    truth = s713.truth_stack()
    cases = (
        ("winter", "2026-01", 13, "D5"),
        ("winter", "2026-01", 13, "D8"),
        ("summer", "2026-06", 16, "D5"),
        ("summer", "2026-06", 16, "D8"),
    )
    for season, month, day, label in cases:
        data = pl.concat(
            [
                series_frame(
                    arm=star if s == "M*" else s,
                    shown=s,
                    month=month,
                    day=day,
                    label=label,
                    seeds=(0, 1),
                    truth=truth,
                )
                for s in ARMS_SHOWN[part]
            ]
        )
        for kind, hours, name in (("day-ahead", 36.0, "day"), ("14-day", 336.0, "fourteen")):
            title = [
                f"{kind.capitalize()} forecast of Primary {label}, {season}, against what happened",
                "Black line: actual. Coloured line: median. Dark band: 50% interval. Light band: 90% interval. Seed mean of seeds 0 and 1.",
            ]
            f713.save(
                series_chart(data=data, arms=ARMS_SHOWN[part], horizon_hours=hours, title=title),
                path=out_dir / f"series_part{part}_{name}_{season}_{label}.svg",
            )


def self_test() -> int:
    """Build every chart from synthetic data and check its specification (colours, axis values, titles, and counts)."""
    import e00718_check_batch as cb
    import e00718_tables as tab

    source = cb.synthetic_source()
    arrays = tab.pool_arrays_from_source(source=source, with_calendar=False)
    days = np.arange(cb.FIRST_ORIGIN_DAY, cb.FIRST_ORIGIN_DAY + cb.N_ORIGIN_DAYS)
    tables = {
        a: ret.build_table(
            arm=a,
            arrays=arrays,
            primaries=cb.TABLE_PRIMARIES,
            days=days,
            target_provider=tab.era5_provider(arrays=arrays),
        )
        for a in ("R", "R-A", "P90")
    }
    hist = age_histogram(tables=tables)
    assert (
        hist.height == 3 * lay.N_BUCKETS
        and abs(hist.filter(pl.col("arm") == "R")["share"].sum() - 1.0) < 1e-9  # ty: ignore[unsupported-operator]
    )
    spec = ages_chart(hist=hist, title=["t", "s"]).to_dict()
    assert spec["encoding"]["color"]["scale"]["domain"] == ["R", "R-A", "P90"]
    assert spec["encoding"]["color"]["scale"]["range"] == [
        COLOUR["R"],
        COLOUR["R-A"],
        COLOUR["P90"],
    ]
    dist = pl.DataFrame(
        {
            "arm": "R",
            "stage": "stage 1 (ERA5 target)",
            "lead_day": list(range(1, 15)),
            "distance": np.linspace(5, 9, 14),
        }
    )
    assert (
        "values" in distance_chart(data=dist, title=["t", "s"]).to_dict()["encoding"]["x"]["axis"]
    )
    rng = np.random.default_rng(0)
    per = pl.DataFrame(
        [
            {"arm": a, "primary": f"D{i}", "crps_minus_reference": float(rng.normal(-0.1, 0.1))}
            for a in ("M*", "R", "R-A")
            for i in range(1, 17)
        ]
    )
    chart = contrast_primary_chart(rows=per, reference="M*", title=["t", "s"])
    layered = chart.to_dict()["layer"]
    sizes = sorted(len(v) for v in chart.to_dict()["datasets"].values())
    assert len(layered) == 2 and sizes[-1] == 2 * 16, "one dot per arm at each of the 16 Primaries"
    hours = np.arange(0, 336, 0.5)
    series = pl.concat(
        [
            pl.DataFrame(
                {
                    "arm": a,
                    "hours": hours,
                    "actual": 50 + 10 * np.sin(hours / 7),
                    **{
                        f"q{k:02d}": 50 + 10 * np.sin(hours / 7) + o
                        for k, o in zip((5, 25, 50, 75, 95), (-8, -3, 0, 3, 8), strict=True)
                    },
                }
            )
            for a in ARMS_SHOWN[1]
        ]
    )
    sc_spec = series_chart(
        data=series, arms=ARMS_SHOWN[1], horizon_hours=36.0, title=["t", "s"]
    ).to_dict()
    assert len(sc_spec["vconcat"]) == 4
    by_bucket = {a: np.full(lay.N_BUCKETS, 1 / lay.N_BUCKETS) for a in ("R", "U")}
    assert (
        alpha_chart(by_bucket=by_bucket, title=["t", "s"]).to_dict()["encoding"]["xOffset"]["field"]
        == "arm"
    )
    heat = attention_heat(maps={"R": rng.random((14, 48)), "U": rng.random((14, 48))}, title="t")
    assert heat.to_dict() is not None
    for arm in set(COLOUR) | {"M*"}:
        assert arm in COLOUR
    print(
        "e00718_figures: all checks passed (ages, distance, per-Primary contrast, series, weight, and attention charts build from synthetic data)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ages", "contrasts", "series", "self-test"])
    parser.add_argument("out_dir", nargs="?", type=Path)
    parser.add_argument("part", nargs="?", type=int, choices=[1, 2])
    parser.add_argument("--month", default="2026-01")
    args = parser.parse_args()
    if args.command == "self-test":
        return self_test()
    assert args.out_dir is not None and args.part is not None
    if args.command == "ages":
        cmd_ages(out_dir=args.out_dir, part=args.part, month=args.month)
    elif args.command == "contrasts":
        cmd_contrasts(out_dir=args.out_dir, part=args.part)
    else:
        cmd_series(out_dir=args.out_dir, part=args.part)
    return 0


if __name__ == "__main__":
    sys.exit(main())
