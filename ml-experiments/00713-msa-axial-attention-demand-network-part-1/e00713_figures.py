# ruff: noqa: E501, D103
"""Experiment 713: forecast, injected-world, and contrast figures from the saved predictions.

Usage:
    e00713_figures.py series OUT_DIR           # day-ahead and 14-day forecasts against actuals
    e00713_figures.py injected OUT_DIR         # permanent change and temporary move, onset marked
    e00713_figures.py contrasts TAG OUT_DIR    # per-Primary and per-fold CRPS contrasts with intervals

These scripts read the stored forecasts of the fits (`work/fits/<arm>/`), the bank forecasts
(`bank713/`), and the protocol rows (`score/protocol_<TAG>/rows.parquet`). They run after the fits and
fit nothing. `M*` is the arm in `work/selection.json`.

**Anonymisation.** The forecasts are for Primary substations, which the rule does not cover, but every
axis counts hours or days from the origin and carries no calendar date, so no chart could be matched
to public generation data.

Every chart is written as SVG and optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.
Colour is by arm: Candidate `C` Data Blue, `M*` Brand Orange, `E` Data Green, `P` Data Purple,
`K` Data Sky.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import altair as alt
import e00713_arms as arms
import e00713_score as score
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WIDTH: Final = 640
COLOUR: Final = {
    "C": theme.DATA_BLUE, "M*": theme.BRAND_ORANGE, "E": theme.DATA_GREEN, "P": theme.DATA_PURPLE,
    "K": theme.DATA_SKY,
}  # fmt: skip
LIGHT: Final = {
    "C": theme.DATA_BLUE_LIGHT, "M*": theme.BRAND_ORANGE_LIGHT, "E": theme.DATA_GREEN_LIGHT,
    "P": theme.DATA_PURPLE_LIGHT, "K": theme.DATA_SKY_LIGHT,
}  # fmt: skip
CASES: Final = (
    # (season, fold month, origin day within the month, Primary label)
    ("winter", "2026-01", 13, "D5"), ("winter", "2026-01", 13, "D12"), ("winter", "2026-01", 13, "D8"),
    ("summer", "2026-06", 16, "D5"), ("summer", "2026-06", 16, "D12"), ("summer", "2026-06", 16, "D8"),
)  # fmt: skip
ARMS_SHOWN: Final = ("C", "M*", "E")


def save(chart: alt.TopLevelMixin, *, path: Path) -> None:
    """Write `chart` as SVG and optimise it with svgo, as `CLAUDE.md` requires."""
    path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(path)
    if shutil.which("npx") is None:
        print("npx is missing: run svgo by hand on", path)
        return
    subprocess.run(
        [
            "npx",
            "svgo@4",
            "--multipass",
            "--precision=1",
            "--final-newline",
            str(path),
            "-o",
            str(path),
        ],
        check=True,
    )
    print("wrote", path, f"({path.stat().st_size // 1024} KB)")


def arm_name(*, shown: str, star: str) -> str:
    """Return the stored arm name behind a shown label."""
    return star if shown == "M*" else shown


def seed_mean_frame(*, arm: str, month: str, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return an arm's seed-mean forecast of one fold in capacity shares, at the contract origin."""
    return score.seed_average(frame=score.load_arm(arm=arm, seeds=seeds, months=(month,)))


def case_frame(
    *,
    arm: str,
    shown: str,
    month: str,
    day: int,
    label: str,
    seeds: tuple[int, ...],
    truth: np.ndarray,
) -> pl.DataFrame:
    """Return the quantiles (points of capacity) and the actual of one Primary and one origin."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]

    sites = k578.sites()
    row = sites.filter(pl.col("label") == label)
    tid, primary = int(row["time_series_id"][0]), int(row["primary"][0])
    frame = seed_mean_frame(arm=arm, month=month, seeds=seeds).filter(
        pl.col("time_series_id") == tid
    )
    origin = frame["power_fcst_init_time"].unique().sort()
    chosen = origin.filter(origin.dt.day() == day)
    assert len(chosen) == 1, (month, day, len(chosen))
    out = frame.filter(pl.col("power_fcst_init_time") == chosen[0]).sort("valid_time")
    grid = (
        (out["valid_time"].dt.epoch("us").to_numpy() - prep422.GRID_START_US) // prep422.STEP_US
    ).astype(np.int64)
    hours = (out["valid_time"] - chosen[0]).dt.total_minutes().to_numpy() / 60.0
    q = {c: out[c].to_numpy() * 100.0 for c in score.QCOLS}
    return pl.DataFrame(
        {"arm": shown, "label": label, "hours": hours, "actual": truth[primary, grid] * 100.0, **q}
    )


def forecast_chart(*, data: pl.DataFrame, horizon_hours: float, title: list[str]) -> alt.Chart:
    """Return a column of panels (one per arm) with the 90% and 50% bands, the median, and the actual."""
    d = data.filter(pl.col("hours") <= horizon_hours).with_columns(
        x=pl.col("hours") if horizon_hours <= 36 else pl.col("hours") / 24.0
    )
    xaxis = (
        alt.Axis(values=[0, 6, 12, 18, 24, 30, 36])
        if horizon_hours <= 36
        else alt.Axis(values=list(range(0, 15, 2)))
    )
    base = alt.Chart(d, width=WIDTH, height=110)
    x = alt.X(
        "x:Q",
        title="Hours after the 12:00 UTC origin"
        if horizon_hours <= 36
        else "Days after the 12:00 UTC origin",
        axis=xaxis,
    )
    layers = []
    for arm in ARMS_SHOWN:
        sub = base.transform_filter(alt.datum.arm == arm)
        layers.append(
            alt.layer(
                sub.mark_area(color=LIGHT[arm], opacity=0.45).encode(  # ty: ignore[unresolved-attribute]
                    x=x, y=alt.Y("q05:Q", title="Power (% of capacity)"), y2="q95:Q"
                ),
                sub.mark_area(color=LIGHT[arm], opacity=0.9).encode(x=x, y="q25:Q", y2="q75:Q"),  # ty: ignore[unresolved-attribute]
                sub.mark_line(color=COLOUR[arm], strokeWidth=1.6, clip=True).encode(x=x, y="q50:Q"),  # ty: ignore[unresolved-attribute]
                sub.mark_line(color="#222222", strokeWidth=1.0, clip=True).encode(  # ty: ignore[unresolved-attribute]
                    x=x, y="actual:Q"
                ),
            ).properties(
                title=alt.TitleParams(
                    arm if arm != "M*" else "M* (selected MSA arm)", anchor="start", fontSize=11
                )
            )
        )
    return alt.vconcat(*layers).properties(  # ty: ignore[invalid-return-type]
        title=alt.TitleParams(title, anchor="start", fontSize=12, subtitleFontSize=11)
    )


def cmd_series(*, out_dir: Path) -> None:
    star = arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    truth = score.truth_stack()
    for season, month, day, label in CASES:
        data = pl.concat(
            [
                case_frame(
                    arm=arm_name(shown=s, star=star),
                    shown=s,
                    month=month,
                    day=day,
                    label=label,
                    seeds=(0, 1),
                    truth=truth,
                )
                for s in ARMS_SHOWN
            ]
        )
        for kind, hours, name in (("day-ahead", 36.0, "day"), ("14-day", 336.0, "fourteen")):
            title = [
                f"{kind.capitalize()} forecast of Primary {label}, {season}, against what happened",
                "Black line: actual. Coloured line: median. Dark band: 50% interval. Light band: 90% interval. Seed mean of seeds 0 and 1.",
            ]
            save(
                forecast_chart(data=data, horizon_hours=hours, title=title),
                path=out_dir / f"series_{name}_{season}_{label}.svg",
            )


def event_daily(*, folder: Path, truth_column: str, event: dict, ids: np.ndarray) -> pl.DataFrame:
    """Return, per origin day, the mean median forecast of lead day 1 at the event's host, and the truth."""
    f = pl.read_parquet(folder)
    host = ids[event["host"]]
    d = f.filter(pl.col("time_series_id") == host)
    lead_h = (d["valid_time"] - d["power_fcst_init_time"]).dt.total_minutes() / 60.0
    d = d.filter((lead_h > 0) & (lead_h <= 24))
    day = d["power_fcst_init_time"].dt.epoch("s") // 86400
    return (
        d.with_columns(day=day)
        .group_by("day")
        .agg(median=pl.col("q50").mean() * 100.0, truth=pl.col(truth_column).mean() * 100.0)
        .sort("day")
    )


def cmd_injected(*, out_dir: Path) -> None:
    """Draw one permanent change and one temporary move: forecasts of C, M*, E, P with the onset marked."""
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00709_features as features  # ty: ignore[unresolved-import]
    import e00713_era as era

    star = arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    ids = k578.sites().sort("primary")["time_series_id"].to_numpy()
    month, seed = arms.BANK_FOLDS[0], 0
    grid0 = int(features.GRID_START.timestamp() // 86400)
    for truth_kind, column, name in (
        ("era", "truth_share", "permanent change"),
        ("switch", "clean_share", "temporary move"),
    ):
        picked = None
        for path in sorted(
            (era.BANK713 / f"{star}_{month}_s{seed}").glob("pass[0-9][0-9].parquet")
        ):
            events = pl.read_parquet(path.with_name(path.stem + "_truth.parquet"))
            hit = events.filter(pl.col("truth") == truth_kind)
            if not hit.is_empty():
                picked = (path, hit.row(0, named=True))
                break
        if picked is None:
            print("no", name, "in", month)
            continue
        path, event = picked
        frames = []
        for shown in ("C", "M*", "E", "P"):
            folder = (
                era.D709 / "bank" / f"none_{month}_s{seed}" / path.name
                if shown == "C"
                else era.BANK713 / f"{arm_name(shown=shown, star=star)}_{month}_s{seed}" / path.name
            )
            daily = event_daily(folder=folder, truth_column=column, event=event, ids=ids)
            frames.append(
                daily.with_columns(
                    arm=pl.lit(shown), days=pl.col("day") - grid0 - event["onset_day"]
                )
            )
        data = pl.concat(frames)
        truth_line = frames[0].select("days", pl.col("truth"))
        end = None if event["end_day"] < 0 else event["end_day"] - event["onset_day"]
        x = alt.X("days:Q", title="Days since the injected onset (origin day)")
        lines = (
            alt.Chart(data, width=WIDTH, height=220)  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1.6, clip=True)
            .encode(
                x=x,
                y=alt.Y(
                    "median:Q",
                    title="Mean median forecast, lead day 1 (% of capacity)",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(
                        domain=list(COLOUR)[:4], range=[COLOUR[a] for a in list(COLOUR)[:4]]
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
        )
        truth = (
            alt.Chart(truth_line, width=WIDTH, height=220)  # ty: ignore[unresolved-attribute]
            .mark_line(color="#222222", strokeDash=[4, 3], clip=True)
            .encode(x=x, y="truth:Q")
        )
        rules = [alt.Chart(pl.DataFrame({"d": [0]})).mark_rule(color=theme.GREY_1).encode(x="d:Q")]  # ty: ignore[unresolved-attribute]
        if end is not None:
            rules.append(
                alt.Chart(pl.DataFrame({"d": [end]}))  # ty: ignore[unresolved-attribute]
                .mark_rule(color=theme.GREY_1, strokeDash=[2, 2])
                .encode(x="d:Q")
            )
        chart = alt.layer(lines, truth, *rules).properties(
            title=alt.TitleParams(
                [
                    f"A {name} injected into the history power of one Primary, {event['kind']}",
                    "Dashed black: the power the forecast should follow (the truth). Solid rule: onset. Dotted rule: end of the move.",
                ],
                anchor="start",
                fontSize=12,
                subtitleFontSize=11,
            )
        )
        save(chart, path=out_dir / f"injected_{truth_kind}.svg")


def month_interval(
    *, month: np.ndarray, values: np.ndarray, seed: int = 713
) -> tuple[float, float, float]:
    labels, inv = np.unique(month, return_inverse=True)
    sums, counts = np.bincount(inv, weights=values), np.bincount(inv).astype(np.float64)
    pick = np.random.default_rng(seed).integers(0, len(labels), size=(10_000, len(labels)))
    draws = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def contrast_table(*, rows: pl.DataFrame, arm: str, by: str) -> pl.DataFrame:
    """Return the mean CRPS difference `arm` minus `C` and its month-block interval, per `by`."""
    h = rows.filter(pl.col("all")).select(
        "arm",
        "label",
        "month",
        "time_series_id" if False else "label",
        "crps",
        "valid_time" if False else "month",
    )
    a = rows.filter((pl.col("arm") == arm) & pl.col("all")).select(
        "power_fcst_init_time", "valid_time", "label", "month", "crps"
    )
    c = rows.filter((pl.col("arm") == "C") & pl.col("all")).select(
        "power_fcst_init_time", "valid_time", "label", "crps"
    )
    del h
    j = a.join(c, on=["power_fcst_init_time", "valid_time", "label"], suffix="_c").with_columns(
        diff=pl.col("crps") - pl.col("crps_c")
    )
    out = []
    for key in sorted(j[by].unique().to_list()):
        g = j.filter(pl.col(by) == key)
        mean, lo, hi = month_interval(
            month=g["month"].to_numpy(), values=g["diff"].to_numpy().astype(np.float64)
        )
        out.append({by: key, "mean": mean, "lo": lo, "hi": hi, "arm": arm})
    return pl.DataFrame(out)


def cmd_contrasts(*, tag: str, out_dir: Path) -> None:
    rows = pl.read_parquet(score.SCORE_DIR / f"protocol_{tag}" / "rows.parquet")
    star = arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    shown = {
        a: ("M*" if a == star else a)
        for a in rows["arm"].unique().to_list()
        if a not in {"C", "Climatology", "Manual heuristic"}
    }
    for by, title, xname in (
        ("label", "Primary", "Primary"),
        ("month", "fold (month of the forecast origin)", "Fold"),
    ):
        table = pl.concat(
            [
                contrast_table(rows=rows, arm=a, by=by).with_columns(arm=pl.lit(shown[a]))
                for a in shown
            ]
        )
        order = sorted(table[by].unique().to_list())
        dom = list(shown.values())
        colour = alt.Color(
            "arm:N",
            scale=alt.Scale(domain=dom, range=[COLOUR.get(a, theme.GREY_2) for a in dom]),
            legend=alt.Legend(title=None, orient="top"),
        )
        y = alt.Y(f"{by}:N", sort=order, title=xname)
        base = alt.Chart(table, width=WIDTH, height=max(160, 22 * len(order)))
        chart = alt.layer(
            base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
                y=y,
                x=alt.X(
                    "lo:Q",
                    title="CRPS minus the Candidate's (points of capacity; more negative is better)",
                ),
                x2="hi:Q",
                color=colour,
                yOffset="arm:N",
            ),
            base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
                y=y, x="mean:Q", color=colour, yOffset="arm:N"
            ),
            alt.Chart(pl.DataFrame({"z": [0.0]})).mark_rule(color=theme.GREY_1).encode(x="z:Q"),  # ty: ignore[unresolved-attribute]
        ).properties(
            title=alt.TitleParams(
                [
                    f"CRPS difference from the Candidate by {title}",
                    "Dot: mean. Line: 95% interval from resampling whole months. Zero: same as the Candidate.",
                ],
                anchor="start",
                fontSize=12,
                subtitleFontSize=11,
            )
        )
        save(chart, path=out_dir / f"contrast_by_{by}.svg")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("series", "injected"):
        sub.add_parser(name).add_argument("out_dir", type=Path)
    c = sub.add_parser("contrasts")
    c.add_argument("tag")
    c.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    alt.themes.enable("default")
    if args.command == "series":
        cmd_series(out_dir=args.out_dir)
    elif args.command == "injected":
        cmd_injected(out_dir=args.out_dir)
    else:
        cmd_contrasts(tag=args.tag, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
