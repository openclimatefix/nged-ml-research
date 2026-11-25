# ruff: noqa: E501, D103
"""Experiment 713: forecast, injected-world, and contrast figures from the saved predictions.

Usage:
    e00713_figures.py series OUT_DIR           # day-ahead and 14-day forecasts against actuals
    e00713_figures.py events TAG OUT_DIR       # the Candidate and M* on real logged events, three Primaries per group
    e00713_figures.py injected OUT_DIR         # permanent change and temporary move, onset marked
    e00713_figures.py contrasts OUT_DIR        # per-Primary and per-fold CRPS and CRPS minus C, six and eight apart, 24-240 h

These scripts read the stored forecasts of the fits (`work/fits/<arm>/`), the bank forecasts
(`bank713/`), and the protocol rows (`score/protocol_<TAG>/rows.parquet`). They run after the fits and
fit nothing. `M*` is the arm in `work/selection.json`.

**Anonymisation.** The forecasts are for Primary substations, which the rule does not cover. The series,
injected, and contrast charts count hours or days from the origin; the event charts carry calendar dates
because the events are logged at substations, not at metered generators.

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

from studies import charts

WIDTH: Final = charts.PLOT_WIDTH_PX
COLOUR: Final = {
    "C": theme.DATA_BLUE, "M*": theme.BRAND_ORANGE, "E": theme.DATA_GREEN, "P": theme.DATA_PURPLE,
    "K": theme.DATA_SKY, "A": theme.DATA_AMBER, "N": theme.DATA_DEEP_TEAL,
}  # fmt: skip
LIGHT: Final = {
    "C": theme.DATA_BLUE_LIGHT, "M*": theme.BRAND_ORANGE_LIGHT, "E": theme.DATA_GREEN_LIGHT,
    "P": theme.DATA_PURPLE_LIGHT, "K": theme.DATA_SKY_LIGHT,
}  # fmt: skip
CASES: Final = (
    # (season, fold month, origin day within the month, Primary label)
    ("winter", "2026-01", 13, "D5"),
    ("summer", "2026-06", 16, "D19"),
)
# Hosts allowed for a series chart. No table records which Primary a metered generator connects to,
# so the evidence is circumstantial (experiment 722 found no metered generator inside D11, D19, or
# D21 and NGED's monitor-direction file marks no reverse power flow there; D5 is a signed-MW Primary
# already on this page). D8, D12, and D22 are refused.
ALLOWED_SERIES_HOSTS: Final = ("D5", "D11", "D19", "D21")
REFUSED_SERIES_HOSTS: Final = ("D8", "D12", "D22")
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
    for _, _, _, label in CASES:
        assert label not in REFUSED_SERIES_HOSTS, f"{label} is refused as a series host"
        assert label in ALLOWED_SERIES_HOSTS, f"{label} is not in {ALLOWED_SERIES_HOSTS}"
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
        rules = [alt.Chart(pl.DataFrame({"d": [0]})).mark_rule(color=theme.TEXT).encode(x="d:Q")]  # ty: ignore[unresolved-attribute]
        if end is not None:
            rules.append(
                alt.Chart(pl.DataFrame({"d": [end]}))  # ty: ignore[unresolved-attribute]
                .mark_rule(color=theme.ENSEMBLE_LINE, strokeDash=[2, 2])
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


def event_runs(*, rows: pl.DataFrame, groups: dict[str, np.ndarray], per_group: int) -> list[dict]:
    """Return the longest runs of consecutive origin days with `event_rows` for each group, one run per Primary.

    `rows` are the protocol rows of the Candidate. A run is a stretch of origin days on which the Primary
    has at least one half-hour inside a logged event.
    """
    import e00578_common as k578  # ty: ignore[unresolved-import]

    sites = k578.sites()
    label_of = dict(zip(sites["time_series_id"].to_list(), sites["label"].to_list(), strict=True))
    primary_of = dict(
        zip(sites["time_series_id"].to_list(), sites["primary"].to_list(), strict=True)
    )
    days = (
        rows.filter(pl.col("event_rows"))
        .with_columns(day=pl.col("power_fcst_init_time").dt.date())
        .select("time_series_id", "day")
        .unique()
        .sort("time_series_id", "day")
    )
    runs: list[dict] = []
    for tid, frame in days.group_by("time_series_id", maintain_order=True):
        t = int(tid[0])
        ordinal = frame["day"].dt.epoch("d").to_numpy()
        start = 0
        for i in range(1, len(ordinal) + 1):
            if i == len(ordinal) or ordinal[i] != ordinal[i - 1] + 1:
                runs.append(
                    {
                        "time_series_id": t, "label": label_of[t], "primary": int(primary_of[t]),
                        "first": frame["day"][start], "last": frame["day"][i - 1], "length": i - start,
                    }
                )  # fmt: skip
                start = i
    chosen = []
    for name in ("six", "eight"):
        ids = set(groups[name].tolist())
        mine = sorted((r for r in runs if r["time_series_id"] in ids), key=lambda r: -r["length"])
        seen: set[int] = set()
        picked = []
        for r in mine:
            if r["time_series_id"] in seen:
                continue
            seen.add(r["time_series_id"])
            picked.append({**r, "group": name})
            if len(picked) == per_group:
                break
        chosen.extend(picked)
    return chosen


def cmd_events(*, tag: str, out_dir: Path, per_group: int = 3) -> None:
    """Draw the Candidate and `M*` on real logged events: lead-day-1 median by origin day against the actual."""
    from datetime import timedelta

    star = arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    truth = score.truth_stack()
    rows = pl.read_parquet(score.SCORE_DIR / f"protocol_{tag}" / "rows.parquet").filter(
        pl.col("arm") == "C"
    )
    picked = event_runs(rows=rows, groups=score.group_ids(), per_group=per_group)
    assert picked, "no real events found in the protocol rows"
    for k, run in enumerate(picked):
        pad = timedelta(days=7)
        lo, hi = run["first"] - pad, run["last"] + pad
        months = [m for m in arms.ALL_FOLDS if f"{lo:%Y-%m}" <= m <= f"{hi:%Y-%m}"]
        frames = []
        for shown in ("C", "M*"):
            f = pl.concat(
                [
                    seed_mean_frame(arm=arm_name(shown=shown, star=star), month=m, seeds=(0, 1))
                    for m in months
                ]
            )
            lead = (f["valid_time"] - f["power_fcst_init_time"]).dt.total_minutes() / 60.0
            f = f.with_columns(lead_h=lead).filter(
                (pl.col("time_series_id") == run["time_series_id"])
                & (pl.col("lead_h") > 0)
                & (pl.col("lead_h") <= 24)
            )
            grid = (
                (f["valid_time"].dt.epoch("us").to_numpy() - 1_567_296_000_000_000) // 1_800_000_000
            ).astype(np.int64)
            f = f.with_columns(
                actual=pl.Series(truth[run["primary"], grid] * 100.0),
                day=pl.col("power_fcst_init_time").dt.date(),
            )
            daily = (
                f.group_by("day")
                .agg(
                    median=pl.col("q50").mean() * 100.0,
                    actual=pl.col("actual").fill_nan(None).mean(),
                )
                .sort("day")
            )
            frames.append(daily.with_columns(arm=pl.lit(shown)))
        data = (
            pl.concat(frames)
            .filter((pl.col("day") >= lo) & (pl.col("day") <= hi))
            .with_columns(day=pl.col("day").cast(pl.Datetime("us")))
        )
        actual = data.filter(pl.col("arm") == "C").select("day", "actual")
        band = pl.DataFrame(
            {"a": [run["first"]], "b": [run["last"] + timedelta(days=1)]}
        ).with_columns(pl.all().cast(pl.Datetime("us")))
        x = alt.X("day:T", title="Origin day (12:00 UTC)")
        lines = (
            alt.Chart(data, width=WIDTH, height=200)  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1.6, clip=True)
            .encode(
                x=x,
                y=alt.Y(
                    "median:Q",
                    title="Mean power, lead day 1 (% of capacity)",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(domain=["C", "M*"], range=[COLOUR["C"], COLOUR["M*"]]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
        )
        truth_line = (
            alt.Chart(actual, width=WIDTH, height=200)  # ty: ignore[unresolved-attribute]
            .mark_line(color="#222222", strokeDash=[4, 3], clip=True)
            .encode(x=x, y="actual:Q")
        )  # ty: ignore[unresolved-attribute]
        shade = (
            alt.Chart(band)
            .mark_rect(color=theme.ENSEMBLE_LINE, opacity=0.2)
            .encode(x="a:T", x2="b:T")  # ty: ignore[unresolved-attribute]
        )
        chart = alt.layer(shade, lines, truth_line).properties(
            title=alt.TitleParams(
                [
                    f"Primary {run['label']}, {GROUP_PHRASE[run['group']]}: a logged event of {run['length']} origin days",
                    "Dashed black: actual. Coloured: forecast of the next 24 hours from each origin day. Shaded: days with half-hours inside the event.",
                ],
                anchor="start",
                fontSize=12,
                subtitleFontSize=11,
            )
        )
        save(chart, path=out_dir / f"event_{FILE_GROUP[run['group']]}_{run['label']}_{k}.svg")


def month_interval(
    *, month: np.ndarray, values: np.ndarray, seed: int = 713
) -> tuple[float, float, float]:
    labels, inv = np.unique(month, return_inverse=True)
    sums, counts = np.bincount(inv, weights=values), np.bincount(inv).astype(np.float64)
    pick = np.random.default_rng(seed).integers(0, len(labels), size=(10_000, len(labels)))
    draws = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


LABEL_ORDER_KEY = lambda label: int(label[1:])  # noqa: E731
GROUP_TITLE: Final = {"six": "The six signed-MW Primaries", "eight": "The eight MVA-only Primaries"}
FILE_GROUP: Final = {"six": "signed_mw", "eight": "mva_only"}
GROUP_PHRASE: Final = {
    "six": "one of the six signed-MW Primaries",
    "eight": "one of the eight MVA-only Primaries",
}
CONTRAST_ARMS: Final = ("M*", "P", "E", "K", "N", "A")
DRAWS: Final = 10_000


def contrast_cells(*, star: str, months: tuple[str, ...], seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return per (arm, month, label, origin day) sums of CRPS and of the CRPS difference from `C`, headline band.

    One fold and one arm are in memory at a time. The seed mean of each arm's quantiles is scored, as
    in `e00713_score.load_all`, and only the 24 to 240 h rows of the six and the eight are kept.
    """
    import e00578_common as k578  # ty: ignore[unresolved-import]

    truth = score.truth_stack()
    sites = k578.sites()
    keep = {"six": score.SIX, "eight": score.EIGHT}
    label_of = dict(zip(sites["time_series_id"].to_list(), sites["label"].to_list(), strict=True))
    group_of = {label: g for g, labels in keep.items() for label in labels}
    out = []
    for month in months:
        frames = {}
        for shown in ("C", *CONTRAST_ARMS):
            f = score.seed_average(
                frame=score.load_arm(
                    arm=arm_name(shown=shown, star=star), seeds=seeds, months=(month,)
                )
            )
            f = score.add_losses(frame=f, truth=truth)
            f = f.filter(
                (pl.col("lead_hours") >= score.HEADLINE_BAND_HOURS[0])
                & (pl.col("lead_hours") < score.HEADLINE_BAND_HOURS[1])
            ).select(*score.KEYS, "crps")
            frames[shown] = f.with_columns(
                label=pl.col("time_series_id").replace_strict(label_of, return_dtype=pl.String),
                day=pl.col("power_fcst_init_time").dt.date(),
            ).filter(pl.col("label").is_in(list(group_of)))
        ref = frames["C"].select(*score.KEYS, pl.col("crps").alias("crps_c"))
        for shown, f in frames.items():
            j = f.join(ref, on=list(score.KEYS), how="inner")
            assert j.height == f.height == ref.height, (
                f"arm {shown!r} and C hold different keys in {month}"
            )
            cell = (
                j.group_by("label", "day")
                .agg(
                    crps=pl.col("crps").sum(),
                    diff=(pl.col("crps") - pl.col("crps_c")).sum(),
                    n=pl.len(),
                )
                .with_columns(arm=pl.lit(shown), month=pl.lit(month))
            )
            out.append(cell)
    return pl.concat(out).with_columns(
        group=pl.col("label").replace_strict(group_of, return_dtype=pl.String)
    )


def block_interval(
    *, blocks: np.ndarray, num: np.ndarray, den: np.ndarray
) -> tuple[float, float, float]:
    """Return the ratio of sums and a 95% interval from resampling whole blocks (months or origin days)."""
    labels, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=num)
    counts = np.bincount(inv, weights=den)
    pick = np.random.default_rng(713).integers(0, len(labels), size=(DRAWS, len(labels)))
    draws = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return (
        float(sums.sum() / counts.sum()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def contrast_summary(*, cells: pl.DataFrame, by: str) -> pl.DataFrame:
    """Return, per arm, group and `by`, the mean CRPS and the mean difference from `C` with block intervals.

    Per Primary the blocks are the 8 fold months; per fold the blocks are origin days of that fold.
    """
    block = "month" if by == "label" else "day"
    rows = []
    for (arm, group, key), g in cells.group_by("arm", "group", by):
        mean_crps, c_lo, c_hi = block_interval(
            blocks=g[block].cast(pl.String).to_numpy(),
            num=g["crps"].to_numpy(),
            den=g["n"].to_numpy().astype(float),
        )
        mean_diff, d_lo, d_hi = block_interval(
            blocks=g[block].cast(pl.String).to_numpy(),
            num=g["diff"].to_numpy(),
            den=g["n"].to_numpy().astype(float),
        )
        rows.append(
            {
                "arm": arm,
                "group": group,
                by: key,
                "crps": mean_crps,
                "crps_lo": c_lo,
                "crps_hi": c_hi,
                "diff": mean_diff,
                "diff_lo": d_lo,
                "diff_hi": d_hi,
            }
        )
    return pl.DataFrame(rows)


def contrast_chart(
    *,
    table: pl.DataFrame,
    by: str,
    value: str,
    arms_shown: tuple[str, ...],
    title: list[str],
    x_title: str,
) -> alt.TopLevelMixin:
    """Draw dots and intervals of `value` (`crps` or `diff`), one panel per Primary group, rows of `by`."""
    table = table.filter(pl.col("arm").is_in(list(arms_shown)))
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=list(arms_shown), range=[COLOUR[a] for a in arms_shown]),
        legend=alt.Legend(title=None, orient="top"),
    )
    panels = []
    for group in ("six", "eight"):
        t = table.filter(pl.col("group") == group)
        order = sorted(t[by].unique().to_list(), key=LABEL_ORDER_KEY if by == "label" else str)
        y = alt.Y(f"{by}:N", sort=order, title=None)
        base = alt.Chart(t, width=WIDTH, height=40 * len(order), title=GROUP_TITLE[group])  # ty: ignore[unresolved-attribute]
        layers = [
            base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
                y=y,
                x=alt.X(f"{value}_lo:Q", title=x_title),
                x2=f"{value}_hi:Q",
                color=colour,
                yOffset="arm:N",
            ),
            base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
                y=y, x=f"{value}:Q", color=colour, yOffset="arm:N"
            ),
        ]
        if value == "diff":
            layers.append(
                alt.Chart(pl.DataFrame({"z": [0.0]})).mark_rule(color=theme.TEXT).encode(x="z:Q")  # ty: ignore[unresolved-attribute]
            )  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(*layers))
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(title, anchor="start", fontSize=12, subtitleFontSize=11)
    )


def cmd_contrasts(*, out_dir: Path) -> None:
    """Draw per-Primary and per-fold CRPS (absolute and as a difference from `C`) in the 24 to 240 h band, six and eight apart."""
    star = arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    cells = contrast_cells(star=star, months=arms.ALL_FOLDS, seeds=arms.SCREEN_SEEDS)
    # The check against the stored headline: M* minus C pooled over each group's cells.
    for g in ("six", "eight"):
        c = cells.filter((pl.col("arm") == "M*") & (pl.col("group") == g))
        pooled = c["diff"].sum() / c["n"].sum()  # ty: ignore[unsupported-operator]
        print(
            f"{star} minus C, {g}, seeds 0 and 1, pooled: {pooled:.4f} (report_only.json holds the 95% figure)"
        )
    for by, what in (("label", "Primary"), ("month", "fold")):
        table = contrast_summary(cells=cells, by=by)
        interval = "months" if by == "label" else "origin days"
        sub = f"Dot: mean CRPS of the seed-mean forecast (seeds 0 and 1). Line: 95% interval from resampling whole {interval}.{' Origin days overlap, so this interval is too narrow.' if by == 'month' else ''}"
        save(
            contrast_chart(
                table=table,
                by=by,
                value="diff",
                arms_shown=CONTRAST_ARMS,
                title=[f"CRPS minus the Candidate's, by {what}, 24 to 240 h ahead", sub],
                x_title="Points of capacity; more negative is better than the Candidate",
            ),
            path=out_dir / f"contrast_by_{by}.svg",
        )
        save(
            contrast_chart(
                table=table,
                by=by,
                value="crps",
                arms_shown=("C", "M*", "P"),
                title=[f"CRPS by {what}, 24 to 240 h ahead", sub],
                x_title="CRPS, points of capacity (lower is better)",
            ),
            path=out_dir / f"crps_by_{by}.svg",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("series", "injected"):
        sub.add_parser(name).add_argument("out_dir", type=Path)
    e = sub.add_parser("events")
    e.add_argument("tag")
    e.add_argument("out_dir", type=Path)
    c = sub.add_parser("contrasts")
    c.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    if args.command == "series":
        cmd_series(out_dir=args.out_dir)
    elif args.command == "injected":
        cmd_injected(out_dir=args.out_dir)
    elif args.command == "events":
        cmd_events(tag=args.tag, out_dir=args.out_dir)
    else:
        cmd_contrasts(out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
