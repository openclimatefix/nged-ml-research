# ruff: noqa: E501, PERF401
"""Experiment 716: the validation-loss, attention-map, and forecast-example figures of the result page.

Usage: run.sh e00716_writeup_figures.py contrasts OUT_DIR   # A1 and A2: intervals, by fold, by Primary
       run.sh e00716_writeup_figures.py tables               # stopping-pass table of the fit logs
       run.sh e00716_writeup_figures.py losses OUT_DIR
       run.sh e00716_writeup_figures.py attention OUT_DIR
       run.sh e00716_writeup_figures.py forecasts OUT_DIR

Nothing is fitted. `losses` reads the validation losses per pass that the fit logs hold under
`work/logs/<arm>/<month>_s<seed>_fit.json` (arms `H` and `H-S` here, and experiment 713's arm
`M1`, the stored `M*`). The fit logs hold no training loss, so no training-loss curve is drawn. `attention`
reads `work/attention/Mstar_2026-05_s0.npz` and `H_2026-05_s0.npz`, which
`e00716_attention_extract.py` wrote. `forecasts` reads the same two files for the day-ahead cases and
the stored forecast files of `M*` and `H` for the 14-day forecasts.

**Anonymisation.** Every chart is of a Primary substation. The forecast examples use hosts D11,
D19, and D21 only. The attention maps use D11, D19, and D21. D19 and D21 read a capacity factor
that looks ahead of fold 2026-05, and the page says so. No chart names a metered generator.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import altair as alt
import e00716_arms as arms
import e00716_figures as figs
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WIDTH: Final = 640
MONTH: Final = "2026-05"
SEED: Final = 0
HOSTS_ATTENTION: Final = ("D11", "D19", "D21")
HOSTS_FORECAST: Final = ("D11", "D19", "D21")
CASES: Final = {
    "clear_day": "clear target day",
    "dull_after_sunny": "dull target day after a sunny window",
}
ARM_COLOUR: Final = {"M*": theme.BRAND_ORANGE, "H": theme.DATA_BLUE, "H-S": theme.DATA_PURPLE}
QUANTILES: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
ATTENTION_DIR: Final = arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR, arm="Z") / "work" / "attention"
M713_FITS: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00713-msa-axial-attention-demand-network-part-1" / "work"
)


def lines_of(text: str, width: int = 100) -> list[str]:
    """Wrap a title or subtitle to the plot width."""
    import textwrap

    return textwrap.wrap(text, width=width)


def title_of(*, title: str, subtitle: str) -> alt.TitleParams:
    """Return a wrapped title with a subtitle."""
    return alt.TitleParams(
        text=lines_of(title, 80),
        subtitle=lines_of(subtitle, 110),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def loss_rows() -> pl.DataFrame:
    """Return one row per arm, fold, seed, stage, and pass with the validation loss of that pass."""
    rows = []
    sources = {
        "H": arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR, arm="Z") / "work" / "logs" / "H",
        "H-S": arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR, arm="Z") / "work" / "logs" / "H-S",
        "M*": M713_FITS / "logs" / "M1",
    }
    for arm, folder in sources.items():
        for month in arms.ALL_FOLDS:
            for seed in range(5):
                path = folder / f"{month}_s{seed}_fit.json"
                assert path.exists(), f"missing {path}"
                log = json.loads(path.read_text())
                for entry in log["stages"]:
                    first = entry["first"]
                    hist = first["val_history"]
                    if entry.get("rerun"):
                        hist = entry["rerun"]["val_history"]
                    rows.extend(
                        {
                            "arm": arm,
                            "fold": month,
                            "seed": seed,
                            "stage": int(entry["stage"]),
                            "pass": i + 1,
                            "loss": float(v),
                            "run": f"{arm}-{seed}",
                        }
                        for i, v in enumerate(hist)
                    )
    return pl.DataFrame(rows)


def cmd_losses(*, out_dir: Path) -> None:
    """Draw the validation loss per pass: one panel per fold and stage, one line per arm and seed."""
    table = loss_rows()
    fold_number = {m: i + 1 for i, m in enumerate(arms.ALL_FOLDS)}
    table = table.with_columns(
        pl.col("fold").replace_strict(fold_number).alias("fold_n"),
        pl.format("Stage {}", pl.col("stage")).alias("stage_label"),
    )
    summary = (
        table.group_by("arm", "stage")
        .agg(pl.col("pass").max().alias("max_pass"), pl.len().alias("n"))
        .sort("arm", "stage")
    )
    print(summary)
    for stage in (2, 3):
        sub = table.filter(pl.col("stage") == stage)
        chart = (
            alt.Chart(sub)
            .mark_line(opacity=0.8, strokeWidth=1)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("pass:Q", title="Training pass over the stage's training pairs"),
                y=alt.Y("loss:Q", title="Validation loss", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "arm:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["M*", "H", "H-S"],
                        range=[ARM_COLOUR["M*"], ARM_COLOUR["H"], ARM_COLOUR["H-S"]],
                    ),
                    legend=alt.Legend(orient="top"),
                ),
                detail="run:N",
            )
            .properties(width=WIDTH - 80, height=70)
            .facet(row=alt.Row("fold_n:O", title="Fold (1 = 2025-11 to 8 = 2026-06)"))
            .resolve_scale(y="independent")
            .properties(
                title=title_of(
                    title=f"Validation loss per pass, stage {stage}, every seed of M*, H, and H-S",
                    subtitle="One line per seed (0 to 4). Smaller is better. The loss is the validation loss that the training loop of experiment 713 reports for each pass. The fit logs hold no training loss, so none is drawn.",
                )
            )
        )
        figs.save(chart, path=out_dir / f"validation_loss_stage{stage}.svg")


def tod(idx: np.ndarray) -> np.ndarray:
    """Return the UTC hour of day of each half-hour grid index."""
    return (idx % 48) / 2.0


def load_arm(*, arm: str) -> dict[str, np.ndarray]:
    """Return the arrays of an arm's attention file."""
    name = "Mstar" if arm == "M*" else arm
    z = np.load(ATTENTION_DIR / f"{name}_{MONTH}_s{SEED}.npz")
    return {k: z[k] for k in z.files}


def across_frame(*, files: dict[str, dict[str, np.ndarray]], host: str) -> pl.DataFrame:
    """Return the head-mean across-day weights of every target token to the target row and the 14 history rows."""
    rows = []
    for arm, arrays in files.items():
        for case in CASES:
            w = arrays[f"{host}/{case}/across"].mean(axis=1)  # (48, 15)
            tgt = arrays[f"{host}/{case}/target_idx"]
            for t in range(48):
                for slot in range(15):
                    rows.append(
                        {
                            "panel": f"{arm}, {CASES[case]}",
                            "arm": arm,
                            "case": case,
                            "hour": float((tgt[t] - tgt[0]) / 2.0),
                            "age": "target" if slot == 0 else f"{slot + 1}",
                            "slot": slot,
                            "weight": float(w[t, slot]),
                        }
                    )
    table = pl.DataFrame(rows).with_columns((pl.col("hour") // 3 * 3).alias("hour"))
    return table.group_by("panel", "arm", "case", "age", "slot", "hour").agg(
        pl.col("weight").mean()
    )


def alpha_frame(*, files: dict[str, dict[str, np.ndarray]], host: str) -> pl.DataFrame:
    """Return the read-out weights over the 14 history rows and the 48 half-hours of each row."""
    rows = []
    for arm, arrays in files.items():
        for case in CASES:
            a = arrays[f"{host}/{case}/alpha"]  # (14, 48)
            for d in range(14):
                for h in range(48):
                    rows.append(
                        {
                            "panel": f"{arm}, {CASES[case]}",
                            "arm": arm,
                            "case": case,
                            "age": d + 2,
                            "hour": float(h // 2),
                            "weight": float(a[d, h]),
                        }
                    )
    table = pl.DataFrame(rows)
    return table.group_by("panel", "arm", "case", "age", "hour").agg(pl.col("weight").mean())


def heat(
    *, table: pl.DataFrame, x: alt.X, y: alt.Y, title: alt.TitleParams, height: int
) -> alt.TopLevelMixin:
    """Return a heat map with one row of panels per `panel` value."""
    return (
        alt.Chart(table)
        .mark_rect()
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=y,
            color=alt.Color(
                "weight:Q",
                title="Weight",
                scale=alt.Scale(range=[theme.WHITE, theme.DATA_BLUE]),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=WIDTH - 110, height=height)
        .facet(
            row=alt.Row(
                "panel:N",
                title=None,
                sort=[f"{a}, {c}" for a in ("M*", "H") for c in CASES.values()],
                header=alt.Header(
                    labelAngle=0,
                    labelAlign="left",
                    labelOrient="top",
                    labelFontSize=10,
                    labelPadding=2,
                ),
            ),
            spacing=14,
        )
        .properties(title=title)
    )


def cmd_attention(*, out_dir: Path) -> None:
    """Draw the attention maps and read-out weights of M* and H at three hosts, and the mean weight by age."""
    files = {"M*": load_arm(arm="M*"), "H": load_arm(arm="H")}
    order = ["target", *(str(a) for a in range(2, 16))]
    for host in HOSTS_ATTENTION:
        cases = {
            c: (
                float(files["H"][f"{host}/{c}/history_mean"]),
                float(files["H"][f"{host}/{c}/target_mean"]),
            )
            for c in CASES
        }
        print(host, {c: (round(a, 3), round(b, 3)) for c, (a, b) in cases.items()})
        across = across_frame(files=files, host=host)
        chart = heat(
            table=across,
            x=alt.X(
                "age:O",
                title="Row attended to: the target day, then the history row of that age in days",
                sort=order,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "hour:O",
                title="Hours after start",
                axis=alt.Axis(labelExpr="datum.value % 6 == 0 ? datum.value : ''"),
            ),
            title=title_of(
                title=f"Across-day attention of the last block, {host}, fold 2026-05, seed 0",
                subtitle="Each row of a panel is a 3-hour block of target half-hours (the mean of 6 tokens). Each column is one row that the token attends to. Weights are the mean over the 4 heads and sum to 1 in each row of a panel.",
            ),
            height=110,
        )
        figs.save(chart, path=out_dir / f"attention_across_{host}.svg")
        alpha = alpha_frame(files=files, host=host)
        chart = heat(
            table=alpha,
            x=alt.X(
                "hour:O",
                title="Target half-hour: hours after the first one (2 half-hours averaged)",
                axis=alt.Axis(labelExpr="datum.value % 6 == 0 ? datum.value : ''", labelAngle=0),
            ),
            y=alt.Y(
                "age:O", title="Row age (days)", sort="ascending", axis=alt.Axis(labelFontSize=9)
            ),
            title=title_of(
                title=f"Read-out weights over the 14 history rows, {host}, fold 2026-05, seed 0",
                subtitle="The read-out builds the base of the forecast for each target half-hour as a weighted mean of the 14 history rows. The weights in one column of a panel sum to 1. Each panel is one forecast origin, as the case labels say.",
            ),
            height=170,
        )
        figs.save(chart, path=out_dir / f"attention_readout_{host}.svg")
    rows = []
    for arm, arrays in files.items():
        for lead in (1, 7):
            w = arrays[f"summary_lead{lead}/mean_weight"]
            cnt = arrays[f"summary_lead{lead}/count"]
            pooled = (w * cnt[:, None]).sum(axis=0) / cnt.sum()
            rows.extend(
                {"arm": arm, "lead_day": f"Lead day {lead}", "slot": int(s), "weight": float(v)}
                for s, v in enumerate(pooled)
            )
    table = pl.DataFrame(rows)
    print(
        table.pivot(on="arm", index=["lead_day", "slot"], values="weight").sort("lead_day", "slot")
    )
    chart = (
        alt.Chart(table)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "slot:O",
                title="Slot 0 = the target row; slot k = the history row that starts k+1 days before the origin",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("weight:Q", title="Mean across-day weight"),
            color=alt.Color(
                "arm:N",
                title=None,
                scale=alt.Scale(domain=["M*", "H"], range=[ARM_COLOUR["M*"], ARM_COLOUR["H"]]),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=WIDTH - 90, height=130)
        .facet(row=alt.Row("lead_day:N", title=None))
        .properties(
            title=title_of(
                title="Mean across-day weight by row, all 16 Primaries and every scored origin of fold 2026-05, seed 0",
                subtitle="Weights are the last block's, the mean over heads and over the origins of the fold, weighted by origin count. They sum to 1 over the 15 slots.",
            )
        )
    )
    figs.save(chart, path=out_dir / "attention_weight_by_age.svg")


def panel_chart(
    *,
    bands: pl.DataFrame,
    lines: pl.DataFrame,
    x: alt.X,
    panel: str,
    stroke: float,
    scale: alt.Scale,
) -> alt.TopLevelMixin:
    """Return the layered chart of one panel: a band for each arm, and the median and demand lines."""
    layers = [
        alt.Chart(bands.filter((pl.col("panel") == panel) & (pl.col("arm") == arm)))
        .mark_area(opacity=0.18, color=ARM_COLOUR[arm])
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("lo:Q", title="Demand (share of capacity)", scale=alt.Scale(zero=False)),
            y2="hi:Q",
        )
        for arm in ("M*", "H")
    ]
    layers.append(
        alt.Chart(lines.filter(pl.col("panel") == panel))
        .mark_line(strokeWidth=stroke)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("value:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None, scale=scale, legend=alt.Legend(orient="top")),
        )
    )
    return alt.layer(*layers).properties(
        width=WIDTH - 90, height=110, title=alt.TitleParams(text=panel, anchor="start", fontSize=11)
    )


def stacked(
    *,
    bands: pl.DataFrame,
    lines: pl.DataFrame,
    x: alt.X,
    stroke: float,
    scale: alt.Scale,
    title: alt.TitleParams,
) -> alt.TopLevelMixin:
    """Return the panels of a table, one above the other."""
    panels = sorted(bands["panel"].unique().to_list())
    charts = [
        panel_chart(bands=bands, lines=lines, x=x, panel=p, stroke=stroke, scale=scale)
        for p in panels
    ]
    return alt.vconcat(*charts)  # ty: ignore[invalid-argument-type]


def cmd_forecasts(*, out_dir: Path) -> None:
    """Draw the first-day forecasts from the stored cases and the full-horizon forecasts from stored files."""
    import e00578_common as k578  # ty: ignore[unresolved-import]

    files = {"M*": load_arm(arm="M*"), "H": load_arm(arm="H")}
    sites = k578.sites()
    id_of = dict(zip(sites["label"].to_list(), sites["time_series_id"].to_list(), strict=True))
    scale = alt.Scale(
        domain=["Cleaned demand", "M* median", "H median"],
        range=[theme.BLACK_1, theme.BRAND_ORANGE, theme.DATA_BLUE],
    )
    bands, lines = [], []
    for host in HOSTS_FORECAST:
        for case in CASES:
            panel = f"{host}, {CASES[case]}"
            for arm, arrays in files.items():
                q = arrays[f"{host}/{case}/q"]
                for i in range(48):
                    bands.append(
                        {
                            "panel": panel,
                            "arm": arm,
                            "hour": i / 2.0,
                            "lo": float(q[i, 1]),
                            "hi": float(q[i, 9]),
                        }
                    )
                    lines.append(
                        {
                            "panel": panel,
                            "series": f"{arm} median",
                            "hour": i / 2.0,
                            "value": float(q[i, 5]),
                        }
                    )
            actual = files["H"][f"{host}/{case}/actual"]
            lines.extend(
                {
                    "panel": panel,
                    "series": "Cleaned demand",
                    "hour": i / 2.0,
                    "value": float(actual[i]),
                }
                for i in range(48)
            )
    x = alt.X(
        "hour:Q",
        title="Hours after the first forecast half-hour",
        axis=alt.Axis(values=[0, 6, 12, 18, 24]),
        scale=alt.Scale(domain=[0, 24]),
    )
    chart = stacked(
        bands=pl.DataFrame(bands),
        lines=pl.DataFrame(lines),
        x=x,
        stroke=1.6,
        scale=scale,
        title=title_of(
            title="First-day forecast of M* and H, fold 2026-05, seed 0, at two Primaries",
            subtitle="Lines: the median forecast and the cleaned demand. Shaded bands: the 5% to 95% range of each arm. The 48 half-hours of the first target day.",
        ),
    )
    figs.save(chart, path=out_dir / "forecast_day_ahead.svg")
    frames = {
        "M*": pl.read_parquet(M713_FITS / "fits" / "M1" / f"{MONTH}_s{SEED}.parquet"),
        "H": pl.read_parquet(ATTENTION_DIR.parent / "fits" / "H" / f"{MONTH}_s{SEED}.parquet"),
    }
    grid_start = np.datetime64("2019-09-01T00:00", "us")
    bands, lines = [], []
    for host in HOSTS_FORECAST:
        for case in CASES:
            panel = f"{host}, {CASES[case]}"
            first_valid = (
                (
                    grid_start
                    + np.timedelta64(int(files["H"][f"{host}/{case}/target_idx"][0]) * 30, "m")
                )
                .astype("datetime64[us]")
                .astype(object)
            )
            ts = id_of[host]
            ref = frames["H"].filter(pl.col("time_series_id") == ts)
            inits = (
                ref.filter(pl.col("valid_time").dt.replace_time_zone(None) == first_valid)[
                    "power_fcst_init_time"
                ]
                .unique()
                .to_list()
            )
            origin = max(inits)
            print(
                host,
                case,
                "first valid",
                first_valid,
                "origin of the case",
                origin,
                "origins whose forecast reaches it",
                len(inits),
            )
            for arm, frame in frames.items():
                sub = frame.filter(
                    (pl.col("time_series_id") == ts) & (pl.col("power_fcst_init_time") == origin)
                ).sort("valid_time")
                t0 = sub["valid_time"].min()
                days = ((sub["valid_time"] - t0).dt.total_minutes() / 1440.0).to_list()
                bands.extend(
                    {"panel": panel, "arm": arm, "day": d, "lo": lo, "hi": hi}
                    for d, lo, hi in zip(
                        days, sub["q05"].to_list(), sub["q95"].to_list(), strict=True
                    )
                )
                lines.extend(
                    {"panel": panel, "series": f"{arm} median", "day": d, "value": v}
                    for d, v in zip(days, sub["q50"].to_list(), strict=True)
                )
                if arm == "H":
                    lines.extend(
                        {"panel": panel, "series": "Cleaned demand", "day": d, "value": v}
                        for d, v in zip(days, sub["actual"].to_list(), strict=True)
                        if v is not None
                    )
                    print(
                        host,
                        case,
                        "rows",
                        sub.height,
                        "max lead hours",
                        sub["lead_hours"].max(),
                        "first valid",
                        t0,
                    )
    x = alt.X(
        "day:Q",
        title="Days after the first forecast half-hour",
        axis=alt.Axis(values=[0, 2, 4, 6, 8, 10, 12, 14]),
    )
    chart = stacked(
        bands=pl.DataFrame(bands),
        lines=pl.DataFrame(lines),
        x=x,
        stroke=1.2,
        scale=scale,
        title=title_of(
            title="Full-horizon forecast of M* and H, fold 2026-05, seed 0, at the same two Primaries",
            subtitle="Lines: the median forecast and the cleaned demand. Shaded bands: the 5% to 95% range of each arm. The stored forecast file covers the leads it holds.",
        ),
    )
    figs.save(chart, path=out_dir / "forecast_full_horizon.svg")


PART1: Final = arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR, arm="Z") / "score" / "part1.json"
GROUP_NAME: Final = {"six": "Six signed-MW Primaries", "eight": "Eight MVA-only Primaries"}
CONTRAST_COLOUR: Final = {
    "A1: H minus M*": theme.DATA_BLUE,
    "A2: H minus H-S": theme.DATA_PURPLE,
    "Control: H-S minus M*": theme.DATA_SKY,
}


def cmd_contrasts(*, out_dir: Path) -> None:
    """Draw A1 and A2 with their 98.75% intervals, A1 and A2 by fold, and the first contrast by Primary."""
    import e00578_common as k578  # ty: ignore[unresolved-import]

    res = json.loads(PART1.read_text())
    rows = []
    for key, label in (
        ("first", "A1: H minus M*"),
        ("placebo", "A2: H minus H-S"),
        ("control", "Control: H-S minus M*"),
    ):
        for g in ("six", "eight"):
            c = res[f"counted:{key}:{g}"]
            rows.append(
                {
                    "contrast": label,
                    "group": GROUP_NAME[g],
                    "point": c["point"],
                    "lower": c["lower"],
                    "upper": c["upper"],
                    "delta": c["delta"],
                }
            )
            print(
                label,
                g,
                round(c["point"], 4),
                round(c["lower"], 4),
                round(c["upper"], 4),
                "limit",
                c["delta"],
            )
    table = pl.DataFrame(rows)
    order = list(CONTRAST_COLOUR)
    colour = alt.Color(
        "contrast:N",
        legend=None,
        scale=alt.Scale(domain=order, range=list(CONTRAST_COLOUR.values())),
    )
    base = alt.Chart(table).encode(y=alt.Y("contrast:N", title=None, sort=order))
    bars = base.mark_rule(strokeWidth=4).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="CRPS difference, points of capacity (negative: the first arm is better)",
        ),
        x2="upper:Q",
        color=colour,
    )
    dots = base.mark_point(filled=True, size=80).encode(x="point:Q", color=colour)  # ty: ignore[unresolved-attribute]
    zero = base.mark_rule(color="#292B2B").encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(bars, dots, zero, data=table)
        .properties(width=230, height=110)
        .facet(column=alt.Column("group:N", title=None, sort=list(GROUP_NAME.values())))
        .resolve_scale(x="independent")
        .properties(
            title=title_of(
                title="Counted contrasts A1 and A2 and the placebo control, 98.75% intervals",
                subtitle="Dot: mean difference in CRPS over 8 folds and 5 seeds. Line: 98.75% interval from resampling whole months. Vertical line: no difference. The no-detectable-difference limits of the decision rule are plus or minus 0.06 points (six signed-MW Primaries) and 0.16 (eight MVA-only Primaries).",
            )
        )
    )
    figs.save(chart, path=out_dir / "contrasts.svg")
    # by fold
    rows = []
    for key, label in (("first", "A1: H minus M*"), ("placebo", "A2: H minus H-S")):
        for g in ("six", "eight"):
            for i, v in enumerate(res[f"counted:{key}:{g}"]["monthly"]):
                rows.append(
                    {"contrast": label, "group": GROUP_NAME[g], "fold": i + 1, "difference": v}
                )
    fold_table = pl.DataFrame(rows)
    print(
        fold_table.pivot(on="contrast", index=["group", "fold"], values="difference").sort(
            "group", "fold"
        )
    )
    lines = (
        alt.Chart()
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "fold:O", title="Fold (1 = 2025-11 to 8 = 2026-06)", axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y("difference:Q", title="CRPS difference, points of capacity"),
            color=alt.Color(
                "contrast:N",
                title=None,
                scale=alt.Scale(domain=order[:2], range=[CONTRAST_COLOUR[o] for o in order[:2]]),
                legend=alt.Legend(orient="top"),
            ),
        )
    )
    zero = alt.Chart().mark_rule(color="#292B2B").encode(y=alt.datum(0))  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(lines, zero, data=fold_table)
        .properties(width=WIDTH - 90, height=130)
        .facet(row=alt.Row("group:N", title=None, sort=list(GROUP_NAME.values())))
        .properties(
            title=title_of(
                title="A1 and A2 by fold: the difference in CRPS in each monthly test period",
                subtitle="Each point is the mean difference over the origins of one fold, averaged over 5 seeds. Negative favours the first arm.",
            )
        )
    )
    figs.save(chart, path=out_dir / "contrasts_by_fold.svg")
    # by Primary
    sites = k578.sites()
    label_of = dict(zip(sites["time_series_id"].to_list(), sites["label"].to_list(), strict=True))
    group_of = dict(zip(sites["time_series_id"].to_list(), sites["group"].to_list(), strict=True))
    rows = []
    for pid, v in res["per_primary"].items():
        f = v["first"]
        rows.append(
            {
                "primary": label_of[int(pid)],
                "group": group_of[int(pid)],
                "point": f["point"],
                "lower": f["lower"],
                "upper": f["upper"],
            }
        )
    prim = pl.DataFrame(rows).sort("point")
    print(prim)
    order_p = prim["primary"].to_list()
    # the D12 and D15 intervals are far wider than the rest: clip the axis and say so
    base = alt.Chart(prim).encode(y=alt.Y("primary:N", title=None, sort=order_p))
    gcol = alt.Color(
        "group:N",
        title=None,
        scale=alt.Scale(
            domain=["signed-MW six", "MVA-only eight", "D15 and D18"],
            range=[theme.DATA_BLUE, theme.DATA_PURPLE, theme.BRAND_ORANGE],
        ),
        legend=alt.Legend(orient="top"),
    )
    bars = base.mark_rule(strokeWidth=2, clip=True).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="CRPS difference H minus M*, points of capacity (negative: H is better)",
            scale=alt.Scale(domain=[-0.25, 0.15]),
        ),
        x2="upper:Q",
        color=gcol,
    )
    dots = base.mark_point(filled=True, size=60, clip=True).encode(x="point:Q", color=gcol)  # ty: ignore[unresolved-attribute]
    zero = base.mark_rule(color="#292B2B").encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(bars, dots, zero)
        .properties(width=WIDTH - 90, height=300)
        .properties(
            title=title_of(
                title="A1 by Primary: H minus M*, 95% intervals",
                subtitle="Report-only. Dot: mean difference in CRPS over 8 folds and 5 seeds. Line: 95% interval from resampling whole months, clipped to the axis. D15 (-0.345) and D18 read no solar estimate.",
            )
        )
    )
    figs.save(chart, path=out_dir / "contrasts_by_primary.svg")


def cmd_tables() -> None:
    """Print the stopping passes of every fit log: mean and range by arm and stage, and cap hits."""
    rows = []
    for arm in ("H", "H-S", "H-X", "Z"):
        for path in sorted((ATTENTION_DIR.parent / "logs" / arm).glob("*_fit.json")):
            log = json.loads(path.read_text())
            rows.extend(
                {
                    "arm": arm,
                    "stage": int(e["stage"]),
                    "passes": int(e["passes"]),
                    "cap_hit": bool(e["cap_hit"]),
                    "rerun": bool(e.get("rerun")),
                }
                for e in log["stages"]
            )
    table = pl.DataFrame(rows)
    out = (
        table.group_by("arm", "stage")
        .agg(
            pl.len().alias("fits"),
            pl.col("passes").mean().round(1).alias("mean"),
            pl.col("passes").min().alias("min"),
            pl.col("passes").max().alias("max"),
            pl.col("cap_hit").sum().alias("cap_hits"),
            pl.col("rerun").sum().alias("reruns"),
        )
        .sort("arm", "stage")
    )
    with pl.Config(tbl_rows=50):
        print(out)


def main() -> int:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("losses", "attention", "forecasts", "contrasts", "tables")
    )
    parser.add_argument("out_dir", type=Path, nargs="?")
    args = parser.parse_args()
    if args.command == "tables":
        cmd_tables()
        return 0
    assert args.out_dir is not None, "pass OUT_DIR"
    commands = {
        "losses": cmd_losses,
        "attention": cmd_attention,
        "forecasts": cmd_forecasts,
        "contrasts": cmd_contrasts,
    }
    commands[args.command](out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
