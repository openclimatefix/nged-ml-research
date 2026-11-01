# ruff: noqa: E501
"""Figures of experiment 477: neural capacity beside the GAM estimate, recovery, and learned PV.

Substations may be named; no metered generator's series is drawn (the injected farms appear only
as recovery ratios). Writes SVG figures into the experiment's docs folder.

```bash
. <the environment script that sets DATA_PATH_INTERNAL and PYTHONPATH>
uv run --no-sync python ml-experiments/00477-*/e00477_plots.py
```
"""

import json
from datetime import timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00405_prep as p5  # ty: ignore[unresolved-import]
import e00416_score as s416  # ty: ignore[unresolved-import]
import e00477_inject as inj  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WIDTH: Final[int] = 680
DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00477-neural-pv-module-placebo-gate-and-known-answer-on-stored-fits"
)
GAM: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals"
    / "work"
    / "nged_table.parquet"
)
W416: Final[Path] = s416.WORK / "seed_preds"
SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
SITES: Final[tuple[str, ...]] = (*SIX, "D15", "D18")
CELL: Final[int] = 2
"""Season-year cell 2 is July 2025 to June 2026."""
REPORT_ONLY: Final[dict[str, str]] = {
    "D5": "export limit",
    "D15": "report-only",
    "D18": "report-only",
}


def neural_capacity() -> pl.DataFrame:
    """Return one row per fit, arm, and site: the learned capacity (MW) of the latest season-year."""
    rows = []
    for arm, folder in (("C3", W416), ("C3-nolag", inj.WORK / "seed_preds")):
        for path in sorted((folder / arm).glob("*.capacity.json")):
            caps = json.loads(path.read_text())
            rows.extend(
                {"arm": arm, "file": path.name, "site": s, "mw": caps[s][CELL]} for s in caps
            )
    return pl.DataFrame(rows)


def capacity_figure() -> alt.Chart:
    """Neural and GAM capacity per Primary, with the 10th to 90th percentile."""
    neural = (
        neural_capacity()
        .group_by("arm", "site")
        .agg(
            low=pl.col("mw").quantile(0.1),
            mid=pl.col("mw").median(),
            high=pl.col("mw").quantile(0.9),
        )
    )
    gam = (
        pl.read_parquet(GAM)
        .filter(pl.col("season_year") == 2025)
        .select(
            "site",
            arm=pl.lit("GAM (experiment 474)"),
            low=pl.col("ac_p10"),
            mid=pl.col("ac_p50"),
            high=pl.col("ac_p90"),
        )
    )
    neural = neural.with_columns(
        arm=pl.col("arm").replace(
            {
                "C3": "Neural PV-split (C3)",
                "C3-nolag": "Neural, no lags (C3-nolag)",
            }
        )
    )
    long = pl.concat([neural, gam.select(neural.columns)]).filter(pl.col("site").is_in(SITES))
    long = long.with_columns(
        label=pl.col("site")
        + pl.col("site").replace_strict(
            {s: f" ({REPORT_ONLY[s]})" for s in REPORT_ONLY}, default=""
        )
    )
    prior = pl.DataFrame(
        {
            "site": list(SITES),
            "prior": [p5.studies_capacity_2025(site=s) for s in SITES],
        }
    ).with_columns(
        label=pl.col("site")
        + pl.col("site").replace_strict(
            {s: f" ({REPORT_ONLY[s]})" for s in REPORT_ONLY}, default=""
        )
    )
    order = prior.sort(pl.col("site").str.slice(1).cast(pl.Int64))["label"].to_list()
    scale = alt.Scale(
        domain=[
            "Neural PV-split (C3)",
            "Neural, no lags (C3-nolag)",
            "GAM (experiment 474)",
        ],
        range=[ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.BRAND_ORANGE],
    )
    enc = alt.Chart(long).encode(
        y=alt.Y("label:N", sort=order, title=None),
        yOffset=alt.YOffset("arm:N"),
        color=alt.Color(
            "arm:N", scale=scale, legend=alt.Legend(orient="top", title=None, columns=1)
        ),
    )
    bars = enc.mark_rule(strokeWidth=4, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Estimated unmetered solar capacity (MW AC)"), x2="high:Q"
    )
    dots = enc.mark_point(filled=True, size=50, aria=False).encode(x="mid:Q")  # ty: ignore[unresolved-attribute]
    ticks = (
        alt.Chart(prior)  # ty: ignore[unresolved-attribute]
        .mark_tick(color="#606060", thickness=2, size=30, aria=False)
        .encode(y=alt.Y("label:N", sort=order), x="prior:Q")
    )
    return alt.layer(bars, dots, ticks).properties(  # ty: ignore[invalid-return-type]
        width=WIDTH - 200,
        height=80 * 3 + 40,
        title=alt.TitleParams(
            text="Neural capacity medians sit below the GAM at 5 of 7 Primaries and fail the placebo gate",
            subtitle=[
                "Season-year July 2025 to June 2026. Dot: median. Line: 10th to 90th percentile",
                "over fits (neural) or the bootstrap interval (GAM). Grey tick: the studies page's capacity.",
                "No neural capacity passed the placebo gate. Intervals are wide at D5 and D9.",
            ],
            anchor="start",
        ),
    )


def recovery_figure() -> alt.Chart:
    """Recovery of the injected 4 MW farm by condition and arm, per Primary."""
    rec = json.loads((inj.WORK / "recovery.json").read_text())
    rows = []
    for key, v in rec.items():
        arm, cond, site = key.split("|")
        if cond not in {"held_C_mw4", "placebo_C_mw4", "profile_C_mw4"}:
            continue
        rows.append(
            {
                "arm": arm,
                "cond": {
                    "held_C_mw4": "Real farm",
                    "placebo_C_mw4": "Placebo",
                    "profile_C_mw4": "Profile only",
                }[cond],
                "site": site,
                "mid": v["equivalent_ratio_mean"],
                "low": v["equivalent_ratio_mean"] - v["equivalent_ratio_sd"],
                "high": v["equivalent_ratio_mean"] + v["equivalent_ratio_sd"],
            }
        )
    data = pl.DataFrame(rows).filter(pl.col("site").is_in(SIX))
    arms = {
        "C3": "Neural PV-split (C3)",
        "C3-nolag": "Neural, no lags (C3-nolag)",
        "A0-doy": "Neural, free irradiance (A0-doy)",
    }
    data = data.with_columns(arm=pl.col("arm").replace(arms))
    base_chart = alt.Chart(data).encode(
        y=alt.Y("site:N", sort=list(SIX), title=None),
        yOffset=alt.YOffset("cond:N"),
        color=alt.Color(
            "cond:N",
            scale=alt.Scale(
                domain=["Real farm", "Placebo", "Profile only"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.GREY_2],
            ),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    bars = base_chart.mark_rule(strokeWidth=3, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Read-out over injected MW"),
        x2="high:Q",
    )
    dots = base_chart.mark_point(filled=True, size=40, aria=False).encode(x="mid:Q")  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(bars, dots)  # ty: ignore[invalid-return-type]
        .properties(width=170, height=170)
        .facet(column=alt.Column("arm:N", title=None))
        .properties(
            title=alt.TitleParams(
                text="Every network reads out the placebo as much as the real farm",
                subtitle=[
                    "A 4 MW farm is injected into six Primaries' target and lags, 2026-06 fold, 3 seeds.",
                    "Dot: seed mean. Line: one seed standard deviation. 1 is the full injection.",
                    "The free-irradiance network was fitted on the placebo and profile only; the no-lag network on the real farm only.",
                ],
                anchor="start",
            )
        )
    )


def pv_by_hour_figure() -> alt.Chart:
    """Mean learned PV (MW) by local hour on sunny and cloudy days, C3, seed 0, 12 months."""
    cams = (
        pl.read_parquet(
            s416.CAMS_FRAME_PATH, columns=["site", "kind", "time", "ghi_u", "ghi_u_clear"]
        )
        .filter(pl.col("kind") == "substation", pl.col("ghi_u_clear") > 0)
        .select(label="site", valid_time="time", csi=pl.col("ghi_u") / pl.col("ghi_u_clear"))
    )
    ts = adapter.series_table().select("time_series_id", "label")
    pieces = []
    for month in s416.TEST_MONTHS_12:
        f = W416 / "C3" / f"{month}_s0.solar.parquet"
        if not f.exists():
            f = s416.WORK_409 / "seed_preds" / "C3" / f"{month}_s0.solar.parquet"
        pieces.append(pl.read_parquet(f))
    pv = (
        pl.concat(pieces)
        .join(ts, on="time_series_id")
        .filter(pl.col("label").is_in(SIX))
        .filter(pl.col("valid_time") - pl.col("power_fcst_init_time") < timedelta(hours=36))
        .join(cams, on=["label", "valid_time"], how="inner")
    )
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    pv = pv.with_columns(day=local.dt.date(), hour=local.dt.hour() + local.dt.minute() / 60)
    midday = (
        pv.filter(pl.col("hour").is_between(10, 15))
        .group_by("label", "day")
        .agg(csi_day=pl.col("csi").mean())
    )
    pv = (
        pv.join(midday, on=["label", "day"])
        .with_columns(
            sky=pl.when(pl.col("csi_day") >= 0.7)
            .then(pl.lit("Sunny day (index 0.7+)"))
            .when(pl.col("csi_day") <= 0.3)
            .then(pl.lit("Cloudy day (index 0.3-)"))
            .otherwise(None)
        )
        .drop_nulls("sky")
    )
    curve = pv.group_by("label", "sky", pl.col("hour").floor().alias("h")).agg(
        mw=pl.col("solar_mw").mean()
    )
    return (
        alt.Chart(curve)  # ty: ignore[unresolved-attribute]
        .mark_line(aria=False)
        .encode(
            x=alt.X(
                "h:Q", title="Hour of day (UK local time)", axis=alt.Axis(values=[0, 6, 12, 18, 24])
            ),
            y=alt.Y("mw:Q", title="Learned PV (MW)"),
            color=alt.Color(
                "sky:N",
                scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=150, height=110)
        .facet(facet=alt.Facet("label:N", sort=list(SIX), title=None), columns=3)
        .resolve_scale(y="independent")
        .properties(
            title=alt.TitleParams(
                text="Learned PV output of the PV module, sunny against cloudy days",
                subtitle=[
                    "C3, seed 0, 12 test months, forecasts up to 36 hours ahead. D5 has an export limit."
                ],
                anchor="start",
            )
        )
    )


def main() -> None:
    """Write the figures."""
    DOCS.mkdir(parents=True, exist_ok=True)
    capacity_figure().save(DOCS / "figure_capacity_neural_and_gam.svg")
    recovery_figure().save(DOCS / "figure_recovery.svg")
    pv_by_hour_figure().save(DOCS / "figure_pv_by_hour.svg")


if __name__ == "__main__":
    main()
