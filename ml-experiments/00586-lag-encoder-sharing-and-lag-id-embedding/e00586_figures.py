# ruff: noqa: E501, D103, PERF401
"""Experiment 586: figures of the result page (contrasts, absolute scores, leak, example forecasts, tails)."""

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00581_summary as summary  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00586-lag-encoder-sharing-and-lag-id-embedding"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
SCORES = WORK / "work" / "scores"
DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
W572 = ML_EXPERIMENTS_DATA_DIR / "00572-two-offset-part-1-lags-versus-the-slow-output-offset/work"
NAMES = {
    "S0": "S0 one encoder, all lags",
    "S1": "S1 shared + learnt ID",
    "S2": "S2 shared, no ID",
    "S3": "S3 separate encoders",
    "S4": "S4 shared + fixed period code",
}
COLOURS = {
    NAMES["S0"]: theme.DATA_BLUE, NAMES["S1"]: theme.BRAND_ORANGE, NAMES["S2"]: "#777777",
    NAMES["S3"]: theme.DATA_GREEN, NAMES["S4"]: theme.DATA_PURPLE,
}  # fmt: skip
GROUPS = {"eight": "eight MVA Primaries", "six": "six MW Primaries"}
BANDS = ["0-24h", "24-120h", "120-240h"]
WIDTH = 600
summary.NAME = NAME


def save(chart: alt.TopLevelMixin, stem: str) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / f"{stem}.svg"
    chart.save(out)
    subprocess.run(
        ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(out)],
        check=False, capture_output=True,
    )  # fmt: skip
    print("wrote", out)


def contrasts() -> None:
    c = pl.read_csv(SCORES / "part1_contrasts.csv").filter(
        (pl.col("slice") == "all rows") & (pl.col("band") != "all") & (pl.col("set") != "report")
    )
    pairs = [("S1", "S0"), ("S2", "S0"), ("S3", "S0"), ("S4", "S0")]
    rows = []
    for a, b in pairs:
        for r in c.filter((pl.col("a") == a) & (pl.col("b") == b)).iter_rows(named=True):
            rows.append({"arm": NAMES[a], "group": GROUPS[r["set"]], "band": r["band"], "diff": r["difference"],
                         "lo": r["low_month"], "hi": r["high_month"], "mde": r["mde"]})  # fmt: skip
    src = pl.DataFrame(rows)
    charts = []
    order = [NAMES[k] for k in ("S1", "S2", "S3", "S4")]
    for g in GROUPS.values():
        for band in BANDS:
            sub = src.filter((pl.col("group") == g) & (pl.col("band") == band))
            base = alt.Chart(sub).encode(
                y=alt.Y("arm:N", title=None, sort=order),
                color=alt.Color("arm:N", scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())), legend=None),
            )  # fmt: skip
            last = g == list(GROUPS.values())[-1] and band == BANDS[-1]
            rule = base.mark_rule(strokeWidth=2).encode(
                x=alt.X(
                    "lo:Q",
                    title="CRPS minus S0's (points of capacity; negative is better)"
                    if last
                    else None,
                    scale=alt.Scale(domain=[-0.5, 0.35]),
                ),
                x2="hi:Q",
            )  # ty: ignore[unresolved-attribute]
            dot = base.mark_point(filled=True, size=60).encode(x="diff:Q")  # ty: ignore[unresolved-attribute]
            zero = (
                alt.Chart(pl.DataFrame({"z": [0.0]}))
                .mark_rule(color="#222222", strokeDash=[3, 3])
                .encode(x="z:Q")
            )
            charts.append(
                alt.layer(zero, rule, dot).properties(
                    width=WIDTH - 150, height=62, title=alt.TitleParams(f"{g}, {band}", fontSize=10)
                )
            )
    chart = alt.vconcat(*charts, spacing=14).properties(
        title=alt.TitleParams(
            text=["No wiring of the lag encoder is shown to beat the one that reads both lags at once"],
            subtitle=[
                "Dot: two-seed mean CRPS of the arm minus S0 (experiment 572's v3+D5), by lead band.",
                "Line: 5-month-block bootstrap, Bonferroni-widened for the 24 counted tests (S1 only; the others are descriptive).",
            ],
            anchor="start", fontSize=13, subtitleFontSize=10,
        )
    )  # fmt: skip
    save(chart, "crps_minus_s0_by_band")


def leak() -> None:
    lk = pl.read_csv(SCORES / "part1_leak.csv").filter(pl.col("arm").is_in(list(NAMES)))
    lk = lk.with_columns(arm=pl.col("arm").replace_strict(NAMES))
    base = alt.Chart(lk).encode(
        y=alt.Y("arm:N", title=None, sort=list(NAMES.values())),
        color=alt.Color("arm:N", scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())), legend=None),
    )  # fmt: skip
    rule = base.mark_rule(strokeWidth=2).encode(
        x=alt.X("lo:Q", title="Leak (points of capacity; smaller is better)"), x2="hi:Q"
    )  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=70).encode(x="leak:Q")  # ty: ignore[unresolved-attribute]
    chart = alt.layer(rule, dot).properties(
        width=WIDTH - 140, height=130,
        title=alt.TitleParams(
            text=["Every wiring leaks an injected 7-day shift about as much as S0"],
            subtitle=["Dot: mean over 5 folds (seed 0). Line: smallest to largest fold.", "Leak: how far the median forecast moves, in points of capacity, when the lag inputs carry a 7-day injected shift."],
            anchor="start", fontSize=13, subtitleFontSize=10,
        ),
    )  # fmt: skip
    save(chart, "leak_by_arm")


def forecast_rows(
    *,
    arm: str,
    month: str,
    ts: int,
    lo: datetime,
    hi: datetime,
    lead: tuple[int, int] | None,
    origin: datetime | None,
) -> pl.DataFrame:
    root = W572 if arm == "S0" else WORK / "arms" / arm.lower() / "work"
    parts = [
        pl.read_parquet(
            root / "fits" / "V3D5" / f"{month}_{d}_s{s}.parquet",
            columns=["time_series_id", "power_fcst_init_time", "valid_time", "q05", "q50", "q95"],
        )
        for d in ("d0", "d14")
        for s in (0, 1)
    ]
    df = pl.concat(parts).filter(pl.col("time_series_id") == ts).group_by(["time_series_id", "power_fcst_init_time", "valid_time"]).agg(pl.col("q05").mean(), pl.col("q50").mean(), pl.col("q95").mean())  # fmt: skip
    df = df.with_columns(
        lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours()
    )
    df = df.filter((pl.col("valid_time") >= lo) & (pl.col("valid_time") < hi))
    if lead:
        df = df.filter((pl.col("lead") >= lead[0]) & (pl.col("lead") < lead[1]))
    if origin:
        df = df.filter(pl.col("power_fcst_init_time") == origin)
    return df.sort("valid_time").with_columns(arm=pl.lit(NAMES[arm]))


def examples() -> None:
    sm = k578.sites()
    ts = int(sm.filter(pl.col("label") == "D8")["time_series_id"][0])
    p = int(sm.filter(pl.col("label") == "D8")["primary"][0])
    cap = float(sm.filter(pl.col("label") == "D8")["capacity_mw"][0])
    clean = k578.clean()
    panels = []
    for month, d0, label, origin in (
        ("2026-01", datetime(2026, 1, 12, tzinfo=UTC), "Winter (January 2026)", datetime(2026, 1, 12, 9, tzinfo=UTC)),
        ("2026-06", datetime(2026, 6, 8, tzinfo=UTC), "Summer (June 2026)", datetime(2026, 6, 8, 9, tzinfo=UTC)),
    ):  # fmt: skip
        for kind in ("day", "horizon"):
            if kind == "day":
                lo, hi = d0, d0.replace(day=d0.day + 3)
                fr = pl.concat(
                    [
                        forecast_rows(
                            arm=a, month=month, ts=ts, lo=lo, hi=hi, lead=(24, 48), origin=None
                        )
                        for a in ("S0", "S1", "S3")
                    ]
                )
                title = f"{label}: day-ahead forecasts (lead 24 to 48 h), 3 days"
            else:
                lo, hi = origin, origin.replace(day=origin.day + 10)
                fr = pl.concat(
                    [
                        forecast_rows(
                            arm=a, month=month, ts=ts, lo=lo, hi=hi, lead=None, origin=origin
                        )
                        for a in ("S0", "S1", "S3")
                    ]
                )
                title = f"{label}: one forecast run, all leads to 10 days"
            idx = (
                (fr["valid_time"].dt.epoch("us").to_numpy() - k578.prep.GRID_START_US)
                // k578.prep.STEP_US
            ).astype(int)
            fr = fr.with_columns(y=pl.Series(100 * clean[p, idx]), day=(pl.col("valid_time") - pl.lit(lo)).dt.total_hours() / 24.0, q05=100 * pl.col("q05"), q50=100 * pl.col("q50"), q95=100 * pl.col("q95"))  # fmt: skip
            shown = [NAMES[k] for k in ("S0", "S1", "S3")]
            col = alt.Color(
                "arm:N",
                scale=alt.Scale(domain=shown, range=[COLOURS[n] for n in shown]),
                legend=alt.Legend(orient="top", title=None) if not panels else None,
            )
            base = alt.Chart(fr).encode(
                x=alt.X(
                    "day:Q",
                    title="Days from the start of the window"
                    if kind == "horizon" and month == "2026-06"
                    else None,
                ),
                color=col,
            )
            band = (
                alt.Chart(fr.filter(pl.col("arm") == NAMES["S0"]))
                .mark_area(opacity=0.18, color=theme.DATA_BLUE)
                .encode(
                    x="day:Q",
                    y=alt.Y("q05:Q", title="Power (% of capacity)", scale=alt.Scale(zero=False)),
                    y2="q95:Q",
                )
            )  # ty: ignore[unresolved-attribute]
            med = base.mark_line(strokeWidth=1.2).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
            act = (
                alt.Chart(fr.filter(pl.col("arm") == NAMES["S0"]))
                .mark_line(color="#222222", strokeWidth=1.0)
                .encode(x="day:Q", y="y:Q")
            )  # ty: ignore[unresolved-attribute]
            panels.append(
                alt.layer(band, med, act).properties(
                    width=WIDTH, height=110, title=alt.TitleParams(title, fontSize=10)
                )
            )
    chart = alt.vconcat(*panels, spacing=8).properties(
        title=alt.TitleParams(
            text=["Example forecasts at substation D8: S0, S1, and S3 are close"],
            subtitle=["Line: median. Shaded: S0's q05 to q95. Dark line: observed power (cleaned). Mean of seeds 0 and 1.", f"Power is a share of the substation's {cap:.1f} MVA capacity."],
            anchor="start", fontSize=13, subtitleFontSize=10,
        )
    )  # fmt: skip
    save(chart, "example_forecasts_d8")


def tails() -> None:
    arms = {"v3d5": "S0", "s1": "S1", "s2": "S2", "s3": "S3", "s4": "S4"}
    t = summary.injected(arms=list(arms)).with_columns(
        arm=pl.col("arm").replace_strict({k: NAMES[v] for k, v in arms.items()})
    )
    order = ["before", "d0-3", "d4-end", "after0-7", "after7-14"]
    charts = []
    for metric, title, domain in (
        ("follow_q50", "Share of shift followed", None),
        ("nra_side95", "Safe-side share", [0, 1]),
    ):
        ch = alt.Chart(t).mark_line(point=True).encode(
            x=alt.X("phase:N", sort=order, title="Forecast origin relative to the injected event" if metric == "nra_side95" else None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{metric}:Q", title=title, scale=alt.Scale(domain=domain) if domain else alt.Scale(zero=True)),
            color=alt.Color("arm:N", scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())), legend=alt.Legend(orient="top", title=None)),
        ).properties(width=WIDTH - 60, height=100).facet(row=alt.Row("group:N", title=None))  # ty: ignore[unresolved-attribute]  # fmt: skip
        charts.append(ch)
    chart = alt.vconcat(*charts, spacing=12).properties(
        title=alt.TitleParams(text=["Lag-following on injected 3, 7, and 15-day events"], subtitle=["Experiment 581's injected bank; lead 24 to 120 h; mean over events and 5 folds.", "Upper panels: share of the shift the median follows (smaller is better).", "Lower panels: share of true readings on the safe side of q95, or q05 for a shift that lowered load (larger is better)."], anchor="start", fontSize=13, subtitleFontSize=10)
    )  # fmt: skip
    save(chart, "injected_event_following_by_phase")


WIDE_NAMES = {
    "S0w": "S0w one encoder, 4 lags",
    "S1w": "S1w shared + learnt ID",
    "S3w": "S3w separate encoders",
}
WIDE_COLOURS = {
    WIDE_NAMES["S0w"]: theme.DATA_BLUE, WIDE_NAMES["S1w"]: theme.BRAND_ORANGE, WIDE_NAMES["S3w"]: theme.DATA_GREEN,
}  # fmt: skip
WIDE_SCORES = WORK / "work_wide" / "scores"


def wide() -> None:
    c = pl.read_csv(WIDE_SCORES / "part1_contrasts.csv").filter(
        (pl.col("band") == "24-120h") & (pl.col("set") != "report") & (pl.col("b") == "S0w")
        & pl.col("slice").is_in(["all rows", "cold winter nights", "Christmas and New Year", "bank-holiday days"])
    )  # fmt: skip
    rows = [
        {"arm": WIDE_NAMES[r["a"]], "group": GROUPS[r["set"]], "slice": r["slice"], "diff": r["difference"],
         "lo": r["low_month"], "hi": r["high_month"], "folds": r["folds"]}
        for r in c.iter_rows(named=True)
    ]  # fmt: skip
    src = pl.DataFrame(rows)
    order = [WIDE_NAMES["S1w"], WIDE_NAMES["S3w"]]
    charts = []
    slices = ["all rows", "cold winter nights", "Christmas and New Year", "bank-holiday days"]
    for g in GROUPS.values():
        for sl in slices:
            sub = src.filter((pl.col("group") == g) & (pl.col("slice") == sl))
            base = alt.Chart(sub).encode(
                y=alt.Y("arm:N", title=None, sort=order),
                color=alt.Color("arm:N", scale=alt.Scale(domain=list(WIDE_COLOURS), range=list(WIDE_COLOURS.values())), legend=None),
            )  # fmt: skip
            last = g == list(GROUPS.values())[-1] and sl == slices[-1]
            rule = base.mark_rule(strokeWidth=2).encode(
                x=alt.X("lo:Q", title="CRPS minus S0w's (points of capacity; negative is better)" if last else None, scale=alt.Scale(domain=[-1.5, 0.7]), axis=alt.Axis(values=[-1.5, -1.0, -0.5, 0.0, 0.5])),
                x2="hi:Q",
            )  # ty: ignore[unresolved-attribute]  # fmt: skip
            dot = base.mark_point(filled=True, size=60).encode(x="diff:Q")  # ty: ignore[unresolved-attribute]
            zero = (
                alt.Chart(pl.DataFrame({"z": [0.0]}))
                .mark_rule(color="#222222", strokeDash=[3, 3])
                .encode(x="z:Q")
            )
            n = int(sub["folds"][0]) if sub.height else 0
            charts.append(
                alt.layer(zero, rule, dot).properties(
                    width=WIDTH - 150,
                    height=50,
                    title=alt.TitleParams(
                        f"{g}, {sl} ({n} fold{'s' if n != 1 else ''})", fontSize=10
                    ),
                )
            )
    chart = alt.vconcat(*charts, spacing=10).properties(
        title=alt.TitleParams(
            text=["With four lags, neither shared-weight wiring beats one encoder reading all lags"],
            subtitle=[
                "Dot: two-seed mean CRPS of the arm minus S0w (one encoder reading day, week, 2-week, and year-ago lags), lead 24 to 120 h.",
                "Line: smallest to largest month-block fold mean; absent where one fold holds the slice. Slices are descriptive.",
            ],
            anchor="start", fontSize=13, subtitleFontSize=10,
        )
    )  # fmt: skip
    save(chart, "wide_crps_minus_s0w_by_slice")


def wide_tails() -> None:
    arms = {"s0w": "S0w", "s1w": "S1w", "s3w": "S3w"}
    t = summary.injected(arms=list(arms)).with_columns(
        arm=pl.col("arm").replace_strict({k: WIDE_NAMES[v] for k, v in arms.items()})
    )
    order = ["before", "d0-3", "d4-end", "after0-7", "after7-14"]
    charts = []
    for metric, title, domain in (
        ("follow_q50", "Share of shift followed", None),
        ("nra_side95", "Safe-side share", [0, 1]),
    ):
        ch = alt.Chart(t).mark_line(point=True).encode(
            x=alt.X("phase:N", sort=order, title="Forecast origin relative to the injected event" if metric == "nra_side95" else None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{metric}:Q", title=title, scale=alt.Scale(domain=domain) if domain else alt.Scale(zero=True)),
            color=alt.Color("arm:N", scale=alt.Scale(domain=list(WIDE_COLOURS), range=list(WIDE_COLOURS.values())), legend=alt.Legend(orient="top", title=None)),
        ).properties(width=WIDTH - 60, height=100).facet(row=alt.Row("group:N", title=None))  # ty: ignore[unresolved-attribute]  # fmt: skip
        charts.append(ch)
    chart = alt.vconcat(*charts, spacing=12).properties(
        title=alt.TitleParams(text=["Four-lag arms on experiment 581's injected load shifts"], subtitle=["Lead 24 to 120 h; mean over events and 5 folds.", "Upper panels: share of the shift the median follows (smaller is better).", "Lower panels: share of true readings on the safe side of q95, or q05 for a shift that lowered load (larger is better)."], anchor="start", fontSize=13, subtitleFontSize=10)
    )  # fmt: skip
    save(chart, "wide_injected_event_following_by_phase")


if __name__ == "__main__":
    for fn in sys.argv[1:] or ["contrasts", "leak", "examples", "tails"]:
        globals()[fn]()
