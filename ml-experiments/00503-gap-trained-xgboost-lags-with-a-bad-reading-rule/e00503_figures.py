# ruff: noqa: E501
r"""Draw experiment 503's figures from the scored rows, the sweep sums, and the stored forecasts.

Writes `docs/ml-experiments/00503-*/figure_*.svg` and `contrasts.csv`. Arms: (a) the original
forecast, (f) the 50/50 mixture of (a) and the lagless forecast, G the gap-trained forecast, and G-B
the gap-trained forecast with all three serving rules.

```bash
PYTHONPATH=<the paths of e00503_score.py plus ml-experiments/00465-...> DATA_PATH_INTERNAL=... \
    E503_DEVICE=cpu uv run --no-sync python ml-experiments/00503-*/e00503_figures.py
```
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00392_score as s392  # ty: ignore[unresolved-import]
import e00465_figures as fig465  # ty: ignore[unresolved-import]
import e00465_sweep as sweep465  # ty: ignore[unresolved-import]
import e00503_fit as fit  # ty: ignore[unresolved-import]
import e00503_rules as rules  # ty: ignore[unresolved-import]
import e00503_score as score  # ty: ignore[unresolved-import]
import e00503_sweep as sweep  # ty: ignore[unresolved-import]
import e00503_sweep_report as report  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from plotting import ocf_theme as theme
from studies.eval_protocol import metrics

SLUG: Final[str] = "00503-gap-trained-xgboost-lags-with-a-bad-reading-rule"
FIGURE_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 600
RULE_LABELS: Final[dict[str, str]] = {
    "zero": "Zero",
    "far": "Far from the lagless forecast",
    "stuck": "Stuck run",
}
ARM_LABELS: Final[dict[str, str]] = {
    "a": "(a) original forecast",
    "f": "(f) mixture of (a) and lagless",
    "gb": "G-B (gap-trained, three rules)",
    "g": "G (gap-trained, no rule)",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "a": theme.DATA_BLUE,
    "f": theme.DATA_PURPLE,
    "gb": theme.BRAND_ORANGE,
    "g": theme.DATA_GREEN,
}
CASES: Final[dict[str, str]] = {
    "dropout 6h": "Dropout to zero, 6 h",
    "stuck 24h": "Stuck reading, 24 h",
    "spike x5": "Spike, 5 MAD",
    "spike x8": "Spike, 8 MAD",
    "spike x15": "Spike, 15 MAD",
}


def arm_scale() -> alt.Scale:
    """Return the colour scale of the plotted arms."""
    return alt.Scale(
        domain=[ARM_LABELS[a] for a in ("a", "f", "gb")],
        range=[ARM_COLOURS[a] for a in ("a", "f", "gb")],
    )


def contrasts() -> pl.DataFrame:
    """Compute the pre-registered contrasts and write `contrasts.csv`."""
    rows = pl.read_parquet(score.OUT / "rows_w1.0.parquet")
    six = score.SIX
    out = []
    for arm, control in (*score.CONTRASTS, ("arep", "a")):
        for row_set in ("outside", "harm"):
            cond = score.BANDS["0-240 h"] & six & score.ROW_SETS[row_set]
            level = score.LEVEL_PERCENT
            if row_set == "outside":
                result = metrics.month_bootstrap_contrast(
                    treatment=rows.filter(pl.col("arm") == arm),
                    reference=rows.filter(pl.col("arm") == control),
                    metric="crps",
                    where=cond,
                    level_percent=level,
                    n_resamples=score.RESAMPLES,
                )
                low, high, difference = result["lower"], result["upper"], result["difference"]
            else:
                (difference, low, high, _), _, _ = score.harm_contrast(
                    treatment=rows.filter((pl.col("arm") == arm) & cond),
                    reference=rows.filter((pl.col("arm") == control) & cond),
                    level=level,
                )
            out.append(
                {
                    "arm": arm,
                    "control": control,
                    "row_set": row_set,
                    "difference": difference,
                    "lower": low,
                    "upper": high,
                }
            )
    frame = pl.DataFrame(out)
    frame.write_csv(FIGURE_DIR / "contrasts.csv")
    return frame


def contrast_figure(*, frame: pl.DataFrame) -> None:
    """Draw G, G-A, and G-B minus (a) and (b), with the noise floor, on both row sets."""
    names = {"g": "G", "ga": "G-A", "gb": "G-B", "arep": "(a) repeat"}
    data = frame.with_columns(
        label=pl.col("arm").replace(names)
        + pl.lit(" minus ")
        + pl.col("control").replace({"a": "(a)", "b": "(b)"}),
        panel=pl.col("row_set").replace(
            {
                "outside": "Rows with no labelled event (Bonferroni 95% month-block interval)",
                "harm": "Harm rows: a lag in a labelled event, the target outside (whole-event interval)",
            }
        ),
    )
    base = alt.Chart(data.to_pandas())
    band = base.mark_rect(opacity=0.12, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.datum(-score.EQUIVALENCE_LIMIT), x2=alt.datum(score.EQUIVALENCE_LIMIT)
    )
    zero = base.mark_rule(color=theme.GREY_2, strokeDash=[4, 3]).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    line = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower:Q"),
        x2="upper:Q",
        y=alt.Y("label:N", title=None, sort=None),
        color=alt.value(theme.DATA_BLUE),
    )
    dots = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="CRPS difference (points of capacity; negative = first arm better)",
        ),
        y=alt.Y("label:N", title=None, sort=None),
        color=alt.value(theme.BRAND_ORANGE),
    )
    chart = (
        alt.layer(band, zero, line, dots)
        .properties(width=WIDTH - 140, height=150)
        .facet(
            row=alt.Row(
                "panel:N",
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left", labelLimit=560),
            )
        )
        .resolve_scale(x="independent")
        .properties(
            title=alt.TitleParams(
                [
                    "Gap-trained forecasts (G; G-A, which equals G on real rows; G-B) minus the stored forecasts,",
                    "six Primaries metered in signed MW, leads up to 240 hours. Dot: estimate. Line: interval.",
                    "Blue band: the equivalence limit of 0.11 points. The repeat of (a) gives the noise floor.",
                ],
                anchor="start",
                fontSize=11,
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_1_contrasts.svg")


def shift_figure() -> None:
    """Draw the median shift as a share of the corruption, by injection and arm (leads 24-72 h)."""
    table = report.pooled(widening=1.0).filter(
        (pl.col("band") == "24-72 h") & pl.col("kind").is_in(["stuck", "dropout", "spike"])
    )
    arms = {
        "a": "(a) original forecast",
        "G": "G (gap-trained, no rule)",
        "G-A": "G-A (rules zero and stuck)",
        "G-B": "G-B (three rules)",
    }
    data = (
        table.filter(pl.col("arm").is_in(list(arms)))
        .with_columns(
            case=(pl.col("kind") + pl.lit(" ") + pl.col("variant")).replace(CASES),
            arm=pl.col("arm").replace(arms),
        )
        .select("case", "arm", "share")
    )
    chart = (
        alt.Chart(data.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("case:N", title=None, sort=list(CASES.values())),
            x=alt.X(
                "share:Q",
                title="Shift of the median as a share of the corruption of the lags (smaller is better)",
            ),
            yOffset=alt.YOffset("arm:N", sort=list(arms.values())),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(arms.values()),
                    range=[theme.DATA_BLUE, theme.DATA_GREEN, theme.DATA_SKY, theme.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=300),
            ),
        )
        .properties(
            width=WIDTH - 140,
            height=260,
            title=alt.TitleParams(
                [
                    "Bad readings injected into the six Primaries' own readings, lead 24 to 72 hours.",
                    "Share = summed absolute shift of the median over the summed absolute change of",
                    "the three lag values. MAD: the stage-1 residual median absolute deviation.",
                ],
                anchor="start",
                fontSize=11,
            ),
        )
    )
    chart.save(FIGURE_DIR / "figure_2_shift_share.svg")


def firing_figure() -> None:
    """Draw how often each rule sets a lag to missing on real rows, by row set."""
    rows = pl.read_parquet(score.OUT / "rows_w1.0.parquet")
    fire = pl.concat([pl.read_parquet(p) for p in sorted((fit.ARM_DIR / "fire").glob("*.parquet"))])
    on_g = rows.filter(
        (pl.col("arm") == "g") & score.SIX & (pl.col("lead_hours") <= score.MAX_LEAD_HOURS)
    ).join(fire, on=list(s392.KEY_COLUMNS), how="left", validate="1:1")
    out = []
    for name, cond in (
        ("Rows with no labelled event", score.ROW_SETS["outside"]),
        ("Harm rows", score.ROW_SETS["harm"]),
        ("Inside labelled events", score.ROW_SETS["during"]),
        ("7 days after an event", score.ROW_SETS["after7d"]),
    ):
        sub = on_g.filter(cond)
        out.extend(
            {
                "row set": name,
                "rule": RULE_LABELS[rule],
                "share": float(sub[f"fire_{rule}"].gt(0).cast(pl.Float64).fill_null(0.0).mean()),  # ty: ignore[invalid-argument-type]
                "rows": sub.height,
            }
            for rule in rules.RULE_NAMES
        )
    frame = pl.DataFrame(out)
    print(frame)
    chart = (
        alt.Chart(frame.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("row set:N", title=None, sort=None),
            x=alt.X(
                "share:Q",
                title="Share of forecast rows in which the rule sets at least one lag to missing",
                axis=alt.Axis(format="%"),
            ),
            yOffset=alt.YOffset("rule:N"),
            color=alt.Color(
                "rule:N",
                scale=alt.Scale(
                    domain=["Zero", "Far from the lagless forecast", "Stuck run"],
                    range=[theme.DATA_BLUE, theme.BRAND_ORANGE, theme.DATA_PURPLE],
                ),
                legend=alt.Legend(title=None, orient="top", columns=3),
            ),
        )
        .properties(
            width=WIDTH - 140,
            height=170,
            title=alt.TitleParams(
                [
                    "How often each serving rule fires on the real rows of the four scored months,",
                    "six Primaries metered in signed MW, leads up to 240 hours.",
                ],
                anchor="start",
                fontSize=11,
            ),
        )
    )
    chart.save(FIGURE_DIR / "figure_3_rule_firing.svg")


def window_from(
    *,
    label: str,
    origin: datetime,
    days: int,
    arms: dict[str, pl.DataFrame],
    actuals: pl.DataFrame,
    series: pl.DataFrame,
) -> pl.DataFrame:
    """Return the plotted quantiles (% of capacity) and the actual for each arm in `arms`."""
    one = series.filter(pl.col("label") == label)
    capacity, series_id = one["capacity_mw"][0], one["time_series_id"][0]
    q = list(s392.QUANTILES)
    pieces = [
        frame.sort("valid_time")
        .filter(pl.col("valid_time") <= origin + pl.duration(days=days))
        .with_columns(arm=pl.lit(ARM_LABELS[name]))
        .select("valid_time", "arm", *[(pl.col(c) * 100.0 / capacity).alias(c) for c in q])
        for name, frame in arms.items()
    ]
    out = pl.concat(pieces).with_columns(
        hour=(pl.col("valid_time") - origin).dt.total_minutes() / 60.0
    )
    act = actuals.filter(pl.col("time_series_id") == series_id).select(
        "valid_time", actual=pl.col("actual") * 100.0 / capacity
    )
    return out.join(act, on="valid_time", how="left")


def stored_arms(*, label: str, origin: datetime, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the stored forecasts of arms a, f, and G-B for one Primary and origin (MW)."""
    one = series.filter(pl.col("label") == label)
    fold = s392.FOLD_STARTS.index(max(r for r in s392.FOLD_STARTS if r <= origin))
    frames = {}
    for arm in ("a", "b", "gb"):
        stored = s392.read_fold(directory=score.SOURCES[arm], sites=[label], fold=fold)
        assert stored is not None
        frames[arm] = adapter.forecast_frame(stored=stored, series=one, unit="fraction").filter(
            pl.col("power_fcst_init_time") == origin
        )
    frames["f"] = score.mix_frames(first=frames["a"], second=frames["b"])
    return {k: frames[k] for k in ("a", "f", "gb")}


def injected_arms(
    *, label: str, origin: datetime, kind: str, series: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    """Return forecasts of a, f, and G-B (as MW-fraction frames) after injecting `kind` readings."""
    fd_series, arrays = fd._series_and_arrays()
    row = fd_series.filter(pl.col("site") == label).to_dicts()[0]
    arrs = arrays[label]
    refit = max(r for r in s392.FOLD_STARTS if r <= origin)
    columns = fit.columns()
    positions = [columns.index(c) for c in common.LAG_COLUMNS]
    median, mad = rules.stage1_median(
        site=label, capacity_mw=row["effective_capacity_mw"], refit=refit
    )
    frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
    prepared = fd.prepared_run(
        run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
        row=row,
        arrays=arrs,
        a7=False,
    )
    members = prepared["ensemble_member"].n_unique()
    index = int(common.grid_index(times=origin))
    depth = 15.0 * mad if kind == "spike" else 0.0
    power = sweep465.modified_readings(
        power=arrs.power, kind=kind, origin=index, size=0, depth=depth, first_origin=0
    )
    target = np.asarray(common.grid_index(times=prepared["valid_time"]))
    origin_index = np.asarray(common.grid_index(times=prepared["power_fcst_init_time"]))
    matrix = sweep.lag_matrix(prepared=prepared, power=power, columns=columns)
    flags = sweep.case_flags(
        power=power, target=target, origin=origin_index, median=median, mad=mad
    )
    keys = prepared.select(*common.KEYS).unique(maintain_order=True)
    out: dict[str, pl.DataFrame] = {}
    for booster, name, rule_set in (("a", "a", "none"), ("g", "gb", "iii2")):
        model = xgb.Booster()
        model.load_model(sweep.BOOSTER_DIRS[booster] / f"{label}_s0_{refit:%Y%m%d}.ubj")
        model.set_param({"device": "cpu", "nthread": 4})
        ruled = rules.apply_rules(
            matrix=matrix,
            lag_positions=positions,
            flags=flags,
            rules=rules.rules_to_apply(name=rule_set),
        )
        raw = model.predict(xgb.DMatrix(ruled))
        out[name] = fd.mixture_frame(keys=keys, raw=raw, members=members, seed=0)
    stored = {}
    for arm in ("b",):
        fold = s392.FOLD_STARTS.index(refit)
        got = s392.read_fold(directory=score.SOURCES[arm], sites=[label], fold=fold)
        assert got is not None
        stored[arm] = got.filter(pl.col("power_fcst_init_time") == origin)
    adapted = {
        name: adapter.forecast_frame(
            stored=frame_, series=series.filter(pl.col("label") == label), unit="fraction"
        )
        for name, frame_ in {**out, **stored}.items()
    }
    adapted["f"] = score.mix_frames(first=adapted["a"], second=adapted["b"])
    return {k: adapted[k] for k in ("a", "f", "gb")}


def series_chart(
    *, data: pl.DataFrame, title: list[str], span: tuple[float, float] | None
) -> alt.LayerChart:
    """Draw one forecast window: median, 50% and 90% bands of three arms, and the actual."""
    base = alt.Chart(data.to_pandas())
    colour = alt.Color(
        "arm:N", scale=arm_scale(), legend=alt.Legend(title=None, orient="top", labelLimit=400)
    )
    band90 = base.mark_area(opacity=0.10).encode(x="hour:Q", y="q05:Q", y2="q95:Q", color=colour)  # ty: ignore[unresolved-attribute]
    band50 = base.mark_area(opacity=0.20).encode(  # ty: ignore[unresolved-attribute]
        x="hour:Q", y="q25:Q", y2="q75:Q", color=alt.Color("arm:N", scale=arm_scale(), legend=None)
    )  # ty: ignore[unresolved-attribute]
    median = base.mark_line().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("hour:Q", title="Hours after the forecast origin"),
        y=alt.Y("q50:Q", title="Demand (% of effective capacity)", scale=alt.Scale(zero=False)),
        color=alt.Color("arm:N", scale=arm_scale(), legend=None),
    )
    actual = (
        alt.Chart(data.filter(pl.col("arm") == ARM_LABELS["a"]).to_pandas())
        .mark_line(color="black", strokeWidth=1.5)
        .encode(x="hour:Q", y="actual:Q")  # ty: ignore[unresolved-attribute]
    )
    layers = [band90, band50, median, actual]
    if span is not None:
        rect = (
            alt.Chart(pl.DataFrame({"a": [span[0]], "b": [span[1]]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rect(opacity=0.10, color="grey")
            .encode(x="a:Q", x2="b:Q")
        )  # ty: ignore[unresolved-attribute]
        layers.insert(0, rect)
    return alt.layer(*layers).properties(  # ty: ignore[invalid-return-type]
        width=WIDTH - 60, height=190, title=alt.TitleParams(title, anchor="start", fontSize=11)
    )


def main() -> None:
    """Draw every figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = contrasts()
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=200, float_precision=4):
        print(frame)
    contrast_figure(frame=frame)
    shift_figure()
    firing_figure()
    series = adapter.series_table(labels=adapter.PRIMARY_LABELS)
    actuals, _, _ = s392.load_actuals(source="observed", series=series)
    _, events = s392.event_tables(series=series)
    origins = common.all_origins()

    def nearest(month: tuple[int, int, int]) -> datetime:
        return min(
            origins, key=lambda o: abs((o - datetime(*month, 12, tzinfo=UTC)).total_seconds())
        )

    number = 4
    for name, label, month, days in (
        ("day_ahead_winter", "D5", (2026, 1, 15), 1),
        ("fourteen_day_summer", "D9", (2026, 6, 15), 14),
    ):
        origin = nearest(month)
        data = window_from(
            label=label,
            origin=origin,
            days=days,
            arms=stored_arms(label=label, origin=origin, series=series),
            actuals=actuals,
            series=series,
        )
        series_chart(
            data=data,
            title=[
                f"{label}, forecast issued {origin:%B %Y}: median, 50% and 90% bands, and the",
                f"actual (black); {days} day{'s' if days > 1 else ''} ahead.",
            ],
            span=None,
        ).save(FIGURE_DIR / f"figure_{number}_{name}.svg")
        number += 1
    origin = nearest((2026, 1, 15))
    for kind, text in (
        ("dropout", "a 6-hour dropout to zero"),
        ("spike", "a 4-reading spike of 15 MAD"),
        ("stuck", "a 24-hour stuck reading"),
    ):
        data = window_from(
            label="D14",
            origin=origin,
            days=14,
            arms=injected_arms(label="D14", origin=origin, kind=kind, series=series),
            actuals=actuals,
            series=series,
        )
        series_chart(
            data=data,
            title=[
                f"D14, forecast issued {origin:%B %Y}, with {text} injected into the readings",
                "that end 3.5 hours before the origin: median, 50% and 90% bands, and the actual (black).",
            ],
            span=None,
        ).save(FIGURE_DIR / f"figure_{number}_injected_{kind}.svg")
        number += 1
    six_events = fig465.pick_events(events=events, origins=origins)
    for event, event_origin in six_events:
        label = event["label"]
        if label not in fit.SIGNED_MW_LABELS:
            continue
        data = window_from(
            label=label,
            origin=event_origin,
            days=14,
            arms=stored_arms(label=label, origin=event_origin, series=series),
            actuals=actuals,
            series=series,
        )
        span = (
            (event["start"] - event_origin).total_seconds() / 3600.0,
            (event["end"] - event_origin).total_seconds() / 3600.0,
        )
        series_chart(
            data=data,
            title=[
                f"{label} (metered in signed MW), forecast issued {event_origin:%B %Y}: median, 50% and",
                "90% bands and the actual (black); the labelled event is shaded grey.",
            ],
            span=span,
        ).save(FIGURE_DIR / f"figure_{number}_event_{label}.svg")
        number += 1


if __name__ == "__main__":
    main()
