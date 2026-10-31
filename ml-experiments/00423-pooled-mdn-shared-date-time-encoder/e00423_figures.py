# ruff: noqa: E501
"""Draw the figures of experiment 423 from its stored seed forecasts, and print every number shown.

The script imports `e00423_score` for the loaders and the scoring code, so the figures use the
same arms (seeds averaged in quantile space, widened by 1.15) as the scorer's tables. Figures:

1. CRPS at days 1 to 10 per arm, for the signed-MW six and the MVA-only eight, with a 95% interval
   from resampling whole months (the five screen folds), and the reference R as a rule.
2. CRPS minus R's per arm and fold.
3. CRPS by lead day (1 to 14) per arm at the signed-MW six, and the same minus R's.
4. CRPS on bank-holiday-week rows per arm and fold, with R plus and minus the minimum detectable
   effect (six-seed means) as a band.
5. Example forecasts of R, E1, and E2 at one signed-MW Primary and one MVA-only Primary, from the
   origin on 2025-12-24 (the 2025-12 fold, around Christmas): the first 36 hours, and the 14 days.
   Primaries are substations, so no anonymisation is needed.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00407-pooled-mixture-density-network-tail-arms-and-recency-weights:\
ml-experiments/00423-pooled-mdn-shared-date-time-encoder \
OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00423-*/e00423_figures.py
```
"""

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00423_features as feats  # ty: ignore[unresolved-import]
import e00423_score as s423  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00423-pooled-mdn-shared-date-time-encoder"
)
WIDTH: Final[int] = 600
ARMS: Final[tuple[str, ...]] = ("R", "E0", "E1n", "E1", "E2")
ARM_NAMES: Final[dict[str, str]] = {
    "R": "R: hand-made calendar columns",
    "E0": "E0: half-hour and weekday only",
    "E1n": "E1n: E0 + bank-holiday inputs",
    "E1": "E1: E1n + day of month, year, holiday name",
    "E2": "E2: E0 + day of month and day of year",
}
ARM_ORDER: Final[list[str]] = [ARM_NAMES[a] for a in ARMS]
ARM_COLOURS: Final[dict[str, str]] = {
    "R": "#1A1A1A",
    "E0": theme.DATA_PURPLE,
    "E1n": theme.DATA_SKY,
    "E1": theme.DATA_BLUE,
    "E2": theme.BRAND_ORANGE,
}
ARM_DASHES: Final[dict[str, list[int]]] = {
    "R": [1, 0],
    "E0": [2, 2],
    "E1n": [6, 3],
    "E1": [1, 0],
    "E2": [6, 3],
}
COLOUR_SCALE: Final[alt.Scale] = alt.Scale(domain=ARM_ORDER, range=[ARM_COLOURS[a] for a in ARMS])
GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "Signed-MW six": s423.SIGNED,
    "MVA-only eight": s407.SUBSETS["MVA-only eight"],
}
N_BOOT: Final[int] = 20_000
LEVEL: Final[tuple[float, float]] = (2.5, 97.5)
EXAMPLE_ORIGIN: Final[datetime] = datetime(2025, 12, 24, 9, 0, tzinfo=UTC)
EXAMPLE_PRIMARIES: Final[tuple[tuple[str, str], ...]] = (
    ("D5", "signed-MW"),
    ("D2", "MVA-only"),
)
EXAMPLE_ARMS: Final[tuple[str, ...]] = ("R", "E1", "E2")
SCORED_ARMS: Final[tuple[str, ...]] = ("R", "R_s012", "R_s345", "E0", "E1n", "E1", "E2")


def title(*, text: str, subtitle: list[str]) -> alt.TitleParams:
    """Return a left-aligned title with subtitle lines."""
    return alt.TitleParams(
        text, subtitle=subtitle, anchor="start", fontSize=14, subtitleFontSize=11, offset=8
    )


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG, then optimise it with svgo."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / name
    chart.save(path)
    subprocess.run(
        ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(path)],
        check=True,
        capture_output=True,
    )
    print(f"wrote {path} ({path.stat().st_size / 1000:.0f} kB)")


def fold_expr() -> pl.Expr:
    """Return the fold (month of forecast origin) of a row."""
    return pl.col("power_fcst_init_time").dt.strftime("%Y-%m").alias("fold")


def label_fold_sums(
    *, rows: pl.DataFrame, members: tuple[str, ...], where: pl.Expr | None
) -> pl.DataFrame:
    """Return, per Primary and fold, the sum and the count of the headline-band CRPS."""
    keep = pl.col("label").is_in(list(members)) & pl.col(s423.HEADLINE)
    if where is not None:
        keep &= where
    return (
        rows.filter(keep)
        .group_by("label", fold_expr())
        .agg(total=pl.col("crps").sum(), n=pl.len())
        .sort("label", "fold")
    )


def pooled_with_interval(
    *, sums: pl.DataFrame, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return the equal-weight-per-Primary mean CRPS and its interval from resampling folds."""
    folds = sorted(sums["fold"].unique().to_list())
    labels = sorted(sums["label"].unique().to_list())
    total = np.zeros((len(labels), len(folds)))
    count = np.zeros((len(labels), len(folds)))
    for label, fold, t, n in sums.select("label", "fold", "total", "n").iter_rows():
        total[labels.index(label), folds.index(fold)] = t
        count[labels.index(label), folds.index(fold)] = n
    point = float(np.mean(total.sum(axis=1) / count.sum(axis=1)))
    draws = rng.integers(0, len(folds), size=(N_BOOT, len(folds)))
    multiplicity = np.stack([np.bincount(d, minlength=len(folds)) for d in draws])
    num = multiplicity @ total.T
    den = multiplicity @ count.T
    valid = (den > 0).all(axis=1)
    estimates = np.mean(num[valid] / den[valid], axis=1)
    low, high = np.percentile(estimates, LEVEL)
    return point, float(low), float(high)


def figure_headline(*, rows: dict[str, pl.DataFrame]) -> None:
    """Draw figure 1: CRPS at days 1 to 10 per arm and group, with month-block intervals."""
    rng = np.random.default_rng(423)
    records = []
    for group, members in GROUPS.items():
        for arm in ARMS:
            sums = label_fold_sums(rows=rows[f"{arm}_w115"], members=members, where=None)
            point, low, high = pooled_with_interval(sums=sums, rng=rng)
            records.append(
                {"group": group, "arm": ARM_NAMES[arm], "crps": point, "low": low, "high": high}
            )
    data = pl.DataFrame(records)
    print("\n== figure 1: CRPS at days 1 to 10 (points of capacity), 95% interval over months ==")
    with pl.Config(tbl_rows=40, tbl_cols=10):
        print(data.with_columns(pl.col("crps", "low", "high").round(3)))
    panels = []
    for group in GROUPS:
        part = data.filter(pl.col("group") == group)
        reference = float(part.filter(pl.col("arm") == ARM_NAMES["R"])["crps"][0])
        low = min(part["low"].to_list()) - 0.02
        high = max(part["high"].to_list()) + 0.02
        x = alt.X(
            "crps:Q",
            title="CRPS, days 1 to 10 (points of capacity; smaller is better)",
            scale=alt.Scale(domain=[low, high], nice=False),
        )
        y = alt.Y("arm:N", sort=ARM_ORDER, title=None, axis=alt.Axis(labelLimit=320))
        colour = alt.Color("arm:N", scale=COLOUR_SCALE, legend=None)
        line = (
            alt.Chart(part)
            .mark_rule(strokeWidth=2, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                y=y,
                x=alt.X("low:Q", scale=alt.Scale(domain=[low, high], nice=False)),
                x2="high:Q",
                color=colour,
            )
        )
        dot = (
            alt.Chart(part)
            .mark_point(filled=True, size=90, aria=False)
            .encode(y=y, x=x, color=colour)  # ty: ignore[unresolved-attribute]
        )
        rule = (
            alt.Chart(pl.DataFrame({"x": [reference]}))
            .mark_rule(strokeDash=[4, 3], color=theme.GREY_1, aria=False)
            .encode(x=alt.X("x:Q", scale=alt.Scale(domain=[low, high], nice=False)))  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            (rule + line + dot).properties(
                width=WIDTH,
                height=150,
                title=alt.TitleParams(group, anchor="start", fontSize=12, subtitle=[], offset=2),
            )
        )
    chart = (
        alt.vconcat(*panels, spacing=20)
        .properties(
            title=title(
                text="No calendar encoder beats the hand-made columns; E1 and E2 are worse",
                subtitle=[
                    "CRPS at forecast days 1 to 10, equal weight per Primary, over the five screen months",
                    "(2025-10, 2025-12, 2026-01, 2026-03, 2026-06). Dot: estimate. Line: 95% interval from",
                    "resampling whole months. Dashed rule: R, the hand-made columns (six seeds); E0 has three.",
                ],
            )
        )
        .configure_view(stroke=None)
    )
    save(chart=chart, name="figure_1_crps_days_1_to_10.svg")


def fold_values(*, rows: dict[str, pl.DataFrame], where: pl.Expr | None) -> pl.DataFrame:
    """Return CRPS per arm, group, and fold: equal weight per Primary over the fold's rows."""
    parts = []
    for arm in ARMS:
        for group, members in GROUPS.items():
            sums = label_fold_sums(rows=rows[f"{arm}_w115"], members=members, where=where)
            parts.append(
                sums.with_columns(mean=pl.col("total") / pl.col("n"))
                .group_by("fold")
                .agg(crps=pl.col("mean").mean(), n_half_hours=pl.col("n").sum())
                .with_columns(arm=pl.lit(arm), group=pl.lit(group))
            )
    return pl.concat(parts).sort("group", "arm", "fold")


def figure_by_fold(*, rows: dict[str, pl.DataFrame]) -> None:
    """Draw figure 2: CRPS minus R's per arm and fold."""
    values = fold_values(rows=rows, where=None)
    reference = values.filter(pl.col("arm") == "R").select(
        "group", "fold", reference=pl.col("crps")
    )
    data = (
        values.filter(pl.col("arm") != "R")
        .join(reference, on=["group", "fold"])
        .with_columns(
            difference=pl.col("crps") - pl.col("reference"),
            arm_name=pl.col("arm").replace_strict(ARM_NAMES),
        )
        .sort("group", "fold", "arm")
    )
    print("\n== figure 2: CRPS minus R's at days 1 to 10, per fold (points of capacity) ==")
    with pl.Config(tbl_rows=100, tbl_cols=12):
        print(
            data.select("group", "fold", "arm", "crps", "reference", "difference").with_columns(
                pl.col("crps", "reference", "difference").round(3)
            )
        )
    bound = float(data["difference"].abs().max()) * 1.1  # ty: ignore[invalid-argument-type]
    panels = []
    arm_order = [ARM_NAMES[a] for a in ARMS if a != "R"]
    for group in GROUPS:
        part = data.filter(pl.col("group") == group)
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(strokeDash=[4, 3], color=theme.GREY_1, aria=False)
            .encode(x=alt.X("x:Q", scale=alt.Scale(domain=[-bound, bound])))  # ty: ignore[unresolved-attribute]
        )
        dots = (
            alt.Chart(part)
            .mark_point(filled=True, size=70, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("fold:N", title=None, axis=alt.Axis(labelAngle=0)),
                yOffset=alt.YOffset("arm_name:N", sort=arm_order),
                x=alt.X(
                    "difference:Q",
                    title="CRPS minus R's (points of capacity; more negative is better)",
                    scale=alt.Scale(domain=[-bound, bound]),
                ),
                color=alt.Color(
                    "arm_name:N",
                    scale=alt.Scale(
                        domain=arm_order, range=[ARM_COLOURS[a] for a in ARMS if a != "R"]
                    ),
                    legend=alt.Legend(orient="top", title=None, columns=2, labelLimit=400),
                ),
            )
        )
        panels.append(
            (zero + dots).properties(
                width=WIDTH,
                height=170,
                title=alt.TitleParams(group, anchor="start", fontSize=12, offset=2),
            )
        )
    chart = (
        alt.vconcat(*panels, spacing=20)
        .resolve_scale(color="shared")
        .properties(
            title=title(
                text="E1 and E2 trail R in every month; E0 and E1n sit close to R",
                subtitle=[
                    "CRPS at days 1 to 10 minus R's, per screen month (fold), equal weight per Primary.",
                    "Dot: one arm in one month. Dashed rule: same as R. E0 is a mean of three seeds, the rest six.",
                ],
            )
        )
        .configure_view(stroke=None)
    )
    save(chart=chart, name="figure_2_crps_difference_by_fold.svg")


def figure_by_lead_day(*, rows: dict[str, pl.DataFrame]) -> None:
    """Draw figure 3: CRPS by lead day per arm at the signed-MW six."""
    records = []
    for arm in ARMS:
        frame = rows[f"{arm}_w115"]
        by_day = (
            frame.filter(pl.col("label").is_in(list(s423.SIGNED)))
            .with_columns(lead_day=(pl.col("lead_hours") // 24).cast(pl.Int32) + 1)
            .filter(pl.col("lead_day") <= 14)
            .group_by("label", "lead_day")
            .agg(pl.col("crps").mean())
            .group_by("lead_day")
            .agg(crps=pl.col("crps").mean(), n_primaries=pl.len())
            .with_columns(arm=pl.lit(arm))
        )
        records.append(by_day)
    data = pl.concat(records).sort("arm", "lead_day")
    reference = data.filter(pl.col("arm") == "R").select("lead_day", reference=pl.col("crps"))
    data = (
        data.join(reference, on="lead_day")
        .with_columns(
            difference=pl.col("crps") - pl.col("reference"),
            arm_name=pl.col("arm").replace_strict(ARM_NAMES),
        )
        .sort("arm", "lead_day")
    )
    print("\n== figure 3: CRPS by lead day at the signed-MW six (points of capacity) ==")
    table = data.pivot(on="arm", index="lead_day", values="crps").sort("lead_day")
    with pl.Config(tbl_rows=40, tbl_cols=12):
        print(table.with_columns(pl.exclude("lead_day").round(3)))
        print("-- minus R --")
        print(
            data.pivot(on="arm", index="lead_day", values="difference")
            .sort("lead_day")
            .drop("R")
            .with_columns(pl.exclude("lead_day").round(3))
        )
    panels = []
    for column, text, kind in (
        ("crps", "CRPS (points of capacity; smaller is better)", "absolute"),
        ("difference", "CRPS minus R's (points of capacity; more negative is better)", "minus R"),
    ):
        part = data if kind == "absolute" else data.filter(pl.col("arm") != "R")
        base = alt.Chart(part).encode(
            x=alt.X(
                "lead_day:O",
                title="Forecast lead (days; day 1 is the first 24 hours)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(f"{column}:Q", title=text, scale=alt.Scale(zero=kind == "minus R")),
            color=alt.Color(
                "arm_name:N",
                scale=COLOUR_SCALE,
                legend=alt.Legend(orient="top", title=None, columns=2, labelLimit=400),
            ),
        )
        dash = alt.StrokeDash(
            "arm_name:N",
            scale=alt.Scale(domain=ARM_ORDER, range=[ARM_DASHES[a] for a in ARMS]),
            legend=None,
        )
        layers = alt.layer(
            base.mark_line(aria=False).encode(strokeDash=dash),  # ty: ignore[unresolved-attribute, invalid-argument-type]
            base.mark_point(filled=True, size=35, aria=False),  # ty: ignore[invalid-argument-type]
        )
        if kind == "minus R":
            zero = (
                alt.Chart(pl.DataFrame({"y": [0.0]}))
                .mark_rule(color=theme.GREY_1, aria=False)
                .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
            )
            layers = zero + layers
        panels.append(layers.properties(width=WIDTH, height=190))
    chart = (
        alt.vconcat(*panels, spacing=20)
        .resolve_scale(color="shared")
        .properties(
            title=title(
                text="At the signed-MW six, the encoder arms fall behind R as the lead grows",
                subtitle=[
                    "CRPS by lead day at the signed-MW six, equal weight per Primary, all five screen months.",
                    "Top: each arm. Bottom: each arm minus R (grey rule: same as R). E0 is three seeds, the rest six.",
                ],
            )
        )
        .configure_view(stroke=None)
    )
    save(chart=chart, name="figure_3_crps_by_lead_day.svg")


def figure_bank_holiday(
    *,
    rows: dict[str, pl.DataFrame],
    mde: dict[tuple[str, str], tuple[float, float]],
) -> None:
    """Draw figure 4: bank-holiday-week CRPS per arm and fold, with R plus or minus the MDE."""
    values = fold_values(rows=rows, where=pl.col("bank_holiday_week"))
    reference = values.filter(pl.col("arm") == "R").select(
        "group", "fold", reference=pl.col("crps")
    )
    data = values.join(reference, on=["group", "fold"]).with_columns(
        arm_name=pl.col("arm").replace_strict(ARM_NAMES)
    )
    group_key = {"Signed-MW six": "signed-MW six", "MVA-only eight": "MVA-only eight"}
    bands = []
    for group in GROUPS:
        figure = mde.get((group_key[group], "bank-holiday weeks"))
        if figure is None:
            print(f"no minimum detectable effect for {group}: band not drawn")
            continue
        for fold, ref in (
            data.filter(pl.col("arm") == "R", pl.col("group") == group)
            .select("fold", "reference")
            .iter_rows()
        ):
            bands.append(
                {
                    "group": group,
                    "fold": fold,
                    "low": ref - figure[1],
                    "high": ref + figure[1],
                    "mde": figure[1],
                }
            )
    band_frame = pl.DataFrame(bands)
    print("\n== figure 4: CRPS on bank-holiday-week rows, days 1 to 10 (points of capacity) ==")
    with pl.Config(tbl_rows=100, tbl_cols=12):
        print(
            data.select("group", "fold", "arm", "crps", "n_half_hours").with_columns(
                pl.col("crps").round(3)
            )
        )
        print("band: R plus or minus the six-seed minimum detectable effect")
        print(band_frame.with_columns(pl.col("low", "high", "mde").round(3)))
    panels = []
    for group in GROUPS:
        part = data.filter(pl.col("group") == group)
        band = band_frame.filter(pl.col("group") == group)
        low = min(float(part["crps"].min()), float(band["low"].min())) - 0.05  # ty: ignore[invalid-argument-type]
        high = max(float(part["crps"].max()), float(band["high"].max())) + 0.05  # ty: ignore[invalid-argument-type]
        domain = alt.Scale(domain=[low, high], nice=False)
        y = alt.Y("fold:N", title=None, axis=alt.Axis(labelAngle=0))
        shade = (
            alt.Chart(band)
            .mark_bar(color=theme.GREY_3, opacity=0.5, aria=False)
            .encode(y=y, x=alt.X("low:Q", scale=domain), x2="high:Q")  # ty: ignore[unresolved-attribute]
        )
        dots = (
            alt.Chart(part)
            .mark_point(filled=True, size=70, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                y=y,
                yOffset=alt.YOffset("arm_name:N", sort=ARM_ORDER),
                x=alt.X(
                    "crps:Q",
                    title="CRPS on bank-holiday-week rows, days 1 to 10 (points of capacity; smaller is better)",
                    scale=domain,
                ),
                color=alt.Color(
                    "arm_name:N",
                    scale=COLOUR_SCALE,
                    legend=alt.Legend(orient="top", title=None, columns=2, labelLimit=400),
                ),
            )
        )
        panels.append(
            (shade + dots).properties(
                width=WIDTH,
                height=140,
                title=alt.TitleParams(group, anchor="start", fontSize=12, offset=2),
            )
        )
    chart = (
        alt.vconcat(*panels, spacing=20)
        .resolve_scale(color="shared")
        .properties(
            title=title(
                text="On bank-holiday weeks the noise band is wide and no arm clearly beats R",
                subtitle=[
                    "CRPS at days 1 to 10 on rows from bank-holiday weeks, per screen month that has any,",
                    "equal weight per Primary. Dot: one arm. Grey band: R plus or minus the minimum detectable",
                    "effect, from the gap between R's two three-seed halves (a lower bound, on few months).",
                    "E0 is a mean of three seeds, so its own band would be wider.",
                ],
            )
        )
        .configure_view(stroke=None)
    )
    save(chart=chart, name="figure_4_bank_holiday_weeks.svg")


def figure_examples(
    *, frames: dict[str, pl.DataFrame], series: pl.DataFrame, actuals: pl.DataFrame
) -> None:
    """Draw figure 5: the first 36 hours and the 14 days of one forecast, R beside E1 and E2."""
    label_of = dict(zip(series["label"], series["time_series_id"], strict=True))
    for view, horizon, name in (
        ("first 36 hours", 36.0, "figure_5a_day_ahead.svg"),
        ("14 days", 14 * 24.0, "figure_5b_14_day.svg"),
    ):
        panels = []
        print(f"\n== figure 5, {view}: forecasts from {EXAMPLE_ORIGIN:%Y-%m-%d %H:%M} UTC ==")
        for primary, kind in EXAMPLE_PRIMARIES:
            sid = label_of[primary]
            cut = {}
            for arm in EXAMPLE_ARMS:
                frame = (
                    frames[f"{arm}_w115"]
                    .filter(
                        pl.col("time_series_id") == sid,
                        pl.col("power_fcst_init_time") == EXAMPLE_ORIGIN,
                    )
                    .join(
                        actuals.select("time_series_id", "valid_time", "actual"),
                        on=["time_series_id", "valid_time"],
                        how="left",
                    )
                    .with_columns(
                        hours=(
                            pl.col("valid_time") - pl.col("power_fcst_init_time")
                        ).dt.total_minutes()
                        / 60.0
                    )
                    .filter(pl.col("hours") <= horizon)
                    .sort("valid_time")
                )
                cut[arm] = frame
            low = min(
                float(f.select(pl.min_horizontal("q05", "actual")).min().item())
                for f in cut.values()
            )
            high = max(
                float(f.select(pl.max_horizontal("q95", "actual")).max().item())
                for f in cut.values()
            )
            pad = 0.05 * (high - low)
            for arm, frame in cut.items():
                inside = frame.filter(pl.col("actual").is_not_null())
                stats = inside.select(
                    n=pl.len(),
                    median_abs_error_mw=(pl.col("actual") - pl.col("q50")).abs().mean(),
                    share_inside_90=(
                        (pl.col("actual") >= pl.col("q05")) & (pl.col("actual") <= pl.col("q95"))
                    ).mean(),
                    share_inside_50=(
                        (pl.col("actual") >= pl.col("q25")) & (pl.col("actual") <= pl.col("q75"))
                    ).mean(),
                )
                print(
                    f"{primary} ({kind}), {arm}: {frame.height} half-hours, "
                    f"{stats['n'][0]} with actuals, MAE of median {stats['median_abs_error_mw'][0]:.3f} MW, "
                    f"actuals inside 50% band {stats['share_inside_50'][0]:.2f}, inside 90% band "
                    f"{stats['share_inside_90'][0]:.2f}; y range {low:.2f} to {high:.2f} MW"
                )
                base = alt.Chart(frame.to_pandas()).encode(
                    x=alt.X(
                        "valid_time:T",
                        title="Valid time (UTC)" if arm == EXAMPLE_ARMS[-1] else None,
                        axis=alt.Axis(
                            format="%d %b" if horizon > 48 else "%d %b %H:%M",
                            labels=arm == EXAMPLE_ARMS[-1],
                        ),
                    )
                )
                scale = alt.Scale(domain=[low - pad, high + pad], nice=False)
                band90 = base.mark_area(opacity=0.2, color=theme.DATA_BLUE, aria=False).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("q05:Q", title="MW", scale=scale), y2="q95:Q"
                )
                band50 = base.mark_area(opacity=0.35, color=theme.DATA_BLUE, aria=False).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("q25:Q", scale=scale), y2="q75:Q"
                )
                median = base.mark_line(color=theme.DATA_BLUE, aria=False).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("q50:Q", scale=scale)
                )
                actual = base.mark_line(color=theme.BRAND_ORANGE, aria=False).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("actual:Q", scale=scale)
                )
                panels.append(
                    (band90 + band50 + median + actual).properties(
                        width=WIDTH,
                        height=95,
                        title=alt.TitleParams(
                            f"{primary} ({kind} Primary), {ARM_NAMES[arm]}",
                            anchor="start",
                            fontSize=11,
                            offset=2,
                        ),
                    )
                )
        chart = (
            alt.vconcat(*panels, spacing=12)
            .properties(
                title=title(
                    text=f"Forecasts around Christmas, {view}: R beside E1 and E2",
                    subtitle=[
                        f"Forecast made on {EXAMPLE_ORIGIN:%d %B %Y} at 09:00 UTC (the 2025-12 fold; seed mean, widened 1.15).",
                        "Blue line: median. Dark band: 50% interval. Light band: 90% interval. Orange line: actual.",
                    ],
                )
            )
            .configure_view(stroke=None)
        )
        save(chart=chart, name=name)


def main() -> None:
    """Score the arms as the scorer does, then draw every figure."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    flagged = pl.read_parquet(s407.FLAGGED_KEYS).select("time_series_id", "valid_time")
    ctx["actuals"] = ctx["actuals"].join(flagged, on=["time_series_id", "valid_time"], how="anti")
    arm_defs = tuple(a for a in s423.ARM_DEFS if a.name in SCORED_ARMS)
    frames = s423.load_frames(series=series, arm_defs=arm_defs, raw_arms=())
    missing = [a for a in ARMS if f"{a}_w115" not in frames]
    if missing:
        msg = f"arms {missing} are not complete: run the figures when every stored arm exists"
        raise ValueError(msg)
    s423.check_keys(frames=frames, reference="R_w115")
    _, _, scored = s381.score_arms(arms=frames, series=series, ctx=ctx)
    rows = {n: feats.add_row_sets(frame=r) for n, r in scored.items()}
    mde = s423.minimum_detectable_effect(rows=rows)
    figure_headline(rows=rows)
    figure_by_fold(rows=rows)
    figure_by_lead_day(rows=rows)
    figure_bank_holiday(rows=rows, mde=mde)
    figure_examples(frames=frames, series=series, actuals=ctx["actuals"])


if __name__ == "__main__":
    main()
