r"""Score experiment 707's arms against the references and print every number the page quotes.

The scorer follows experiment 425's `e00425_score.py`. It reads the per-seed raw quantiles that
`e00707_fit.py` wrote (and, for `A_24` and `R_24`, experiment 425's stored arms `A` and `R`),
averages the seeds that both arms of a contrast hold in quantile space, widens each average by 1.15
about the median, and scores it. The references are scored on the half-hours that every arm shares:
`S0M_w115` (experiment 389's stored monthly XGBoost forecast, widened here), and `bg2`,
`manual_heuristic`, and `climatology` (experiment 87's stored forecasts, as stored).

Printed in this order: the arms scored, the rows dropped per fold, **the minimum detectable effect
(before any contrast)**, CRPS beside the normalised MAE of the median per group, per Primary, and
per lead band, the slices of the protocol (cold winter nights, sunny half-hours, the Economy 7 step,
the load-shift event slices, and cold-start months), the tails, the avoidable curtailment cost, the
eight contrasts at the Bonferroni level, the verdict of the decision rule for C1, the descriptive
readings, and the checks. Every contrast is read per group at 1 - 0.05/16 = 99.69%, with the
month-block bootstrap of `metrics.month_bootstrap_contrast` and 50,000 resamples. The group is the
signed-MW six or the MVA-only eight; D15 and D18 are report-only.

The two figures (`figure_gain_against_ens_months` and `figure_example_forecasts`) are written as
SVG under the experiment's docs folder, and audit triggers 2 and 6 are decided from a descriptive
line (stored 425 `A` minus `R`) on the same rows, level, and resamples as C1. Trigger 7 prints a
line saying that the page's author must judge it.

```bash
# PYTHONPATH: this folder and the folders of experiments 377, 381, 397, 407 (their scorers' helpers)
uv run --no-sync python ml-experiments/00707-*/e00707_score.py            # all 12 folds
uv run --no-sync python ml-experiments/00707-*/e00707_score.py --screen   # the 4 screen folds
```
"""

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as s377  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00397_score as s397  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00707_arms as arms  # ty: ignore[unresolved-import]
import e00707_common as common  # ty: ignore[unresolved-import]
import e00707_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme
from scipy import stats
from studies.charts import CONTENT_WIDTH_PX, wrapped
from studies.eval_protocol import metrics

HEADLINE: Final[str] = s407.HEADLINE
SUBSETS: Final[dict[str, tuple[str, ...]]] = s407.SUBSETS
SIGNED: Final[tuple[str, ...]] = s407.SIGNED
OUTER_LEVELS: Final[tuple[str, ...]] = s407.OUTER_LEVELS
KEYS: Final[list[str]] = list(common.KEYS)
SIX: Final[str] = "signed-MW six"
EIGHT: Final[str] = "MVA-only eight"
REPORT_ONLY_GROUP: Final[str] = "D15 and D18"
TESTED_GROUPS: Final[tuple[str, ...]] = (SIX, EIGHT)
N_READINGS: Final[int] = 16
"""Eight contrasts in two groups."""
BONFERRONI_LEVEL_PERCENT: Final[float] = 100.0 - 5.0 / N_READINGS
N_RESAMPLES: Final[int] = 50_000
GAIN_THRESHOLD: Final[float] = -0.02
"""Points of CRPS: the protocol's minimum for a new fitted step, a new input, or a new source."""
CURTAILMENT_FRACTIONS: Final[tuple[float, ...]] = (0.9, 0.75, 0.5)
LEAD_BIN_HOURS: Final[int] = 6
EARLY_LEAD_HOURS: Final[float] = 144.0
"""Days 1 to 6 end here; days 7 to 10 end at 240 hours."""
DROPPED_FOLD: Final[str] = "2026-01"
"""The fold that dominated experiments 449 and 487; descriptive readings drop it."""
OUT_DIR: Path = arms.WORK_DIR / "work"
FIGURE_TABLE_DIR: Path = OUT_DIR / "figure_tables"
REFERENCE_NAMES: Final[tuple[str, ...]] = ("bg2", "manual_heuristic", "climatology")
SLICES: Final[tuple[tuple[str, str | None], ...]] = (
    ("days 1 to 10", None),
    ("cold winter nights", "cold_nights"),
    ("winter nights", "winter_nights"),
    ("sunny half-hours", "sunny_unit_shape"),
    ("Economy 7 step", "economy_7_step"),
    ("load-shift event rows", "event_rows"),
    ("7 days after a load-shift event", "event_following_7_days"),
    ("cold-start months", "cold_start_months"),
)
SCALE_METRICS: Final[tuple[str, ...]] = ("crps", "mae_median")
"""CRPS and the normalised MAE of the median, both in points of effective capacity."""
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1 A_F minus R_F": ("A_F", "R_F"),
    "C2 A_E24 minus R_E24": ("A_E24", "R_E24"),
    "C3 R_F minus A_24": ("R_F", "A_24"),
    "C4 A_F minus A_E24": ("A_F", "A_E24"),
    "C5 A_24 minus A1x_24": ("A_24", "A1x_24"),
    "C6 A1x_24 minus R_24": ("A1x_24", "R_24"),
    "C7 R_C minus R_F": ("R_C", "R_F"),
    "C8 R_F+flags minus R_F": ("R_F+flags", "R_F"),
}
"""The eight contrasts: label, (treatment, reference)."""
MDE_HALVES: Final[tuple[tuple[int, ...], tuple[int, ...]]] = ((0, 1, 2), (3, 4))
"""The two seed sets whose scores the minimum detectable effect compares."""
MDE_ARM: Final[str] = "R_F"
"""The arm whose seed halves give the minimum detectable effect; `R_24` (stored `R`) otherwise."""


DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / arms.EXPERIMENT_DIR_NAME
)
"""The experiment's docs folder, where the two figures are written."""
TRIGGER_6_FACTOR: Final[float] = 2.0
"""Audit trigger 6 fires if C1's interval is wider than this multiple of the stored line's, or
narrower than its reciprocal."""
DAYS_PER_MONTH: Final[float] = 30.4375
LEAD_BAND_DAY_HOURS: Final[int] = 24
FIGURE_LEVEL_TEXT: Final[str] = f"{BONFERRONI_LEVEL_PERCENT:.2f}%"
EXAMPLE_ORIGIN: Final[datetime] = datetime(2026, 1, 14, 9, tzinfo=UTC)
"""The origin of the example forecasts: the 00Z run of 2026-01-14, issued 9 hours earlier."""
EXAMPLE_PRIMARIES: Final[dict[str, str]] = {SIX: "D5", EIGHT: "D2"}
"""The Primary drawn per group. Primaries are substations, so naming them is allowed."""
EXAMPLE_ARMS: Final[tuple[str, str]] = ("A_F", "R_F")
SVGO_COMMAND: Final[tuple[str, ...]] = ("npx", "svgo@4", "--multipass", "--precision=1")


@dataclass(frozen=True)
class GainWindow:
    """One x position of the figure of gain against ENS months.

    Attributes:
        label: The legend text.
        treatment: The pre-trained arm.
        reference: The ENS-only arm.
        window: The ENS window that both arms share.
        colour: A main data colour of the OCF palette.
    """

    label: str
    treatment: str
    reference: str
    window: arms.Window
    colour: str


GAIN_WINDOWS: Final[tuple[GainWindow, ...]] = (
    GainWindow(
        label="12 months (A_12 minus R_12, descriptive)",
        treatment="A_12",
        reference="R_12",
        window=arms.W12,
        colour=ocf_theme.DATA_SKY,
    ),
    GainWindow(
        label="24 months (C2, A_E24 minus R_E24)",
        treatment="A_E24",
        reference="R_E24",
        window=arms.W24,
        colour=ocf_theme.DATA_BLUE,
    ),
    GainWindow(
        label="Window F (C1, A_F minus R_F)",
        treatment="A_F",
        reference="R_F",
        window=arms.W_F,
        colour=ocf_theme.BRAND_ORANGE,
    ),
    GainWindow(
        label="Window C (A_C minus R_C, descriptive)",
        treatment="A_C",
        reference="R_C",
        window=arms.W_C,
        colour=ocf_theme.DATA_PURPLE,
    ),
)
"""The four windows of the figure of gain against ENS months."""


@dataclass(frozen=True)
class Variant:
    """One scored forecast: an arm averaged over a set of seeds."""

    arm: str
    seeds: tuple[int, ...]

    @property
    def name(self) -> str:
        """Return the arm's name, with the seeds appended when they are not the arm's own."""
        full = arms.ARMS[self.arm].seeds
        if self.seeds == full:
            return self.arm
        return f"{self.arm}@{''.join(str(s) for s in self.seeds)}"


def arm_folder(*, arm: str) -> Path:
    """Return the folder of an arm's per-seed forecasts: experiment 425's for a stored arm."""
    spec = arms.ARMS[arm]
    if spec.stored_folder is not None:
        return arms.STORED_425_DIR / "seed_preds" / spec.stored_folder
    return OUT_DIR / "seed_preds" / spec.folder


def available_seeds(*, arm: str, months: tuple[str, ...]) -> tuple[int, ...]:
    """Return the arm's seeds whose forecast file exists for every month."""
    folder = arm_folder(arm=arm)
    return tuple(
        s
        for s in arms.ARMS[arm].seeds
        if all((folder / f"{m}_s{s}.parquet").exists() for m in months)
    )


def seed_average(*, variant: Variant, months: tuple[str, ...]) -> pl.DataFrame:
    """Return a variant's forecast (fraction of capacity) averaged over its seeds in quantile space.

    Raises:
        ValueError: If a seed differs in height or keys from the first.
    """
    folder = arm_folder(arm=variant.arm)
    pieces = []
    for month in months:
        frames = [pl.read_parquet(folder / f"{month}_s{seed}.parquet") for seed in variant.seeds]
        first_keys = frames[0].select(KEYS).sort(KEYS)
        for frame in frames:
            if frame.height != frames[0].height or not frame.select(KEYS).sort(KEYS).equals(
                first_keys
            ):
                msg = f"{variant.name} {month}: a seed differs in height or keys from the first"
                raise ValueError(msg)
        pieces.append(
            pl.concat(frames).group_by(KEYS).agg(*[pl.col(c).mean() for c in common.LEVEL_COLUMNS])
        )
    frame = pl.concat(pieces).sort(KEYS)
    values = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    return frame.with_columns(
        *[pl.Series(c, values[:, i]).cast(pl.Float32) for i, c in enumerate(common.LEVEL_COLUMNS)]
    )


def needed_variants(*, months: tuple[str, ...]) -> tuple[list[Variant], dict[str, tuple[int, ...]]]:
    """Return the variants to score, and each arm's available seeds.

    The variants are every arm with at least one complete seed (all the seeds that exist), and the
    seed subsets that the contrasts use: a contrast averages the seeds that both of its arms hold.
    The minimum detectable effect uses seeds 0 to 2 and 3 and 4 of `MDE_ARM` and of `R_24`.
    """
    have = {name: available_seeds(arm=name, months=months) for name in arms.ARMS}
    variants: dict[str, Variant] = {}
    for name, seeds in have.items():
        if seeds:
            variants[Variant(name, seeds).name] = Variant(name, seeds)
    for treatment, reference in CONTRASTS.values():
        shared = tuple(sorted(set(have[treatment]) & set(have[reference])))
        if shared:
            for name in (treatment, reference):
                variants[Variant(name, shared).name] = Variant(name, shared)
    for name in (MDE_ARM, "R_24"):
        for part in MDE_HALVES:
            if all(s in have[name] for s in part):
                variants[Variant(name, part).name] = Variant(name, part)
    return list(variants.values()), have


def load_frames(
    *, series: pl.DataFrame, months: tuple[str, ...], variants: list[Variant]
) -> dict[str, pl.DataFrame]:
    """Return the scored forecasts (MW): each variant widened by 1.15, and the references."""
    frames: dict[str, pl.DataFrame] = {}
    for variant in variants:
        stored = seed_average(variant=variant, months=months)
        frames[variant.name] = adapter.forecast_frame(
            stored=s407.widen_global(frame=stored), series=series, unit="fraction"
        )
    references = s397.references(series=series)
    frames["S0M_w115"] = references["xgb_monthly_e389_widened"]
    for name in REFERENCE_NAMES:
        frames[name] = references[name]
    print("scored:", ", ".join(frames))
    missing = [a for a in arms.ARMS if not available_seeds(arm=a, months=months)]
    if missing:
        print(f"arms with no complete seed over these folds, not scored: {missing}")
    return frames


def cleaned_scoring_context(*, series: pl.DataFrame) -> dict:
    """Return experiment 381's scoring context with the flagged exact-zero actuals removed.

    The flagged half-hours are the same ones the training rows drop (experiment 425's rule).
    """
    ctx = s381.build_context(series=series)
    flagged = pl.read_parquet(prep.FLAGGED_KEYS).select("time_series_id", "valid_time")
    before = ctx["actuals"].height
    ctx["actuals"] = ctx["actuals"].join(flagged, on=["time_series_id", "valid_time"], how="anti")
    print(f"scoring actuals dropped for a flagged exact zero: {before - ctx['actuals'].height}")
    return ctx


def check_logs(*, months: tuple[str, ...]) -> None:
    """Print the training rows dropped per fold and the ENS window coverage, from the fit logs."""
    print("\n== training rows dropped and window coverage per fold (from the fit logs, seed 0) ==")
    for month in months:
        for name, arm in arms.ARMS.items():
            path = OUT_DIR / "logs" / arm.folder / f"{month}_s0.json"
            if not path.exists():
                continue
            log = json.loads(path.read_text())
            coverage = log.get("ens_window_coverage")
            print(
                month,
                name,
                "fine-tuning rows",
                log.get("fine_tune_rows"),
                "first origin",
                log.get("fine_tune_first_origin"),
                "dropped",
                log.get("fine_tune_rows_dropped"),
                "S_ens coverage",
                coverage,
            )


def per_label_fold(
    *, rows: pl.DataFrame, members: tuple[str, ...], metric: str = "crps"
) -> pl.DataFrame:
    """Return, per fold, the group's metric: the mean over Primaries of each Primary's fold mean."""
    return (
        rows.filter(pl.col("label").is_in(list(members)) & pl.col(HEADLINE))
        .with_columns(fold=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
        .group_by("label", "fold")
        .agg(pl.col(metric).mean())
        .group_by("fold")
        .agg(pl.col(metric).mean())
        .sort("fold")
    )


def group_metric(
    *, rows: pl.DataFrame, members: tuple[str, ...], metric: str, where: pl.Expr | None
) -> float:
    """Return the equal-weight mean over Primaries of the headline-band metric on a row set."""
    keep = pl.col("label").is_in(list(members)) & pl.col(HEADLINE)
    if where is not None:
        keep &= where
    per_label = rows.filter(keep).group_by("label").agg(pl.col(metric).mean())
    return float("nan") if per_label.is_empty() else float(np.mean(per_label[metric].to_numpy()))


def minimum_detectable_effect(*, rows: dict[str, pl.DataFrame]) -> float | None:
    """Print and return the minimum detectable effect at the signed-MW six (a lower bound).

    The effect is the standard deviation over folds of the signed-MW six CRPS difference between
    the mean of an arm's seeds 0 to 2 and the mean of its seeds 3 and 4, divided by the square root
    of two. It ignores the interaction of folds with seed sets. The arm is `MDE_ARM`, or `R_24`
    (experiment 425's `R`) if `MDE_ARM` has not all five seeds.
    """
    print("\n== minimum detectable effect, signed-MW six, days 1 to 10 (before any contrast) ==")
    for arm in (MDE_ARM, "R_24"):
        first, second = (Variant(arm, half).name for half in MDE_HALVES)
        if first in rows and second in rows:
            a = per_label_fold(rows=rows[first], members=SIGNED)
            b = per_label_fold(rows=rows[second], members=SIGNED)
            gap = a.join(b, on="fold", suffix="_second").with_columns(
                difference=pl.col("crps") - pl.col("crps_second")
            )
            sd = float(np.std(gap["difference"].to_numpy(), ddof=1))
            mde = sd / np.sqrt(2)
            print(
                f"arm {arm}: {gap.height} folds, SD of the seed-half difference {sd:.4f}; "
                f"divided by sqrt(2): MDE {mde:.4f} points (a lower bound); the C1 minus C2 "
                f"interval is about 1.4 times wider"
            )
            return mde
    print("not computed: no arm holds seeds 0 to 2 and 3 and 4 on every fold")
    return None


def metric_tables(*, rows: dict[str, pl.DataFrame]) -> None:
    """Print CRPS and the MAE of the median per group, per Primary, and per lead band."""
    for metric in SCALE_METRICS:
        print(f"\n== {metric} at days 1 to 10 (equal weight per Primary, points) ==")
        for group, members in SUBSETS.items():
            print(
                group,
                {
                    n: round(group_metric(rows=r, members=members, metric=metric, where=None), 3)
                    for n, r in rows.items()
                },
            )
    print("\n== CRPS and MAE of the median at days 1 to 10 per Primary ==")
    per_primary = pl.concat(
        [
            r.filter(pl.col(HEADLINE))
            .group_by("label")
            .agg(*[pl.col(m).mean() for m in SCALE_METRICS])
            .with_columns(arm=pl.lit(n))
            for n, r in rows.items()
        ]
    )
    with pl.Config(tbl_rows=40, tbl_cols=30):
        for metric in SCALE_METRICS:
            print(metric)
            print(
                per_primary.pivot(on="arm", index="label", values=metric).sort(
                    pl.col("label").str.slice(1).cast(pl.Int32)
                )
            )
    FIGURE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    per_primary.write_parquet(FIGURE_TABLE_DIR / "metrics_per_primary.parquet")
    by_lead = []
    for name, frame in rows.items():
        by_lead.append(
            frame.filter(pl.col("lead_hours") < 336.0)
            .with_columns(lead_bin_hours=(pl.col("lead_hours") // LEAD_BIN_HOURS) * LEAD_BIN_HOURS)
            .group_by("label", "lead_bin_hours")
            .agg(*[pl.col(m).mean() for m in SCALE_METRICS])
            .with_columns(arm=pl.lit(name))
        )
    pl.concat(by_lead).write_parquet(FIGURE_TABLE_DIR / "metrics_by_lead.parquet")
    print(f"figure tables written to {FIGURE_TABLE_DIR}")
    lead_band_tables(rows=rows)


def lead_band_tables(*, rows: dict[str, pl.DataFrame]) -> None:
    """Print each group's CRPS and MAE of the median per 24-hour lead band, one column per arm.

    A cell is the equal-weight mean over the group's Primaries of each Primary's mean in the band.
    Band 1 is the first 24 hours after the origin.
    """
    per_band = pl.concat(
        [
            frame.filter(pl.col("lead_hours") < 336.0)
            .with_columns(lead_day=(pl.col("lead_hours") // LEAD_BAND_DAY_HOURS).cast(pl.Int32) + 1)
            .group_by("label", "lead_day")
            .agg(*[pl.col(m).mean() for m in SCALE_METRICS])
            .with_columns(arm=pl.lit(name))
            for name, frame in rows.items()
        ]
    )
    for group, members in SUBSETS.items():
        for metric in SCALE_METRICS:
            print(
                f"\n== {metric} per lead day (24-hour bands), {group} (equal weight per Primary) =="
            )
            table = (
                per_band.filter(pl.col("label").is_in(list(members)))
                .group_by("lead_day", "arm")
                .agg(pl.col(metric).mean())
                .pivot(on="arm", index="lead_day", values=metric)
                .sort("lead_day")
            )
            with pl.Config(tbl_rows=20, tbl_cols=30):
                print(table)


def slice_tables(*, scored: dict[str, pl.DataFrame]) -> None:
    """Print each metric on each protocol slice, per group."""
    one = next(iter(scored.values()))
    print("\n== metrics available:", sorted(one["metric"].unique().to_list()))
    print("== slices available:", sorted(one["slice"].unique().to_list()))
    for slice_label, slice_name in SLICES[1:]:
        assert slice_name is not None
        for group, members in SUBSETS.items():
            for metric in SCALE_METRICS:
                means = {}
                for arm, t in scored.items():
                    sub = t.filter(
                        pl.col("metric") == metric,
                        pl.col("slice") == slice_name,
                        pl.col("label").is_in(list(members)),
                    )
                    if sub.height:
                        means[arm] = round(float(np.nanmean(sub["value"].to_numpy())), 3)
                if means:
                    print(f"{metric} on {slice_label}, {group}: {means}")
                else:
                    print(f"{metric} on {slice_label}, {group}: the slice is empty")


def tails(*, rows: dict[str, pl.DataFrame], table: pl.DataFrame, reference: pl.DataFrame) -> None:
    """Print coverage and pinball loss at the four outer levels, overall and sunny low-demand.

    `reference` is the forecast (MW, with `q50`) whose median defines the sunny low-demand rows.
    """
    low = s407.sunny_low_demand(table=table, reference=reference)
    print(f"\nsunny low-demand half-hours: {low.height}")
    s407.tail_table(rows=rows, low=low)
    out = []
    for arm, frame in rows.items():
        part = frame.filter(pl.col(HEADLINE))
        for group, members in SUBSETS.items():
            sub = part.filter(pl.col("label").is_in(list(members)))
            per_primary = sub.group_by("label").agg(
                *[pl.col(f"coverage_{lv}").mean() for lv in OUTER_LEVELS],
                *[pl.col(f"pinball_{lv}").mean() for lv in OUTER_LEVELS],
            )
            out.append(
                {"arm": arm, "group": group}
                | {
                    c: float(np.nanmean(per_primary[c].to_numpy()))
                    for c in per_primary.columns
                    if c != "label"
                }
            )
    pl.DataFrame(out).write_parquet(FIGURE_TABLE_DIR / "tails.parquet")


def curtailment(
    *, frames: dict[str, pl.DataFrame], series: pl.DataFrame, ctx: dict, table: pl.DataFrame
) -> None:
    """Print the avoidable curtailment cost per series (signed-MW six) at each limit fraction."""
    flows = ctx["actuals"].rename({"actual": "flow_mw"})
    forecasts = {
        n: metrics.restrict_to_slice_table(forecast=f, slice_table=table).join(
            ctx["actuals"], on=["time_series_id", "valid_time"]
        )
        for n, f in frames.items()
    }
    for fraction in CURTAILMENT_FRACTIONS:
        cost = metrics.avoidable_curtailment_cost(
            forecasts=forecasts,
            series=series,
            flows=flows,
            scored_start=s377.SCORED_FIRST,
            fraction=fraction,
        )
        print(f"\n== avoidable curtailment cost, pounds, limit fraction {fraction} ==")
        print(cost.pivot(on="arm", index="label", values="avoidable_cost_gbp").sort("label"))
        print("total:", cost.group_by("arm").agg(pl.col("avoidable_cost_gbp").sum()).sort("arm"))


def median_minus_mean_per_site(*, frames: dict[str, pl.DataFrame], series: pl.DataFrame) -> None:
    """Print, per site and arm, the mean forecast median minus the mean of the quantile mixture.

    The mean of a forecast is the mass-weighted mean of its 11 quantiles (`common.ATOM_MASSES`).
    A spread that shifts the forecast shows as a median that sits away from the mean, in MW.
    """
    quantiles = [c.lower() for c in common.LEVEL_COLUMNS]
    labels = dict(zip(series["time_series_id"].to_list(), series["label"].to_list(), strict=True))
    for name, frame in frames.items():
        mean = sum(pl.col(q) * float(m) for q, m in zip(quantiles, common.ATOM_MASSES, strict=True))
        per_series = (
            frame.with_columns(gap=pl.col("q50") - mean)
            .group_by("time_series_id")
            .agg(pl.col("gap").mean())
            .sort("time_series_id")
        )
        print(
            f"median minus mean (MW), {name}:",
            {labels[int(i)]: round(float(g), 3) for i, g in per_series.iter_rows()},
        )


def paired_difference(*, plus: pl.DataFrame, minus: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Return the keys and label with `metric` set to `plus` minus `minus`, on the shared rows."""
    keys = [*KEYS, "label"]
    return (
        plus.select(*keys, metric)
        .join(minus.select(*KEYS, other=pl.col(metric)), on=KEYS, how="inner", validate="1:1")
        .with_columns((pl.col(metric) - pl.col("other")).alias(metric))
        .drop("other")
    )


def contrast_line(
    *,
    label: str,
    group: str,
    treatment: pl.DataFrame,
    reference: pl.DataFrame,
    table: pl.DataFrame,
    mde: float | None,
    metric: str = "crps",
    where: pl.Expr | None = None,
    level_percent: float = BONFERRONI_LEVEL_PERCENT,
    tag: str = "",
) -> dict[str, float] | None:
    """Print one contrast and return its figures, or None if the forecasts share no row.

    Args:
        label: The contrast's name.
        group: A key of `SUBSETS`.
        treatment: Scored rows of the treatment.
        reference: Scored rows of the reference.
        table: The shared slice table.
        mde: The minimum detectable effect, for the verdict at the signed-MW six.
        metric: A scale metric.
        where: An extra filter, such as a lead band or a dropped fold.
        level_percent: The interval's coverage.
        tag: A suffix for the printed label, such as `(descriptive)`.
    """
    members = SUBSETS[group]
    shown = f"{group} (report-only)" if group == REPORT_ONLY_GROUP else group
    keep = pl.col("label").is_in(list(members)) & pl.col(HEADLINE)
    if where is not None:
        keep &= where
    try:
        r = metrics.month_bootstrap_contrast(
            treatment=treatment,
            reference=reference,
            metric=metric,
            where=keep,
            level_percent=level_percent,
            n_resamples=N_RESAMPLES,
            slice_table=table,
        )
    except ValueError as error:
        print(f"{label:30s} {shown:18s} {metric} unavailable: {error}")
        return None
    reading = ""
    if mde is not None and group == SIX and metric == "crps":
        reading = "inside the noise" if abs(r["difference"]) < mde else "larger than the MDE"
    print(
        f"{label:30s} {shown:18s} {metric:10s} {r['difference']:+.3f} "
        f"({r['lower']:+.3f} to {r['upper']:+.3f}) level {r['level_percent']:.2f}% "
        f"n={r['n_half_hours']} {reading} {tag}"
    )
    return {k: float(r[k]) for k in ("difference", "lower", "upper")}


def c1_verdict(*, lower: float, upper: float) -> str:
    """Return the decision rule's verdict for C1 from its interval, in points of CRPS.

    The rule (plan page), applied in this order to the interval [lower, upper]: `refuted` if the
    lower limit is above 0; `supported` if the upper limit is below -0.02; `not supported` if the
    lower limit is at or above -0.02; `small gain` if the upper limit is below 0 (so the interval
    straddles -0.02); otherwise `inconclusive` (the interval straddles both -0.02 and 0), with its
    width printed.
    """
    if lower > 0:
        return "refuted (interval wholly above 0)"
    if upper < GAIN_THRESHOLD:
        return "supported (interval wholly below -0.02)"
    if lower >= GAIN_THRESHOLD:
        return "not supported (interval at or above -0.02)"
    if upper < 0:
        return "small gain (interval below 0 and straddling -0.02)"
    return f"inconclusive (straddles -0.02 and 0; width {upper - lower:.3f})"


def check_c1_verdict_branches() -> None:
    """Check, on synthetic intervals, that `c1_verdict` reaches each of its five branches.

    Raises:
        AssertionError: If an interval gives a verdict other than its branch's.
    """
    cases = {
        "refuted": (0.01, 0.05),
        "supported": (-0.10, -0.03),
        "not supported": (-0.02, 0.03),
        "small gain": (-0.05, -0.01),
        "inconclusive": (-0.08, 0.02),
    }
    for expected, (lower, upper) in cases.items():
        verdict = c1_verdict(lower=lower, upper=upper)
        assert verdict.startswith(expected), (expected, lower, upper, verdict)
    print("c1_verdict: all five branches pass on synthetic intervals")


def tested_contrasts(
    *,
    rows: dict[str, pl.DataFrame],
    have: dict[str, tuple[int, ...]],
    table: pl.DataFrame,
    mde: float | None,
) -> dict[tuple[str, str, str], dict[str, float]]:
    """Print the eight contrasts per group (CRPS, then the MAE of the median), and C1's verdicts.

    Returns:
        The CRPS figures by (contrast number, group, metric).
    """
    print(
        f"\n== tested contrasts (treatment minus reference, days 1 to 10, month-block bootstrap, "
        f"{N_RESAMPLES:,} resamples, level {BONFERRONI_LEVEL_PERCENT:.2f}%, Bonferroni alpha "
        f"{5.0 / N_READINGS / 100:.4f}; D15 and D18 report-only) =="
    )
    found: dict[tuple[str, str, str], dict[str, float]] = {}
    for label, (treatment, reference) in CONTRASTS.items():
        names = variant_names(treatment=treatment, reference=reference, rows=rows, have=have)
        if names is None:
            print(f"{label}: skipped, an arm is not scored")
            continue
        for group in SUBSETS:
            for metric in SCALE_METRICS:
                figures = contrast_line(
                    label=label,
                    group=group,
                    treatment=rows[names[0]],
                    reference=rows[names[1]],
                    table=table,
                    mde=mde,
                    metric=metric,
                )
                if figures is not None:
                    found[label.split()[0], group, metric] = figures
    for group in TESTED_GROUPS:
        c1 = found.get(("C1", group, "crps"))
        if c1 is not None:
            print(f"\nC1 verdict, {group}: {c1_verdict(lower=c1['lower'], upper=c1['upper'])}")
    return found


def variant_names(
    *,
    treatment: str,
    reference: str,
    rows: dict[str, pl.DataFrame],
    have: dict[str, tuple[int, ...]],
) -> tuple[str, str] | None:
    """Return the scored names of a contrast's two arms on their shared seeds, or None."""
    shared = tuple(sorted(set(have[treatment]) & set(have[reference])))
    first, second = (Variant(a, shared).name for a in (treatment, reference))
    return (first, second) if shared and first in rows and second in rows else None


def descriptive_readings(
    *,
    rows: dict[str, pl.DataFrame],
    have: dict[str, tuple[int, ...]],
    table: pl.DataFrame,
    mde: float | None,
) -> None:
    """Print the readings outside the 16 tested ones, each labelled descriptive.

    They are C1 with and without fold 2026-01, C1 for days 1 to 6 and 7 to 10, the 12-month gain,
    C1 per Primary, a t test on the 12 monthly differences of C1, and C1 minus C2 with its own
    interval.
    """
    print("\n== descriptive readings (not among the 16 tested; same level and resamples) ==")
    names = variant_names(treatment="A_F", reference="R_F", rows=rows, have=have)
    if names is not None:
        treatment, reference = rows[names[0]], rows[names[1]]
        month = pl.col("power_fcst_init_time").dt.strftime("%Y-%m")
        for group in TESTED_GROUPS:
            contrast_line(
                label="C1",
                group=group,
                treatment=treatment,
                reference=reference,
                table=table,
                mde=mde,
                where=month != DROPPED_FOLD,
                tag="(descriptive: without fold 2026-01)",
            )
            contrast_line(
                label="C1",
                group=group,
                treatment=treatment,
                reference=reference,
                table=table,
                mde=mde,
                where=pl.col("lead_hours") <= EARLY_LEAD_HOURS,
                tag="(descriptive: days 1 to 6)",
            )
            contrast_line(
                label="C1",
                group=group,
                treatment=treatment,
                reference=reference,
                table=table,
                mde=mde,
                where=pl.col("lead_hours") > EARLY_LEAD_HOURS,
                tag="(descriptive: days 7 to 10)",
            )
            monthly = per_label_fold(rows=treatment, members=SUBSETS[group]).join(
                per_label_fold(rows=reference, members=SUBSETS[group]),
                on="fold",
                suffix="_reference",
            )
            difference = (monthly["crps"] - monthly["crps_reference"]).to_numpy()
            test = stats.ttest_1samp(difference, 0.0)
            print(
                f"C1, {group} (descriptive): per-fold differences "
                f"{[round(float(v), 3) for v in difference]}; {int((difference < 0).sum())} of "
                f"{len(difference)} negative; t test on the monthly differences "
                f"t={float(test.statistic):.2f}, p={float(test.pvalue):.3f}"
            )
        print("\nC1 per Primary (descriptive, 95% intervals):")
        with pl.Config(tbl_rows=40):
            print(
                metrics.reported_contrast_by_primary(
                    treatment=treatment,
                    reference=reference,
                    metric="crps",
                    where=pl.col(HEADLINE),
                    slice_table=table,
                )
            )
        c2_names = variant_names(treatment="A_E24", reference="R_E24", rows=rows, have=have)
        if c2_names is not None:
            plus = paired_difference(plus=treatment, minus=reference, metric="crps")
            other = paired_difference(
                plus=rows[c2_names[0]], minus=rows[c2_names[1]], metric="crps"
            )
            for group in TESTED_GROUPS:
                contrast_line(
                    label="C1 minus C2",
                    group=group,
                    treatment=plus,
                    reference=other,
                    table=table,
                    mde=None,
                    tag="(descriptive)",
                )
    twelve = variant_names(treatment="A_12", reference="R_12", rows=rows, have=have)
    if twelve is not None:
        for group in TESTED_GROUPS:
            contrast_line(
                label="12-month gain A_12 minus R_12",
                group=group,
                treatment=rows[twelve[0]],
                reference=rows[twelve[1]],
                table=table,
                mde=mde,
                tag="(descriptive)",
            )


def audit_triggers(*, found: dict[tuple[str, str, str], dict[str, float]]) -> None:
    """Print the audit triggers that the contrasts decide: 1 (placebo), 3 (MAE), and 5 (R_F)."""
    print("\n== audit triggers decided by the contrasts ==")
    for group in TESTED_GROUPS:
        c5 = found.get(("C5", group, "crps"))
        if c5 is not None and c5["upper"] >= 0:
            print(f"TRIGGER 1 (placebo failed): C5 is not wholly below 0 at the {group}")
        for number in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"):
            crps = found.get((number, group, "crps"))
            mae = found.get((number, group, "mae_median"))
            if crps is not None and mae is not None and crps["difference"] * mae["difference"] < 0:
                print(
                    f"TRIGGER 3: {number} at the {group}: CRPS {crps['difference']:+.3f} and "
                    f"the MAE of the median {mae['difference']:+.3f} have opposite signs"
                )
        c3 = found.get(("C3", group, "crps"))
        if c3 is not None and c3["upper"] < 0:
            print(
                f"TRIGGER 5: R_F beats A_24 at the {group} (C3 wholly below 0): "
                f"an arm without pre-training beats the stored recipe"
            )


def triggers_2_6_7(
    *,
    rows: dict[str, pl.DataFrame],
    have: dict[str, tuple[int, ...]],
    table: pl.DataFrame,
    found: dict[tuple[str, str, str], dict[str, float]],
) -> None:
    """Print audit triggers 2, 6, and 7.

    Both 2 and 6 compare C1 with the descriptive line "stored 425 `A` minus `R`" (`A_24` minus
    `R_24`), computed on the same rows, at the same level, and with the same resamples as C1.

    - **Trigger 2** (C1 contradicts experiment 425) fires if C1 and the line have opposite signs, or
      if their intervals do not overlap.
    - **Trigger 6** (the headline interval is much wider or narrower than the line's) fires if C1's
      interval width is more than twice or less than half the line's.
    - **Trigger 7** (a later design depends on the headline) needs a judgement about later
      designs, so a line says that the page's author must make it.

    Args:
        rows: The scored rows of every variant.
        have: Each arm's available seeds.
        table: The shared slice table.
        found: The tested contrasts' figures, from `tested_contrasts`.
    """
    print("\n== audit triggers 2, 6, and 7 (C1 against the stored 425 A minus R) ==")
    names = variant_names(treatment="A_24", reference="R_24", rows=rows, have=have)
    if names is None:
        print("triggers 2 and 6 not decided: A_24 or R_24 is not scored")
    else:
        for group in TESTED_GROUPS:
            line = contrast_line(
                label="A_24 minus R_24 (stored 425)",
                group=group,
                treatment=rows[names[0]],
                reference=rows[names[1]],
                table=table,
                mde=None,
                tag="(descriptive)",
            )
            c1 = found.get(("C1", group, "crps"))
            if line is None or c1 is None:
                print(f"triggers 2 and 6 not decided at the {group}: C1 or the line is missing")
                continue
            opposite = c1["difference"] * line["difference"] < 0
            apart = c1["lower"] > line["upper"] or c1["upper"] < line["lower"]
            if opposite or apart:
                print(
                    f"TRIGGER 2: C1 {c1['difference']:+.3f} ({c1['lower']:+.3f} to "
                    f"{c1['upper']:+.3f}) contradicts the stored 425 line "
                    f"{line['difference']:+.3f} ({line['lower']:+.3f} to {line['upper']:+.3f}) "
                    f"at the {group}: "
                    f"{'opposite signs' if opposite else 'intervals do not overlap'}"
                )
            width_c1, width_line = c1["upper"] - c1["lower"], line["upper"] - line["lower"]
            ratio = width_c1 / width_line
            print(
                f"C1 interval width {width_c1:.3f}, stored line {width_line:.3f}, ratio {ratio:.2f}"
            )
            if ratio > TRIGGER_6_FACTOR or ratio < 1 / TRIGGER_6_FACTOR:
                print(
                    f"TRIGGER 6: at the {group} C1's interval is {ratio:.2f} times the width of "
                    f"the stored 425 line's (limits: {1 / TRIGGER_6_FACTOR:.1f} and "
                    f"{TRIGGER_6_FACTOR:.1f})"
                )
    print(
        "TRIGGER 7 not decided by the scorer: the page's author must judge whether a later design "
        "depends on the headline"
    )


def mean_ens_months(*, window: arms.Window, months: tuple[str, ...]) -> float:
    """Return a window's months of ENS history before the cut, averaged over the folds.

    A trailing window holds its length. A floored window holds the months from the floor to each
    fold's cut.
    """
    if window.days is not None:
        return window.days / DAYS_PER_MONTH
    assert window.floor is not None
    spans = [(prep.refit_of(month=m) - window.floor).days / DAYS_PER_MONTH for m in months]
    return float(np.mean(spans))


def optimise_svg(*, path: Path) -> None:
    """Run `svgo` on an SVG in place, as the `check-added-large-files` hook needs."""
    subprocess.run([*SVGO_COMMAND, "--final-newline", str(path)], check=True)


def gain_panel(*, points: pl.DataFrame, group: str) -> alt.LayerChart | alt.FacetChart:
    """Return one group's panel of the pre-trained minus ENS-only CRPS against ENS months."""
    data = points.filter(pl.col("group") == group)
    ticks = sorted({round(v) for v in data["x"].to_list()})
    labels = [w.label for w in GAIN_WINDOWS]
    colour = alt.Color(
        "window:N",
        scale=alt.Scale(domain=labels, range=[w.colour for w in GAIN_WINDOWS]),
        legend=alt.Legend(orient="top", direction="vertical", title=None),
    )
    x = alt.X(
        "x:Q",
        title="Mean months of ENS history before the cut",
        scale=alt.Scale(domain=[0, 75]),
        axis=alt.Axis(values=ticks),
    )
    y_title = wrapped(
        text="Pre-trained minus ENS-only CRPS (points of capacity; more negative is better)",
        width=40,
    )
    zero = (
        alt.Chart(pl.DataFrame({"zero": [0.0]}))
        .mark_rule(color="#555555", strokeDash=[4, 3])
        .encode(y=alt.Y("zero:Q", title=y_title))  # ty: ignore[unresolved-attribute]
    )
    zero_text = (
        alt.Chart(pl.DataFrame({"zero": [0.0], "x": [75.0], "text": ["same as ENS-only"]}))
        .mark_text(align="right", baseline="bottom", dy=-3, color="#555555")
        .encode(x="x:Q", y="zero:Q", text="text:N")  # ty: ignore[unresolved-attribute]
    )
    interval = (
        alt.Chart(data)
        .mark_rule(strokeWidth=2.5)
        .encode(x=x, y="lower:Q", y2="upper:Q", color=colour)  # ty: ignore[unresolved-attribute]
    )
    estimate = (
        alt.Chart(data)
        .mark_point(filled=True, size=90, opacity=1)
        .encode(x=x, y="difference:Q", color=colour)  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(zero, zero_text, interval, estimate).properties(
        width=CONTENT_WIDTH_PX - 110,
        height=190,
        title=alt.TitleParams(text=group, anchor="start"),
    )


def figure_gain_against_ens_months(
    *,
    rows: dict[str, pl.DataFrame],
    have: dict[str, tuple[int, ...]],
    table: pl.DataFrame,
    months: tuple[str, ...],
) -> None:
    """Write the figure of the pre-trained minus ENS-only CRPS against the months of ENS history.

    The x axis is the mean months of ENS history before the cut for the windows 12 months, 24
    months, F, and C. The y axis is the CRPS of the pre-trained arm minus the ENS-only arm of that
    window, from `A_12` minus `R_12`, C2, C1, and `A_C` minus `R_C`, each with its interval at the
    Bonferroni level (the 12-month and C points are descriptive). One panel per group.
    """
    print("\n== figure: gain against ENS months (each point is a contrast at the same level) ==")
    points = []
    for gain in GAIN_WINDOWS:
        names = variant_names(
            treatment=gain.treatment, reference=gain.reference, rows=rows, have=have
        )
        if names is None:
            print(f"{gain.label}: not drawn, an arm is not scored")
            continue
        x = mean_ens_months(window=gain.window, months=months)
        for group in TESTED_GROUPS:
            figures = contrast_line(
                label=f"{gain.treatment} minus {gain.reference}",
                group=group,
                treatment=rows[names[0]],
                reference=rows[names[1]],
                table=table,
                mde=None,
                tag=f"(figure point at {x:.1f} months)",
            )
            if figures is not None:
                points.append({"group": group, "window": gain.label, "x": x, **figures})
    if not points:
        print("figure not drawn: no window has both arms scored")
        return
    data = pl.DataFrame(points)
    title = alt.TitleParams(
        text="Does the gain from pre-training shrink as the ENS history grows?",
        subtitle=wrapped(
            text=(
                "Dot: difference of CRPS in days 1 to 10 (equal weight per Primary). Line: "
                f"{FIGURE_LEVEL_TEXT} interval from resampling whole months. The 12-month and "
                "window C points are descriptive. Rows that every arm shares."
            ),
            width=95,
        ),
        anchor="start",
    )
    chart = alt.vconcat(*[gain_panel(points=data, group=g) for g in TESTED_GROUPS]).properties(
        title=title
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / "gain_against_ens_months.svg"
    chart.save(path)
    optimise_svg(path=path)
    print(f"figure written to {path}")


def example_panel(
    *, forecast: pl.DataFrame, actual: pl.DataFrame, title: str, ticks: list[int]
) -> alt.LayerChart | alt.FacetChart:
    """Return one panel: each arm's 5% to 95% band and median, and the actual, against the lead."""
    names = [*EXAMPLE_ARMS, "Actual"]
    colour = alt.Color(
        "series:N",
        scale=alt.Scale(
            domain=names, range=[ocf_theme.DATA_BLUE, ocf_theme.BRAND_ORANGE, "#222222"]
        ),
        legend=alt.Legend(orient="top", title=None),
    )
    x = alt.X(
        "lead_hours:Q",
        title="Hours after the origin",
        axis=alt.Axis(values=ticks),
        scale=alt.Scale(domain=[min(ticks), max(ticks)]),
    )
    band = (
        alt.Chart(forecast)
        .mark_area(opacity=0.2)
        .encode(x=x, y=alt.Y("q05:Q", title="Demand (MW)"), y2="q95:Q", color=colour)  # ty: ignore[unresolved-attribute]
    )
    median = (
        alt.Chart(forecast).mark_line().encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
    )
    observed = (
        alt.Chart(actual).mark_line(strokeWidth=2).encode(x=x, y="actual:Q", color=colour)  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(band, median, observed).properties(
        width=CONTENT_WIDTH_PX - 110, height=170, title=alt.TitleParams(text=title, anchor="start")
    )


def figure_example_forecasts(
    *,
    frames: dict[str, pl.DataFrame],
    have: dict[str, tuple[int, ...]],
    series: pl.DataFrame,
    ctx: dict,
) -> None:
    """Write a day-ahead and a full 14-day example forecast against the actual, per group.

    The arms are `EXAMPLE_ARMS` (C1's two arms), the origin is `EXAMPLE_ORIGIN`, and the Primary of
    each group is in `EXAMPLE_PRIMARIES`. Primaries are substations, so no NGED metered generator's
    series is drawn. The band is the 5% to 95% interval of the widened forecast, and the line is the
    median. The x axis counts hours after the origin.
    """
    print("\n== figure: example forecasts ==")
    names = {a: Variant(a, have[a]).name for a in EXAMPLE_ARMS if have.get(a)}
    if len(names) != len(EXAMPLE_ARMS) or any(n not in frames for n in names.values()):
        print(f"figure not drawn: {EXAMPLE_ARMS} are not both scored")
        return
    lead = (
        (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    ).alias("lead_hours")
    day_start = (EXAMPLE_ORIGIN + timedelta(days=1)).replace(hour=0)
    panels = []
    for group, label in EXAMPLE_PRIMARIES.items():
        series_id = int(series.filter(pl.col("label") == label)["time_series_id"][0])
        pieces = [
            frames[names[a]]
            .filter(
                pl.col("time_series_id") == series_id,
                pl.col("power_fcst_init_time") == EXAMPLE_ORIGIN,
            )
            .with_columns(lead, series=pl.lit(a))
            for a in EXAMPLE_ARMS
        ]
        forecast = pl.concat(pieces)
        if forecast.is_empty():
            print(f"{group}: no forecast of {label} from the origin {EXAMPLE_ORIGIN}; not drawn")
            continue
        actual = (
            ctx["actuals"]
            .filter(pl.col("time_series_id") == series_id)
            .join(
                forecast.select("valid_time", "lead_hours").unique(), on="valid_time", how="inner"
            )
            .with_columns(series=pl.lit("Actual"))
        )
        for horizon, ticks, keep in (
            (
                "day-ahead",
                [12, 18, 24, 30, 36, 42],
                (pl.col("valid_time") >= day_start)
                & (pl.col("valid_time") < day_start + timedelta(days=1)),
            ),
            ("full 14-day", list(range(0, 361, 48)), pl.lit(True)),
        ):
            panels.append(
                example_panel(
                    forecast=forecast.filter(keep),
                    actual=actual.filter(keep),
                    title=(
                        f"{group}: Primary {label}, {horizon} forecast from "
                        f"{EXAMPLE_ORIGIN:%Y-%m-%d %H:%M} UTC"
                    ),
                    ticks=ticks,
                )
            )
    if not panels:
        print("figure not drawn: no panel has a forecast")
        return
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            text="Example forecasts of the two arms of C1 against the actual demand",
            subtitle=wrapped(
                text=(
                    "Band: 5% to 95% of the widened forecast. Line: median. Each arm is the "
                    "average of its seeds in quantile space. One Primary per group."
                ),
                width=95,
            ),
            anchor="start",
        )
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / "example_forecasts.svg"
    chart.save(path)
    optimise_svg(path=path)
    print(f"figure written to {path}")


def main() -> None:
    """Print the checks, the minimum detectable effect, the tables, and the contrasts."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--months", nargs="+", default=None)
    parser.add_argument("--screen", action="store_true", help="the 4 screen folds; interim")
    parser.add_argument("--smoke", action="store_true", help="read work/smoke")
    args = parser.parse_args()
    global OUT_DIR, FIGURE_TABLE_DIR  # noqa: PLW0603
    if args.smoke:
        OUT_DIR = arms.WORK_DIR / "work" / "smoke"
        FIGURE_TABLE_DIR = OUT_DIR / "figure_tables"
    check_c1_verdict_branches()
    months = tuple(args.months or (arms.SCREEN_MONTHS if args.screen else arms.ALL_MONTHS))
    if args.screen:
        print("SCREEN: all numbers are interim")
    series = adapter.series_table()
    ctx = cleaned_scoring_context(series=series)
    variants, have = needed_variants(months=months)
    frames = load_frames(series=series, months=months, variants=variants)
    check_logs(months=months)
    table, scored, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    print(f"shared half-hours across every arm and reference: {table.height}")
    mde = minimum_detectable_effect(rows=rows)
    display = [Variant(a, seeds).name for a, seeds in have.items() if seeds]
    shown = {n: rows[n] for n in (*display, "S0M_w115", *REFERENCE_NAMES) if n in rows}
    metric_tables(rows=shown)
    slice_tables(scored={n: scored[n] for n in shown})
    reference = next((frames[n] for n in shown if n.startswith("R_24")), frames["bg2"])
    tails(rows=shown, table=table, reference=reference)
    curtailment(frames={n: frames[n] for n in shown}, series=series, ctx=ctx, table=table)
    median_minus_mean_per_site(frames={n: frames[n] for n in shown}, series=series)
    found = tested_contrasts(rows=rows, have=have, table=table, mde=mde)
    descriptive_readings(rows=rows, have=have, table=table, mde=mde)
    audit_triggers(found=found)
    triggers_2_6_7(rows=rows, have=have, table=table, found=found)
    figure_gain_against_ens_months(rows=rows, have=have, table=table, months=months)
    figure_example_forecasts(frames=frames, have=have, series=series, ctx=ctx)


if __name__ == "__main__":
    main()
