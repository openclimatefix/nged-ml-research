# ruff: noqa: E501
r"""Score experiment 437's feature-bundle arms against experiment 425's reference network `R`.

Reads the per-seed raw quantiles `e00437_fit.py` wrote (seeds 0 to 2 of each arm on the four screen
folds) and experiment 425's stored `R` (seeds 0 to 2 for the control; seeds 3 and 4 only for the
minimum detectable effect), averages the seeds in quantile space, widens every arm by 1.15 about
the median, and scores them on the half-hours they share with `scoring actuals without the flagged
exact zeros`. Printed in this order: the match of the keys and the rows dropped per fold, the
minimum detectable effect (before any contrast), CRPS per group and per Primary, the tails
(coverage and pinball loss at q01, q05, q95, and q99), the protocol's slices and four slices of
this experiment (cold sunny days, cold overcast days, shoulder seasons, morning ramp hours), the
avoidable curtailment cost, the contrasts of each arm minus `R` with the decision rule applied,
and the fold-by-fold table. Tables for the figures go to `work/figure_tables/`.

The decision rule (registered in `result.md` before any score was read): an arm advances if it
beats its control in at least 3 of the 4 folds, its pooled gain is at least 0.03 at the signed-MW
six or the MVA-only eight, its point estimate at the other group is below +0.03, and its increase
in pinball loss at q95 and q99 is at most 0.02 at both groups. The bootstrap intervals are printed
as a description only. Arms that are halves of a bundle are descriptive.

```bash
# PYTHONPATH as in e00437_fit.py
uv run --no-sync python ml-experiments/00437-*/e00437_score.py
```
"""

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00397_score as s397  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00425_score as sc  # ty: ignore[unresolved-import]
import e00437_features as feats  # ty: ignore[unresolved-import]
import e00437_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics

SCREEN_ARMS: Final[tuple[str, ...]] = ("T", "S", "C", "C_tariff", "F", "T_pre", "S_pre", "F_pre")
"""The eight tested arms; every other arm is a descriptive half or a combination.

A `_pre` arm is compared with experiment 425's pretrained network `A`; every other arm with `R`.
"""
N_TESTED: Final[int] = len(SCREEN_ARMS)
BONFERRONI_LEVEL_PERCENT: Final[float] = 100.0 - 5.0 / (2 * N_TESTED)
CONTROLS: Final[tuple[str, ...]] = ("R", "A")
HEADLINE: Final[str] = sc.HEADLINE
SUBSETS: Final[dict[str, tuple[str, ...]]] = sc.SUBSETS
SIX: Final[str] = sc.SIX
EIGHT: Final[str] = sc.EIGHT
MIN_GAIN: Final[float] = 0.03
MAX_HARM: Final[float] = 0.03
MIN_FOLDS: Final[int] = 3
"""Folds out of four in which a bundle must beat its control."""
MAX_TAIL_HARM: Final[float] = 0.02
COLD_C: Final[float] = 5.0
SUNNY_INDEX: Final[float] = 0.6
OVERCAST_INDEX: Final[float] = 0.3
SHOULDER_MONTHS: Final[tuple[int, ...]] = (3, 4, 5, 9, 10, 11)
SLICE_COLUMNS: Final[dict[str, pl.Expr]] = {
    "cold sunny days": pl.col("cold_day") & pl.col("sunny_day"),
    "cold overcast days": pl.col("cold_day") & pl.col("overcast_day"),
    "shoulder seasons": pl.col("shoulder"),
    "morning ramp hours": pl.col("morning_ramp"),
}
FIGURE_TABLE_DIR: Path = fit.OUT_DIR / "figure_tables"


def average_seeds(*, arm: str, months: tuple[str, ...], seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return an arm's forecast (fraction of capacity) averaged over its seeds in quantile space."""
    pieces = []
    for month in months:
        frames = [pl.read_parquet(fit.seed_path(arm=arm, month=month, seed=s)) for s in seeds]
        first = frames[0].select(sc.KEYS).sort(sc.KEYS)
        for frame in frames:
            if frame.height != frames[0].height or not frame.select(sc.KEYS).sort(sc.KEYS).equals(
                first
            ):
                msg = f"{arm} {month}: a seed differs in height or keys from the first"
                raise ValueError(msg)
        pieces.append(
            pl.concat(frames)
            .group_by(sc.KEYS)
            .agg(*[pl.col(c).mean() for c in common.LEVEL_COLUMNS])
        )
    frame = pl.concat(pieces).sort(sc.KEYS)
    values = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    return frame.with_columns(
        *[pl.Series(c, values[:, i]).cast(pl.Float32) for i, c in enumerate(common.LEVEL_COLUMNS)]
    )


def load_frames(
    *, series: pl.DataFrame, months: tuple[str, ...], arms: list[str]
) -> dict[str, pl.DataFrame]:
    """Return the widened forecasts (MW): the control from experiment 425, the arms, the references."""
    frames: dict[str, pl.DataFrame] = {}
    controls = [
        (f"{control}{suffix}", control, seeds)
        for control in CONTROLS
        for suffix, seeds in (("", (0, 1, 2)), ("_s012", (0, 1, 2)), ("_s34", (3, 4)))
    ]
    for name, control, seeds in controls:
        stored = sc.seed_average(arm=sc.ArmFiles(name, control, seeds, months))
        frames[f"{name}_w115"] = adapter.forecast_frame(
            stored=s407.widen_global(frame=stored), series=series, unit="fraction"
        )
    for arm in arms:
        missing = [
            fit.seed_path(arm=arm, month=m, seed=s)
            for m in months
            for s in fit.SEEDS
            if not fit.seed_path(arm=arm, month=m, seed=s).exists()
        ]
        if missing:
            print(f"arm {arm} skipped: {len(missing)} files missing")
            continue
        stored = average_seeds(arm=arm, months=months, seeds=fit.SEEDS)
        frames[f"{arm}_w115"] = adapter.forecast_frame(
            stored=s407.widen_global(frame=stored), series=series, unit="fraction"
        )
    references = s397.references(series=series)
    frames["S0M_w115"] = references["xgb_monthly_e389_widened"]
    for name in sc.REFERENCE_NAMES:
        frames[name] = references[name]
    print("arms scored:", ", ".join(frames))
    return frames


def diagnostics(*, months: tuple[str, ...]) -> pl.DataFrame:
    """Return per (Primary, origin, valid time): the ENS-mean temperature, clear-sky index, and slice masks."""
    first = fit.base.refit_of(month=months[0])
    last = fit.base.refit_of(month=months[-1]) + timedelta(days=40)
    parts = [
        pl.read_parquet(
            feats.table_path(site=row["site"]),
            columns=["origin_us", "valid_us", "diag_temperature", "diag_csi"],
        )
        .filter(
            pl.col("origin_us").is_between(
                int(first.timestamp() * 1e6), int(last.timestamp() * 1e6)
            )
        )
        .with_columns(
            label=pl.lit(row["site"]),
            power_fcst_init_time=pl.col("origin_us").cast(pl.Datetime("us", "UTC")),
            valid_time=pl.col("valid_us").cast(pl.Datetime("us", "UTC")),
        )
        .drop("origin_us", "valid_us")
        for row in fit.base.primaries().iter_rows(named=True)
    ]
    start_local = pl.col("valid_time").dt.offset_by("-30m").dt.convert_time_zone("Europe/London")
    hour = start_local.dt.hour().cast(pl.Float64) + start_local.dt.minute().cast(pl.Float64) / 60.0
    daytime = hour.is_between(9.0, 14.5)
    return pl.concat(parts).with_columns(
        cold_day=(pl.col("diag_temperature") < COLD_C) & daytime,
        sunny_day=pl.col("diag_csi").is_not_nan() & (pl.col("diag_csi") > SUNNY_INDEX),
        overcast_day=pl.col("diag_csi") < OVERCAST_INDEX,
        shoulder=start_local.dt.month().is_in(list(SHOULDER_MONTHS)),
        morning_ramp=hour.is_between(6.0, 9.0, closed="left"),
    )


def with_masks(*, rows: dict[str, pl.DataFrame], diag: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return each arm's scored rows with the slice-mask columns of `diagnostics` joined on."""
    keep = [
        "label",
        "power_fcst_init_time",
        "valid_time",
        "cold_day",
        "sunny_day",
        "overcast_day",
        "shoulder",
        "morning_ramp",
    ]
    out = {}
    for name, frame in rows.items():
        joined = frame.join(
            diag.select(keep),
            on=["label", "power_fcst_init_time", "valid_time"],
            how="left",
            validate="m:1",
        )
        unmatched = int(joined["cold_day"].null_count())
        if unmatched:
            print(f"{name}: {unmatched} scored rows have no diagnostic row; masked out")
        out[name] = joined.with_columns(pl.col(c).fill_null(False) for c in keep[3:])
    return out


def custom_slices(*, rows: dict[str, pl.DataFrame], table: pl.DataFrame, arms: list[str]) -> None:
    """Print each arm's CRPS and its contrast with `R` on the four slices of this experiment."""
    print(
        "\n== slices of this experiment: CRPS days 1 to 10 and (arm minus its control, R or A for a pretrained arm) with the Bonferroni interval =="
    )
    for slice_name, mask in SLICE_COLUMNS.items():
        for group, members in SUBSETS.items():
            where = pl.col("label").is_in(list(members)) & pl.col(HEADLINE) & mask
            print(
                f"-- {slice_name}, {group}: R_w115 CRPS {sc.group_crps(rows=rows['R_w115'], members=members, where=mask):.3f}, "
                f"A_w115 CRPS {sc.group_crps(rows=rows['A_w115'], members=members, where=mask):.3f}"
            )
            for arm in arms:
                if f"{arm}_w115" not in rows:
                    continue
                try:
                    r = metrics.month_bootstrap_contrast(
                        treatment=rows[f"{arm}_w115"],
                        reference=rows[f"{control_of(arm=arm)}_w115"],
                        metric="crps",
                        where=where,
                        level_percent=BONFERRONI_LEVEL_PERCENT,
                        n_resamples=sc.N_RESAMPLES,
                        slice_table=table,
                    )
                except ValueError as error:
                    print(f"   {arm}: unavailable ({error})")
                    continue
                print(
                    f"   {arm:14s} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) n={r['n_half_hours']}"
                )


def tail_harm(*, rows: dict[str, pl.DataFrame], arm: str) -> dict[str, dict[str, float]]:
    """Return, per group, the arm's increase over `R` in mean pinball loss at q95 and q99."""
    out: dict[str, dict[str, float]] = {}
    for group, members in SUBSETS.items():

        def level_mean(
            frame: pl.DataFrame, level: str, members: tuple[str, ...] = members
        ) -> float:
            per_primary = (
                frame.filter(pl.col("label").is_in(list(members)) & pl.col(HEADLINE))
                .group_by("label")
                .agg(pl.col(f"pinball_{level}").mean())
            )
            return float(np.nanmean(per_primary[f"pinball_{level}"].to_numpy()))

        out[group] = {
            lv: level_mean(rows[f"{arm}_w115"], lv)
            - level_mean(rows[f"{control_of(arm=arm)}_w115"], lv)
            for lv in ("q95", "q99")
        }
    return out


def control_of(*, arm: str) -> str:
    """Return the control arm an arm is compared with: `A` for a pretrained arm, else `R`."""
    return "A" if fit.is_pretrained(arm=arm) else "R"


def minimum_detectable_effects(*, rows: dict[str, pl.DataFrame]) -> dict[str, float | None]:
    """Print and return the minimum detectable effect at the signed-MW six for each control.

    The effect is the standard deviation over the folds of the signed-MW six CRPS difference
    between the control's seeds 0 to 2 and its seeds 3 and 4, divided by the square root of two (a
    lower bound, as in experiment 425). Four folds make the estimate rough.
    """
    out: dict[str, float | None] = {}
    for control in CONTROLS:
        first, second = f"{control}_s012_w115", f"{control}_s34_w115"
        if first not in rows or second not in rows:
            out[control] = None
            continue
        gap = sc.per_label_fold(rows=rows[first], members=SUBSETS[SIX]).join(
            sc.per_label_fold(rows=rows[second], members=SUBSETS[SIX]), on="fold", suffix="_second"
        )
        sd = float(np.std((gap["crps"] - gap["crps_second"]).to_numpy(), ddof=1))
        out[control] = sd / np.sqrt(2)
        print(
            f"minimum detectable effect, control {control}, signed-MW six, {gap.height} folds: {out[control]:.4f} points (a lower bound; experiment 425 had 12 folds)"
        )
    return out


def decisions(
    *,
    rows: dict[str, pl.DataFrame],
    table: pl.DataFrame,
    mdes: dict[str, float | None],
    arms: list[str],
) -> None:
    """Print each arm's contrasts at the three groups, the fold signs, the tail harm, and the rule's reading."""
    print(
        f"\n== contrasts, arm minus its control (CRPS days 1 to 10, month-block bootstrap over 4 folds, {sc.N_RESAMPLES:,} resamples, "
        f"level {BONFERRONI_LEVEL_PERCENT:.2f}%; D15 and D18 report-only) =="
    )
    for arm in arms:
        if f"{arm}_w115" not in rows:
            continue
        found = {}
        for group in SUBSETS:
            figures = sc.contrast_line(
                label=f"{arm} minus {control_of(arm=arm)}",
                group=group,
                treatment=rows[f"{arm}_w115"],
                reference=rows[f"{control_of(arm=arm)}_w115"],
                table=table,
                mde=mdes[control_of(arm=arm)],
            )
            if figures is not None:
                found[group] = figures
        negative = {}
        for group in (SIX, EIGHT):
            gap = sc.per_label_fold(rows=rows[f"{arm}_w115"], members=SUBSETS[group]).join(
                sc.per_label_fold(rows=rows[f"{control_of(arm=arm)}_w115"], members=SUBSETS[group]),
                on="fold",
                suffix="_reference",
            )
            difference = (gap["crps"] - gap["crps_reference"]).to_numpy()
            negative[group] = int((difference < 0).sum())
            print(
                f"   {arm}, {group}: fold differences {dict(zip(gap['fold'].to_list(), [round(float(v), 3) for v in difference], strict=True))}"
            )
        harm = tail_harm(rows=rows, arm=arm)
        print(
            f"   {arm}: increase in pinball loss over the control at q95 and q99 by group: "
            + str({g: {k: round(v, 4) for k, v in h.items()} for g, h in harm.items()})
        )
        if arm not in SCREEN_ARMS or SIX not in found or EIGHT not in found:
            print(f"   {arm}: descriptive, no advance decision")
            continue
        wins_six = (
            found[SIX]["difference"] <= -MIN_GAIN
            and negative[SIX] >= MIN_FOLDS
            and found[EIGHT]["difference"] < MAX_HARM
        )
        wins_eight = (
            found[EIGHT]["difference"] <= -MIN_GAIN
            and negative[EIGHT] >= MIN_FOLDS
            and found[SIX]["difference"] < MAX_HARM
        )
        tail_ok = all(v <= MAX_TAIL_HARM for g in (SIX, EIGHT) for v in harm[g].values())
        reading = "ADVANCES" if (wins_six or wins_eight) and tail_ok else "does not advance"
        note = "" if tail_ok else " (tail harm above 0.02)"
        print(
            f"   {arm}: {reading}{note}; inside the noise at the six: {mdes[control_of(arm=arm)] is not None and abs(found[SIX]['difference']) < mdes[control_of(arm=arm)]}"
        )


def main() -> None:
    """Print the checks, the minimum detectable effect, the tables, and the contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--arms", nargs="+", default=list(SCREEN_ARMS))
    parser.add_argument("--smoke", action="store_true", help="read work/smoke")
    args = parser.parse_args()
    if args.smoke:
        fit.OUT_DIR = feats.WORK_DIR / "work" / "smoke"
    global FIGURE_TABLE_DIR  # noqa: PLW0603
    FIGURE_TABLE_DIR = fit.OUT_DIR / "figure_tables"
    sc.FIGURE_TABLE_DIR = FIGURE_TABLE_DIR
    sc.BONFERRONI_LEVEL_PERCENT = BONFERRONI_LEVEL_PERCENT
    months = tuple(args.months)
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    frames = load_frames(series=series, months=months, arms=args.arms)
    print("\n== training rows dropped per fold (from the fit logs) ==")
    for month in months:
        log_path = fit.OUT_DIR / "logs" / f"{month}.json"
        if log_path.exists():
            log = json.loads(log_path.read_text())
            print(month, {k: v["dropped"] for k, v in log.items() if k.endswith("_rows")})
    table, scored, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    print(f"shared half-hours across every arm and reference: {table.height}")
    mdes = minimum_detectable_effects(rows=rows)
    shown = {n: r for n, r in rows.items() if not n.startswith(("R_s", "A_s"))}
    sc.crps_tables(rows=shown)
    sc.slice_tables(scored={n: scored[n] for n in shown})
    sc.tails(rows=shown, table=table, reference=frames["R_w115"])
    masked = with_masks(rows=rows, diag=diagnostics(months=months))
    custom_slices(rows=masked, table=table, arms=args.arms)
    sc.curtailment(frames={n: frames[n] for n in shown}, series=series, ctx=ctx, table=table)
    decisions(rows=rows, table=table, mdes=mdes, arms=args.arms)


if __name__ == "__main__":
    main()
