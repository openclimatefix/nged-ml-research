# ruff: noqa: E501
"""Score experiment 425's arms against the references and print every number the page quotes.

Reads the per-seed raw quantiles `e00425_fit.py` wrote (arms `R`, `A`, and `Anoemb`, seeds 0 to 4,
12 monthly folds), averages the seeds in quantile space, and scores each arm widened by 1.15 about
the median (`_w115`) and unwidened (`_raw`). The references are scored on the half-hours every arm
shares, and the shared row count is printed:

- `S0M_raw` and `S0M_w115`: the monthly XGBoost recipe, experiment 389's stored arm `S0M_raw`
  (read through experiment 397's `references`), raw and widened by 1.15 here. Experiment 389
  predates the D15 and D18 start cut and the flagged-zero drop, so S0M is trained without either
  (not verified against the stored fit, which this script cannot see);
- `bg2`, `manual_heuristic`, and `climatology`: experiment 87's stored forecasts, as stored.

Printed in this order: the match of every arm's keys, the training rows dropped per fold (from the
fit logs), the minimum detectable effect (before any contrast), CRPS per group and per Primary,
coverage and pinball loss at q01, q05, q95, and q99, the cold winter night, winter night, sunny
half-hour, and Economy 7 step slices, the avoidable curtailment cost, the four tested contrasts
C1 to C4 and their verdicts, and the `--no-cut` cross-check if it was fitted. Tables for the
figures go to `work/figure_tables/`.

```bash
# PYTHONPATH as in e00425_fit.py (experiments 87, 48, 377, 381, 397, 407, 418, and 425)
uv run --no-sync python ml-experiments/00425-*/e00425_score.py
```
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as s377  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00397_score as s397  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00425_fit as fit  # ty: ignore[unresolved-import]
import e00425_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics

HEADLINE: Final[str] = s407.HEADLINE
SUBSETS: Final[dict[str, tuple[str, ...]]] = s407.SUBSETS
SIGNED: Final[tuple[str, ...]] = s407.SIGNED
OUTER_LEVELS: Final[tuple[str, ...]] = s407.OUTER_LEVELS
KEYS: Final[list[str]] = list(common.KEYS)
SIX: Final[str] = "signed-MW six"
EIGHT: Final[str] = "MVA-only eight"
REPORT_ONLY_GROUP: Final[str] = "D15 and D18"
SEEDS: Final[tuple[int, ...]] = fit.DEFAULT_SEEDS
SEEDS_FIRST: Final[tuple[int, ...]] = (0, 1, 2)
SEEDS_SECOND: Final[tuple[int, ...]] = (3, 4)
N_TESTED_CONTRASTS: Final[int] = 4
BONFERRONI_LEVEL_PERCENT: Final[float] = 100.0 - 5.0 / N_TESTED_CONTRASTS
"""The two-sided interval level that keeps a family-wise 5% over the four tested contrasts."""
N_RESAMPLES: Final[int] = 20_000
MIN_GAIN: Final[float] = 0.03
"""The smallest gain worth porting the pretraining stage for (experiment 418's threshold)."""
CURTAILMENT_FRACTIONS: Final[tuple[float, ...]] = (0.9, 0.75, 0.5)
SLICES: Final[tuple[tuple[str, str | None], ...]] = (
    ("days 1 to 10", None),
    ("cold nights", "cold_nights"),
    ("winter nights", "winter_nights"),
    ("sunny half-hours", "sunny_unit_shape"),
    ("Economy 7 step", "economy_7_step"),
)
LEAD_BIN_HOURS: Final[int] = 6
FIGURE_TABLE_DIR: Path = fit.OUT_DIR / "figure_tables"
ARM_NAMES: Final[tuple[str, ...]] = tuple(fit.ARMS)
REFERENCE_NAMES: Final[tuple[str, ...]] = ("bg2", "manual_heuristic", "climatology")
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1 A minus R (widened)": ("A_w115", "R_w115"),
    "C2 A-noemb minus A (widened)": ("Anoemb_w115", "A_w115"),
    "C3 A unwidened minus A widened": ("A_raw", "A_w115"),
    "C4 A minus S0M (both widened)": ("A_w115", "S0M_w115"),
}
"""The four tested contrasts: label, (treatment, reference)."""


@dataclass(frozen=True)
class ArmFiles:
    """An arm's forecasts: the folder name, the seeds averaged, and the folds."""

    name: str
    folder: str
    seeds: tuple[int, ...]
    months: tuple[str, ...]

    def files(self) -> list[Path]:
        """Return every per-seed forecast file the arm needs."""
        return [
            fit.OUT_DIR / "seed_preds" / self.folder / f"{month}_s{seed}.parquet"
            for month in self.months
            for seed in self.seeds
        ]

    def complete(self) -> bool:
        """Return whether every file exists."""
        return all(f.exists() for f in self.files())


def seed_average(*, arm: ArmFiles) -> pl.DataFrame:
    """Return an arm's forecast (fraction of capacity) averaged over its seeds in quantile space."""
    pieces = []
    for month in arm.months:
        frames = [
            pl.read_parquet(fit.OUT_DIR / "seed_preds" / arm.folder / f"{month}_s{seed}.parquet")
            for seed in arm.seeds
        ]
        first_keys = frames[0].select(KEYS).sort(KEYS)
        for frame in frames:
            if frame.height != frames[0].height or not frame.select(KEYS).sort(KEYS).equals(
                first_keys
            ):
                msg = f"{arm.name} {month}: a seed differs in height or keys from the first"
                raise ValueError(msg)
        pieces.append(
            pl.concat(frames).group_by(KEYS).agg(*[pl.col(c).mean() for c in common.LEVEL_COLUMNS])
        )
    frame = pl.concat(pieces).sort(KEYS)
    values = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    return frame.with_columns(
        *[pl.Series(c, values[:, i]).cast(pl.Float32) for i, c in enumerate(common.LEVEL_COLUMNS)]
    )


def load_frames(
    *,
    series: pl.DataFrame,
    months: tuple[str, ...],
    seeds: tuple[int, ...] = SEEDS,
    with_references: bool = True,
) -> dict[str, pl.DataFrame]:
    """Return the scored forecasts (MW): each complete arm raw and widened, and the references."""
    frames: dict[str, pl.DataFrame] = {}
    defs = [ArmFiles(name, name, seeds, months) for name in ARM_NAMES]
    defs += [
        ArmFiles("R_s012", "R", SEEDS_FIRST, months),
        ArmFiles("R_s34", "R", SEEDS_SECOND, months),
    ]
    for arm in defs:
        if not arm.complete():
            missing = sum(not f.exists() for f in arm.files())
            print(f"arm {arm.name} skipped: {missing} of {len(arm.files())} files missing")
            continue
        stored = seed_average(arm=arm)
        frames[f"{arm.name}_w115"] = adapter.forecast_frame(
            stored=s407.widen_global(frame=stored), series=series, unit="fraction"
        )
        if arm.name in ARM_NAMES:
            frames[f"{arm.name}_raw"] = adapter.forecast_frame(
                stored=stored, series=series, unit="fraction"
            )
    if with_references:
        references = s397.references(series=series)
        frames["S0M_raw"] = references["xgb_monthly_e389"]
        frames["S0M_w115"] = references["xgb_monthly_e389_widened"]
        for name in REFERENCE_NAMES:
            frames[name] = references[name]
    print("arms scored:", ", ".join(frames))
    return frames


def check_keys(*, frames: dict[str, pl.DataFrame], months: tuple[str, ...]) -> None:
    """Print the rows per arm and the training rows dropped per fold, from the fit logs."""
    print("\n== rows per arm (before restriction to the shared half-hours) ==")
    print({name: frame.height for name, frame in frames.items()})
    print("\n== training rows dropped per fold (from the fit logs) ==")
    for month in months:
        path = fit.OUT_DIR / "logs" / f"{month}.json"
        if path.exists():
            log = json.loads(path.read_text())
            print(
                month,
                "fine-tuning:",
                log.get("fine_tune_rows_dropped"),
                "pretraining:",
                log.get("pretrain_rows_dropped"),
            )
        else:
            print(month, "no fit log")


def per_label_fold(*, rows: pl.DataFrame, members: tuple[str, ...]) -> pl.DataFrame:
    """Return, per fold, the group's CRPS: the mean over Primaries of each Primary's fold mean."""
    return (
        rows.filter(pl.col("label").is_in(list(members)) & pl.col(HEADLINE))
        .with_columns(fold=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
        .group_by("label", "fold")
        .agg(pl.col("crps").mean())
        .group_by("fold")
        .agg(pl.col("crps").mean())
        .sort("fold")
    )


def group_crps(*, rows: pl.DataFrame, members: tuple[str, ...], where: pl.Expr | None) -> float:
    """Return the equal-weight mean over Primaries of the headline-band CRPS on a row set."""
    keep = pl.col("label").is_in(list(members)) & pl.col(HEADLINE)
    if where is not None:
        keep &= where
    per_label = rows.filter(keep).group_by("label").agg(pl.col("crps").mean())
    return float("nan") if per_label.is_empty() else float(np.mean(per_label["crps"].to_numpy()))


def minimum_detectable_effect(*, rows: dict[str, pl.DataFrame]) -> float | None:
    """Print and return the minimum detectable effect at the signed-MW six (a lower bound).

    The effect is the standard deviation over folds of the signed-MW six CRPS difference between
    the mean of R's seeds 0 to 2 and the mean of R's seeds 3 and 4, divided by the square root of
    two. It ignores the interaction of folds with seed sets.
    """
    print("\n== minimum detectable effect, signed-MW six, days 1 to 10 (before any contrast) ==")
    if "R_s012_w115" not in rows or "R_s34_w115" not in rows:
        print("not computed: R seeds 0 to 2 or seeds 3 and 4 are not complete")
        return None
    first = per_label_fold(rows=rows["R_s012_w115"], members=SIGNED)
    second = per_label_fold(rows=rows["R_s34_w115"], members=SIGNED)
    gap = first.join(second, on="fold", suffix="_second").with_columns(
        difference=pl.col("crps") - pl.col("crps_second")
    )
    sd = float(np.std(gap["difference"].to_numpy(), ddof=1))
    mde = sd / np.sqrt(2)
    print(gap.rename({"crps": "R_seeds_0_to_2", "crps_second": "R_seeds_3_and_4"}))
    print(
        f"{gap.height} folds, SD of the difference {sd:.4f}; divided by sqrt(2): "
        f"MDE {mde:.4f} points (a lower bound)"
    )
    return mde


def crps_tables(*, rows: dict[str, pl.DataFrame]) -> None:
    """Print group CRPS for every arm and reference, and per Primary, and store the figure tables."""
    print("\n== CRPS at days 1 to 10 (equal weight per Primary, points) ==")
    for group, members in SUBSETS.items():
        print(
            group,
            {n: round(group_crps(rows=r, members=members, where=None), 3) for n, r in rows.items()},
        )
    print("\n== CRPS at days 1 to 10 per Primary ==")
    per_primary = pl.concat(
        [
            r.filter(pl.col(HEADLINE))
            .group_by("label")
            .agg(pl.col("crps").mean())
            .with_columns(arm=pl.lit(n))
            for n, r in rows.items()
        ]
    )
    with pl.Config(tbl_rows=40, tbl_cols=30):
        print(
            per_primary.pivot(on="arm", index="label", values="crps").sort(
                pl.col("label").str.slice(1).cast(pl.Int32)
            )
        )
    FIGURE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    per_primary.write_parquet(FIGURE_TABLE_DIR / "crps_per_primary.parquet")
    by_lead = []
    for name, frame in rows.items():
        by_lead.append(
            frame.filter(pl.col("lead_hours") < 336.0)
            .with_columns(lead_bin_hours=(pl.col("lead_hours") // LEAD_BIN_HOURS) * LEAD_BIN_HOURS)
            .group_by("label", "lead_bin_hours")
            .agg(pl.col("crps").mean())
            .with_columns(arm=pl.lit(name))
        )
    pl.concat(by_lead).write_parquet(FIGURE_TABLE_DIR / "crps_by_lead.parquet")
    print(f"figure tables written to {FIGURE_TABLE_DIR}")


def slice_tables(*, scored: dict[str, pl.DataFrame]) -> None:
    """Print group CRPS on each slice, and the metric and slice names the table offers."""
    one = next(iter(scored.values()))
    print("\n== metrics available:", sorted(one["metric"].unique().to_list()))
    print("== slices available:", sorted(one["slice"].unique().to_list()))
    for slice_label, slice_name in SLICES[1:]:
        assert slice_name is not None
        for group, members in SUBSETS.items():
            means = {}
            for arm, t in scored.items():
                sub = t.filter(
                    pl.col("metric") == "crps",
                    pl.col("slice") == slice_name,
                    pl.col("label").is_in(list(members)),
                )
                if sub.height:
                    means[arm] = round(float(np.nanmean(sub["value"].to_numpy())), 3)
            if means:
                print(f"CRPS on {slice_label}, {group}: {means}")


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


def contrast_line(
    *,
    label: str,
    group: str,
    treatment: pl.DataFrame,
    reference: pl.DataFrame,
    table: pl.DataFrame,
    mde: float | None,
) -> dict[str, float] | None:
    """Print one tested contrast at the Bonferroni level and return the figures, or None."""
    members = SUBSETS[group]
    shown = f"{group} (report-only)" if group == REPORT_ONLY_GROUP else group
    try:
        r = metrics.month_bootstrap_contrast(
            treatment=treatment,
            reference=reference,
            metric="crps",
            where=pl.col("label").is_in(list(members)) & pl.col(HEADLINE),
            level_percent=BONFERRONI_LEVEL_PERCENT,
            n_resamples=N_RESAMPLES,
            slice_table=table,
        )
    except ValueError as error:
        print(f"{label:34s} {shown:26s} unavailable: {error}")
        return None
    verdict = ""
    if mde is not None and group == SIX:
        verdict = "inside the noise" if abs(r["difference"]) < mde else "larger than the MDE"
    print(
        f"{label:34s} {shown:26s} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) "
        f"level {r['level_percent']:.2f}% n={r['n_half_hours']} {verdict}"
    )
    return {k: float(r[k]) for k in ("difference", "lower", "upper")}


def tested_contrasts(
    *, rows: dict[str, pl.DataFrame], table: pl.DataFrame, mde: float | None
) -> None:
    """Print C1 to C4 per group, the per-fold signs of C1, and the decision-rule verdicts."""
    print(
        f"\n== tested contrasts (treatment minus reference, CRPS days 1 to 10, month-block "
        f"bootstrap, {N_RESAMPLES:,} resamples, level {BONFERRONI_LEVEL_PERCENT:.2f}%, "
        f"Bonferroni alpha {5.0 / N_TESTED_CONTRASTS / 100:.4f}; D15 and D18 report-only) =="
    )
    found: dict[tuple[str, str], dict[str, float]] = {}
    for label, (treatment, reference) in CONTRASTS.items():
        if treatment not in rows or reference not in rows:
            print(f"{label}: skipped, an arm is not scored")
            continue
        for group in SUBSETS:
            figures = contrast_line(
                label=label,
                group=group,
                treatment=rows[treatment],
                reference=rows[reference],
                table=table,
                mde=mde,
            )
            if figures is not None:
                found[label.split()[0], group] = figures
    for (treatment, reference), name in (
        (("A_w115", "R_w115"), "C1 A minus R"),
        (("Anoemb_w115", "A_w115"), "C2 A-noemb minus A"),
    ):
        if treatment in rows and reference in rows:
            for group in (SIX, EIGHT):
                members = SUBSETS[group]
                gap = per_label_fold(rows=rows[treatment], members=members).join(
                    per_label_fold(rows=rows[reference], members=members),
                    on="fold",
                    suffix="_reference",
                )
                difference = (gap["crps"] - gap["crps_reference"]).to_numpy()
                print(
                    f"{name}, {group}: per-fold differences "
                    f"{[round(float(v), 3) for v in difference]}; {int((difference < 0).sum())} of "
                    f"{len(difference)} folds negative"
                )
    verdicts(found=found)


def verdicts(*, found: dict[tuple[str, str], dict[str, float]]) -> None:
    """Print the decision rule's reading of C1, and the audit triggers of C2 and C3."""
    c1_six, c1_eight = found.get(("C1", SIX)), found.get(("C1", EIGHT))
    if c1_six is not None:
        if c1_six["upper"] < -MIN_GAIN:
            reading = "supported (six interval wholly below -0.03)"
        elif (c1_eight is not None and c1_eight["upper"] < -MIN_GAIN) or c1_six[
            "difference"
        ] < -MIN_GAIN:
            reading = "partly supported (eight meets the bar, or the six point estimate passes with an interval reaching -0.03 or above)"
        else:
            reading = "not supported"
        print(f"\nC1 reading against the decision rule: {reading}")
    c2 = found.get(("C2", SIX))
    if c2 is not None and c2["difference"] <= 0:
        print("AUDIT TRIGGER: C2 (A-noemb minus A) is not worse than zero at the signed-MW six")
    c3 = found.get(("C3", SIX))
    if c3 is not None and c3["difference"] < 0:
        print("AUDIT TRIGGER: C3 (A unwidened minus A widened) reversed sign at the signed-MW six")


def cross_check(*, series: pl.DataFrame, ctx: dict) -> None:
    """Print the cut effect: arms fitted without the cuts (seed 0) beside the same fold and seed with."""
    months = tuple(
        m
        for m in prep.MONTHS
        if all(
            (
                fit.OUT_DIR / "seed_preds" / fit.folder_name(arm=a, cut=False) / f"{m}_s0.parquet"
            ).exists()
            for a in ("R",)
        )
    )
    if not months:
        print("\n== cut cross-check: no --no-cut fit found ==")
        return
    print(f"\n== cut cross-check, seed 0, folds {months} (widened 1.15, scored on shared rows) ==")
    frames = {}
    for arm in ARM_NAMES:
        for cut in (True, False):
            files = ArmFiles(arm, fit.folder_name(arm=arm, cut=cut), (0,), months)
            if files.complete():
                label = f"{arm}_{'cut' if cut else 'nocut'}"
                frames[label] = adapter.forecast_frame(
                    stored=s407.widen_global(frame=seed_average(arm=files)),
                    series=series,
                    unit="fraction",
                )
    if not frames:
        print("no complete arm")
        return
    _, _, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    for group, members in SUBSETS.items():
        print(
            group,
            {n: round(group_crps(rows=r, members=members, where=None), 3) for n, r in rows.items()},
        )


def main() -> None:
    """Print the checks, the minimum detectable effect, the tables, and the contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(prep.MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--smoke", action="store_true", help="read work/smoke")
    args = parser.parse_args()
    if args.smoke:
        fit.OUT_DIR = prep.WORK_DIR / "work" / "smoke"
        global FIGURE_TABLE_DIR  # noqa: PLW0603
        FIGURE_TABLE_DIR = fit.OUT_DIR / "figure_tables"
    months = tuple(args.months)
    series = adapter.series_table()
    ctx = prep.cleaned_scoring_context(series=series)
    frames = load_frames(series=series, months=months, seeds=tuple(args.seeds))
    if "R_w115" not in frames:
        msg = "arm R is incomplete: no reference for the contrasts"
        raise ValueError(msg)
    check_keys(frames=frames, months=months)
    table, scored, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    print(f"shared half-hours across every arm and reference: {table.height}")
    mde = minimum_detectable_effect(rows=rows)
    keep = [n for n in rows if not n.startswith("R_s")]
    shown_rows = {n: rows[n] for n in keep}
    crps_tables(rows=shown_rows)
    slice_tables(scored={n: scored[n] for n in keep})
    tails(rows=shown_rows, table=table, reference=frames["R_w115"])
    curtailment(frames={n: frames[n] for n in keep}, series=series, ctx=ctx, table=table)
    tested_contrasts(rows=rows, table=table, mde=mde)
    cross_check(series=series, ctx=ctx)


if __name__ == "__main__":
    main()
