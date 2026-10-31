# ruff: noqa: E501
"""Score experiment 423's calendar-encoder arms against the stored pooled-network reference.

Reads the per-seed forecasts `e00423_fit.py` wrote, the stored `ctrl` of experiment 407 (seeds 0 to
2), and the stored `ctrl_s3` of experiment 419 (seeds 3 to 5). Seeds are averaged in quantile space
(as experiment 419's scorer does), every arm is widened by 1.15 about the median (`_w115`), and the
scorer prints, in this order:

- the match of every arm's keys to the reference's, and the per-fold counts of rows dropped as
  flagged exact zeros;
- the minimum detectable effect on all rows and on the bank-holiday-week rows, before any contrast:
  2 times the standard deviation over folds of the signed-MW six CRPS gap between `ctrl` (seeds 0
  to 2) and `ctrl_s3` (seeds 3 to 5), divided by the square root of two for six-seed means;
- the R0 reproduction check: R refitted through experiment 423's code path (seed 0, fold 2025-12)
  against the stored `ctrl` seed 0 and against the stored seed-to-seed gap;
- CRPS at days 1 to 10 per group on every row set, per Primary on all rows, and coverage and
  pinball loss at q01, q05, q95, and q99 (raw and widened) on all rows and on bank-holiday weeks;
- the four tested contrasts C1 to C4 (month-block bootstrap of `studies.eval_protocol`, 20,000
  resamples, 95% level Bonferroni-corrected over four contrasts), each per group (the signed-MW
  six, the MVA-only eight, and D15 and D18 report-only), and C3's sign in each fold;
- the descriptive contrasts at an uncorrected 95% level: E1 minus R on the Christmas and Easter
  sets, and E3 minus E1 if E3 was fitted;
- the audit check per site: the median forecast's error minus the error of the mean of the
  forecast distribution (the mean of the 11 quantiles, weighted by the share of probability each
  represents), beside the reference;
- the event fold 2026-04, if fitted: point estimates only (one fold gives no month-block interval),
  with R0 as the reference because the stored reference holds no 2026-04.

Arms that are not complete (every screen month and seed) are skipped with a message.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00407-pooled-mixture-density-network-tail-arms-and-recency-weights:\
ml-experiments/00423-pooled-mdn-shared-date-time-encoder \
uv run --no-sync python ml-experiments/00423-*/e00423_score.py
```
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00423_features as feats  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT_419: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00419-pooled-mixture-density-network-solar-summary-sunny-flag-and-half-life-input-ports"
)
SEED_419: Final[Path] = ROOT_419 / "work" / "seed_preds"
ROOT_423: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00423-pooled-mdn-shared-date-time-encoder"
SEED_423: Final[Path] = ROOT_423 / "work" / "seed_preds"
MONTHS: Final[tuple[str, ...]] = s407.MONTHS
EVENT_MONTH: Final[str] = "2026-04"
HEADLINE: Final[str] = s407.HEADLINE
SUBSETS: Final[dict[str, tuple[str, ...]]] = s407.SUBSETS
SIGNED: Final[tuple[str, ...]] = s407.SIGNED
OUTER_LEVELS: Final[tuple[str, ...]] = s407.OUTER_LEVELS
KEYS: Final[list[str]] = list(common.KEYS)
SEEDS_0_TO_2: Final[tuple[int, ...]] = (0, 1, 2)
SEEDS_3_TO_5: Final[tuple[int, ...]] = (3, 4, 5)
SEEDS_0_TO_5: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5)
N_TESTED_CONTRASTS: Final[int] = 4
BONFERRONI_LEVEL_PERCENT: Final[float] = 100.0 - 5.0 / N_TESTED_CONTRASTS
"""The two-sided interval level that keeps a family-wise 5% over the four tested contrasts."""
DESCRIPTIVE_LEVEL_PERCENT: Final[float] = 95.0
N_RESAMPLES: Final[int] = 20_000
ROW_SETS: Final[dict[str, pl.Expr | None]] = {
    "all rows": None,
    "bank-holiday weeks": pl.col("bank_holiday_week"),
    "Christmas set": pl.col("christmas_set"),
    "Easter set": pl.col("easter_set"),
}
"""The row sets of the plan, as filters on rows that `add_row_sets` has widened."""
REPORT_ONLY_GROUP: Final[str] = "D15 and D18"
"""The group that is reported and never tested."""


@dataclass(frozen=True)
class ArmDef:
    """An arm's forecasts: folders, each with the seeds that live there."""

    name: str
    sources: tuple[tuple[Path, tuple[int, ...]], ...]
    months: tuple[str, ...] = MONTHS

    def files(self) -> list[Path]:
        """Return every per-seed forecast file the arm needs."""
        return [
            folder / f"{month}_s{seed}.parquet"
            for month in self.months
            for folder, seeds in self.sources
            for seed in seeds
        ]

    def complete(self) -> bool:
        """Return whether every file exists."""
        return all(f.exists() for f in self.files())


STORED_CTRL: Final[Path] = s407.SEED_407 / "ctrl"
STORED_CTRL_S3: Final[Path] = SEED_419 / "ctrl_s3"
REFERENCE: Final[str] = "R"
ARM_DEFS: Final[tuple[ArmDef, ...]] = (
    ArmDef("R", ((STORED_CTRL, SEEDS_0_TO_2), (STORED_CTRL_S3, SEEDS_3_TO_5))),
    ArmDef("R_s012", ((STORED_CTRL, SEEDS_0_TO_2),)),
    ArmDef("R_s345", ((STORED_CTRL_S3, SEEDS_3_TO_5),)),
    ArmDef("E0", ((SEED_423 / "E0", SEEDS_0_TO_2),)),
    ArmDef("E1n", ((SEED_423 / "E1n", SEEDS_0_TO_5),)),
    ArmDef("E1", ((SEED_423 / "E1", SEEDS_0_TO_5),)),
    ArmDef("E2", ((SEED_423 / "E2", SEEDS_0_TO_5),)),
    ArmDef("E2_s012", ((SEED_423 / "E2", SEEDS_0_TO_2),)),
    ArmDef("E3", ((SEED_423 / "E3", SEEDS_0_TO_5),)),
)
RAW_TAIL_ARMS: Final[tuple[str, ...]] = ("R", "E0", "E1n", "E1", "E2", "E3")
"""The arms whose raw (unwidened) forecasts are also scored, for the tails table."""
CONTRASTS: Final[dict[str, tuple[str, str, str, int]]] = {
    "C1 E1 minus R": ("E1", "R", "all rows", 6),
    "C2 E2 minus R": ("E2", "R", "all rows", 6),
    "C3 E1 minus E1n": ("E1", "E1n", "bank-holiday weeks", 6),
    "C4 E2 minus E0 (seeds 0-2 both)": ("E2_s012", "E0", "all rows", 3),
}
"""The four tested contrasts: label, (treatment, reference, row set, seeds per mean)."""
DESCRIPTIVE_CONTRASTS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("E1 minus R, Christmas set", "E1", "R", "Christmas set"),
    ("E1 minus R, Easter set", "E1", "R", "Easter set"),
    ("E3 minus E1, all rows", "E3", "E1", "all rows"),
    ("E3 minus E1, bank-holiday weeks", "E3", "E1", "bank-holiday weeks"),
)
LEVEL_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (common.LEVELS[:-1] + common.LEVELS[1:]) / 2, [1.0]])
)
"""The probability each of the 11 quantiles stands for, so a weighted sum gives the mean."""


def seed_average(*, arm: ArmDef) -> pl.DataFrame:
    """Return an arm's forecast (fraction of capacity) averaged over its seeds in quantile space."""
    pieces = []
    for month in arm.months:
        files = [
            folder / f"{month}_s{seed}.parquet" for folder, seeds in arm.sources for seed in seeds
        ]
        seeds = [pl.read_parquet(f) for f in files]
        for f, frame in zip(files, seeds, strict=True):
            same = frame.height == seeds[0].height and frame.select(KEYS).sort(KEYS).equals(
                seeds[0].select(KEYS).sort(KEYS)
            )
            if not same:
                msg = f"{f} differs in height or keys from {files[0]}"
                raise ValueError(msg)
        stacked = pl.concat(seeds)
        pieces.append(stacked.group_by(KEYS).agg(*[pl.col(c).mean() for c in common.LEVEL_COLUMNS]))
    frame = pl.concat(pieces).sort(KEYS)
    values = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    return frame.with_columns(
        *[pl.Series(c, values[:, i]).cast(pl.Float32) for i, c in enumerate(common.LEVEL_COLUMNS)]
    )


def load_frames(
    *, series: pl.DataFrame, arm_defs: tuple[ArmDef, ...], raw_arms: tuple[str, ...]
) -> dict[str, pl.DataFrame]:
    """Return the scored forecasts (MW): each complete arm widened by 1.15, and `raw_arms` raw."""
    frames: dict[str, pl.DataFrame] = {}
    for arm in arm_defs:
        if not arm.complete():
            missing = sum(not f.exists() for f in arm.files())
            print(f"arm {arm.name} skipped: {missing} of {len(arm.files())} files missing")
            continue
        stored = seed_average(arm=arm)
        frames[f"{arm.name}_w115"] = adapter.forecast_frame(
            stored=s407.widen_global(frame=stored), series=series, unit="fraction"
        )
        if arm.name in raw_arms:
            frames[f"{arm.name}_raw"] = adapter.forecast_frame(
                stored=stored, series=series, unit="fraction"
            )
    print("arms scored:", ", ".join(frames))
    return frames


def check_keys(*, frames: dict[str, pl.DataFrame], reference: str) -> None:
    """Print per-fold row counts and raise if any arm's keys differ from the reference's."""
    reference_keys = frames[reference].select(KEYS)
    counts = {
        n: f.group_by(pl.col("power_fcst_init_time").dt.strftime("%Y-%m").alias("fold"))
        .len()
        .sort("fold")
        for n, f in frames.items()
    }
    print("\n== rows per fold (month of forecast origin) ==")
    print(
        pl.concat([c.with_columns(arm=pl.lit(n)) for n, c in counts.items()]).pivot(
            on="arm", index="fold", values="len"
        )
    )
    for name, frame in frames.items():
        keys = frame.select(KEYS)
        extra = keys.join(reference_keys, on=KEYS, how="anti").height
        missing = reference_keys.join(keys, on=KEYS, how="anti").height
        if extra or missing:
            msg = f"{name}: {extra} keys not in the reference and {missing} reference keys missing"
            raise ValueError(msg)
    print("every arm's keys match the reference's")
    path = ROOT_423 / "work" / "dropped_rows.json"
    if path.exists():
        print(
            "training rows dropped as flagged exact zeros, per fold:", json.loads(path.read_text())
        )


def group_crps(*, rows: pl.DataFrame, members: tuple[str, ...], where: pl.Expr | None) -> float:
    """Return the equal-weight mean over Primaries of the headline-band CRPS on a row set."""
    keep = pl.col("label").is_in(list(members)) & pl.col(HEADLINE)
    if where is not None:
        keep &= where
    per_label = rows.filter(keep).group_by("label").agg(pl.col("crps").mean())
    return float("nan") if per_label.is_empty() else float(np.mean(per_label["crps"].to_numpy()))


def fold_gap(
    *,
    first: pl.DataFrame,
    second: pl.DataFrame,
    where: pl.Expr | None,
    members: tuple[str, ...] = SIGNED,
) -> pl.DataFrame:
    """Return, per fold with rows, the group's CRPS of two fits and their difference."""
    parts = []
    for name, rows in (("first", first), ("second", second)):
        keep = pl.col("label").is_in(list(members)) & pl.col(HEADLINE)
        if where is not None:
            keep &= where
        parts.append(
            rows.filter(keep)
            .with_columns(fold=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
            .group_by("label", "fold")
            .agg(pl.col("crps").mean())
            .group_by("fold")
            .agg(pl.col("crps").mean().alias(name))
        )
    return (
        parts[0]
        .join(parts[1], on="fold")
        .sort("fold")
        .with_columns(difference=pl.col("first") - pl.col("second"))
    )


P1_MDE_LIMIT: Final[float] = 0.10
"""P1 is read only if the minimum detectable effect on bank-holiday-week rows is at most this."""
MIN_FOLDS_FOR_SD: Final[int] = 4


def minimum_detectable_effect(
    *, rows: dict[str, pl.DataFrame]
) -> dict[tuple[str, str], tuple[float, float]]:
    """Print and return the minimum detectable effect per group and row set.

    The effect is 2 times the standard deviation over folds of the group's CRPS difference between
    `ctrl` (seeds 0 to 2) and `ctrl_s3` (seeds 3 to 5). That figure holds for contrasts of
    three-seed means (C4). Contrasts of six-seed means (C1 to C3) use it divided by the square root
    of two. It ignores the interaction of folds with seed sets, so it is a lower bound.

    Args:
        rows: The scored rows of each arm, keyed by arm label (`R_s012_w115` and `R_s345_w115` are
            read).

    Returns:
        For each (group, row set) with at least two folds: (three-seed figure, six-seed figure).
    """
    print(
        "\n== minimum detectable effect per group and row set, days 1 to 10 (before any contrast) =="
    )
    out: dict[tuple[str, str], tuple[float, float]] = {}
    if "R_s012_w115" not in rows or "R_s345_w115" not in rows:
        print("not computed: R_s012 or R_s345 is not in this pass")
        return out
    for set_name in ("all rows", "bank-holiday weeks"):
        for group, members in SUBSETS.items():
            gap = fold_gap(
                first=rows["R_s012_w115"],
                second=rows["R_s345_w115"],
                where=ROW_SETS[set_name],
                members=members,
            )
            n = gap.height
            if n < 2:
                print(f"{set_name}, {group}: {n} folds with rows; cannot be estimated")
                continue
            sd = float(np.std(gap["difference"].to_numpy(), ddof=1))
            out[group, set_name] = (2 * sd, 2 * sd / np.sqrt(2))
            caveat = (
                f" CAVEAT: the SD rests on only {n - 1} degrees of freedom."
                if n < MIN_FOLDS_FOR_SD
                else ""
            )
            print(
                f"{set_name}, {group}: {n} folds, SD {sd:.4f}; 2 x SD (three-seed means, C4) "
                f"{2 * sd:.4f}; divided by sqrt(2) (six-seed means, C1 to C3) "
                f"{2 * sd / np.sqrt(2):.4f}.{caveat}"
            )
    bank = out.get(("signed-MW six", "bank-holiday weeks"))
    if bank is not None:
        testable = "yes" if bank[1] <= P1_MDE_LIMIT else "no"
        print(
            f"\nP1 testable: {testable} (MDE {bank[1]:.3f} vs {P1_MDE_LIMIT:.2f}, signed-MW six, "
            "bank-holiday-week rows, six-seed means)"
        )
    return out


def interval_line(
    *,
    label: str,
    group: str,
    treatment: pl.DataFrame,
    reference: pl.DataFrame,
    where: pl.Expr | None,
    level_percent: float,
    mde: float | None,
) -> None:
    """Print one contrast: difference, interval, half-hour count, and a verdict against the MDE."""
    members = SUBSETS[group]
    if group == REPORT_ONLY_GROUP:
        group = f"{group} (report-only)"
    keep = pl.col("label").is_in(list(members)) & pl.col(HEADLINE)
    if where is not None:
        keep &= where
    try:
        r = metrics.month_bootstrap_contrast(
            treatment=treatment,
            reference=reference,
            metric="crps",
            where=keep,
            level_percent=level_percent,
            n_resamples=N_RESAMPLES,
        )
    except ValueError as error:
        print(f"{label:42s} {group:15s} unavailable: {error}")
        return
    verdict = ""
    if mde is not None:
        verdict = "inside the noise" if abs(r["difference"]) < mde else "larger than the MDE"
    print(
        f"{label:42s} {group:15s} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) "
        f"level {r['level_percent']:.2f}% n={r['n_half_hours']} {verdict}"
    )


def absolute_tables(*, rows: dict[str, pl.DataFrame]) -> None:
    """Print group CRPS on each row set for every widened arm, and per Primary on all rows."""
    widened = {n: r for n, r in rows.items() if n.endswith("_w115")}
    for set_name, where in ROW_SETS.items():
        print(f"\n== CRPS at days 1 to 10, {set_name} (equal weight per Primary, points) ==")
        for group, members in SUBSETS.items():
            print(
                group,
                {
                    n: round(group_crps(rows=r, members=members, where=where), 3)
                    for n, r in widened.items()
                },
            )
    print("\n== CRPS at days 1 to 10 per Primary, all rows ==")
    table = pl.concat(
        [
            r.filter(pl.col(HEADLINE))
            .group_by("label")
            .agg(pl.col("crps").mean())
            .with_columns(arm=pl.lit(n))
            for n, r in widened.items()
        ]
    ).pivot(on="arm", index="label", values="crps")
    with pl.Config(tbl_rows=40, tbl_cols=30):
        print(table.sort(pl.col("label").str.slice(1).cast(pl.Int32)))


def tails_table(*, rows: dict[str, pl.DataFrame]) -> None:
    """Print coverage and pinball loss at the four outer levels, raw and widened, by row set."""
    for set_name in ("all rows", "bank-holiday weeks"):
        where = ROW_SETS[set_name]
        out = []
        for arm, frame in rows.items():
            if not (arm.endswith("_raw") or arm.removesuffix("_w115") in RAW_TAIL_ARMS):
                continue
            keep = pl.col(HEADLINE) if where is None else pl.col(HEADLINE) & where
            part = frame.filter(keep)
            for group, members in SUBSETS.items():
                sub = part.filter(pl.col("label").is_in(list(members)))
                if sub.is_empty():
                    continue
                per_primary = sub.group_by("label").agg(
                    *[pl.col(f"coverage_{lv}").mean() for lv in OUTER_LEVELS],
                    *[pl.col(f"pinball_{lv}").mean() for lv in OUTER_LEVELS],
                )
                out.append(
                    {"arm": arm, "group": group, "n_half_hours": sub.height}
                    | {
                        c: round(float(np.nanmean(per_primary[c].to_numpy())), 3)
                        for c in per_primary.columns
                        if c != "label"
                    }
                )
        with pl.Config(tbl_rows=200, tbl_cols=20):
            print(
                f"\n== tails, {set_name} (coverage nominal q01 0.01, q05 0.05, q95 0.95, q99 0.99) =="
            )
            print(pl.DataFrame(out).sort("group", "arm"))


def tested_contrasts(
    *, rows: dict[str, pl.DataFrame], mde: dict[tuple[str, str], tuple[float, float]]
) -> None:
    """Print C1 to C4 per group at the Bonferroni-corrected level, and C3's sign per fold."""
    print(
        f"\n== tested contrasts C1 to C4 (treatment minus reference, CRPS days 1 to 10, "
        f"month-block bootstrap, {N_RESAMPLES:,} resamples, level {BONFERRONI_LEVEL_PERCENT:.2f}%; "
        "D15 and D18 report-only; the five screen folds are interim) ==\n"
        "Note: month-block bootstrap intervals with 3 to 5 blocks (folds) are close to the minimum "
        "and maximum of the fold differences, so read them as a range, not a 95% interval."
    )
    for label, (treatment, reference, set_name, n_seeds) in CONTRASTS.items():
        if f"{treatment}_w115" not in rows or f"{reference}_w115" not in rows:
            print(f"{label}: skipped, an arm is not complete")
            continue
        for group in SUBSETS:
            figures = mde.get((group, set_name))
            interval_line(
                label=label + f" [{set_name}]",
                group=group,
                treatment=rows[f"{treatment}_w115"],
                reference=rows[f"{reference}_w115"],
                where=ROW_SETS[set_name],
                level_percent=BONFERRONI_LEVEL_PERCENT,
                mde=None if figures is None else figures[0 if n_seeds == 3 else 1],
            )
    if "E1_w115" in rows and "E1n_w115" in rows:
        gap = fold_gap(
            first=rows["E1_w115"], second=rows["E1n_w115"], where=ROW_SETS["bank-holiday weeks"]
        )
        print(
            "\n== C3 by fold: E1 minus E1n, bank-holiday-week rows, signed-MW six (negative favours E1) =="
        )
        print(gap.rename({"first": "E1", "second": "E1n"}))


def descriptive_contrasts(*, rows: dict[str, pl.DataFrame]) -> None:
    """Print the contrasts outside the correction, with uncorrected 95% intervals."""
    print("\n== descriptive contrasts (not tested, uncorrected 95% intervals; read as a sketch) ==")
    for label, treatment, reference, set_name in DESCRIPTIVE_CONTRASTS:
        if f"{treatment}_w115" not in rows or f"{reference}_w115" not in rows:
            continue
        for group in SUBSETS:
            interval_line(
                label=label,
                group=group,
                treatment=rows[f"{treatment}_w115"],
                reference=rows[f"{reference}_w115"],
                where=ROW_SETS[set_name],
                level_percent=DESCRIPTIVE_LEVEL_PERCENT,
                mde=None,
            )


def median_minus_mean_check(
    *, rows: dict[str, pl.DataFrame], frames: dict[str, pl.DataFrame]
) -> None:
    """Print, per site, the median's mean absolute error minus the mean forecast's, in points.

    This is a check defined here, not a stored statistic. The mean of a forecast distribution is
    approximated by the sum of its 11 raw (unwidened) quantiles weighted by the probability each
    stands for (`LEVEL_MASSES`). A negative value means the median forecast is the better point
    forecast; the audit reads it beside each contrast.
    """
    print(
        "\n== audit check: MAE of the median minus MAE of the mean forecast, days 1 to 10, points =="
    )
    weights = pl.Series(LEVEL_MASSES.tolist())
    out = []
    for name in ("R_raw", "E0_raw", "E1n_raw", "E1_raw", "E2_raw", "E3_raw"):
        if name not in rows:
            continue
        quantiles = frames[name].select(*KEYS, *common.LEVEL_COLUMNS)
        joined = rows[name].filter(pl.col(HEADLINE)).join(quantiles, on=KEYS, validate="1:1")
        mean_mw = pl.sum_horizontal(
            *[pl.col(c) * w for c, w in zip(common.LEVEL_COLUMNS, weights, strict=True)]
        )
        scale = 100.0 / pl.col("capacity_mw")
        per_site = (
            joined.group_by("label")
            .agg(
                (pl.col("actual_points") - pl.col("q50") * scale).abs().mean().alias("mae_median"),
                (pl.col("actual_points") - mean_mw * scale).abs().mean().alias("mae_mean"),
            )
            .with_columns(
                arm=pl.lit(name), median_minus_mean=pl.col("mae_median") - pl.col("mae_mean")
            )
        )
        out.append(per_site)
    if out:
        table = pl.concat(out).pivot(on="arm", index="label", values="median_minus_mean")
        with pl.Config(tbl_rows=40):
            print(table.sort(pl.col("label").str.slice(1).cast(pl.Int32)))


def r0_check(*, series: pl.DataFrame, ctx: dict) -> None:
    """Print how closely R refitted through this code path (R0) matches the stored `ctrl`.

    R0 (seed 0, fold 2025-12) is compared with the stored seed 0 of `ctrl`, and the yardstick is
    the gap between the stored seeds 0 and 1 on the same fold.
    """
    r0_file = SEED_423 / "R0" / "2025-12_s0.parquet"
    print("\n== R0 reproduction check (fold 2025-12, seed 0) ==")
    if not r0_file.exists():
        print("R0 not fitted for 2025-12: check skipped")
        return
    months = ("2025-12",)
    defs = (
        ArmDef("R0", ((SEED_423 / "R0", (0,)),), months=months),
        ArmDef("ctrl_s0", ((STORED_CTRL, (0,)),), months=months),
        ArmDef("ctrl_s1", ((STORED_CTRL, (1,)),), months=months),
    )
    frames = load_frames(series=series, arm_defs=defs, raw_arms=())
    check_keys(frames=frames, reference="R0_w115")
    _, _, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    for group, members in SUBSETS.items():
        values = {
            n: round(group_crps(rows=r, members=members, where=None), 3) for n, r in rows.items()
        }
        print(
            f"{group}: {values}; R0 minus stored seed 0 "
            f"{values['R0_w115'] - values['ctrl_s0_w115']:+.3f}; seed 1 minus seed 0 "
            f"{values['ctrl_s1_w115'] - values['ctrl_s0_w115']:+.3f}"
        )


def event_fold(*, series: pl.DataFrame, ctx: dict) -> None:
    """Print point estimates on the Easter fold 2026-04, with R0 as the reference."""
    print(
        f"\n== event fold {EVENT_MONTH} (point estimates only; reference is R0, one seed vs the encoder arms' seed means) =="
    )
    months = (EVENT_MONTH,)
    defs = (
        ArmDef("R0", ((SEED_423 / "R0", (0,)),), months=months),
        ArmDef("E0", ((SEED_423 / "E0", SEEDS_0_TO_2),), months=months),
        ArmDef("E1n", ((SEED_423 / "E1n", SEEDS_0_TO_5),), months=months),
        ArmDef("E1", ((SEED_423 / "E1", SEEDS_0_TO_5),), months=months),
        ArmDef("E2", ((SEED_423 / "E2", SEEDS_0_TO_5),), months=months),
        ArmDef("E3", ((SEED_423 / "E3", SEEDS_0_TO_5),), months=months),
    )
    present = tuple(d for d in defs if d.complete())
    if not any(d.name != "R0" for d in present):
        print(f"no encoder arm is fitted for {EVENT_MONTH}: skipped")
        return
    frames = load_frames(series=series, arm_defs=present, raw_arms=())
    check_keys(frames=frames, reference="R0_w115")
    _, _, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    rows = {n: feats.add_row_sets(frame=r) for n, r in rows.items()}
    for set_name, where in ROW_SETS.items():
        for group, members in SUBSETS.items():
            values = {
                n: round(group_crps(rows=r, members=members, where=where), 3)
                for n, r in rows.items()
            }
            print(f"{set_name}, {group}: {values}")


def main() -> None:
    """Print the checks, the minimum detectable effect, the tables, and the contrasts."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    flagged = pl.read_parquet(s407.FLAGGED_KEYS).select("time_series_id", "valid_time")
    before = ctx["actuals"].height
    ctx["actuals"] = ctx["actuals"].join(flagged, on=["time_series_id", "valid_time"], how="anti")
    print(f"scoring actuals dropped for a flagged exact zero: {before - ctx['actuals'].height}")
    frames = load_frames(series=series, arm_defs=ARM_DEFS, raw_arms=RAW_TAIL_ARMS)
    if f"{REFERENCE}_w115" not in frames:
        msg = f"the reference {REFERENCE} (407 ctrl seeds 0 to 2 and 419 ctrl_s3 seeds 3 to 5) is incomplete"
        raise ValueError(msg)
    check_keys(frames=frames, reference=f"{REFERENCE}_w115")
    _, _, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    rows = {n: feats.add_row_sets(frame=r) for n, r in rows.items()}
    mde = minimum_detectable_effect(rows=rows)
    r0_check(series=series, ctx=ctx)
    absolute_tables(rows=rows)
    tails_table(rows=rows)
    tested_contrasts(rows=rows, mde=mde)
    descriptive_contrasts(rows=rows)
    median_minus_mean_check(rows=rows, frames=frames)
    event_fold(series=series, ctx=ctx)


if __name__ == "__main__":
    main()
