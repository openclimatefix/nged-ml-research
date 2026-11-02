r"""Score experiment 503: the gap-trained forecast and its serving rules against the stored arms.

Arms, all on the half-hours every arm shares (finite observed actuals with flagged zeros dropped,
valid time before 2026-07-01, lead up to 240 hours):

- `a`: experiment 402's `r0_lagged`; `b`: experiment 392's `r0_lagless`; `arep`: experiment 502's
  CUDA repeat of (a), which gives the noise floor.
- `g`: the gap-trained forecast with no rule; `ga`: G-A (rules zero and stuck); `gb`: G-B (zero,
  far, and stuck).
- `ar1`, `ar2`: the (a-rule) arms: arm (a)'s boosters with the same two rule sets.
- `f`: the fixed 50/50 mixture of the probability distributions of (a) and (b) (experiment 402's
  algebra, `e00402_mixture.Pair`).

Row sets: `outside` (no labelled event: not inside an event and not within 48 hours after one),
`harm` (a lag lies in an unpadded corrected event and the target lies outside every event: the
tags of experiment 395 Part F), `during`, and `after7d`. Real-event rows use the metered readings,
which are not corrected for the normal running arrangement. Intervals on `outside` are month
blocks (`metrics.month_bootstrap_contrast`); on `harm` they resample whole events. The 12
pre-registered contrasts are g, ga, and gb each minus a and minus b, on `outside` and `harm`.
Run one widening per call (`--widening 1.0` or `1.15`) to bound memory.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\\
ml-experiments/00392-xgboost-nra-load-shift-masked-targets:\\
ml-experiments/00395-xgboost-level-state-winter-spread-anchor-stored-forecasts:\\
ml-experiments/00402-lag-gate-learned-weight-injected-load-shifts-part-1:\\
ml-experiments/00503-gap-trained-xgboost-lags-with-a-bad-reading-rule \\
    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00503-*/e00503_score.py
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00392_score as s392  # ty: ignore[unresolved-import]
import e00395_part_f_build as pf  # ty: ignore[unresolved-import]
import e00402_mixture as mixture  # ty: ignore[unresolved-import]
import e00503_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = fit.WORK_DIR / "scores"
WORK_402: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00402-lag-gate-learned-weight-injected-load-shifts-part-1" / "work"
)
SOURCES: Final[dict[str, Path]] = {
    "a": WORK_402 / "arm_preds" / "r0_lagged",
    "b": ML_EXPERIMENTS_DATA_DIR
    / "00392-xgboost-nra-load-shift-masked-targets"
    / "work"
    / "arm_preds"
    / "r0_lagless",
    "arep": ML_EXPERIMENTS_DATA_DIR
    / "00502-leave-one-lag-out-power-lag-ablation-xgboost-demand"
    / "work"
    / "arm_preds"
    / "a_rep",
    "g": fit.ARM_DIR / "g_gap",
    "ga": fit.ARM_DIR / "g_iii",
    "gb": fit.ARM_DIR / "g_iii2",
    "ar1": fit.ARM_DIR / "a_iii",
    "ar2": fit.ARM_DIR / "a_iii2",
}
ARM_ORDER: Final[tuple[str, ...]] = ("a", "arep", "b", "f", "g", "ga", "gb", "ar1", "ar2")
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    *((arm, control) for arm in ("g", "ga", "gb") for control in ("a", "b")),
)
"""The pre-registered contrasts, on `outside` and `harm` rows: 6 x 2 = 12."""
DESCRIPTIVE: Final[tuple[tuple[str, str], ...]] = (
    ("arep", "a"),
    ("ar1", "a"),
    ("ar2", "a"),
    ("f", "a"),
    ("g", "f"),
    ("ga", "f"),
    ("gb", "f"),
    ("gb", "ga"),
)
MAX_LEAD_HOURS: Final[float] = 240.0
BANDS: Final = {
    "0-240 h": pl.col("lead_hours") <= MAX_LEAD_HOURS,
    "12-36 h": (pl.col("lead_hours") >= 12.0) & (pl.col("lead_hours") <= 36.0),
}
ROW_SETS: Final[dict[str, pl.Expr]] = {
    "outside": s392.ROW_SETS["outside"],
    "harm": pl.col("harm"),
    "during": s392.ROW_SETS["during"],
    "after7d": s392.ROW_SETS["after7d"],
}
SIX: Final = s392.GROUP_VIEWS["signed-MW six"]
BONFERRONI_CONTRASTS: Final[int] = 12
LEVEL_PERCENT: Final[float] = 100.0 * (1.0 - 0.05 / BONFERRONI_CONTRASTS)
RESAMPLES: Final[int] = 20_000
METRIC_COLUMNS: Final[tuple[str, ...]] = ("crps", *s392.LOSS_COLUMNS[1:])
EQUIVALENCE_LIMIT: Final[float] = 0.11


def mix_frames(*, first: pl.DataFrame, second: pl.DataFrame) -> pl.DataFrame:
    """Return the 50/50 mixture of two forecasts' distributions, aligned on the keys."""
    keys = list(s392.KEY_COLUMNS)
    q = list(s392.QUANTILES)
    joined = first.sort(*keys).join(
        second.sort(*keys).select(*keys, *[pl.col(c).alias(f"{c}_2") for c in q]),
        on=keys,
        how="inner",
        validate="1:1",
    )
    pieces = []
    for start in range(0, joined.height, 200_000):
        part = joined.slice(start, 200_000)
        pair = mixture.Pair(
            q1=part.select(q).to_numpy(), q2=part.select([f"{c}_2" for c in q]).to_numpy()
        )
        weights = np.full(part.height, 0.5)
        pieces.append(
            part.select(*keys).hstack(
                pl.DataFrame(pair.quantiles(w=weights), schema=q, orient="row")
            )
        )
    return pl.concat(pieces)


def event_interval(*, sums: np.ndarray, counts: np.ndarray, level: float) -> tuple[float, ...]:
    """Return the pooled mean and an interval from resampling whole events."""
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(sums), size=(RESAMPLES, len(sums)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (100 - level) / 2
    return (
        float(sums.sum() / counts.sum()),
        float(np.percentile(means, tail)),
        float(np.percentile(means, 100 - tail)),
        float(len(sums)),
    )


def harm_contrast(*, treatment: pl.DataFrame, reference: pl.DataFrame, level: float) -> tuple:
    """Return the harm-row CRPS difference with a whole-event interval and the leave-outs.

    Leave-one-Primary-out reports the range of the pooled difference when each of the six Primaries
    is dropped in turn, and the largest event is dropped in a last line.
    """
    keys = [*s392.KEY_COLUMNS, "lag_event_id", "label"]
    joined = treatment.select(*keys, "crps").join(
        reference.select(*s392.KEY_COLUMNS, reference=pl.col("crps")),
        on=list(s392.KEY_COLUMNS),
        validate="1:1",
    )
    per = (
        joined.with_columns(diff=pl.col("crps") - pl.col("reference"))
        .group_by("lag_event_id", "label")
        .agg(s=pl.col("diff").sum(), n=pl.len())
    )
    result = event_interval(sums=per["s"].to_numpy(), counts=per["n"].to_numpy(), level=level)
    leave_out = [
        float(sub["s"].sum()) / float(sub["n"].sum())
        for label in per["label"].unique().to_list()
        if (sub := per.filter(pl.col("label") != label)).height
    ]
    largest = per.sort("n", descending=True).slice(1)
    return (
        result,
        (min(leave_out), max(leave_out)),
        float(largest["s"].sum()) / float(largest["n"].sum()),
    )


def main() -> None:  # noqa: PLR0912, PLR0915
    """Print the scores and write the per-row table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widening", type=float, default=1.0)
    parser.add_argument(
        "--skip", nargs="*", default=[], help="arms to leave out (for example arep)"
    )
    args = parser.parse_args()
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    OUT.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table(labels=adapter.PRIMARY_LABELS)
    sites = [s for s in series["label"].to_list() if s in fit.SIGNED_MW_LABELS]
    series = series.filter(pl.col("label").is_in(sites))
    actuals, cache_actuals, _ = s392.load_actuals(source="observed", series=series)
    s392.print_dropped(cache_actuals=cache_actuals, actuals=actuals, series=series)
    thresholds = metrics.twcrps_thresholds(
        history=cache_actuals.rename({"actual": "power_mw"}),
        series=series,
        before=s392.SCORED_FIRST,
    )
    flags, _ = s392.event_tables(series=series)
    events = pl.read_parquet(pf.WORK / "events_unpadded.parquet")
    arms = [a for a in SOURCES if a not in args.skip]
    scored: list[pl.DataFrame] = []
    for fold, name in enumerate(s392.FOLD_NAMES):
        stored = {}
        for arm in arms:
            frame = s392.read_fold(directory=SOURCES[arm], sites=sites, fold=fold)
            if frame is None:
                msg = f"fold {name}: arm {arm} is incomplete"
                raise ValueError(msg)
            stored[arm] = adapter.forecast_frame(
                stored=frame, series=series, unit="fraction"
            ).filter(s392.lead_expression() <= MAX_LEAD_HOURS)
        stored["f"] = mix_frames(first=stored["a"], second=stored["b"])
        finite = actuals.filter(pl.col("valid_time") < metrics.HOLDOUT_START)
        shared = None
        for frame in stored.values():
            keys = frame.select(*s392.KEY_COLUMNS).join(
                finite.select("time_series_id", "valid_time"), on=["time_series_id", "valid_time"]
            )
            shared = keys if shared is None else shared.join(keys, on=list(s392.KEY_COLUMNS))
        assert shared is not None
        print(f"fold {name}: shared half-hours {shared.height:,}")
        tags = pf.tag_rows(keys=shared.select(*s392.KEY_COLUMNS), events=events)
        for arm, frame in stored.items():
            widened = s392.widen(
                frame=frame.join(shared, on=list(s392.KEY_COLUMNS), how="semi"),
                factor=args.widening,
            )
            scored.append(
                s392.score_arm(
                    forecast=widened,
                    actuals=actuals,
                    series=series,
                    thresholds=thresholds,
                    flags=flags,
                )
                .join(tags, on=list(s392.KEY_COLUMNS), how="left", validate="1:1")
                .with_columns(
                    arm=pl.lit(arm),
                    fold=pl.lit(fold, dtype=pl.Int64),
                    harm=(pl.col("lag_count") > 0) & (pl.col("target_event_id") < 0),
                )
            )
    rows = pl.concat(scored)
    rows.write_parquet(OUT / f"rows_w{args.widening}.parquet")
    print(f"\n{'=' * 90}\nWIDENING {args.widening}\n{'=' * 90}")
    arm_order = {a: i for i, a in enumerate(ARM_ORDER)}
    for band, band_cond in BANDS.items():
        for row_set, row_cond in ROW_SETS.items():
            cond = band_cond & SIX & row_cond
            print(f"\n-- absolute scores: signed-MW six, {row_set}, lead {band}")
            print(
                s392.means(rows=rows, where=cond, by=["arm"], columns=METRIC_COLUMNS)
                .with_columns(pl.col("arm").replace_strict(arm_order, default=99).alias("o"))
                .sort("o")
                .drop("o")
            )
    for band, band_cond in BANDS.items():
        print(f"\n== MDE: arm (a) minus its repeat (arep), lead {band} ==")
        if "arep" in arms:
            cond = band_cond & SIX & ROW_SETS["outside"]
            print(
                metrics.month_bootstrap_contrast(
                    treatment=rows.filter(pl.col("arm") == "arep"),
                    reference=rows.filter(pl.col("arm") == "a"),
                    metric="crps",
                    where=cond,
                    level_percent=LEVEL_PERCENT,
                    n_resamples=RESAMPLES,
                )
            )
    for pre_registered, pairs in (("pre-registered", CONTRASTS), ("descriptive", DESCRIPTIVE)):
        for arm, control in pairs:
            if (arm not in arms and arm != "f") or (control not in arms and control != "f"):
                continue
            for band, band_cond in BANDS.items():
                for row_set in ("outside", "harm", "during", "after7d"):
                    cond = band_cond & SIX & ROW_SETS[row_set]
                    print(f"\n[{pre_registered}] ({arm}) - ({control}), {row_set}, lead {band}")
                    print(
                        s392.contrast_table(rows=rows, arm=arm, control=control, where=cond, by=[])
                    )
                    treatment = rows.filter((pl.col("arm") == arm) & cond)
                    reference = rows.filter((pl.col("arm") == control) & cond)
                    if row_set == "harm":
                        if treatment.height:
                            print(
                                "harm (difference, [low, high], events), leave-one-Primary-out "
                                "range, largest event dropped:",
                                harm_contrast(
                                    treatment=treatment, reference=reference, level=LEVEL_PERCENT
                                ),
                            )
                    else:
                        print(
                            metrics.month_bootstrap_contrast(
                                treatment=rows.filter(pl.col("arm") == arm),
                                reference=rows.filter(pl.col("arm") == control),
                                metric="crps",
                                where=cond,
                                level_percent=LEVEL_PERCENT,
                                n_resamples=RESAMPLES,
                            )
                        )
    fire = pl.concat([pl.read_parquet(p) for p in sorted((fit.ARM_DIR / "fire").glob("*.parquet"))])
    print(
        "\n== Rule firing on real forecast keys: "
        "share of keys where a rule sets at least one lag =="
    )
    on_g = rows.filter(pl.col("arm") == "g").join(
        fire, on=list(s392.KEY_COLUMNS), how="left", validate="1:1"
    )
    for row_set, row_cond in ROW_SETS.items():
        sub = on_g.filter(SIX & row_cond & (pl.col("lead_hours") <= MAX_LEAD_HOURS))
        print(
            row_set,
            sub.select(
                n=pl.len(),
                zero=(pl.col("fire_zero") > 0).mean(),
                far=(pl.col("fire_far") > 0).mean(),
                stuck=(pl.col("fire_stuck") > 0).mean(),
                gap=(pl.col("fire_table_gap") > 0).mean(),
            ).row(0),
        )
    print("\n-- by Primary, rows with no labelled event --")
    print(
        on_g.filter(SIX & ROW_SETS["outside"])
        .group_by("label")
        .agg(
            n=pl.len(),
            zero=(pl.col("fire_zero") > 0).mean(),
            far=(pl.col("fire_far") > 0).mean(),
            stuck=(pl.col("fire_stuck") > 0).mean(),
            gap=(pl.col("fire_table_gap") > 0).mean(),
        )
        .sort("label")
    )
    print("\n-- CRPS where a rule fires and where none fires (rows with no labelled event) --")
    fired = on_g.select(
        *s392.KEY_COLUMNS,
        any_fire=(pl.col("fire_zero") + pl.col("fire_far") + pl.col("fire_stuck")) > 0,
    )
    tagged = rows.join(fired, on=list(s392.KEY_COLUMNS), how="left", validate="m:1")
    print(
        s392.means(
            rows=tagged, where=SIX & ROW_SETS["outside"], by=["arm", "any_fire"], columns=("crps",)
        ).sort("any_fire", "arm")
    )


if __name__ == "__main__":
    main()
