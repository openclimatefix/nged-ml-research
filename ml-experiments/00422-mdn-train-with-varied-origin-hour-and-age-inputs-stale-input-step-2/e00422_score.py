"""Score the stale-input forecasts of experiment 422 (no fit).

For each training arm the script averages the three seeds in quantile space, applies the 1.15
widening, and scores each test arm against the planned references on their shared half-hours, with
the month-block 95% bootstrap of experiment 410's scorer:

- C1: a test arm minus the same training arm's FRESH forecast (experiment 410's contrast).
- C2: a training arm minus T0 in the same test arm.
- C3 is C2 at the FRESH test arm.

The stored forecasts of experiment 410 (`S397`, the unmodified network trained at 09:00 on
unfiltered power, and `S1M`, the XGBoost model) are scored for C1 on the arms they hold.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00410-mdn-stale-weather-and-delayed-power-rescore:\
ml-experiments/00422-mdn-train-with-varied-origin-hour-and-age-inputs-stale-input-step-2 \
OMP_NUM_THREADS=2 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00422-*/e00422_score.py
```
"""

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00410_stale_predict as stale  # ty: ignore[unresolved-import]
import e00410_stale_score as score410  # ty: ignore[unresolved-import]
import e00422_predict as predict422  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics

HEADLINE: Final[str] = score410.HEADLINE
GROUPS: Final[dict[str, tuple[str, ...]]] = score410.GROUPS
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
OLD_REFERENCES: Final[dict[str, tuple[str, str]]] = {
    "S397": ("pooled", "pooled"),
    "S1M": ("xgb_s1m", "xgb_s1m"),
}
"""Stored forecasts of experiment 410: label, then (folder under 410's `preds`, seed layout)."""
RESAMPLES: Final[int] = score410.RESAMPLES


def seed_average(*, folder: Path, months: list[str], seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return a folder's seed-averaged forecast (fraction of capacity) over the months, sorted."""
    keys = list(common.KEYS)
    pieces = []
    for month in months:
        files = [folder / f"{month}_s{s}.parquet" for s in seeds]
        stack = pl.concat([pl.read_parquet(f) for f in files])
        pieces.append(stack.group_by(keys).agg(*[pl.col(c).mean() for c in common.LEVEL_COLUMNS]))
    frame = pl.concat(pieces).sort(keys)
    values = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    return frame.with_columns(
        *[pl.Series(c, values[:, i]).cast(pl.Float32) for i, c in enumerate(common.LEVEL_COLUMNS)]
    )


def forecast(
    *, source: str, test_arm: str, months: list[str], series: pl.DataFrame
) -> pl.DataFrame:
    """Return the widened forecast of a training arm (or a stored reference) under a test arm."""
    if source in OLD_REFERENCES:
        folder_name, _ = OLD_REFERENCES[source]
        seeds = (0,) if source == "S1M" else SEEDS
        folder = stale.OUT_DIR / "preds" / folder_name / test_arm
    else:
        seeds = SEEDS
        folder = predict422.OUT_DIR / "preds" / source / test_arm
    stored = seed_average(folder=folder, months=months, seeds=seeds)
    return score410.widen(
        frame=adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    )


def contrast_records(
    *,
    name: str,
    kind: str,
    treatment: pl.DataFrame,
    reference: pl.DataFrame,
    series: pl.DataFrame,
    ctx: dict,
) -> list[dict]:
    """Return one record per group for treatment minus reference."""
    table, scored, rows = s381.score_arms(
        arms={"ref": reference, "trt": treatment}, series=series, ctx=ctx
    )
    records = []
    for group, labels in GROUPS.items():
        where = pl.col("label").is_in(list(labels)) & pl.col(HEADLINE)
        r = metrics.month_bootstrap_contrast(
            treatment=rows["trt"],
            reference=rows["ref"],
            metric="crps",
            where=where,
            level_percent=95.0,
            n_resamples=RESAMPLES,
            slice_table=table,
        )
        record = {
            "contrast": name,
            "kind": kind,
            "group": group,
            "n_half_hours": r["n_half_hours"],
            "crps_ref": score410.group_mean(scored=scored["ref"], metric="crps", labels=labels),
            "crps_trt": score410.group_mean(scored=scored["trt"], metric="crps", labels=labels),
            "diff": r["difference"],
            "lower": r["lower"],
            "upper": r["upper"],
        }
        for tag, frame in (("ref", reference), ("trt", treatment)):
            for key, value in score410.widths(
                frame=frame, series=series, table=table, labels=labels
            ).items():
                record[f"{key}_{tag}"] = value
            for metric in (
                "coverage_q95",
                "coverage_q99",
                "coverage_q01",
                "coverage_q05",
                "pinball_q01",
                "pinball_q05",
                "pinball_q95",
                "pinball_q99",
            ):
                record[f"{metric}_{tag}"] = score410.group_mean(
                    scored=scored["ref" if tag == "ref" else "trt"], metric=metric, labels=labels
                )
        records.append(record)
    return records


def main() -> None:
    """Score every contrast and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(prep.SCREEN_MONTHS))
    parser.add_argument("--train-arms", nargs="+", default=[a.name for a in prep422.TRAIN_ARMS])
    parser.add_argument(
        "--test-arms",
        nargs="+",
        default=[a.name for a in predict422.TEST_ARMS if a.name != "FRESH"],
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--no-old", action="store_true")
    args = parser.parse_args()
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    flagged = pl.read_parquet(stale.FLAGGED_KEYS).select("time_series_id", "valid_time")
    for month in args.months:
        origins = prep.test_origins(month=month)
        window = flagged.filter(
            pl.col("valid_time").is_between(origins[0], origins[-1] + timedelta(days=15))
        )
        count = ctx["actuals"].join(window, on=["time_series_id", "valid_time"], how="semi").height
        print(f"{month}: flagged substation_zero half-hours dropped from the actuals: {count}")
    ctx["actuals"] = ctx["actuals"].join(flagged, on=["time_series_id", "valid_time"], how="anti")
    cache: dict[tuple[str, str], pl.DataFrame] = {}

    def get(source: str, test_arm: str) -> pl.DataFrame:
        if (source, test_arm) not in cache:
            cache[source, test_arm] = forecast(
                source=source, test_arm=test_arm, months=args.months, series=series
            )
        return cache[source, test_arm]

    records: list[dict] = []
    for source in [*args.train_arms, *([] if args.no_old else OLD_REFERENCES)]:
        for test_arm in args.test_arms:
            try:
                trt = get(source, test_arm)
                ref = get(source, "FRESH")
            except FileNotFoundError, pl.exceptions.ComputeError:
                continue
            records += [
                {"source": source, "test_arm": test_arm, **r}
                for r in contrast_records(
                    name=f"{source}:{test_arm}-FRESH",
                    kind="C1",
                    treatment=trt,
                    reference=ref,
                    series=series,
                    ctx=ctx,
                )
            ]
            print(f"C1 {source} {test_arm}", flush=True)
    for source in args.train_arms:
        if source == "T0":
            continue
        for test_arm in ["FRESH", *args.test_arms]:
            records += [
                {"source": source, "test_arm": test_arm, **r}
                for r in contrast_records(
                    name=f"{source}-T0:{test_arm}",
                    kind="C3" if test_arm == "FRESH" else "C2",
                    treatment=get(source, test_arm),
                    reference=get("T0", test_arm),
                    series=series,
                    ctx=ctx,
                )
            ]
            print(f"C2/C3 {source} {test_arm}", flush=True)
    out = pl.DataFrame(records)
    target = predict422.OUT_DIR / "scores" / f"contrasts{args.tag}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(target)
    with pl.Config(tbl_rows=400, tbl_cols=12, tbl_width_chars=220):
        for group in ("signed-MW six", "MVA-only eight"):
            for kind in ("C1", "C2", "C3"):
                sub = out.filter((pl.col("group") == group) & (pl.col("kind") == kind))
                print(f"\n== {group}, {kind}: CRPS days 1 to 10 (points), interim ==")
                print(
                    sub.select(
                        "source", "test_arm", "crps_ref", "crps_trt", "diff", "lower", "upper"
                    )
                )
    limits = pl.DataFrame(
        [
            {
                "source": r["source"],
                "test_arm": r["test_arm"],
                "group": r["group"],
                "limit": score410.THRESHOLDS[r["test_arm"]],
                "loss": r["diff"],
                "upper": r["upper"],
                "cov95_fall": r["coverage_q95_ref"] - r["coverage_q95_trt"],
                "cov99_fall": r["coverage_q99_ref"] - r["coverage_q99_trt"],
            }
            for r in out.filter(
                (pl.col("kind") == "C1")
                & pl.col("test_arm").is_in(list(score410.THRESHOLDS))
                & pl.col("group").is_in(["signed-MW six", "MVA-only eight"])
            ).iter_rows(named=True)
        ]
    ).with_columns(
        passes=(pl.col("loss") < pl.col("limit"))
        & (pl.max_horizontal("cov95_fall", "cov99_fall") < 0.02)
    )
    with pl.Config(tbl_rows=400, tbl_cols=12, tbl_width_chars=220):
        print("\n== experiment 410 limits: point estimate (upper bound beside it) ==")
        print(limits)


if __name__ == "__main__":
    main()
