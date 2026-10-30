# ruff: noqa: E501
"""Score experiment 416's arms and print the pre-registered contrasts.

Seed-averages each arm in quantile space, restricts every arm to the half-hours they share, adds a
`sunny_midday` slice (realised CAMS clear-sky index of 0.7 or more, local time 10:00 to 15:00), and
prints absolute scores, tails, per-Primary tails, and the contrasts with the month-block bootstrap
and the minimum detectable effect. `--pre-fit` prints only the minimum detectable effect of
experiment 409's stored arms (C3 minus A0-noirr-doy, seeds 0 to 2), which the plan commits before
any new fit.

```bash
export PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:ml-experiments/00403-pooled-versus-per-primary-mixture-density-network:ml-experiments/00416-mdn-explicit-solar-term-better-powered-follow-up
uv run --no-sync python ml-experiments/00416-*/e00416_score.py [--pre-fit]
```
"""

import argparse
import itertools
import json
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00403_rescore as r3  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.eval_protocol.metrics import ContrastInterval
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = DATA / "00416-mdn-explicit-solar-term-better-powered-follow-up" / "work"
WORK_409: Final[Path] = DATA / "00409-mdn-solar-term-day-of-year-and-anomaly-known-answer" / "work"
CAMS_FRAME_PATH: Final[Path] = (
    DATA / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
MONTHS: Final = r3.MONTHS
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
REUSED_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
REFERENCE: Final[str] = "A0-doy"
ARMS: Final[tuple[str, ...]] = ("A0-doy", "C3", "H1", "A0-noirr-doy")
FROM_409: Final[tuple[str, ...]] = ("C3", "A0-noirr-doy")
PRIMARY_ARMS: Final[tuple[str, ...]] = ("C3", "H1", "A0-noirr-doy")
SECONDARY: Final[tuple[tuple[str, str], ...]] = (("C3", "A0-noirr-doy"), ("H1", "C3"))
PRIMARY_METRICS: Final[tuple[str, ...]] = ("crps", "pinball_q99")
N_PRIMARY_TESTS: Final[int] = 6
LEVEL: Final[float] = 100.0 - 5.0 / N_PRIMARY_TESTS
"""The Bonferroni-corrected interval level in percent for a family-wise 95% across six tests."""
SUNNY_CSI: Final[float] = 0.7
MIDDAY_START_HOUR: Final[int] = 10
MIDDAY_END_HOUR: Final[int] = 15
NON_INFERIORITY_CRPS: Final[float] = 0.10
NON_INFERIORITY_Q99_RELATIVE: Final[float] = 0.05
SIGNED: Final = r3.SIGNED
SLICES: Final[tuple[str, ...]] = ("sunny_midday", "lead_days_1_10", "sunny_unit_shape")
TAIL_METRICS: Final[tuple[str, ...]] = (
    "pinball_q01",
    "pinball_q05",
    "pinball_q95",
    "pinball_q99",
    "coverage_q01",
    "coverage_q05",
    "coverage_q95",
    "coverage_q99",
    "twcrps",
)
RESULTS_PATH: Final[Path] = WORK / "score_results.json"


def seed_file(*, arm: str, month: str, seed: int) -> Path:
    """Return a seed's forecast file: experiment 409's for its reused seeds, else this experiment's."""
    reused = arm in FROM_409 and seed in REUSED_SEEDS
    return (WORK_409 if reused else WORK) / "seed_preds" / arm / f"{month}_s{seed}.parquet"


def arm_frame(*, arm: str, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return an arm's forecast (fraction of capacity) averaged over the seeds, all five months."""
    pieces = []
    for month in MONTHS:
        files = [seed_file(arm=arm, month=month, seed=s) for s in seeds]
        missing = [f for f in files if not f.exists()]
        assert not missing, f"missing forecasts: {missing}"
        stacked = pl.concat([pl.read_parquet(f) for f in files])
        counts = stacked.group_by(r3.KEYS).len()["len"].unique().to_list()
        assert counts == [len(files)], f"{arm} {month}: keys are not in every seed file: {counts}"
        pieces.append(
            stacked.group_by(r3.KEYS).agg(*[pl.col(c).mean() for c in common.LEVEL_COLUMNS])
        )
    frame = pl.concat(pieces).sort(r3.KEYS)
    values = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    return frame.with_columns(
        *[pl.Series(c, values[:, i]).cast(pl.Float32) for i, c in enumerate(common.LEVEL_COLUMNS)]
    )


def sunny_midday_flags(*, table: pl.DataFrame, series: pl.DataFrame) -> pl.DataFrame:
    """Return `table`'s keys with a `sunny_midday` flag from the realised CAMS clear-sky index.

    A valid time is the end of its half-hour, as in the CAMS frame's `time`. The flag is true when
    the half-hour starts at 10:00 or later and before 15:00 UK local time and CAMS global
    irradiance is at least `SUNNY_CSI` of the clear-sky value. Half-hours with no CAMS row, or with
    zero clear-sky irradiance, are false.
    """
    cams = (
        pl.read_parquet(
            CAMS_FRAME_PATH,
            columns=["site", "kind", "time", "ghi_u", "ghi_u_clear"],
        )
        .filter(pl.col("kind") == "substation", pl.col("ghi_u_clear") > 0)
        .select(
            label=pl.col("site"),
            valid_time=pl.col("time"),
            csi=pl.col("ghi_u") / pl.col("ghi_u_clear"),
        )
    )
    start = (pl.col("valid_time") - timedelta(minutes=30)).dt.convert_time_zone("Europe/London")
    keyed = (
        table.select("time_series_id", "power_fcst_init_time", "valid_time")
        .join(series.select("time_series_id", "label"), on="time_series_id", how="left")
        .join(cams, on=["label", "valid_time"], how="left")
        .with_columns(
            sunny_midday=(
                (pl.col("csi").is_finite() & (pl.col("csi") >= SUNNY_CSI))
                & (start.dt.hour() >= MIDDAY_START_HOUR)
                & (start.dt.hour() < MIDDAY_END_HOUR)
            ).fill_null(value=False)
        )
    )
    return table.join(
        keyed.select("time_series_id", "power_fcst_init_time", "valid_time", "sunny_midday"),
        on=["time_series_id", "power_fcst_init_time", "valid_time"],
        how="left",
        validate="1:1",
    )


def contrast(
    *, rows: dict, table: pl.DataFrame, a: str, b: str, sl: str, metric: str, level: float
) -> ContrastInterval:
    """Return the month-block contrast of arm `a` minus arm `b` at the signed-MW six."""
    where = pl.col("label").is_in(list(SIGNED)) & pl.col(sl) & pl.col("lead_days_1_10")
    return metrics.month_bootstrap_contrast(
        treatment=rows[a],
        reference=rows[b],
        metric=metric,
        where=where,
        level_percent=level,
        n_resamples=20_000,
        slice_table=table,
    )


def monthly_mean(*, rows: pl.DataFrame, sl: str, metric: str) -> dict:
    """Return the signed-six mean of a row metric in a slice (within days 1 to 10) per test month."""
    kept = rows.filter(
        pl.col("label").is_in(list(SIGNED)), pl.col(sl), pl.col("lead_days_1_10")
    ).with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
    return dict(kept.group_by("month").agg(pl.col(metric).mean()).iter_rows())


_PER_SEED: dict = {}


def per_seed_means(
    *, arm: str, seed: int, series: pl.DataFrame, ctx: dict, table: pl.DataFrame
) -> dict:
    """Return one seed's signed-six monthly mean of each metric in each slice, scored once."""
    if (arm, seed) not in _PER_SEED:
        frame = adapter.forecast_frame(
            stored=arm_frame(arm=arm, seeds=(seed,)), series=series, unit="fraction"
        )
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        rows = metrics.scored_rows(
            forecast=kept,
            actuals=ctx["actuals"],
            series=series,
            thresholds=ctx["thresholds"],
            slice_table=table,
        )
        _PER_SEED[arm, seed] = {
            (sl, metric): monthly_mean(rows=rows, sl=sl, metric=metric)
            for sl in SLICES
            for metric in PRIMARY_METRICS
        }
    return _PER_SEED[arm, seed]


def minimum_detectable_effect(
    *,
    a: str,
    b: str,
    sl: str,
    metric: str,
    seeds: tuple[int, ...],
    series: pl.DataFrame,
    ctx: dict,
    table: pl.DataFrame,
) -> tuple[float, float]:
    """Return two minimum detectable effects of the contrast `a - b`.

    The first is the pre-registered definition: 2 x the SD over folds of the difference between two
    single seeds' contrasts. The second is for the contrast as scored, the seed-average of
    `len(seeds)` seeds: the first divided by `sqrt(2 * len(seeds))`, because the SD of a difference
    of two single-seed contrasts is `sqrt(2)` times one seed's SD and averaging divides by
    `sqrt(len(seeds))`.
    """
    per_seed = {
        (arm, s): per_seed_means(arm=arm, seed=s, series=series, ctx=ctx, table=table)[sl, metric]
        for arm in (a, b)
        for s in seeds
    }
    sds = []
    for s1, s2 in itertools.combinations(seeds, 2):
        diff = [
            (per_seed[a, s1][m] - per_seed[b, s1][m]) - (per_seed[a, s2][m] - per_seed[b, s2][m])
            for m in per_seed[a, s1]
        ]
        sds.append(float(np.std(diff, ddof=1)))
    single = 2.0 * float(np.mean(sds))
    return single, single / float(np.sqrt(2 * len(seeds)))


def build(*, arms: tuple[str, ...], seeds: tuple[int, ...]) -> tuple:
    """Return the series, context, slice table (with `sunny_midday`), scores, and row losses."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    frames = {
        n: adapter.forecast_frame(
            stored=arm_frame(arm=n, seeds=seeds), series=series, unit="fraction"
        )
        for n in arms
    }
    table, scored, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    table = sunny_midday_flags(table=table, series=series)
    flags = table.select(*metrics.KEY_COLUMNS, "sunny_midday")
    rows = {
        n: r.join(flags, on=list(metrics.KEY_COLUMNS), how="left", validate="m:1")
        for n, r in rows.items()
    }
    return series, ctx, table, scored, rows


def pre_fit() -> None:
    """Print the minimum detectable effect from experiment 409's stored seeds 0 to 2."""
    arms = ("C3", "A0-noirr-doy")
    series, ctx, table, _, _ = build(arms=arms, seeds=REUSED_SEEDS)
    print(f"sunny-midday half-hours in the shared set: {int(table['sunny_midday'].sum())}")
    for sl in ("sunny_midday", "lead_days_1_10"):
        for metric in PRIMARY_METRICS:
            mde, mde_avg = minimum_detectable_effect(
                a="C3",
                b="A0-noirr-doy",
                sl=sl,
                metric=metric,
                seeds=REUSED_SEEDS,
                series=series,
                ctx=ctx,
                table=table,
            )
            print(
                f"pre-fit MDE, C3 minus A0-noirr-doy [{sl}, {metric}], seeds 0-2: "
                f"single-seed definition {mde:.3f}, 3-seed average {mde_avg:.3f}"
            )


def main() -> None:
    """Print the absolute scores, tails, and contrasts, and store them for the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-fit", action="store_true")
    args = parser.parse_args()
    if args.pre_fit:
        pre_fit()
        return
    series, ctx, table, scored, rows = build(arms=ARMS, seeds=SEEDS)
    out: dict = {"absolute": {}, "tails": {}, "primary": {}, "secondary": {}}
    print("\n== absolute scores, signed-MW six / MVA-only eight (days 1 to 10; points) ==")
    for n in ARMS:
        t = scored[n].filter(pl.col("metric") == "crps", pl.col("slice") == "lead_days_1_10")
        signed = t.filter(pl.col("label").is_in(list(SIGNED)))["value"].mean()
        mva = t.filter(pl.col("label").is_in(list(r3.MVA_ONLY)))["value"].mean()
        out["absolute"][n] = {"signed": signed, "mva": mva}
        print(f"{n:14s} {signed:.3f}  {mva:.3f}")
    print("\n== tails, days 1 to 10, signed-MW six (mean over Primaries) ==")
    tail_table = pl.concat(
        [
            scored[n]
            .filter(
                pl.col("slice") == "lead_days_1_10",
                pl.col("metric").is_in(TAIL_METRICS),
                pl.col("label").is_in(list(SIGNED)),
            )
            .group_by("metric")
            .agg(pl.col("value").mean())
            .with_columns(arm=pl.lit(n))
            for n in ARMS
        ]
    ).pivot(on="metric", index="arm", values="value")
    with pl.Config(tbl_cols=20, tbl_rows=20):
        print(tail_table)
    print("\n== sunny-midday scores by Primary (days 1 to 10): CRPS, q99 pinball, q99 coverage ==")
    per_primary = pl.concat(
        [
            rows[n]
            .filter(pl.col("sunny_midday"), pl.col("lead_days_1_10"))
            .group_by("label")
            .agg(
                pl.col("crps").mean(),
                pl.col("pinball_q99").mean(),
                pl.col("pinball_q95").mean(),
                pl.col("coverage_q99").mean(),
                pl.col("coverage_q95").mean(),
                pl.len().alias("n"),
            )
            .with_columns(arm=pl.lit(n))
            for n in ARMS
        ]
    ).sort("label", "arm")
    with pl.Config(tbl_rows=100, tbl_cols=20):
        print(per_primary)
    out["per_primary_sunny_midday"] = per_primary.to_dicts()
    print("\n== sunny-midday means, signed-MW six, days 1 to 10 ==")
    means = {}
    for n in ARMS:
        kept = rows[n].filter(
            pl.col("label").is_in(list(SIGNED)), pl.col("sunny_midday"), pl.col("lead_days_1_10")
        )
        means[n] = {m: float(kept[m].mean()) for m in PRIMARY_METRICS} | {"n": kept.height}
        out["sunny_midday_means"] = means
        print(
            f"{n:14s} CRPS {means[n]['crps']:.3f}  q99 pinball {means[n]['pinball_q99']:.3f}  n {kept.height}"
        )
    print(
        f"\n== primary contrasts vs {REFERENCE}, sunny-midday slice, signed-MW six, level {LEVEL:.2f}% =="
    )
    print("MDE(1) is the pre-registered single-seed definition; MDE(5) is for the 5-seed average.")
    for arm in PRIMARY_ARMS:
        for metric in PRIMARY_METRICS:
            r = contrast(
                rows=rows,
                table=table,
                a=arm,
                b=REFERENCE,
                sl="sunny_midday",
                metric=metric,
                level=LEVEL,
            )
            mde1, mde5 = minimum_detectable_effect(
                a=arm,
                b=REFERENCE,
                sl="sunny_midday",
                metric=metric,
                seeds=SEEDS,
                series=series,
                ctx=ctx,
                table=table,
            )
            margin = (
                NON_INFERIORITY_CRPS
                if metric == "crps"
                else NON_INFERIORITY_Q99_RELATIVE * means[REFERENCE]["pinball_q99"]
            )
            key = f"{arm} minus {REFERENCE} [sunny_midday, {metric}]"
            verdict = {
                "non_inferior": r["upper"] < margin,
                "gain": r["upper"] < 0 and -r["difference"] > mde5,
                "mde_below_margin": mde5 < margin,
            }
            out["primary"][key] = {
                **r,
                "mde_single": mde1,
                "mde_avg": mde5,
                "margin": margin,
                **verdict,
            }
            print(
                f"{key}: {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) "
                f"MDE(1) {mde1:.3f} MDE(5) {mde5:.3f} margin {margin:.3f} {verdict}"
            )
    print("\n== headline-band contrasts vs the reference (secondary) ==")
    for arm in PRIMARY_ARMS:
        for metric in PRIMARY_METRICS:
            r = contrast(
                rows=rows,
                table=table,
                a=arm,
                b=REFERENCE,
                sl="lead_days_1_10",
                metric=metric,
                level=LEVEL,
            )
            key = f"{arm} minus {REFERENCE} [lead_days_1_10, {metric}]"
            out["secondary"][key] = dict(r)
            print(f"{key}: {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f})")
    print("\n== secondary contrasts ==")
    for a, b in SECONDARY:
        for metric in PRIMARY_METRICS:
            for sl in ("sunny_midday", "lead_days_1_10"):
                r = contrast(rows=rows, table=table, a=a, b=b, sl=sl, metric=metric, level=LEVEL)
                key = f"{a} minus {b} [{sl}, {metric}]"
                out["secondary"][key] = dict(r)
                print(f"{key}: {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f})")
    RESULTS_PATH.write_text(json.dumps(out, default=float, indent=1))
    print(f"\nstored {RESULTS_PATH}")


if __name__ == "__main__":
    main()
