# ruff: noqa: E501
"""Score experiment 409's arms against pooled A0 (experiment 397, stored) and print the contrasts.

Seed-averages each arm in quantile space, restricts every arm to the shared half-hours, and prints
absolute CRPS and tails per group, the pre-registered contrasts with the month-block bootstrap
(95%, 20,000 resamples, Bonferroni across the seven contrasts) and the minimum detectable effect
(2 times the standard deviation over folds of the seed-to-seed difference of the contrast).

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:ml-experiments/00403-pooled-versus-per-primary-mixture-density-network:ml-experiments/00409-mdn-solar-term-day-of-year-and-anomaly-known-answer \
  uv run --no-sync python ml-experiments/00409-*/e00409_score.py
```
"""

import itertools
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

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00409-mdn-solar-term-day-of-year-and-anomaly-known-answer" / "work"
)
MONTHS: Final = r3.MONTHS
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
ARMS: Final[tuple[str, ...]] = ("A0-noirr-doy", "C1", "C2", "C3", "C4", "C5a", "C5b")
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("C3", "A0-noirr-doy"),
    ("C2", "A0-noirr-doy"),
    ("C3", "C2"),
    ("C2", "C1"),
    ("C4", "C3"),
    ("C5a", "C3"),
    ("C5b", "C3"),
    ("C3", "A0"),
)
"""The last contrast is reported as the practical comparison and is not one of the seven."""
N_BONFERRONI: Final[int] = 7
LEVEL: Final[float] = 100.0 - 5.0 / N_BONFERRONI
"""The Bonferroni-corrected interval level in percent for a family-wise 95%."""
SLICES: Final[tuple[str, ...]] = ("lead_days_1_10", "sunny_unit_shape", "sunny_april_june")
SIGNED: Final = r3.SIGNED
TAILS: Final[tuple[str, ...]] = (
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


def arm_frame(*, arm: str, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return an arm's forecast (fraction of capacity) averaged over the seeds, all five months."""
    pieces = []
    for month in MONTHS:
        folder = r3.SEED_DIR / "A0" if arm == "A0" else WORK / "seed_preds" / arm
        files = [folder / f"{month}_s{s}.parquet" for s in seeds]
        missing = [f for f in files if not f.exists()]
        assert not missing, f"missing forecasts: {missing}"
        stacked = pl.concat([pl.read_parquet(f) for f in files])
        pieces.append(
            stacked.group_by(r3.KEYS).agg(*[pl.col(c).mean() for c in common.LEVEL_COLUMNS])
        )
    frame = pl.concat(pieces).sort(r3.KEYS)
    values = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    return frame.with_columns(
        *[pl.Series(c, values[:, i]).cast(pl.Float32) for i, c in enumerate(common.LEVEL_COLUMNS)]
    )


def contrast_text(*, rows: dict, table: pl.DataFrame, a: str, b: str, sl: str) -> ContrastInterval:
    """Return the Bonferroni-level month-block contrast of arm `a` minus arm `b` at the signed-MW six."""
    where = pl.col("label").is_in(list(SIGNED)) & pl.col(sl) & pl.col("lead_days_1_10")
    return metrics.month_bootstrap_contrast(
        treatment=rows[a],
        reference=rows[b],
        metric="crps",
        where=where,
        level_percent=LEVEL,
        n_resamples=20_000,
        slice_table=table,
    )


def monthly_signed_crps(*, rows: pl.DataFrame, sl: str) -> dict:
    """Return the signed-six mean CRPS in a slice (within days 1 to 10) for each test month."""
    kept = rows.filter(
        pl.col("label").is_in(list(SIGNED)), pl.col(sl), pl.col("lead_days_1_10")
    ).with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
    return dict(kept.group_by("month").agg(pl.col("crps").mean()).iter_rows())


def minimum_detectable_effect(
    *, a: str, b: str, sl: str, series: pl.DataFrame, ctx: dict, table: pl.DataFrame
) -> float:
    """Return 2 x the SD over folds of the seed-to-seed difference of the contrast `a - b`."""
    per_seed = {}
    for arm in (a, b):
        for s in SEEDS:
            frame = adapter.forecast_frame(
                stored=arm_frame(arm=arm, seeds=(s,)), series=series, unit="fraction"
            )
            kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
            per_seed[arm, s] = monthly_signed_crps(
                sl=sl,
                rows=metrics.scored_rows(
                    forecast=kept,
                    actuals=ctx["actuals"],
                    series=series,
                    thresholds=ctx["thresholds"],
                    slice_table=table,
                ),
            )
    sds = []
    for s1, s2 in itertools.combinations(SEEDS, 2):
        diff = [
            (per_seed[a, s1][m] - per_seed[b, s1][m]) - (per_seed[a, s2][m] - per_seed[b, s2][m])
            for m in per_seed[a, s1]
        ]
        sds.append(float(np.std(diff, ddof=1)))
    return 2.0 * float(np.mean(sds))


def main() -> None:
    """Print the absolute scores, tails, and contrasts."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    names = ["A0", *ARMS]
    frames = {
        n: adapter.forecast_frame(
            stored=arm_frame(arm=n, seeds=SEEDS), series=series, unit="fraction"
        )
        for n in names
    }
    table, scored, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    print("\n== absolute CRPS at days 1 to 10, signed-MW six / MVA-only eight (points) ==")
    for n in names:
        t = scored[n].filter(pl.col("metric") == "crps", pl.col("slice") == "lead_days_1_10")
        signed = t.filter(pl.col("label").is_in(list(SIGNED)))["value"].mean()
        mva = t.filter(pl.col("label").is_in(list(r3.MVA_ONLY)))["value"].mean()
        print(f"{n:14s} {signed:.3f}  {mva:.3f}")
    print("\n== tails, days 1 to 10, signed-MW six (mean over Primaries) ==")
    tail_table = pl.concat(
        [
            scored[n]
            .filter(
                pl.col("slice") == "lead_days_1_10",
                pl.col("metric").is_in(TAILS),
                pl.col("label").is_in(list(SIGNED)),
            )
            .group_by("metric")
            .agg(pl.col("value").mean())
            .with_columns(arm=pl.lit(n))
            for n in names
        ]
    ).pivot(on="metric", index="arm", values="value")
    with pl.Config(tbl_cols=20, tbl_rows=20):
        print(tail_table)
    print(
        f"\n== contrasts at the signed-MW six: difference (95% interval), Bonferroni level {LEVEL:.2f}% =="
    )
    for a, b in CONTRASTS:
        for sl in SLICES:
            mde = (
                minimum_detectable_effect(a=a, b=b, sl=sl, series=series, ctx=ctx, table=table)
                if b != "A0"
                else float("nan")
            )
            r = contrast_text(rows=rows, table=table, a=a, b=b, sl=sl)
            print(
                f"{a} minus {b} [{sl}]: {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) MDE {mde:.3f}"
            )


if __name__ == "__main__":
    main()
