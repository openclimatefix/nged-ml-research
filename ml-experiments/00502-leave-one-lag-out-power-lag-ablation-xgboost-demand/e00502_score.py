r"""Score experiment 502: the leave-one-lag-out arms against the stored lagged and lagless arms.

Arms, all on the half-hours every arm shares (finite observed actuals with flagged zeros dropped,
valid time before 2026-07-01, lead up to 336 hours; scored at leads up to 240 hours):

- `a`: experiment 402's `r0_lagged`; `b`: experiment 392's `r0_lagless`; `rep`: this experiment's
  CUDA repeat of (a) (`a_rep`), the noise-floor control.
- `nowk`, `noday`: this experiment's arms without the week lag and without the day lag.
- `noorigin`: experiment 388's arm E (the baseline without the newest reading), re-scored with no
  fits.
- `f`: the fixed 50/50 mixture of the probability distributions of (a) and (b)
  (`e00402_mixture.Pair`).

Row sets as in experiment 392 (`outside` = no labelled event: not inside an event and not within 48
hours after one; `during`; `after7d`), plus `harm`: a lag lies in an unpadded corrected event and
the target lies outside every event (experiment 402's
`gate_scores_r0_forward` table: real rows with `lags_real` > 0 and not `target_real`). Only the
six Primaries with signed-MW readings are scored. The minimum detectable effect (MDE), (rep) minus
(a) on the four refit dates, is printed before any contrast.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1:\\
ml-experiments/00392-xgboost-nra-load-shift-masked-targets:\\
ml-experiments/00402-lag-gate-learned-weight-injected-load-shifts-part-1:\\
ml-experiments/00502-leave-one-lag-out-power-lag-ablation-xgboost-demand \\
    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00502-*/e00502_score.py
```
"""

from pathlib import Path
from typing import Final

import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00392_score as s392  # ty: ignore[unresolved-import]
import e00402_mixture as mixture  # ty: ignore[unresolved-import]
import e00502_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.eval_protocol.slices import SIGNED_MW_LABELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final = fit.WORK_DIR / "scores"
ARM_ORDER: Final[tuple[str, ...]] = ("a", "rep", "b", "f", "nowk", "noday", "noorigin")
SOURCES: Final[dict[str, Path]] = {
    "a": ML_EXPERIMENTS_DATA_DIR
    / "00402-lag-gate-learned-weight-injected-load-shifts-part-1"
    / "work"
    / "arm_preds"
    / "r0_lagged",
    "b": ML_EXPERIMENTS_DATA_DIR
    / "00392-xgboost-nra-load-shift-masked-targets"
    / "work"
    / "arm_preds"
    / "r0_lagless",
    "rep": fit.ARM_DIR / "a_rep",
    "nowk": fit.ARM_DIR / "nowk",
    "noday": fit.ARM_DIR / "noday",
    "noorigin": ML_EXPERIMENTS_DATA_DIR
    / "00388-xgboost-lagged-power-column-sets-weekly-demand"
    / "work"
    / "arm_preds"
    / "monthly"
    / "e",
}
CONTRASTS: Final[tuple[tuple[str, str], ...]] = tuple(
    (arm, control) for arm in ("nowk", "noday", "noorigin") for control in ("a", "b")
)
"""The six contrasts, each read on two row sets: 3 arms x 2 controls x 2 row sets = 12."""
ROW_SETS: Final[dict[str, pl.Expr]] = {
    "outside": s392.ROW_SETS["outside"],
    "harm": pl.col("harm"),
    "during": s392.ROW_SETS["during"],
    "after7d": s392.ROW_SETS["after7d"],
}
CONTRAST_ROW_SETS: Final[tuple[str, ...]] = ("outside", "harm")
BANDS: Final = {
    "0-240 h": pl.col("lead_hours") <= 240.0,
    "12-36 h": (pl.col("lead_hours") >= 12.0) & (pl.col("lead_hours") <= 36.0),
}
METRIC_COLUMNS: Final[tuple[str, ...]] = ("crps", *s392.LOSS_COLUMNS[1:])
BONFERRONI_CONTRASTS: Final[int] = 12
"""Contrasts times row sets, on the 0-240 h band and widening 1.0; the rest is descriptive."""
LEVEL_PERCENT: Final[float] = 100.0 * (1.0 - 0.05 / BONFERRONI_CONTRASTS)
RESAMPLES: Final[int] = 20_000
GROUP: Final[str] = "signed-MW six"


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


def harm_keys(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the (time_series_id, power_fcst_init_time, valid_time) keys of the harm rows."""
    scores = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR
        / "00402-lag-gate-learned-weight-injected-load-shifts-part-1"
        / "work"
        / "gate_scores_r0_forward.parquet",
        columns=["kind", "site", "power_fcst_init_time", "valid_time", "lags_real", "target_real"],
    ).filter((pl.col("kind") == "real") & (pl.col("lags_real") > 0) & ~pl.col("target_real"))
    ids = series.filter(pl.col("label").is_in(list(SIGNED_MW_LABELS))).select(
        pl.col("label").alias("site"), "time_series_id"
    )
    return scores.join(ids, on="site", how="inner").select(
        "time_series_id",
        pl.col("power_fcst_init_time", "valid_time").dt.replace_time_zone("UTC"),
    )


def main() -> None:
    """Print the MDE, the absolute scores, and the contrasts, and write the per-row table."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    OUT.mkdir(parents=True, exist_ok=True)
    full = adapter.series_table(labels=adapter.PRIMARY_LABELS)
    series = full.filter(pl.col("label").is_in(list(fit.SIX)))
    sites = series["label"].to_list()
    actuals, cache_actuals, _ = s392.load_actuals(source="observed", series=series)
    s392.print_dropped(cache_actuals=cache_actuals, actuals=actuals, series=series)
    thresholds = metrics.twcrps_thresholds(
        history=cache_actuals.rename({"actual": "power_mw"}),
        series=series,
        before=s392.SCORED_FIRST,
    )
    flags, _ = s392.event_tables(series=series)
    harm = harm_keys(series=series).with_columns(harm=pl.lit(value=True))
    scored: list[pl.DataFrame] = []
    for fold, name in enumerate(s392.FOLD_NAMES):
        stored = {}
        for arm, directory in SOURCES.items():
            frame = s392.read_fold(directory=directory, sites=sites, fold=fold)
            if frame is None:
                msg = f"fold {name}: arm {arm} is incomplete"
                raise ValueError(msg)
            stored[arm] = adapter.forecast_frame(
                stored=frame, series=series, unit="fraction"
            ).filter(s392.lead_expression() <= s392.MAX_LEAD_HOURS)
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
        for arm, frame in stored.items():
            for width in (1.0, 1.15):
                widened = s392.widen(
                    frame=frame.join(shared, on=list(s392.KEY_COLUMNS), how="semi"), factor=width
                )
                scored.append(
                    s392.score_arm(
                        forecast=widened,
                        actuals=actuals,
                        series=series,
                        thresholds=thresholds,
                        flags=flags,
                    )
                    .join(
                        harm,
                        on=["time_series_id", "power_fcst_init_time", "valid_time"],
                        how="left",
                        validate="m:1",
                    )
                    .with_columns(
                        pl.col("harm").fill_null(value=False),
                        arm=pl.lit(arm),
                        fold=pl.lit(fold, dtype=pl.Int64),
                        widening=pl.lit(width),
                    )
                )
    rows = pl.concat(scored)
    rows.write_parquet(OUT / "rows.parquet")
    for width in (1.0, 1.15):
        part = rows.filter(pl.col("widening") == width)
        print(f"\n{'=' * 90}\nWIDENING {width}\n{'=' * 90}")
        outside = BANDS["0-240 h"] & ROW_SETS["outside"]
        print("\n-- MDE: (rep) minus (a), CRPS, rows with no labelled event, lead 0-240 h")
        print(s392.contrast_table(rows=part, arm="rep", control="a", where=outside, by=[]))
        print(
            metrics.month_bootstrap_contrast(
                treatment=part.filter(pl.col("arm") == "rep"),
                reference=part.filter(pl.col("arm") == "a"),
                metric="crps",
                where=outside,
                level_percent=LEVEL_PERCENT,
                n_resamples=RESAMPLES,
            )
        )
        for band, band_cond in BANDS.items():
            for row_set, row_cond in ROW_SETS.items():
                cond = band_cond & row_cond
                print(f"\n-- absolute scores: {GROUP}, {row_set}, lead {band}")
                print(
                    s392.means(rows=part, where=cond, by=["arm"], columns=METRIC_COLUMNS)
                    .with_columns(
                        pl.col("arm")
                        .replace_strict({a: i for i, a in enumerate(ARM_ORDER)})
                        .alias("o")
                    )
                    .sort("o")
                    .drop("o")
                )
                if row_set not in CONTRAST_ROW_SETS:
                    continue
                for arm, control in CONTRASTS:
                    table = s392.contrast_table(
                        rows=part, arm=arm, control=control, where=cond, by=[]
                    )
                    interval = metrics.month_bootstrap_contrast(
                        treatment=part.filter(pl.col("arm") == arm),
                        reference=part.filter(pl.col("arm") == control),
                        metric="crps",
                        where=cond,
                        level_percent=LEVEL_PERCENT,
                        n_resamples=RESAMPLES,
                    )
                    print(f"contrast ({arm}) - ({control}), CRPS, {row_set}, lead {band}:")
                    print(table)
                    print(interval)


if __name__ == "__main__":
    main()
