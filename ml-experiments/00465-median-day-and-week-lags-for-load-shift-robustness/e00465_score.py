r"""Score experiment 465: arm (c), median lags, against the stored arms and the mixtures.

Arms, all on the half-hours every arm shares (finite observed actuals with flagged zeros dropped,
valid time before 2026-07-01, lead up to 336 hours; scored at leads up to 240 hours):

- `a`: experiment 402's `r0_lagged`; `b`: experiment 392's `r0_lagless`; `c`: this experiment's
  `c_robust`.
- `e`: the fixed 50/50 mixture of the probability distributions of (c) and (b); `f`: the fixed 50/50
  mixture of (a) and (b). A forecast is the cumulative distribution function that is linear between
  its 11 quantiles (experiment 402's algebra, `e00402_mixture.Pair`); the mixture's cumulative
  distribution function is the average of the two, inverted at the 11 levels. It is not an average
  of quantiles.

Row sets as in experiment 392 (`outside` = no labelled event: not inside an event and not within 48
hours after one; `during`; `after7d`). Real-event rows use the observed readings, which are not
corrected for the normal running arrangement, so the real-event slice measures the forecast against
the load-shifted readings. Lead bands: 0-240 hours, and 12-36 hours on its own. Intervals: month
blocks (`metrics.month_bootstrap_contrast`) at the Bonferroni level of 18 contrasts, with each
month's difference beside it (four months).

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1:\\
ml-experiments/00392-xgboost-nra-load-shift-masked-targets:\\
ml-experiments/00402-lag-gate-learned-weight-injected-load-shifts-part-1:\\
ml-experiments/00465-median-day-and-week-lags-for-load-shift-robustness \\
    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00465-*/e00465_score.py
```
"""

from typing import Final

import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00392_score as s392  # ty: ignore[unresolved-import]
import e00402_mixture as mixture  # ty: ignore[unresolved-import]
import e00465_early as early  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics

OUT: Final = early.fit.WORK_DIR / "scores"
ARM_ORDER: Final[tuple[str, ...]] = ("a", "b", "c", "e", "f")
SOURCES: Final = {"a": "a_r0_lagged", "b": "b_r0_lagless", "c": "c_robust"}
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    *((arm, control) for arm in ("c", "e") for control in ("a", "b", "f")),
    ("c", "e"),
)
"""The six pre-registered contrasts, then (c) minus (e), which is descriptive."""
GROUPS: Final = ("signed-MW six", "MVA-only eight")
ROW_SETS: Final = ("outside", "during", "after7d")
BANDS: Final = {
    "0-240 h": pl.col("lead_hours") <= 240.0,
    "12-36 h": (pl.col("lead_hours") >= 12.0) & (pl.col("lead_hours") <= 36.0),
}
METRIC_COLUMNS: Final[tuple[str, ...]] = ("crps", *s392.LOSS_COLUMNS[1:])
BONFERRONI_CONTRASTS: Final[int] = 18 * 1  # contrasts at the six: 2 arms x 3 controls x 3 row sets
LEVEL_PERCENT: Final[float] = 100.0 * (1.0 - 0.05 / BONFERRONI_CONTRASTS)
RESAMPLES: Final[int] = 20_000


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


def main() -> None:
    """Print the scores and write the per-row table."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    OUT.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table(labels=adapter.PRIMARY_LABELS)
    sites = series["label"].to_list()
    actuals, cache_actuals, _ = s392.load_actuals(source="observed", series=series)
    s392.print_dropped(cache_actuals=cache_actuals, actuals=actuals, series=series)
    thresholds = metrics.twcrps_thresholds(
        history=cache_actuals.rename({"actual": "power_mw"}),
        series=series,
        before=s392.SCORED_FIRST,
    )
    flags, _ = s392.event_tables(series=series)
    scored: list[pl.DataFrame] = []
    for fold, name in enumerate(s392.FOLD_NAMES):
        stored = {}
        for arm, source in SOURCES.items():
            frame = s392.read_fold(directory=early.ARMS[source], sites=sites, fold=fold)
            if frame is None:
                msg = f"fold {name}: arm {arm} is incomplete"
                raise ValueError(msg)
            stored[arm] = adapter.forecast_frame(
                stored=frame, series=series, unit="fraction"
            ).filter(s392.lead_expression() <= s392.MAX_LEAD_HOURS)
        stored["e"] = mix_frames(first=stored["c"], second=stored["b"])
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
                    ).with_columns(
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
        for band, band_cond in BANDS.items():
            for group in GROUPS:
                for row_set in ROW_SETS:
                    cond = band_cond & s392.GROUP_VIEWS[group] & s392.ROW_SETS[row_set]
                    print(f"\n-- absolute scores: {group}, {row_set}, lead {band}")
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
                    if group != GROUPS[0]:
                        continue
                    for arm, control in CONTRASTS:
                        table = s392.contrast_table(
                            rows=part, arm=arm, control=control, where=cond, by=[]
                        )
                        treatment = part.filter(pl.col("arm") == arm)
                        reference = part.filter(pl.col("arm") == control)
                        interval = metrics.month_bootstrap_contrast(
                            treatment=treatment,
                            reference=reference,
                            metric="crps",
                            where=cond,
                            level_percent=LEVEL_PERCENT,
                            n_resamples=RESAMPLES,
                        )
                        print(f"contrast ({arm}) - ({control}), CRPS:")
                        print(table)
                        print(interval)


if __name__ == "__main__":
    main()
