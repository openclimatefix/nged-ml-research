r"""Early check of the experiment 465 stop rule: arm (c) against arm (a) on rows with no event.

Scores each fold whose 16 Primaries are all fitted for arm (c), and prints the CRPS in points of
effective capacity on the "outside" rows (not inside an event, not within 48 hours after one) at
leads up to 240 hours, for arms (a) experiment 402's `r0_lagged`, (b) experiment 392's
`r0_lagless`, and (c) this experiment's `c_robust`, on the half-hours all three share and with
finite observed actuals (flagged zeros dropped). It reuses experiment 392's scoring functions.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1:\\
ml-experiments/00392-xgboost-nra-load-shift-masked-targets:\\
ml-experiments/00465-median-day-and-week-lags-for-load-shift-robustness \\
    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00465-*/e00465_early.py
```
"""

from pathlib import Path
from typing import Final

import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00392_score as s392  # ty: ignore[unresolved-import]
import e00465_fit as fit  # ty: ignore[unresolved-import]
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARMS: Final[dict[str, Path]] = {
    "a_r0_lagged": ML_EXPERIMENTS_DATA_DIR
    / "00402-lag-gate-learned-weight-injected-load-shifts-part-1"
    / "work"
    / "arm_preds"
    / "r0_lagged",
    "b_r0_lagless": ML_EXPERIMENTS_DATA_DIR
    / "00392-xgboost-nra-load-shift-masked-targets"
    / "work"
    / "arm_preds"
    / "r0_lagless",
    "c_robust": fit.ARM_DIR / fit.ARM_NAME,
}


def main() -> None:
    """Print the outside-row CRPS of the arms for each complete fold of arm (c)."""
    series = adapter.series_table(labels=adapter.PRIMARY_LABELS)
    sites = series["label"].to_list()
    actuals, cache_actuals, _ = s392.load_actuals(source="observed", series=series)
    thresholds = metrics.twcrps_thresholds(
        history=cache_actuals.rename({"actual": "power_mw"}),
        series=series,
        before=s392.SCORED_FIRST,
    )
    flags, _ = s392.event_tables(series=series)
    for fold, name in enumerate(s392.FOLD_NAMES):
        stored = {}
        for arm, directory in ARMS.items():
            frame = s392.read_fold(directory=directory, sites=sites, fold=fold)
            if frame is None:
                break
            stored[arm] = adapter.forecast_frame(
                stored=frame, series=series, unit="fraction"
            ).filter(s392.lead_expression() <= s392.MAX_LEAD_HOURS)
        if len(stored) < len(ARMS):
            print(f"fold {name}: not complete, skipped")
            continue
        finite = actuals.filter(pl.col("valid_time") < metrics.HOLDOUT_START)
        shared = None
        for frame in stored.values():
            keys = frame.select(*s392.KEY_COLUMNS).join(
                finite.select("time_series_id", "valid_time"), on=["time_series_id", "valid_time"]
            )
            shared = keys if shared is None else shared.join(keys, on=list(s392.KEY_COLUMNS))
        assert shared is not None
        print(f"fold {name}: shared half-hours {shared.height:,}")
        for width in s392.WIDENINGS:
            rows = []
            for arm, frame in stored.items():
                widened = s392.widen(
                    frame=frame.join(shared, on=list(s392.KEY_COLUMNS), how="semi"), factor=width
                )
                scored = s392.score_arm(
                    forecast=widened,
                    actuals=actuals,
                    series=series,
                    thresholds=thresholds,
                    flags=flags,
                )
                rows.append(scored.with_columns(arm=pl.lit(arm)))
            table = pl.concat(rows)
            for view in ("signed-MW six", "MVA-only eight", "all 16"):
                where = s392.ROW_SETS["outside"] & s392.HEADLINE & s392.GROUP_VIEWS[view]
                result = s392.means(rows=table, where=where, by=["arm"])
                print(f"widening {width}, {view}, rows with no labelled event, days 1-10:")
                print(result)
            during = s392.means(
                rows=table,
                where=s392.ROW_SETS["during"] & s392.HEADLINE & s392.GROUP_VIEWS["signed-MW six"],
                by=["arm"],
            )
            print(f"widening {width}, signed-MW six, inside labelled events:")
            print(during)


if __name__ == "__main__":
    main()
