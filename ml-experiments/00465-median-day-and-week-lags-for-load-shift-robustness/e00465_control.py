r"""Score the CPU noise-floor control (original lags refitted with this experiment's settings).

Compares `a_cpu_control` with the stored `r0_lagged` forecasts on the refit month the control
covers, so the difference between them is the refit-to-refit and CPU-versus-CUDA noise that any
contrast in `result.md` has to exceed.
"""

from typing import Final

import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00392_score as s392  # ty: ignore[unresolved-import]
import e00465_early as early  # ty: ignore[unresolved-import]
import polars as pl
from studies.eval_protocol import metrics

CONTROL_DIR: Final = early.fit.ARM_DIR / "a_cpu_control"


def main() -> None:
    """Print the mean CRPS of both arms per refit month and group, and their difference."""
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
        scores = {}
        for arm, directory in (("a", early.ARMS["a_r0_lagged"]), ("control", CONTROL_DIR)):
            frame = s392.read_fold(directory=directory, sites=sites, fold=fold)
            if frame is None:
                break
            forecast = adapter.forecast_frame(stored=frame, series=series, unit="fraction").filter(
                s392.lead_expression() <= s392.MAX_LEAD_HOURS
            )
            scores[arm] = s392.score_arm(
                forecast=forecast,
                actuals=actuals,
                series=series,
                thresholds=thresholds,
                flags=flags,
            )
        if len(scores) < 2:
            continue
        keys = list(s392.KEY_COLUMNS)
        joined = scores["a"].join(scores["control"], on=keys, suffix="_ctl", validate="1:1")
        print(f"fold {name}: {joined.height:,} shared rows")
        print(
            joined.group_by("group").agg(
                a=pl.col("crps").mean(),
                control=pl.col("crps_ctl").mean(),
                difference=(pl.col("crps_ctl") - pl.col("crps")).mean(),
            )
        )


if __name__ == "__main__":
    main()
