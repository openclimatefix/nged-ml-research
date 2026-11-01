r"""Avoidable curtailment cost of experiment 465's arms at the signed-MW six.

Uses experiment 392's `cost_report` (which calls `metrics.avoidable_curtailment_cost`) on the arms
(a), (b), (c), and the mixtures (e) and (f), with the 1.15 widening applied once, at the limit
fractions 0.9, 0.75, and 0.5, in three row sets: inside labelled events, in the 7 days after, and
all rows.

```bash
PYTHONPATH=<as in e00465_score.py> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00465-*/e00465_cost.py
```
"""

import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00392_score as s392  # ty: ignore[unresolved-import]
import e00465_early as early  # ty: ignore[unresolved-import]
import e00465_score as score  # ty: ignore[unresolved-import]
import polars as pl
from studies.eval_protocol import metrics

FRACTIONS = (0.9, 0.75, 0.5)
WIDTH = 1.15


def main() -> None:
    """Print the avoidable curtailment cost for each limit fraction."""
    series = adapter.series_table(labels=adapter.PRIMARY_LABELS)
    sites = series["label"].to_list()
    actuals, _, _ = s392.load_actuals(source="observed", series=series)
    flags, _ = s392.event_tables(series=series)
    flows = actuals.rename({"actual": "flow_mw"})
    frames: dict[str, list[pl.DataFrame]] = {}
    for fold in range(len(s392.FOLD_NAMES)):
        stored = {}
        for arm, source in score.SOURCES.items():
            frame = s392.read_fold(directory=early.ARMS[source], sites=sites, fold=fold)
            assert frame is not None
            stored[arm] = adapter.forecast_frame(
                stored=frame, series=series, unit="fraction"
            ).filter(s392.lead_expression() <= s392.MAX_LEAD_HOURS)
        stored["e"] = score.mix_frames(first=stored["c"], second=stored["b"])
        stored["f"] = score.mix_frames(first=stored["a"], second=stored["b"])
        finite = actuals.filter(pl.col("valid_time") < metrics.HOLDOUT_START)
        shared = None
        for frame in stored.values():
            keys = frame.select(*s392.KEY_COLUMNS).join(
                finite.select("time_series_id", "valid_time"), on=["time_series_id", "valid_time"]
            )
            shared = keys if shared is None else shared.join(keys, on=list(s392.KEY_COLUMNS))
        assert shared is not None
        for arm, frame in stored.items():
            widened = s392.widen(
                frame=frame.join(shared, on=list(s392.KEY_COLUMNS), how="semi"), factor=WIDTH
            )
            cost_frame = (
                widened.filter(
                    (s392.lead_expression() >= metrics.CURTAILMENT_LEAD_HOURS[0])
                    & (s392.lead_expression() < metrics.CURTAILMENT_LEAD_HOURS[1])
                )
                .join(actuals, on=["time_series_id", "valid_time"])
                .join(
                    flags.select("time_series_id", "valid_time", "in_event", "post7d"),
                    on=["time_series_id", "valid_time"],
                    how="left",
                )
                .with_columns(
                    pl.col("in_event", "post7d").fill_null(value=False),
                    fold=pl.lit(fold, dtype=pl.Int64),
                )
            )
            frames.setdefault(arm, []).append(cost_frame)
    for fraction in FRACTIONS:
        s392.COST_FRACTION = fraction
        print(f"\n######## limit fraction {fraction}, widening {WIDTH} ########")
        s392.cost_report(frames=dict(frames), series=series, flows=flows)


if __name__ == "__main__":
    main()
