r"""Avoidable curtailment cost of experiment 503's arms (signed-MW six), by month and row set.

The cost is `metrics.avoidable_curtailment_cost` (node-only mode, the 12 to 36 hour lead window, a
limit of `f` times the series' maximum flow before the scored year) at f = 0.9, 0.75, and 0.5. It
is read only as a difference between arms, per refit month, so the sign count over the four months
stands in for an interval. Row sets: all rows, rows inside labelled events, and rows within 7 days
after an event. The arms are those of `e00503_score.py` (BG2, S1, and `manual_heuristic` are not
reproduced here).

```bash
PYTHONPATH=... DATA_PATH_INTERNAL=... \\
    uv run --no-sync python ml-experiments/00503-*/e00503_curtail.py
```
"""

import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00392_score as s392  # ty: ignore[unresolved-import]
import e00503_fit as fit  # ty: ignore[unresolved-import]
import e00503_score as score  # ty: ignore[unresolved-import]
import polars as pl
from studies.eval_protocol import metrics

FRACTIONS = (0.9, 0.75, 0.5)
CONTROLS = ("a", "b", "f")


def main() -> None:
    """Print the cost per arm and month, and each arm minus the controls."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(200)
    series = adapter.series_table(labels=adapter.PRIMARY_LABELS)
    series = series.filter(pl.col("label").is_in(list(fit.SIGNED_MW_LABELS)))
    actuals, _, _ = s392.load_actuals(source="observed", series=series)
    flags, _ = s392.event_tables(series=series)
    arms = [a for a in score.SOURCES if a != "arep"]
    results = []
    for fold, name in enumerate(s392.FOLD_NAMES):
        stored = {}
        for arm in arms:
            frame = s392.read_fold(
                directory=score.SOURCES[arm], sites=list(fit.SIGNED_MW_LABELS), fold=fold
            )
            stored[arm] = adapter.forecast_frame(
                stored=frame, series=series, unit="fraction"
            ).filter(s392.lead_expression() <= score.MAX_LEAD_HOURS)
        stored["f"] = score.mix_frames(first=stored["a"], second=stored["b"])
        shared = None
        for frame in stored.values():
            keys = frame.select(*s392.KEY_COLUMNS)
            shared = keys if shared is None else shared.join(keys, on=list(s392.KEY_COLUMNS))
        assert shared is not None
        for row_set, keep in (
            ("all rows", pl.lit(value=True)),
            ("inside events", pl.col("in_event")),
            ("7 days after", pl.col("post7d")),
        ):
            selected = {
                arm: frame.join(shared, on=list(s392.KEY_COLUMNS), how="semi")
                .join(actuals, on=["time_series_id", "valid_time"])
                .join(
                    flags.select("time_series_id", "valid_time", "in_event", "post7d"),
                    on=["time_series_id", "valid_time"],
                    how="left",
                )
                .with_columns(pl.col("in_event", "post7d").fill_null(value=False))
                .filter(keep)
                .drop("in_event", "post7d")
                for arm, frame in stored.items()
            }
            for fraction in FRACTIONS:
                try:
                    cost = metrics.avoidable_curtailment_cost(
                        forecasts=selected,
                        series=series,
                        flows=actuals.rename({"actual": "flow_mw"}),
                        scored_start=s392.SCORED_FIRST,
                        fraction=fraction,
                    )
                except ValueError as error:
                    print(f"{name} | {row_set} | f={fraction}: cost undefined ({error})")
                    continue
                results.append(
                    cost.with_columns(
                        month=pl.lit(name), row_set=pl.lit(row_set), fraction=pl.lit(fraction)
                    )
                )
    table = pl.concat(results).filter(pl.col("status") == "ok")
    table.write_parquet(fit.WORK_DIR / "scores" / "curtailment.parquet")
    pooled = table.group_by("arm", "row_set", "fraction", "month").agg(
        cost=pl.col("avoidable_cost_gbp").sum()
    )
    for row_set in ("all rows", "inside events", "7 days after"):
        for fraction in FRACTIONS:
            part = pooled.filter((pl.col("row_set") == row_set) & (pl.col("fraction") == fraction))
            if part.is_empty():
                print(f"\n-- {row_set}, f = {fraction}: cost undefined in every month --")
                continue
            wide = part.pivot(on="arm", index="month", values="cost").sort("month")
            print(
                f"\n-- {row_set}, f = {fraction}: avoidable cost, pounds, "
                f"six Primaries, by month --"
            )
            print(wide.with_columns(pl.selectors.float().round(0)))
            total = part.group_by("arm").agg(cost=pl.col("cost").sum())
            print("total:", {r["arm"]: round(r["cost"]) for r in total.iter_rows(named=True)})
            for arm in ("g", "ga", "gb"):
                for control in CONTROLS:
                    d = wide.select(diff=pl.col(arm) - pl.col(control))["diff"]
                    print(
                        f"{arm} minus {control}: total {d.sum():+.0f}; months lower: "
                        f"{int((d < 0).sum())} of {d.len()}"
                    )


if __name__ == "__main__":
    main()
