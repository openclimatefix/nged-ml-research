"""Per-Primary tail measures and avoidable curtailment cost during load-shift events.

Prints, for the full stack, the Kalman-off arm, the plain GAM, and the references: per Primary the
q01, q05, q95, q99 coverage and pinball loss at days 1 to 10; and the signed-MW six's avoidable
curtailment cost (limit fractions 0.9 and 0.5 of the earlier maximum) on three row sets: the
hand-labelled event rows, the 7 days after each event, and all rows.

    PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility \
    uv run --no-sync python ml-experiments/00380-*/e00380_tails_curtailment.py
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00380_score as score
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics

ARMS = ("full", "drop_k", "build0")
TAILS = (
    "coverage_q01",
    "coverage_q05",
    "coverage_q95",
    "coverage_q99",
    "pinball_q01",
    "pinball_q05",
    "pinball_q95",
    "pinball_q99",
)
SIGNED = ["D5", "D9", "D10", "D11", "D14", "D16"]


def main() -> None:
    """Print the tables."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=base.CACHE_DIR)
    actuals, context = adapter.actuals_and_context(
        cache=cache, series=series, first=datetime(2024, 4, 1, tzinfo=UTC),
        last=datetime(2026, 7, 1, tzinfo=UTC),
    )  # fmt: skip
    events = adapter.events_from_masks(cache=cache, series=series)
    first_readings = {
        int(sid): next(
            t for t, p in zip(cache.times_series(), cache.sites[lab].present, strict=True) if p
        )
        for sid, lab, _ in series.iter_rows()
    }
    thresholds = metrics.twcrps_thresholds(
        history=actuals.rename({"actual": "power_mw"}), series=series, before=base.SCORED_FIRST
    )
    arms = {n: score.arm_frame(name=n, series=series) for n in ARMS}
    arms.update(base.reference_frames(series=series))
    table = metrics.shared_slice_table(
        forecasts=arms, actuals=actuals, context=context, events=events,
        first_readings=first_readings,
    )  # fmt: skip
    scored = {}
    for name, frame in arms.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        scored[name] = metrics.score_forecast(
            forecast=kept, actuals=actuals, series=series, slice_table=table, thresholds=thresholds
        )
    for measure_group in (TAILS[:4], TAILS[4:]):
        parts = [
            scored[n]
            .filter(pl.col("slice") == base.HEADLINE_BAND, pl.col("metric").is_in(measure_group))
            .select("label", "group", "metric", pl.col("value").alias(n))
            for n in scored
        ]
        joined = parts[0]
        for p in parts[1:]:
            joined = joined.join(p.drop("group"), on=["label", "metric"])
        print(f"\n== Per-Primary {measure_group[0].split('_')[0]}, days 1 to 10, by arm ==")
        print(joined.sort("metric", "group", "label"))
    print("\n== Avoidable curtailment cost (pounds), signed-MW six, scored year ==")
    flows = actuals.rename({"actual": "flow_mw"})
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    row_sets = {
        "all rows": pl.lit(True),
        "event rows": pl.col("event_rows"),
        "7 days after events": pl.col("event_following_7_days"),
    }
    signed = series.filter(pl.col("label").is_in(SIGNED))
    for set_name, flag in row_sets.items():
        selected = table.filter(flag).select(keys)
        print(f"\n-- {set_name}: {selected.height} half-hours across the six --")
        forecasts = {
            n: arms[n]
            .join(selected, on=keys, how="semi")
            .join(actuals, on=["time_series_id", "valid_time"])
            for n in arms
            if n in (*ARMS, "bg2", "manual_heuristic")
        }
        for fraction in (0.9, 0.5):
            pieces = []
            for label in SIGNED:
                one = signed.filter(pl.col("label") == label)
                try:
                    pieces.append(
                        metrics.avoidable_curtailment_cost(
                            forecasts=forecasts,
                            series=one,
                            flows=flows,
                            scored_start=base.SCORED_FIRST,
                            fraction=fraction,
                        )
                    )
                except ValueError:
                    print(f"{label}: no scored half-hour above the limit in these rows (undefined)")
            if not pieces:
                continue
            cost = pl.concat(pieces)
            print(f"limit fraction {fraction}")
            print(cost.pivot(on="arm", index="label", values="avoidable_cost_gbp").sort("label"))
            print(cost.group_by("arm").agg(pl.col("avoidable_cost_gbp").sum()).sort("arm"))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
