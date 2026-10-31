r"""Score the window-surprise arms (WS A1 to A3) against experiment 391's `b0` control, both seeds.

Reads the monthly refit forecasts of `b0`, `b0s1`, `ws_a1`, `ws_a2`, and `ws_a3` for the seven
Primaries of experiment 391's secondary arms, scores them with experiment 391's machinery (the
shared slice table, `build_rows`), and prints, by metering group, the CRPS difference from `b0`
and the q95 coverage on three slices: the rows whose lag lies inside a labelled event and that fall
outside the events (experiment 391's primary slice), the rows with no lag inside an event, and all
rows, all for days 1 to 10. The difference between `b0s1` and `b0` is the noise floor. Nothing is
written to experiment 391's folders.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\\
ml-experiments/00391-xgboost-lag-surprise-second-stage:\\
$(ls -d ml-experiments/00396-*) \\
    uv run --no-sync python ml-experiments/00396-*/e00396_score_ws.py
```
"""

from datetime import UTC, datetime
from typing import Final

import e00379_score as ref379  # ty: ignore[unresolved-import]
import e00391_assemble as assemble  # ty: ignore[unresolved-import]
import e00391_fit as fit  # ty: ignore[unresolved-import]
import e00391_score as score  # ty: ignore[unresolved-import]
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.eval_protocol.slices import KEY_COLUMNS

ARMS: Final[tuple[str, ...]] = ("b0", "b0s1", "ws_a1", "ws_a2", "ws_a3")
CONTROL: Final[str] = "b0"
NOISE_ARM: Final[str] = "b0s1"
SLICE_NAMES: Final[dict[str, str]] = {
    "event-lag rows outside events": "PRIMARY: contaminated-lag, outside events, days 1-10",
    "clean rows": "clean rows (no lag in event), outside events",
    "all rows": "all rows, days 1-10",
}


def main() -> None:
    """Print the CRPS difference from `b0` and the q95 coverage of each arm."""
    series = score.adapter.series_table()
    sites = list(fit.SUBSET_SITES)
    chosen = series.filter(pl.col("label").is_in(sites))
    frames: dict[str, pl.DataFrame] = {}
    for name in ARMS:
        stored = assemble.read_arm(
            directory=assemble.ARM_DIR,
            name=name,
            sites=sites,
            refits=assemble.MONTHLY_REFITS,
            stored=False,
        )
        if stored is None:
            print(f"no files for {name}; skipped")
            continue
        stored = stored.filter(pl.col("lead_hours") <= score.MAX_LEAD_HOURS)
        frames[name] = score.adapter.forecast_frame(stored=stored, series=chosen, unit="fraction")
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=ref379.CACHE_DIR)
    actuals, context = score.adapter.actuals_and_context(
        cache=cache,
        series=chosen,
        first=datetime(2024, 4, 1, tzinfo=UTC),
        last=datetime(2026, 7, 1, tzinfo=UTC),
    )
    all_events = score.adapter.events_from_masks(cache=cache, series=chosen)
    events = (
        all_events.join(chosen.select("time_series_id", "label"), on="time_series_id")
        .filter(pl.col("end") >= score.SCORED_FIRST)
        .sort("label", "start")
        .with_row_index("event")
    )
    first_readings = {
        int(sid): next(
            t for t, p in zip(cache.times_series(), cache.sites[lab].present, strict=True) if p
        )
        for sid, lab, _ in chosen.iter_rows()
    }
    thresholds = metrics.twcrps_thresholds(
        history=actuals.rename({"actual": "power_mw"}), series=chosen, before=score.SCORED_FIRST
    )
    table = metrics.shared_slice_table(
        forecasts=frames,
        actuals=actuals,
        context=context,
        events=all_events,
        first_readings=first_readings,
    )
    flags = pl.read_parquet(assemble.OUT_DIR / "contaminated.parquet").with_columns(
        pl.col("time_series_id").cast(chosen.schema["time_series_id"])
    )
    table = table.join(flags, on=list(KEY_COLUMNS), how="left", validate="1:1").with_columns(
        pl.col("lag_in_event").fill_null(value=False)
    )
    ids = score.event_ids(table=table, events=events)
    rows = score.build_rows(
        arms=frames,
        table=table,
        flags=flags,
        ids=ids,
        actuals=actuals,
        series=chosen,
        thresholds=thresholds,
    )
    slices = score.slice_definitions(columns=table.columns)
    print(f"arms {list(rows)}; shared half-hours {table.height:,}; sites {sites}")
    for group in score.GROUPS:
        for label, key in SLICE_NAMES.items():
            where = slices[key][0] & (pl.col("group") == group)
            base = rows[CONTROL].filter(where)
            if base.is_empty():
                continue
            print(f"\n{group}, {label}: rows {base.height:,}, b0 CRPS {base['crps'].mean():.3f}")
            out = [
                {
                    "arm": name,
                    "crps_minus_b0": round(
                        rows[name].filter(where)["crps"].mean() - base["crps"].mean(), 3
                    ),
                    "coverage_q95": round(rows[name].filter(where)["coverage_q95"].mean(), 3),
                }
                for name in rows
                if name != CONTROL
            ]
            print(pl.DataFrame(out))


if __name__ == "__main__":
    main()
