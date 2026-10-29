# ruff: noqa: E501, PLR0915
"""Score BG2's components at all 16 Primaries under demand-v1 and print every number quoted.

Arms: BG2 (`bg2`), the five members alone (`only_W`, `only_M`, `only_AN`, `only_PT`, `only_ST`),
BG2 without each member (`drop_<member>`), the staircase of experiment 87 (`s1_frozen`,
`s2_stack`, `s3_monthly`, `s4_weekly`, `s5_blend_no_offset`), BG2 with the spread factor 1.0
(`a5_no_spread`), and the references `manual_heuristic` and `climatology`. The component arms come
from `e00379_build.py`; the staircase, A5, and the references from experiment 87's stored arms.
Experiment 377's stack is scored on its own shared rows with BG2 (`--with-stack`), because its
rows start later at some Primaries and would otherwise shrink the shared rows of every arm.

    PYTHONPATH=ml-experiments/00379-* DATA_PATH_INTERNAL=... uv run --no-sync python \
        ml-experiments/00379-*/e00379_score.py [--with-stack]
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00379_adapter as adapter
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = DATA / "00379-bg2-components-all-16-primaries-to-14-days" / "work"
STORED_87: Final[Path] = (
    DATA / "00087-bg2-best-guess-assembled-with-drop-one-ablations" / "work" / "arms"
)
WORK_377: Final[Path] = DATA / "00377-demand-v1-first-run-feasibility" / "work"
CACHE_DIR: Final[Path] = (
    DATA / "00106-differentiable-physics-4-all-years-drift-and-recent-level" / "cache"
)
SCORED_FIRST: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
MAX_LEAD_HOURS: Final[float] = 336.0
MEMBERS: Final[tuple[str, ...]] = ("W", "M", "AN", "PT", "ST")
STORED: Final[dict[str, str]] = {
    "s1_frozen": "p_s1_frozen",
    "s2_stack": "p_s2_stack",
    "s3_monthly": "p_s3_monthly",
    "s4_weekly": "p_s4_weekly",
    "s5_blend_no_offset": "p_s5_blend_no_offset",
    "a5_no_spread": "p_a5_spread_off",
    "bg2": "p_s6_bg2",
    "manual_heuristic": "ref_manual_heuristic",
    "climatology": "ref_climatology",
}
STAIRCASE: Final[tuple[str, ...]] = (
    "s1_frozen",
    "s2_stack",
    "s3_monthly",
    "s4_weekly",
    "s5_blend_no_offset",
    "bg2",
)
HEADLINE_BAND: Final[str] = "lead_days_1_10"
BANDS: Final[tuple[str, ...]] = (HEADLINE_BAND, "lead_days_8_10", "lead_days_11_14")
CENTRAL_STACK_SITES: Final[tuple[str, ...]] = adapter.PRIMARY_LABELS
CONTROL_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16", "D18")


def arm_frames(*, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return every arm's forecast frame in MW, leads to `MAX_LEAD_HOURS`."""
    ids = series["time_series_id"].to_list()
    out: dict[str, pl.DataFrame] = {}
    for arm, file in STORED.items():
        stored = pl.read_parquet(STORED_87 / f"{file}.parquet").filter(
            pl.col("time_series_id").is_in(ids), pl.col("lead_hours") <= MAX_LEAD_HOURS
        )
        out[arm] = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    for arm in ("full", *(f"drop_{m}" for m in MEMBERS), *(f"only_{m}" for m in MEMBERS)):
        stored = pl.read_parquet(WORK / "arms" / f"{arm}.parquet").filter(
            pl.col("lead_hours") <= MAX_LEAD_HOURS
        )
        out["gate_full" if arm == "full" else arm] = adapter.forecast_frame(
            stored=stored, series=series, unit="fraction"
        )
    return out


def stack_frame(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 377's stack (D8's own-residual arm) in MW, read only."""
    pieces = []
    for site in CENTRAL_STACK_SITES:
        source = "d15" if site == "D15" else "control" if site in CONTROL_SITES else "mva"
        path = WORK_377 / source / "arms" / "a0" / "dp_preds_s2" / f"stack_d_{site}.parquet"
        pieces.append(
            adapter.forecast_frame(
                stored=pl.read_parquet(path),
                series=series.filter(pl.col("label") == site),
                unit="points",
            ).filter(
                pl.col("valid_time") - pl.col("power_fcst_init_time") <= pl.duration(hours=336)
            )
        )
    return pl.concat(pieces)


def main() -> None:
    """Print the scores."""
    with_stack = "--with-stack" in sys.argv
    pl.Config.set_tbl_rows(300)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(260)
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)
    actuals, context = adapter.actuals_and_context(
        cache=cache,
        series=series,
        first=datetime(2024, 4, 1, tzinfo=UTC),
        last=datetime(2026, 7, 1, tzinfo=UTC),
    )
    events = adapter.events_from_masks(cache=cache, series=series)
    first_readings = {
        int(sid): next(
            t for t, p in zip(cache.times_series(), cache.sites[lab].present, strict=True) if p
        )
        for sid, lab, _ in series.iter_rows()
    }
    thresholds = metrics.twcrps_thresholds(
        history=actuals.rename({"actual": "power_mw"}), series=series, before=SCORED_FIRST
    )
    arms = arm_frames(series=series)
    # Unit check: the stored actual of experiment 87 equals the cache actual.
    stored_actual = pl.read_parquet(STORED_87 / "p_s6_bg2.parquet").select(
        "time_series_id", "valid_time", "actual", "capacity_mw"
    )
    check = stored_actual.join(
        actuals, on=["time_series_id", "valid_time"], suffix="_cache"
    ).filter(pl.col("actual").is_finite(), pl.col("actual_cache").is_finite())
    gap = (check["actual"] * check["capacity_mw"] - check["actual_cache"]).abs()
    print(
        f"unit check: {check.height:,} rows, max |stored actual - cache actual| = {gap.max():.4f} MW"
    )
    print(
        "gate: re-blended BG2 vs stored BG2 on shared keys, max |q50 gap| MW =",
        (arms["gate_full"]["q50"] - arms["bg2"]["q50"]).abs().max(),
    )
    del arms["gate_full"]
    if with_stack:
        arms = {"bg2": arms["bg2"], "stack_377": stack_frame(series=series)}
    table = metrics.shared_slice_table(
        forecasts=arms,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    )
    print(f"arms: {list(arms)}")
    print(f"shared half-hours: {table.height}")
    scored, rows = {}, {}
    for name, frame in arms.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        scored[name] = metrics.score_forecast(
            forecast=kept, actuals=actuals, series=series, slice_table=table, thresholds=thresholds
        )
        rows[name] = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        )
    tag = "stack" if with_stack else "components"
    out = WORK / "scores"
    out.mkdir(exist_ok=True)
    pl.concat([t.with_columns(arm=pl.lit(n)) for n, t in scored.items()]).write_parquet(
        out / f"scores_{tag}.parquet"
    )
    pl.concat([r.with_columns(arm=pl.lit(n)) for n, r in rows.items()]).write_parquet(
        out / f"rows_{tag}.parquet"
    )
    table.write_parquet(out / f"slice_table_{tag}.parquet")

    def crps_table(slice_name: str) -> pl.DataFrame:
        parts = [
            t.filter(pl.col("metric") == "crps", pl.col("slice") == slice_name).select(
                "label", "group", "n_half_hours", pl.col("value").alias(n)
            )
            for n, t in scored.items()
        ]
        joined = parts[0]
        for p in parts[1:]:
            joined = joined.join(p.drop("group", "n_half_hours"), on="label")
        return joined.sort("group", "label")

    for band in (HEADLINE_BAND, "lead_day_1", "lead_days_5_7", "lead_days_8_10", "lead_days_11_14"):
        print(f"\n== CRPS, points, per Primary, slice {band} ==")
        print(crps_table(band))
    print("\n== MAE of the median, days 1 to 10, per Primary ==")
    print(
        pl.concat(
            [
                t.filter(pl.col("metric") == "mae_median", pl.col("slice") == HEADLINE_BAND).select(
                    "label", pl.lit(n).alias("arm"), "value"
                )
                for n, t in scored.items()
            ]
        )
        .pivot(on="arm", index="label", values="value")
        .sort("label")
    )
    print("\n== Load-shift slices, CRPS, days 1 to 10, by group ==")
    for sl in ("event_rows", "event_following_7_days", "sunny_april_june", "cold_nights"):
        parts = []
        for name in scored:
            t = rows[name].filter(pl.col(sl), pl.col(HEADLINE_BAND))
            parts.append(t.group_by("group").agg(pl.col("crps").mean().alias(name), n=pl.len()))
        joined = parts[0]
        for p in parts[1:]:
            joined = joined.join(p.drop("n"), on="group")
        print(sl)
        print(joined.sort("group"))
    if not with_stack:
        print("\n== Avoidable curtailment cost, signed-MW six, pounds ==")
        flows = actuals.rename({"actual": "flow_mw"})
        signed = series.filter(pl.col("label").is_in(["D5", "D9", "D10", "D11", "D14", "D16"]))
        cost_arms = ("bg2", "s4_weekly", "only_ST", "manual_heuristic", "climatology")
        cost_forecasts = {
            n: metrics.restrict_to_slice_table(forecast=arms[n], slice_table=table).join(
                actuals, on=["time_series_id", "valid_time"]
            )
            for n in cost_arms
        }
        for fraction in (0.5, 0.9):
            cost = metrics.avoidable_curtailment_cost(
                forecasts=cost_forecasts,
                series=signed,
                flows=flows,
                scored_start=SCORED_FIRST,
                fraction=fraction,
            )
            print(f"limit fraction {fraction}")
            print(cost.pivot(on="arm", index="label", values="avoidable_cost_gbp").sort("label"))


if __name__ == "__main__":
    main()
