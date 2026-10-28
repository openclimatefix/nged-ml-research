"""Score Primary demand research default 1, extended to 336 hours, under demand-v1.

Prints every number the write-up quotes. Arms: `stack` (default 1 extended; D8 uses its
own-residual spread from fold month 2025-08), `stack_d8_donor` (D8's donor-spread arm), and the
stored `bg2`, `climatology`, and `manual_heuristic` of experiment 87 on `fold_a_daily_gap_v1`.
D15 and D18 are report only.

    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00377-*/e00377_score.py
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import e00377_adapter as adapter
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA = ML_EXPERIMENTS_DATA_DIR
WORK = DATA / "00377-demand-v1-first-run-feasibility" / "work"
CACHE_DIR = DATA / "00106-differentiable-physics-4-all-years-drift-and-recent-level" / "cache"
SCORED_FIRST = datetime(2025, 7, 1, tzinfo=UTC)
STACK_SITES = (
    "D2",
    "D3",
    "D4",
    "D5",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D14",
    "D15",
    "D16",
    "D18",
    "D19",
    "D21",
    "D22",
)
CONTROL_SITES = ("D5", "D9", "D10", "D11", "D14", "D16", "D18")
HEADLINE_BAND = "lead_days_1_10"
LATE_BAND = "lead_days_8_10"
BEYOND_BAND = "lead_days_11_14"


def stage_2_path(*, site: str, name: str | None = None) -> Path:
    """Return the stored stage-2 forecast of a site (`name` overrides the file stem)."""
    source = "d15" if site == "D15" else "control" if site in CONTROL_SITES else "mva"
    stem = name or f"stack_d_{site}"
    return WORK / source / "arms" / "a0" / "dp_preds_s2" / f"{stem}.parquet"


def stack_frames(*, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the stack's forecast frame, and D8's donor-spread frame, in MW."""
    pieces = []
    for site in STACK_SITES:
        one = series.filter(pl.col("label") == site)
        stored = pl.read_parquet(stage_2_path(site=site))
        pieces.append(adapter.forecast_frame(stored=stored, series=one, unit="points"))
    d8 = series.filter(pl.col("label") == "D8")
    donor = pl.read_parquet(stage_2_path(site="D8", name="stack_d_donor_D8"))
    return {
        "stack": pl.concat(pieces),
        "stack_d8_donor": adapter.forecast_frame(stored=donor, series=d8, unit="points"),
    }


def reference_frames(*, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return BG2, climatology, and the manual heuristic of experiment 87."""
    out = {}
    for arm, name in (
        ("bg2", "p_s6_bg2"),
        ("climatology", "ref_climatology"),
        ("manual_heuristic", "ref_manual_heuristic"),
    ):
        stored = pl.read_parquet(
            adapter.stored_path(root=DATA, pattern=f"00087*/predictions/{name}/predictions.parquet")
        ).filter(pl.col("time_series_id").is_in(series["time_series_id"].to_list()))
        out[arm] = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    return out


def main() -> None:  # noqa: PLR0912, PLR0915
    """Print the scores."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
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
    arms = {**stack_frames(series=series), **reference_frames(series=series)}
    fitted = series.filter(pl.col("label").is_in(STACK_SITES))
    main_arms = {n: arms[n] for n in ("stack", "bg2", "climatology", "manual_heuristic")}
    table = metrics.shared_slice_table(
        forecasts=main_arms,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    )
    print(f"shared half-hours of the four arms: {table.height}")
    print(
        "runs per Primary in the shared rows (distinct origins):",
        table.group_by("time_series_id")
        .agg(n=pl.col("power_fcst_init_time").n_unique())
        .get_column("n")
        .unique()
        .to_list(),
    )
    scored = {}
    rows = {}
    for name, frame in main_arms.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        scored[name] = metrics.score_forecast(
            forecast=kept, actuals=actuals, series=fitted, slice_table=table, thresholds=thresholds
        )
        rows[name] = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=fitted, thresholds=thresholds, slice_table=table
        )
    out = WORK / "scores"
    out.mkdir(exist_ok=True)
    pl.concat([t.with_columns(arm=pl.lit(n)) for n, t in scored.items()]).write_parquet(
        out / "scores.parquet"
    )

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

    for band in (HEADLINE_BAND, "lead_day_1", LATE_BAND, BEYOND_BAND, "all"):
        print(
            f"\n== CRPS, points, slice {band} "
            "(arms: stack = Primary demand research default 1, extended to 336 h) =="
        )
        print(crps_table(band))
    print("\n== MAE of the median, slice lead_days_1_10 ==")
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

    groups = {
        "signed-MW": pl.col("group") == "signed-MW",
        "MVA-only with D8": pl.col("group") == "MVA-only",
        "MVA-only without D8": (pl.col("group") == "MVA-only") & (pl.col("label") != "D8"),
        "short-history (D15, D18, report only)": pl.col("group") == "short-history",
    }
    print(
        "\n== HEADLINE contrasts (99% interval, 20,000 resamples), "
        "treatment minus reference, CRPS =="
    )
    for band in (HEADLINE_BAND, LATE_BAND, BEYOND_BAND):
        for ref in ("bg2", "manual_heuristic", "climatology"):
            for gname, where in groups.items():
                result = metrics.headline_month_bootstrap_contrast(
                    treatment=rows["stack"],
                    reference=rows[ref],
                    metric="crps",
                    where=where & pl.col(band),
                    slice_table=table,
                )
                print(
                    f"{band:15s} stack - {ref:16s} {gname:34s} "
                    f"{result['difference']:+.3f} "
                    f"({result['lower']:+.3f} to {result['upper']:+.3f}) "
                    f"n={result['n_half_hours']}, months={result['n_months']}"
                )
    print(
        "\n== Per-Primary contrasts stack - bg2, 95% interval, 2,000 resamples, "
        "slice lead_days_1_10 =="
    )
    print(
        metrics.reported_contrast_by_primary(
            treatment=rows["stack"],
            reference=rows["bg2"],
            metric="crps",
            where=pl.col(HEADLINE_BAND),
            slice_table=table,
        )
    )
    print("\n== Leave-one-out stack - bg2 (95%), lead_days_1_10 ==")
    for gname, labels in (
        ("signed-MW", ["D5", "D9", "D10", "D11", "D14", "D16"]),
        ("MVA-only", ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]),
    ):
        for left_out in labels:
            keep = pl.col("label").is_in([x for x in labels if x != left_out])
            result = metrics.month_bootstrap_contrast(
                treatment=rows["stack"],
                reference=rows["bg2"],
                metric="crps",
                where=keep & pl.col(HEADLINE_BAND),
                slice_table=table,
            )
            print(
                f"{gname:10s} without {left_out:4s} {result['difference']:+.3f} "
                f"({result['lower']:+.3f} to {result['upper']:+.3f})"
            )
    print(
        "\n== Primaries won by stack against each reference "
        "(CRPS, days 1 to 10; D15 and D18 not counted) =="
    )
    countable = lambda t: t.filter(~pl.col("label").is_in(["D15", "D18"]))  # noqa: E731
    for ref in ("bg2", "manual_heuristic", "climatology"):
        print(
            ref,
            metrics.primaries_won(
                treatment=countable(scored["stack"]),
                reference=countable(scored[ref]),
                slice_name=HEADLINE_BAND,
            ),
        )
    print(
        "\n== Tail measures, stack and bg2, slice lead_days_1_10, "
        "per group (mean over Primaries) =="
    )
    tails = [
        "pinball_q01",
        "pinball_q05",
        "pinball_q95",
        "pinball_q99",
        "coverage_q01",
        "coverage_q05",
        "coverage_q95",
        "coverage_q99",
        "twcrps",
    ]
    for name in ("stack", "bg2", "manual_heuristic", "climatology"):
        t = scored[name].filter(pl.col("slice") == HEADLINE_BAND, pl.col("metric").is_in(tails))
        print(name)
        print(
            t.group_by("group", "metric")
            .agg(pl.col("value").mean())
            .pivot(on="metric", index="group", values="value")
            .sort("group")
        )
    print(
        "\n== Slice CRPS, stack vs bg2, pooled by group: "
        "sunny/winter nights/cold nights/E7/Christmas/events =="
    )
    for sl in (
        "sunny_unit_shape",
        "sunny_april_june",
        "winter_nights",
        "cold_nights",
        "economy_7_step",
        "christmas",
        "bank_holidays",
        "event_rows",
        "event_following_7_days",
        "cold_start_months",
    ):
        parts = []
        for name in ("stack", "bg2"):
            t = rows[name].filter(pl.col(sl), pl.col(HEADLINE_BAND))
            parts.append(t.group_by("group").agg(pl.col("crps").mean().alias(name), n=pl.len()))
        print(sl)
        print(parts[0].join(parts[1].drop("n"), on="group").sort("group"))
    print("\n== Median minus mean of the stack, per site, in points, and q05 coverage ==")
    stored_mean = []
    for site in STACK_SITES:
        f = pl.read_parquet(stage_2_path(site=site)).filter(
            pl.col("power_fcst_init_time") >= SCORED_FIRST
        )
        stored_mean.append(
            f.select(site=pl.lit(site), diff=(pl.col("q50") - pl.col("mean")).mean())
        )
    print(pl.concat(stored_mean))
    print("\n== D8: own-residual arm (main) vs donor-spread arm, on shared rows, CRPS ==")
    d8 = series.filter(pl.col("label") == "D8")
    d8_id = d8["time_series_id"][0]
    d8_arms = {
        n: arms[n].filter(pl.col("time_series_id") == d8_id)
        for n in ("stack", "stack_d8_donor", "bg2")
    }
    table8 = metrics.shared_slice_table(forecasts=d8_arms, actuals=actuals, context=context)
    for name, frame in d8_arms.items():
        s = metrics.score_forecast(
            forecast=metrics.restrict_to_slice_table(forecast=frame, slice_table=table8),
            actuals=actuals,
            series=d8,
            slice_table=table8,
            thresholds=thresholds,
            slices=[HEADLINE_BAND, BEYOND_BAND, "all"],
        )
        print(
            name,
            s.filter(pl.col("metric") == "crps").select("slice", "n_half_hours", "value").rows(),
        )
    print("\n== Avoidable curtailment cost, signed-MW six, pounds, scored year ==")
    flows = actuals.rename({"actual": "flow_mw"})
    cost_forecasts = {
        n: metrics.restrict_to_slice_table(forecast=main_arms[n], slice_table=table).join(
            actuals, on=["time_series_id", "valid_time"]
        )
        for n in main_arms
    }
    signed = series.filter(pl.col("label").is_in(["D5", "D9", "D10", "D11", "D14", "D16"]))
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
    print(
        "\n== Replicate months 2025-04 to 2025-06 and the same months a year later, "
        "stack CRPS days 1 to 10 (no reference forecasts exist for 2025-04 to 2025-06) =="
    )
    only = metrics.row_losses(
        forecast=arms["stack"], actuals=actuals, series=fitted, thresholds=thresholds
    )
    print(
        only.filter(pl.col("lead_hours") <= 240)
        .with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
        .filter(
            pl.col("month").is_in(
                ["2025-04", "2025-05", "2025-06", "2026-04", "2026-05", "2026-06"]
            )
        )
        .group_by("group", "month")
        .agg(pl.col("crps").mean(), n=pl.len())
        .sort("group", "month")
    )
    print("\n== D15, report only: stack and references on shared rows ==")
    d15 = series.filter(pl.col("label") == "D15")
    d15_id = d15["time_series_id"][0]
    arms15 = {n: arms[n].filter(pl.col("time_series_id") == d15_id) for n in main_arms}
    t15 = metrics.shared_slice_table(
        forecasts=arms15,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    )
    for n, f in arms15.items():
        s = metrics.score_forecast(
            forecast=metrics.restrict_to_slice_table(forecast=f, slice_table=t15),
            actuals=actuals,
            series=d15,
            slice_table=t15,
            thresholds=thresholds,
            slices=[HEADLINE_BAND, BEYOND_BAND, "all"],
        )
        print(
            n,
            s.filter(pl.col("metric") == "crps")
            .select("slice", "n_half_hours", "value", "status")
            .rows(),
        )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
