"""Score experiment 480's A0 and A1 under demand-v1 and print every number the write-up quotes.

Scored rows are the half-hours shared by every arm in the comparison (`metrics.shared_slice_table`),
so a partly finished fit scores only the refits both arms have finished. Prints:

- the code check: A0 against experiment 381's stored `monthly/control` at D5 and D12;
- C1 (A1 minus A0, CRPS at days 1 to 10) pooled over the eight MVA-only Primaries, with and without
  D8, at the 99.67% level, and the no-harm contrasts at the signed-MW six and at D15 and D18;
- each Primary at 95%, lead bands, the sunny slices, event slices, tails, MAE of the median,
  the avoidable curtailment cost at the signed-MW six, and D12's sunny bias by lead band;
- the absolute scores of A0, A1, the stored control, the GAM stack, BG2, `manual_heuristic`, and
  `climatology`.

```bash
PYTHONPATH=ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    uv run --no-sync python ml-experiments/00480-*/e00480_score.py
```
"""

from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00480_fit as fit480  # ty: ignore[unresolved-import]
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00480-xgboost-cams-columns-at-mva-only-primaries-lab-conditions"
)
HEADLINE: Final[str] = s381.HEADLINE
LEVEL: Final[float] = 100 * (1 - 0.01 / 3)
RESAMPLES: Final[int] = 20_000
SIGNED: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
MVA_ONLY: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
SHORT: Final[tuple[str, ...]] = ("D15", "D18")
GROUPS: Final[dict[str, pl.Expr]] = {
    "MVA-only eight": pl.col("label").is_in(MVA_ONLY),
    "MVA-only seven (no D8)": pl.col("label").is_in([s for s in MVA_ONLY if s != "D8"]),
    "signed-MW six": pl.col("label").is_in(SIGNED),
    "D15 and D18": pl.col("label").is_in(SHORT),
    "D12": pl.col("label") == "D12",
    "D8": pl.col("label") == "D8",
}
SLICES: Final[tuple[str, ...]] = (
    "sunny_unit_shape",
    "sunny_cams_index_07",
    "sunny_cams_index_09",
    "sunny_april_june",
    "winter_nights",
    "cold_nights",
    "economy_7_step",
    "event_rows",
    "event_following_7_days",
)
TAILS: Final[list[str]] = [
    "pinball_q01", "pinball_q05", "pinball_q95", "pinball_q99",
    "coverage_q01", "coverage_q05", "coverage_q95", "coverage_q99", "twcrps",
]  # fmt: skip
SUNNY_LOW: Final[float] = 0.7
SUNNY_HIGH: Final[float] = 0.9
MARKED: Final[tuple[str, ...]] = ("D5", "D12", "D15", "D18")


def own_arm(*, name: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return an arm of this experiment in MW."""
    files = sorted((WORK / "arm_preds" / "monthly" / name).glob("*.parquet"))
    if not files:
        msg = f"no files for arm {name}"
        raise FileNotFoundError(msg)
    stored = pl.concat([pl.read_parquet(f) for f in files]).filter(
        pl.col("power_fcst_init_time") >= base.SCORED_FIRST
    )
    return adapter.forecast_frame(stored=stored, series=series, unit="fraction")


def cams_flags(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the two sunny flags from the realised CAMS clear-sky index of each Primary."""
    coordinates = common.series_table()
    pieces = []
    for row in coordinates.iter_rows(named=True):
        index = fit480.cams_columns(
            site=row["site"], latitude=row["latitude"], longitude=row["longitude"]
        )
        pieces.append(
            index.select(
                pl.lit(row["time_series_id"]).alias("time_series_id"),
                "valid_time",
                sunny_cams_index_07=pl.col("cams_clear_sky_index") >= SUNNY_LOW,
                sunny_cams_index_09=pl.col("cams_clear_sky_index") >= SUNNY_HIGH,
            )
        )
    return pl.concat(pieces).with_columns(
        pl.col("time_series_id").cast(series.schema["time_series_id"])
    )


def safe_contrast(*, rows: dict, table: pl.DataFrame, where: pl.Expr, treat: str, ref: str) -> str:
    """Return the formatted contrast, or a note when the filter shares no half-hour."""
    try:
        return fmt(contrast(rows=rows, table=table, where=where, treat=treat, ref=ref))
    except ValueError as error:
        return f"not scored: {error}"


def contrast(
    *, rows: dict, table: pl.DataFrame, where: pl.Expr, treat: str, ref: str
) -> metrics.ContrastInterval:
    """Return the 99.67% C1-type contrast of CRPS at days 1 to 10 for a row filter."""
    return metrics.month_bootstrap_contrast(
        treatment=rows[treat],
        reference=rows[ref],
        metric="crps",
        where=where & pl.col(HEADLINE),
        level_percent=LEVEL,
        n_resamples=RESAMPLES,
        slice_table=table,
    )


def fmt(r: metrics.ContrastInterval) -> str:
    """Format a contrast."""
    return (
        f"{r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) "
        f"n={r['n_half_hours']:,} months={r['n_months']}"
    )


def main() -> None:  # noqa: PLR0912, PLR0915
    """Print the report."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    arms = {
        "a0": own_arm(name="a0", series=series),
        "a1": own_arm(name="a1", series=series),
        "stored_control": s381.arm_frame(regime="monthly", arm="control", series=series),
        "stack": base.stack_frames(series=series)["stack"],
        **base.reference_frames(series=series),
    }
    table, scored, rows = s381.score_arms(arms=arms, series=series, ctx=ctx)
    flag_frame = cams_flags(series=series)
    keys = ["time_series_id", "valid_time"]
    rows = {n: r.join(flag_frame, on=keys, how="left") for n, r in rows.items()}
    table = table.join(flag_frame, on=keys, how="left")
    missing = (
        ctx["actuals"]
        .join(table.select(keys).unique(), on=keys)
        .filter(pl.col("actual").is_null() | pl.col("actual").is_nan())
    )
    print(f"scored half-hours with a missing actual (flagged or absent readings): {missing.height}")
    WORK.mkdir(exist_ok=True)
    (WORK / "scores").mkdir(exist_ok=True)
    pl.concat([t.with_columns(arm=pl.lit(n)) for n, t in scored.items()]).write_parquet(
        WORK / "scores" / "scores.parquet"
    )
    core = {n: arms[n] for n in ("a0", "a1", "stored_control")}
    print("\n-- core row set: A0, A1, and the stored control only (all 16 Primaries) --")
    table_c, _, rows_c = s381.score_arms(arms=core, series=series, ctx=ctx)
    flag_c = flag_frame
    rows_c = {n: r.join(flag_c, on=keys, how="left") for n, r in rows_c.items()}
    table_c = table_c.join(flag_c, on=keys, how="left")
    print(f"\nlevel {LEVEL:.2f}%, {RESAMPLES} resamples; days 1 to 10 unless stated")
    print("\n== code check: A0 minus stored control (CRPS) at D5 and D12 ==")
    for lab in ("D5", "D12"):
        r = safe_contrast(
            rows=rows_c,
            table=table_c,
            where=pl.col("label") == lab,
            treat="a0",
            ref="stored_control",
        )
        print(f"{lab}: {r}")
    print("\n== C1 and no-harm: A1 minus A0 ==")
    for name, where in GROUPS.items():
        r = safe_contrast(rows=rows_c, table=table_c, where=where, treat="a1", ref="a0")
        print(f"{name:24s} {r}")
    print("\n== A1 minus A0 by slice (99.67%) ==")
    for sl in SLICES:
        if sl not in rows_c["a0"].columns:
            print(f"(slice {sl} not in the rows)")
            continue
        for name in ("MVA-only eight", "signed-MW six", "D12"):
            r = safe_contrast(
                rows=rows_c, table=table_c, where=GROUPS[name] & pl.col(sl), treat="a1", ref="a0"
            )
            print(f"{sl:24s} {name:16s} {r}")
    print("\n== A1 minus A0 by lead band (99.67%) ==")
    for band in metrics.LEAD_BAND_SLICES:
        for name in ("MVA-only eight", "signed-MW six", "D12"):
            try:
                r = fmt(
                    metrics.month_bootstrap_contrast(
                        treatment=rows_c["a1"],
                        reference=rows_c["a0"],
                        metric="crps",
                        where=GROUPS[name] & pl.col(band),
                        level_percent=LEVEL,
                        n_resamples=RESAMPLES,
                        slice_table=table_c,
                    )
                )
            except ValueError as error:
                r = f"not scored: {error}"
            print(f"{band:24s} {name:16s} {r}")
    print("\n== per Primary, A1 minus A0, 95% ==")
    print(
        metrics.reported_contrast_by_primary(
            treatment=rows_c["a1"], reference=rows_c["a0"], metric="crps",
            where=pl.col(HEADLINE), slice_table=table_c,
        ).with_columns(marked=pl.col("label").is_in(MARKED))
    )  # fmt: skip
    print("\n== absolute CRPS, days 1 to 10, per Primary ==")
    print(
        pl.concat(
            [
                t.filter(pl.col("metric") == "crps", pl.col("slice") == HEADLINE).select(
                    "label", "group", pl.lit(n).alias("arm"), "value"
                )
                for n, t in scored.items()
            ]
        )
        .pivot(on="arm", index=["label", "group"], values="value")
        .sort("group", "label")
    )
    print("\n== MAE of the median, days 1 to 10 ==")
    print(
        pl.concat(
            [
                t.filter(pl.col("metric") == "mae_median", pl.col("slice") == HEADLINE).select(
                    "label", pl.lit(n).alias("arm"), "value"
                )
                for n, t in scored.items()
            ]
        )
        .pivot(on="arm", index="label", values="value")
        .sort("label")
    )
    print("\n== tails and twCRPS, mean over Primaries per group ==")
    for name in ("a0", "a1", "stored_control", "stack"):
        t = scored[name].filter(pl.col("slice") == HEADLINE, pl.col("metric").is_in(TAILS))
        print(name)
        print(
            t.group_by("group", "metric").agg(pl.col("value").mean())
            .pivot(on="metric", index="group", values="value").sort("group")
        )  # fmt: skip
    print("\n== tails per Primary, A0 then A1 ==")
    for name in ("a0", "a1"):
        t = scored[name].filter(pl.col("slice") == HEADLINE, pl.col("metric").is_in(TAILS))
        print(name)
        print(t.pivot(on="metric", index="label", values="value").sort("label"))
    print("\n== D12 and D8 bias of the median on sunny half-hours, points, by lead band ==")
    flags = table_c.select(
        "time_series_id", "power_fcst_init_time", "valid_time", *metrics.LEAD_BAND_SLICES,
        "sunny_unit_shape", "sunny_cams_index_07", "sunny_cams_index_09",
    )  # fmt: skip
    for sunny in ("sunny_unit_shape", "sunny_cams_index_07", "sunny_cams_index_09"):
        parts = []
        for name in ("a0", "a1"):
            joined = (
                arms[name]
                .join(flags, on=["time_series_id", "power_fcst_init_time", "valid_time"])
                .join(ctx["actuals"], on=["time_series_id", "valid_time"])
                .join(series.select("time_series_id", "label", "capacity_mw"), on="time_series_id")
                .filter(pl.col("label").is_in(["D12", "D8"]), pl.col(sunny))
                .with_columns(err=(pl.col("q50") - pl.col("actual")) / pl.col("capacity_mw") * 100)
            )
            parts.extend(
                joined.filter(pl.col(band))
                .group_by("label")
                .agg(pl.col("err").mean().alias("bias"), pl.len().alias("n"))
                .with_columns(arm=pl.lit(name), band=pl.lit(band), sunny=pl.lit(sunny))
                for band in metrics.LEAD_BAND_SLICES
            )
        print(pl.concat(parts).sort("sunny", "label", "band", "arm"))
    print("\n== avoidable curtailment cost, signed-MW six, pounds, scored year ==")
    flows = ctx["actuals"].rename({"actual": "flow_mw"})
    cost_forecasts = {
        n: metrics.restrict_to_slice_table(forecast=arms[n], slice_table=table).join(
            ctx["actuals"], on=["time_series_id", "valid_time"]
        )
        for n in ("a0", "a1", "stack", "manual_heuristic")
    }
    signed = series.filter(pl.col("label").is_in(list(SIGNED)))
    key = ["time_series_id", "power_fcst_init_time", "valid_time"]
    row_sets = {
        "all rows": None,
        "during a labelled event": "event_rows",
        "7 days after a labelled event": "event_following_7_days",
    }
    for set_name, flag in row_sets.items():
        subset = (
            cost_forecasts
            if flag is None
            else {
                n: f.join(table.filter(pl.col(flag)).select(key), on=key)
                for n, f in cost_forecasts.items()
            }
        )
        for fraction in (0.5, 0.9):
            print(f"{set_name}, limit fraction {fraction}")
            try:
                cost = metrics.avoidable_curtailment_cost(
                    forecasts=subset, series=signed, flows=flows,
                    scored_start=base.SCORED_FIRST, fraction=fraction,
                )  # fmt: skip
                print(
                    cost.pivot(on="arm", index="label", values="avoidable_cost_gbp").sort("label")
                )
            except ValueError as error:
                print(f"not scored: {error}")


if __name__ == "__main__":
    main()
