# ruff: noqa: E501, PLR0915
"""Score experiment 383's arms under demand-v1 and print every number quoted.

Arms (from `e00383_assemble.py`): `d`, `c1`, `c2`, each raw (`<arm>`), widened by the fixed factor
(`<arm>_w`), and widened by the as-of factor (`<arm>_wa`); references `bg2`, `manual_heuristic`,
and `climatology` from experiment 87's stored arms. The scoring machinery is experiment 379's.

Headline rows: the contaminated-lag rows (a lag inside an event) that lie outside the events,
days 1 to 10. Prints, per group and arm: CRPS, tail pinball loss and coverage, the 5% to 95% and
1% to 99% band coverage and mean width, then the pre-registered contrasts with their intervals and
the gate-O checks, the placebo identity check, the event and 7-days-after slices, the by-days-
since-onset split, the Christmas split, the per-event table, and the avoidable-curtailment cost.

```bash
PYTHONPATH=ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:ml-experiments/00383-* \
    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00383-*/e00383_score.py
```
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00379_adapter as adapter
import e00379_score as ref379
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.eval_protocol.slices import KEY_COLUMNS, MVA_ONLY_LABELS, SIGNED_MW_LABELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00383-load-shift-xgboost-oracle-first-part-1" / "work"
)
ARMS_DIR: Final[Path] = WORK / "arms"
OUT: Final[Path] = WORK / "scores"
SCORED_FIRST: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
MAX_LEAD_HOURS: Final[float] = 336.0
HEADLINE_BAND: Final[str] = "lead_days_1_10"
OWN_ARMS: Final[tuple[str, ...]] = (
    "d", "c1", "c2", "d_w", "c1_w", "c2_w", "d_wa", "c1_wa", "c2_wa",
)  # fmt: skip
REFERENCE_ARMS: Final[tuple[str, ...]] = ("bg2", "manual_heuristic", "climatology")
PLACEBOS: Final[tuple[str, ...]] = ("D2", "D9", "D11", "D21")
TARGETS: Final[tuple[str, ...]] = ("D3", "D4", "D5", "D10", "D12", "D14", "D16", "D19", "D22")
N_CONTRASTS: Final[int] = 6
BONFERRONI_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / N_CONTRASTS)
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("c1_w", "d"), ("c2_w", "d"), ("c1", "d"), ("c2", "d"), ("d_w", "d"), ("c2_wa", "d"),
)  # fmt: skip
GATE_GROUP_TOLERANCE: Final[float] = 0.02
GATE_PRIMARY_TOLERANCE: Final[float] = 0.05
BAND_NOMINAL: Final[float] = 0.90
CHRISTMAS_START: Final[tuple[int, int]] = (12, 20)
CHRISTMAS_END: Final[tuple[int, int]] = (1, 10)


def load_arms(*, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return every arm's forecast frame in MW, leads to `MAX_LEAD_HOURS`."""
    ids = series["time_series_id"].to_list()
    out: dict[str, pl.DataFrame] = {}
    for name in OWN_ARMS:
        stored = pl.read_parquet(ARMS_DIR / f"{name}.parquet").filter(
            pl.col("lead_hours") <= MAX_LEAD_HOURS
        )
        out[name] = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    for name in REFERENCE_ARMS:
        file = ref379.STORED[name]
        stored = pl.read_parquet(ref379.STORED_87 / f"{file}.parquet").filter(
            pl.col("time_series_id").is_in(ids), pl.col("lead_hours") <= MAX_LEAD_HOURS
        )
        out[name] = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    return out


def add_band_columns(
    *, rows: pl.DataFrame, forecast: pl.DataFrame, series: pl.DataFrame
) -> pl.DataFrame:
    """Add the 5% to 95% and 1% to 99% band coverage (0 or 1) and width (points) to `rows`."""
    widths = forecast.join(
        series.select("time_series_id", "capacity_mw"), on="time_series_id"
    ).select(
        *KEY_COLUMNS,
        width_90=(pl.col("q95") - pl.col("q05")) * 100.0 / pl.col("capacity_mw"),
        width_98=(pl.col("q99") - pl.col("q01")) * 100.0 / pl.col("capacity_mw"),
    )
    return rows.join(widths, on=list(KEY_COLUMNS), how="left", validate="1:1").with_columns(
        band_90=pl.col("coverage_q95") - pl.col("coverage_q05"),
        band_98=pl.col("coverage_q99") - pl.col("coverage_q01"),
        pinball_tail_up=(pl.col("pinball_q95") + pl.col("pinball_q99")) / 2.0,
    )


def summarise(*, rows: dict[str, pl.DataFrame], where: pl.Expr, by: str = "group") -> pl.DataFrame:
    """Return, per `by` value and arm, the mean losses and band measures over the filtered rows."""
    metric_names = (
        "crps", "pinball_q01", "pinball_q05", "pinball_q95", "pinball_q99", "pinball_tail_up",
        "coverage_q01", "coverage_q05", "coverage_q95", "coverage_q99", "band_90", "band_98",
        "width_90", "width_98",
    )  # fmt: skip
    pieces = [
        frame.filter(where)
        .group_by(by)
        .agg(pl.len().alias("n"), *[pl.col(m).mean().alias(m) for m in metric_names])
        .with_columns(arm=pl.lit(name))
        for name, frame in rows.items()
    ]
    return pl.concat(pieces).sort(by, "arm")


def christmas(*, valid: pl.Expr) -> pl.Expr:
    """Return the expression true on the valid times from 20 December to 10 January."""
    return ((valid.dt.month() == CHRISTMAS_START[0]) & (valid.dt.day() >= CHRISTMAS_START[1])) | (
        (valid.dt.month() == CHRISTMAS_END[0]) & (valid.dt.day() <= CHRISTMAS_END[1])
    )


def main() -> None:
    """Print the scores."""
    pl.Config.set_tbl_rows(400)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(260)
    OUT.mkdir(exist_ok=True)
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=ref379.CACHE_DIR)
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
    arms = load_arms(series=series)
    table = metrics.shared_slice_table(
        forecasts=arms,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    )
    flags = pl.read_parquet(ARMS_DIR / "contaminated.parquet").with_columns(
        pl.col("time_series_id").cast(series.schema["time_series_id"])
    )
    print(f"arms: {list(arms)}; shared half-hours: {table.height:,}")
    table = table.join(flags, on=list(KEY_COLUMNS), how="left", validate="1:1").with_columns(
        pl.col("lag_in_event").fill_null(value=False)
    )
    rows: dict[str, pl.DataFrame] = {}
    for name, frame in arms.items():
        kept = metrics.restrict_to_slice_table(
            forecast=frame, slice_table=table.drop("lag_in_event")
        )
        scored = metrics.scored_rows(
            forecast=kept,
            actuals=actuals,
            series=series,
            thresholds=thresholds,
            slice_table=table.drop("lag_in_event"),
        )
        scored = add_band_columns(rows=scored, forecast=kept, series=series).join(
            table.select(*KEY_COLUMNS, "lag_in_event"), on=list(KEY_COLUMNS), validate="1:1"
        )
        rows[name] = scored
    pl.concat([r.with_columns(arm=pl.lit(n)) for n, r in rows.items()]).write_parquet(
        OUT / "rows.parquet"
    )
    outside = ~pl.col("event_rows") & pl.col(HEADLINE_BAND)
    headline = outside & pl.col("lag_in_event")
    ordinary = outside & ~pl.col("lag_in_event")
    print("\n== Row counts, days 1 to 10, outside events, per group ==")
    first = rows["d"].filter(outside)
    print(
        first.group_by("group")
        .agg(all_rows=pl.len(), contaminated=pl.col("lag_in_event").sum())
        .sort("group")
    )
    print("\n== HEADLINE: contaminated-lag rows, days 1 to 10, outside events, by group ==")
    print(summarise(rows=rows, where=headline))
    print("\n== Ordinary rows (no lag inside an event), days 1 to 10, outside events, by group ==")
    print(summarise(rows=rows, where=ordinary))
    print("\n== Main score: all rows outside events, days 1 to 10, by group ==")
    print(summarise(rows=rows, where=outside))
    print("\n== Main score by Primary (CRPS, days 1 to 10, outside events) ==")
    print(
        pl.concat(
            [
                f.filter(outside)
                .group_by("label")
                .agg(pl.col("crps").mean())
                .with_columns(arm=pl.lit(n))
                for n, f in rows.items()
            ]
        )
        .pivot(on="arm", index="label", values="crps")
        .sort("label")
    )
    print("\n== Contaminated-lag rows by Primary (CRPS, n) ==")
    contaminated = pl.concat(
        [
            f.filter(headline)
            .group_by("label")
            .agg(pl.col("crps").mean(), n=pl.len())
            .with_columns(arm=pl.lit(n))
            for n, f in rows.items()
            if n in ("d", "c1", "c2", "d_w", "c1_w", "c2_w")
        ]
    )
    print(
        contaminated.pivot(on="arm", index=["label"], values="crps")
        .join(contaminated.filter(pl.col("arm") == "d").select("label", "n"), on="label")
        .sort("label")
    )
    print(
        "\n== Placebo identity: c2, d_w, c2_w, c2_wa against d at the placebos (max |q50 gap|, points) =="
    )
    for arm in ("c2", "c2_w", "c2_wa", "d_w", "d_wa"):
        gap = (
            arms[arm]
            .join(arms["d"], on=list(KEY_COLUMNS), suffix="_d")
            .join(series.select("time_series_id", "label"), on="time_series_id")
            .filter(pl.col("label").is_in(list(PLACEBOS)))
            .select((pl.col("q05") - pl.col("q05_d")).abs().max())
            .item()
        )
        print(f"{arm}: max |q05 gap| at placebos = {gap:.6f} MW")
    # Contrasts
    print(
        f"\n== Contrasts on the contaminated-lag rows, 95% Bonferroni over {N_CONTRASTS} (level {BONFERRONI_LEVEL:.2f}%) =="
    )
    for group_name, labels in (("signed-MW", SIGNED_MW_LABELS), ("MVA-only", MVA_ONLY_LABELS)):
        where_group = pl.col("label").is_in(list(labels))
        for treatment, reference in CONTRASTS:
            for metric in ("crps", "pinball_tail_up"):
                result = metrics.month_bootstrap_contrast(
                    treatment=rows[treatment].with_columns(pl.col("lag_in_event")),
                    reference=rows[reference],
                    metric=metric,
                    where=headline & where_group,
                    level_percent=BONFERRONI_LEVEL,
                    n_resamples=5000,
                )
                print(
                    f"{group_name} {treatment} - {reference} {metric}: "
                    f"{result['difference']:+.4f} ({result['lower']:+.4f} to {result['upper']:+.4f}), "
                    f"n={result['n_half_hours']:,} months={result['n_months']}"
                )
    print(
        "\n== Harm: main score (outside events, days 1 to 10), arm minus d, 90% interval (one-sided 95% bound = upper) =="
    )
    for group_name, labels in (("signed-MW", SIGNED_MW_LABELS), ("MVA-only", MVA_ONLY_LABELS)):
        for treatment in ("c1_w", "c2_w", "c1", "c2"):
            result = metrics.month_bootstrap_contrast(
                treatment=rows[treatment],
                reference=rows["d"],
                metric="crps",
                where=outside & pl.col("label").is_in(list(labels)),
                level_percent=90.0,
                n_resamples=5000,
            )
            print(
                f"{group_name} {treatment} - d: {result['difference']:+.4f} "
                f"({result['lower']:+.4f} to {result['upper']:+.4f})"
            )
    print("\n== Per Primary: main score difference (point estimate), arm minus d ==")
    per_primary = pl.concat(
        [
            f.filter(outside)
            .group_by("label")
            .agg(pl.col("crps").mean())
            .with_columns(arm=pl.lit(n))
            for n, f in rows.items()
            if n in ("d", "c1", "c2", "c1_w", "c2_w")
        ]
    ).pivot(on="arm", index="label", values="crps")
    print(
        per_primary.with_columns(
            *[(pl.col(a) - pl.col("d")).alias(f"{a}-d") for a in ("c1", "c2", "c1_w", "c2_w")]
        )
        .select("label", "d", pl.col("^.*-d$"))
        .sort("label")
    )
    print("\n== Load-shift slices (CRPS, days 1 to 10), by group ==")
    for sl in ("event_rows", "event_following_7_days"):
        print(sl)
        print(
            summarise(rows=rows, where=pl.col(sl) & pl.col(HEADLINE_BAND)).select(
                "group", "arm", "n", "crps", "pinball_q95", "pinball_q99", "band_90", "width_90"
            )
        )
    print("\n== Christmas split on the headline rows (20 Dec to 10 Jan against the rest) ==")
    xmas = christmas(valid=pl.col("valid_time"))
    for label, cond in (("inside", xmas), ("outside", ~xmas)):
        print(label)
        print(
            summarise(
                rows={
                    k: v for k, v in rows.items() if k in ("d", "c1", "c2", "d_w", "c1_w", "c2_w")
                },
                where=headline & cond,
            ).select("group", "arm", "n", "crps", "pinball_tail_up", "band_90", "band_98")
        )
    print("\n== Headline rows by days since the event ended or began are in e00383_events.py ==")
    cost_report(rows=rows, arms=arms, series=series, actuals=actuals, table=table)


def cost_report(
    *,
    rows: dict[str, pl.DataFrame],
    arms: dict[str, pl.DataFrame],
    series: pl.DataFrame,
    actuals: pl.DataFrame,
    table: pl.DataFrame,
) -> None:
    """Print the avoidable curtailment cost at the signed-MW six in three row sets."""
    del rows
    flows = actuals.rename({"actual": "flow_mw"})
    signed = series.filter(pl.col("label").is_in(list(SIGNED_MW_LABELS)))
    chosen = ("d", "c1", "c2", "d_w", "c1_w", "c2_w", "manual_heuristic", "climatology")
    sets = {
        "during events": pl.col("event_rows"),
        "7 days after events": pl.col("event_following_7_days"),
        "all rows": pl.lit(value=True),
    }
    for fraction in (0.9,):
        for name, cond in sets.items():
            keys = table.filter(cond).select(*KEY_COLUMNS)
            forecasts = {
                n: arms[n]
                .join(keys, on=list(KEY_COLUMNS), how="semi")
                .join(actuals, on=["time_series_id", "valid_time"])
                for n in chosen
            }
            parts = []
            for label in SIGNED_MW_LABELS:
                one = signed.filter(pl.col("label") == label)
                try:
                    parts.append(
                        metrics.avoidable_curtailment_cost(
                            forecasts=forecasts,
                            series=one,
                            flows=flows,
                            scored_start=SCORED_FIRST,
                            fraction=fraction,
                        )
                    )
                except ValueError as error:
                    print(f"{label}: {name}: no cost ({error})")
            cost = pl.concat(parts)
            print(cost.pivot(on="arm", index="label", values="avoidable_cost_gbp").sort("label"))


if __name__ == "__main__":
    main()
