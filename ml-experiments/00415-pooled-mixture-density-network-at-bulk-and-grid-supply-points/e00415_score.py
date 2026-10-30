# ruff: noqa: E501, PLR0915
"""Score the pooled-network arms of experiment 415 against the stored references (no refit).

Reads the per-seed forecasts of `e00415_fit.py`, averages the seeds in quantile space, restricts
every arm to the half-hours they share inside the fitted test months, and prints:

- absolute CRPS at days 1 to 10 per node, with the stored references of experiment 386 (BG2, the
  monthly XGBoost control raw and widened by 1.15, the GAM stack, `manual_heuristic`, and
  `climatology`);
- the pre-registered contrasts per group (BSPs and GSPs apart), by the month-block bootstrap of
  `studies.eval_protocol` at 95% (interim, because a few test months give few resamples);
- the tail measures (pinball loss and coverage at q01, q05, q95, q99) per node and arm;
- the CRPS per fold, the seed spread of P0 (the refit noise floor), and the CRPS at the 14
  countable Primaries for the contrast N1 minus P0.

```bash
PYTHONPATH=ml-experiments/00386-bsp-gsp-demand-v1-top-down-bg2-gam-simple-xgboost:\
ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00415-pooled-mixture-density-network-at-bulk-and-grid-supply-points \
uv run --no-sync python ml-experiments/00415-*/e00415_score.py
```
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00386_adapter as adapter  # ty: ignore[unresolved-import]
import e00386_common as common386  # ty: ignore[unresolved-import]
import e00386_score_a as stage_a  # ty: ignore[unresolved-import]
import e00386_score_b as stage_b  # ty: ignore[unresolved-import]
import e00386d_score as stage_d  # ty: ignore[unresolved-import]
import e00415_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics

SEED_DIR: Final[Path] = prep.WORK_DIR / "seed_preds"
HEADLINE: Final[str] = stage_a.HEADLINE_BAND
KEYS: Final[list[str]] = list(common.KEYS)
GROUPS: Final[dict[str, tuple[str, ...]]] = {"BSP": ("D13", "D17"), "GSP": ("D1", "D6")}
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("N1", "P0"),
    ("N1", "xgb_control_w"),
    ("N1", "bg2"),
    ("N2", "N1"),
    ("N2rw", "N1"),
    ("N2", "N2rw"),
    ("P0", "xgb_control_w"),
    ("N1", "manual_heuristic"),
    ("N1", "climatology"),
)


def seed_files(*, arm: str) -> dict[int, list[Path]]:
    """Return an arm's forecast files grouped by seed."""
    out: dict[int, list[Path]] = {}
    for f in sorted((SEED_DIR / arm).glob("*_s*.parquet")):
        out.setdefault(int(f.stem.rsplit("_s", 1)[1]), []).append(f)
    return out


def seed_average(*, files: list[Path]) -> pl.DataFrame:
    """Return the mean over files of the quantiles (sorted again), keyed by series, run, and time."""
    frame = (
        pl.concat([pl.read_parquet(f) for f in files])
        .group_by(KEYS)
        .agg(*[pl.col(c).mean() for c in common.LEVEL_COLUMNS])
        .sort(KEYS)
    )
    values = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    return frame.with_columns(
        *[pl.Series(c, values[:, i]).cast(pl.Float32) for i, c in enumerate(common.LEVEL_COLUMNS)]
    )


def arm_frames(*, arm: str) -> dict[str, pl.DataFrame]:
    """Return an arm's seed-averaged forecast, and each seed's own forecast if there are several."""
    by_seed = seed_files(arm=arm)
    if not by_seed:
        return {}
    out = {arm: seed_average(files=[f for fs in by_seed.values() for f in fs])}
    if len(by_seed) > 1:
        for seed, files in by_seed.items():
            out[f"{arm}_s{seed}"] = seed_average(files=files)
    return out


def score_nodes() -> None:
    """Print the node scores, contrasts, and tails."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    adapter.register_groups()
    series = adapter.series_table()
    cache = common386.load_cache()
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
        history=actuals.rename({"actual": "power_mw"}),
        series=series,
        before=common386.SCORED_FIRST,
    )
    node_ids = series["time_series_id"].to_list()
    mdn: dict[str, pl.DataFrame] = {}
    for arm in prep.ARMS:
        for name, frame in arm_frames(arm=arm).items():
            kept = frame.filter(pl.col("time_series_id").is_in(node_ids))
            mdn[name] = adapter.forecast_frame(stored=kept, series=series, unit="fraction")
    if not mdn:
        msg = "no forecast files found"
        raise SystemExit(msg)
    months = sorted(
        {f.stem.split("_s")[0] for a in prep.ARMS for f in (SEED_DIR / a).glob("*.parquet")}
    )
    first = pl.col("power_fcst_init_time")
    in_months = pl.any_horizontal(*[first.dt.strftime("%Y-%m") == m for m in months])
    raw = {n: stage_d.xgboost_frame(arm=n, series=series) for n in stage_d.ARMS}
    references = {
        "xgb_control": raw["control"],
        "xgb_control_w": stage_d.widen(frame=raw["control"], factor=stage_d.WIDENING),
        "stack": stage_b.stack_frame(series=series),
        **stage_a.reference_frames(series=series),
    }
    arms = {n: f.filter(in_months) for n, f in {**mdn, **references}.items()}
    table = metrics.shared_slice_table(
        forecasts=arms,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    )
    print(f"test months {months}; shared half-hours of {len(arms)} arms: {table.height}")
    scored, rows = {}, {}
    for name, frame in arms.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        scored[name] = metrics.score_forecast(
            forecast=kept, actuals=actuals, series=series, slice_table=table, thresholds=thresholds
        )
        rows[name] = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        )
    crps = pl.concat(
        [
            t.filter(pl.col("metric") == "crps", pl.col("slice") == HEADLINE).select(
                "label", pl.lit(n).alias("arm"), "value"
            )
            for n, t in scored.items()
        ]
    ).pivot(on="arm", index="label", values="value")
    print("\n== CRPS days 1 to 10, points of effective capacity (all test months) ==")
    print(crps.sort("label"))
    print("\n== CRPS days 1 to 10 per test month (rows are scored half-hours) ==")
    for name in sorted(n for n in rows if n in prep.ARMS):
        per_month = (
            rows[name]
            .filter(pl.col(HEADLINE))
            .group_by("group", first.dt.strftime("%Y-%m").alias("month"))
            .agg(pl.col("crps").mean(), n=pl.len())
            .sort("group", "month")
        )
        print(name)
        print(per_month)
    print("\n== Refit noise floor: P0 CRPS by seed, days 1 to 10 ==")
    for name in sorted(n for n in scored if n.startswith("P0_s")):
        print(
            name,
            scored[name]
            .filter(pl.col("metric") == "crps", pl.col("slice") == HEADLINE)
            .select("label", "value")
            .sort("label")["value"]
            .round(3)
            .to_list(),
        )
    print("\n== Contrasts (treatment minus reference), 95% month-block bootstrap, interim ==")
    for treatment, reference in CONTRASTS:
        if treatment not in rows or reference not in rows:
            continue
        for group in GROUPS:
            r = metrics.month_bootstrap_contrast(
                treatment=rows[treatment],
                reference=rows[reference],
                metric="crps",
                where=(pl.col("group") == group) & pl.col(HEADLINE),
                level_percent=95.0,
                n_resamples=20_000,
                slice_table=table,
            )
            print(
                f"{treatment:6s} - {reference:18s} {group} {r['difference']:+.3f} "
                f"({r['lower']:+.3f} to {r['upper']:+.3f}) n={r['n_half_hours']}"
            )
    print("\n== Tails, days 1 to 10, per node ==")
    for name in [n for n in scored if n in prep.ARMS] + ["bg2", "xgb_control_w", "climatology"]:
        print(name)
        print(
            scored[name]
            .filter(pl.col("slice") == HEADLINE, pl.col("metric").is_in(stage_a.TAILS))
            .pivot(on="metric", index=["label", "group"], values="value")
            .sort("group", "label")
        )
    out = prep.WORK_DIR / "scores"
    out.mkdir(parents=True, exist_ok=True)
    pl.concat([t.with_columns(arm=pl.lit(n)) for n, t in scored.items()]).write_parquet(
        out / "node_scores.parquet"
    )
    scored_per_series = table.group_by("label").len().sort("label")
    bg2_per_series = (
        arms["bg2"]
        .join(series.select("time_series_id", "label"), on="time_series_id")
        .group_by("label")
        .len()
        .sort("label")
    )
    print(
        "\n== Half-hours without a cleaned actual or an arm's forecast (BG2 rows minus scored) =="
    )
    print(
        bg2_per_series.join(scored_per_series, on="label", suffix="_scored").with_columns(
            dropped=pl.col("len") - pl.col("len_scored")
        )
    )


def main() -> None:
    """Score the nodes."""
    score_nodes()


if __name__ == "__main__":
    main()
