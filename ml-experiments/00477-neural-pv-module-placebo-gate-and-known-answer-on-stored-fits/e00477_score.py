# ruff: noqa: E501
"""Re-score the stored arms of experiments 416 and 477 beside the stored references (no refit).

Uses experiment 416's scorer machinery. The arms are seed-averaged in quantile space and scored with
the stored references (the GAM stack, BG2, the manual heuristic, and climatology of experiments 377
and 87) on one shared set of half-hours (row set `fold_a_daily_gap_v1`). Added here: a daylight and
a cloudy slice (CAMS), cold winter nights, Christmas week, the 12 to 36 hour band, the
avoidable-curtailment cost at three limit fractions, and the arm `C3-nolag` for the five fold months.

```bash
. <the environment script that sets DATA_PATH_INTERNAL and PYTHONPATH>
uv run --no-sync python ml-experiments/00477-*/e00477_score.py --set twelve
uv run --no-sync python ml-experiments/00477-*/e00477_score.py --set folds
```
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as s377  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00397_score as s397  # ty: ignore[unresolved-import]
import e00403_rescore as r3  # ty: ignore[unresolved-import]
import e00416_score as s416  # ty: ignore[unresolved-import]
import e00477_inject as inj  # ty: ignore[unresolved-import]
import polars as pl
from studies.eval_protocol import metrics

OUT: Final[Path] = inj.WORK
SEED_DIR_477: Final[Path] = inj.WORK / "seed_preds"
FOLDS: Final[tuple[str, ...]] = tuple(r3.MONTHS)
TWELVE: Final[tuple[str, ...]] = s416.TEST_MONTHS_12
STORED: Final[tuple[str, ...]] = ("A0-doy", "A0-noirr-doy", "C3", "H1", "C3p-weak")
REFERENCES: Final[tuple[str, ...]] = ("stack", "bg2", "manual_heuristic", "climatology")
CLOUDY_CSI: Final[float] = 0.4
SIGNED: Final = list(s416.SIGNED)
CURTAILMENT_FRACTIONS: Final[tuple[float, ...]] = (0.9, 0.75, 0.5)
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("C3", "A0-doy"),
    ("C3", "A0-noirr-doy"),
    ("C3-nolag", "C3"),
)
SLICES: Final[tuple[str, ...]] = ("sunny_midday", "daylight", "cloudy", "lead_days_1_10")
_seed_file = s416.seed_file


def seed_file(*, arm: str, month: str, seed: int) -> Path:
    """Return a seed's forecast file; `C3-nolag` lives in this experiment's folder."""
    if arm == "C3-nolag":
        return SEED_DIR_477 / arm / f"{month}_s{seed}.parquet"
    return _seed_file(arm=arm, month=month, seed=seed)


s416.seed_file = seed_file  # ty: ignore[invalid-assignment]


def cams_flags(*, table: pl.DataFrame, series: pl.DataFrame) -> pl.DataFrame:
    """Return the table with `sunny_midday`, `daylight`, `cloudy`, `christmas_week` flags."""
    cams = (
        pl.read_parquet(
            s416.CAMS_FRAME_PATH, columns=["site", "kind", "time", "ghi_u", "ghi_u_clear"]
        )
        .filter(pl.col("kind") == "substation", pl.col("ghi_u_clear") > 0)
        .select(
            label=pl.col("site"),
            valid_time=pl.col("time"),
            csi=pl.col("ghi_u") / pl.col("ghi_u_clear"),
        )
    )
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    keyed = (
        table.select("time_series_id", "power_fcst_init_time", "valid_time")
        .join(series.select("time_series_id", "label"), on="time_series_id", how="left")
        .join(cams, on=["label", "valid_time"], how="left")
        .with_columns(
            daylight=pl.col("csi").is_finite().fill_null(value=False),
            cloudy=(pl.col("csi") < CLOUDY_CSI).fill_null(value=False),
            christmas_week=(
                ((local.dt.month() == 12) & (local.dt.day() >= 24))
                | ((local.dt.month() == 1) & (local.dt.day() <= 1))
            ),
        )
        .select(
            "time_series_id",
            "power_fcst_init_time",
            "valid_time",
            "daylight",
            "cloudy",
            "christmas_week",
        )
    )
    return table.join(
        keyed,
        on=["time_series_id", "power_fcst_init_time", "valid_time"],
        how="left",
        validate="1:1",
    )


def frames(*, months: tuple[str, ...], arms: tuple[str, ...], series: pl.DataFrame) -> dict:
    """Return each NN arm (seed-averaged, MW) and each reference in MW, as stored."""
    s416.MONTHS[:] = months
    seeds = {a: (0, 1, 2) if a == "C3-nolag" else s416.SEEDS for a in arms}
    out = {
        a: adapter.forecast_frame(
            stored=s416.arm_frame(arm=a, seeds=seeds[a]), series=series, unit="fraction"
        )
        for a in arms
    }
    refs = s397.references(series=series)
    first = datetime(int(months[0][:4]), int(months[0][5:7]), 1, tzinfo=UTC)
    for name in REFERENCES:
        out[name] = refs[name].filter(pl.col("power_fcst_init_time") >= first)
    return out


def build(*, months: tuple[str, ...], arms: tuple[str, ...]) -> tuple:
    """Return series, context, slice table with flags, scored metrics, row losses, and frames."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    arm_frames = frames(months=months, arms=arms, series=series)
    table, scored, rows = s381.score_arms(arms=arm_frames, series=series, ctx=ctx)
    table = s416.sunny_midday_flags(table=table, series=series)
    table = cams_flags(table=table, series=series)
    flags = table.select(
        *metrics.KEY_COLUMNS, "sunny_midday", "daylight", "cloudy", "christmas_week"
    )
    rows = {
        n: r.join(flags, on=list(metrics.KEY_COLUMNS), how="left", validate="m:1")
        for n, r in rows.items()
    }
    return series, ctx, table, scored, rows, arm_frames


def mean_in(*, rows: pl.DataFrame, where: pl.Expr, metric: str) -> float:
    """Return the signed-MW-six mean of a row metric over a row set."""
    kept = rows.filter(pl.col("label").is_in(SIGNED), where)
    return float(kept[metric].mean()) if kept.height else float("nan")  # ty: ignore[invalid-argument-type]


def band_36(*, rows: pl.DataFrame) -> pl.Expr:
    """Return the 12 to 36 hour lead band as an expression."""
    return (pl.col("lead_hours") >= 12) & (pl.col("lead_hours") < 36)


def main() -> None:
    """Print and store the scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("twelve", "folds"), required=True)
    args = parser.parse_args()
    months = TWELVE if args.set == "twelve" else FOLDS
    arms = (*STORED, "C3-nolag") if args.set == "folds" else STORED
    series, ctx, table, _scored, rows, arm_frames = build(months=months, arms=arms)
    names = (*arms, *REFERENCES)
    out: dict = {"months": list(months), "arms": list(names), "slices": {}, "contrasts": {}}
    slices: dict[str, pl.Expr] = {
        "days_1_10": pl.col("lead_days_1_10"),
        "sunny_midday": pl.col("sunny_midday") & pl.col("lead_days_1_10"),
        "daylight": pl.col("daylight") & pl.col("lead_days_1_10"),
        "cloudy": pl.col("cloudy") & pl.col("lead_days_1_10"),
        "band_12_36h": pl.col("lead_hours").is_between(12, 36, closed="left"),
        "cold_nights": pl.col("cold_nights") & pl.col("lead_days_1_10"),
        "christmas_week": pl.col("christmas_week") & pl.col("lead_days_1_10"),
    }
    cols = (
        "crps",
        "pinball_q01",
        "pinball_q05",
        "pinball_q95",
        "pinball_q99",
        "coverage_q01",
        "coverage_q05",
        "coverage_q95",
        "coverage_q99",
    )
    for sl, where in slices.items():
        out["slices"][sl] = {}
        print(f"\n== {sl}: signed-MW six, mean over rows ==")
        for n in names:
            r = {m: mean_in(rows=rows[n], where=where, metric=m) for m in cols}
            r["n"] = int(rows[n].filter(pl.col("label").is_in(SIGNED), where).height)
            out["slices"][sl][n] = r
            print(
                f"{n:14s} CRPS {r['crps']:.3f}  q99 pin {r['pinball_q99']:.3f}  q01 pin {r['pinball_q01']:.3f}  "
                f"cov q01 {r['coverage_q01']:.3f} q05 {r['coverage_q05']:.3f} q95 {r['coverage_q95']:.3f} q99 {r['coverage_q99']:.3f}  n {r['n']}"
            )
    print(
        f"\n== contrasts, level {s416.LEVEL:.2f}% (Bonferroni across 6: 3 contrasts x 2 metrics) =="
    )
    for a, b in CONTRASTS:
        if a not in rows or b not in rows:
            continue
        for sl in SLICES:
            if sl not in table.columns:
                continue
            for metric in s416.PRIMARY_METRICS:
                r = s416.contrast(
                    rows=rows, table=table, a=a, b=b, sl=sl, metric=metric, level=s416.LEVEL
                )
                seeds = (0, 1, 2) if "C3-nolag" in (a, b) else s416.SEEDS
                if sl in s416.SLICES:
                    mde1, mde_avg = s416.minimum_detectable_effect(
                        a=a,
                        b=b,
                        sl=sl,
                        metric=metric,
                        seeds=seeds,
                        series=series,
                        ctx=ctx,
                        table=table,
                    )
                else:
                    mde1 = mde_avg = float("nan")
                key = f"{a} minus {b} [{sl}, {metric}]"
                out["contrasts"][key] = {**dict(r), "mde_single": mde1, "mde_avg": mde_avg}
                print(
                    f"{key}: {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) "
                    f"MDE single {mde1:.3f} avg {mde_avg:.3f}"
                )
    print("\n== avoidable-curtailment cost (GBP), signed-MW six ==")
    flows = ctx["actuals"].rename({"actual": "flow_mw"})
    cost_in = {
        n: metrics.restrict_to_slice_table(forecast=arm_frames[n], slice_table=table).join(
            ctx["actuals"], on=["time_series_id", "valid_time"]
        )
        for n in names
    }
    signed_series = series.filter(pl.col("label").is_in(SIGNED))
    out["curtailment"] = {}
    for fraction in CURTAILMENT_FRACTIONS:
        cost = metrics.avoidable_curtailment_cost(
            forecasts=cost_in,
            series=signed_series,
            flows=flows,
            scored_start=s377.SCORED_FIRST,
            fraction=fraction,
        )
        pivot = cost.pivot(on="arm", index="label", values="avoidable_cost_gbp").sort("label")
        print(f"limit fraction {fraction}")
        print(pivot)
        out["curtailment"][str(fraction)] = pivot.to_dicts()
    suffix = args.set
    (OUT / f"score_{suffix}.json").write_text(json.dumps(out, default=float, indent=1))
    print(f"stored {OUT / f'score_{suffix}.json'}")


if __name__ == "__main__":
    main()
