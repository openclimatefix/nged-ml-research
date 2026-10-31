# ruff: noqa: E501
"""Average the seeds, score the screen folds, and test the attention predictions (397).

Modes:

- `average`: average each arm's seeds in quantile space and store one prediction set per scheme and
  arm (`<scheme>-<arm>`), with the 1.15 widening of the outer quantiles about the median applied
  only at scoring time (the stored sets are raw).
- `folds`: the fold-by-fold table. For each of the five screen months it scores every neural arm
  and the stored references on the same half-hours, with `studies.eval_protocol`. It prints
  pooled quantile score (CRPS from the 11 quantiles, points of capacity) at days 1 to 10 per
  group, raw and widened, then the tails. The screen has five months, so it prints no interval.
- `gate`: the pre-registered predictions W1 to W6 from the gate logs of the attention arms.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand \
uv run --no-sync python ml-experiments/00397-*/e00397_score.py folds --scheme summary
```
"""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00397_models as models  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

from studies import experiment_store

EXPERIMENT_ID: Final[int] = 397
SEED_DIR: Final[Path] = prep.WORK_DIR / "work" / "seed_preds"
E389_ARMS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00389-xgboost-tail-calibration-and-ensemble-spread-columns"
    / "work"
    / "arms"
)
WIDEN: Final[float] = common.SPREAD_FACTOR
OUTER: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
HEADLINE: Final[str] = base.HEADLINE_BAND
LABEL_GROUPS: Final[dict[str, str]] = s381.LABEL_GROUPS
COUNTABLE_EXCLUDED: Final[tuple[str, ...]] = s381.COUNTABLE_EXCLUDED
WEIGHT_GROUPS: Final[tuple[str, ...]] = models.GROUPS
LEAD_BANDS_DAYS: Final[dict[str, tuple[float, float]]] = {
    "day 1": (0.0, 24.0),
    "days 2-3": (24.0, 72.0),
    "days 4-7": (72.0, 168.0),
    "days 8-10": (168.0, 240.0),
}
"""Lead bands for the gate weights, in hours."""
W_MARGIN: Final[float] = 0.05
CHRISTMAS: Final[tuple[str, str]] = ("12-24", "01-01")


def widen(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Widen q01, q05, q95, and q99 about the median by the spread factor 1.15."""
    return frame.with_columns(
        *[(pl.col("q50") + WIDEN * (pl.col(c) - pl.col("q50"))).alias(c) for c in OUTER]
    )


def seed_files(*, scheme: str, arm: str, month: str) -> list[Path]:
    """Return a fold's per-seed forecast files, excluding the gate logs."""
    folder = SEED_DIR / scheme / arm
    return sorted(p for p in folder.glob(f"{month}_s*.parquet") if not p.stem.endswith("_gate"))


def average_arm(*, scheme: str, arm: str, months: list[str]) -> pl.DataFrame:
    """Return an arm's seed-averaged forecast (fraction of capacity), months stacked."""
    keys = list(common.KEYS)
    pieces = []
    for month in months:
        files = seed_files(scheme=scheme, arm=arm, month=month)
        if not files:
            continue
        stack = pl.concat([pl.read_parquet(f) for f in files])
        pieces.append(
            stack.group_by(keys)
            .agg(*[pl.col(c).mean() for c in common.LEVEL_COLUMNS])
            .sort(keys)
            .with_columns(pl.col(*common.LEVEL_COLUMNS).cast(pl.Float32))
        )
    if not pieces:
        msg = f"no forecasts for {scheme} {arm}"
        raise FileNotFoundError(msg)
    out = pl.concat(pieces)
    # Quantiles of a mean across seeds can cross only through rounding; sort across levels.
    values = np.sort(out.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    return out.with_columns(
        *[pl.Series(c, values[:, i]) for i, c in enumerate(common.LEVEL_COLUMNS)]
    )


def average(*, scheme: str, arms: list[str], months: list[str]) -> None:
    """Store one prediction set per arm: the seed average over the screen months."""
    for arm in arms:
        frame = average_arm(scheme=scheme, arm=arm, months=months).filter(
            pl.col("valid_time") < common.HOLDOUT
        )
        path = experiment_store.write_predictions(
            experiment_id=EXPERIMENT_ID,
            name=f"{scheme}-{arm.lower().replace('_', '-')}",
            frame=frame,
            kind="quantiles",
            row_set_id=common.ROW_SET_ID,
        )
        print(f"{arm}: {frame.height:,} rows -> {path}")


def references(*, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the stored references in MW: XGBoost (weekly, monthly), the stack, BG2, and the rest."""
    out: dict[str, pl.DataFrame] = {
        "xgb_weekly_e381": s381.arm_frame(
            regime="weekly", arm="bundle_solar+sunny+hl1", series=series
        ),
        "xgb_monthly_e381": s381.arm_frame(regime="monthly", arm="control", series=series),
    }
    for name, label in (("S0M_raw", "xgb_monthly_e389"), ("S1M_raw", "xgb_monthly_spread_e389")):
        stored = pl.read_parquet(E389_ARMS / f"{name}.parquet")
        out[label] = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    out["xgb_monthly_e389_widened"] = widen(frame=out["xgb_monthly_e389"])
    out["stack"] = base.stack_frames(series=series)["stack"]
    out.update(base.reference_frames(series=series))
    return out


def month_origins(*, month: str) -> set:
    """Return the set of `power_fcst_init_time` values of a test month."""
    return set(prep.test_origins(month=month))


def only_month(*, frame: pl.DataFrame, month: str) -> pl.DataFrame:
    """Return a frame's rows whose forecast origin lies in the month."""
    return frame.filter(pl.col("power_fcst_init_time").is_in(list(month_origins(month=month))))


def group_means(*, scored: pl.DataFrame, metric: str, slice_name: str) -> dict[str, float]:
    """Return the equal-weight mean over countable Primaries of a metric in each group."""
    table = scored.filter(
        pl.col("metric") == metric,
        pl.col("slice") == slice_name,
        ~pl.col("label").is_in(COUNTABLE_EXCLUDED),
    )
    signed = table.filter(pl.col("group") == "signed-MW")["value"].to_numpy()
    mva = table.filter(pl.col("group") == "MVA-only")["value"].to_numpy()
    return {"signed-MW six": float(signed.mean()), "MVA-only eight": float(mva.mean())}


def folds(*, scheme: str, arms: list[str], months: list[str]) -> None:
    """Print the fold-by-fold table of every arm and reference."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    refs = references(series=series)
    neural = {
        f"nn_{arm}": adapter.forecast_frame(
            stored=average_arm(scheme=scheme, arm=arm, months=months),
            series=series,
            unit="fraction",
        )
        for arm in arms
    }
    neural_widened = {f"{k}_w": widen(frame=v) for k, v in neural.items()}
    all_arms = {**neural, **neural_widened, **refs}
    table: list[pl.DataFrame] = []
    tails: list[pl.DataFrame] = []
    for month in months:
        arms_month = {k: only_month(frame=v, month=month) for k, v in all_arms.items()}
        shared, scored, _ = s381.score_arms(arms=arms_month, series=series, ctx=ctx)
        del shared
        for name, frame in scored.items():
            for group, value in group_means(
                scored=frame, metric="crps", slice_name=HEADLINE
            ).items():
                table.append(
                    pl.DataFrame({"fold": month, "arm": name, "group": group, "crps": value})
                )
            for metric in ("coverage_q95", "coverage_q99", "pinball_q95", "pinball_q99"):
                for group, value in group_means(
                    scored=frame, metric=metric, slice_name=HEADLINE
                ).items():
                    tails.append(
                        pl.DataFrame(
                            {
                                "fold": month,
                                "arm": name,
                                "group": group,
                                "metric": metric,
                                "value": value,
                            }
                        )
                    )
    crps = pl.concat(table)
    with pl.Config(tbl_rows=200, tbl_cols=20):
        for group in ("signed-MW six", "MVA-only eight"):
            print(f"\n== quantile score (CRPS), days 1 to 10, {group}, points ==")
            print(
                crps.filter(pl.col("group") == group)
                .pivot(on="fold", index="arm", values="crps")
                .with_columns(mean=pl.mean_horizontal(pl.exclude("arm")))
                .sort("mean")
            )
        print("\n== tails, days 1 to 10 (coverage nominal 0.95 / 0.99; pinball in points) ==")
        print(
            pl.concat(tails)
            .group_by("arm", "group", "metric")
            .agg(pl.col("value").mean())
            .pivot(on="metric", index=["arm", "group"], values="value")
            .sort("group", "arm")
        )


def gate_logs(*, scheme: str, arm: str, months: list[str]) -> pl.DataFrame:
    """Return the gate logs of an attention arm's first seed per month, with local-date columns."""
    pieces = []
    for month in months:
        files = sorted((SEED_DIR / scheme / arm).glob(f"{month}_s*_gate.parquet"))
        if files:
            pieces.append(pl.concat([pl.read_parquet(f) for f in files]))
    frame = pl.concat(pieces)
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    return frame.with_columns(
        day_of_year_key=local.dt.strftime("%m-%d"),
        month_key=local.dt.strftime("%Y-%m"),
        hour_local=local.dt.hour(),
        weekday=local.dt.weekday(),
    )


def band_mean(*, logs: pl.DataFrame, column: str, lo: float, hi: float) -> float:
    """Return the mean of a gate column at leads in (lo, hi] hours."""
    kept = logs.filter(pl.col("lead_hours").is_between(lo, hi, closed="right"))
    return float(kept[column].to_numpy().astype(np.float64).mean())


def gate(*, scheme: str, months: list[str]) -> None:
    """Print W1 to W6 for the attention arms, and say whether each held."""
    for arm in ("A1", "A2", "A3"):
        try:
            logs = gate_logs(scheme=scheme, arm=arm, months=months)
        except ValueError, FileNotFoundError:
            continue
        print(f"\n== {arm}: mean gate weight by lead band ==")
        rows = []
        for name, (lo, hi) in LEAD_BANDS_DAYS.items():
            rows.append(
                {"band": name}
                | {g: band_mean(logs=logs, column=f"w_{g}", lo=lo, hi=hi) for g in WEIGHT_GROUPS}
            )
        bands = pl.DataFrame(rows)
        print(bands)
        day1, late = bands.row(0, named=True), bands.row(-1, named=True)
        print(f"W1 weather falls: {day1['weather'] - late['weather']:+.3f} (needs >= {W_MARGIN})")
        print(f"W2 lags fall: {day1['lags'] - late['lags']:+.3f} (needs >= {W_MARGIN})")
        rise = (late["calendar"] + late["holidays"]) - (day1["calendar"] + day1["holidays"])
        print(f"W3 calendar + holidays rise: {rise:+.3f} (needs >= {W_MARGIN})")
        december = logs.filter(pl.col("month_key").str.ends_with("-12"))
        xmas = december.filter(pl.col("day_of_year_key") >= CHRISTMAS[0])
        rest = december.filter(pl.col("day_of_year_key") < CHRISTMAS[0])
        if xmas.height and rest.height:
            gain = float(
                xmas["w_holidays"].to_numpy().mean() - rest["w_holidays"].to_numpy().mean()
            )
            print(f"W4 holiday weight, 24-31 Dec minus 1-23 Dec: {gain:+.3f} (needs >= {W_MARGIN})")
        spans = {
            g: max(
                band_mean(logs=logs, column=f"w_{g}", lo=lo, hi=hi)
                for lo, hi in LEAD_BANDS_DAYS.values()
            )
            - min(
                band_mean(logs=logs, column=f"w_{g}", lo=lo, hi=hi)
                for lo, hi in LEAD_BANDS_DAYS.values()
            )
            for g in WEIGHT_GROUPS
        }
        print(f"W6 range over lead bands per group: {spans} (needs one >= {W_MARGIN})")
    print("W5 (bank holidays) is read on the full scored year only.")


MODES: Final[dict[str, Callable[..., None]]] = {"average": average, "folds": folds, "gate": gate}


def main() -> None:
    """Run a mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(MODES))
    parser.add_argument("--scheme", choices=prep.SCHEMES, default="summary")
    parser.add_argument("--arms", nargs="+", default=["A0", "A2"])
    parser.add_argument("--months", nargs="+", default=list(prep.SCREEN_MONTHS))
    args = parser.parse_args()
    if args.mode == "gate":
        gate(scheme=args.scheme, months=args.months)
    else:
        MODES[args.mode](scheme=args.scheme, arms=args.arms, months=args.months)


if __name__ == "__main__":
    main()
