# ruff: noqa: E501
r"""Descriptive tables, checks, and explanation tests for experiment 504 (reads the stored scores).

Writes CSV files under `work/scores/report/` and prints them:

- `slices.csv`: each arm minus K3 on the eight, the signed-MW six, D12 alone, the eight without D12,
  cold winter nights, the 12 to 36 hour band, sunny-midday rows (CAMS clear-sky index at least 0.7,
  valid times after 10:00 and up to 15:00 UTC), and lead bands of days 1, 2 to 5, and 6 to 10 (95%
  origin-day block bootstrap, not Bonferroni-corrected; descriptive);
- `per_primary.csv`: each arm minus K3 at each of the eight;
- `median_checks.csv`: the mean absolute error of the median forecast beside CRPS, and the mean of
  the 11 stored quantiles minus the median, per group (the stored forecasts hold quantiles only);
- `val_nll.csv`: the validation loss of the fine-tuned A-cams network against the stored arm A;
- `autocorrelation.csv`: the correlation between the daily cloudiness of the CAMS day D and of the
  day D + lag, over the training years, per lag.

```bash
export PYTHONPATH=$(ls -d $PWD/ml-experiments/*/ | tr '\n' ':')
uv run --no-sync python ml-experiments/00504-*/e00504_report.py
```
"""

import json
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep397  # ty: ignore[unresolved-import]
import e00454_score as score454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00504_cams as cams_mod  # ty: ignore[unresolved-import]
import e00504_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

ARMS: Final[tuple[str, ...]] = ("A-cams", "A-cams-real", "A-cams-lag-only")
OUT: Final = score.WORK_DIR / "scores" / "report"
SUNNY_INDEX: Final[float] = 0.7
RESAMPLES: Final[int] = 20_000


def rows_with_context() -> pl.DataFrame:
    """Return the stored row table with the target time, site, and CAMS clear-sky index added."""
    table = score470.load_rows().filter(pl.col("main16"))
    grid = common.grid_times()
    table = table.with_columns(valid_time=grid.gather(table["target"].to_numpy().astype(np.int64)))
    primaries = prep397.primaries()
    sunny = np.full(table.height, np.nan)
    sites = dict(zip(primaries["primary"].to_list(), primaries.iter_rows(named=True), strict=True))
    for p, row in sites.items():
        sel = (table["primary"] == p).to_numpy()
        cams = score.cams_for(row=row)
        valid_us = table.filter(pl.col("primary") == p)["valid_time"].dt.epoch("us").to_numpy()
        index = (valid_us - cams.first_us) // cams_mod.STEP_US
        sunny[sel] = cams.clear_sky_index[index]
    hour = table["valid_time"].dt.hour() + table["valid_time"].dt.minute() / 60.0
    return table.with_columns(
        sunny=pl.Series(sunny) >= SUNNY_INDEX,
        midday=(hour > 10.0) & (hour <= 15.0),
    )


def slice_masks(*, table: pl.DataFrame) -> dict[str, pl.Expr]:
    """Return the descriptive slices as filter expressions."""
    d12 = int(prep397.primaries().filter(pl.col("site") == score.D12)["primary"][0])
    eight = pl.col("group") == "mva_eight"
    six = (pl.col("group") == "signed_six") & (pl.col("cls") == "clean")
    lead = pl.col("lead_hours")
    return {
        "eight": eight,
        "six_clean": six,
        "D12 alone": eight & (pl.col("primary") == d12),
        "eight without D12": eight & (pl.col("primary") != d12),
        "eight, cold winter nights": eight & pl.col("cold"),
        "six, cold winter nights": six & pl.col("cold"),
        "eight, 12 to 36 hours": eight & pl.col("band"),
        "six, 12 to 36 hours": six & pl.col("band"),
        "eight, sunny midday": eight & pl.col("sunny") & pl.col("midday"),
        "six, sunny midday": six & pl.col("sunny") & pl.col("midday"),
        "eight, day 1": eight & (lead < 24),
        "eight, days 2 to 5": eight & (lead >= 24) & (lead < 120),
        "eight, days 6 to 10": eight & (lead >= 120),
        "six, day 1": six & (lead < 24),
        "six, days 2 to 5": six & (lead >= 24) & (lead < 120),
        "six, days 6 to 10": six & (lead >= 120),
    }


def contrasts(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return each arm minus K3 on every descriptive slice, with 95% block-bootstrap intervals."""
    score470.RESAMPLES = RESAMPLES
    rng = np.random.default_rng(504)
    records = []
    for name, expr in slice_masks(table=table).items():
        part = table.filter(expr)
        for arm in ARMS:
            diff = part.select("month", "day", d=pl.col(f"{arm}__crps") - pl.col("K3__crps"))
            result = score470.block_bootstrap(diff=diff, rng=rng)
            records.append(
                {
                    "slice": name, "arm": arm, "rows": part.height,
                    "k3_crps": float(part["K3__crps"].mean()),
                    "difference": result["difference"], "low": result["low"], "high": result["high"],
                }
            )  # fmt: skip
    return pl.DataFrame(records)


def per_primary(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return each arm minus K3 at each Primary (CRPS, no interval)."""
    names = prep397.primaries().select("primary", "site")
    return (
        table.filter(pl.col("group") != "report_only")
        .group_by("primary")
        .agg(
            pl.col("K3__crps").mean().alias("k3_crps"),
            *[(pl.col(f"{a}__crps") - pl.col("K3__crps")).mean().alias(a) for a in ARMS],
        )
        .join(names, on="primary")
        .sort("site")
    )


def median_checks() -> pl.DataFrame:
    """Return the median forecast's absolute error, and the mean of the quantiles minus the median."""
    records = []
    for month in score.FIVE:
        base = score454.month_base(month=month).select(
            "primary", "origin", "target", "actual", "group"
        )
        for arm in ("K3", "A-cams", "A-cams-real"):
            frame = pl.read_parquet(build470.prediction_path(arm=arm, month=month, cell=0))
            joined = base.join(frame, on=["primary", "origin", "target"], how="inner")
            mean_q = joined.select(list(build470.LEVELS)).to_numpy().mean(axis=1)
            records.append(
                joined.with_columns(
                    arm=pl.lit(arm),
                    mae_median=100.0 * (pl.col("actual") - pl.col("q50")).abs(),
                    quantile_mean_minus_median=pl.Series(
                        100.0 * (mean_q - joined["q50"].to_numpy())
                    ),
                ).select("primary", "group", "arm", "mae_median", "quantile_mean_minus_median")
            )
    both = pl.concat(records)
    names = prep397.primaries().select("primary", "site")
    return (
        both.join(names, on="primary")
        .group_by("arm", "group", "site")
        .agg(pl.col("mae_median").mean(), pl.col("quantile_mean_minus_median").mean())
        .sort("arm", "group", "site")
    )


def val_nll() -> pl.DataFrame:
    """Return the validation loss of each fold's fine-tuned networks against the stored arm A."""
    records = []
    for month in score.FIVE:
        stored = json.loads((score.heads470.E425 / "logs" / f"{month}.json").read_text())
        new = json.loads((score.WORK_DIR / "logs" / f"{month}.json").read_text())
        for seed in score.SEEDS:
            records.append(  # noqa: PERF401
                {
                    "month": month, "seed": seed,
                    "stored_A": stored[f"A_s{seed}"]["val_nll"],
                    "A-cams": new[f"A-cams_s{seed}"]["val_nll"],
                    "A-cams-lag-only": new[f"A-cams-lag-only_s{seed}"]["val_nll"],
                    "A-cams-real": new[f"A-cams-real_s{seed}"]["val_nll"],
                }
            )  # fmt: skip
    return pl.DataFrame(records)


def autocorrelation() -> pl.DataFrame:
    """Return the correlation between the daily cloudiness of two days `lag` days apart."""
    records = []
    for row in prep397.primaries().iter_rows(named=True):
        cams = score.cams_for(row=row)
        daily = cams.cloudiness.astype(np.float64)
        start = (np.datetime64("2023-07-01") - np.datetime64("1970-01-01")).astype(
            int
        ) - cams.first_day
        series = daily[start:]
        for lag in range(1, 8):
            records.append(  # noqa: PERF401
                {
                    "site": row["site"], "lag_days": lag,
                    "correlation": float(np.corrcoef(series[:-lag], series[lag:])[0, 1]),
                }
            )  # fmt: skip
    return pl.DataFrame(records)


def main() -> None:
    """Write and print every table."""
    score.configure(comparison="main")
    score.link_stored()
    OUT.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    table = rows_with_context()
    tables = {
        "slices": contrasts(table=table),
        "per_primary": per_primary(table=table),
        "median_checks": median_checks(),
        "val_nll": val_nll(),
        "autocorrelation": autocorrelation(),
    }
    for name, frame in tables.items():
        frame.write_csv(OUT / f"{name}.csv")
        print(f"\n== {name}")
        print(frame.with_columns(pl.col(pl.Float64).round(4)))
    print("\n== autocorrelation, mean over the 16 Primaries")
    print(
        tables["autocorrelation"]
        .group_by("lag_days")
        .agg(pl.col("correlation").mean())
        .sort("lag_days")
    )


if __name__ == "__main__":
    main()
