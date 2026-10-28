# ruff: noqa: E501
"""Fit the experiment 373 meta-model at every Primary and write a scored frame for each (experiment 375).

Copies experiment 373's recipe (features, expanding monthly training window, target the actual
minus the base median, XGBoost quantile parameters) with the worker list set per Primary: where no
GAM stack was stored (D15 and D18) the stack columns are absent and the base is BG2. No tuning.

    PYTHONPATH=ml-experiments/00373-d8-meta-model-weights-worker-forecasts:ml-experiments/00354-d8-own-residual-bands-and-longer-fit-history:ml-experiments/00264-dp-gas-engine-asymmetric-bands-at-d8:ml-experiments/00252-gam-stack-at-eight-more-primaries \
        systemd-run --user --scope -q -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00375-*/e00375_meta.py [--sites D8 D2 ...]
"""

import multiprocessing as mp
import os
import sys
from typing import Final

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

import e00373_meta as m373  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final = "00375-meta-model-over-worker-forecasts-at-all-primaries"
WORK: Final = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
SITES: Final = (
    "D2", "D3", "D4", "D5", "D8", "D9", "D10", "D11", "D12", "D14", "D15", "D16", "D18", "D19", "D21", "D22",
)  # fmt: skip
QUANTILES: Final = m373.QUANTILES
SCORED_MONTHS: Final = m373.SCORED_MONTHS
KNOWN_AFTER_HOURS: Final = m373.KNOWN_AFTER_HOURS
RECENT_DAYS: Final = m373.RECENT_DAYS
ARMS: Final = m373.ARMS
WORKERS_FULL: Final = m373.WORKERS
WORKERS_NO_STACK: Final = {k: v for k, v in m373.WORKERS.items() if k != "GAM"}
THREADS: Final = 2


def workers_of(frame: pl.DataFrame) -> dict[str, str]:
    """Return the worker prefixes available in a Primary's frame."""
    return WORKERS_FULL if "gam__q50" in frame.columns else WORKERS_NO_STACK


def base_median(frame: pl.DataFrame) -> pl.Series:
    """Return the base median: the 50/50 blend where a stack exists, BG2 alone otherwise."""
    if "gam__q50" in frame.columns:
        return 0.5 * (frame["gam__q50"] + frame["bg2_q50"])
    return frame["bg2_q50"]


def origin_features(frame: pl.DataFrame, workers: dict[str, str]) -> pl.DataFrame:
    """Return one row per run: lagged power at the origin and each worker's seven-day median error."""
    shortest = frame.sort(["valid_time", "lead_hours"]).unique(
        "valid_time", keep="first", maintain_order=True
    )
    valid = shortest["valid_time"].to_numpy().astype("datetime64[us]")
    actual = shortest["actual"].to_numpy().astype(np.float64)
    errors = {
        k: np.abs(actual - shortest[f"{p}q50"].to_numpy().astype(np.float64))
        for k, p in workers.items()
    }
    lag = np.timedelta64(KNOWN_AFTER_HOURS, "h")
    rows = []
    for init in frame["power_fcst_init_time"].unique().sort().to_numpy().astype("datetime64[us]"):
        known = valid <= init - lag
        recent = known & (valid > init - lag - np.timedelta64(RECENT_DAYS, "D"))
        last = np.flatnonzero(known)
        row = {
            "power_fcst_init_time": init.astype("datetime64[us]").item(),
            "lag_origin": actual[last[-1]] if last.size else np.nan,
        }
        for k, err in errors.items():
            row[f"err_{k}"] = float(np.nanmean(err[recent])) if recent.any() else np.nan
        rows.append(row)
    return pl.DataFrame(
        rows, schema_overrides={"power_fcst_init_time": pl.Datetime("us")}
    ).with_columns(pl.col("power_fcst_init_time").dt.replace_time_zone("UTC"))


def features(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Return the frame with feature columns, and the column groups of each meta arm."""
    workers = workers_of(frame)
    frame = frame.join(origin_features(frame, workers), on="power_fcst_init_time", how="left")
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    bank = [d.isoformat() for d in m373.BANK_HOLIDAYS]
    frame = frame.with_columns(
        hour=local.dt.hour() + local.dt.minute() / 60.0,
        month_num=local.dt.month(),
        weekday=local.dt.weekday(),
        bank_holiday=local.dt.date().cast(pl.String).is_in(bank),
        christmas_period=((local.dt.month() == 12) & (local.dt.day() >= 24))
        | ((local.dt.month() == 1) & (local.dt.day() <= 2)),
    ).with_columns(
        event_day=pl.col("bank_holiday") | pl.col("christmas_period"),
        spread_medians=pl.concat_list([pl.col(f"{p}q50") for p in workers.values()]).list.std(),
    )
    worker_cols = [f"{p}{q}" for p in workers.values() for q in ("q05", "q50", "q95")]
    calendar = ["hour", "month_num", "weekday", "bank_holiday", "christmas_period"]
    base = [*worker_cols, "spread_medians", "lead_hours", *[f"err_{k}" for k in workers]]
    return frame, {
        "meta": [*base, "lag_origin", *calendar],
        "meta_nolag": [*base, *calendar],
        "meta_nocal": [*base, "lag_origin"],
    }


def month_start_cutoff(frame: pl.DataFrame, month: str) -> np.datetime64:
    """Return the first issue time of the month's runs, less the delay after which a reading is known."""
    issue = frame.filter(pl.col("month") == month)["power_fcst_init_time"]
    first = issue.dt.replace_time_zone(None).to_numpy().min().astype("datetime64[us]")
    return first - np.timedelta64(KNOWN_AFTER_HOURS, "h")


def fit_predict(*, frame: pl.DataFrame, columns: list[str], month: str) -> pl.DataFrame:
    """Train on rows read before `month`'s first run and return sorted quantiles for the month."""
    cutoff = month_start_cutoff(frame, month)
    valid = frame["valid_time"].to_numpy().astype("datetime64[us]")
    train = frame.filter(pl.Series(valid <= cutoff))
    test = frame.filter(pl.col("month") == month)
    target = (train["actual"] - base_median(train)).to_numpy().astype(np.float64)
    dtrain = xgb.DMatrix(train.select(columns).cast(pl.Float64).to_numpy(), label=target)
    params = m373.XGB_PARAMS | {"nthread": THREADS}
    booster = xgb.train(params, dtrain, num_boost_round=m373.ROUNDS)
    pred = booster.predict(xgb.DMatrix(test.select(columns).cast(pl.Float64).to_numpy()))
    pred = np.sort(pred, axis=1) + base_median(test).to_numpy().astype(np.float64)[:, None]
    return test.select(["power_fcst_init_time", "valid_time"]).with_columns(
        *[pl.Series(f"__{q}", pred[:, i]) for i, q in enumerate(QUANTILES)]
    )


def row_score(*, frame: pl.DataFrame, prefix: str) -> np.ndarray:
    """Return per-row quantile scores of one prefixed set of quantile columns."""
    return m373.lib.row_scores(
        actual=frame["actual"].to_numpy().astype(np.float64), frame=frame, prefix=prefix
    )


def run_site(site: str) -> str:
    """Fit the three meta arms month by month at one Primary and write `meta_wide_<site>.parquet`."""
    frame = pl.read_parquet(WORK / f"workers_{site}.parquet").with_columns(
        month=pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.strftime("%Y-%m")
    )
    workers = workers_of(frame)
    scores = {k: row_score(frame=frame, prefix=p) for k, p in workers.items()}
    frame, groups = features(frame)
    parts, chosen = [], {}
    for month in SCORED_MONTHS:
        out = frame.filter(pl.col("month") == month).select(["power_fcst_init_time", "valid_time"])
        for arm in ARMS:
            pred = fit_predict(frame=frame, columns=groups[arm], month=month)
            out = out.join(
                pred.rename({f"__{q}": f"{arm}__{q}" for q in QUANTILES}),
                on=["power_fcst_init_time", "valid_time"],
                how="left",
            )
        cutoff = month_start_cutoff(frame, month)
        mask = frame["valid_time"].to_numpy().astype("datetime64[us]") <= cutoff
        chosen[month] = min(
            (k for k in workers if k != "BG2"), key=lambda k: float(scores[k][mask].mean())
        )
        parts.append(out)
        print(site, month, "best single", chosen[month], flush=True)
    wide = frame.filter(pl.col("month").is_in(list(SCORED_MONTHS))).join(
        pl.concat(parts), on=["power_fcst_init_time", "valid_time"], how="left"
    )
    single = {
        q: np.select(
            [wide["month"].to_numpy() == m for m in SCORED_MONTHS],
            [wide[f"{workers[chosen[m]]}{q}"].to_numpy() for m in SCORED_MONTHS],
        )
        for q in QUANTILES
    }
    blend = {
        q: 0.5 * wide[f"gam__{q}"].to_numpy() + 0.5 * wide[f"bg2_{q}"].to_numpy()
        if "GAM" in workers
        else wide[f"bg2_{q}"].to_numpy()
        for q in QUANTILES
    }
    wide = wide.with_columns(
        *[pl.Series(f"single__{q}", single[q]) for q in QUANTILES],
        *[pl.Series(f"b50__{q}", blend[q]) for q in QUANTILES],
    ).with_columns(best_single=pl.col("month").replace_strict(chosen))
    wide.write_parquet(WORK / f"meta_wide_{site}.parquet")
    return site


def main() -> None:
    """Run the Primaries in parallel (spawn start method: polars is loaded in this process)."""
    sites = tuple(sys.argv[sys.argv.index("--sites") + 1 :]) if "--sites" in sys.argv else SITES
    workers = int(os.environ.get("E00375_WORKERS", "4"))
    with mp.get_context("spawn").Pool(workers) as pool:
        for done in pool.imap_unordered(run_site, sites):
            print("finished", done, flush=True)


if __name__ == "__main__":
    main()
