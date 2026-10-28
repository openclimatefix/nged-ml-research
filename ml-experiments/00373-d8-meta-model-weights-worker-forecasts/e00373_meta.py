# ruff: noqa: E501
"""Fit and score the meta-model over the stored worker forecasts at D8 (experiment 373).

Workers are BG2's five members (experiment 360's `only_*` columns), the GAM stack (experiment 355,
arm W), and BG2. The meta-model is trained on an expanding window of earlier valid months and scores
October 2025 to June 2026. A reading counts as known six hours after its valid time, the as-of rule
of experiments 264 and 354.

Run it with the folders of experiments 252, 264, and 354 on `PYTHONPATH`, under
`systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue` where it fits models.
"""

import json
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

from datetime import date, timedelta
from typing import Final

import e00354_lib as lib  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final = "00373-d8-meta-model-weights-worker-forecasts"
WORK: Final = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
D355: Final = ML_EXPERIMENTS_DATA_DIR / "00355-d8-weekly-refit-and-median-band-swap" / "work"
D360: Final = ML_EXPERIMENTS_DATA_DIR / "00360-bg2-member-drop-one-at-d8" / "work"
QUANTILES: Final = lib.QUANTILES
LEVELS: Final = [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99]
WORKERS: Final[dict[str, str]] = {
    "W": "only_W__",
    "M": "only_M__",
    "AN": "only_AN__",
    "PT": "only_PT__",
    "ST": "only_ST__",
    "GAM": "gam__",
    "BG2": "bg2_",
}
SCORED_MONTHS: Final = (
    "2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
)  # fmt: skip
KNOWN_AFTER_HOURS: Final = 6
RECENT_DAYS: Final = 7
BANK_HOLIDAYS: Final = (
    date(2025, 8, 25), date(2025, 12, 25), date(2025, 12, 26), date(2026, 1, 1),
    date(2026, 4, 3), date(2026, 4, 6), date(2026, 5, 4), date(2026, 5, 25),
)  # fmt: skip
"""England and Wales bank holidays inside the scored year."""
PERCENT: Final = 100.0
"""Experiment 360's frame stores quantiles as fractions of capacity; experiment 355's stores points."""
GATE_MEMBERS: Final[dict[str, float]] = {
    "W": 7.560,
    "M": 7.694,
    "AN": 7.472,
    "PT": 7.633,
    "ST": 8.562,
}
GATE_GAM: Final = 8.338
GATE_BG2: Final = 7.593
XGB_PARAMS: Final = {
    "objective": "reg:quantileerror",
    "quantile_alpha": np.array(LEVELS),
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "nthread": 12,
    "seed": 0,
}
ROUNDS: Final = 200
ARMS: Final = ("meta", "meta_nolag", "meta_nocal")


def load() -> pl.DataFrame:
    """Return the joined worker frame: actual, 11 quantiles per worker, flags, on the 96,986 rows."""
    members = pl.read_parquet(
        D360 / "frame_D8.parquet",
        columns=[
            "power_fcst_init_time",
            "valid_time",
            *[f"only_{k}__{q}" for k in ("W", "M", "AN", "PT", "ST") for q in QUANTILES],
        ],
    )
    members = members.with_columns(pl.exclude("power_fcst_init_time", "valid_time") * PERCENT)
    wide = pl.read_parquet(D355 / "scores_wide.parquet")
    keep = [
        "power_fcst_init_time", "valid_time", "lead_hours", "actual", "month",
        *[f"W__{q}" for q in QUANTILES], *[f"bg2_{q}" for q in QUANTILES],
        *[f"climatology_{q}" for q in QUANTILES], *[f"manual_{q}" for q in QUANTILES],
    ]  # fmt: skip
    wide = wide.select(keep).rename({f"W__{q}": f"gam__{q}" for q in QUANTILES})
    frame = wide.join(members, on=["power_fcst_init_time", "valid_time"], how="inner")
    if frame.height != 96_986:
        raise SystemExit(f"row count {frame.height} is not 96,986")
    return frame.sort(["power_fcst_init_time", "valid_time"])


def row_score(*, frame: pl.DataFrame, prefix: str) -> np.ndarray:
    """Return per-row quantile scores of one prefixed set of quantile columns."""
    return lib.row_scores(
        actual=frame["actual"].to_numpy().astype(np.float64), frame=frame, prefix=prefix
    )


def origin_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Return one row per run: lagged power at the origin and each worker's seven-day median error."""
    shortest = frame.sort(["valid_time", "lead_hours"]).unique(
        "valid_time", keep="first", maintain_order=True
    )
    valid = shortest["valid_time"].to_numpy().astype("datetime64[us]")
    actual = shortest["actual"].to_numpy().astype(np.float64)
    errors = {
        k: np.abs(actual - shortest[f"{p}q50"].to_numpy().astype(np.float64))
        for k, p in WORKERS.items()
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
    frame = frame.join(origin_features(frame), on="power_fcst_init_time", how="left")
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    bank = [d.isoformat() for d in BANK_HOLIDAYS]
    frame = frame.with_columns(
        hour=local.dt.hour() + local.dt.minute() / 60.0,
        month_num=local.dt.month(),
        weekday=local.dt.weekday(),
        bank_holiday=local.dt.date().cast(pl.String).is_in(bank),
        christmas_period=((local.dt.month() == 12) & (local.dt.day() >= 24))
        | ((local.dt.month() == 1) & (local.dt.day() <= 2)),
    ).with_columns(
        event_day=pl.col("bank_holiday") | pl.col("christmas_period"),
        spread_medians=pl.concat_list([pl.col(f"{p}q50") for p in WORKERS.values()]).list.std(),
    )
    worker_cols = [f"{p}{q}" for p in WORKERS.values() for q in ("q05", "q50", "q95")]
    calendar = ["hour", "month_num", "weekday", "bank_holiday", "christmas_period"]
    recent_err = [f"err_{k}" for k in WORKERS]
    base = [*worker_cols, "spread_medians", "lead_hours", *recent_err]
    groups = {
        "meta": [*base, "lag_origin", *calendar],
        "meta_nolag": [*base, *calendar],
        "meta_nocal": [*base, "lag_origin"],
    }
    return frame, groups


def month_start_cutoff(frame: pl.DataFrame, month: str) -> np.datetime64:
    """Return the first issue time of runs whose first valid time falls in `month`, less the known delay."""
    issue = frame.filter(pl.col("month") == month)["power_fcst_init_time"]
    first = issue.dt.replace_time_zone(None).to_numpy().min().astype("datetime64[us]")
    return first - np.timedelta64(KNOWN_AFTER_HOURS, "h")


def fit_predict(*, frame: pl.DataFrame, columns: list[str], month: str) -> pl.DataFrame:
    """Train on rows read before `month`'s first run, and return sorted quantiles for the month's rows."""
    cutoff = month_start_cutoff(frame, month)
    valid = frame["valid_time"].to_numpy().astype("datetime64[us]")
    train = frame.filter(pl.Series(valid <= cutoff))
    test = frame.filter(pl.col("month") == month)
    base_train = 0.5 * (train["gam__q50"] + train["bg2_q50"])
    base_test = 0.5 * (test["gam__q50"] + test["bg2_q50"])
    target = (train["actual"] - base_train).to_numpy().astype(np.float64)
    dtrain = xgb.DMatrix(train.select(columns).cast(pl.Float64).to_numpy(), label=target)
    booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=ROUNDS)
    pred = booster.predict(xgb.DMatrix(test.select(columns).cast(pl.Float64).to_numpy()))
    pred = np.sort(pred, axis=1) + base_test.to_numpy().astype(np.float64)[:, None]
    return test.select(["power_fcst_init_time", "valid_time"]).with_columns(
        *[pl.Series(f"__{q}", pred[:, i]) for i, q in enumerate(QUANTILES)]
    )


def best_single(*, frame: pl.DataFrame, scores: dict[str, np.ndarray], month: str) -> str:
    """Return the family (BG2, itself a blend, excluded) with the lowest training score before `month`."""
    cutoff = month_start_cutoff(frame, month)
    mask = frame["valid_time"].to_numpy().astype("datetime64[us]") <= cutoff
    return min((k for k in WORKERS if k != "BG2"), key=lambda k: float(scores[k][mask].mean()))


def main() -> None:
    """Reproduce the gate, fit the meta arms month by month, and write `work/meta_wide_D8.parquet`."""
    WORK.mkdir(parents=True, exist_ok=True)
    frame = load()
    scores = {k: row_score(frame=frame, prefix=p) for k, p in WORKERS.items()}
    gate = {"gam": float(scores["GAM"].mean()), "bg2": float(scores["BG2"].mean())}
    gate |= {k: float(scores[k].mean()) for k in GATE_MEMBERS}
    print("gate", gate, "expected", GATE_GAM, GATE_BG2, GATE_MEMBERS)
    if any(abs(gate[k] - v) > 0.0015 for k, v in GATE_MEMBERS.items()):
        raise SystemExit("member reproduction gate failed")
    if abs(gate["gam"] - GATE_GAM) > 0.0015 or abs(gate["bg2"] - GATE_BG2) > 0.0015:
        raise SystemExit("reproduction gate failed")
    frame, groups = features(frame)
    parts = []
    chosen = {}
    for month in SCORED_MONTHS:
        out = frame.filter(pl.col("month") == month).select(["power_fcst_init_time", "valid_time"])
        for arm in ARMS:
            pred = fit_predict(frame=frame, columns=groups[arm], month=month)
            out = out.join(
                pred.rename({f"__{q}": f"{arm}__{q}" for q in QUANTILES}),
                on=["power_fcst_init_time", "valid_time"],
                how="left",
            )
        chosen[month] = best_single(frame=frame, scores=scores, month=month)
        parts.append(out)
        print(month, "best single", chosen[month], flush=True)
    meta = pl.concat(parts)
    wide = frame.filter(pl.col("month").is_in(list(SCORED_MONTHS))).join(
        meta, on=["power_fcst_init_time", "valid_time"], how="left"
    )
    single = {
        q: np.select(
            [wide["month"].to_numpy() == m for m in SCORED_MONTHS],
            [wide[f"{WORKERS[chosen[m]]}{q}"].to_numpy() for m in SCORED_MONTHS],
        )
        for q in QUANTILES
    }
    b50 = {
        q: 0.5 * wide[f"gam__{q}"].to_numpy() + 0.5 * wide[f"bg2_{q}"].to_numpy() for q in QUANTILES
    }
    wide = wide.with_columns(
        *[pl.Series(f"single__{q}", single[q]) for q in QUANTILES],
        *[pl.Series(f"b50__{q}", b50[q]) for q in QUANTILES],
    )
    wide.write_parquet(WORK / "meta_wide_D8.parquet")
    (WORK / "meta_gate_and_choices.json").write_text(
        json.dumps({"gate": gate, "best_single_by_month": chosen}, indent=2)
    )
    print("wrote", wide.height, "rows;", timedelta(0))


if __name__ == "__main__":
    main()
