# ruff: noqa: E501
"""Build the daily efficiency ratio of each solar farm and the nine trailing estimates of it.

Source (a) is an XGBoost regressor of power over capacity on observed weather, cross-fitted by
leaving out one calendar quarter at a time. Source (b) is the farm's clear-sky power, scaled once on
its first 365 days, on near-clear half-hours.

    PYTHONPATH=ml-experiments/00464-* uv run --no-sync python ml-experiments/00464-*/e00464_ratio.py
"""

import math

import e00464_common as common
import numpy as np
import polars as pl
import xgboost as xgb

FEATURES: list[str] = [
    "ghi_w_m2",
    "dni_w_m2",
    "dhi_w_m2",
    "clear_sky",
    "csi",
    "temperature_c",
    "elevation_deg",
    "sun_azimuth_deg",
]


def load_rows() -> pl.DataFrame:
    """Return every farm half-hour with a validity flag, the clear-sky index and a quarter label."""
    frame = (
        pl.read_parquet(common.FARM_ROWS)
        .rename({"site": "farm", "output": "power"})
        .sort("farm", "time")
        .with_columns(
            csi=pl.col("ghi_w_m2") / pl.col("clear_sky").clip(lower_bound=1.0),
            date=pl.col("time").dt.date(),
            quarter=pl.col("time").dt.year() * 4 + (pl.col("time").dt.month() - 1) // 3,
        )
    )
    return frame.pipe(_flag_validity)


def _flag_validity(frame: pl.DataFrame) -> pl.DataFrame:
    """Add `stuck_day` (a day containing a stuck run) and `valid` (passes every rule)."""
    out = []
    for _, part in frame.group_by("farm", maintain_order=True):
        p = part["power"].to_numpy().astype(np.float64)
        t = part["time"].dt.epoch("s").to_numpy()
        stuck = np.zeros(len(p), dtype=bool)
        run_start = 0
        for i in range(1, len(p) + 1):
            same = (
                i < len(p)
                and np.isfinite(p[i])
                and np.isfinite(p[i - 1])
                and abs(p[i] - p[i - 1]) <= common.STUCK_TOLERANCE
                and t[i] - t[i - 1] == 1800
            )
            if same:
                continue
            if i - run_start >= common.STUCK_RUN and p[run_start] > common.STUCK_MIN_POWER:
                stuck[run_start:i] = True
            run_start = i
        out.append(part.with_columns(stuck=pl.Series(stuck)))
    frame = pl.concat(out)
    # A day with a stuck run at its daily maximum is a curtailment proxy: drop the whole day.
    day_max = (
        frame.filter(pl.col("power").is_not_null() & (pl.col("power") <= common.SPIKE_POWER))
        .group_by("farm", "date")
        .agg(pl.col("power").max().alias("day_max"))
    )
    flat_days = (
        frame.join(day_max, on=["farm", "date"])
        .filter(pl.col("stuck") & ((pl.col("day_max") - pl.col("power")).abs() <= 0.005))
        .select("farm", "date")
        .unique()
        .with_columns(curtailed_day=pl.lit(value=True))
    )
    return (
        frame.join(flat_days, on=["farm", "date"], how="left")
        .with_columns(pl.col("curtailed_day").fill_null(value=False))
        .with_columns(
            valid=(
                pl.col("power").is_not_null()
                & pl.col("power").is_finite()
                & ~pl.col("stuck")
                & ~pl.col("curtailed_day")
                & ~((pl.col("power") == 0) & (pl.col("clear_sky") > common.ZERO_DAYLIGHT_CLEAR_SKY))
                & (pl.col("power") <= common.SPIKE_POWER)
                & pl.col("ghi_w_m2").is_not_null()
            )
        )
    )


def _fit_predict(
    *, x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray
) -> np.ndarray:
    """Return predictions for the test rows from a small regressor fitted on the train rows."""
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, n_jobs=1, tree_method="hist"
    )
    model.fit(x[train], y[train])
    return model.predict(x[test])


def expected_observed_weather(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return rows with `expected_a` (leave-one-quarter-out) and `expected_c` (forward-only).

    For `expected_c`, a year-quarter is predicted by a regressor trained on earlier year-quarters;
    the farm's first four year-quarters use leave-one-quarter-out within that first year.
    """
    parts = []
    for farm in common.FARMS:
        part = rows.filter(pl.col("farm") == farm)
        x = part.select(FEATURES).fill_null(0.0).to_numpy()
        valid = part["valid"].to_numpy()
        y = part["power"].to_numpy().astype(np.float64)
        quarter = part["quarter"].to_numpy()
        quarters = np.unique(quarter)
        pred_a = np.full(len(part), np.nan)
        pred_c = np.full(len(part), np.nan)
        for rank, q in enumerate(quarters):
            test = quarter == q
            pred_a[test] = _fit_predict(x=x, y=y, train=valid & ~test, test=test)
            if rank < 4:
                first_year = np.isin(quarter, quarters[:4])
                train = valid & first_year & ~test
            else:
                train = valid & (quarter < q)
            pred_c[test] = _fit_predict(x=x, y=y, train=train, test=test)
        parts.append(
            part.with_columns(
                expected_a=pl.Series(pred_a.clip(0.0, None)),
                expected_c=pl.Series(pred_c.clip(0.0, None)),
            )
        )
    return pl.concat(parts)


def expected_b(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return rows with `expected_b` (clear-sky power) and `near_clear` (the half-hours it applies to)."""
    parts = []
    near = (pl.col("csi") >= common.NEAR_CLEAR_INDEX) & (
        pl.col("clear_sky") > common.NEAR_CLEAR_MIN_CLEAR_SKY
    )
    for farm in common.FARMS:
        part = rows.filter(pl.col("farm") == farm).with_columns(near_clear=near)
        first = part.filter(pl.col("valid") & pl.col("near_clear")).filter(
            pl.col("time") < pl.col("time").min() + pl.duration(days=365)
        )
        k = float((first["power"] / (first["clear_sky"] / 1000.0)).median())  # ty: ignore[invalid-argument-type]
        parts.append(part.with_columns(expected_b=pl.col("clear_sky") / 1000.0 * k))
    return pl.concat(parts)


def daily_ratio(*, rows: pl.DataFrame, source: str) -> pl.DataFrame:
    """Return per farm and date the ratio of summed power to summed expected power and its weight."""
    exp = pl.col(f"expected_{source}")
    keep = pl.col("valid") & (exp > common.MIN_EXPECTED)
    if source == "b":
        keep &= pl.col("near_clear")
    daily = (
        rows.filter(keep)
        .group_by("farm", "date")
        .agg(
            power=pl.col("power").sum(),
            expected=exp.sum(),
            weight=pl.len(),
        )
        .filter(pl.col("weight") >= common.MIN_HALF_HOURS_PER_DAY)
        .with_columns(ratio=pl.col("power") / pl.col("expected"))
        .filter(pl.col("ratio") > 0)
        .sort("farm", "date")
    )
    return daily.select("farm", "date", "ratio", "weight").with_columns(source=pl.lit(source))


def _dense(*, daily: pl.DataFrame, farm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dates, ratio, weight) on a gap-free daily calendar; missing days have weight 0."""
    part = daily.filter(pl.col("farm") == farm)
    dates = pl.date_range(part["date"].min(), part["date"].max(), "1d", eager=True)  # ty: ignore[no-matching-overload]
    full = pl.DataFrame({"date": dates}).join(part, on="date", how="left")
    ratio = full["ratio"].fill_null(1.0).to_numpy().astype(np.float64)
    weight = full["weight"].fill_null(0).to_numpy().astype(np.float64)
    return dates.to_numpy(), ratio, weight


def weighted_median(*, values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted median of the values (NaN if the weights sum to zero)."""
    if weights.sum() <= 0:
        return math.nan
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, 0.5 * cum[-1])])


def trailing_median(*, ratio: np.ndarray, weight: np.ndarray, window: int) -> np.ndarray:
    """Return, per day, the weighted median over the last `window` days (NaN if under 40% valid)."""
    out = np.full(len(ratio), math.nan)
    for i in range(len(ratio)):
        lo = max(0, i - window + 1)
        w = weight[lo : i + 1]
        if (w > 0).sum() < common.MIN_VALID_SHARE * window:
            continue
        out[i] = weighted_median(values=ratio[lo : i + 1], weights=w)
    return out


def exponential_level(*, ratio: np.ndarray, weight: np.ndarray, half_life: int) -> np.ndarray:
    """Return the exponentially weighted mean of the log ratio, exponentiated (NaN before any data)."""
    decay = 0.5 ** (1.0 / half_life)
    num = den = 0.0
    out = np.full(len(ratio), math.nan)
    log_ratio = np.log(ratio)
    for i in range(len(ratio)):
        num = decay * num + weight[i] * log_ratio[i]
        den = decay * den + weight[i]
        if den > 1e-9:
            out[i] = math.exp(num / den)
    return out


def kalman_level(
    *, ratio: np.ndarray, weight: np.ndarray, sigma: float, reference_weight: float
) -> np.ndarray:
    """Return a local-linear-trend filter's level of the log ratio, exponentiated.

    The observation variance of a day is `sigma**2 * reference_weight / weight`.
    """
    obs = np.log(ratio)
    f = np.array([[1.0, 1.0], [0.0, 1.0]])
    q = np.diag([common.KALMAN_LEVEL_SHARE, common.KALMAN_TREND_SHARE]) * sigma**2
    state = np.array([0.0, 0.0])
    cov = np.diag([1e4, 1e-2])
    started = False
    out = np.full(len(ratio), math.nan)
    for i in range(len(ratio)):
        state = f @ state
        cov = f @ cov @ f.T + q
        if weight[i] > 0:
            if not started:
                state = np.array([obs[i], 0.0])
                started = True
            r = sigma**2 * reference_weight / weight[i]
            s = cov[0, 0] + r
            gain = cov[:, 0] / s
            state = state + gain * (obs[i] - state[0])
            cov = cov - np.outer(gain, cov[0, :])
        if started:
            out[i] = math.exp(state[0])
    return out


def robust_sigma(*, ratio: np.ndarray, weight: np.ndarray) -> float:
    """Return 1.4826 x the median absolute deviation of the log ratio around its 14-day median."""
    med = trailing_median(ratio=ratio, weight=weight, window=14)
    ok = (weight > 0) & np.isfinite(med)
    dev = np.abs(np.log(ratio[ok]) - np.log(med[ok]))
    return 1.4826 * float(np.median(dev)) if ok.any() else 0.1


def estimates(*, daily: pl.DataFrame, farm: str, refit_index: int | None = None) -> pl.DataFrame:
    """Return per date the nine estimates for one farm (the Kalman filter is per refit)."""
    dates, ratio, weight = _dense(daily=daily, farm=farm)
    cols: dict[str, np.ndarray] = {
        f"med{w}": trailing_median(ratio=ratio, weight=weight, window=w) for w in common.WINDOWS
    }
    for h in common.HALF_LIVES:
        cols[f"ew{h}"] = exponential_level(ratio=ratio, weight=weight, half_life=h)
    refit = common.REFITS[refit_index or 0]
    before = dates < np.datetime64(refit.date())
    sigma = robust_sigma(ratio=ratio[before], weight=weight[before])
    ref_w = float(np.median(weight[before & (weight > 0)]))
    cols["kalman"] = kalman_level(
        ratio=ratio, weight=weight, sigma=max(sigma, 1e-3), reference_weight=ref_w
    )
    ref = weighted_median(values=ratio[before], weights=weight[before])
    return pl.DataFrame({"date": dates, **cols}).with_columns(
        farm=pl.lit(farm), reference=pl.lit(ref), refit=pl.lit(refit)
    )


def main() -> None:
    """Write the daily ratio and the estimates for both sources."""
    common.RATIO_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    print(
        rows.group_by("farm")
        .agg(
            n=pl.len(),
            invalid=(~pl.col("valid")).sum(),
            stuck=pl.col("stuck").sum(),
            curtailed=pl.col("curtailed_day").sum(),
        )
        .sort("farm")
    )
    rows = expected_b(rows=expected_observed_weather(rows=rows))
    rows.select(
        "farm",
        "time",
        "date",
        "power",
        "valid",
        "near_clear",
        "expected_a",
        "expected_b",
        "expected_c",
    ).write_parquet(common.RATIO_DIR / "rows.parquet")
    for source in common.SOURCES:
        daily = daily_ratio(rows=rows, source=source)
        daily.write_parquet(common.RATIO_DIR / f"daily_{source}.parquet")
        est = pl.concat(
            [
                estimates(daily=daily, farm=farm, refit_index=i)
                for farm in common.FARMS
                for i in range(len(common.REFITS))
            ]
        )
        est.write_parquet(common.RATIO_DIR / f"estimates_{source}.parquet")
        print(
            source,
            daily.group_by("farm").agg(n=pl.len(), med=pl.col("ratio").median()).sort("farm"),
        )


if __name__ == "__main__":
    main()
