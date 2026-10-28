# ruff: noqa: E501
"""Build the band head's training and scoring frame: the stack's stored forecasts at eight Primaries.

Each row is a stored forecast of the GAM stack (experiment 252's stage-2 arm d; at D8, experiment
354's fit `b` monthly) with its median, its donated band, the reading, and the features the heads
condition on. The as-of residual features use only readings that arrived (valid time plus 6 hours)
before the forecast's origin.

    PYTHONPATH=ml-experiments/00252-gam-stack-at-eight-more-primaries:ml-experiments/00264-dp-gas-engine-asymmetric-bands-at-d8:ml-experiments/00354-d8-own-residual-bands-and-longer-fit-history \
        uv run --no-sync python ml-experiments/00368-*/e00368_features.py
"""

import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

from typing import Final

import e00252_common as common252  # ty: ignore[unresolved-import]
import e00354_lib as lib  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00368-d8-conditional-band-head-on-gam-stack-residuals" / "work"
)
SITES: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
D8_STORED: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00354-d8-own-residual-bands-and-longer-fit-history"
    / "work"
    / "fits_b"
    / "d_scored_D8.parquet"
)
LATENCY_NS: Final[int] = 6 * 3600 * 10**9
NS_PER_DAY: Final[int] = 86_400 * 10**9
FEATURES: Final[tuple[str, ...]] = (
    "lead_hours",
    "median",
    "band_width",
    "hour_local",
    "day_of_week",
    "temperature_c",
    "pv_unit_shape",
    "elevation_deg",
    "abs_res_1d",
    "abs_res_7d",
    "res_1d",
    *(f"site_{s}" for s in SITES),
)


def stored_frame(*, site: str) -> pl.DataFrame:
    """Return the cleaned stored forecasts of the stack at a site."""
    path = (
        D8_STORED
        if site == "D8"
        else common252.ARM_DIR / "dp_preds_s2" / f"d_scored_{site}.parquet"
    )
    return lib.clean(frame=pl.read_parquet(path))


def trailing_means(*, frame: pl.DataFrame, window_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean absolute and mean signed residual known at each row's origin.

    A residual is `actual - q50` at the shortest lead of each valid time, known 6 hours after the
    valid time. NaN where fewer than 48 half-hours are known in the trailing window.
    """
    shortest = frame.sort("lead_hours").group_by("valid_time").first().sort("valid_time")
    valid_ns = shortest["valid_time"].dt.epoch("ns").to_numpy()
    residual = (shortest["actual"] - shortest["q50"]).to_numpy().astype(np.float64)
    cum_abs = np.concatenate([[0.0], np.cumsum(np.abs(residual))])
    cum_sig = np.concatenate([[0.0], np.cumsum(residual)])
    known = frame["power_fcst_init_time"].dt.epoch("ns").to_numpy() - LATENCY_NS
    hi = np.searchsorted(valid_ns, known, side="right")
    lo = np.searchsorted(valid_ns, known - window_days * NS_PER_DAY, side="right")
    count = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = count >= 48
        abs_mean = np.where(ok, (cum_abs[hi] - cum_abs[lo]) / np.maximum(count, 1), np.nan)
        sig_mean = np.where(ok, (cum_sig[hi] - cum_sig[lo]) / np.maximum(count, 1), np.nan)
    return abs_mean, sig_mean


def site_frame(*, site: str, cache) -> pl.DataFrame:  # noqa: ANN001
    """Return one site's rows with every feature column and the site's one-hot columns."""
    frame = stored_frame(site=site)
    features = cache.sites[site]
    index = np.searchsorted(
        cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    local = frame["valid_time"].dt.convert_time_zone("Europe/London")
    abs_1d, sig_1d = trailing_means(frame=frame, window_days=1)
    abs_7d, _ = trailing_means(frame=frame, window_days=7)
    return frame.with_columns(
        site=pl.lit(site),
        median=pl.col("q50"),
        band_width=pl.col("q95") - pl.col("q05"),
        hour_local=local.dt.hour().cast(pl.Float64) + local.dt.minute().cast(pl.Float64) / 60.0,
        day_of_week=local.dt.weekday().cast(pl.Float64),
        temperature_c=pl.Series(features.temperature_c[index].astype(np.float64)),
        pv_unit_shape=pl.Series(features.pv_unit_shape[index].astype(np.float64)),
        elevation_deg=pl.Series(features.elevation_deg[index].astype(np.float64)),
        abs_res_1d=pl.Series(abs_1d),
        abs_res_7d=pl.Series(abs_7d),
        res_1d=pl.Series(sig_1d),
        residual=pl.col("actual") - pl.col("q50"),
        valid_ns=pl.col("valid_time").dt.epoch("ns"),
        origin_ns=pl.col("power_fcst_init_time").dt.epoch("ns"),
        **{f"site_{s}": pl.lit(float(s == site)) for s in SITES},
    )


def main() -> None:
    """Write `work/features.parquet`, one row per stored forecast at the eight Primaries."""
    cache = common252.load_cache()
    WORK.mkdir(parents=True, exist_ok=True)
    keep = [
        *lib.KEY,
        "site",
        "run",
        "lead_hours",
        "actual",
        "mean",
        *lib.QUANTILES,
        "residual",
        "valid_ns",
        "origin_ns",
        *FEATURES,
    ]
    frames = [
        site_frame(site=site, cache=cache).select(list(dict.fromkeys(keep))) for site in SITES
    ]
    out = pl.concat(frames)
    out.write_parquet(WORK / "features.parquet")
    print(
        out.group_by("site")
        .agg(pl.len(), pl.col("abs_res_1d").is_null().mean().alias("null_1d"))
        .sort("site")
    )


if __name__ == "__main__":
    main()
