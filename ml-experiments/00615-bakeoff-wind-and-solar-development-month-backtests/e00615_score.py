# ruff: noqa: E501
"""Score the assembled candidates by lead band, month, season, hour, and day type.

All scores are in points of capacity (the quantile score is the mean pinball loss over the 11
levels times 100; CRPS is twice that mean, as experiment 609 defines it). Two row versions are
scored: `all` (every row in the table) and `nonzero` (rows whose observed power is not exactly
zero, which is the filter experiment 605 and 609 call version A-dev). Contrasts use a paired
bootstrap over calendar months of forecast origin, so each resample draws 5 months with replacement.

    source the environment helper; uv run --no-sync python ml-experiments/00615-*/e00615_score.py
"""

import json
import os
from typing import Final

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00615_common as common
import numpy as np
import polars as pl
from scipy import stats

LEVELS: Final[np.ndarray] = np.array(common.LEVELS)
Q: Final[list[str]] = list(common.QUANTILES)
POINTS: Final[float] = 100.0
TW_THRESHOLD: Final[float] = 0.8
"""Threshold-weighted CRPS weights outcomes above 80% of capacity (this experiment's choice)."""
RESAMPLES: Final[int] = 4000
LEAD_BANDS: Final[dict[str, tuple[float, float]]] = {
    "0-24 h": (0.0, 24.0),
    "24-120 h": (24.0, 120.0),
    "120-240 h": (120.0, 240.0),
}
SEASON: Final[dict[str, str]] = {
    "2025-10": "autumn",
    "2025-12": "winter",
    "2026-01": "winter",
    "2026-03": "spring",
    "2026-06": "summer",
}
SCORES_DIR: Final = common.DATA_DIR / "scores"


def pinball(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the pinball loss per row and level (rows, 11), in fractions of capacity."""
    error = actual[:, None] - quantiles
    return np.maximum(LEVELS * error, (LEVELS - 1.0) * error)


def tw_pinball(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the threshold-weighted pinball loss, with both sides clipped below at the threshold."""
    return pinball(
        quantiles=np.maximum(quantiles, TW_THRESHOLD), actual=np.maximum(actual, TW_THRESHOLD)
    )


def add_row_metrics(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the per-row metrics (quantile score and tail/coverage indicators) to a candidate table."""
    q = frame.select(Q).to_numpy()
    y = frame["actual"].to_numpy()
    loss = pinball(quantiles=q, actual=y)
    tw = tw_pinball(quantiles=q, actual=y)
    return frame.with_columns(
        qs=pl.Series(loss.mean(axis=1) * POINTS),
        crps=pl.Series(2.0 * loss.mean(axis=1) * POINTS),
        twcrps=pl.Series(2.0 * tw.mean(axis=1) * POINTS),
        pin01=pl.Series(loss[:, 0] * POINTS),
        pin99=pl.Series(loss[:, -1] * POINTS),
        covered=pl.Series(((y >= q[:, 1]) & (y <= q[:, 9])).astype(np.float64)),
        above99=pl.Series((y > q[:, -1]).astype(np.float64)),
        below01=pl.Series((y < q[:, 0]).astype(np.float64)),
        month=pl.col("init_time").dt.strftime("%Y-%m"),
        hour=pl.col("valid_time").dt.hour(),
        day=pl.col("valid_time").dt.date(),
    ).with_columns(
        season=pl.col("month").replace_strict(SEASON),
        lead_band=pl.when(pl.col("lead_hours") < 24.0)
        .then(pl.lit("0-24 h"))
        .when(pl.col("lead_hours") < 120.0)
        .then(pl.lit("24-120 h"))
        .otherwise(pl.lit("120-240 h")),
    )


def load(*, tech: str) -> pl.DataFrame:
    """Load every candidate of a technology with row metrics and the day-type flags."""
    frames = [
        add_row_metrics(frame=pl.read_parquet(p))
        for p in sorted(common.TABLE_DIR.glob(f"{tech}_*.parquet"))
    ]
    frame = pl.concat(frames)
    return add_day_flags(frame=frame, tech=tech)


def add_day_flags(*, frame: pl.DataFrame, tech: str) -> pl.DataFrame:
    """Flag very sunny days (solar) and high-wind days (wind) from the observed power only."""
    if tech == "solar":
        # Very sunny day: observed output reaches 90% of that farm's own 95th percentile for the
        # calendar month on at least half the daylight half-hours (a clear-sky proxy built from
        # the farm's observed power, because no clear-sky series is stored with the forecasts).
        obs = frame.filter(pl.col("candidate") == "solar_a2_member_quarterly").unique(
            ["series", "valid_time"]
        )
        ref = obs.group_by("series", pl.col("valid_time").dt.month().alias("m")).agg(
            pl.col("actual").quantile(0.95).alias("ref")
        )
        flags = (
            obs.with_columns(m=pl.col("valid_time").dt.month())
            .join(ref, on=["series", "m"])
            .with_columns(hit=(pl.col("actual") >= 0.9 * pl.col("ref")).cast(pl.Float64))
            .group_by("series", "day")
            .agg(share=pl.col("hit").mean())
            .with_columns(special=pl.col("share") >= 0.5)
            .select("series", "day", "special")
        )
        return frame.join(flags, on=["series", "day"], how="left").with_columns(
            special=pl.col("special").fill_null(value=False)
        )
    obs = frame.filter(pl.col("candidate") == "wind_default2").unique(["series", "valid_time"])
    high = obs.group_by("series", "day").agg(
        hi=(pl.col("actual") > TW_THRESHOLD).any().cast(pl.Int32)
    )
    two_farms = (
        high.group_by("day")
        .agg(n=pl.col("hi").sum())
        .with_columns(special=pl.col("n") >= 2)
        .select("day", "special")
    )
    return frame.join(two_farms, on="day", how="left").with_columns(
        special=pl.col("special").fill_null(value=False)
    )


def table(*, frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Mean metrics per candidate and slice; the pooled mean is the mean over series."""
    per_series = frame.group_by(["candidate", "series", *by]).agg(
        pl.col("qs", "crps", "twcrps", "pin01", "pin99", "covered", "above99", "below01").mean(),
        n=pl.len(),
    )
    return (
        per_series.group_by(["candidate", *by])
        .agg(
            pl.col(
                "qs", "crps", "twcrps", "pin01", "pin99", "covered", "above99", "below01"
            ).mean(),
            n=pl.col("n").sum(),
        )
        .sort(["candidate", *by])
    )


def month_matrix(*, frame: pl.DataFrame, candidate: str, key: str = "qs") -> pl.DataFrame:
    """Return the mean-over-series score per forecast-origin month for a candidate."""
    per = (
        frame.filter(pl.col("candidate") == candidate)
        .group_by("series", "month")
        .agg(pl.col(key).mean())
    )
    return per.group_by("month").agg(pl.col(key).mean()).sort("month")


def paired_months(*, frame: pl.DataFrame, a: str, b: str, rows_key: list[str]) -> dict[str, float]:
    """Return a-minus-b mean score and a month-bootstrap interval on the rows both share."""
    left = frame.filter(pl.col("candidate") == a).select(
        *rows_key, "month", pl.col("qs").alias("qa")
    )
    right = frame.filter(pl.col("candidate") == b).select(*rows_key, pl.col("qs").alias("qb"))
    joined = left.join(right, on=rows_key).with_columns(d=pl.col("qa") - pl.col("qb"))
    per = joined.group_by("series", "month").agg(pl.col("d").mean())
    months = per.group_by("month").agg(pl.col("d").mean()).sort("month")["d"].to_numpy()
    rng = np.random.default_rng(615)
    draws = months[rng.integers(0, months.size, size=(RESAMPLES, months.size))].mean(axis=1)
    return {  # ty: ignore[invalid-return-type]
        "diff": float(months.mean()),
        "draws": draws,
        "n_months": months.size,
        "rows": joined.height,
    }


def interval(*, draws: np.ndarray, level: float) -> tuple[float, float]:
    """Return the equal-tailed interval at `level` (0.95 for 95%)."""
    low, high = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(low), float(high)


def mde(*, draws: np.ndarray, contrasts: int) -> float:
    """Return the minimum detectable effect: (z at Bonferroni alpha + z at 80% power) * bootstrap SE."""
    alpha = 0.05 / contrasts
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(0.8)) * draws.std())


def contrasts(*, frame: pl.DataFrame, default: str, others: list[str]) -> pl.DataFrame:
    """Return each other candidate minus the default, per lead band and for the whole range."""
    bands = [*LEAD_BANDS, "all leads"]
    count = len(others) * len(LEAD_BANDS)
    out = []
    for other in others:
        for band in bands:
            sub = frame if band == "all leads" else frame.filter(pl.col("lead_band") == band)
            result = paired_months(
                frame=sub, a=other, b=default, rows_key=["series", "init_time", "valid_time"]
            )
            level = 1 - 0.05 / count
            low, high = interval(draws=result["draws"], level=level)  # ty: ignore[invalid-argument-type]
            out.append(
                {
                    "candidate": other,
                    "default": default,
                    "lead_band": band,
                    "diff_points": result["diff"],
                    "low": low,
                    "high": high,
                    "interval_level": level,
                    "mde": mde(draws=result["draws"], contrasts=count),  # ty: ignore[invalid-argument-type]
                    "bonferroni_count": count,
                    "paired_rows": result["rows"],
                    "months": result["n_months"],
                }
            )
    return pl.DataFrame(out)


def run(*, tech: str, default: str, others: list[str]) -> None:
    """Write every score table of one technology."""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    frame = load(tech=tech)
    versions = {"all": frame, "nonzero": frame.filter(pl.col("actual") > 0.0)}
    for version, sub in versions.items():
        slices = {
            "overall": [],
            "lead_band": ["lead_band"],
            "month": ["month"],
            "season": ["season"],
            "hour": ["hour"],
            "series": ["series"],
            "day_type": ["special"],
            "month_lead": ["month", "lead_band"],
        }
        for name, by in slices.items():
            if name == "overall":
                result = table(frame=sub.with_columns(all=pl.lit("all")), by=["all"])
            elif name == "series":
                result = (
                    sub.group_by("candidate", "series")
                    .agg(
                        pl.col(
                            "qs",
                            "crps",
                            "twcrps",
                            "pin01",
                            "pin99",
                            "covered",
                            "above99",
                            "below01",
                        ).mean(),
                        n=pl.len(),
                    )
                    .sort("candidate", "series")
                )
            else:
                result = table(frame=sub, by=by)
            result.write_csv(SCORES_DIR / f"{tech}_{version}_{name}.csv")
        present = [c for c in others if c in sub["candidate"].unique().to_list()]
        contrasts(frame=sub, default=default, others=present).write_csv(
            SCORES_DIR / f"{tech}_{version}_contrasts.csv"
        )
    counts = {
        "rows_all": frame.group_by("candidate").len().sort("candidate").to_dicts(),
        "rows_nonzero": frame.filter(pl.col("actual") > 0.0)
        .group_by("candidate")
        .len()
        .sort("candidate")
        .to_dicts(),
        "special_days": frame.filter(pl.col("special")).select(pl.col("day").n_unique()).item(),
    }
    (SCORES_DIR / f"{tech}_counts.json").write_text(json.dumps(counts, default=str, indent=1))
    print(tech, "scored", flush=True)


def main() -> None:
    """Score solar, and wind when its tables exist."""
    run(
        tech="solar",
        default="solar_a2_member_quarterly",
        others=[
            "solar_a2_member_annual",
            "solar_gam",
            "solar_xgboost",
            "solar_climatology",
            "solar_manual_heuristic",
        ],
    )
    if (common.TABLE_DIR / "wind_default2.parquet").exists():
        run(
            tech="wind",
            default="wind_default2",
            others=[
                "wind_climatology",
                "wind_default1_stored_tuesdays",
                "wind_default2_stored_tuesdays",
            ],
        )


if __name__ == "__main__":
    main()
