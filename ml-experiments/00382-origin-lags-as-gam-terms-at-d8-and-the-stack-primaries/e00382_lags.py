"""Add three origin lags to experiment 377's stack as stage 2 smooth terms (experiment 382).

The treated mean is the stack's corrected mean (arm d of experiment 155, as experiment 377 builds
it) plus a shift: the sum, over the newest usable reading and the day and week lags, of a penalised
cubic B-spline of that lag's anomaly (the reading minus the stack's own mean for that half-hour),
fitted per lead band by penalised least squares on earlier months only. The spread is then fitted
fold by fold on earlier months' residuals of the treated mean, as in experiment 377.

    PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility \
    E00377_SOURCE=mva E00377_ARM=a0 uv run --no-sync python e00382_lags.py \
        --arm spline --sites D2 D3 D4 D8 D12 D19 D21 D22
"""

import argparse
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00377s2_common as common  # ty: ignore[unresolved-import]
import e00377s3_stack as stack  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy.interpolate import BSpline
from studies.dp_features import FeatureCache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00382-origin-lags-as-gam-terms-at-d8-and-the-stack-primaries"
    / "work"
)
ARMS: Final[tuple[str, ...]] = ("spline", "linear", "zero")
"""`zero` adds no shift: the gate that the pipeline reproduces experiment 377's stored stack."""
FIRST_CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
LAST_CUT: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
N_BASIS: Final[int] = 8
DEGREE: Final[int] = 3
PENALTY_WEIGHT: Final[float] = 10.0
RIDGE: Final[float] = 1e-6
MIN_FIT_ROWS: Final[int] = 2_000
CLIP_QUANTILES: Final[tuple[float, float]] = (0.01, 0.99)
NEWEST_AGE: Final[int] = 7
"""Half-hours between the origin and the newest usable reading (3.5 hours), as experiment 87."""
STEPS_PER_DAY: Final[int] = 48
N_LAGS: Final[int] = 3


@dataclass(frozen=True)
class Term:
    """One lag's term in one lead band: clip range, knots, and coefficients."""

    low: float
    high: float
    coefficients: np.ndarray


def basis(*, x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Return the centred B-spline basis of `x` (clipped), zero at x = 0, shape (rows, N_BASIS)."""
    inner = np.linspace(low, high, N_BASIS - DEGREE + 1)
    knots = np.concatenate([[low] * DEGREE, inner, [high] * DEGREE])
    clipped = np.clip(x, low, high)
    design = BSpline.design_matrix(clipped, knots, DEGREE).toarray()
    at_zero = BSpline.design_matrix(np.clip([0.0], low, high), knots, DEGREE).toarray()
    return design - at_zero


def term_columns(*, anomaly: np.ndarray, term: Term | None, arm: str) -> np.ndarray:
    """Return a lag's design columns; a missing anomaly gives zeros."""
    present = np.isfinite(anomaly)
    value = np.where(present, anomaly, 0.0)
    if arm == "linear":
        low, high = (term.low, term.high) if term is not None else (-np.inf, np.inf)
        return np.clip(value, low, high)[:, None]
    low, high = (term.low, term.high) if term is not None else (-1.0, 1.0)
    columns = basis(x=value, low=low, high=high)
    columns[~present] = 0.0
    return columns


def fit_term_set(*, anomalies: np.ndarray, target: np.ndarray, arm: str) -> list[Term] | None:
    """Fit the three terms of one lead band jointly; `None` if the band is too small."""
    keep = np.isfinite(anomalies).any(axis=1)
    if keep.sum() < MIN_FIT_ROWS:
        return None
    anomalies, target = anomalies[keep], target[keep]
    ranges = []
    for k in range(N_LAGS):
        column = anomalies[:, k]
        column = column[np.isfinite(column)]
        low, high = (
            np.quantile(column, CLIP_QUANTILES) if len(column) else (np.float64(-1), np.float64(1))
        )
        ranges.append((float(low), float(max(high, low + 1e-6))))
    placeholder = [Term(low, high, np.zeros(1)) for low, high in ranges]
    blocks = [
        term_columns(anomaly=anomalies[:, k], term=placeholder[k], arm=arm) for k in range(N_LAGS)
    ]
    design = np.hstack(blocks)
    width = [b.shape[1] for b in blocks]
    penalty = np.eye(design.shape[1]) * RIDGE
    if arm == "spline":
        diff = np.diff(np.eye(N_BASIS), n=2, axis=0)
        block = PENALTY_WEIGHT * diff.T @ diff
        for k in range(N_LAGS):
            start = k * N_BASIS
            penalty[start : start + N_BASIS, start : start + N_BASIS] += block
    gram = design.T @ design + penalty
    solution = np.linalg.solve(gram, design.T @ target)
    terms, start = [], 0
    for k in range(N_LAGS):
        terms.append(Term(ranges[k][0], ranges[k][1], solution[start : start + width[k]]))
        start += width[k]
    return terms


def shift_of(*, anomalies: np.ndarray, terms: list[Term], arm: str) -> np.ndarray:
    """Return the summed shift of the three terms."""
    out = np.zeros(len(anomalies))
    for k in range(N_LAGS):
        out += term_columns(anomaly=anomalies[:, k], term=terms[k], arm=arm) @ terms[k].coefficients
    return out


def lag_anomalies(*, frame: pl.DataFrame, cache: FeatureCache, site: str) -> np.ndarray:
    """Return the (rows, 3) anomalies of the newest, day, and week lags of each frame row.

    Args:
        frame: The corrected stack frame of one site (mean, origin, valid time).
        cache: The feature cache, which holds the site's power on the half-hourly grid.
        site: The site's label.

    Returns:
        An array of shape (rows, 3): the reading minus the stack's short-lead mean for that
        half-hour, `NaN` where either is missing.

    Raises:
        ValueError: If a lag index is younger than the newest usable reading, or a row's valid
            time is not after its origin.
    """
    ends = cache.end_times
    y = np.asarray(cache.sites[site].y, dtype=np.float64)
    frame = frame.with_columns(
        pl.col("valid_time").dt.replace_time_zone(None).alias("_v"),
        pl.col("power_fcst_init_time").dt.replace_time_zone(None).alias("_o"),
    )
    # The stack's short-lead mean for each half-hour: the row with the latest origin before it.
    short = (
        frame.sort("_v", "_o")
        .group_by("_v", maintain_order=True)
        .agg(pl.col("mean").last(), pl.col("lead_hours").last())
    )
    m1 = np.full(len(ends), np.nan)
    position = np.searchsorted(ends, short["_v"].to_numpy())
    inside = (position < len(ends)) & (
        ends[np.minimum(position, len(ends) - 1)] == short["_v"].to_numpy()
    )
    m1[position[inside]] = short["mean"].to_numpy().astype(np.float64)[inside]
    anomaly_series = y - m1
    target = np.searchsorted(ends, frame["_v"].to_numpy())
    origin = np.searchsorted(ends, frame["_o"].to_numpy())
    if (target <= origin).any():
        msg = "every valid time must be after its origin"
        raise ValueError(msg)
    newest = origin - NEWEST_AGE
    gap = target - newest
    day = target - STEPS_PER_DAY * np.ceil(gap / STEPS_PER_DAY).astype(int)
    week = target - 7 * STEPS_PER_DAY * np.ceil(gap / (7 * STEPS_PER_DAY)).astype(int)
    if (day > newest).any() or (week > newest).any():
        msg = "a lag index is younger than the newest usable reading"
        raise ValueError(msg)
    index = np.stack([newest, day, week], axis=1)
    return np.where(index >= 0, anomaly_series[np.maximum(index, 0)], np.nan)


def treat(*, frame: pl.DataFrame, cache, site: str, arm: str) -> pl.DataFrame:  # noqa: ANN001
    """Return the frame with the treated `mean` and `resid`, and the `lags_missing` count."""
    anomalies = lag_anomalies(frame=frame, cache=cache, site=site)
    mean = frame["mean"].to_numpy().astype(np.float64)
    resid = frame["resid"].to_numpy().astype(np.float64)
    origin = frame["power_fcst_init_time"]
    valid = frame["valid_time"]
    band = stack.lead_band(lead_hours=frame["lead_hours"].to_numpy())
    usable = ~frame["switching_mask"].to_numpy() & np.isfinite(resid)
    shift = np.zeros(frame.height)
    month = FIRST_CUT
    while month <= LAST_CUT:
        following = month.replace(year=month.year + (month.month == 12), month=month.month % 12 + 1)
        before = (origin < month).to_numpy() & (valid < month).to_numpy()
        now = ((origin >= month) & (origin < following)).to_numpy()
        if arm != "zero" and now.any():
            for b in range(stack.N_BANDS):
                fit_rows = before & usable & (band == b)
                rows = now & (band == b)
                terms = fit_term_set(anomalies=anomalies[fit_rows], target=resid[fit_rows], arm=arm)
                if terms is None:
                    print(f"{site} {month:%Y-%m} band {b}: {fit_rows.sum()} fit rows, no shift")
                    continue
                shift[rows] = shift_of(anomalies=anomalies[rows], terms=terms, arm=arm)
        month = following
    treated = mean + shift
    return frame.with_columns(
        pl.Series("mean", treated.astype(np.float32)),
        resid=pl.col("actual").cast(pl.Float64) - pl.Series(treated),
        lags_missing=pl.Series((~np.isfinite(anomalies)).sum(axis=1).astype(np.int8)),
    )


def first_reading(*, cache, site: str) -> datetime:  # noqa: ANN001
    """Return the first time with a reading at a site."""
    times = cache.times_series()
    return next(t for t, p in zip(times, cache.sites[site].present, strict=True) if p)


def write(*, frame: pl.DataFrame, site: str, arm: str, stem: str) -> None:
    """Write a treated, spread-fitted frame atomically."""
    target = OUT_DIR / "arms" / arm / f"{stem}_{site}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    frame.select([*common.BASE_COLUMNS, "arm", "lags_missing"]).write_parquet(temporary)
    temporary.rename(target)


def run_site(*, site: str, arm: str, cache) -> None:  # noqa: ANN001
    """Treat one site (and, for D8, its donors) and write the stack's quantiles."""
    first = {s: first_reading(cache=cache, site=s) for s in ("D2", site)}
    frame = treat(frame=stack.corrected_frame(site=site), cache=cache, site=site, arm=arm)
    if site == "D8":
        donors = [
            treat(frame=stack.corrected_frame(site=d), cache=cache, site=d, arm=arm)
            for d in common.DONORS
        ]
        donor_months = [
            m
            for m in stack.eligible_months(first_reading=first["D2"])
            if m >= stack.D8_DONOR_MONTHS_FROM
        ]
        write(
            frame=stack.with_fold_spread(
                frame=frame, pools=donors, months=donor_months, arm=stack.DONOR_ARM
            ),
            site=site,
            arm=arm,
            stem="stack_d_donor",
        )
        own = stack.eligible_months(first_reading=first["D2"], first_month=stack.D8_FIRST_OWN_MONTH)
        write(
            frame=stack.with_fold_spread(frame=frame, pools=[frame], months=own, arm=stack.OWN_ARM),
            site=site,
            arm=arm,
            stem="stack_d",
        )
        return
    months = stack.eligible_months(first_reading=first[site])
    write(
        frame=stack.with_fold_spread(frame=frame, pools=[frame], months=months, arm=stack.OWN_ARM),
        site=site,
        arm=arm,
        stem="stack_d",
    )


def main() -> None:
    """Treat the named sites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--sites", nargs="+", required=True)
    args = parser.parse_args()
    cache = common.load_cache()
    for site in args.sites:
        run_site(site=site, arm=args.arm, cache=cache)
        print(f"{site} {args.arm} done", flush=True)


if __name__ == "__main__":
    main()
