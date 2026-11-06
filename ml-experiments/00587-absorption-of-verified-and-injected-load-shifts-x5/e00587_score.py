# ruff: noqa: E501
"""Experiment 587: absorbed shares of every case, arm, refit day, and phase, and their summaries.

Reads the forecast files `e00587_run.py` wrote (E: fit and forecast in the event world; E_on_N: the
same weights forecasting with no-event inputs; N: fit and forecast in the no-event world) and writes
`scores/event_shares.parquet` (per case, event, arm, refit day, phase, quantile: total and refit
share, and the number of half-hours) in the experiment's data folder. Shares follow the plan:

    total = s * (q[E] - q[N]) / D        refit = s * (q[E_on_N] - q[N]) / D

where `s` is the sign of the event's depth at the Primary, `D` its absolute depth (share of
capacity), and `q` is a forecast quantile at that Primary, averaged over seeds and half-hours.
"""

from itertools import product
from typing import Final

import e00587_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

QUANTILES: Final[tuple[str, ...]] = ("q01", "q05", "q50", "q95", "q99")
PHASES: Final[tuple[str, ...]] = ("before", "first_4_days", "day_4_to_end", "after_7_days")
MIN_HALF_HOURS: Final[int] = 48
KEYS: Final[list[str]] = ["seed", "refit_day", "time_series_id", "power_fcst_init_time", "valid_time"]


def read_world(*, arm: str, world: str, on: str | None, months: list[str]) -> pl.DataFrame:
    """Return the forecasts of an arm's fits in a world, all months, refit days, and seeds."""
    tag = common.ARMS[arm][1]
    d = common.world_dir(arm=arm, world=world, on=on) / "work" / "fits" / tag
    frames = []
    for month, day, seed in product(months, (0, 14), common.SEEDS):
        f = d / f"{month}_d{day}_s{seed}.parquet"
        if f.exists():
            frames.append(
                pl.read_parquet(f).with_columns(
                    pl.lit(seed).alias("seed"), pl.lit(day).alias("refit_day"),
                    pl.lit(month).alias("month"),
                )
            )  # fmt: skip
    assert frames, f"no forecasts in {d} for {months}"
    return pl.concat(frames)


def phase_expr(*, onset: int, end: int) -> pl.Expr:
    """Return the phase of a row from its valid time (the event runs from `onset` to `end`, in grid indices)."""
    idx = (
        (pl.col("valid_time").dt.epoch("us") - int(common.GRID_START.timestamp() * 1e6))
        // 1_800_000_000
    )
    return (
        pl.when(idx < onset).then(pl.lit("before"))
        .when(idx < onset + 4 * common.PER_DAY).then(pl.lit("first_4_days"))
        .when(idx < end).then(pl.lit("day_4_to_end"))
        .when(idx < end + 7 * common.PER_DAY).then(pl.lit("after_7_days"))
    )  # fmt: skip


def score_case(*, arm: str, case: str) -> pl.DataFrame:
    """Return the share table of one arm and case: one row per event, refit day, phase, and quantile."""
    ev, no = common.case_worlds(case=case)
    months = common.case_months(case=case)
    e = read_world(arm=arm, world=ev, on=None, months=months)
    en = read_world(arm=arm, world=ev, on=no, months=months)
    n = read_world(arm=arm, world=no, on=None, months=months)
    rows = []
    for event in common.case_events(case=case):
        cols = [*KEYS, *QUANTILES]

        def sel(frame: pl.DataFrame, event=event, cols=cols) -> pl.DataFrame:
            return frame.filter(pl.col("time_series_id") == event.time_series_id).select(cols)

        j = (
            sel(e).join(sel(en), on=KEYS, suffix="_en", how="inner")
            .join(sel(n), on=KEYS, suffix="_n", how="inner")
            .with_columns(phase_expr(onset=event.onset_idx, end=event.end_idx).alias("phase"))
            .drop_nulls("phase")
        )  # fmt: skip
        for q in QUANTILES:
            j = j.with_columns(
                (event.sign * (pl.col(q) - pl.col(f"{q}_n")) / abs(event.depth)).alias(f"total_{q}"),
                (event.sign * (pl.col(f"{q}_en") - pl.col(f"{q}_n")) / abs(event.depth)).alias(f"refit_{q}"),
            )  # fmt: skip
        agg = (
            j.group_by("refit_day", "phase", "seed")
            .agg(
                pl.len().alias("n_half_hours"),
                *[pl.col(f"{k}_{q}").mean() for k in ("total", "refit") for q in QUANTILES],
            )
            .group_by("refit_day", "phase")
            .agg(pl.col("n_half_hours").mean(), *[pl.col(f"{k}_{q}").mean() for k in ("total", "refit") for q in QUANTILES])
            .with_columns(
                pl.lit(arm).alias("arm"), pl.lit(case).alias("case"), pl.lit(event.label).alias("label"),
                pl.lit(event.depth).alias("depth"),
            )
        )  # fmt: skip
        rows.append(agg)
    return pl.concat(rows)


def bootstrap_mean(*, values: np.ndarray, blocks: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Return the mean and its 95% interval over block resamples (a block is one design, both Primaries)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(blocks)
    means = np.array([values[blocks == b].mean() for b in ids])
    draws = rng.choice(means, size=(n, len(ids))).mean(axis=1)
    return float(means.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    cases = ["verified", *[f"inj{i}" for i in common.designs()["spec_id"].to_list()]]
    frames = []
    for arm in common.ARMS:
        for case in cases:
            try:
                frames.append(score_case(arm=arm, case=case))
            except (AssertionError, FileNotFoundError) as exc:
                print("missing", arm, case, exc)
    out = common.OUT / "scores"
    out.mkdir(parents=True, exist_ok=True)
    shares = pl.concat(frames)
    shares.write_parquet(out / "event_shares.parquet")
    print(shares.height, "rows")


if __name__ == "__main__":
    main()
