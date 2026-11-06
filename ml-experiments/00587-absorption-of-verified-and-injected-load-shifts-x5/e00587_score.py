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
KEYS: Final[list[str]] = [
    "seed",
    "refit_day",
    "time_series_id",
    "power_fcst_init_time",
    "valid_time",
]


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
        pl.col("valid_time").dt.epoch("us") - int(common.GRID_START.timestamp() * 1e6)
    ) // 1_800_000_000
    return (
        pl.when(idx < onset).then(pl.lit("before"))
        .when(idx < onset + 4 * common.PER_DAY).then(pl.lit("first_4_days"))
        .when(idx < end).then(pl.lit("day_4_to_end"))
        .when(idx < end + 7 * common.PER_DAY).then(pl.lit("after_7_days"))
    )  # fmt: skip


def check_complete(*, frame: pl.DataFrame, months: list[str], label: str) -> None:
    """Assert that every month, refit day, and seed of a world has forecasts (an `.empty` window would hide here)."""
    have = set(frame.select("month", "refit_day", "seed").unique().iter_rows())
    want = set(product(months, (0, 14), common.SEEDS))
    assert have == want, f"{label}: missing {sorted(want - have)}"


def score_case(*, arm: str, case: str) -> tuple[pl.DataFrame, int]:
    """Return the share table of one arm and case (one row per event, month, refit day, phase) and the number of dropped thin cells."""
    ev, no = common.case_worlds(case=case)
    months = common.case_months(case=case)
    e = read_world(arm=arm, world=ev, on=None, months=months)
    en = e if arm in common.LAG_FREE else read_world(arm=arm, world=ev, on=no, months=months)
    n = read_world(arm=arm, world=no, on=None, months=months)
    for frame, label in ((e, "E"), (en, "E_on_N"), (n, "N")):
        check_complete(frame=frame, months=months, label=f"{arm} {case} {label}")
    rows, dropped = [], 0
    for event in common.case_events(case=case):
        cols = [*KEYS, "month", *QUANTILES]

        def sel(
            frame: pl.DataFrame, event: common.Event = event, cols: list[str] = cols
        ) -> pl.DataFrame:
            return frame.filter(pl.col("time_series_id") == event.time_series_id).select(cols)

        parts = [sel(e), sel(en), sel(n)]
        j = (
            parts[0].join(parts[1], on=[*KEYS, "month"], suffix="_en", how="inner")
            .join(parts[2], on=[*KEYS, "month"], suffix="_n", how="inner")
        )  # fmt: skip
        assert j.height == parts[0].height == parts[1].height == parts[2].height, (
            f"{arm} {case} {event.label}: join lost rows {[p.height for p in parts]} -> {j.height}"
        )
        j = j.with_columns(
            phase_expr(onset=event.onset_idx, end=event.end_idx).alias("phase")
        ).drop_nulls("phase")
        for q in QUANTILES:
            j = j.with_columns(
                (event.sign * (pl.col(q) - pl.col(f"{q}_n")) / abs(event.depth)).alias(f"total_{q}"),
                (event.sign * (pl.col(f"{q}_en") - pl.col(f"{q}_n")) / abs(event.depth)).alias(f"refit_{q}"),
            )  # fmt: skip
        per_seed = j.group_by("month", "refit_day", "phase", "seed").agg(
            pl.len().alias("n_half_hours"),
            *[pl.col(f"{k}_{q}").mean() for k in ("total", "refit") for q in QUANTILES],
        )
        agg = per_seed.group_by("month", "refit_day", "phase").agg(
            pl.col("n_half_hours").mean(),
            *[pl.col(f"{k}_{q}").mean() for k in ("total", "refit") for q in QUANTILES],
        )
        dropped += agg.filter(pl.col("n_half_hours") < MIN_HALF_HOURS).height
        agg = agg.filter(pl.col("n_half_hours") >= MIN_HALF_HOURS).with_columns(
            pl.lit(arm).alias("arm"), pl.lit(case).alias("case"), pl.lit(event.label).alias("label"),
            pl.lit(event.depth).alias("depth"),
        )  # fmt: skip
        rows.append(agg)
    return pl.concat(rows), dropped


def noise_floor(*, arm: str) -> pl.DataFrame:
    """Return the largest and mean absolute difference of the q50 forecast between this experiment's `obs` refit and the stored fit of the harness' own experiment.

    A refit on the same power with the same seeds should reproduce the stored forecast; the difference
    (as a share of capacity) is the GPU non-determinism floor that a share is read against.
    """
    harness, tag = common.ARMS[arm]
    stored_root = {"565": "00565-drift-embedding-variants-six-designs",
                   "574": "00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms"}[harness]  # fmt: skip
    rows = []
    for month, day, seed in product(common.ALL_MONTHS, (0, 14), common.SEEDS):
        mine = (
            common.world_dir(arm=arm, world="obs")
            / "work"
            / "fits"
            / tag
            / f"{month}_d{day}_s{seed}.parquet"
        )
        theirs = (
            common.ML_EXPERIMENTS_DATA_DIR
            / stored_root
            / "work"
            / "fits"
            / tag
            / f"{month}_d{day}_s{seed}.parquet"
        )
        if not (mine.exists() and theirs.exists()):
            continue
        j = pl.read_parquet(mine).join(pl.read_parquet(theirs), on=KEYS[2:], suffix="_stored")
        d = (j["q50"] - j["q50_stored"]).abs()
        rows.append({"arm": arm, "month": month, "refit_day": day, "seed": seed, "rows": j.height,
                     "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean())})  # fmt: skip  # ty: ignore[invalid-argument-type]
    return pl.DataFrame(rows)


def bootstrap_mean(
    *, values: np.ndarray, blocks: np.ndarray, n: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Return the mean and its 95% interval over block resamples (a block is one design, both Primaries)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(blocks)
    means = np.array([values[blocks == b].mean() for b in ids])
    draws = rng.choice(means, size=(n, len(ids))).mean(axis=1)
    return float(means.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    """Score every arm and case and write the share and noise-floor tables."""
    cases = ["verified", *[f"inj{i}" for i in common.designs()["spec_id"].to_list()]]
    frames, dropped = [], 0
    for arm in common.ARMS:
        for case in cases:
            table, n_dropped = score_case(arm=arm, case=case)
            frames.append(table)
            dropped += n_dropped
    out = common.OUT / "scores"
    out.mkdir(parents=True, exist_ok=True)
    shares = pl.concat(frames)
    shares.write_parquet(out / "event_shares.parquet")
    floor = pl.concat([noise_floor(arm=a) for a in common.ARMS])
    floor.write_parquet(out / "noise_floor.parquet")
    print(
        shares.height,
        "rows;",
        dropped,
        "cells dropped for fewer than",
        MIN_HALF_HOURS,
        "half-hours",
    )
    print(floor.group_by("arm").agg(pl.col("max_abs_diff").max(), pl.col("mean_abs_diff").mean()))


if __name__ == "__main__":
    main()
