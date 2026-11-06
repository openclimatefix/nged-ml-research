# ruff: noqa: E501
"""Experiment 595: absorbed shares of C1 and A1, joined to experiment 587's shares of M0, D5, and P5.

The share definitions are experiment 587's (`e00587_score.score_case`, reused unchanged): with `s` the
sign of the event's depth at the Primary and `D` its absolute depth,

    total = s * (q[E] - q[N]) / D        refit = s * (q[E_on_N] - q[N]) / D

for a forecast quantile `q` at that Primary, averaged over seeds and half-hours, by phase (before, first
4 days, day 4 to the end, 7 days after the end). `score_case` reads `common.world_dir`, `common.ARMS`,
and `common.LAG_FREE`, so this module hands it a copy of experiment 587's common module with those
three replaced by this experiment's.
"""

from itertools import product
from types import SimpleNamespace

import e00587_common as c587  # ty: ignore[unresolved-import]
import e00587_score as s587  # ty: ignore[unresolved-import]
import e00595_common as common  # ty: ignore[unresolved-import]
import polars as pl

s587.common = SimpleNamespace(**{k: getattr(c587, k) for k in dir(c587) if not k.startswith("__")})
s587.common.world_dir = common.world_dir
s587.common.ARMS = common.NEW_ARMS
s587.common.LAG_FREE = frozenset()

STORED = {
    "C1": common.ML_EXPERIMENTS_DATA_DIR
    / "00584-nra-candidate-pack-x6"
    / "c1"
    / "work"
    / "fits"
    / "P5",
    "A1": common.ML_EXPERIMENTS_DATA_DIR
    / "00591-shared-lag-encoder-auxiliary-shift-depth-loss"
    / "A1"
    / "work"
    / "fits"
    / "P5",
}
"""Where experiments 584 and 591 stored the arms' fits on the observed power (the noise-floor reference)."""


def noise_floor(*, arm: str) -> pl.DataFrame:
    """Return the q50 difference between this experiment's `obs` refit and the stored fit of the arm."""
    rows = []
    for month, day, seed in product(c587.ALL_MONTHS, (0, 14), c587.SEEDS):
        mine = (
            common.world_dir(arm=arm, world="obs")
            / "work"
            / "fits"
            / "P5"
            / f"{month}_d{day}_s{seed}.parquet"
        )
        theirs = STORED[arm] / f"{month}_d{day}_s{seed}.parquet"
        if not (mine.exists() and theirs.exists()):
            continue
        j = pl.read_parquet(mine).join(pl.read_parquet(theirs), on=s587.KEYS[2:], suffix="_stored")
        d = (j["q50"] - j["q50_stored"]).abs()
        rows.append({"arm": arm, "month": month, "refit_day": day, "seed": seed, "rows": j.height,
                     "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean())})  # fmt: skip  # ty: ignore[invalid-argument-type]
    return pl.DataFrame(rows)


def main() -> None:
    """Score C1 and A1 on every case, add experiment 587's shares, and write the tables."""
    cases = ["verified", *[f"inj{i}" for i in c587.designs()["spec_id"].to_list()]]
    frames, dropped = [], 0
    for arm in common.NEW_ARMS:
        for case in cases:
            table, n = s587.score_case(arm=arm, case=case)
            frames.append(table)
            dropped += n
    reference = pl.read_parquet(c587.OUT / "scores" / "event_shares.parquet").filter(
        pl.col("arm").is_in(common.REFERENCE_ARMS)
    )
    out = common.OUT / "scores"
    out.mkdir(parents=True, exist_ok=True)
    shares = pl.concat([reference, *frames], how="diagonal_relaxed")
    shares.write_parquet(out / "event_shares.parquet")
    floor = pl.concat([noise_floor(arm=a) for a in common.NEW_ARMS])
    floor.write_parquet(out / "noise_floor.parquet")
    print(
        shares.height,
        "rows;",
        dropped,
        "cells dropped for fewer than",
        s587.MIN_HALF_HOURS,
        "half-hours",
    )
    print(floor.group_by("arm").agg(pl.col("max_abs_diff").max(), pl.col("mean_abs_diff").mean()))


if __name__ == "__main__":
    main()
