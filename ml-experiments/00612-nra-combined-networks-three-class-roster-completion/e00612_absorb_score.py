# ruff: noqa: D103, ANN201
"""Experiment 612: absorbed shares of the class 2 candidate and the lag-free K3 (CPU).

`total = s * (q[E] - q[N]) / D` and `refit = s * (q[E_on_N] - q[N]) / D`, by phase, as experiments
595 and 599. `score_case` of experiment 587 is reused unchanged, with its `common` module replaced
by one that points at this experiment's worlds. Writes `absorb/event_shares_<kind>.parquet`.

Usage: python e00612_absorb_score.py KIND   (c2 or k3d5)
"""

import sys

import e00587_common as c587  # ty: ignore[unresolved-import]
import e00587_score as s587  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

KIND = sys.argv[1]
OUT = (
    ML_EXPERIMENTS_DATA_DIR / "00612-nra-combined-networks-three-class-roster-completion" / "absorb"
)
TAG = {"c2": "P5", "k3d5": "D5"}[KIND]


def world_dir(*, arm: str, world: str, on: str | None = None):
    return OUT / KIND / (world if on is None else f"{world}_on_{on}")


s587.common = type("Common", (), {k: getattr(c587, k) for k in dir(c587) if not k.startswith("__")})
s587.common.world_dir = world_dir
s587.common.ARMS = {KIND: ("", TAG)}
s587.common.LAG_FREE = frozenset({"k3d5"}) if KIND == "k3d5" else frozenset()

cases = ["verified", *[f"inj{i}" for i in c587.designs()["spec_id"].to_list()]]
frames, dropped = [], 0
for case in cases:
    table, n = s587.score_case(arm=KIND, case=case)
    frames.append(table)
    dropped += n
shares = pl.concat(frames, how="diagonal_relaxed")
shares.write_parquet(OUT / f"event_shares_{KIND}.parquet")
print(shares.height, "rows;", dropped, "cells dropped")
