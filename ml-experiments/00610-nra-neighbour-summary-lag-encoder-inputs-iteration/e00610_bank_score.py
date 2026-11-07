# ruff: noqa: E402, D103, E501
"""Experiment 610: absorbed shares of one arm on the target-side bank, beside experiment 595's A1.

Usage: python e00610_bank_score.py ARM

The share definitions are experiment 587's (`e00587_score.score_case`), as in experiment 595's
`e00595_score.py`: with `s` the sign of the event's depth at the Primary and `D` its absolute depth,
`total = s * (q[E] - q[N]) / D` and `refit = s * (q[E_on_N] - q[N]) / D`, by phase. This module hands
`score_case` a copy of experiment 587's common module whose `world_dir`, `ARMS`, and `LAG_FREE` are
this experiment's, and adds experiment 595's rows for A1 and the reference arms.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ARM = sys.argv[1]

import e00587_common as c587  # ty: ignore[unresolved-import]
import e00587_score as s587  # ty: ignore[unresolved-import]
import e00595_common as c595  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT = ML_EXPERIMENTS_DATA_DIR / "00610-nra-neighbour-summary-lag-encoder-inputs-iteration"
DIR = ARM.replace("-", "").replace("+", "_")


def world_dir(*, arm: str, world: str, on: str | None = None) -> Path:
    return ROOT / "bank" / DIR / (world if on is None else f"{world}_on_{on}")


s587.common = SimpleNamespace(**{k: getattr(c587, k) for k in dir(c587) if not k.startswith("__")})
s587.common.world_dir = world_dir
s587.common.ARMS = {ARM: ("610", "P5")}
s587.common.LAG_FREE = frozenset()


def main() -> None:
    cases = ["verified", *[f"inj{i}" for i in c587.designs()["spec_id"].to_list()]]
    frames, dropped = [], 0
    for case in cases:
        table, n = s587.score_case(arm=ARM, case=case)
        frames.append(table)
        dropped += n
    reference = pl.read_parquet(c595.OUT / "scores" / "event_shares.parquet").filter(
        pl.col("arm").is_in(["A1", *c595.REFERENCE_ARMS])
    )
    out = ROOT / "bank" / "scores"
    out.mkdir(parents=True, exist_ok=True)
    shares = pl.concat([reference, *frames], how="diagonal_relaxed")
    shares.write_parquet(out / f"event_shares_{DIR}.parquet")
    print(shares.height, "rows;", dropped, "cells dropped")
    kind = (
        pl.when(pl.col("case") == "verified").then(pl.lit("verified")).otherwise(pl.lit("injected"))
    )
    print(
        shares.with_columns(kind=kind)
        .group_by("arm", "kind", "phase")
        .agg(
            pl.col("total_q50").mean(),
            pl.col("refit_q50").mean(),
            pl.len().alias("cells"),
        )
        .sort("kind", "phase", "arm")
    )


if __name__ == "__main__":
    main()
