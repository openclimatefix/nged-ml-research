# ruff: noqa: ANN201, D103, E501
"""Seed spread and minimum detectable effect of experiment 562 (the formula of experiment 545's scorer).

Per-seed CRPS of M0 and of arms B1 and B1p give the fold-by-seed spread; the minimum detectable
effect of a contrast of two 2-seed means is 1.96 x sqrt(2) x (residual sd) / sqrt(seeds).
"""

import math

import e00521_stats as s  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W = ML_EXPERIMENTS_DATA_DIR / "00562-modular-lag-free-network-null-input-retry-with-placebos"
K = ["primary", "origin", "target"]
PAIRS = {"M0": ("M0s0", "M0s1"), "B1": ("B1s0", "B1s1"), "B1p": ("B1ps0", "B1ps1")}


def main():
    cols = [f"{a}__crps" for p in PAIRS.values() for a in p]
    base = s.load()
    mine = pl.concat(
        [
            pl.read_parquet(W / "work/scores" / f"rows_{m}.parquet", columns=[*K, *cols])
            for m in s.MONTHS
        ]
    )
    t = base.join(mine, on=K, how="left", validate="1:1", maintain_order="left").filter(
        pl.col("main16")
    )
    rows = []
    for group in ("eight", "six"):
        part = s.row_set(table=t, name=group)
        for slice_name, sub in (("all", part), ("cold_night", part.filter(pl.col("cold")))):
            for name, (a, b) in PAIRS.items():
                mm = (
                    sub.group_by("month")
                    .agg(pl.col(f"{a}__crps").mean(), pl.col(f"{b}__crps").mean())
                    .sort("month")
                )
                v = mm.select(f"{a}__crps", f"{b}__crps").to_numpy()
                f, k = v.shape
                r = v - v.mean(axis=1, keepdims=True)
                mde = (
                    1.96
                    * math.sqrt(2)
                    * math.sqrt(float((r**2).sum()) / (f * (k - 1)))
                    / math.sqrt(k)
                )
                rows.append({"set": group, "slice": slice_name, "arm": name, "seed0": float(sub[f"{a}__crps"].mean()), "seed1": float(sub[f"{b}__crps"].mean()), "mde": mde})  # fmt: skip
    out = pl.DataFrame(rows)
    out.write_csv(W / "seed_spread_mde.csv")
    with pl.Config(tbl_rows=30):
        print(out)


main()
