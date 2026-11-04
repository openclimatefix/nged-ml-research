# ruff: noqa: ANN201, D103, E402, PTH123, SIM115
"""Score the 558 arms against R-DW (539), L-1d, A-pool and the manual heuristic.

Uses 533's rows, 533's loader and 521's statistics.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path[:0] = [str(p) for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]
import e00521_stats as s  # ty: ignore[unresolved-import]

DD = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
W = DD / "00558-lags-in-pretraining-and-24-hour-cap-lag-light-demand-network"
E524 = DD / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator/work/scores"
K = ["primary", "origin", "target"]
NEW = tuple(
    a
    for a in (
        "R-DW-24",
        "R-DW-24-absent",
        "R-DW-24-PT",
        "R-DW-24-PT-absent",
        "A-pool-24",
        "A-pool-24-absent",
    )
    if (W / "work/seed_preds" / a).is_dir()
)
OLD = ("R-DW", "A-pool", "L-1d", "manual_heuristic")
ARMS = {a: f"{a}__crps" for a in (*NEW, *OLD)}
# counted contrasts: P1 cap cost, P2 pretraining lags, P3 A-pool-24 versus R-DW-24-PT (4 tests as pre-registered)
CONTRASTS = [
    ("R-DW-24", "R-DW", True),
    ("R-DW-24-PT", "R-DW-24", True),
    ("A-pool-24", "R-DW-24-PT", True),
    ("R-DW-24", "L-1d", False),
    ("R-DW-24-PT", "L-1d", False),
    ("R-DW-24-PT", "R-DW", False),
    ("A-pool-24", "A-pool", False),
    ("R-DW-24-absent", "L-1d", False),
    ("R-DW-24-PT-absent", "L-1d", False),
    ("R-DW-24", "manual_heuristic", False),
]
CONTRASTS = [c for c in CONTRASTS if c[0] in NEW and (c[1] in OLD or c[1] in NEW)]
TESTS = 4  # pre-registered Bonferroni count


def table():
    base = s.load()
    extra = pl.concat(
        [
            pl.read_parquet(
                E524 / f"rows_{m}.parquet",
                columns=[*K, "A-pool__crps", "manual_heuristic__crps", "S2-J-0.4x__crps"],
            )
            for m in s.MONTHS
        ]
    )
    mine = pl.concat(
        [
            pl.read_parquet(
                W / "work/scores" / f"rows_{m}.parquet",
                columns=[*K, *(f"{a}__crps" for a in NEW)],
            )
            for m in s.MONTHS
        ]
    )
    dw = pl.concat(
        [
            pl.read_parquet(
                DD / "00539-l1d-plus-day-and-week-lags/work/scores" / f"rows_{m}.parquet",
                columns=[*K, "R-DW__crps"],
            )
            for m in s.MONTHS
        ]
    )
    mine = mine.join(dw, on=K, how="left", validate="1:1", maintain_order="left")
    assert base.select(K).equals(extra.select(K))
    drop = [c for c in ("A-pool__crps", "S2-J-0.4x__crps") if c in base.columns]
    t = pl.concat([base.drop(drop), extra.drop(K)], how="horizontal")
    n0 = t.height
    t = t.join(mine, on=K, how="left", validate="1:1", maintain_order="left")
    assert t.height == n0
    return t


def main():
    t = table().with_columns(band=s.band_expr())
    need = [c for k2, c in ARMS.items() if k2 != "manual_heuristic" and c in t.columns]
    fin = pl.all_horizontal([pl.col(c).is_finite() for c in need])
    t = t.with_columns(main16=pl.col("main16") & fin)
    chk = {
        "rows": t.height,
        "rows main16 with all arms finite": int(t["main16"].sum()),
        "columns used": need,
    }
    json.dump(chk, open(W / "scores/checks.json", "w"), indent=1)
    print(chk)
    rng = np.random.default_rng(0)
    absolute, contrasts = [], []
    seedcols = ["L-1d-seed0__crps", "L-1d-seed1__crps"]
    for name in ("eight", "six", "report"):
        part = s.row_set(table=t, name=name)
        for band in (*s.BANDS, s.ALL):
            sub = s.in_band(part=part, band=band)
            if sub.height == 0:
                continue
            for arm, col in ARMS.items():
                if col in sub.columns:
                    absolute.append(
                        {
                            "set": name,
                            "band": band,
                            "arm": arm,
                            "n": sub.height,
                            "crps": float(sub.filter(pl.col(col).is_finite())[col].mean()),
                        }
                    )
            mm = (
                sub.group_by("month")
                .agg([pl.col(c).mean() for c in seedcols])
                .sort("month")
                .select(seedcols)
                .to_numpy()
            )
            f, k = mm.shape
            r = mm - mm.mean(axis=1, keepdims=True)
            mde = (
                1.96 * math.sqrt(2) * math.sqrt(float((r**2).sum()) / (f * (k - 1))) / math.sqrt(k)
            )
            for a, b, counted in CONTRASTS:
                if ARMS[b] not in sub.columns:
                    continue
                d = sub.select("month", "day", d=pl.col(ARMS[a]) - pl.col(ARMS[b])).filter(
                    pl.col("d").is_finite()
                )
                st = s.block_stats(diff=d, rng=rng, tests=TESTS)
                contrasts.append(
                    {
                        "set": name,
                        "band": band,
                        "contrast": f"{a} - {b}",
                        "counted": counted and band != s.ALL and name != "report",
                        "tests": TESTS,
                        "mde": mde,
                        **st,
                    }
                )
    pl.DataFrame(absolute).write_csv(W / "scores/absolute.csv")
    pl.DataFrame(contrasts).write_csv(W / "scores/contrasts.csv")


main()
