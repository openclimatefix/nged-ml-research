# ruff: noqa: ANN201, D103, PTH123, SIM115
"""Score M0, M0lr, and M1 (545) against L-1d, A-pool, and R-DW (539) on 511's rows.

Reuses experiment 539's scorer, 533's loader, and 521's statistics.
"""

import json
import math

import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DD = ML_EXPERIMENTS_DATA_DIR
W = DD / "00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out"
W539 = DD / "00539-l1d-plus-day-and-week-lags"
E524 = DD / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator/work/scores"
K = ["primary", "origin", "target"]
ARMS = {
    "M0": "M0__crps",
    "M1": "M1__crps",
    "M0lr": "M0lr__crps",
    "R-DW": "R-DW__crps",
    "A-pool": "A-pool__crps",
    "L-1d": "L-1d__crps",
    "L-Ww": "L-Ww__crps",
    "manual_heuristic": "manual_heuristic__crps",
}
CONTRASTS = [
    ("M0", "L-1d", True),
    ("M1", "L-1d", True),
    ("M1", "M0", False),
    ("M0lr", "L-1d", False),
    ("M0", "M0lr", False),
    ("M1", "M0lr", False),
    ("M0", "A-pool", False),
    ("M1", "A-pool", False),
    ("M0", "R-DW", False),
    ("M1", "R-DW", False),
    ("M1", "manual_heuristic", False),
    ("L-1d", "A-pool", False),
]
TESTS = 2 * 7 * 2  # 2 counted contrasts x 7 bands x 2 sets
NEW = [
    a
    for a in ("M0", "M1", "M0lr")
    if (W / "work/scores" / f"rows_{s.MONTHS[0]}.parquet").exists()
    and f"{a}__crps" in pl.read_parquet_schema(W / "work/scores" / f"rows_{s.MONTHS[0]}.parquet")
]


def table():
    base = s.load()
    extra = pl.concat(
        [
            pl.read_parquet(
                E524 / f"rows_{m}.parquet", columns=[*K, "A-pool__crps", "manual_heuristic__crps"]
            )
            for m in s.MONTHS
        ]
    )
    rdw = pl.concat(
        [
            pl.read_parquet(W539 / "work/scores" / f"rows_{m}.parquet", columns=[*K, "R-DW__crps"])
            for m in s.MONTHS
        ]
    )
    mine = pl.concat(
        [
            pl.read_parquet(
                W / "work/scores" / f"rows_{m}.parquet", columns=[*K, *[f"{a}__crps" for a in NEW]]
            )
            for m in s.MONTHS
        ]
    )
    mine = mine.join(rdw, on=K, how="left", validate="1:1", maintain_order="left")
    assert base.select(K).equals(extra.select(K))
    drop = [c for c in ("A-pool__crps", "manual_heuristic__crps") if c in base.columns]
    t = pl.concat([base.drop(drop), extra.drop(K)], how="horizontal")
    n0 = t.height
    t = t.join(mine, on=K, how="left", validate="1:1", maintain_order="left")
    assert t.height == n0
    return t


def main():
    t = table().with_columns(band=s.band_expr())
    need = [
        c for k2, c in ARMS.items() if k2 not in ("manual_heuristic", "L-Ww") and c in t.columns
    ]
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
                if ARMS[b] not in sub.columns or ARMS[a] not in sub.columns:
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
