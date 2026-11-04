# ruff: noqa: E501, ANN201, D103, PTH123, SIM115
"""Score the student S-DW (546) against L-1d, A-pool, and the teacher R-DW (539) on 533's rows. Reuses 533's loader and 521's statistics."""

import json
import math
import os

import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

s.MONTHS = tuple(os.environ.get("E546_MONTHS", " ".join(s.MONTHS)).split())
OUT = os.environ.get("E546_OUT", "scores")

DD = ML_EXPERIMENTS_DATA_DIR
W = DD / "00546-teacher-student-distillation-into-lag-free-demand-network"
E524 = DD / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator/work/scores"
K = ["primary", "origin", "target"]
ARMS = {
    "S-DW": "S-DW__crps",
    "R-DW": "R-DW__crps",
    "A-pool": "A-pool__crps",
    "L-1d": "L-1d__crps",
    "L-Ww": "L-Ww__crps",
    "S2-J-0.4x": "S2-J-0.4x__crps",
    "K2-S1": "K2-S1__crps",
    "manual_heuristic": "manual_heuristic__crps",
}
CONTRASTS = [
    ("S-DW", "A-pool", True),
    ("S-DW", "L-1d", True),
    ("S-DW", "R-DW", False),
    ("S-DW", "K2-S1", False),
    ("S-DW", "manual_heuristic", False),
    ("R-DW", "A-pool", False),
    ("R-DW", "L-1d", False),
    ("L-1d", "A-pool", False),
]
TESTS = 2 * 7 * 2  # 2 counted contrasts x 7 bands x 2 sets


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
                W / "work/scores" / f"rows_{m}.parquet", columns=[*K, "S-DW__crps", "R-DW__crps"]
            )
            for m in s.MONTHS
        ]
    )
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
        "rows main16 with S-DW,R-DW finite": int(t["main16"].sum()),
        "columns used": need,
    }
    (W / OUT).mkdir(exist_ok=True, parents=True)
    json.dump(chk, open(W / OUT / "checks.json", "w"), indent=1)
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
    pl.DataFrame(absolute).write_csv(W / OUT / "absolute.csv")
    pl.DataFrame(contrasts).write_csv(W / OUT / "contrasts.csv")


main()
