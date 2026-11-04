# ruff: noqa: ANN201, D103, PTH123, SIM115
"""Score M2-a, M2-b, and M2-c against M0 (experiment 554's M0, seeds 0 and 1) on 511's rows.

Reuses experiment 554's scorer structure, 533's loader, and 521's statistics. The
contrast set is: each M2 arm minus M0 over the same 2 seeds (counted), M2-b and M2-c minus M2-a, and
the references M0 (5 seeds) minus `L-1d5`. 42 counted tests: 3 arms x 2 sets x 7 lead bands.
"""

import json
import math

import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DD = ML_EXPERIMENTS_DATA_DIR
W = DD / "00564-pooled-primary-bsp-gsp-m2-reduced"
E524 = DD / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator/work/scores"
K = ["primary", "origin", "target"]
SEED_ARMS = (
    "M2-a", "M2-b", "M2-c", "M0-2", "M0", "L-1d5",
    *[f"{a}-s{i}" for a in ("M2-a", "M2-b", "M2-c", "M0-2") for i in (0, 1)],
)  # fmt: skip
ARMS = {
    **{a: f"{a}__crps" for a in SEED_ARMS if "-s" not in a.removeprefix("M")},
    "A-pool": "A-pool__crps",
    "L-1d": "L-1d__crps",
    "manual_heuristic": "manual_heuristic__crps",
}
COUNTED = [("M2-a", "M0-2"), ("M2-b", "M0-2"), ("M2-c", "M0-2")]
CONTRASTS = [
    *[(a, b, True) for a, b in COUNTED],
    ("M2-b", "M2-a", False),
    ("M2-c", "M2-a", False),
    ("M0-2", "L-1d5", False),
    ("M0", "L-1d5", False),
    ("M2-a", "M0", False),
]
TESTS = 3 * 7 * 2  # 3 counted contrasts x 7 bands x 2 sets
POOLED = "24-120h"
"""The 24-48 h, 48-72 h, and 72-120 h bands pooled (predictions P1 and P2)."""
NEW = [
    a
    for a in SEED_ARMS
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
    mine = pl.concat(
        [
            pl.read_parquet(
                W / "work/scores" / f"rows_{m}.parquet", columns=[*K, *[f"{a}__crps" for a in NEW]]
            )
            for m in s.MONTHS
        ]
    )
    assert base.select(K).equals(extra.select(K))
    drop = [c for c in ("A-pool__crps", "manual_heuristic__crps") if c in base.columns]
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
    seedcols = [f"{a}-s{i}__crps" for a in ("M2-a",) for i in (0, 1)]
    d8 = [p for p, lab in s.PRIMARY_LABELS.items() if lab == "D8"]
    assert len(d8) == 1, d8
    for name in ("eight", "six", "report", "eight_without_D8", "D8_alone"):
        base_name = "eight" if name.startswith(("eight", "D8")) else name
        part = s.row_set(table=t, name=base_name)
        if name == "eight_without_D8":
            part = part.filter(pl.col("primary") != d8[0])
        elif name == "D8_alone":
            part = part.filter(pl.col("primary") == d8[0])
        for band in (*s.BANDS, s.ALL, POOLED):
            if band == POOLED:
                sub = part.filter((pl.col("lead_hours") >= 24) & (pl.col("lead_hours") < 120))
            else:
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
                            "crps": float(sub.filter(pl.col(col).is_finite())[col].mean()),  # ty: ignore[invalid-argument-type]
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
                        "counted": counted
                        and band not in {s.ALL, POOLED}
                        and name in {"eight", "six"},
                        "tests": TESTS,
                        "mde": mde,
                        **st,
                    }
                )
    pl.DataFrame(absolute).write_csv(W / "scores/absolute.csv")
    pl.DataFrame(contrasts).write_csv(W / "scores/contrasts.csv")


main()
