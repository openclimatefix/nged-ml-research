# ruff: noqa: ANN201, D103, E501
"""Score experiment 570's arms: absolute CRPS and month-block paired contrasts with the MDE beside each.

Reuses experiment 521's loader, row sets, and `block_stats` (5 monthly blocks, so intervals sit near
min-to-max); the per-arm row CRPS comes from `e00570_rows.py`. MDEs are the brief's two-seed values
from experiments 554 and 564 (eight 0.113, six 0.128, eight without D8 0.092, D8 alone 0.593).
"""

import json

import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W = ML_EXPERIMENTS_DATA_DIR / "00570-price-encoder-part-1-known-answer-test-and-pretrained-stack" / "work"
K = ["primary", "origin", "target"]
POOLED = "24-120h"
TESTS = 6
MDE = {"eight": 0.113, "six": 0.128, "eight_without_D8": 0.092, "D8_alone": 0.593, "report": float("nan")}
ARMS = (
    "M0", "A6-res", "A6-raw", "A6-p1", "A6-p2", "A6-noise", "A6-const",
    "A1-res", "A1-p2", "A1-noise", "A1-const", "A3-res", "A3-noise",
    "A1-res-oracle", "A1-p2-oracle", "A3-res-oracle",
)  # fmt: skip
CONTRASTS = [  # (a, b, counted test label)
    ("A6-res", "A6-noise", "T1/T2"), ("A6-res", "A6-const", ""), ("A6-res", "A6-p1", ""), ("A6-res", "A6-p2", ""),
    ("A6-raw", "A6-res", ""), ("A6-res", "M0", ""), ("A6-noise", "M0", ""), ("A6-const", "M0", ""),
    ("A1-res", "A1-noise", "T3/T4"), ("A1-res", "A1-const", ""), ("A1-p2", "A1-noise", ""), ("A1-res", "M0", ""),
    ("A1-res-oracle", "A1-res", ""), ("A1-p2-oracle", "A1-p2", ""),
    ("A3-res", "A3-noise", ""), ("A3-res", "M0", ""), ("A3-res-oracle", "A3-res", ""),
]  # fmt: skip


def table() -> pl.DataFrame:
    base = s.load().select(*K, "group", "cls", "lead_hours", "tgt_real", "cold", "xmas", "month", "day", "main16")
    cols = [f"{a}__crps" for a in ARMS]
    mine = pl.concat([pl.read_parquet(W / "scores" / f"rows_{m}.parquet", columns=[*K, *cols]) for m in s.MONTHS])
    t = base.join(mine, on=K, how="left", validate="1:1", maintain_order="left")
    fin = pl.all_horizontal([pl.col(c).is_finite() for c in cols])
    return t.with_columns(main16=pl.col("main16") & fin, band=s.band_expr())


def main():
    t = table()
    rng = np.random.default_rng(0)
    d8 = [p for p, lab in s.PRIMARY_LABELS.items() if lab == "D8"]
    assert len(d8) == 1, d8
    absolute, contrasts = [], []
    for name in ("eight", "six", "eight_without_D8", "D8_alone", "report"):
        base_name = "eight" if name.startswith(("eight", "D8")) else name
        part = s.row_set(table=t, name=base_name)
        if name == "eight_without_D8":
            part = part.filter(pl.col("primary") != d8[0])
        elif name == "D8_alone":
            part = part.filter(pl.col("primary") == d8[0])
        for band in (*s.BANDS, s.ALL, POOLED):
            sub = part.filter((pl.col("lead_hours") >= 24) & (pl.col("lead_hours") < 120)) if band == POOLED else s.in_band(part=part, band=band)
            if sub.height == 0:
                continue
            for arm in ARMS:
                absolute.append({"set": name, "band": band, "arm": arm, "n": sub.height, "crps": float(sub[f"{arm}__crps"].mean())})
            for a, b, label in CONTRASTS:
                d = sub.select("month", "day", d=pl.col(f"{a}__crps") - pl.col(f"{b}__crps"))
                st = s.block_stats(diff=d, rng=rng, tests=TESTS)
                fp = st["folds_positive"]
                contrasts.append({
                    "set": name, "band": band, "contrast": f"{a} - {b}", "counted": label if band == POOLED and name in {"eight", "six"} else "",
                    "tests": TESTS, "mde": MDE[name], **st,
                    "exceeds_mde_5_of_5": bool(abs(st["difference"]) > MDE[name] and fp in {0, st["folds"]}),
                })  # fmt: skip
    pl.DataFrame(absolute).write_csv(W / "scores/absolute.csv")
    pl.DataFrame(contrasts).write_csv(W / "scores/contrasts.csv")
    json.dump({"rows": t.height, "main16 rows with all arms finite": int(t["main16"].sum())}, open(W / "scores/checks.json", "w"), indent=1)
    print("rows", t.height, "main16", int(t["main16"].sum()))


main()
