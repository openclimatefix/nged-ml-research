# ruff: noqa: ANN001, ANN201, D103, E501, PERF401
"""Part 3 scores: CRPS of the lag-removal and late-reading scenarios against L-1d and against the readings present.

Pre-registered counted contrasts (Bonferroni count 28 = 2 contrasts x 7 lead bands x 2 row sets):
R-DROP-none minus L-1d, and R-DW-none minus L-1d. Every other contrast is report-only.
"""

import json
import math
import os
from pathlib import Path

import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

DD = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
OUT = DD / "00551-lag-light-release-safety-lag-timestamp-audit-holidays-and-lag-removal"
E524 = DD / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator/work/scores"
K = ["primary", "origin", "target"]
SCEN = ("base", "none", "noday", "noweek", "age1", "age2", "age3", "rand25")
ARMS = [f"{a}-{sc}" for a in ("R-DW", "R-DROP") for sc in SCEN]
TESTS = 28


def table():
    base = s.load()
    a = pl.concat(
        [
            pl.read_parquet(E524 / f"rows_{m}.parquet", columns=[*K, "A-pool__crps"])
            for m in s.MONTHS
        ]
    )
    base = base.drop("A-pool__crps", strict=False).join(
        a, on=K, how="left", validate="1:1", maintain_order="left"
    )
    mine = pl.concat(
        [
            pl.read_parquet(
                OUT / "work/scores" / f"rows_{m}.parquet",
                columns=[*K, *[f"{x}__crps" for x in ARMS]],
            )
            for m in s.MONTHS
        ]
    )
    n0 = base.height
    t = base.join(mine, on=K, how="left", validate="1:1", maintain_order="left")
    assert t.height == n0
    return t.with_columns(
        band=s.band_expr(), headline=(pl.col("lead_hours") >= 24) & (pl.col("lead_hours") < 120)
    )


def mde(part):
    seeds = ["L-1d-seed0__crps", "L-1d-seed1__crps"]
    m = (
        part.group_by("month")
        .agg([pl.col(c).mean() for c in seeds])
        .sort("month")
        .select(seeds)
        .to_numpy()
    )
    f, k = m.shape
    r = m - m.mean(axis=1, keepdims=True)
    return 1.96 * math.sqrt(2) * math.sqrt(float((r**2).sum()) / (f * (k - 1))) / math.sqrt(k)


def main():
    t = table()
    need = [f"{a}__crps" for a in ARMS] + ["L-1d__crps", "A-pool__crps"]
    t = t.with_columns(
        main16=pl.col("main16") & pl.all_horizontal([pl.col(c).is_finite() for c in need])
    )
    chk = {"rows": t.height, "rows_main16_all_finite": int(t["main16"].sum())}
    (OUT / "scores/removal_checks.json").write_text(json.dumps(chk, indent=1))
    print(chk)
    rng = np.random.default_rng(0)
    absolute, contrasts = [], []
    for name in ("eight", "six", "report"):
        part = s.row_set(table=t, name=name)
        for band in (*s.BANDS, s.ALL, "24-120h"):
            sub = (
                part.filter(pl.col("headline"))
                if band == "24-120h"
                else s.in_band(part=part, band=band)
            )
            if sub.height == 0:
                continue
            m = mde(sub)
            for col in ("L-1d", "A-pool", *ARMS):
                absolute.append(
                    {
                        "set": name,
                        "band": band,
                        "arm": col,
                        "n": sub.height,
                        "crps": float(sub[f"{col}__crps"].mean()),
                    }
                )
            changed = [a for a in ARMS if not a.endswith("-base")]
            pairs = [(a, "L-1d") for a in changed] + [
                (a, f"{a.rsplit('-', 1)[0]}-base") for a in changed
            ]
            pairs += [("R-DW-base", "L-1d"), ("R-DROP-base", "L-1d"), ("R-DROP-none", "R-DW-none")]
            for a, b in pairs:
                d = sub.select("month", "day", d=pl.col(f"{a}__crps") - pl.col(f"{b}__crps"))
                st = s.block_stats(diff=d, rng=rng, tests=TESTS)
                counted = (
                    (a, b) in (("R-DROP-none", "L-1d"), ("R-DW-none", "L-1d"))
                    and band in s.BANDS
                    and name != "report"
                )
                contrasts.append(
                    {
                        "set": name,
                        "band": band,
                        "contrast": f"{a} - {b}",
                        "counted": counted,
                        "tests": TESTS,
                        "mde": m,
                        **st,
                    }
                )
    pl.DataFrame(absolute).write_csv(OUT / "scores/removal_absolute.csv")
    c = pl.DataFrame(contrasts)
    c.write_csv(OUT / "scores/removal_contrasts.csv")
    with pl.Config(tbl_rows=200, tbl_width_chars=250):
        show = c.filter((pl.col("band") == "24-120h") | (pl.col("band") == s.ALL)).filter(
            pl.col("set") != "report"
        )
        print(
            show.select(
                "set",
                "band",
                "contrast",
                pl.col("difference").round(3),
                pl.col("low_month").round(3),
                pl.col("high_month").round(3),
                pl.col("mde").round(3),
                "folds_positive",
                "folds",
            )
        )
    print("counted tests:", int(c["counted"].sum()))


main()
