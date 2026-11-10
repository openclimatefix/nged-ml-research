# ruff: noqa: D103, E731, PTH123, SIM115
"""Experiment 534, `e00534_stats.py`: kept for the record.

Score experiment 534's arms against A-pool, L-1d (= R-90), and K2-S1 with experiment 521's row
sets, bands, bootstrap, and MDE formula.
"""

import json
import math

import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = (
    ML_EXPERIMENTS_DATA_DIR / "00534-lag-free-recency-sweep-and-member-ensemble" / "work" / "scores"
)
K = ["primary", "origin", "target"]
NEW = ("R-3", "R-7", "R-14", "R-30", "ENS-3", "ENS-5")
COUNTED = ("R-3", "R-7", "R-14", "R-30", "ENS-3")
ARMS = (*NEW, "R-90", "A-pool", "K2-S1", "manual_heuristic")
TESTS = len(COUNTED) * 7 * 2
SLICES = ("all rows", "no labelled event", "cold nights")


def load() -> pl.DataFrame:
    out = []
    for m in s.MONTHS:
        r = pl.read_parquet(OUT / f"rows_{m}.parquet")
        seeds = pl.read_parquet(
            s.WORK / f"refs_{m}.parquet", columns=[*K, "L-1d-seed0__crps", "L-1d-seed1__crps"]
        )
        out.append(
            r.join(seeds, on=K, how="left", maintain_order="left").with_columns(
                day=(pl.col("origin") // 48).cast(pl.Int32)
            )
        )
    return pl.concat(out).with_columns(band=s.band_expr())


def main() -> None:
    t = load()
    t = t.with_columns(
        main16=pl.col("main16")
        & pl.all_horizontal(
            [pl.col(f"{a}__crps").is_finite() for a in (*NEW, "R-90", "A-pool", "K2-S1")]
        )
    )
    crps = lambda a: f"{a}__crps"
    sl = {
        "all rows": pl.lit(True),
        "no labelled event": ~pl.col("tgt_real"),
        "cold nights": pl.col("cold").cast(pl.Boolean),
    }
    contrasts_def = (
        [(a, "A-pool", a in COUNTED) for a in NEW]
        + [("R-90", "A-pool", False)]
        + [(a, "R-90", False) for a in NEW]
        + [(a, "K2-S1", False) for a in (*NEW, "R-90")]
        + [(a, "manual_heuristic", False) for a in ("R-90", "ENS-3")]
    )
    rng = np.random.default_rng(0)
    absolute, contrasts = [], []
    for name in ("eight", "six", "report"):
        part0 = s.row_set(table=t, name=name)
        for sname, ex in sl.items():
            part = part0.filter(ex)
            for band in (*s.BANDS, s.ALL):
                sub = s.in_band(part=part, band=band)
                if sub.height == 0:
                    continue
                for arm in ARMS:
                    col = crps(arm)
                    absolute.append(
                        {
                            "set": name,
                            "slice": sname,
                            "band": band,
                            "arm": arm,
                            "n": sub.height,
                            "crps": float(sub.filter(pl.col(col).is_finite())[col].mean()),
                            "n_arm": int(sub[col].is_finite().sum()),
                        }
                    )
                cols = ["L-1d-seed0__crps", "L-1d-seed1__crps"]
                mm = (
                    sub.group_by("month")
                    .agg([pl.col(c).mean() for c in cols])
                    .sort("month")
                    .select(cols)
                    .to_numpy()
                )
                f, k = mm.shape
                r = mm - mm.mean(axis=1, keepdims=True)
                mde = (
                    1.96
                    * math.sqrt(2)
                    * math.sqrt(float((r**2).sum()) / (f * (k - 1)))
                    / math.sqrt(k)
                )
                for a, b, counted in contrasts_def:
                    d = sub.select("month", "day", d=pl.col(crps(a)) - pl.col(crps(b))).filter(
                        pl.col("d").is_finite()
                    )
                    st = s.block_stats(diff=d, rng=rng, tests=TESTS)
                    contrasts.append(
                        {
                            "set": name,
                            "slice": sname,
                            "band": band,
                            "contrast": f"{a} - {b}",
                            "counted": counted
                            and sname == "all rows"
                            and band != s.ALL
                            and name != "report",
                            "tests": TESTS,
                            "mde": mde,
                            **st,
                        }
                    )
    pl.DataFrame(absolute).write_csv(OUT / "absolute.csv")
    pl.DataFrame(contrasts).write_csv(OUT / "contrasts.csv")
    json.dump(
        {"rows main16": int(t["main16"].sum()), "bonferroni_count": TESTS},
        open(OUT / "checks.json", "w"),
    )
    print("rows", int(t["main16"].sum()), "tests", TESTS)


if __name__ == "__main__":
    main()
