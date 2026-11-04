"""Score F1, F2, F3 (543) against L-1d, A-pool, R-DW (539), R1 and R9 (538) on 533's rows. Reuses 533's loader and 521's statistics."""
import json, math, sys
from pathlib import Path
import numpy as np
import polars as pl

sys.path[:0] = [str(p) for p in Path("/home/jack/dev/nged-ml-research/ml-experiments").iterdir() if p.is_dir()]
import e00521_stats as s

DD = Path("/home/jack/dev/nged-substation-forecast/data/ml-experiments")
W = DD / "00543-lag-free-network-auxiliary-recent-level-and-reverse-flow-heads"
E524 = DD / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator/work/scores"
K = ["primary", "origin", "target"]
NEWARMS = ("F1", "F2", "F3")
ARMS = {**{n: f"{n}__crps" for n in NEWARMS}, "R-DW": "R-DW__crps", "R1": "R1__crps", "R9": "R9__crps", "A-pool": "A-pool__crps", "L-1d": "L-1d__crps",
        "L-Ww": "L-Ww__crps", "K2-S1": "K2-S1__crps", "manual_heuristic": "manual_heuristic__crps"}
CONTRASTS = [("F1", "A-pool", True), ("F1", "L-1d", True), ("F2", "L-1d", True), ("F3", "L-1d", True),
             ("F2", "A-pool", False), ("F3", "A-pool", False), ("F3", "F1", False), ("F1", "R-DW", False), ("F3", "R-DW", False),
             ("F1", "R1", False), ("F1", "manual_heuristic", False), ("R-DW", "A-pool", False), ("L-1d", "A-pool", False)]
POOLED = "24-72h"
TESTS = 4 * 7 * 2  # 4 counted contrasts x 7 bands x 2 sets


def table():
    base = s.load()
    extra = pl.concat([pl.read_parquet(E524 / f"rows_{m}.parquet", columns=[*K, "A-pool__crps", "manual_heuristic__crps", "S2-J-0.4x__crps"]) for m in s.MONTHS])
    mine = pl.concat([pl.read_parquet(W / "work/scores" / f"rows_{m}.parquet", columns=[*K, *[f"{n}__crps" for n in NEWARMS]]) for m in s.MONTHS])
    for folder, cols in (("00539-l1d-plus-day-and-week-lags", ["R-DW__crps"]), ("00538-l1d-plus-recent-level-and-lags", ["R1__crps", "R9__crps"])):
        old = pl.concat([pl.read_parquet(DD / folder / "work/scores" / f"rows_{m}.parquet", columns=[*K, *cols]) for m in s.MONTHS])
        mine = mine.join(old, on=K, how="left", validate="1:1", maintain_order="left")
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
    chk = {"rows": t.height, "rows main16 with all arms finite": int(t["main16"].sum()), "columns used": need}
    json.dump(chk, open(W / "scores/checks.json", "w"), indent=1); print(chk)
    rng = np.random.default_rng(0)
    absolute, contrasts = [], []
    seedcols = ["L-1d-seed0__crps", "L-1d-seed1__crps"]
    for name in ("eight", "six", "report"):
        part = s.row_set(table=t, name=name)
        for band in (*s.BANDS, POOLED, s.ALL):
            sub = part.filter((pl.col('lead_hours') >= 24) & (pl.col('lead_hours') < 72)) if band == POOLED else s.in_band(part=part, band=band)
            if sub.height == 0:
                continue
            for arm, col in ARMS.items():
                if col in sub.columns:
                    absolute.append({"set": name, "band": band, "arm": arm, "n": sub.height, "crps": float(sub.filter(pl.col(col).is_finite())[col].mean())})
            mm = sub.group_by("month").agg([pl.col(c).mean() for c in seedcols]).sort("month").select(seedcols).to_numpy()
            f, k = mm.shape; r = mm - mm.mean(axis=1, keepdims=True)
            mde = 1.96 * math.sqrt(2) * math.sqrt(float((r**2).sum()) / (f * (k - 1))) / math.sqrt(k)
            for a, b, counted in CONTRASTS:
                if ARMS[b] not in sub.columns:
                    continue
                d = sub.select("month", "day", d=pl.col(ARMS[a]) - pl.col(ARMS[b])).filter(pl.col("d").is_finite())
                st = s.block_stats(diff=d, rng=rng, tests=TESTS)
                contrasts.append({"set": name, "band": band, "contrast": f"{a} - {b}", "counted": counted and band not in (s.ALL, POOLED) and name != "report", "tests": TESTS, "mde": mde, **st})
    pl.DataFrame(absolute).write_csv(W / "scores/absolute.csv")
    pl.DataFrame(contrasts).write_csv(W / "scores/contrasts.csv")

main()
