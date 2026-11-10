# ruff: noqa: ANN201, D103, E402, F841, PTH123, SIM115
"""Experiment 533, `e00533_score.py`: kept for the record.

Score L-1d (511, 2 seeds; 527, 4 seeds) against A-pool on the same stored rows. No fit, no GPU.

Reuses experiment 521's row sets, lead bands, bootstrap and MDE formula (imported from the
nged-ml-research checkout) and 511's, 521's, 524's, and 527's stored per-row scores.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

sys.path[:0] = [str(p) for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]
import e00521_stats as s  # ty: ignore[unresolved-import]

D = Path(str(ML_EXPERIMENTS_DATA_DIR))
OUT = D / "00533-l1d-vs-a-pool-same-rows" / "scores"
E524 = D / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator/work/scores"
E527 = D / "00527-lag-gap-head-count-test-k2-s1-vs-four-seed-daily-lag-free/work"
K = ["primary", "origin", "target"]
ARMS = {
    "A-pool": "A-pool__crps",
    "L-1d": "L-1d__crps",
    "L-1d-4q": "L-1d-4q__crps",
    "L-Ww": "L-Ww__crps",
    "S2-J-0.4x": "S2-J-0.4x__crps",
    "K2-S1": "K2-S1__crps",
    "manual_heuristic": "manual_heuristic__crps",
}
CONTRASTS = [
    ("L-1d", "A-pool", True),
    ("L-1d-4q", "A-pool", True),
    ("L-1d", "K2-S1", False),
    ("L-1d-4q", "K2-S1", False),
    ("A-pool", "K2-S1", False),
    ("L-1d", "S2-J-0.4x", False),
    ("L-1d", "L-Ww", False),
    ("L-1d", "manual_heuristic", False),
]
TESTS = 2 * 7 * 2
SLICES = ("all rows", "no labelled event", "cold nights", "Christmas week")


def load():
    base = s.load()  # 511 rows + 521 refs (L-1d seeds 0, 1)
    out = []
    for m in s.MONTHS:
        a = pl.read_parquet(
            E524 / f"rows_{m}.parquet",
            columns=[*K, "S2-J-0.4x__crps", "A-pool__crps", "manual_heuristic__crps"],
        )
        b = pl.read_parquet(
            E527 / f"seeds_{m}.parquet",
            columns=[*K, "L-1d-seed2__crps", "L-1d-seed3__crps", "L-1d-4q__crps"],
        )
        e = pl.read_parquet(
            D
            / "00511-lagless-network-refit-cadence-half-life-and-mixed-pretraining/work/scores"
            / f"rows_{m}.parquet",
            columns=[*K, "A-pool__crps", "manual_heuristic__crps", "xmas", "cold"],
        ).rename(
            {
                "A-pool__crps": "A511",
                "manual_heuristic__crps": "M511",
                "xmas": "xmas_",
                "cold": "cold_",
            }
        )
        out.append(pl.concat([a, b.drop(K), e.drop(K)], how="horizontal"))
    extra = pl.concat(out)
    t = (
        pl.concat(
            [
                base.drop(["A-pool__crps"], strict=False)
                if "A-pool__crps" in base.columns
                else base,
                extra.drop(["manual_heuristic__crps"]),
            ],
            how="horizontal",
        )
        if False
        else None
    )
    return base, extra


def main():
    base, extra = load()
    assert base.select(K).equals(extra.select(K)) if all(c in extra.columns for c in K) else True
    t = pl.concat([base, extra.drop(K, strict=False)], how="horizontal")
    t = t.with_columns(band=s.band_expr())
    # checks: A-pool and manual_heuristic agree between 511 and 524 row tables
    chk = {
        "A-pool 511 vs 524 max abs diff": float((t["A-pool__crps"] - t["A511"]).abs().max()),
        "manual 511 vs 524 max abs diff": float(
            (t["manual_heuristic__crps"] - t["M511"]).abs().max()
        ),
        "rows": t.height,
    }
    fin = pl.all_horizontal(
        [pl.col(c).is_finite() for k2, c in ARMS.items() if k2 != "manual_heuristic"]
    )
    t = t.with_columns(main16=pl.col("main16") & fin)
    chk["rows with all arms finite and main16"] = int(t["main16"].sum())
    chk["rows main16 in 511 only"] = int(base["main16"].sum())
    json.dump(chk, open(OUT / "checks.json", "w"), indent=1)
    print(chk)
    sl = {
        "all rows": pl.lit(True),
        "no labelled event": ~pl.col("tgt_real"),
        "cold nights": pl.col("cold").cast(pl.Boolean),
        "Christmas week": pl.col("xmas").cast(pl.Boolean),
    }
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
                for arm, col in ARMS.items():
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
                # MDE from seed spread
                mdes = {}
                for lab, cols in {
                    "L-1d": ["L-1d-seed0__crps", "L-1d-seed1__crps"],
                    "L-1d-4q": [
                        "L-1d-seed0__crps",
                        "L-1d-seed1__crps",
                        "L-1d-seed2__crps",
                        "L-1d-seed3__crps",
                    ],
                }.items():
                    mm = (
                        sub.group_by("month")
                        .agg([pl.col(c).mean() for c in cols])
                        .sort("month")
                        .select(cols)
                        .to_numpy()
                    )
                    f, k = mm.shape
                    r = mm - mm.mean(axis=1, keepdims=True)
                    mdes[lab] = (
                        1.96
                        * math.sqrt(2)
                        * math.sqrt(float((r**2).sum()) / (f * (k - 1)))
                        / math.sqrt(k)
                    )
                for a, b, counted in CONTRASTS:
                    d = sub.select("month", "day", d=pl.col(ARMS[a]) - pl.col(ARMS[b])).filter(
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
                            "mde": mdes.get(a, float("nan")),
                            **st,
                        }
                    )
    pl.DataFrame(absolute).write_csv(OUT / "absolute.csv")
    pl.DataFrame(contrasts).write_csv(OUT / "contrasts.csv")


main()
