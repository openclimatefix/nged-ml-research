# ruff: noqa: ANN001, ANN201, D103, E501, PLW2901
"""Score the attention-mixer arms T0 to T5 (568), copied from experiment 566's scorer.

The references are M0 (545), L-1d, A-pool, R-DW (539), and, when scored, R-DW-24 and A-pool-24 (558).

Reuses experiment 521's rows and block bootstrap. The counted contrasts are listed in `COUNTED`; the
Bonferroni count is `TESTS`. Every scoring variant (`-removed`, `-aged1` to `-aged3`) is a column.
"""

import math

import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DD = ML_EXPERIMENTS_DATA_DIR
W = DD / "00568-attention-mixer-modular-network-with-lead-and-spread-token"
W545 = DD / "00545-modular-lag-free-demand-network-weather-and-datetime-encoders"
W539 = DD / "00539-l1d-plus-day-and-week-lags"
W560 = DD / "00560-modular-lag-encoder-capped-lags-and-block-dropout"
E524 = DD / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator/work/scores"
K = ["primary", "origin", "target"]
BAND_24_120 = "24-120h"
MODES = ("T0", "T0z", "T1", "T2", "T3", "T4", "T5", "T0-daily", "T4-daily")
TESTS = 36
"""Counted contrasts: C1 - C0, C2 - C1, C3 - C1, and C4 - C1, each at 24-120 h at the eight and at the six."""
SUFFIXES = ("",)


def own_arms() -> list[str]:
    path = W / "work/scores" / f"rows_{s.MONTHS[0]}.parquet"
    if not path.exists():
        return []
    have = pl.read_parquet_schema(path)
    return [f"{m}{x}" for m in MODES for x in SUFFIXES if f"{m}{x}__crps" in have]


def optional(folder, names):
    path = folder / "work/scores" / f"rows_{s.MONTHS[0]}.parquet"
    if not path.exists():
        return []
    have = pl.read_parquet_schema(path)
    return [n for n in names if f"{n}__crps" in have]


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
    assert base.select(K).equals(extra.select(K))
    drop = [c for c in ("A-pool__crps", "manual_heuristic__crps") if c in base.columns]
    t = pl.concat([base.drop(drop), extra.drop(K)], how="horizontal")
    for folder, names in (
        (W545, ["M0"]),
        (W539, ["R-DW"]),
        (W, own_arms()),
    ):
        names = optional(folder, names) if folder != W else names
        if not names:
            continue
        cols = [f"{a}__crps" for a in names]
        add = pl.concat(
            [
                pl.read_parquet(folder / "work/scores" / f"rows_{m}.parquet", columns=[*K, *cols])
                for m in s.MONTHS
            ]
        )
        n0 = t.height
        t = t.join(add, on=K, how="left", validate="1:1", maintain_order="left")
        assert t.height == n0
    return t


def main():
    t = table().with_columns(band=s.band_expr())
    mine = own_arms()
    primaries = [f"{a}__crps" for a in ("M0", "L-1d", *mine) if f"{a}__crps" in t.columns]
    fin = pl.all_horizontal([pl.col(c).is_finite() for c in primaries])
    t = t.with_columns(main16=pl.col("main16") & fin)
    print({"rows": t.height, "main16 rows with all arms finite": int(t["main16"].sum())})
    sites = pl.read_csv(s.DATA / "site_map.csv")
    d8 = int(sites.filter(pl.col("label") == "D8")["primary"][0])
    rng = np.random.default_rng(0)
    refs = [a for a in ("M0", "L-1d", "A-pool", "manual_heuristic") if f"{a}__crps" in t.columns]
    arms = [*refs, *mine]
    contrasts_spec = [("T1", "T0", True), ("T1", "T2", True), ("T1", "T3", True),
                      ("T4", "T1", True), ("T5", "T2", True)]  # fmt: skip
    contrasts_spec += [("T1", "T0z", False), ("T2", "T0z", False), ("T0z", "T0", False),
                       ("T3", "T2", False), ("T4", "T0", False), ("T5", "T1", False),
                       ("T0", "M0", False), ("T1", "L-1d", False),
                       ("T4-daily", "T0-daily", False), ("T0-daily", "T0", False), ("T4-daily", "T4", False),
                       ("T0-daily", "M0", False), ("T4-daily", "L-1d", False)]  # fmt: skip
    unique: dict = {}
    for a, b, counted in contrasts_spec:
        unique[(a, b)] = unique.get((a, b), False) or counted
    contrasts_spec = [(a, b, counted) for (a, b), counted in unique.items()]
    absolute, contrasts = [], []
    for name in ("eight", "eight_no_d8", "six", "report"):
        part = s.row_set(table=t, name="eight" if name == "eight_no_d8" else name)
        if name == "eight_no_d8":
            part = part.filter(pl.col("primary") != d8)
        bands = [*s.BANDS, BAND_24_120, s.ALL]
        for band in bands:
            if band == BAND_24_120:
                sub = part.filter(pl.col("lead_hours").is_between(24, 120, closed="left"))
            else:
                sub = s.in_band(part=part, band=band)
            if sub.height == 0:
                continue
            for arm in arms:
                col = f"{arm}__crps"
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
            seedcols = ["L-1d-seed0__crps", "L-1d-seed1__crps"]
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
            for a, b, counted in contrasts_spec:
                ca, cb = f"{a}__crps", f"{b}__crps"
                if ca not in sub.columns or cb not in sub.columns:
                    continue
                dd = sub.select("month", "day", d=pl.col(ca) - pl.col(cb)).filter(
                    pl.col("d").is_finite()
                )
                st = s.block_stats(diff=dd, rng=rng, tests=TESTS)
                contrasts.append(
                    {
                        "set": name,
                        "band": band,
                        "contrast": f"{a} - {b}",
                        "counted": counted and band == BAND_24_120 and name in {"eight", "six"},
                        "tests": TESTS,
                        "mde": mde,
                        **st,
                    }
                )
    (W / "scores").mkdir(exist_ok=True)
    pl.DataFrame(absolute).write_csv(W / "scores/absolute.csv")
    pl.DataFrame(contrasts).write_csv(W / "scores/contrasts.csv")


main()
