# ruff: noqa: ANN001, ANN201, D103, PTH123, SIM115, E501, PLW2901
"""Score the lag-encoder arms (560) against M0 (545), L-1d, A-pool, R-DW (539), and, when scored, R-DW-24 and A-pool-24 (558).

Reuses experiment 521's rows and block bootstrap. The counted contrasts are listed in `COUNTED`; the
Bonferroni count is `TESTS`. Every scoring variant (`-removed`, `-aged1` to `-aged3`) is a column.
"""

import json
import math

import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DD = ML_EXPERIMENTS_DATA_DIR
W = DD / "00560-modular-lag-encoder-capped-lags-and-block-dropout"
W545 = DD / "00545-modular-lag-free-demand-network-weather-and-datetime-encoders"
W539 = DD / "00539-l1d-plus-day-and-week-lags"
W558 = DD / "00558-lags-in-pretraining-and-24-hour-cap-lag-light-demand-network"
E524 = DD / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator/work/scores"
K = ["primary", "origin", "target"]
BAND_24_120 = "24-120h"
MODES = ("v1", "v2", "v3", "v4", "b_shared", "b_unshared")
TESTS = 4
"""Counted contrasts: v3 - v1 (1), v4 - M0 (1), best - A-pool-24 and best - R-DW-24 (2), each at 24-120 h at the eight."""
SUFFIXES = ("", "-removed", "-aged1", "-aged2", "-aged3")


def own_arms() -> list[str]:
    path = W / "work/scores" / f"rows_{s.MONTHS[0]}.parquet"
    if not path.exists():
        return []
    have = pl.read_parquet_schema(path)
    return [f"LAG-{m}{x}" for m in MODES for x in SUFFIXES if f"LAG-{m}{x}__crps" in have]


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
        (W558, ["R-DW-24", "A-pool-24"]),
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


def best_arm(t, sub_eight):
    """Return the stage-A or stage-B arm with the lowest mean CRPS at 24-120 h at the eight (named in the page)."""
    cand = [a for a in own_arms() if a.count("-") == 1 and not a.startswith("LAG-v4")]
    sub = sub_eight.filter(pl.col("lead_hours").is_between(24, 120, closed="left"))
    means = {
        a: float(sub.filter(pl.col(f"{a}__crps").is_finite())[f"{a}__crps"].mean()) for a in cand
    }
    return min(means, key=means.get), means


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
    best, means = best_arm(t, s.row_set(table=t, name="eight"))
    json.dump(
        {"best_stage_a_or_b_arm": best, "mean_crps_24_120_eight": means},
        open(W / "scores/best_arm.json", "w"),
        indent=1,
    )
    refs = [
        a
        for a in ("M0", "L-1d", "A-pool", "R-DW", "R-DW-24", "A-pool-24", "manual_heuristic")
        if f"{a}__crps" in t.columns
    ]
    arms = [*refs, *mine]
    contrasts_spec = [
        ("LAG-v3", "LAG-v1", True),
        ("LAG-v4", "M0", True),
        (best, "A-pool-24", True),
        (best, "R-DW-24", True),
    ]
    contrasts_spec += [(a, "M0", False) for a in mine if not a.count("-") > 1 and a != "LAG-v4"]
    contrasts_spec += [(a, "L-1d", False) for a in mine if a.count("-") == 1]
    contrasts_spec += [(f"LAG-{m}{x}", f"LAG-{m}", False) for m in MODES for x in SUFFIXES[1:]]
    contrasts_spec += [
        (f"LAG-{a}", f"LAG-{b}", False) for i, a in enumerate(MODES) for b in MODES[:i]
    ]
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
                        "counted": counted and band == BAND_24_120 and name == "eight",
                        "tests": TESTS,
                        "mde": mde,
                        **st,
                    }
                )
    (W / "scores").mkdir(exist_ok=True)
    pl.DataFrame(absolute).write_csv(W / "scores/absolute.csv")
    pl.DataFrame(contrasts).write_csv(W / "scores/contrasts.csv")


main()
