# ruff: noqa: E501, PTH123, SIM115, PERF401
"""Experiment 565, stages 2 to 4: score the coarse screening arms against C-M0 and C-A10.

Each seed of each arm is scored on its own (widened by 1.15, as every stored row table is). A row's
contrast is the mean over the two seeds of the arm's CRPS minus C-M0's CRPS of the same seed, so
arms are paired by fold and seed. The Bonferroni count is 36 (the whole experiment: 6 designs x 2
row sets x 3 lead bands of 24-48, 48-72, and 72-120 h). Positive differences mean the arm is worse.

The injected-shift leak is the mean of `sign x 100 x (median with the block injected - median of the
control)` over the forecast window of the day-14 fit (seed 0, both signs), in points of capacity.

Outputs under `data/ml-experiments/00565-*/work/scores/`: `screen_absolute.csv`,
`screen_contrasts.csv`, `screen_leak.csv`, `screen_seed_signs.csv`.
"""

import json
import math
import os
from typing import Final

import e00454_score as score454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00521_stats as s  # ty: ignore[unresolved-import]
import e00565_design6 as d6  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SEEDS: Final[tuple[int, ...]] = tuple(int(x) for x in os.environ.get("SCREEN_SEEDS", "0 1").split())
ARMS: Final[tuple[str, ...]] = tuple(
    os.environ.get("SCREEN_ARMS", "C-M0 C-A10 D3m D3q D4 D4-null D5").split()
)
PREFIX: Final[str] = os.environ.get("SCREEN_PREFIX", "screen")
"""`SCREEN_SEEDS`, `SCREEN_ARMS`, and `SCREEN_PREFIX` select a subset run, such as the five-seed D5 and C-M0 pair."""
BASE: Final[str] = "C-M0"
BANDS: Final[tuple[str, ...]] = ("24-48h", "48-72h", "72-120h")
TESTS: Final[int] = 36
KEYS: Final = ["time_series_id", "power_fcst_init_time", "valid_time"]


def arms_present() -> list[str]:
    """Return the arms whose assembled forecasts exist for every month and seed."""
    return [
        a
        for a in ARMS
        if all(
            (d6.W / "seed_preds" / a / f"{m}_s{x}.parquet").exists()
            for m in d6.MONTHS
            for x in SEEDS
        )
    ]


def build_table(*, arms: list[str]) -> pl.DataFrame:
    """Return rows with `<arm>_s<seed>` widened CRPS columns."""
    sites = build470.site_map()
    frames = []
    for month in d6.MONTHS:
        rows = pl.read_parquet(
            d6.W559 / "scores" / f"rows_{month}.parquet",
            columns=[*d6.KEYS, "group", "cls", "lead_hours", "tgt_real", "cold", "main16"],
        )
        actual = score454.month_base(month=month).select(*d6.KEYS, "actual")
        rows = rows.join(actual, on=d6.KEYS, how="left", maintain_order="left", validate="1:1")
        cols = {}
        for arm in arms:
            for seed in SEEDS:
                frame = build470.pool_stored(
                    folder=d6.W / "seed_preds" / arm, month=month, sites=sites, seeds=(seed,)
                )
                fr = rows.select(*d6.KEYS, "actual").join(
                    frame.select(*d6.KEYS, *d6.LEVELS),
                    on=d6.KEYS,
                    how="left",
                    maintain_order="left",
                )
                ok = fr["q50"].is_not_null().to_numpy()
                fr = fr.with_columns(*[pl.col(q).fill_null(0.0) for q in d6.LEVELS])
                fr = score470.widen_arm(arm="M0", frame=fr, factor=score470.HEADLINE)
                cols[f"{arm}_s{seed}"] = np.where(
                    ok, score470.row_metrics(frame=fr)["crps"], np.nan
                )
        frames.append(
            rows.hstack(
                pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float64) for k, v in cols.items()})
            )
            .with_columns(
                month=pl.lit(month), day=(pl.col("origin") // 48).cast(pl.Int32), band=s.band_expr()
            )
            .drop("actual")
        )
    t = pl.concat(frames)
    fin = pl.all_horizontal([pl.col(f"{a}_s{x}").is_finite() for a in arms for x in SEEDS])
    return t.with_columns(main16=pl.col("main16") & fin)


def mde(*, sub: pl.DataFrame, arm: str, ref: str) -> float:
    """Return the seed-noise half-width of a contrast.

    The half-width is 1.96 x the seed-to-seed sd of the paired contrast (pooled within folds) /
    sqrt(seeds).

    This is a half-width, not a power-based minimum detectable effect, and it uses the normal
    quantile with 5 degrees of freedom.
    """
    cols = [f"d_{arm}_{ref}_s{x}" for x in SEEDS]
    mm = (
        sub.group_by("month")
        .agg([pl.col(c).mean() for c in cols])
        .sort("month")
        .select(cols)
        .to_numpy()
    )
    f, k = mm.shape
    r = mm - mm.mean(axis=1, keepdims=True)
    return 1.96 * math.sqrt(float((r**2).sum()) / (f * (k - 1))) / math.sqrt(k)


def leak(*, arms: list[str]) -> pl.DataFrame:
    """Return the injected-shift leak of each arm (mean over the five folds, with min and max)."""
    rows = []
    for arm in arms:
        for month in d6.MONTHS:
            ctrl = pl.read_parquet(d6.W / "fits" / arm / f"{month}_d14_s0.parquet").select(
                *KEYS, c="q50"
            )
            for sign, tag in ((1, "p"), (-1, "n")):
                p = d6.W / "fits" / arm / f"{month}_d14_s0_stress{tag}.parquet"
                j = ctrl.join(pl.read_parquet(p).select(*KEYS, s="q50"), on=KEYS)
                assert j.height == ctrl.height, (arm, month, tag)
                rows.append(
                    {
                        "arm": arm,
                        "month": month,
                        "sign": sign,
                        "leak": float((sign * 100.0 * (j["s"] - j["c"])).mean()),  # ty: ignore[invalid-argument-type]
                    }
                )
    t = pl.DataFrame(rows)
    fold = t.group_by("arm", "month").agg(pl.col("leak").mean())
    return (
        fold.group_by("arm")
        .agg(leak=pl.col("leak").mean(), lo=pl.col("leak").min(), hi=pl.col("leak").max())
        .sort("arm")
    )


def main() -> None:
    """Score and write the tables."""
    arms = arms_present()
    assert BASE in arms, arms
    t = build_table(arms=arms)
    out = d6.W / "scores"
    out.mkdir(parents=True, exist_ok=True)
    others = [a for a in arms if a != BASE]
    refs = [BASE] + (["C-A10"] if "C-A10" in arms else [])
    t = t.with_columns(
        *[
            (pl.col(f"{a}_s{x}") - pl.col(f"{r}_s{x}")).alias(f"d_{a}_{r}_s{x}")
            for a in others
            for r in refs
            if a != r
            for x in SEEDS
        ]
    )
    absolute, contrasts, signs = [], [], []
    for name in ("eight", "six", "report"):
        part = s.row_set(table=t, name=name)
        for band in (*BANDS, "24-120h"):
            sub = (
                part.filter(pl.col("band").is_in(BANDS))
                if band == "24-120h"
                else s.in_band(part=part, band=band)
            )
            if sub.height == 0:
                continue
            for arm in arms:
                absolute.append(
                    {
                        "set": name,
                        "band": band,
                        "arm": arm,
                        "n": sub.height,
                        "mean_single_seed_crps": float(
                            np.mean([sub[f"{arm}_s{x}"].mean() for x in SEEDS])
                        ),
                    }
                )
            for arm in others:
                for ref in refs:
                    if arm == ref:
                        continue
                    sub2 = sub.with_columns(
                        d=pl.mean_horizontal([pl.col(f"d_{arm}_{ref}_s{x}") for x in SEEDS])
                    )
                    st = s.block_stats(
                        diff=sub2.select("month", "day", "d"),
                        rng=np.random.default_rng(0),
                        tests=TESTS,
                    )
                    contrasts.append(
                        {
                            "set": name,
                            "band": band,
                            "arm": arm,
                            "ref": ref,
                            "counted": ref == BASE and band != "24-120h" and name != "report",
                            "pass_rule": ref == BASE and band == "24-120h" and name != "report",
                            "tests": TESTS,
                            "mde": mde(sub=sub2, arm=arm, ref=ref),
                            **st,
                        }
                    )
                    signs.append(
                        {
                            "set": name,
                            "band": band,
                            "arm": arm,
                            "ref": ref,
                            **{
                                f"seed{x}": float(sub2[f"d_{arm}_{ref}_s{x}"].mean()) for x in SEEDS
                            },
                        }
                    )
    pl.DataFrame(absolute).write_csv(out / f"{PREFIX}_absolute.csv")
    pl.DataFrame(contrasts).write_csv(out / f"{PREFIX}_contrasts.csv")
    pl.DataFrame(signs).write_csv(out / f"{PREFIX}_seed_signs.csv")
    lk = leak(arms=arms) if os.environ.get("SCREEN_LEAK", "1") == "1" else pl.DataFrame()
    lk.write_csv(out / f"{PREFIX}_leak.csv")
    json.dump(
        {"arms": arms, "rows": t.height, "main16": int(t["main16"].sum())},
        open(out / f"{PREFIX}_checks.json", "w"),
        indent=1,
    )
    with pl.Config(tbl_rows=80, tbl_cols=14):
        c = pl.DataFrame(contrasts)
        print(
            c.filter((pl.col("ref") == BASE) & (pl.col("set") != "report"))
            .select(
                "set",
                "band",
                "arm",
                "difference",
                "low_bonferroni_month",
                "high_bonferroni_month",
                "mde",
                "folds_positive",
            )
            .with_columns(pl.col(pl.Float64).round(3))
        )
        print(lk.with_columns(pl.col(pl.Float64).round(2)))


if __name__ == "__main__":
    main()
