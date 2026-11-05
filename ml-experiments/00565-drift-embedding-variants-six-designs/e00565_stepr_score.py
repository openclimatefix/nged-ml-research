# ruff: noqa: E501, PTH123, SIM115, PERF401
"""Experiment 565, step R: A10 (drift embedding) minus M0, five seeds, five folds of 511.

Each seed's stored forecast is scored on its own (widened by 1.15 as every stored row table is), and
the contrast of a row is the mean over the five seeds of A10's CRPS minus M0's CRPS of the same seed,
so the two sets of fits are paired by fold and seed. A10 seeds 0 and 1 are experiment 559's stored
fits, seeds 2 to 4 are this experiment's; M0 seeds 0 to 4 are experiment 554's.

Outputs under `data/ml-experiments/00565-*/work/scores/`: `stepr_absolute.csv`, `stepr_contrasts.csv`,
`stepr_seed_signs.csv`, `stepr_checks.json`.

Positive differences mean A10 has the larger (worse) CRPS.
"""

import json
import math
from typing import Final

import e00454_score as score454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00521_stats as s  # ty: ignore[unresolved-import]
import e00565_design6 as d6  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DD: Final = ML_EXPERIMENTS_DATA_DIR
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
FOLDERS: Final = {
    "M0": d6.W545.parent.parent
    / "00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out"
    / "work"
    / "seed_preds"
    / "M0",
    "A10": d6.W559 / "seed_preds" / "A10",
}
BANDS: Final[tuple[str, ...]] = ("24-48h", "48-72h", "72-120h")
TESTS: Final[int] = 8
"""2 groups x 3 lead bands (6) + 2 cold-night contrasts (the eight and the six, 24-120 h, 3 folds)."""
LEAVE_OUT: Final[str] = "D8"


def seed_crps(
    *, arm: str, month: str, seed: int, rows: pl.DataFrame, sites: pl.DataFrame
) -> np.ndarray:
    """Return one seed's widened per-row CRPS on the table's rows (NaN where the seed has no row)."""
    frame = build470.pool_stored(folder=FOLDERS[arm], month=month, sites=sites, seeds=(seed,))
    fr = rows.select(*d6.KEYS, "actual").join(
        frame.select(*d6.KEYS, *d6.LEVELS), on=d6.KEYS, how="left", maintain_order="left"
    )
    ok = fr["q50"].is_not_null().to_numpy()
    fr = fr.with_columns(*[pl.col(q).fill_null(0.0) for q in d6.LEVELS])
    fr = score470.widen_arm(arm=arm, frame=fr, factor=score470.HEADLINE)
    return np.where(ok, score470.row_metrics(frame=fr)["crps"], np.nan)


def build_table() -> tuple[pl.DataFrame, dict]:
    """Return rows with `<arm>_s<seed>` CRPS columns, and the reproduction checks."""
    sites = build470.site_map()
    checks: dict = {}
    frames = []
    for month in d6.MONTHS:
        rows = pl.read_parquet(
            d6.W559 / "scores" / f"rows_{month}.parquet",
            columns=[
                *d6.KEYS,
                "group",
                "cls",
                "lead_hours",
                "tgt_real",
                "cold",
                "main16",
                "A10__crps",
            ],
        )
        m0 = pl.read_parquet(
            d6.W545 / "scores" / f"rows_{month}.parquet", columns=[*d6.KEYS, "M0__crps"]
        )
        rows = rows.join(m0, on=d6.KEYS, how="left", maintain_order="left", validate="1:1")
        actual = score454.month_base(month=month).select(*d6.KEYS, "actual")
        rows = rows.join(actual, on=d6.KEYS, how="left", maintain_order="left", validate="1:1")
        cols = {}
        for arm in FOLDERS:
            for seed in SEEDS:
                cols[f"{arm}_s{seed}"] = seed_crps(
                    arm=arm, month=month, seed=seed, rows=rows, sites=sites
                )
            # Reproduce the stored two-seed pooled table from the seeds: pooled quantiles of seeds 0 and 1.
            pooled = build470.pool_stored(
                folder=FOLDERS[arm], month=month, sites=sites, seeds=(0, 1)
            )
            fr = rows.select(*d6.KEYS, "actual").join(
                pooled.select(*d6.KEYS, *d6.LEVELS), on=d6.KEYS, how="left", maintain_order="left"
            )
            fr = score470.widen_arm(arm=arm, frame=fr, factor=score470.HEADLINE)
            ours = score470.row_metrics(frame=fr)["crps"]
            stored = rows[f"{arm}__crps"].to_numpy()
            assert (np.isnan(ours) == np.isnan(stored)).all(), (arm, month)
            diff = np.abs(ours - stored)
            checks[f"{arm} {month} max |pooled(0,1) rescored - stored table|"] = float(
                np.nanmax(diff)
            )
            assert np.nanmax(diff) < 1e-4, (arm, month, np.nanmax(diff))
        frames.append(
            rows.hstack(
                pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float64) for k, v in cols.items()})
            )
            .with_columns(
                month=pl.lit(month), day=(pl.col("origin") // 48).cast(pl.Int32), band=s.band_expr()
            )
            .drop("actual", "A10__crps", "M0__crps")
        )
    return pl.concat(frames), checks


def mde(*, sub: pl.DataFrame) -> float:
    """Return the minimum detectable effect of the five-seed mean contrast: 1.96 x seed-to-seed sd / sqrt(5)."""
    cols = [f"d_s{x}" for x in SEEDS]
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


def main() -> None:
    """Score and write the tables."""
    t, checks = build_table()
    sites = pl.read_csv(s.DATA / "site_map.csv")
    label = dict(zip(sites["primary"].to_list(), sites["label"].to_list(), strict=True))
    t = t.with_columns(
        *[(pl.col(f"A10_s{x}") - pl.col(f"M0_s{x}")).alias(f"d_s{x}") for x in SEEDS],
    ).with_columns(
        d=pl.mean_horizontal([pl.col(f"d_s{x}") for x in SEEDS]),
        finite=pl.all_horizontal([pl.col(f"{a}_s{x}").is_finite() for a in FOLDERS for x in SEEDS]),
    )
    t = t.with_columns(main16=pl.col("main16") & pl.col("finite"))
    t = t.with_columns(label=pl.col("primary").replace_strict(label, default="?"))
    checks["rows"] = t.height
    checks["rows main16 finite in all seeds"] = int(t["main16"].sum())
    rng = np.random.default_rng(0)
    absolute, contrasts, signs = [], [], []
    for name in ("eight", "six", "report"):
        base_part = s.row_set(table=t, name=name)
        for kind in ("all", "leave_D8_out", "cold"):
            part = (
                s.row_set(table=t, name="six_all")
                if (kind == "cold" and name == "six")
                else base_part
            )
            if kind == "leave_D8_out":
                if name != "eight":
                    continue
                p = part.filter(pl.col("label") != LEAVE_OUT)
            elif kind == "cold":
                p = part.filter(pl.col("cold"))
            else:
                p = part
            for band in (*BANDS, "24-120h"):
                sub = (
                    p.filter(pl.col("band").is_in(BANDS))
                    if band == "24-120h"
                    else s.in_band(part=p, band=band)
                )
                if sub.height == 0 or (kind == "cold" and band != "24-120h"):
                    continue
                for arm in FOLDERS:
                    absolute.append(
                        {
                            "set": name,
                            "kind": kind,
                            "band": band,
                            "arm": arm,
                            "n": sub.height,
                            "mean_single_seed_crps": float(
                                np.mean([sub[f"{arm}_s{x}"].mean() for x in SEEDS])
                            ),
                        }
                    )
                st = s.block_stats(diff=sub.select("month", "day", "d"), rng=rng, tests=TESTS)
                counted = (
                    kind in ("all", "cold")
                    and name != "report"
                    and (band != "24-120h" or kind == "cold")
                )
                contrasts.append(
                    {
                        "set": name,
                        "kind": kind,
                        "band": band,
                        "contrast": "A10 - M0",
                        "counted": counted,
                        "tests": TESTS,
                        "mde": mde(sub=sub),
                        **st,
                    }
                )
                signs.append(
                    {
                        "set": name,
                        "kind": kind,
                        "band": band,
                        **{f"seed{x}": float(sub[f"d_s{x}"].mean()) for x in SEEDS},
                    }
                )
    out = d6.W / "scores"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(absolute).write_csv(out / "stepr_absolute.csv")
    pl.DataFrame(contrasts).write_csv(out / "stepr_contrasts.csv")
    pl.DataFrame(signs).write_csv(out / "stepr_seed_signs.csv")
    json.dump(checks, open(out / "stepr_checks.json", "w"), indent=1)
    with pl.Config(tbl_rows=60, tbl_cols=12):
        print(
            pl.DataFrame(contrasts)
            .select(
                "set",
                "kind",
                "band",
                "counted",
                "difference",
                "low_bonferroni_month",
                "high_bonferroni_month",
                "mde",
                "folds_positive",
            )
            .with_columns(pl.col(pl.Float64).round(3))
        )
        print(pl.DataFrame(signs).with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
