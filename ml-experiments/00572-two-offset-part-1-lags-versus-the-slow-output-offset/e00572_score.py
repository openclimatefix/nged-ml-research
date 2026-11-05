# ruff: noqa: E501, PTH123, SIM115, PT018, PERF401
"""Experiment 572: score the Part 1 arms in three lead bands, with slices, contrasts, and the injected-shift leak.

Arms: M0 and E2 are experiment 565's stored `C-M0` and `D5` (coarse refits, seeds 0 and 1); E2ft, V3,
V3D5, and V35 are this experiment's fits. Each seed of each arm is scored on its own (widened by 1.15,
as every stored row table is), and a contrast is the mean over the two seeds of the paired CRPS
difference, so arms are paired by fold, seed, and row. Positive differences mean the first arm is
worse.

Lead bands (hours): 0-24, 24-120, and 120-240. The Bonferroni count is 24: 4 counted contrasts (E2
minus M0, E2 minus V3, E2 minus V3D5, V3D5 minus V3) x 2 groups x 3 lead bands, all on all rows of the
group. Every other number is descriptive.

Outputs under `data/ml-experiments/00572-*/work/scores/`: `part1_absolute.csv`, `part1_contrasts.csv`,
`part1_seed_signs.csv`, `part1_leak.csv`, `part1_checks.json`.
"""

import json
import math
from pathlib import Path
from typing import Final

import e00454_score as score454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00521_stats as s  # ty: ignore[unresolved-import]
import e00565_design6 as d6  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W572: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00572-two-offset-part-1-lags-versus-the-slow-output-offset/work"
)
SEEDS: Final[tuple[int, ...]] = (0, 1)
TESTS: Final[int] = 24
BANDS: Final[dict[str, tuple[float, float]]] = {
    "0-24h": (0.0, 24.0),
    "24-120h": (24.0, 120.0),
    "120-240h": (120.0, 240.0),
}
ARM_DIRS: Final[dict[str, tuple[Path, str]]] = {
    "M0": (d6.W, "C-M0"),
    "E2": (d6.W, "D5"),
    "E2ft": (W572, "E2ft"),
    "V3": (W572, "V3"),
    "V3D5": (W572, "V3D5"),
    "V35": (W572, "V35"),
}
SIBLING_SUFFIXES: Final[tuple[str, ...]] = ("removed", "aged1", "aged2", "aged3")
COUNTED: Final[tuple[tuple[str, str], ...]] = (
    ("E2", "M0"),
    ("E2", "V3"),
    ("E2", "V3D5"),
    ("V3D5", "V3"),
)
DESCRIPTIVE: Final[tuple[tuple[str, str], ...]] = (
    ("V3D5", "E2"),
    ("E2", "E2ft"),
    ("V35", "E2"),
    ("V35", "V3"),
    ("V3", "M0"),
    ("E2ft", "M0"),
    ("V3D5", "M0"),
)
KEYS: Final = ["time_series_id", "power_fcst_init_time", "valid_time"]
D8: Final[int] = 4
D14: Final[int] = 10
"""Primary indices of D8 and D14 in experiment 521's site map (asserted in `build_table`)."""


def arms_present() -> dict[str, tuple[Path, str]]:
    """Return the arms (including lag-sibling twins) whose assembled forecasts exist for every fold and seed."""
    out = {}
    for name, (root, tag) in ARM_DIRS.items():
        variants = [(name, tag)]
        if name in {"V3", "V3D5", "V35"}:
            variants += [(f"{name}-{x}", f"{tag}-{x}") for x in SIBLING_SUFFIXES]
        for label, folder in variants:
            if all(
                (root / "seed_preds" / folder / f"{m}_s{x}.parquet").exists()
                for m in d6.MONTHS
                for x in SEEDS
            ):
                out[label] = (root, folder)
    return out


def band_of(lead: pl.Expr) -> pl.Expr:
    """Return the three-band label of a lead in hours."""
    return (
        pl.when(lead < 24.0)
        .then(pl.lit("0-24h"))
        .when(lead < 120.0)
        .then(pl.lit("24-120h"))
        .otherwise(pl.lit("120-240h"))
    )


def build_table(*, arms: dict[str, tuple[Path, str]]) -> pl.DataFrame:
    """Return rows with `<arm>_s<seed>` widened CRPS columns."""
    sites = build470.site_map()
    frames = []
    for month in d6.MONTHS:
        rows = pl.read_parquet(
            d6.W559 / "scores" / f"rows_{month}.parquet",
            columns=[*d6.KEYS, "group", "cls", "lead_hours", "tgt_real", "cold", "xmas", "main16"],
        )
        actual = score454.month_base(month=month).select(*d6.KEYS, "actual")
        rows = rows.join(actual, on=d6.KEYS, how="left", maintain_order="left", validate="1:1")
        cols = {}
        for label, (root, folder) in arms.items():
            for seed in SEEDS:
                frame = build470.pool_stored(
                    folder=root / "seed_preds" / folder, month=month, sites=sites, seeds=(seed,)
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
                cols[f"{label}_s{seed}"] = np.where(
                    ok, score470.row_metrics(frame=fr)["crps"], np.nan
                )
        frames.append(
            rows.hstack(
                pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float64) for k, v in cols.items()})
            )
            .with_columns(
                month=pl.lit(month),
                day=(pl.col("origin") // 48).cast(pl.Int32),
                band=band_of(pl.col("lead_hours")),
            )
            .drop("actual")
        )
    t = pl.concat(frames)
    main_arms = [a for a in arms if "-" not in a]
    fin = pl.all_horizontal([pl.col(f"{a}_s{x}").is_finite() for a in main_arms for x in SEEDS])
    labels = pl.read_csv(
        d6.W559.parent.parent
        / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
    )
    lab = dict(zip(labels["primary"].to_list(), labels["label"].to_list(), strict=True))
    assert lab[D8] == "D8" and lab[D14] == "D14", lab
    return t.with_columns(main16=pl.col("main16") & fin)


def mde(*, sub: pl.DataFrame, a: str, b: str) -> float:
    """Return the seed-noise half-width of a paired contrast (1.96 x the seed sd, pooled within folds, over sqrt(seeds))."""
    cols = [f"d_{a}_{b}_s{x}" for x in SEEDS]
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


SLICES: Final[dict[str, pl.Expr]] = {
    "all rows": pl.lit(True),
    "no labelled event": ~pl.col("tgt_real"),
    "cold winter nights": pl.col("cold"),
    "Christmas week": pl.col("xmas"),
    "D8 alone": pl.col("primary") == D8,
    "D14 alone": pl.col("primary") == D14,
    "leave D8 out": pl.col("primary") != D8,
}


def leak(*, arms: list[str]) -> pl.DataFrame:
    """Return the injected-shift leak of each arm (mean over the five folds, with min and max), in points of capacity."""
    rows = []
    for arm in arms:
        root, folder = ARM_DIRS[arm]
        for month in d6.MONTHS:
            ctrl_path = root / "fits" / folder / f"{month}_d14_s0.parquet"
            stress = [
                root / "fits" / folder / f"{month}_d14_s0_stress{t}.parquet" for t in ("p", "n")
            ]
            if not (ctrl_path.exists() and all(p.exists() for p in stress)):
                break
            ctrl = pl.read_parquet(ctrl_path).select(*KEYS, c="q50")
            for sign, p in zip((1, -1), stress, strict=True):
                j = ctrl.join(pl.read_parquet(p).select(*KEYS, s="q50"), on=KEYS)
                assert j.height == ctrl.height, (arm, month)
                rows.append(
                    {
                        "arm": arm,
                        "month": month,
                        "sign": sign,
                        "leak": float((sign * 100.0 * (j["s"] - j["c"])).mean()),  # ty: ignore[invalid-argument-type]
                    }
                )  # ty: ignore[invalid-argument-type]
    if not rows:
        return pl.DataFrame()
    fold = pl.DataFrame(rows).group_by("arm", "month").agg(pl.col("leak").mean())
    return (
        fold.group_by("arm")
        .agg(
            leak=pl.col("leak").mean(),
            lo=pl.col("leak").min(),
            hi=pl.col("leak").max(),
            folds=pl.len(),
        )
        .sort("arm")
    )


def main() -> None:
    """Score and write the tables."""
    arms = arms_present()
    present = [a for a in arms if "-" not in a]
    print("arms present:", list(arms))
    assert {"M0", "E2"} <= set(present), present
    t = build_table(arms=arms)
    out = W572 / "scores"
    out.mkdir(parents=True, exist_ok=True)
    pairs = [(a, b) for a, b in (*COUNTED, *DESCRIPTIVE) if a in present and b in present]
    t = t.with_columns(
        *[
            (pl.col(f"{a}_s{x}") - pl.col(f"{b}_s{x}")).alias(f"d_{a}_{b}_s{x}")
            for a, b in pairs
            for x in SEEDS
        ]
    )
    absolute, contrasts, signs = [], [], []
    for name in ("eight", "six", "report"):
        part = s.row_set(table=t, name=name)
        for slice_name, expr in SLICES.items():
            if slice_name == "leave D8 out" and name != "eight":
                continue
            sl = part.filter(expr.fill_null(value=False))
            for band in (*BANDS, "all"):
                sub = sl if band == "all" else sl.filter(pl.col("band") == band)
                if sub.height < 200:
                    continue
                for arm in arms:
                    absolute.append({"set": name, "slice": slice_name, "band": band, "arm": arm, "n": sub.height,
                                     "mean_single_seed_crps": float(np.mean([sub[f"{arm}_s{x}"].mean() for x in SEEDS]))})  # fmt: skip
                for a, b in pairs:
                    sub2 = sub.with_columns(
                        d=pl.mean_horizontal([pl.col(f"d_{a}_{b}_s{x}") for x in SEEDS])
                    )
                    counted = (
                        (a, b) in COUNTED
                        and slice_name == "all rows"
                        and band != "all"
                        and name != "report"
                    )
                    st = s.block_stats(
                        diff=sub2.select("month", "day", "d"),
                        rng=np.random.default_rng(0),
                        tests=TESTS if counted else 1,
                    )
                    contrasts.append({"set": name, "slice": slice_name, "band": band, "a": a, "b": b, "counted": counted,
                                      "tests": TESTS if counted else 1, "mde": mde(sub=sub2, a=a, b=b), **st})  # fmt: skip
                    signs.append({"set": name, "slice": slice_name, "band": band, "a": a, "b": b,
                                  **{f"seed{x}": float(sub2[f"d_{a}_{b}_s{x}"].mean()) for x in SEEDS}})  # fmt: skip
    pl.DataFrame(absolute).write_csv(out / "part1_absolute.csv")
    pl.DataFrame(contrasts).write_csv(out / "part1_contrasts.csv")
    pl.DataFrame(signs).write_csv(out / "part1_seed_signs.csv")
    lk = leak(arms=present)
    lk.write_csv(out / "part1_leak.csv")
    json.dump(
        {"arms": list(arms), "rows": t.height, "main16": int(t["main16"].sum())},
        open(out / "part1_checks.json", "w"),
        indent=1,
    )
    c = pl.DataFrame(contrasts)
    with pl.Config(tbl_rows=120, tbl_cols=14, tbl_width_chars=200):
        print(
            c.filter(
                (pl.col("slice") == "all rows")
                & (pl.col("set") != "report")
                & (pl.col("band") != "all")
            )
            .select(
                "set",
                "band",
                "a",
                "b",
                "counted",
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
