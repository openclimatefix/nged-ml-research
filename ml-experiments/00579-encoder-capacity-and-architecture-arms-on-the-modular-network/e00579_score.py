# ruff: noqa: B023, E501, FURB171, PLR1714, PT018, PTH123, SIM115
"""Experiment 579: score the arms against the base B, with slices, MDE, and the Stage 1 screen.

B is experiment 572's v3+D5 (stored forecasts in `seed_preds/V3D5`, seeds 0 and 1); M0 and E2 are
experiment 565's stored `C-M0` and `D5`. Each arm and seed is scored on its own (widened by 1.15, as
every stored row table is). A contrast is the mean over the arm's seeds of the paired difference to
B in the same seeds, so arms are paired by fold, seed, and row; a positive difference means the arm
is worse. Every Stage 1 arm has seed 0 only, so its contrast is arm seed 0 minus B seed 0.

Measures: CRPS and the summed pinball loss at the 5%, 95%, and 99% quantiles ("tail"), both in
points of effective capacity. Lead bands (hours): 0-24, 24-120, 120-240; the screen uses 24-120 h.
Slices: all rows, no labelled event, cold winter nights, Christmas week, evenings (17:00 to 20:59
local), sunny days (the top tercile of the ensemble-mean shortwave radiation among daylight
half-hours, per Primary; taken from the smallest-lead run's value at the target, the same for every
arm), D8 alone, D14 alone, and leave D8 out.

MDE: the seed noise of B, measured on its two stored seeds: sigma is the pooled within-fold standard
deviation of the per-fold mean of B's seed 0 and seed 1, `mde_2seed` = 1.96 sigma / sqrt(2) (the
half-width of a two-seed mean, as experiment 572 states it) and `mde_1seed` = 1.96 sqrt(2) sigma
(the half-width of a difference between two single-seed fits). The screen uses `mde_2seed`, as the
brief does, and reports the single-seed value beside it.

Outputs under `work/scores/`: `absolute.csv`, `contrasts.csv`, `mde.csv`, `screen.csv`,
`primaries.csv`, `seed_signs.csv`, `leak.csv`, `checks.json`.
"""

import json
import math
import sys
from pathlib import Path
from typing import Final

import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00454_score as score454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00521_stats as s  # ty: ignore[unresolved-import]
import e00565_design6 as d6  # ty: ignore[unresolved-import]
import e00579_arms as arms_mod  # ty: ignore[unresolved-import]
import e00579_columns as xcols  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W579: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00579-encoder-capacity-and-architecture-arms-on-the-modular-network/work"
)
W572: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00572-two-offset-part-1-lags-versus-the-slow-output-offset/work"
)
W560: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00560-modular-lag-encoder-capped-lags-and-block-dropout/work"
)
BASE: Final[str] = "B"
BANDS: Final[dict[str, tuple[float, float]]] = {
    "0-24h": (0.0, 24.0),
    "24-120h": (24.0, 120.0),
    "120-240h": (120.0, 240.0),
}
SCREEN_BAND: Final[str] = "24-120h"
REFS: Final[dict[str, tuple[Path, str]]] = {"M0": (d6.W, "C-M0"), "E2": (d6.W, "D5")}
SIBLING_SUFFIXES: Final[tuple[str, ...]] = ("removed", "aged1", "aged2", "aged3")
D8: Final[int] = 4
D14: Final[int] = 9
TAIL_LEVELS: Final[tuple[str, ...]] = ("q05", "q95", "q99")
MDE_QUOTED: Final[dict[str, float]] = {"eight": 0.113, "six": 0.128, "eight_no_d8": 0.092}
TAIL_GAIN: Final[float] = 0.02
HARM_FLOOR: Final[float] = 0.03
SLICE_NAMES: Final[tuple[str, ...]] = (
    "no labelled event",
    "cold winter nights",
    "Christmas week",
    "evenings",
    "sunny days",
)
SLICES: Final[dict[str, pl.Expr]] = {
    "all rows": pl.lit(True),
    "no labelled event": ~pl.col("tgt_real"),
    "cold winter nights": pl.col("cold"),
    "Christmas week": pl.col("xmas"),
    "evenings": pl.col("evening"),
    "sunny days": pl.col("sunny"),
    "D8 alone": pl.col("primary") == D8,
    "D14 alone": pl.col("primary") == D14,
    "leave D8 out": pl.col("primary") != D8,
}


def arm_paths() -> dict[str, tuple[Path, str]]:
    """Return (root, folder) of every arm with seed 0 forecasts for all five folds, with lag twins."""
    out: dict[str, tuple[Path, str]] = {BASE: (W572, "V3D5")}
    for tag in arms_mod.ARMS:
        if tag != BASE:
            out[tag] = (W579, tag)
    out |= REFS
    found = {}
    for name, (root, folder) in out.items():
        variants = [(name, folder)]
        if name != "M0" and name != "E2":
            variants += [(f"{name}-{x}", f"{folder}-{x}") for x in SIBLING_SUFFIXES]
        for label, fold in variants:
            if all((root / "seed_preds" / fold / f"{m}_s0.parquet").exists() for m in d6.MONTHS):
                found[label] = (root, fold)
    return found


def seeds_of(*, root: Path, folder: str) -> tuple[int, ...]:
    """Return the seeds whose forecasts exist for every fold."""
    return tuple(
        x
        for x in range(5)
        if all((root / "seed_preds" / folder / f"{m}_s{x}.parquet").exists() for m in d6.MONTHS)
    )


def band_of(lead: pl.Expr) -> pl.Expr:
    """Return the lead-band label of a lead in hours."""
    return (
        pl.when(lead < 24.0)
        .then(pl.lit("0-24h"))
        .when(lead < 120.0)
        .then(pl.lit("24-120h"))
        .otherwise(pl.lit("120-240h"))
    )


def slice_columns(*, t: pl.DataFrame) -> pl.DataFrame:
    """Add the `evening` and `sunny` slice columns (functions of the target time and the ensemble-mean shortwave)."""
    hh = xcols.grid_features(kind="P2")[:, 0]
    evening = (hh[t["target"].to_numpy()] >= 34) & (hh[t["target"].to_numpy()] < 42)
    cols = np.load(W560 / "expected" / "expected_ft_cols.npy", mmap_mode="r")
    sw = np.asarray(cols[t["primary"].to_numpy(), t["target"].to_numpy(), 0], dtype=np.float64)
    sunny = np.zeros(len(sw), dtype=bool)
    prim = t["primary"].to_numpy()
    for p in np.unique(prim):
        m = (prim == p) & (sw > 50.0)
        if m.any():
            sunny[m] = sw[m] >= np.quantile(sw[m], 2.0 / 3.0)
    return t.with_columns(evening=pl.Series(evening), sunny=pl.Series(sunny))


def build_table(*, arms: dict[str, tuple[Path, str]]) -> pl.DataFrame:
    """Return rows with `<arm>_c<seed>` (CRPS) and `<arm>_t<seed>` (summed tail pinball) columns."""
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
            for seed in seeds_of(root=root, folder=folder):
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
                met = score470.row_metrics(frame=fr)
                cols[f"{label}_c{seed}"] = np.where(ok, met["crps"], np.nan)
                cols[f"{label}_t{seed}"] = np.where(
                    ok, sum(met[f"pin_{q}"] for q in TAIL_LEVELS), np.nan
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
    fin = pl.all_horizontal(
        [
            pl.col(f"{a}_c{x}").is_finite()
            for a in main_arms
            for x in seeds_of(root=arms[a][0], folder=arms[a][1])[:1]
        ]
    )
    labels = pl.read_csv(
        d6.W559.parent.parent
        / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
    )
    lab = dict(zip(labels["primary"].to_list(), labels["label"].to_list(), strict=True))
    assert lab[D8] == "D8" and lab[D14] == "D14", lab
    t = t.with_columns(main16=pl.col("main16") & fin)
    return slice_columns(t=t)


def seed_sigma(*, sub: pl.DataFrame, col: str) -> float:
    """Return the pooled within-fold sd of the fold mean of B's seed 0 and seed 1."""
    mm = (
        sub.group_by("month")
        .agg(pl.col(f"{BASE}_{col}0").mean().alias("a"), pl.col(f"{BASE}_{col}1").mean().alias("b"))
        .sort("month")
    )
    d = (mm["a"] - mm["b"]).to_numpy()
    return math.sqrt(float((d**2).mean()) / 2.0)


def contrasts_for(
    *, t: pl.DataFrame, arms: list[str], seeds: dict[str, tuple[int, ...]], measure: str
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return absolute, contrast, and MDE records of every arm against B for one measure (`c` CRPS or `t` tail)."""
    absolute, contrasts, mdes = [], [], []
    rng = np.random.default_rng(0)
    for name in ("eight", "six", "report"):
        part = s.row_set(table=t, name=name)
        for slice_name, expr in SLICES.items():
            if slice_name in {"leave D8 out"} and name != "eight":
                continue
            sl = part.filter(expr.fill_null(value=False)) if slice_name != "all rows" else part
            for band in (*BANDS, "all"):
                sub = sl if band == "all" else sl.filter(pl.col("band") == band)
                if sub.height < 200:
                    continue
                if {0, 1} <= set(seeds[BASE]):
                    sig = seed_sigma(sub=sub, col=measure)
                    mdes.append(
                        {
                            "set": name,
                            "slice": slice_name,
                            "band": band,
                            "measure": measure,
                            "sigma": sig,
                            "mde_2seed": 1.96 * sig / math.sqrt(2),
                            "mde_1seed": 1.96 * math.sqrt(2) * sig,
                        }
                    )
                for arm in arms:
                    ss = seeds[arm]
                    absolute.append({"set": name, "slice": slice_name, "band": band, "measure": measure, "arm": arm, "n": sub.height, "seeds": len(ss),
                                     "mean": float(np.mean([sub[f"{arm}_{measure}{x}"].mean() for x in ss]))})  # fmt: skip
                    if arm == BASE:
                        continue
                    use = [x for x in ss if x in seeds[BASE]]
                    if not use:
                        continue
                    d = sub.with_columns(
                        d=pl.mean_horizontal(
                            [
                                pl.col(f"{arm}_{measure}{x}") - pl.col(f"{BASE}_{measure}{x}")
                                for x in use
                            ]
                        )
                    )
                    st = s.block_stats(diff=d.select("month", "day", "d"), rng=rng, tests=1)
                    base_mean = float(np.mean([sub[f"{BASE}_{measure}{x}"].mean() for x in use]))
                    contrasts.append(
                        {
                            "set": name,
                            "slice": slice_name,
                            "band": band,
                            "measure": measure,
                            "arm": arm,
                            "seeds": len(use),
                            "base_mean": base_mean,
                            "relative": st["difference"] / base_mean,
                            **st,
                        }
                    )
    return absolute, contrasts, mdes


def screen(*, contrasts: pl.DataFrame, mde: pl.DataFrame, arms: list[str]) -> pl.DataFrame:
    """Apply the Stage 1 pass rules (seed 0 contrasts against B) and return one row per arm."""
    rows = []
    for arm in arms:
        if arm == BASE or "-" in arm or arm in REFS:
            continue
        rec = {"arm": arm}
        crps_pass = tail_pass = slice_pass = False
        why = []
        for g in ("eight", "six"):

            def get(slice_name: str, measure: str, g: str = g) -> dict | None:
                r = contrasts.filter(
                    (pl.col("set") == g)
                    & (pl.col("slice") == slice_name)
                    & (pl.col("band") == SCREEN_BAND)
                    & (pl.col("measure") == measure)
                    & (pl.col("arm") == arm)
                )
                return r.row(0, named=True) if r.height else None

            def mde_of(slice_name: str, measure: str, g: str = g) -> float:
                r = mde.filter(
                    (pl.col("set") == g)
                    & (pl.col("slice") == slice_name)
                    & (pl.col("band") == SCREEN_BAND)
                    & (pl.col("measure") == measure)
                )
                return float(r["mde_2seed"][0])

            allc, allt = get("all rows", "c"), get("all rows", "t")
            if allc is None or allt is None:
                continue
            rec[f"{g}_crps_diff"] = allc["difference"]
            rec[f"{g}_crps_folds_better"] = allc["folds"] - allc["folds_positive"]
            rec[f"{g}_tail_relative"] = allt["relative"]
            m_c = mde_of("all rows", "c")
            rec[f"{g}_mde"] = m_c
            if allc["difference"] <= -m_c and (allc["folds"] - allc["folds_positive"]) >= 4:
                crps_pass = True
                why.append(f"CRPS at {g}")
            if allt["relative"] <= -TAIL_GAIN and allc["difference"] <= m_c:
                tail_pass = True
                why.append(f"tail at {g}")
            harmful = False
            gains = []
            for sn in SLICE_NAMES:
                cs = get(sn, "c")
                if cs is None:
                    continue
                m_s = mde_of(sn, "c")
                rec[f"{g}_{sn}"] = cs["difference"]
                if cs["difference"] <= -m_s:
                    gains.append(sn)
                if cs["difference"] > max(HARM_FLOOR, m_s) and (
                    cs["low_month"] > 0 or cs["folds_positive"] >= 4
                ):
                    harmful = True
            if gains and not harmful:
                slice_pass = True
                why.append(f"slice {','.join(gains)} at {g}")
            rec[f"{g}_slice_harm"] = harmful
        rec["pass_crps"], rec["pass_tail"], rec["pass_slice"] = crps_pass, tail_pass, slice_pass
        rec["passes_screen"] = crps_pass or tail_pass or slice_pass
        rec["why"] = "; ".join(why)
        rows.append(rec)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def primaries_table(
    *, t: pl.DataFrame, arms: list[str], seeds: dict[str, tuple[int, ...]]
) -> pl.DataFrame:
    """Return the mean CRPS at 24-120 h of every arm at each of the 16 Primaries (seed 0), with D-labels."""
    labels = pl.read_csv(
        d6.W559.parent.parent
        / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
    )
    sub = t.filter(pl.col("main16") & (pl.col("band") == SCREEN_BAND))
    agg = (
        sub.group_by("primary")
        .agg(
            pl.len().alias("n"), *[pl.col(f"{a}_c0").mean().alias(a) for a in arms if 0 in seeds[a]]
        )
        .sort("primary")
    )
    return agg.join(labels.select("primary", "label"), on="primary", how="left")


def leak(*, arms: list[str], paths: dict[str, tuple[Path, str]]) -> pl.DataFrame:
    """Return the injected-shift leak of each arm with stress fits, in points of capacity."""
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    rows = []
    for arm in arms:
        root, folder = paths[arm]
        for month in d6.MONTHS:
            ctrl_path = root / "fits" / folder / f"{month}_d14_s0.parquet"
            stress = [
                root / "fits" / folder / f"{month}_d14_s0_stress{x}.parquet" for x in ("p", "n")
            ]
            if not (ctrl_path.exists() and all(p.exists() for p in stress)):
                break
            ctrl = pl.read_parquet(ctrl_path).select(*keys, c="q50")
            for sign, p in zip((1, -1), stress, strict=True):
                j = ctrl.join(pl.read_parquet(p).select(*keys, s="q50"), on=keys)
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


STAGE2: Final[tuple[str, ...]] = ("DL", "ML", "P1")
COUNTED_TESTS: Final[int] = 16
"""Stage 2 counted tests: 3 passers x 2 groups x 2 measures = 12, plus DT against B and DL against DT at 2 groups = 4."""


def counted(*, t: pl.DataFrame, seeds: dict[str, tuple[int, ...]]) -> pl.DataFrame:
    """Return the Stage 2 counted contrasts (24-120 h, all rows, seeds 0 and 1) at the Bonferroni level of `COUNTED_TESTS`."""
    rng = np.random.default_rng(0)
    pairs = [(a, BASE, m) for a in STAGE2 for m in ("c", "t")] + [
        ("DT", BASE, "c"),
        ("DL", "DT", "c"),
    ]
    rows = []
    for a, b, m in pairs:
        if a not in seeds or b not in seeds:
            continue
        use = [x for x in seeds[a] if x in seeds[b]]
        if len(use) < 2:
            continue
        for g in ("eight", "six"):
            sub = s.row_set(table=t, name=g).filter(pl.col("band") == SCREEN_BAND)
            d = sub.with_columns(
                d=pl.mean_horizontal([pl.col(f"{a}_{m}{x}") - pl.col(f"{b}_{m}{x}") for x in use])
            )
            st = s.block_stats(diff=d.select("month", "day", "d"), rng=rng, tests=COUNTED_TESTS)
            base_mean = float(np.mean([sub[f"{b}_{m}{x}"].mean() for x in use]))
            rows.append(
                {
                    "set": g,
                    "a": a,
                    "b": b,
                    "measure": m,
                    "seeds": len(use),
                    "base_mean": base_mean,
                    "relative": st["difference"] / base_mean,
                    "tests": COUNTED_TESTS,
                    **st,
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def siblings(
    *, t: pl.DataFrame, arms: list[str], seeds: dict[str, tuple[int, ...]]
) -> pl.DataFrame:
    """Return the seed 0 CRPS at 24-120 h with lags removed and aged by 1, 2, and 3 days."""
    rows = []
    for g in ("eight", "six"):
        sub = s.row_set(table=t, name=g).filter(pl.col("band") == SCREEN_BAND)
        for arm in arms:
            for suffix in ("", *[f"-{x}" for x in SIBLING_SUFFIXES]):
                col = f"{arm}{suffix}_c0"
                if col in sub.columns:
                    rows.append(
                        {
                            "set": g,
                            "arm": arm,
                            "variant": suffix.lstrip("-") or "lags at 24 h",
                            "crps": float(sub[col].mean()),
                        }
                    )
    return pl.DataFrame(rows)


def main() -> None:
    """Score and write the tables."""
    paths = arm_paths()
    assert BASE in paths, "B's stored forecasts are missing"
    arms = list(paths)
    seeds = {a: seeds_of(root=paths[a][0], folder=paths[a][1]) for a in arms}
    print("arms:", {a: seeds[a] for a in arms})
    t = build_table(arms=paths)
    out = W579 / "scores"
    out.mkdir(parents=True, exist_ok=True)
    main_arms = [a for a in arms if "-" not in a]
    allabs, allcon, allmde = [], [], []
    for measure in ("c", "t"):
        a, c, m = contrasts_for(t=t, arms=main_arms, seeds=seeds, measure=measure)
        allabs += a
        allcon += c
        allmde += m
    ab, co, md = (
        pl.DataFrame(allabs),
        pl.DataFrame(allcon, infer_schema_length=None),
        pl.DataFrame(allmde),
    )
    ab.write_csv(out / "absolute.csv")
    co.write_csv(out / "contrasts.csv")
    md.write_csv(out / "mde.csv")
    sc = screen(contrasts=co, mde=md, arms=main_arms)
    if sc.height:
        sc.write_csv(out / "screen.csv")
    cn = counted(t=t, seeds=seeds)
    if cn.height:
        cn.write_csv(out / "counted.csv")
    siblings(t=t, arms=[a for a in main_arms if a not in REFS], seeds=seeds).write_csv(
        out / "siblings.csv"
    )
    prim = primaries_table(t=t, arms=main_arms, seeds=seeds)
    prim.write_csv(out / "primaries.csv")
    lk = leak(arms=[a for a in main_arms if a != "M0" and a != "E2"], paths=paths)
    lk.write_csv(out / "leak.csv")
    json.dump(
        {
            "arms": arms,
            "seeds": {a: list(v) for a, v in seeds.items()},
            "rows": t.height,
            "main16": int(t["main16"].sum()),
        },
        open(out / "checks.json", "w"),
        indent=1,
    )
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=220):
        print(
            md.filter(
                (pl.col("slice") == "all rows") & (pl.col("band") == SCREEN_BAND)
            ).with_columns(pl.col(pl.Float64).round(4))
        )
        if sc.height:
            print(
                sc.select(
                    [
                        c
                        for c in (
                            "arm",
                            "eight_crps_diff",
                            "six_crps_diff",
                            "eight_tail_relative",
                            "six_tail_relative",
                            "passes_screen",
                            "why",
                        )
                        if c in sc.columns
                    ]
                ).with_columns(pl.col(pl.Float64).round(3))
            )
        if cn.height:
            print(
                cn.select(
                    "set",
                    "a",
                    "b",
                    "measure",
                    "difference",
                    "relative",
                    "low_bonferroni_month",
                    "high_bonferroni_month",
                    "folds_positive",
                ).with_columns(pl.col(pl.Float64).round(3))
            )
        print(lk.with_columns(pl.col(pl.Float64).round(2)))
    _ = prep422, sys


if __name__ == "__main__":
    main()
