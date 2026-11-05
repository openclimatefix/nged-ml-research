# ruff: noqa: E501, PTH123, SIM115, PT018, PERF401, PLR0912
"""Experiment 574: score the Part D arms in three lead bands.

The score covers slices, contrasts, per-Primary gains, and leak.

Arms: M0 and E2 are experiment 565's stored `C-M0` and `D5`; P0 is experiment 572's stored `V3D5`
(v3 plus the D5 offset); P1 (spread columns), P3E0 (encoders unfrozen), P2 and P2plac (thermal-lag and
solar-gain columns and their shuffled-day placebo), P4 and P4E (three-step schedule without and with
the 3-day embargo), P5, P6, and the leave-one-out arms are this experiment's fits. A-pool-24 (experiment
558) and L-1d (experiment 511) are stored references with a different refit cadence (daily). Each seed
of each arm is scored on its own (widened by 1.15, as every stored row table is), and a contrast is
the mean over the two seeds of the paired CRPS difference. Positive differences mean the first arm is
worse.

Bonferroni count 42: the add-one contrasts (P1, P3E0, P2, P4 each minus P0) and the final candidate
contrasts (P5 minus P0, E2, and A-pool-24), each x 2 groups x 3 lead bands, all on all rows. A separate
family of 2: P2 minus P2plac on cold winter nights at each group (all lead bands). Everything else is
descriptive.

Outputs under `data/ml-experiments/00574-*/work/scores/`: `absolute.csv`, `contrasts.csv`,
`seed_signs.csv`, `leak.csv`, `per_primary.csv`, `checks.json`.
"""

import json
import math
from pathlib import Path
from typing import Final

import e00422_prep as prep422  # ty: ignore[unresolved-import]
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
W574: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/work"
)
W558: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00558-lags-in-pretraining-and-24-hour-cap-lag-light-demand-network/work"
)
W511: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00511-lagless-network-refit-cadence-half-life-and-mixed-pretraining/work"
)
W568: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00568-attention-mixer-modular-network-with-lead-and-spread-token/work"
)
SEEDS: Final[tuple[int, ...]] = (0, 1)
GRID_US: Final[int] = prep422.GRID_START_US
STEP_US: Final[int] = prep422.STEP_US
TESTS: Final[int] = 42
PLACEBO_TESTS: Final[int] = 2
BANDS: Final[dict[str, tuple[float, float]]] = {
    "0-24h": (0.0, 24.0),
    "24-120h": (24.0, 120.0),
    "120-240h": (120.0, 240.0),
}
LAG_ARMS: Final[tuple[str, ...]] = (
    "P0",
    "P1",
    "P3E0",
    "P2",
    "P2plac",
    "P4",
    "P4E",
    "P5",
    "P6",
    "P7",
)
ARM_DIRS: Final[dict[str, tuple[Path, str]]] = {
    "M0": (d6.W, "C-M0"),
    "E2": (d6.W, "D5"),
    "Apool24": (W558, "A-pool-24"),
    "L1d": (W511, "L-1d"),
    "P0": (W572, "V3D5"),
    "P7": (W572, "V35"),
    **{a: (W574, a) for a in ("P1", "P3E0", "P2", "P2plac", "P4", "P4E", "P5", "P6")},
}
SIBLING_SUFFIXES: Final[tuple[str, ...]] = ("removed", "aged1", "aged2", "aged3")
COUNTED: Final[tuple[tuple[str, str], ...]] = (
    ("P1", "P0"),
    ("P3E0", "P0"),
    ("P2", "P0"),
    ("P4", "P0"),
    ("P5", "P0"),
    ("P5", "E2"),
    ("P5", "Apool24"),
)
PLACEBO_PAIR: Final[tuple[str, str]] = ("P2", "P2plac")
DESCRIPTIVE: Final[tuple[tuple[str, str], ...]] = (
    ("P0", "E2"),
    ("P4E", "P0"),
    ("P4E", "P4"),
    ("P6", "P5"),
    ("P6", "P0"),
    ("P7", "P0"),
    ("P0", "L1d"),
    ("P5", "L1d"),
    ("P0", "M0"),
    ("P2", "P5"),
    ("P4E", "P5"),
)
KEYS: Final = ["time_series_id", "power_fcst_init_time", "valid_time"]
D8: Final[int] = 4
D14: Final[int] = 9
SITE_MAP: Final[pl.DataFrame] = pl.read_csv(
    ML_EXPERIMENTS_DATA_DIR
    / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
)
D15: Final[int] = int(SITE_MAP.filter(pl.col("label") == "D15")["primary"][0])
D18: Final[int] = int(SITE_MAP.filter(pl.col("label") == "D18")["primary"][0])
"""Primary indices of D8 and D14 in experiment 521's site map (asserted in `build_table`)."""


def arms_present() -> dict[str, tuple[Path, str]]:
    """Return the arms (including lag-sibling twins) whose assembled forecasts exist for every fold and seed."""
    out = {}
    for name, (root, tag) in ARM_DIRS.items():
        variants = [(name, tag)]
        if name in LAG_ARMS or name in {"P0", "P7"}:
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
        actual = score454.month_base(month=month).select(*d6.KEYS, "actual", "nra")
        rows = rows.join(actual, on=d6.KEYS, how="left", maintain_order="left", validate="1:1")
        terc = pl.read_parquet(W568 / "terciles" / f"{month}.parquet").select(
            primary=pl.col("primary").cast(pl.Int16),
            origin=((pl.col("power_fcst_init_time") - GRID_US) // STEP_US).cast(pl.Int32),
            target=((pl.col("valid_time") - GRID_US) // STEP_US).cast(pl.Int32),
            tercile=pl.col("tercile"),
        )
        rows = rows.join(terc, on=d6.KEYS, how="left", maintain_order="left", validate="1:1")
        print(
            month,
            "rows without a spread tercile:",
            rows["tercile"].null_count(),
            "of",
            rows.height,
            flush=True,
        )
        rows = rows.with_columns(
            evening=(
                pl.from_epoch(GRID_US + pl.col("target").cast(pl.Int64) * STEP_US, time_unit="us")
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("Europe/London")
                .dt.hour()
                .is_between(17, 20)
            )
        )
        cols = {}
        for label, (root, folder) in arms.items():
            for seed in SEEDS:
                frame = build470.pool_stored(
                    folder=root / "seed_preds" / folder, month=month, sites=sites, seeds=(seed,)
                )
                fr = rows.select(*d6.KEYS, "actual", "nra").join(
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
                fr = fr.with_columns(actual_nra=pl.col("actual") - pl.col("nra"))
                cols[f"{label}_nra_s{seed}"] = np.where(
                    ok, score470.row_metrics(frame=fr, actual="actual_nra")["crps"], np.nan
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
            .drop("actual", "nra")
        )
    t = pl.concat(frames)
    main_arms = [a for a in arms if "-" not in a and a not in {"P7", "Apool24", "L1d"}]
    fin = pl.all_horizontal([pl.col(f"{a}_s{x}").is_finite() for a in main_arms for x in SEEDS])
    labels = pl.read_csv(
        d6.W559.parent.parent
        / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
    )
    lab = dict(zip(labels["primary"].to_list(), labels["label"].to_list(), strict=True))
    assert lab[D8] == "D8" and lab[D14] == "D14", lab
    return t.with_columns(main16=pl.col("main16") & fin)


def mde(*, sub: pl.DataFrame, a: str, b: str) -> float:
    """Return the seed-noise half-width of a paired contrast (1.96 x seed sd, pooled within folds, / sqrt(seeds))."""
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
    "all rows": pl.col("group").is_not_null(),
    "no labelled event": ~pl.col("tgt_real"),
    "cold winter nights": pl.col("cold"),
    "Christmas week": pl.col("xmas"),
    "D8 alone": pl.col("primary") == D8,
    "D14 alone": pl.col("primary") == D14,
    "leave D8 out": pl.col("primary") != D8,
    "evenings (17:00-20:59 local)": pl.col("evening"),
    "high-spread tercile": pl.col("tercile") == 3,
    "D15 alone": pl.col("primary") == D15,
    "D18 alone": pl.col("primary") == D18,
}


def leak(*, arms: list[str]) -> pl.DataFrame:
    """Return the injected-shift leak of each arm (mean over five folds, min, max), in points of capacity."""
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


def real_events(*, out: Path) -> None:
    """Summarise the real-event absorbed share per arm and its paired difference from P0.

    Events are resampled as clusters of (Primary, month), 2000 draws, seed 0.
    """
    frames = {
        p.stem.removeprefix("real_events_"): pl.read_parquet(p)
        for p in sorted(out.glob("real_events_*.parquet"))
    }
    if "P0" not in frames:
        return
    key = ["month", "primary", "onset", "end"]
    base = frames["P0"].select(*key, "depth_points", p0="absorbed_share")
    rng = np.random.default_rng(0)
    rows = []
    for arm, d in frames.items():
        j = base.join(d.select(*key, arm_share="absorbed_share"), on=key)
        clusters = j.with_columns(cl=pl.col("month") + "_" + pl.col("primary").cast(pl.String))
        groups = [g for _, g in clusters.group_by("cl")]
        diffs = []
        for _ in range(2000):
            pick = rng.integers(0, len(groups), len(groups))
            x = pl.concat([groups[i] for i in pick])
            diffs.append(float((x["arm_share"] - x["p0"]).mean()))
        rows.append(
            {
                "arm": arm,
                "events": j.height,
                "mean_share": float(j["arm_share"].mean()),  # ty: ignore[invalid-argument-type]
                "depth_weighted_share": float(
                    (j["arm_share"] * j["depth_points"]).sum() / j["depth_points"].sum()  # ty: ignore[unsupported-operator]
                ),
                "minus_p0": float((j["arm_share"] - j["p0"]).mean()),  # ty: ignore[invalid-argument-type]
                "low": float(np.quantile(diffs, 0.025)),
                "high": float(np.quantile(diffs, 0.975)),
            }
        )
    table = pl.DataFrame(rows)
    table.write_csv(out / "real_events_summary.csv")
    print(table.with_columns(pl.col(pl.Float64).round(4)))


def main() -> None:
    """Score and write the tables."""
    arms = arms_present()
    present = [a for a in arms if "-" not in a]
    print("arms present:", list(arms))
    assert {"M0", "E2", "P0"} <= set(present), present
    t = build_table(arms=arms)
    out = W574 / "scores"
    out.mkdir(parents=True, exist_ok=True)
    pairs = [
        (a, b) for a, b in (*COUNTED, PLACEBO_PAIR, *DESCRIPTIVE) if a in present and b in present
    ]
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
            print("slice", name, slice_name, flush=True)
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
                    placebo_family = (
                        (a, b) == PLACEBO_PAIR
                        and slice_name == "cold winter nights"
                        and band == "all"
                        and name != "report"
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
                        tests=TESTS if counted else (PLACEBO_TESTS if placebo_family else 1),
                    )
                    contrasts.append({"set": name, "slice": slice_name, "band": band, "a": a, "b": b, "counted": counted, "placebo_family": placebo_family,
                                      "tests": TESTS if counted else (PLACEBO_TESTS if placebo_family else 1), "mde": mde(sub=sub2, a=a, b=b), **st})  # fmt: skip
                    signs.append({"set": name, "slice": slice_name, "band": band, "a": a, "b": b,
                                  **{f"seed{x}": float(sub2[f"d_{a}_{b}_s{x}"].mean()) for x in SEEDS}})  # fmt: skip
    pl.DataFrame(absolute).write_csv(out / "absolute.csv")
    pl.DataFrame(contrasts).write_csv(out / "contrasts.csv")
    pl.DataFrame(signs).write_csv(out / "seed_signs.csv")
    # Per-Primary table (all 16 Primaries, all rows of the Primary, three bands): each arm's CRPS and its gain against L-1d.
    full = t.filter(pl.col("main16") | (pl.col("group") == "report_only"))
    pp = []
    for p_idx in range(16):
        for band in BANDS:
            sub_p = full.filter((pl.col("primary") == p_idx) & (pl.col("band") == band))
            for arm in present:
                if sub_p.height:
                    pp.append({"primary": p_idx, "label": SITE_MAP.filter(pl.col("primary") == p_idx)["label"][0], "band": band, "arm": arm,
                               "crps": float(np.mean([sub_p[f"{arm}_s{x}"].mean() for x in SEEDS]))})  # fmt: skip
    pl.DataFrame(pp).write_csv(out / "per_primary.csv")
    # Rows inside labelled events, scored against NRA-corrected actuals (descriptive).
    ev = []
    for name in ("eight", "six_all"):
        part = s.row_set(table=t, name=name).filter(pl.col("tgt_real"))
        for band in BANDS:
            sub_e = part.filter(pl.col("band") == band)
            for arm in present:
                ev.append({"set": name, "band": band, "arm": arm, "n": sub_e.height,
                           "nra_crps": float(np.mean([sub_e[f"{arm}_nra_s{x}"].mean() for x in SEEDS]))})  # fmt: skip
    pl.DataFrame(ev).write_csv(out / "inside_events_nra.csv")
    lk = leak(arms=present)
    lk.write_csv(out / "leak.csv")
    json.dump(
        {"arms": list(arms), "rows": t.height, "main16": int(t["main16"].sum())},
        open(out / "checks.json", "w"),
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
    real_events(out=W574 / "scores")
