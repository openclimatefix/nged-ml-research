# ruff: noqa: E501, PLR0912, PLR0915, PERF401
"""Experiment 577: score the offset arms against D5 (experiment 565) by CRPS, tails, and calibration.

Each seed of each arm is scored on its own, widened by 1.15 about the median as every stored row
table is (`--widen 1.0` re-scores the raw forecast). A row's contrast is the mean over the chosen
seeds of the arm's metric minus the reference arm's metric of the same seed, so arms are paired by
fold and seed. Positive differences mean the arm is worse (CRPS, tail loss) or has the larger
shortfall.

Row metrics, in points of effective capacity unless stated: `crps`; `tail` (the 1% and 99% pinball
losses summed); `in80`, `in90`, `in98` (1 when the actual lies inside the central interval, so a
mean of 0.9 is nominal for `in90`); `w90` (width of the 5% to 95% interval).

Usage: python e00577_score.py --arms D5 O1 ... --seeds 0 1 [--ref D5] [--widen 1.15] [--tag NAME]
Outputs under `data/ml-experiments/00577-*/work/scores/`: `<tag>_absolute.csv`,
`<tag>_contrasts.csv`, `<tag>_slices.csv`, `<tag>_primaries.csv`, `<tag>_pit.csv`,
`<tag>_seed_signs.csv`, `<tag>_checks.json`.
"""

import argparse
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

NAME: Final = "00577-static-between-steps-offsets-and-probabilistic-offset-widening"
W: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
W565: Final[Path] = d6.W
BANDS: Final[tuple[str, ...]] = ("24-48h", "48-72h", "72-120h")
METRICS: Final[tuple[str, ...]] = ("crps", "tail", "in80", "in90", "in98", "w90")
NOMINAL: Final[dict[str, float]] = {"in80": 0.80, "in90": 0.90, "in98": 0.98}
LEVELS: Final = d6.LEVELS
BONFERRONI: Final[int] = 22
CHRISTMAS: Final = score470.CHRISTMAS
CHRISTMAS_MONTH: Final = 12
ARM_DIRS: Final[dict[str, Path]] = {
    "D5": W565 / "seed_preds" / "D5",
    "M0": W565 / "seed_preds" / "C-M0",
}


def arm_dir(*, arm: str) -> Path:
    """Return where an arm's assembled forecasts live."""
    return ARM_DIRS.get(arm, W / "seed_preds" / arm)


def row_metrics(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the per-row metrics of a frame holding the 11 quantile columns and `actual`."""
    q = np.sort(frame.select(list(LEVELS)).to_numpy().astype(np.float64), axis=1)
    y = frame["actual"].to_numpy().astype(np.float64)
    base = score470.row_metrics(frame=frame)
    col = {name: i for i, name in enumerate(LEVELS)}

    def inside(lo: str, hi: str) -> np.ndarray:
        return ((y >= q[:, col[lo]]) & (y <= q[:, col[hi]])).astype(np.float64)

    return {
        "crps": base["crps"],
        "tail": base["pin_q01"] + base["pin_q99"],
        "in80": inside("q10", "q90"),
        "in90": inside("q05", "q95"),
        "in98": inside("q01", "q99"),
        "w90": score470.POINTS * (q[:, col["q95"]] - q[:, col["q05"]]),
    }


def build_table(*, arms: list[str], seeds: tuple[int, ...], factor: float) -> pl.DataFrame:
    """Return rows with `<arm>_s<seed>__<metric>` columns."""
    sites = build470.site_map()
    frames = []
    for month in d6.MONTHS:
        rows = pl.read_parquet(
            d6.W559 / "scores" / f"rows_{month}.parquet",
            columns=[*d6.KEYS, "group", "cls", "lead_hours", "tgt_real", "cold", "main16"],
        )
        actual = score454.month_base(month=month).select(*d6.KEYS, "actual", "nra")
        rows = rows.join(actual, on=d6.KEYS, how="left", maintain_order="left", validate="1:1")
        cols: dict[str, np.ndarray] = {}
        for arm in arms:
            for seed in seeds:
                frame = build470.pool_stored(
                    folder=arm_dir(arm=arm), month=month, sites=sites, seeds=(seed,)
                )
                fr = rows.select(*d6.KEYS, "actual").join(
                    frame.select(*d6.KEYS, *LEVELS), on=d6.KEYS, how="left", maintain_order="left"
                )
                ok = fr["q50"].is_not_null().to_numpy()
                fr = fr.with_columns(*[pl.col(q).fill_null(0.0) for q in LEVELS])
                fr = score470.widen_arm(arm="M0", frame=fr, factor=factor)
                for name, v in row_metrics(frame=fr).items():
                    cols[f"{arm}_s{seed}__{name}"] = np.where(ok, v, np.nan)
        frames.append(
            rows.hstack(
                pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float64) for k, v in cols.items()})
            ).with_columns(
                month=pl.lit(month), day=(pl.col("origin") // 48).cast(pl.Int32), band=s.band_expr()
            )
        )
    t = pl.concat(frames)
    day = np.datetime64("2019-09-01", "D") + (t["target"].to_numpy() // 48).astype("timedelta64[D]")
    month_of = day.astype("datetime64[M]").astype(int) % 12 + 1
    dom = (day - day.astype("datetime64[M]")).astype(int) + 1
    t = t.with_columns(
        christmas=pl.Series(
            (month_of == CHRISTMAS_MONTH) & (dom >= CHRISTMAS[0]) & (dom <= CHRISTMAS[1])
        )
    )
    fin = pl.all_horizontal([pl.col(f"{a}_s{x}__crps").is_finite() for a in arms for x in seeds])
    return t.with_columns(main16=pl.col("main16") & fin)


def mde(*, sub: pl.DataFrame, arm: str, ref: str, metric: str, seeds: tuple[int, ...]) -> float:
    """Return the seed-noise half-width: 1.96 x the seed-to-seed sd of the paired contrast / sqrt(seeds)."""
    if len(seeds) < 2:
        return float("nan")
    cols = [f"d_{arm}_{ref}_{metric}_s{x}" for x in seeds]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--ref", default="D5")
    parser.add_argument("--widen", type=float, default=score470.HEADLINE)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--tests", type=int, default=BONFERRONI)
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    arms = list(dict.fromkeys([args.ref, *args.arms]))
    t = build_table(arms=arms, seeds=seeds, factor=args.widen)
    sites = pl.read_csv(s.DATA / "site_map.csv")
    label = dict(zip(sites["primary"].to_list(), sites["label"].to_list(), strict=True))
    t = t.with_columns(label=pl.col("primary").replace_strict(label, default="?"))
    others = [a for a in arms if a != args.ref]
    for a in others:
        for m in METRICS:
            for x in seeds:
                t = t.with_columns(
                    (pl.col(f"{a}_s{x}__{m}") - pl.col(f"{args.ref}_s{x}__{m}")).alias(
                        f"d_{a}_{args.ref}_{m}_s{x}"
                    )
                )
    out = W / "scores"
    out.mkdir(parents=True, exist_ok=True)
    absolute, contrasts, signs, primaries = [], [], [], []
    rng = np.random.default_rng(0)
    for name in ("eight", "six", "report"):
        part = s.row_set(table=t, name=name)
        for kind in (
            "all",
            "leave_D8_out",
            "D8",
            "D14",
            "cold",
            "christmas",
            "no_event",
            "in_event",
        ):
            if kind in {"leave_D8_out", "D8"} and name != "eight":
                continue
            if kind == "D14" and name != "six":
                continue
            p = part
            if kind == "leave_D8_out":
                p = part.filter(pl.col("label") != "D8")
            elif kind in {"D8", "D14"}:
                p = part.filter(pl.col("label") == kind)
            elif kind == "cold":
                p = (s.row_set(table=t, name="six_all") if name == "six" else part).filter(
                    pl.col("cold")
                )
            elif kind == "christmas":
                p = part.filter(pl.col("christmas"))
            elif kind == "no_event":
                p = part.filter(~pl.col("tgt_real"))
            elif kind == "in_event":
                p = part.filter(pl.col("tgt_real"))
            for band in (*BANDS, "24-120h"):
                sub = (
                    p.filter(pl.col("band").is_in(BANDS))
                    if band == "24-120h"
                    else s.in_band(part=p, band=band)
                )
                if sub.height == 0 or (kind not in {"all", "leave_D8_out"} and band != "24-120h"):
                    continue
                for arm in arms:
                    absolute.append(
                        {
                            "set": name, "kind": kind, "band": band, "arm": arm, "n": sub.height,
                            **{
                                m: float(np.mean([sub[f"{arm}_s{x}__{m}"].mean() for x in seeds]))
                                for m in METRICS
                            },
                        }
                    )  # fmt: skip
                for arm in others:
                    for metric in METRICS:
                        sub2 = sub.with_columns(
                            d=pl.mean_horizontal(
                                [pl.col(f"d_{arm}_{args.ref}_{metric}_s{x}") for x in seeds]
                            )
                        )
                        st = s.block_stats(
                            diff=sub2.select("month", "day", "d"), rng=rng, tests=args.tests
                        )
                        contrasts.append(
                            {
                                "set": name, "kind": kind, "band": band, "arm": arm, "ref": args.ref,
                                "metric": metric, "tests": args.tests,
                                "mde": mde(sub=sub2, arm=arm, ref=args.ref, metric=metric, seeds=seeds),
                                **st,
                            }
                        )  # fmt: skip
                        if metric == "crps":
                            signs.append(
                                {
                                    "set": name, "kind": kind, "band": band, "arm": arm,
                                    **{f"seed{x}": float(sub2[f"d_{arm}_{args.ref}_{metric}_s{x}"].mean()) for x in seeds},
                                }
                            )  # fmt: skip
    # Per-Primary table (24-120 h, all 16 Primaries) and the probability integral transform.
    allp = t.filter(pl.col("band").is_in(BANDS) & pl.col("main16"))
    rep = s.row_set(table=t, name="report").filter(pl.col("band").is_in(BANDS))
    for lab, g in pl.concat([allp, rep], how="diagonal_relaxed").group_by("label"):
        for arm in arms:
            primaries.append(
                {
                    "label": lab[0], "arm": arm, "n": g.height,
                    **{m: float(np.mean([g[f"{arm}_s{x}__{m}"].mean() for x in seeds])) for m in METRICS},
                }
            )  # fmt: skip
    pit = []
    for arm in arms:
        for name in ("eight", "six"):
            sub = s.row_set(table=t, name=name).filter(pl.col("band").is_in(BANDS))
            pit.append(
                {
                    "set": name, "arm": arm,
                    **{f"{m}": float(np.mean([sub[f"{arm}_s{x}__{m}"].mean() for x in seeds])) for m in ("in80", "in90", "in98")},
                }
            )  # fmt: skip
    base = f"{args.tag}"
    pl.DataFrame(absolute).write_csv(out / f"{base}_absolute.csv")
    pl.DataFrame(contrasts).write_csv(out / f"{base}_contrasts.csv")
    pl.DataFrame(signs).write_csv(out / f"{base}_seed_signs.csv")
    pl.DataFrame(primaries).write_csv(out / f"{base}_primaries.csv")
    pl.DataFrame(pit).write_csv(out / f"{base}_pit.csv")
    (out / f"{base}_checks.json").write_text(
        json.dumps(
            {
                "arms": arms,
                "seeds": seeds,
                "widen": args.widen,
                "rows": t.height,
                "main16": int(t["main16"].sum()),
                "tests": args.tests,
            },
            indent=1,
        )
    )
    with pl.Config(tbl_rows=120, tbl_cols=14):
        c = pl.DataFrame(contrasts)
        print(
            c.filter(
                (pl.col("kind") == "all")
                & (pl.col("band") == "24-120h")
                & (pl.col("set") != "report")
            )
            .select(
                "set",
                "arm",
                "metric",
                "difference",
                "low_bonferroni_month",
                "high_bonferroni_month",
                "mde",
                "folds_positive",
            )
            .with_columns(pl.col(pl.Float64).round(3))
        )


if __name__ == "__main__":
    main()
