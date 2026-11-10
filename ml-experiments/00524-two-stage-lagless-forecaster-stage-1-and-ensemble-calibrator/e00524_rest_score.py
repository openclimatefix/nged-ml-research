# ruff: noqa: ANN001, ANN201, C408, D103, RET503
"""Scoring for the 524 remaining arms (interim, unaudited, not for merge).

Commands: `build`, `rows`, `baseline`, `stats`, `tails`, `curtailment`, `bands`. Reads the new
arms' seed forecasts from `addon_rest/work/seed_preds`; the 524 arms are linked from the 524
folder. Writes only under `addon_rest/work`.
"""

import argparse
import math

import e00510_score as s510  # ty: ignore[unresolved-import]
import e00524_bands as b
import e00524_stage1 as s1
import numpy as np
import polars as pl

OUT = s1.WORK_DIR.parent / "addon_rest"
WORK = OUT / "work"
SEED_PREDS = WORK / "seed_preds"
SOURCE = s1.WORK_DIR
STORED = ("S2-0F", "S2-F-A-nll", "S2-F-B-nll", "S2-J-0.4x", "S2-J-0.1x")
LINKED = (
    *STORED,
    "L-W",
    "L-W-s0",
    "L-W-s1",
    "L-Ww",
    "L-Ww-s0",
    "L-Ww-s1",
    "K2-S1",
    "manual_heuristic",
    "climatology",
    "BG2",
    "A-pool",
)
CANDIDATE_NEW = ("S0-A-F1", "S0-A-F2", "S0-S2", "S0-E2E", "S2-R")


def new_arms() -> tuple[str, ...]:
    return tuple(
        a
        for a in CANDIDATE_NEW
        if (SEED_PREDS / a).exists() and len(list((SEED_PREDS / a).glob("*.parquet"))) == 10
    )


def contrasts(new):
    out = [(a, "L-Ww") for a in new] + [(a, "S2-J-0.4x") for a in new] + [(a, "S2-0F") for a in new]
    out += [
        ("S2-J-0.4x", "L-Ww"),
        ("S2-F-B-nll", "L-Ww"),
        ("S2-J-0.4x", "K2-S1"),
        ("S2-F-B-nll", "K2-S1"),
        ("S2-F-B-nll", "S2-0F"),
    ]
    if "S0-A-F1" in new and "S0-A-F2" in new:
        out.append(("S0-A-F2", "S0-A-F1"))
    if "S0-S2" in new:
        out.append(("S0-S2", "S2-F-B-nll"))
    if "S0-E2E" in new:
        out.append(("S0-E2E", "S0-A-F2"))
    return tuple(dict.fromkeys(out))


def configure(*, tails_arms=None) -> tuple[str, ...]:
    new = new_arms()
    s510.WORK_DIR = WORK
    s510.SEED_PREDS = SEED_PREDS
    s510.ARMS = dict.fromkeys(new, (0, 1))
    s510.CONTRASTS = contrasts(new)
    s510.TESTS = 4 * len(s510.CONTRASTS)
    s510.configure()
    s510.link_stored()
    pred = WORK / "predictions"
    for arm in LINKED:
        link = pred / arm
        src = SOURCE / "predictions" / arm
        if not link.exists() and src.exists():
            link.symlink_to(src, target_is_directory=True)
    sp = s510.score470
    main_arms = (*new, *STORED, "L-Ww", "K2-S1")
    sp.CANDIDATES = new or STORED
    sp.CONTROLS = tuple(a for a in main_arms if a not in sp.CANDIDATES)
    sp.MAIN_ARMS = main_arms
    sp.SINGLES = s510.SINGLES
    sp.CONTRASTS = s510.CONTRASTS
    sp.TESTS = s510.TESTS
    s510.tails470.ARMS = main_arms
    s510.curtailment470.ARMS = main_arms
    s510.curtailment470.CANDIDATES = main_arms
    s510.curtailment470.REFERENCES = ("manual_heuristic",)
    s510.curtailment470.COMPARISONS = (
        *[(a, "L-Ww") for a in main_arms if a != "L-Ww"],
        *[(a, "K2-S1") for a in main_arms if a != "K2-S1"],
        *[(a, "manual_heuristic") for a in main_arms],
    )
    return new


def bands() -> None:
    new = configure()
    rows = pl.concat(
        [
            pl.read_parquet(s510.score470.SCORE_DIR / f"rows_{m}.parquet").with_columns(
                fold=pl.lit(m)
            )
            for m in b.FOLDS
        ]
    ).filter(pl.col("main16"))
    rng = np.random.default_rng(524)
    counted = {(a, "S2-J-0.4x") for a in new} | {(a, "L-Ww") for a in new}
    n_tests = len(counted) * 2 * 3
    recs = []
    for first, second in s510.CONTRASTS:
        for group, flt in (
            ("eight", pl.col("group") == "mva_eight"),
            ("six", (pl.col("group") == "signed_six") & (pl.col("cls") == "clean")),
        ):
            for band, (lo, hi) in b.BANDS.items():
                part = rows.filter(flt & (pl.col("lead_hours") >= lo) & (pl.col("lead_hours") < hi))
                diff = part.select(
                    day=pl.col("origin").cast(pl.Int64) // 48,
                    fold="fold",
                    d=pl.col(f"{first}__crps") - pl.col(f"{second}__crps"),
                ).drop_nulls()
                by = diff.group_by("fold", "day").agg(
                    total=pl.col("d").sum(), count=pl.col("d").len()
                )
                t, n = by["total"].to_numpy(), by["count"].to_numpy().astype(float)
                m, lo_, hi_ = b.intervals(total=t, count=n, level=0.95, rng=rng)
                cnt = (first, second) in counted
                _, lb, hb = (
                    b.intervals(total=t, count=n, level=1 - 0.05 / n_tests, rng=rng)
                    if cnt
                    else (math.nan,) * 3
                )
                folds = diff.group_by("fold").agg(m=pl.col("d").mean()).sort("fold")["m"].to_numpy()
                recs.append(
                    dict(
                        first=first,
                        second=second,
                        group=group,
                        band=band,
                        counted=cnt,
                        difference=m,
                        low95=lo_,
                        high95=hi_,
                        low_bonf=lb,
                        high_bonf=hb,
                        bonferroni_count=n_tests,
                        fold_signs="".join("-" if x < 0 else "+" for x in folds),
                    )
                )
    t = pl.DataFrame(recs)
    t.write_csv(OUT / "band_contrasts.csv")
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    print(t.with_columns(pl.col(pl.Float64).round(3)).drop("counted"))


def tails_by_lead() -> None:
    """q01/q05/q95/q99 pinball and coverage by lead band from the stored per-row scores."""
    configure()
    arms = (*STORED[1:], "L-Ww", "K2-S1", *new_arms())
    rows = pl.concat(
        [
            pl.read_parquet(s510.score470.SCORE_DIR / f"rows_{m}.parquet").with_columns(
                fold=pl.lit(m)
            )
            for m in b.FOLDS
        ]
    ).filter(pl.col("main16"))
    recs = []
    for group, flt in (
        ("eight", pl.col("group") == "mva_eight"),
        ("six", (pl.col("group") == "signed_six") & (pl.col("cls") == "clean")),
    ):
        for band, (lo, hi) in b.BANDS.items():
            part = rows.filter(flt & (pl.col("lead_hours") >= lo) & (pl.col("lead_hours") < hi))
            for arm in arms:
                if f"{arm}__crps" not in part.columns:
                    continue
                r = {
                    "group": group,
                    "band": band,
                    "arm": arm,
                    "n": part.height,
                    "crps": part[f"{arm}__crps"].mean(),
                }
                for q in ("q01", "q05", "q95", "q99"):
                    r[f"pin_{q}"] = part[f"{arm}__pin_{q}"].mean()
                    r[f"cov_{q}"] = part[f"{arm}__cov_{q}"].mean()
                recs.append(r)
    t = pl.DataFrame(recs)
    t.write_csv(OUT / "tails_by_lead.csv")
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    print(t.with_columns(pl.col(pl.Float64).round(3)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=(
            "build",
            "rows",
            "baseline",
            "stats",
            "tails",
            "curtailment",
            "bands",
            "tails_by_lead",
        ),
    )
    a = p.parse_args()
    if a.command == "bands":
        return bands()
    if a.command == "tails_by_lead":
        return tails_by_lead()
    configure()
    {
        "build": s510.build,
        "rows": s510.score470.rows_command,
        "baseline": s510.baseline,
        "stats": s510.score470.stats_command,
        "tails": s510.tails470.main,
        "curtailment": s510.curtailment470.main,
    }[a.command]()


if __name__ == "__main__":
    main()
