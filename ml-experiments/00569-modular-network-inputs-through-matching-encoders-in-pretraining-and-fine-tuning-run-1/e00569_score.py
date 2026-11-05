# ruff: noqa: ANN201, PLR0912, PLR0915, PERF401, D103, E501
"""Score experiment 569: real minus placebo (and B3c minus B3a) on the slices of addendum 14, part C.

Reads the per-row CRPS that `e00569_rows.py` wrote. Intervals are 5-block month-block bootstraps (near
min-to-max). Bonferroni count: 6 contrasts x 2 groups x 4 keep-rule slices (all rows, cold nights,
Christmas week, high-spread tercile) = 48 for "improves"; the "hurts" test on every slice (10 slices)
uses the 95% interval without a correction. The minimum detectable effect (MDE) of a slice is
1.96 x sqrt(2) x s, with s the fold-by-seed residual sd of plain M0 seeds 0 and 1.
High spread is the top tercile, within Primary, fold, and lead band, of experiment-569 M0's own
q95 minus q05 (a proxy: the stored scores carry no ensemble-spread column).
"""

import json
import math

import e00470_build as build470  # ty: ignore[unresolved-import]
import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W = (
    ML_EXPERIMENTS_DATA_DIR
    / "00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1"
)
K = ["primary", "origin", "target"]
PAIRS = (
    ("B1", "B1p"),
    ("B2", "B2p"),
    ("B3a", "B3ap"),
    ("B3b", "B3bp"),
    ("B4", "B4p"),
    ("B3c", "B3a"),
)
BONFERRONI = 48
KEEP_SLICES = ("all", "cold_night", "christmas_week", "high_spread")
LEAD_BANDS = ("0-6h", "24-48h", "48-72h", "72-120h")


def load(*, arms: list[str]) -> pl.DataFrame:
    base = s.load()
    mine = pl.concat([pl.read_parquet(W / "work/scores" / f"rows_{m}.parquet", columns=[*K, *[f"{a}__{x}" for a in arms for x in ("crps", "nra_crps")]]) for m in s.MONTHS])  # fmt: skip
    return base.join(mine, on=K, how="left", validate="1:1", maintain_order="left").with_columns(
        band=s.band_expr()
    )


def main():
    schema = pl.read_parquet_schema(W / "work/scores" / f"rows_{s.MONTHS[0]}.parquet")
    arms = sorted({a.split("__")[0] for a in schema if a.endswith("__crps")} & {*(x for p in PAIRS for x in p), "M0", "M0s0", "M0s1"})  # fmt: skip
    t = load(arms=arms)
    need = [f"{a}__crps" for a in arms]
    t = t.with_columns(
        main16=pl.col("main16") & pl.all_horizontal([pl.col(c).is_finite() for c in need])
    )
    # high-spread proxy from M0's forecast width
    widths = []
    for m in s.MONTHS:
        p = W / "work/predictions/M0" / f"{m}_cell0.parquet"
        names = build470.LEVELS
        lo = (
            next(n for n in names if n.endswith("05"))
            if any(n.endswith("05") for n in names)
            else names[1]
        )
        hi = (
            next(n for n in names if n.endswith("95"))
            if any(n.endswith("95") for n in names)
            else names[-2]
        )
        widths.append(pl.read_parquet(p).select(*K, width=pl.col(hi) - pl.col(lo)))
    t = t.join(pl.concat(widths), on=K, how="left", maintain_order="left")
    t = t.with_columns(thr=pl.col("width").quantile(2 / 3).over("primary", "month", "band")).with_columns(high_spread=pl.col("width") >= pl.col("thr"))  # fmt: skip
    slices = {"all": pl.lit(True), "cold_night": pl.col("cold"), "christmas_week": pl.col("xmas"), "high_spread": pl.col("high_spread"),
              "in_event_vs_nra": pl.col("tgt_real"), "no_labelled_event": ~pl.col("tgt_real"), **{f"lead_{b}": pl.col("band") == b for b in LEAD_BANDS}}  # fmt: skip
    rng = np.random.default_rng(0)
    absolute, contrasts, mdes = [], [], []
    for group in ("eight", "six"):
        part = s.row_set(table=t, name=group)
        for scope, lead in (
            ("24-120h", (pl.col("lead_hours") >= 24) & (pl.col("lead_hours") <= 120)),
            ("all_leads", pl.lit(True)),
        ):
            for name, flt in slices.items():
                if scope == "24-120h" and name.startswith("lead_"):
                    continue
                if scope == "all_leads" and name != "all":
                    continue
                sub = part.filter(flt & lead)
                if not sub.height:
                    continue
                col = "nra_crps" if name == "in_event_vs_nra" else "crps"
                for arm in arms:
                    absolute.append({"set": group, "scope": scope, "slice": name, "arm": arm, "n": sub.height, "crps": float(sub[f"{arm}__{col}"].mean())})  # fmt: skip
                if "M0s0" in arms and "M0s1" in arms:
                    mm = (
                        sub.group_by("month")
                        .agg(pl.col(f"M0s0__{col}").mean(), pl.col(f"M0s1__{col}").mean())
                        .sort("month")
                    )
                    v = mm.select(f"M0s0__{col}", f"M0s1__{col}").to_numpy()
                    f_, k_ = v.shape
                    sd = math.sqrt(
                        float(((v - v.mean(axis=1, keepdims=True)) ** 2).sum()) / (f_ * (k_ - 1))
                    )
                    mde = 1.96 * math.sqrt(2) * sd
                    mdes.append(
                        {"set": group, "scope": scope, "slice": name, "mde": mde, "months": f_}
                    )
                    mde_value = mde
                else:
                    mde_value = float("nan")
                for real, ref in PAIRS:
                    if real not in arms or ref not in arms:
                        continue
                    for a, b, kind in (
                        (real, ref, "primary"),
                        (real, "M0", "vs_M0"),
                        (ref, "M0", "ref_vs_M0"),
                    ):
                        if kind != "primary" and real == "B3c":
                            continue
                        d = sub.select(
                            "month", "day", d=pl.col(f"{a}__{col}") - pl.col(f"{b}__{col}")
                        )
                        st = s.block_stats(diff=d, rng=rng, tests=BONFERRONI)
                        contrasts.append({"set": group, "scope": scope, "slice": name, "contrast": f"{a} - {b}", "kind": kind, "mde": mde_value, **st})  # fmt: skip
    pl.DataFrame(absolute).write_csv(W / "scores_absolute.csv")
    out = pl.DataFrame(contrasts)
    out = out.with_columns(
        improves=(pl.col("kind") == "primary")
        & (pl.col("difference") <= -pl.col("mde"))
        & (pl.col("high_bonferroni_month") < 0),
        hurts=(pl.col("kind") == "primary")
        & (pl.col("difference") > pl.max_horizontal(pl.lit(0.03), pl.col("mde")))
        & ((pl.col("low_month") > 0) | (pl.col("folds_positive") >= 4)),
    )
    out.write_csv(W / "scores_contrasts.csv")
    pl.DataFrame(mdes).write_csv(W / "mde.csv")
    json.dump({"arms": arms, "bonferroni": BONFERRONI}, open(W / "scores_arms.json", "w"))  # noqa: SIM115, PTH123
    with pl.Config(tbl_rows=200, tbl_width_chars=220, fmt_str_lengths=60):
        print(out.filter(pl.col("kind") == "primary").select("set", "scope", "slice", "contrast", "difference", "low_bonferroni_month", "high_bonferroni_month", "mde", "folds_positive", "improves", "hurts"))  # fmt: skip
        print(pl.DataFrame(absolute).filter((pl.col("slice") == "all") & (pl.col("scope") == "24-120h")).pivot(on="arm", index="set", values="crps"))  # fmt: skip


main()
