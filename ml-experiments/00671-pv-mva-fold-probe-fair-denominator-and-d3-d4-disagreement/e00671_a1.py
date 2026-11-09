# ruff: noqa: E501
"""Experiment 671, Part 1: A1' with the linear probe's signed-world recovery as the denominator.

Reads experiment 662's stored recoveries (no network is re-probed) and writes `a1_prime.csv`.
A1' = median-over-seeds PF recovery of probe X / median-over-seeds signed-world recovery of L0, per
MW Primary and level, kept where |denominator| >= 0.05.
"""

import json
from pathlib import Path
from typing import Final

import polars as pl

SRC: Final = Path(
    "/home/jack/dev/nged-substation-forecast/data/ml-experiments/00662-pv-probe-through-the-mva-fold-matched-level-redesign/work/analysis/recoveries_2026-06.csv"
)
OUT: Final = Path(
    "/home/jack/dev/nged-substation-forecast/data/ml-experiments/00671-pv-mva-fold-probe-fair-denominator-and-d3-d4-disagreement/work"
)
SIX: Final = ("D5", "D9", "D10", "D11", "D14", "D16")
LEVELS: Final = {"l05": 0.5, "l1": 1.0, "l2": 2.0}


def main() -> None:
    """Compute A1' and the bias reading and print them."""
    OUT.mkdir(parents=True, exist_ok=True)
    rec = pl.read_csv(SRC).filter(pl.col("site").is_in(SIX))
    med = rec.group_by(["condition", "method", "site"]).agg(r=pl.col("rec").median())
    rows = []
    for lv, level in LEVELS.items():
        den = med.filter((pl.col("condition") == f"sig_fleet_{lv}_r0") & (pl.col("method") == "L0"))
        d = dict(zip(den["site"], den["r"], strict=True))
        for meth in ("L0", "L1", "FA"):
            num = med.filter(
                (pl.col("condition") == f"pf_fleet_{lv}_r0") & (pl.col("method") == meth)
            )
            for s, r in zip(num["site"], num["r"], strict=True):
                rows.append(
                    {
                        "level": level,
                        "method": meth,
                        "site": s,
                        "pf": r,
                        "signed_L0": d[s],
                        "ratio": r / d[s] if abs(d[s]) >= 0.05 else None,
                    }
                )
    df = pl.DataFrame(rows).sort(["method", "level", "site"])
    df.write_csv(OUT / "a1_prime.csv")
    summary = {}
    for meth in ("L0", "L1", "FA"):
        for level in LEVELS.values():
            sub = df.filter((pl.col("method") == meth) & (pl.col("level") == level))
            ok = sub.filter(pl.col("ratio").is_between(0.9, 1.1))
            summary[f"{meth}@{level}"] = {
                "n_in_bar": ok.height,
                "n_valid": sub["ratio"].drop_nulls().len(),
                "pass": ok.height >= 5,
                "median_ratio": sub["ratio"].median(),
                "ratios": dict(
                    zip(
                        sub["site"],
                        [None if x is None else round(x, 2) for x in sub["ratio"]],
                        strict=True,
                    )
                ),
            }
    (OUT / "a1_prime_summary.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        print(k, "PASS" if v["pass"] else "fail", v["n_in_bar"], "of", v["n_valid"], v["ratios"])


if __name__ == "__main__":
    main()


def loss_table() -> None:
    """Print PF recovery minus signed-world L0 recovery, in percentage points, per Primary, probe, and level."""
    df = pl.read_csv(OUT / "a1_prime.csv").with_columns(
        pp=(pl.col("pf") - pl.col("signed_L0")) * 100
    )
    for meth in ("L0", "L1", "FA"):
        wide = (
            df.filter(pl.col("method") == meth)
            .pivot(on="level", index="site", values="pp")
            .sort("site")
        )
        print(meth, "loss in pp (PF minus signed L0)\n", wide)
    print(
        df.filter(pl.col("method") == "L0")
        .select("level", "site", "signed_L0")
        .sort("site", "level")
    )
    df.write_csv(OUT / "a1_prime_with_pp.csv")


if __name__ == "__main__":
    loss_table()
