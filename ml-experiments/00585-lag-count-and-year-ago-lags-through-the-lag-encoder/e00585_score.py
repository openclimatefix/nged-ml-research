# ruff: noqa: E501, D103, ANN001, ANN202
"""Experiment 585: CRPS scoring of the lag-count and year-ago arms with experiment 572's scorer (patched, not copied).

Arms: M0, E2 (experiment 565's stored C-M0 and D5), V3 and V3D5 (experiment 572's stored forecasts;
V3D5 is the control N2), N5 (experiment 581's stored X3-med), and this experiment's N1, N3, Y1, Y1c,
and Y3. Counted contrasts: each of N1, N3, N5, Y1, Y1c, and Y3 against V3D5 at the eight MVA
Primaries and the six MW Primaries in the bands 0-24 h, 24-120 h, and 120-240 h (36 tests). Extra
slices: Christmas and New Year, bank-holiday days, and Easter (empty: no fold month contains it).
The scorer drops a row from `main16` when any arm lacks it, so this wrapper first asserts that every
arm has the same finite rows.
"""

import sys
from datetime import date
from pathlib import Path

import e00565_design6 as d6  # ty: ignore[unresolved-import]
import e00572_score as sc  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00585-lag-count-and-year-ago-lags-through-the-lag-encoder"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
W581 = ML_EXPERIMENTS_DATA_DIR / "00581-robust-multiweek-lag-encoder-x3/arms/x3med/work"
ARMS = {"n1": "N1", "n3": "N3", "y1": "Y1", "y1c": "Y1c", "y3": "Y3"}
COUNTED_ARMS = ("N1", "N3", "N5", "Y1", "Y1c", "Y3")
TESTS = 36

# `target` is a half-hour index on the 30-minute UTC grid that starts on 2019-09-01.
DATE = pl.lit(date(2019, 9, 1)) + pl.duration(days=pl.col("target") // 48)
MD = DATE.dt.month() * 100 + DATE.dt.day()
SLICES = {
    "Christmas and New Year": (MD >= 1224) | (MD <= 102),
    "bank-holiday days": MD.is_in([1225, 1226, 101]),
    "Easter": pl.lit(False),
}


def arms_present() -> dict[str, tuple[Path, str]]:
    out = {}
    for name, (root, tag) in sc.ARM_DIRS.items():
        variants = [(name, tag)]
        if name == "V3D5":
            variants += [(f"{name}-{x}", f"{tag}-{x}") for x in sc.SIBLING_SUFFIXES]
        for label, folder in variants:
            if all(
                (root / "seed_preds" / folder / f"{m}_s{x}.parquet").exists()
                for m in d6.MONTHS
                for x in sc.SEEDS
            ):
                out[label] = (root, folder)
    return out


def main() -> None:
    present = [k for k in ARMS if (WORK / "arms" / k / "work" / "seed_preds" / "V3D5").exists()]
    keep = {a: v for a, v in sc.ARM_DIRS.items() if a in {"M0", "E2", "V3", "V3D5"}}
    keep["N5"] = (W581, "V3D5")
    for k in present:
        keep[ARMS[k]] = (WORK / "arms" / k / "work", "V3D5")
    sc.ARM_DIRS = keep
    counted_arms = [a for a in COUNTED_ARMS if a in keep]
    sc.COUNTED = tuple((a, "V3D5") for a in counted_arms)
    sc.DESCRIPTIVE = tuple((a, "E2") for a in counted_arms)
    sc.TESTS = TESTS
    sc.W572 = WORK
    sc.arms_present = arms_present
    sc.SLICES = {**sc.SLICES, **SLICES}
    build = sc.build_table

    def checked(*, arms):
        t = build(arms=arms)
        main_arms = [a for a in arms if "-" not in a]
        masks = {
            a: pl.all_horizontal([pl.col(f"{a}_s{x}").is_finite() for x in sc.SEEDS])
            for a in main_arms
        }
        ref = t.select(m=masks["V3D5"])["m"]
        for a, m in masks.items():
            mine = t.select(m=m)["m"]
            if a in {"M0", "E2"}:
                print(f"rows finite for {a}: {int(mine.sum())}; control: {int(ref.sum())}")
            else:
                assert (mine == ref).all(), f"arm {a} does not cover the same rows as V3D5"
        return t

    sc.build_table = checked
    sc.main()


if __name__ == "__main__":
    sys.exit(main())
