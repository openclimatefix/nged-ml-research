# ruff: noqa: E501, D103
"""Experiment 586, stage 2: CRPS scoring of the wide-layout arms with experiment 572's scorer (patched, not copied).

Arms: M0, E2 (experiment 565's stored C-M0 and D5), S0 (experiment 572's stored two-lag v3+D5, a
report-only reference), and S0w, S1w, S3w with their lags-masked and lags-aged twins. Counted
contrasts: S1w minus S0w and S3w minus S0w, at the eight MVA Primaries and the six MW Primaries, in
the bands 0-24 h, 24-120 h, and 120-240 h: 2 x 2 x 3 = 12 tests. Extra slices (as experiment 585):
Christmas and New Year, bank-holiday days, and Easter (empty: no fold month contains it).
"""

import sys
from datetime import date
from pathlib import Path

import e00565_design6 as d6  # ty: ignore[unresolved-import]
import e00572_score as sc  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00586-lag-encoder-sharing-and-lag-id-embedding"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
KINDS = ("s0w", "s1w", "s3w")
DATE = pl.lit(date(2019, 9, 1)) + pl.duration(days=pl.col("target") // 48)
MD = DATE.dt.month().cast(pl.Int32) * 100 + DATE.dt.day().cast(pl.Int32)
SLICES = {
    "Christmas and New Year": (MD >= 1224) | (MD <= 102),
    "bank-holiday days": MD.is_in([1225, 1226, 101]),
    "Easter": pl.lit(False),
}


def arms_present() -> dict[str, tuple[Path, str]]:
    out = {}
    for name, (root, tag) in sc.ARM_DIRS.items():
        for label, folder in [
            (name, tag),
            *[(f"{name}-{x}", f"{tag}-{x}") for x in sc.SIBLING_SUFFIXES],
        ]:
            if name in {"M0", "E2"} and label != name:
                continue
            if all(
                (root / "seed_preds" / folder / f"{m}_s{x}.parquet").exists()
                for m in d6.MONTHS
                for x in sc.SEEDS
            ):
                out[label] = (root, folder)
    return out


def main() -> None:
    keep = {"M0": sc.ARM_DIRS["M0"], "E2": sc.ARM_DIRS["E2"], "S0": sc.ARM_DIRS["V3D5"]}
    for k in KINDS:
        keep[k.upper().replace("W", "w")] = (WORK / "arms" / k / "work", "V3D5")
    present = {a for a, (r, t) in keep.items() if (r / "seed_preds" / t).exists()}
    sc.ARM_DIRS = {a: v for a, v in keep.items() if a in present}
    pairs = (("S1w", "S0w"), ("S3w", "S0w"))
    sc.COUNTED = tuple(p for p in pairs if set(p) <= present)
    sc.DESCRIPTIVE = tuple(
        p for p in (("S0w", "S0"), ("S1w", "S0"), ("S3w", "S0"), ("S1w", "S3w"), ("S0w", "E2")) if set(p) <= present
    )  # fmt: skip
    sc.TESTS = 12
    sc.W572 = WORK / "work_wide"
    sc.arms_present = arms_present
    sc.SLICES = {**sc.SLICES, **SLICES}
    sc.main()


if __name__ == "__main__":
    sys.exit(main())
