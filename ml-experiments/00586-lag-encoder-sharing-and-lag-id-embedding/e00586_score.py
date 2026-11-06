# ruff: noqa: E501, D103
"""Experiment 586: CRPS scoring of the lag-wiring arms with experiment 572's scorer (patched, not copied).

Arms: M0 and E2 (experiment 565's stored C-M0 and D5), S0 (experiment 572's stored v3+D5, the
control), and this experiment's S1 to S4, each with its lags-masked and lags-aged twins. Counted
contrasts: S1 minus S2, S1 minus S3, S1 minus S0, and S4 minus S1, at the eight MVA Primaries and the
six MW Primaries, bands 0-24 h, 24-120 h, and 120-240 h: 4 x 2 x 3 = 24 tests. The rest are
descriptive.
"""

import sys
from pathlib import Path

import e00565_design6 as d6  # ty: ignore[unresolved-import]
import e00572_score as sc  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00586-lag-encoder-sharing-and-lag-id-embedding"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
KINDS = ("s1", "s2", "s3", "s4")


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
    keep = {
        "M0": sc.ARM_DIRS["M0"],
        "E2": sc.ARM_DIRS["E2"],
        "S0": sc.ARM_DIRS["V3D5"],
    }
    for k in KINDS:
        keep[k.upper()] = (WORK / "arms" / k / "work", "V3D5")
    sc.ARM_DIRS = keep
    pairs = (("S1", "S2"), ("S1", "S3"), ("S1", "S0"), ("S4", "S1"))
    sc.COUNTED = pairs
    sc.DESCRIPTIVE = (
        ("S2", "S0"), ("S3", "S0"), ("S4", "S0"), ("S4", "S2"), ("S3", "S2"), ("S1", "E2"), ("S0", "E2"),
    )  # fmt: skip
    sc.TESTS = 24
    sc.W572 = WORK / "work"
    sc.arms_present = arms_present
    sc.main()


if __name__ == "__main__":
    sys.exit(main())
