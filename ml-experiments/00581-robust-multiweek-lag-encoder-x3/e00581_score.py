# ruff: noqa: E501, D103
"""Experiment 581: CRPS scoring of the X3 arms with experiment 572's scorer (patched, not copied).

Arms: M0, E2 (experiment 565's stored C-M0 and D5), V3 and V3D5 (experiment 572's stored forecasts),
and this experiment's arms with their lags-masked and lags-aged twins. Counted contrasts: each X3 arm
against V3D5, the eight MVA Primaries and the six MW Primaries, bands 0-24 h and 24-120 h (12 tests
with three X3 arms; 2 groups x 2 bands x the number of counted arms otherwise).
"""

import sys
from pathlib import Path

import e00565_design6 as d6  # ty: ignore[unresolved-import]
import e00572_score as sc  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00581-robust-multiweek-lag-encoder-x3"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
ARMS = {
    "x3med": "X3med",
    "placebo": "Placebo",
    "x335": "X335",
    "x3set": "X3set",
    "x3syn": "X3syn",
    "v3syn": "V3D5syn",
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
    present_kinds = [
        k for k in ARMS if (WORK / "arms" / k / "work" / "seed_preds" / "V3D5").exists()
    ]
    keep = {a: v for a, v in sc.ARM_DIRS.items() if a in {"M0", "E2", "V3", "V3D5"}}
    for kind in present_kinds:
        keep[ARMS[kind]] = (WORK / "arms" / kind / "work", "V3D5")
    sc.ARM_DIRS = keep
    counted = tuple((ARMS[k], "V3D5") for k in present_kinds if k in {"x3med", "x3set", "x3syn"})
    sc.COUNTED = counted
    sc.DESCRIPTIVE = tuple(
        pair
        for k in present_kinds
        for pair in ((ARMS[k], "V3D5"), (ARMS[k], "E2"))
        if pair not in counted
    )
    sc.TESTS = 12
    sc.W572 = WORK / "work"
    sc.arms_present = arms_present
    sc.main()


if __name__ == "__main__":
    sys.exit(main())
