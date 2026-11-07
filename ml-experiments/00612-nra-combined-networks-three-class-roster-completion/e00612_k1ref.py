# ruff: noqa: E501, D103
"""Experiment 612: K1-ref gate. Re-score A1 + G4 from experiment 597's stored forecasts and compare.

K1-ref is D5, 3-day embargo, no tail constants: exactly experiment 591's A1 with experiment 592's G4
band. This script runs experiment 597's scoring unchanged, writing to this experiment's folder, and
prints the day-4-to-end coverage (target 0.925, tolerance 0.003).

Usage: python e00612_k1ref.py
"""

import e00597_common as k597  # ty: ignore[unresolved-import]
import e00597_score as s597  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00612-nra-combined-networks-three-class-roster-completion"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
k597.WORK = WORK / "k1ref"
STORED = 0.925
TOLERANCE = 0.003


def main() -> None:
    s597.main("a1")
    tab = pl.read_csv(k597.WORK / "a1" / "bank_summary.csv")
    row = tab.filter(
        (pl.col("group") == "both groups")
        & (pl.col("length_class") == "15 and 30 days")
        & (pl.col("phase") == "days4-end")
        & (pl.col("band") == s597.BANDS[1])
        & (pl.col("system") == "G4")
    )
    print(row.select("cover90", "cover90_lo", "cover90_hi", "n_events"))
    got = float(row["cover90"][0])
    verdict = abs(got - STORED) <= TOLERANCE
    print(
        f"K1-ref day-4-to-end coverage {got:.4f} vs stored {STORED}: {'PASS' if verdict else 'FAIL'}"
    )
    (WORK / "k1ref" / "k1ref_gate.txt").write_text(f"{got:.4f} {'PASS' if verdict else 'FAIL'}\n")


if __name__ == "__main__":
    main()
