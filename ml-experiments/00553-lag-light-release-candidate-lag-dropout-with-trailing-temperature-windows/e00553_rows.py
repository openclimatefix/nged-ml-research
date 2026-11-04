# ruff: noqa: E501
"""Write the per-row CRPS and tail metrics of the experiment 553 arms.

Copies experiment 550's `e00550_rows.py`. Each arm is pooled over seeds 0 and 1 (the seeds of every
stored comparator); the `-s2` arms are seed 2 alone, the replication. Scorer: experiment 470's, on
experiment 533's `main16` rows, with the 1.15 widening.
"""

from pathlib import Path
from typing import Final

import e00454_build as build454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00511_score as k511  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00553-lag-light-release-candidate-lag-dropout-with-trailing-temperature-windows"
    / "work"
)
SRC511: Final[Path] = k511.WORK_DIR / "predictions"
FIVE: Final[tuple[str, ...]] = build470.MONTHS
RATES: Final[tuple[str, ...]] = ("RDWT-D30", "RDWT-D15")
POOLED: Final[tuple[str, ...]] = tuple(a for r in RATES for a in (r, f"{r}-absent"))
REPLICATION: Final[tuple[str, ...]] = tuple(f"{r}-s2" for r in RATES)
NEW: Final[tuple[str, ...]] = (*POOLED, *REPLICATION)


def main() -> None:
    """Pool the two seeds of each arm, write its rows, and score them next to the stored controls."""
    assert FIVE == ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06"), FIVE
    heads470.WORK_DIR = W
    (W / "predictions").mkdir(exist_ok=True, parents=True)
    for arm in ("A-pool", "K2-S1", "manual_heuristic", "L-1d"):
        link = W / "predictions" / arm
        if not link.exists():
            link.symlink_to((SRC511 / arm).resolve(), target_is_directory=True)
    sites = build470.site_map()
    for m in FIVE:
        _, keys = build454.load_heads(month=m)
        rows = pl.DataFrame({k: keys[k] for k in build470.KEYS})
        for arm in NEW:
            replication = arm.endswith("-s2")
            frame = build470.pool_stored(
                folder=W / "seed_preds" / arm.removesuffix("-s2"),
                month=m,
                sites=sites,
                seeds=(2,) if replication else (0, 1),
            )
            joined = build470.on_rows(rows=rows, frame=frame)
            print(
                m, arm, joined.height, "rows; missing", int(joined["q50"].null_count()), flush=True
            )
            build470.write_arm(arm=arm, month=m, cell=0, frame=joined)
    score470.MONTHS = FIVE
    build454.MONTHS = FIVE
    score470.SCORE_DIR = W / "scores"
    score470.CANDIDATES = NEW
    score470.CONTROLS = ("A-pool", "L-1d", "K2-S1")
    score470.DESCRIPTIVE = ()
    score470.SINGLES = ()
    score470.REFERENCES = ("manual_heuristic",)
    score470.MAIN_ARMS = (*NEW, "A-pool", "L-1d", "K2-S1")
    score470.CELLS = {}
    score470.CONTRASTS = tuple((a, b) for a in NEW for b in ("A-pool", "L-1d"))
    score470.GATING_CRPS = ("eight", "six_clean")
    score470.GATING_Q99 = ("eight", "six_clean")
    score470.TESTS = 84

    def masks(*, wide: pl.DataFrame) -> dict[str, pl.Series]:
        have = pl.all_horizontal([pl.col(f"{a}__q50").is_finite() for a in score470.MAIN_ARMS])
        fr = wide.select(main16=have, gam14=have, ref14=have)
        return {col: fr[col] for col in fr.columns}

    score470.masks = masks
    score470.rows_command()


if __name__ == "__main__":
    main()
