# ruff: noqa: E501, E402, ANN001, ANN201, ANN202, D103
"""Write per-row CRPS of the student S-DW (546) with 470's scorer on 511's rows (mirrors e00539_rows)."""

import e00454_build as build454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00511_score as k511  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W = (
    ML_EXPERIMENTS_DATA_DIR
    / "00546-teacher-student-distillation-into-lag-free-demand-network"
    / "work"
)
E539 = ML_EXPERIMENTS_DATA_DIR / "00539-l1d-plus-day-and-week-lags" / "work"
SRC511 = k511.WORK_DIR / "predictions"
import os

FIVE = tuple(os.environ.get("E546_MONTHS", "2025-10 2025-12 2026-01 2026-03 2026-06").split())
NEW = ("S-DW",)


def main():
    heads470.WORK_DIR = W
    (W / "predictions").mkdir(exist_ok=True, parents=True)
    for arm in ("A-pool", "K2-S1", "manual_heuristic", "L-1d"):
        link = W / "predictions" / arm
        if not link.exists():
            link.symlink_to((SRC511 / arm).resolve(), target_is_directory=True)
    link = W / "predictions" / "R-DW"
    if not link.exists():
        link.symlink_to((E539 / "predictions" / "R-DW").resolve(), target_is_directory=True)
    sites = build470.site_map()
    for m in FIVE:
        _, keys = build454.load_heads(month=m)
        rows = pl.DataFrame({k: keys[k] for k in build470.KEYS})
        for arm in NEW:
            frame = build470.pool_stored(
                folder=W / "seed_preds" / arm, month=m, sites=sites, seeds=(0, 1)
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
    score470.CONTROLS = ("A-pool", "L-1d", "K2-S1", "R-DW")
    score470.DESCRIPTIVE = ()
    score470.SINGLES = ()
    score470.REFERENCES = ("manual_heuristic",)
    score470.MAIN_ARMS = (*NEW, "A-pool", "L-1d", "K2-S1", "R-DW")
    score470.CELLS = {}
    score470.CONTRASTS = tuple((a, b) for a in NEW for b in ("A-pool", "L-1d"))
    score470.GATING_CRPS = ("eight", "six_clean")
    score470.GATING_Q99 = ("eight", "six_clean")
    score470.TESTS = 28

    def masks(*, wide):
        have = pl.all_horizontal([pl.col(f"{a}__q50").is_finite() for a in score470.MAIN_ARMS])
        fr = wide.select(main16=have, gam14=have, ref14=have)
        return {c: fr[c] for c in fr.columns}

    score470.masks = masks
    score470.rows_command()


main()
