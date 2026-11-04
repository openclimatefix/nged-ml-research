# ruff: noqa: ANN001, ANN202, D103, E501
"""Write per-row CRPS of the issue-time forecasts with 470's scorer on 511's rows (mirrors `e00551_rows.py`).

One scorer run per issue time, in its own folder, so that the rows of each issue time (valid time
after the issue time) are masked on their own arms and not on the strictest issue time.
"""

import os
from pathlib import Path

import e00454_build as build454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00511_score as k511  # ty: ignore[unresolved-import]
import polars as pl

OUT = (
    Path(os.environ["DATA_PATH_INTERNAL"])
    / "ml-experiments/00555-issue-time-lags-at-06-12-18-utc-lag-light-demand-network"
)
SRC511 = k511.WORK_DIR / "predictions"
FIVE = build470.MONTHS
assert FIVE == ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06"), FIVE
ISSUES = ("09", "12", "18", "00", "06")
NEW = ("R-DW", "R-DROP", "R9")


def one_issue(*, tag: str) -> None:
    w = OUT / f"work_i{tag}"
    heads470.WORK_DIR = w
    (w / "predictions").mkdir(exist_ok=True, parents=True)
    for arm in ("A-pool", "K2-S1", "manual_heuristic", "L-1d"):
        link = w / "predictions" / arm
        if not link.exists():
            link.symlink_to((SRC511 / arm).resolve(), target_is_directory=True)
    sites = build470.site_map()
    for m in FIVE:
        _, keys = build454.load_heads(month=m)
        rows = pl.DataFrame({k: keys[k] for k in build470.KEYS})
        for arm in NEW:
            frame = build470.pool_stored(
                folder=OUT / "work/seed_preds" / f"{arm}-i{tag}", month=m, sites=sites, seeds=(0, 1)
            )
            joined = build470.on_rows(rows=rows, frame=frame)
            print(
                tag,
                m,
                arm,
                joined.height,
                "rows; missing",
                int(joined["q50"].null_count()),
                flush=True,
            )
            build470.write_arm(arm=arm, month=m, cell=0, frame=joined)
    score470.MONTHS = FIVE
    build454.MONTHS = FIVE
    score470.SCORE_DIR = w / "scores"
    score470.CANDIDATES = NEW
    score470.CONTROLS = ("A-pool", "L-1d", "K2-S1")
    score470.DESCRIPTIVE = ()
    score470.SINGLES = ()
    score470.REFERENCES = ("manual_heuristic",)
    score470.MAIN_ARMS = (*NEW, "A-pool", "L-1d", "K2-S1")
    score470.CELLS = {}
    score470.CONTRASTS = (("R-DW", "L-1d"),)
    score470.GATING_CRPS = ("eight", "six_clean")
    score470.GATING_Q99 = ("eight", "six_clean")
    score470.TESTS = 32

    def masks(*, wide):
        have = pl.all_horizontal([pl.col(f"{a}__q50").is_finite() for a in score470.MAIN_ARMS])
        fr = wide.select(main16=have, gam14=have, ref14=have)
        return {c: fr[c] for c in fr.columns}

    score470.masks = masks
    score470.rows_command()


def main() -> None:
    for tag in ISSUES:
        one_issue(tag=tag)


main()
