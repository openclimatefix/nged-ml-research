# ruff: noqa: ANN001, ANN201, ANN202, D103, E501
"""Write per-row CRPS of the cloud arms A11c and A11c-null (2 seeds) on the 4 folds from 2025-12, into `work/scores_cloud`."""

import e00454_build as build454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00511_score as k511  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W = (
    ML_EXPERIMENTS_DATA_DIR
    / "00559-m0-leak-ladder-three-step-schedule-frozen-encoders-and-penalty"
    / "work"
)
SRC511 = k511.WORK_DIR / "predictions"
FIVE = ("2025-12", "2026-01", "2026-03", "2026-06")
assert set(FIVE) <= set(build470.MONTHS), FIVE
TWO_SEEDS = (0, 1)
ARMS: dict[str, tuple[str, tuple[int, ...]]] = {a: (a, TWO_SEEDS) for a in ("A11c", "A11c-null")}


def ready(*, folder: str, seeds: tuple[int, ...]) -> bool:
    """Return whether every month and seed of an arm has been assembled."""
    return all(
        (W / "seed_preds" / folder / f"{m}_s{s}.parquet").exists() for m in FIVE for s in seeds
    )


NEW = tuple(a for a, (folder, seeds) in ARMS.items() if ready(folder=folder, seeds=seeds))


def main():
    heads470.WORK_DIR = W
    (W / "predictions").mkdir(exist_ok=True, parents=True)
    for arm in ("A-pool", "manual_heuristic", "L-1d"):
        link = W / "predictions" / arm
        if not link.exists():
            link.symlink_to((SRC511 / arm).resolve(), target_is_directory=True)
    sites = build470.site_map()
    for m in FIVE:
        _, keys = build454.load_heads(month=m)
        rows = pl.DataFrame({k: keys[k] for k in build470.KEYS})
        for arm in NEW:
            folder, seeds = ARMS[arm]
            frame = build470.pool_stored(
                folder=W / "seed_preds" / folder, month=m, sites=sites, seeds=seeds
            )
            joined = build470.on_rows(rows=rows, frame=frame)
            print(
                m, arm, joined.height, "rows; missing", int(joined["q50"].null_count()), flush=True
            )
            build470.write_arm(arm=arm, month=m, cell=0, frame=joined)
    score470.MONTHS = FIVE
    build454.MONTHS = FIVE
    score470.SCORE_DIR = W / "scores_cloud"
    score470.CANDIDATES = NEW
    score470.CONTROLS = ("A-pool", "L-1d")
    score470.DESCRIPTIVE = ()
    score470.SINGLES = ()
    score470.REFERENCES = ("manual_heuristic",)
    score470.MAIN_ARMS = (*NEW, "A-pool", "L-1d")
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
