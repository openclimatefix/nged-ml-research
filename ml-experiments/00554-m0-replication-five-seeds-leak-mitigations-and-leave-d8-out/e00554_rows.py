# ruff: noqa: ANN001, ANN201, ANN202, D103, E501
"""Write per-row CRPS of the experiment 554 arms with 470's scorer on 511's rows (mirrors 545's e00545_rows).

Arm names carry the seed set: `M0`, `M0lr`, and `L-1d5` average seeds 0 to 4; `M0-3`, `L-1d3`, `M0-sep`,
`M0-mix`, `M0-hl180`, and `M0-c3` average seeds 0 to 2; `L-1d5-s0` to `L-1d5-s4` are single
seeds of L-1d (for the minimum detectable effect). The stored 2-seed L-1d of experiment 511 keeps the name `L-1d`.
"""

import e00454_build as build454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00511_score as k511  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W = (
    ML_EXPERIMENTS_DATA_DIR
    / "00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out"
    / "work"
)
SRC511 = k511.WORK_DIR / "predictions"
FIVE = build470.MONTHS
assert FIVE == ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06"), FIVE
FIVE_SEEDS = (0, 1, 2, 3, 4)
THREE_SEEDS = (0, 1, 2)
ARMS: dict[str, tuple[str, tuple[int, ...]]] = {
    "L-1d5": ("L-1d", FIVE_SEEDS),
    "M0": ("M0", FIVE_SEEDS),
    "M0lr": ("M0lr", FIVE_SEEDS),
    "L-1d3": ("L-1d", THREE_SEEDS),
    "M0-3": ("M0", THREE_SEEDS),
    "M0-sep": ("M0-sep", THREE_SEEDS),
    "M0-mix": ("M0-mix", THREE_SEEDS),
    "M0-hl180": ("M0-hl180", THREE_SEEDS),
    "M0-c3": ("M0-c3", THREE_SEEDS),
    **{f"L-1d5-s{i}": ("L-1d", (i,)) for i in FIVE_SEEDS},
}


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
    score470.SCORE_DIR = W / "scores"
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
