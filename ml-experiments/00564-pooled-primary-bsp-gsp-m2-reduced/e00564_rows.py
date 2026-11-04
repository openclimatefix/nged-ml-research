# ruff: noqa: ANN001, ANN201, ANN202, D103, E501
"""Write per-row CRPS of the experiment 564 arms with 470's scorer on 511's rows (mirrors 554's e00554_rows).

Arm names carry the seed set: `M2-a`, `M2-b`, `M2-c`, and `M0-2` (experiment 554's M0, seeds 0 and 1
only, the control) average seeds 0 and 1; `M2-a-s0`, `M2-a-s1`, `M0-2-s0`, and `M0-2-s1` are single
seeds (for the minimum detectable effect). The 5-seed M0 of experiment 554 keeps the name `M0`, and
the 5-seed `L-1d5` is a report-only reference. Only the Primaries are scored here, because 511's row
set holds Primaries only; `e00564_nodes.py` scores the 4 extra series.
"""

from pathlib import Path

import e00454_build as build454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00511_score as k511  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W554 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out"
    / "work"
)
W = ML_EXPERIMENTS_DATA_DIR / "00564-pooled-primary-bsp-gsp-m2-reduced" / "work"
SRC511 = k511.WORK_DIR / "predictions"
FIVE = build470.MONTHS
assert FIVE == ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06"), FIVE
TWO_SEEDS = (0, 1)
# arm -> (root of the seed_preds folder, folder name, seeds)
ARMS: dict[str, tuple[Path, str, tuple[int, ...]]] = {
    **{a: (W, a, TWO_SEEDS) for a in ("M2-a", "M2-b", "M2-c")},
    "M0-2": (W554, "M0", TWO_SEEDS),
    **{f"{a}-s{i}": (W if a != "M0-2" else W554, a if a != "M0-2" else "M0", (i,)) for a in ("M2-a", "M2-b", "M2-c", "M0-2") for i in TWO_SEEDS},
    "M0": (W554, "M0", (0, 1, 2, 3, 4)),
    "L-1d5": (W554, "L-1d", (0, 1, 2, 3, 4)),
}


def ready(*, root: Path, folder: str, seeds: tuple[int, ...]) -> bool:
    """Return whether every month and seed of an arm has been assembled."""
    return all(
        (root / "seed_preds" / folder / f"{m}_s{s}.parquet").exists() for m in FIVE for s in seeds
    )


NEW = tuple(a for a, (root, folder, seeds) in ARMS.items() if ready(root=root, folder=folder, seeds=seeds))


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
            root, folder, seeds = ARMS[arm]
            frame = build470.pool_stored(
                folder=root / "seed_preds" / folder, month=m, sites=sites, seeds=seeds
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
    score470.TESTS = 42

    def masks(*, wide):
        have = pl.all_horizontal([pl.col(f"{a}__q50").is_finite() for a in score470.MAIN_ARMS])
        fr = wide.select(main16=have, gam14=have, ref14=have)
        return {c: fr[c] for c in fr.columns}

    score470.masks = masks
    score470.rows_command()


main()
