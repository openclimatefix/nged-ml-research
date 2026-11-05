# ruff: noqa: ANN001, ANN201, ANN202, D103, E501
"""Write per-row CRPS of the experiment 569 arms with 470's scorer on 511's rows (mirrors 562's e00562_rows).

Arm names are the folder names under `work/seed_preds`; `M0s0` and `M0s1` are the control's seeds 0 and 1.
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
    / "00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1"
    / "work"
)
SRC511 = k511.WORK_DIR / "predictions"
FIVE = build470.MONTHS
REAL = ("B1", "B2", "B3a", "B3b", "B4", "B3c")
ARMS_ALL = ("M0", *REAL, *[f"{a}p" for a in REAL if a != "B3c"])


def seeds_of(*, arm: str) -> tuple[int, ...]:
    """Return the seeds (0 to 4) assembled for every month of an arm."""
    return tuple(
        s
        for s in range(5)
        if all((W / "seed_preds" / arm / f"{m}_s{s}.parquet").exists() for m in FIVE)
    )


ARMS = {a: (W / "seed_preds" / a, seeds_of(arm=a)) for a in ARMS_ALL if seeds_of(arm=a)}
if "M0" in ARMS and 1 in ARMS["M0"][1]:
    ARMS["M0s0"] = (W / "seed_preds" / "M0", (0,))
    ARMS["M0s1"] = (W / "seed_preds" / "M0", (1,))
NEW = tuple(ARMS)


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
            frame = build470.pool_stored(folder=folder, month=m, sites=sites, seeds=seeds)
            joined = build470.on_rows(rows=rows, frame=frame)
            print(
                m,
                arm,
                seeds,
                joined.height,
                "rows; missing",
                int(joined["q50"].null_count()),
                flush=True,
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
