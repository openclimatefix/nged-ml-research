# ruff: noqa: E501
"""Score experiment 489's students against the pool, using experiment 470's scorer on its rows.

The scorer, the tail script and the curtailment script of experiment 470 read their module-level
settings at call time, so this script points them at this experiment's data folder and arm lists
instead of copying them. `build` puts every student cell on experiment 454's rows (the layout of
experiment 470's `predictions/<arm>/<month>_c<cell>.parquet`, raw quantiles), and symlinks the
stored arms: `A-pool` (the pool, experiment 470), `A-s0` to `A-s4` (single seeds), and `S0`
(`A-s0` to `A-s2` as the three cells of one arm, the single-network reference). Students are
`Dpool50`, `Dpool80`, and `DpoolERA5`, with seeds 0 to 2 as cells 0 to 2.

Commands: `build`, `rows`, `baseline` (minimum detectable effect and the S0-to-pool gap, no student
needed), `stats`, `tails`, `curtailment`.

```bash
source .e489env.sh
uv run --no-sync python ml-experiments/00489-*/e00489_score.py build
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00454_build as build454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_curtailment as curtailment470  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00470_tails as tails470  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00489-distill-the-five-network-pool-into-one-network" / "work"
)
STUDENTS: Final[tuple[str, ...]] = ("Dpool50", "Dpool80", "DpoolERA5")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
TESTS: Final[int] = 27
"""Three students against the pool, times seven CRPS slices and two q99 slices."""
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    *[(s, "A-pool") for s in STUDENTS],
    *[(s, "S0") for s in STUDENTS],
    ("A-pool", "S0"),
)
"""The first three are the counted tests; the rest are descriptive."""


def patch() -> None:
    """Point experiment 470's scorer, tail script, and curtailment script at this experiment."""
    heads470.WORK_DIR = WORK_DIR
    score470.SCORE_DIR = WORK_DIR / "scores"
    score470.CANDIDATES = ("A-pool", *STUDENTS)
    score470.CONTROLS = ("S0",)
    score470.DESCRIPTIVE = ()
    score470.REFERENCES = ()
    score470.MAIN_ARMS = (*score470.CANDIDATES, *score470.CONTROLS)
    score470.CELLS = {"S0": 3, **dict.fromkeys(STUDENTS, 3)}
    score470.CONTRASTS = CONTRASTS
    score470.TESTS = TESTS

    def masks(*, wide: pl.DataFrame) -> dict[str, pl.Series]:
        """Return the row-set masks: every arm present; the same set without the report-only group."""
        have = pl.all_horizontal([pl.col(f"{a}__q50").is_finite() for a in score470.MAIN_ARMS])
        frame = wide.select(
            main16=have,
            gam14=have & (pl.col("group") != "report_only"),
            ref14=have & (pl.col("group") != "report_only"),
        )
        return {c: frame[c] for c in frame.columns}

    score470.masks = masks  # ty: ignore[invalid-assignment]
    tails470.ARMS = (*score470.CANDIDATES, *score470.CONTROLS)
    curtailment470.ARMS = ("A-pool", "S0", *STUDENTS)
    curtailment470.CANDIDATES = ("A-pool", "S0", *STUDENTS)
    curtailment470.REFERENCES = ("manual_heuristic",)
    curtailment470.COMPARISONS = (
        *[(s, "A-pool") for s in STUDENTS],
        *[(s, "S0") for s in STUDENTS],
        ("A-pool", "S0"),
        *[(a, "manual_heuristic") for a in ("A-pool", "S0", *STUDENTS)],
    )


def link_stored() -> None:
    """Symlink the stored arms of experiment 470 that this experiment scores or compares with."""
    source = (
        ML_EXPERIMENTS_DATA_DIR
        / "00470-first-release-demand-forecaster-assembly-and-comparison"
        / "work"
        / "predictions"
    )
    target = WORK_DIR / "predictions"
    target.mkdir(parents=True, exist_ok=True)
    for arm in ("A-pool", "manual_heuristic", *[f"A-s{s}" for s in build470.POOL_SEEDS]):
        link = target / arm
        if not link.exists():
            link.symlink_to(source / arm, target_is_directory=True)
    for month in build470.MONTHS:
        (target / "S0").mkdir(exist_ok=True)
        for cell in SEEDS:
            link = target / "S0" / f"{month}_c{cell}.parquet"
            if not link.exists():
                link.symlink_to(source / f"A-s{cell}" / f"{month}_c0.parquet")


def build_students() -> None:
    """Write each student cell on experiment 454's rows, checking that no row is missing."""
    sites = build470.site_map()
    for month in build470.MONTHS:
        _, keys = build454.load_heads(month=month)
        rows = pl.DataFrame({k: keys[k] for k in build470.KEYS})
        for arm in STUDENTS:
            for seed in SEEDS:
                folder = WORK_DIR / "seed_preds" / arm
                if not (folder / f"{month}_s{seed}.parquet").exists():
                    continue
                frame = build470.pool_stored(folder=folder, month=month, sites=sites, seeds=(seed,))
                joined = build470.on_rows(rows=rows, frame=frame)
                assert joined["q50"].null_count() == 0, f"{arm} {month} s{seed} misses rows"
                build470.write_arm(arm=arm, month=month, cell=seed, frame=joined)
                print("wrote", arm, month, seed, joined.height, flush=True)


def baseline() -> None:
    """Print the minimum detectable effect and the single-seed-to-pool gap (no student score)."""
    score470.CANDIDATES = ("A-pool",)
    score470.MAIN_ARMS = ("A-pool", "S0")
    score470.CELLS = {"S0": 3}
    score470.SCORE_DIR = WORK_DIR / "scores_baseline"
    score470.rows_command()
    table = score470.load_rows()
    print(score470.mde(table=table))
    for name in score470.SLICES:
        part, prefix = score470.slice_filter(table=table, name=name)
        part = part.filter(pl.col("main16"))
        if part.height == 0:
            continue
        pool = part[f"A-pool__{prefix}crps"].mean()
        single = part[f"S0__{prefix}crps"].mean()
        print(f"{name}: S0 {single:.3f}, A-pool {pool:.3f}, gap {single - pool:.3f} points")


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("build", "rows", "baseline", "stats", "tails", "curtailment")
    )
    args = parser.parse_args()
    patch()
    link_stored()
    {
        "build": build_students,
        "rows": score470.rows_command,
        "baseline": baseline,
        "stats": score470.stats_command,
        "tails": tails470.main,
        "curtailment": curtailment470.main,
    }[args.command]()


if __name__ == "__main__":
    main()
