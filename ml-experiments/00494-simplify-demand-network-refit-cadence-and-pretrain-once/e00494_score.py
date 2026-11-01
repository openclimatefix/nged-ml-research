# ruff: noqa: E501
"""Score experiment 494's stale and pretrain-once networks against the fresh two-seed pool (K2).

Uses experiment 470's scorer, tail script, and curtailment script on its rows, pointed at this
experiment's folder (the pattern of experiment 489). K2 is the quantile average of seeds 0 and 1 of
experiment 425's arm A. Each comparison is scored in its own call, on the months the arm has:

| Comparison | Arm | Months |
|---|---|---|
| `Stale3` | networks refitted 3 months earlier | the 5 folds |
| `Stale6` | 6 months earlier | the 5 folds |
| `Stale6_stored` | 6 months earlier | the 3 folds whose refit experiment 425 stored |
| `Stale11` | 11 months earlier (descriptive) | 2026-06 |
| `PO` | fine-tuned from the 2025-07 pretraining | the 5 folds |

Commands: `build`, then per comparison `rows`, `baseline` (minimum detectable effect), `stats`,
`tails`, `curtailment`, each with `--comparison`.

```bash
source <paths>
uv run --no-sync python ml-experiments/00494-*/e00494_score.py build
uv run --no-sync python ml-experiments/00494-*/e00494_score.py rows --comparison Stale3
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
    ML_EXPERIMENTS_DATA_DIR
    / "00494-simplify-demand-network-refit-cadence-and-pretrain-once"
    / "work"
)
E425_SEED_PREDS: Final[Path] = heads470.E425 / "seed_preds"
SEEDS: Final[tuple[int, ...]] = (0, 1)
TESTS: Final[int] = 12
"""Three arms times two groups times CRPS and q99 pinball loss."""
FIVE: Final[tuple[str, ...]] = build470.MONTHS
COMPARISONS: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "Stale3": ("Stale3", FIVE),
    "Stale6": ("Stale6", FIVE),
    "Stale6_stored": ("Stale6", ("2026-01", "2026-03", "2026-06")),
    "Stale11": ("Stale11", ("2026-06",)),
    "PO": ("PO", FIVE),
}
ARM_FOLDERS: Final[dict[str, Path]] = {
    "K2": E425_SEED_PREDS / "A",
    **{a: WORK_DIR / "seed_preds" / a for a in ("Stale3", "Stale6", "Stale11", "PO")},
}


def configure(*, comparison: str) -> None:
    """Point experiment 470's scorer, tails, and curtailment scripts at one comparison."""
    arm, months = COMPARISONS[comparison]
    heads470.WORK_DIR = WORK_DIR
    score470.MONTHS = months
    score470.SCORE_DIR = WORK_DIR / "scores" / comparison
    score470.CANDIDATES = (arm,)
    score470.CONTROLS = ("K2",)
    score470.DESCRIPTIVE = ()
    score470.REFERENCES = ()
    score470.MAIN_ARMS = (arm, "K2")
    score470.CELLS = {}
    score470.CONTRASTS = ((arm, "K2"),)
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
    tails470.ARMS = (arm, "K2")
    curtailment470.ARMS = (arm, "K2")
    curtailment470.CANDIDATES = (arm, "K2")
    curtailment470.REFERENCES = ("manual_heuristic",)
    curtailment470.COMPARISONS = (
        (arm, "K2"),
        (arm, "manual_heuristic"),
        ("K2", "manual_heuristic"),
    )


def link_stored() -> None:
    """Symlink the stored arms of experiment 470 that the scorer reads (singles and references)."""
    source = (
        ML_EXPERIMENTS_DATA_DIR
        / "00470-first-release-demand-forecaster-assembly-and-comparison"
        / "work"
        / "predictions"
    )
    target = WORK_DIR / "predictions"
    target.mkdir(parents=True, exist_ok=True)
    for arm in ("manual_heuristic", "A-pool", *[f"A-s{s}" for s in build470.POOL_SEEDS]):
        link = target / arm
        if not link.exists():
            link.symlink_to(source / arm, target_is_directory=True)


def build() -> None:
    """Write each arm's pooled seeds 0 and 1 on experiment 454's rows (one cell, raw quantiles)."""
    sites = build470.site_map()
    for month in FIVE:
        _, keys = build454.load_heads(month=month)
        rows = pl.DataFrame({k: keys[k] for k in build470.KEYS})
        for arm, folder in ARM_FOLDERS.items():
            if not all((folder / f"{month}_s{s}.parquet").exists() for s in SEEDS):
                print("skipped", arm, month, "(no networks)", flush=True)
                continue
            frame = build470.pool_stored(folder=folder, month=month, sites=sites, seeds=SEEDS)
            joined = build470.on_rows(rows=rows, frame=frame)
            assert joined["q50"].null_count() == 0, f"{arm} {month} misses rows"
            build470.write_arm(arm=arm, month=month, cell=0, frame=joined)
            print("wrote", arm, month, joined.height, flush=True)


def baseline() -> None:
    """Print the minimum detectable effect (experiment 428's definition) before any contrast."""
    score470.rows_command()
    print(score470.mde(table=score470.load_rows()))


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("build", "rows", "baseline", "stats", "tails", "curtailment")
    )
    parser.add_argument("--comparison", choices=list(COMPARISONS), default="Stale3")
    args = parser.parse_args()
    configure(comparison=args.comparison)
    link_stored()
    {
        "build": build,
        "rows": score470.rows_command,
        "baseline": baseline,
        "stats": score470.stats_command,
        "tails": tails470.main,
        "curtailment": curtailment470.main,
    }[args.command]()


if __name__ == "__main__":
    main()
