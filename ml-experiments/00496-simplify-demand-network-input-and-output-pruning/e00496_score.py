# ruff: noqa: E501
"""Score experiment 496's pruned demand networks against PretrainOnce (experiment 494's `PO`).

Uses experiment 470's scorer, tail script, and curtailment script on its rows, pointed at this
experiment's folder (the pattern of experiment 494). Each comparison scores one arm against `PO`
(the equivalence contrast) and against K2 (descriptive), both pooled over seeds 0 and 1. All
comparisons use the 5 folds, except `dF`, the fresh-pretraining check, which is also contrasted with
K2 alone. The Bonferroni count is fixed at 20 (5 arms x 2 groups x CRPS and q99 pinball loss).

Commands: `build`, then per comparison `rows`, `baseline` (minimum detectable effect), `stats`,
`tails`, `curtailment`, each with `--comparison`.

```bash
source <paths>
uv run --no-sync python ml-experiments/00496-*/e00496_score.py build
uv run --no-sync python ml-experiments/00496-*/e00496_score.py rows --comparison a
```
"""

import argparse
import math
from pathlib import Path
from typing import Final

import e00454_build as build454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_curtailment as curtailment470  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00470_tails as tails470  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00496-simplify-demand-network-input-and-output-pruning" / "work"
)
E494_SEED_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00494-simplify-demand-network-refit-cadence-and-pretrain-once"
    / "work"
    / "seed_preds"
)
E425_SEED_PREDS: Final[Path] = heads470.E425 / "seed_preds"
SEEDS: Final[tuple[int, ...]] = (0, 1)
TESTS: Final[int] = 20
"""Five arms times two groups times CRPS and q99 pinball loss."""
RESAMPLES: Final[int] = 100_000
"""Bonferroni at 20 tests leaves few tail draws at 20,000 resamples."""
FIVE: Final[tuple[str, ...]] = build470.MONTHS
ARMS: Final[tuple[str, ...]] = ("a", "c", "d", "b", "X", "dF")
COMPARISONS: Final[dict[str, str]] = {a: a for a in ARMS}
ARM_FOLDERS: Final[dict[str, Path]] = {
    "K2": E425_SEED_PREDS / "A",
    "PO": E494_SEED_PREDS / "PO",
    **{a: WORK_DIR / "seed_preds" / a for a in ARMS},
}


def configure(*, comparison: str) -> None:
    """Point experiment 470's scorer, tails, and curtailment scripts at one comparison."""
    arm, months = COMPARISONS[comparison], FIVE
    heads470.WORK_DIR = WORK_DIR
    score470.MONTHS = months
    score470.RESAMPLES = RESAMPLES
    curtailment470.MONTHS = months
    build454.MONTHS = months
    score470.SCORE_DIR = WORK_DIR / "scores" / comparison
    score470.CANDIDATES = (arm,)
    score470.CONTROLS = ("PO", "K2")
    score470.DESCRIPTIVE = ()
    score470.REFERENCES = ()
    score470.MAIN_ARMS = (arm, "PO", "K2")
    score470.CELLS = {}
    score470.CONTRASTS = ((arm, "PO"), (arm, "K2"))
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
    tails470.ARMS = (arm, "PO", "K2")
    curtailment470.ARMS = (arm, "PO", "K2")
    curtailment470.CANDIDATES = (arm, "PO", "K2")
    curtailment470.REFERENCES = ("manual_heuristic",)
    curtailment470.COMPARISONS = (
        (arm, "PO"),
        (arm, "K2"),
        (arm, "manual_heuristic"),
        ("PO", "manual_heuristic"),
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


def mde_two_seeds(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 428's minimum detectable effect for a pool of 2 seeds (k = 2).

    The seed spread is the spread of the single-seed scores `A-s0` and `A-s1` around their fold
    mean; the effect scales as `1.96 * sqrt(2) * sd / sqrt(2)`, the scorer's formula at k = 2.
    """
    records = []
    for label, group in {"eight": "mva_eight", "six_clean": "signed_six"}.items():
        part = table.filter((pl.col("group") == group) & pl.col("main16"))
        if group == "signed_six":
            part = part.filter(pl.col("cls") == "clean")
        values = np.array(
            [
                [part.filter(pl.col("month") == m)[f"A-s{s}__crps"].mean() for s in SEEDS]
                for m in score470.MONTHS
            ]
        )
        resid = values - values.mean(axis=1, keepdims=True)
        sd = math.sqrt((resid**2).sum() / (len(values) * (len(SEEDS) - 1)))
        records.append(
            {"group": label, "folds": len(values), "seed_sd": sd,
             "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(len(SEEDS))}
        )  # fmt: skip
    return pl.DataFrame(records)


def baseline() -> None:
    """Print the two-seed minimum detectable effect before any contrast."""
    if not all((score470.SCORE_DIR / f"rows_{m}.parquet").exists() for m in score470.MONTHS):
        score470.rows_command()
    minimum = mde_two_seeds(table=score470.load_rows())
    score470.SCORE_DIR.mkdir(parents=True, exist_ok=True)
    minimum.write_csv(score470.SCORE_DIR / "mde_k2.csv")
    print(minimum)


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("build", "rows", "baseline", "stats", "tails", "curtailment")
    )
    parser.add_argument("--comparison", choices=list(COMPARISONS), default="a")
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
