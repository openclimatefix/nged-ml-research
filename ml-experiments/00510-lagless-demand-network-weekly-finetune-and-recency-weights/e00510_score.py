# ruff: noqa: E501
"""Score experiment 510's arms against K2-S1 and A-pool with experiment 470's scorer.

The pattern of experiment 494: experiment 470's scorer, tail script, and curtailment script are
pointed at this folder. Arms are the quantile averages of seeds 0 and 1 (L-M, L-W, L-Ww) or seed 0
(L-Ww-D, L-Ww-O) of the refit forecasts that `e00510_fit.py assemble` wrote, on experiment 454's rows;
single seeds of L-W and L-Ww are also written for the minimum detectable effect. Counted tests: 14
contrasts at the eight and the signed-MW six, for CRPS and q99 pinball loss (56 tests).

Commands: `build`, `rows`, `baseline` (minimum detectable effect, before any contrast), `stats`,
`tails`, `curtailment`.

```bash
source <paths>
uv run --no-sync python ml-experiments/00510-*/e00510_score.py build
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
    ML_EXPERIMENTS_DATA_DIR
    / "00510-lagless-demand-network-weekly-finetune-and-recency-weights"
    / "work"
)
SEED_PREDS: Final[Path] = WORK_DIR / "seed_preds"
ARMS: Final[dict[str, tuple[int, ...]]] = {
    "L-M": (0, 1), "L-W": (0, 1), "L-Ww": (0, 1), "L-Ww-D": (0,), "L-Ww-O": (0,),
}  # fmt: skip
SINGLES: Final[tuple[str, ...]] = ("L-W-s0", "L-W-s1", "L-Ww-s0", "L-Ww-s1")
CONTROLS: Final[tuple[str, ...]] = ("K2-S1", "A-pool")
REFERENCES: Final[tuple[str, ...]] = ("BG2", "manual_heuristic", "climatology")
TESTS: Final[int] = 56
"""14 contrasts x 2 groups x [CRPS, q99 pinball loss]."""
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    *[(a, b) for a in ARMS for b in CONTROLS],
    ("L-W", "L-M"), ("L-Ww", "L-W"), ("L-Ww-D", "L-Ww"), ("L-Ww-O", "L-Ww"),
)  # fmt: skip
FIVE: Final[tuple[str, ...]] = build470.MONTHS


def configure() -> None:
    """Point experiment 470's scorer, tails, and curtailment scripts at this experiment."""
    heads470.WORK_DIR = WORK_DIR
    score470.MONTHS = FIVE
    curtailment470.MONTHS = FIVE
    build454.MONTHS = FIVE
    score470.SCORE_DIR = WORK_DIR / "scores"
    score470.CANDIDATES = tuple(ARMS)
    score470.CONTROLS = CONTROLS
    score470.DESCRIPTIVE = ()
    score470.SINGLES = SINGLES
    score470.REFERENCES = REFERENCES
    score470.MAIN_ARMS = (*ARMS, *CONTROLS)
    score470.CELLS = {}
    score470.CONTRASTS = CONTRASTS
    score470.GATING_CRPS = ("eight", "six_clean")
    score470.GATING_Q99 = ("eight", "six_clean")
    score470.TESTS = TESTS

    def masks(*, wide: pl.DataFrame) -> dict[str, pl.Series]:
        """Return the row sets: every main arm present; the same without the report-only group; plus references."""
        have = pl.all_horizontal([pl.col(f"{a}__q50").is_finite() for a in score470.MAIN_ARMS])
        refs = pl.all_horizontal([pl.col(f"{a}__q50").is_finite() for a in REFERENCES])
        frame = wide.select(
            main16=have,
            gam14=have & (pl.col("group") != "report_only"),
            ref14=have & refs & (pl.col("group") != "report_only"),
        )
        return {c: frame[c] for c in frame.columns}

    score470.masks = masks  # ty: ignore[invalid-assignment]
    tails470.ARMS = (*ARMS, *CONTROLS)
    curtailment470.ARMS = (*ARMS, *CONTROLS)
    curtailment470.CANDIDATES = (*ARMS, *CONTROLS)
    curtailment470.REFERENCES = ("manual_heuristic",)
    curtailment470.COMPARISONS = (
        *[(a, "K2-S1") for a in ARMS],
        *[(a, "manual_heuristic") for a in (*ARMS, "K2-S1")],
    )


def link_stored() -> None:
    """Symlink the stored arms of experiment 470 that the scorer reads (K2-S1, A-pool, references)."""
    source = (
        ML_EXPERIMENTS_DATA_DIR
        / "00470-first-release-demand-forecaster-assembly-and-comparison"
        / "work"
        / "predictions"
    )
    target = WORK_DIR / "predictions"
    target.mkdir(parents=True, exist_ok=True)
    for arm in (*CONTROLS, *REFERENCES):
        link = target / arm
        if not link.exists():
            link.symlink_to(source / arm, target_is_directory=True)


def build() -> None:
    """Write each arm's pooled seeds (and single seeds) on experiment 454's rows."""
    sites = build470.site_map()
    jobs = {arm: (arm, seeds) for arm, seeds in ARMS.items()}
    for single in SINGLES:
        arm, seed = single.rsplit("-s", 1)
        jobs[single] = (arm, (int(seed),))
    for month in FIVE:
        _, keys = build454.load_heads(month=month)
        rows = pl.DataFrame({k: keys[k] for k in build470.KEYS})
        for name, (arm, seeds) in jobs.items():
            folder = SEED_PREDS / arm
            if not all((folder / f"{month}_s{s}.parquet").exists() for s in seeds):
                print("skipped", name, month, "(no forecasts)", flush=True)
                continue
            frame = build470.pool_stored(folder=folder, month=month, sites=sites, seeds=seeds)
            joined = build470.on_rows(rows=rows, frame=frame)
            n_missing = int(joined["q50"].null_count())
            print("wrote", name, month, joined.height, "rows; missing", n_missing, flush=True)
            build470.write_arm(arm=name, month=month, cell=0, frame=joined)


def mde_two_seeds(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 428's minimum detectable effect for a pool of 2 seeds, from L-W and L-Ww seeds.

    The seed spread is the spread of the single-seed scores of L-W and L-Ww around their fold mean;
    the effect scales as `1.96 * sqrt(2) * sd / sqrt(2)`, the scorer's formula at k = 2.
    """
    records = []
    for label, group in {"eight": "mva_eight", "six_clean": "signed_six"}.items():
        part = table.filter((pl.col("group") == group) & pl.col("main16"))
        if group == "signed_six":
            part = part.filter(pl.col("cls") == "clean")
        residuals = []
        for arm in ("L-W", "L-Ww"):
            values = np.array(
                [
                    [part.filter(pl.col("month") == m)[f"{arm}-s{s}__crps"].mean() for s in (0, 1)]
                    for m in FIVE
                ]
            )
            residuals.append(values - values.mean(axis=1, keepdims=True))
        resid = np.concatenate(residuals)
        sd = math.sqrt((resid**2).sum() / (len(resid) * (2 - 1)))
        records.append(
            {"group": label, "folds": len(FIVE), "seed_sd": sd, "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(2)}
        )  # fmt: skip
    return pl.DataFrame(records)


def baseline() -> None:
    """Print the two-seed minimum detectable effect before any contrast."""
    if not all((score470.SCORE_DIR / f"rows_{m}.parquet").exists() for m in FIVE):
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
    args = parser.parse_args()
    configure()
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
