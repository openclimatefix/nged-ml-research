# ruff: noqa: E501
"""Score experiment 514's arms with experiment 470's scorer, pointed at this folder.

The pattern of experiment 510: arms are the quantile averages of seeds 0, 1, and 2 of the forecasts
that `e00514_fit.py` wrote under `seed_preds/<arm>/`; single seeds are also written, for the minimum
detectable effect. Rows are experiment 454's scored rows, restricted to the half-hours where every
arm has a forecast. Contrasts: C1 minus C0, C2 minus C0, and C2 minus C1 at two groups (6 tests); the
Bonferroni count is the family of the whole price-and-lockdown brief, 18.

Commands: `build`, `rows`, `baseline` (minimum detectable effect, before any contrast), `stats`.
"""

import argparse
import math
from pathlib import Path
from typing import Final

import e00454_build as build454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00514-lagless-demand-network-lockdown-flag-removal" / "work"
)
SEED_PREDS: Final[Path] = WORK_DIR / "seed_preds"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
ARMS: Final[dict[str, tuple[int, ...]]] = dict.fromkeys(("C0", "C1", "C2"), SEEDS)
SINGLES: Final[tuple[str, ...]] = tuple(f"{a}-s{s}" for a in ARMS for s in SEEDS)
TESTS: Final[int] = 18
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (("C1", "C0"), ("C2", "C0"), ("C2", "C1"))
FIVE: Final[tuple[str, ...]] = build470.MONTHS


def configure() -> None:
    """Point experiment 470's scorer at this experiment (no stored controls or references)."""
    heads470.WORK_DIR = WORK_DIR
    score470.MONTHS = FIVE
    build454.MONTHS = FIVE
    score470.SCORE_DIR = WORK_DIR / "scores"
    score470.CANDIDATES = tuple(ARMS)
    score470.CONTROLS = ()
    score470.DESCRIPTIVE = ()
    score470.SINGLES = SINGLES
    score470.REFERENCES = ()
    score470.MAIN_ARMS = tuple(ARMS)
    score470.CELLS = {}
    score470.CONTRASTS = CONTRASTS
    score470.GATING_CRPS = ("eight", "six_clean")
    score470.GATING_Q99 = ()
    score470.TESTS = TESTS
    score470.mde = mde_three_seeds  # ty: ignore[invalid-assignment]
    score470.masks = masks  # ty: ignore[invalid-assignment]


def masks(*, wide: pl.DataFrame) -> dict[str, pl.Series]:
    """Return the row sets: every arm present; the same without the report-only group."""
    have = pl.all_horizontal([pl.col(f"{a}__q50").is_finite() for a in ARMS])
    frame = wide.select(
        main16=have,
        gam14=have & (pl.col("group") != "report_only"),
        ref14=have & (pl.col("group") != "report_only"),
    )
    return {c: frame[c] for c in frame.columns}


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
            print(
                "wrote",
                name,
                month,
                joined.height,
                "rows; missing",
                int(joined["q50"].null_count()),
                flush=True,
            )
            build470.write_arm(arm=name, month=month, cell=0, frame=joined)


def mde_three_seeds(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return the minimum detectable effect of a 3-seed mean, from the seed spread of all three arms.

    The spread is that of the single-seed scores around each arm's fold mean, over five folds and
    three arms; the effect is `1.96 * sqrt(2) * sd / sqrt(3)`, the scorer's formula at k = 3.
    """
    records = []
    for label, group in {"eight": "mva_eight", "six_clean": "signed_six"}.items():
        part = table.filter((pl.col("group") == group) & pl.col("main16"))
        if group == "signed_six":
            part = part.filter(pl.col("cls") == "clean")
        residuals = []
        for arm in ARMS:
            values = np.array(
                [
                    [part.filter(pl.col("month") == m)[f"{arm}-s{s}__crps"].mean() for s in SEEDS]
                    for m in FIVE
                ]
            )
            residuals.append(values - values.mean(axis=1, keepdims=True))
        resid = np.concatenate(residuals)
        sd = math.sqrt((resid**2).sum() / (len(resid) * (len(SEEDS) - 1)))
        records.append({"group": label, "folds": len(FIVE), "seed_sd": sd, "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(len(SEEDS))})  # fmt: skip
    return pl.DataFrame(records)


def baseline() -> None:
    """Print the 3-seed minimum detectable effect before any contrast."""
    if not all((score470.SCORE_DIR / f"rows_{m}.parquet").exists() for m in FIVE):
        score470.rows_command()
    minimum = mde_three_seeds(table=score470.load_rows())
    score470.SCORE_DIR.mkdir(parents=True, exist_ok=True)
    minimum.write_csv(score470.SCORE_DIR / "mde.csv")
    print(minimum)


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "rows", "baseline", "stats"))
    args = parser.parse_args()
    configure()
    {
        "build": build,
        "rows": score470.rows_command,
        "baseline": baseline,
        "stats": score470.stats_command,
    }[args.command]()


if __name__ == "__main__":
    main()
