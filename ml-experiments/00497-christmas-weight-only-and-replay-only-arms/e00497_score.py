# ruff: noqa: E501
r"""Score experiment 497's arms (WO, RO) against experiment 425's stored fits, on experiment 470's rows.

Experiment 485's scorer (`e00485_score`) is reused with its arm list, Bonferroni count, folder, and
prediction source redirected to this experiment: the same rows, slices, 1.15 widening, seed-by-seed
scoring, and month-block bootstrap. H0 is experiment 425's arm A seeds 0 to 2 (experiment 470's
stored `A-s0` to `A-s2`).

Commands:

- `rows`: write the row-level metrics of every arm and month;
- `stats`: print the minimum detectable effect first, then absolute scores and contrasts;
- `build`: write each cell (H0, WO, RO; seeds 0 to 2) on experiment 470's rows for the curtailment script;
- `curtailment [--christmas]`: avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the signed-MW six on
  rows with no labelled event, beside `manual_heuristic` and BG2 (`--christmas`: only the 22 to
  31 December 2025 half-hours).

```bash
source .xmasenv.sh
uv run --no-sync python ml-experiments/00497-*/e00497_score.py rows
uv run --no-sync python ml-experiments/00497-*/e00497_score.py stats
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00449_curtailment as c449  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_curtailment as c470  # ty: ignore[unresolved-import]
import e00470_score as s470  # ty: ignore[unresolved-import]
import e00485_fit as fit485  # ty: ignore[unresolved-import]
import e00485_score as score485  # ty: ignore[unresolved-import]
import e00497_fit as fit497  # ty: ignore[unresolved-import]
import polars as pl

ARMS: Final[tuple[str, ...]] = tuple(fit497.ARMS)
SEEDS: Final[tuple[int, ...]] = fit485.SEEDS3
TESTS: Final[int] = 18
"""Two arms times nine tests (experiment 485's count per arm)."""
SCORE_DIR: Final[Path] = fit497.WORK_DIR / "scores"
PRED_DIR: Final[Path] = fit497.WORK_DIR / "work" / "predictions"
REFERENCES: Final[tuple[str, ...]] = ("manual_heuristic",)
LOGICAL_ARMS: Final[tuple[str, ...]] = ("H0", *ARMS)
CELL_ARMS: Final[tuple[str, ...]] = tuple(f"{a}@{s}" for a in LOGICAL_ARMS for s in SEEDS)
"""Each seed cell is costed on its own, then the costs are averaged over seeds."""
COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    *[(a, "H0") for a in ARMS],
    *[(a, "manual_heuristic") for a in LOGICAL_ARMS],
)
ORIGINAL_PATH = build470.prediction_path
WINDOW_FOLD: Final[str] = "2025-12"


def configure() -> None:
    """Point experiment 485's scorer at this experiment's arms, folder, and test count."""
    fit485.OUT_DIR = fit497.OUT_DIR
    score485.ARMS = ARMS
    score485.TESTS = TESTS
    score485.SCORE_DIR = SCORE_DIR


def cell_path(*, arm: str, month: str, cell: int) -> Path:
    """Return where a cell of `arm@seed` is stored; references and others stay in experiment 470's folder."""
    name, _, seed = arm.partition("@")
    if not seed:
        return ORIGINAL_PATH(arm=arm, month=month, cell=cell)
    return PRED_DIR / name / f"{month}_s{seed}.parquet"


def build_command() -> None:
    """Write every cell (H0, WO, RO; seeds 0 to 2) of every fold on experiment 470's rows."""
    configure()
    sites = build470.site_map()
    for month in score485.MONTHS:
        for arm in LOGICAL_ARMS:
            for seed in SEEDS:
                frame = score485.seed_frame(arm=arm, month=month, seed=seed, sites=sites)
                target = cell_path(arm=f"{arm}@{seed}", month=month, cell=0)
                target.parent.mkdir(parents=True, exist_ok=True)
                frame.select(*s470.KEYS, *s470.LEVELS).write_parquet(target)
        print(f"built {month}", flush=True)


def curtailment_command(*, christmas: bool) -> None:
    """Write and print the avoidable-curtailment costs and the bootstrap table."""
    configure()
    build470.prediction_path = cell_path  # ty: ignore[invalid-assignment]
    s470.arm_cells = lambda *, arm: 1  # ty: ignore[invalid-assignment]
    c470.ARMS = CELL_ARMS
    c470.REFERENCES = REFERENCES
    c449.ARMS = LOGICAL_ARMS  # ty: ignore[invalid-assignment]
    c449.REFERENCES = REFERENCES  # ty: ignore[invalid-assignment]
    c449.COMPARISONS = COMPARISONS  # ty: ignore[invalid-assignment]
    c449.RESAMPLES = c470.RESAMPLES  # ty: ignore[invalid-assignment]
    original_base = c470.month_base
    if christmas:
        c470.MONTHS = (WINDOW_FOLD,)

        def christmas_base(*, month: str) -> pl.DataFrame:
            rows = original_base(month=month)
            valid = pl.col("valid_time")
            return rows.filter((valid.dt.month() == 12) & (valid.dt.day() >= 22))

        c470.month_base = christmas_base  # ty: ignore[invalid-assignment]
        c470._BASES.clear()
    cost = c470.build()
    cost = (
        cost.with_columns(arm=pl.col("arm").str.split("@").list.first())
        .group_by([c for c in cost.columns if c not in ("arm", "avoidable_cost_gbp")] + ["arm"])
        .agg(pl.col("avoidable_cost_gbp").mean())
    )
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_christmas" if christmas else ""
    cost.write_parquet(SCORE_DIR / f"curtailment_cells{suffix}.parquet")
    table = c449.bootstrap(cost=cost)
    table.write_csv(SCORE_DIR / f"curtailment_bootstrap{suffix}.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=140):
        print(table.with_columns(pl.col("estimate", "low", "high").round(3)))


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("rows", "stats", "build", "curtailment"))
    parser.add_argument("--christmas", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        build_command()
    elif args.command == "curtailment":
        curtailment_command(christmas=args.christmas)
    else:
        configure()
        {"rows": score485.rows_command, "stats": score485.stats_command}[args.command]()


if __name__ == "__main__":
    main()
