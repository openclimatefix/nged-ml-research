# ruff: noqa: E501
"""Score experiment 556's arms against L-Ww, K2-S1, and A-pool with experiment 470's scorer.

The pattern of experiment 510: experiment 470's scorer, tail script, and curtailment script are pointed
at this folder. Arms are the quantile averages of seeds 0 and 1 of the window forecasts that
`e00511_fit.py assemble` wrote, on experiment 454's rows. L-M, L-W, and L-Ww are experiment 510's stored
seed tables, linked unchanged. The minimum detectable effect is experiment 510's two-seed value
(`mde_k2.csv` of 510), not refitted. Counted tests: 14 contrasts x 2 groups for CRPS (28), and the
same for q99 pinball loss (56 in all).

Commands: `build`, `rows`, `stats`, `tails`, `curtailment`.
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
import e00510_common as c510  # ty: ignore[unresolved-import]
import e00556_fit as fit556
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = fit556.WORK
SEED_PREDS: Final[Path] = WORK_DIR / "seed_preds"
REFITTED: Final[tuple[str, ...]] = ("F3d", "F1d-182", "F3d-E3", "F3d-R")
STORED_510: Final[tuple[str, ...]] = ("L-Ww",)
ARMS: Final[dict[str, tuple[int, ...]]] = dict.fromkeys((*STORED_510, "F1d", *REFITTED), (0, 1))
EXTRA: Final[dict[str, tuple[str, tuple[int, ...]]]] = {}
"""Single-seed report-only arms: name -> (seed-table folder, seeds). Not in any counted contrast."""
CONTROLS: Final[tuple[str, ...]] = ("K2-S1", "A-pool")
REFERENCES: Final[tuple[str, ...]] = ("BG2", "manual_heuristic", "climatology")
TESTS: Final[int] = 48
"""Pre-registered k = 24 counted contrasts x [CRPS, q99 pinball loss] (the code-level count is doubled for the two metrics)."""
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    *[(a, "L-Ww") for a in ("F1d", "F3d", "F1d-182", "F3d-E3", "F3d-R")],
    ("F3d-E3", "F3d"),
    ("F3d-R", "F3d"),
    ("F1d", "K2-S1"),
    ("L-Ww", "K2-S1"),
)
FIVE: Final[tuple[str, ...]] = build470.MONTHS


def configure() -> None:
    """Point experiment 470's scorer, tails, and curtailment scripts at this experiment."""
    heads470.WORK_DIR = WORK_DIR
    score470.MONTHS = FIVE
    curtailment470.MONTHS = FIVE
    build454.MONTHS = FIVE
    score470.SCORE_DIR = WORK_DIR / "scores"
    score470.CANDIDATES = (*ARMS, *EXTRA)
    score470.CONTROLS = CONTROLS
    score470.DESCRIPTIVE = ()
    score470.SINGLES = ()
    score470.REFERENCES = REFERENCES
    score470.MAIN_ARMS = (*ARMS, *EXTRA, *CONTROLS)
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
        return {col: frame[col] for col in frame.columns}

    def mde_510(*, table: pl.DataFrame) -> pl.DataFrame:
        """Return experiment 510's two-seed minimum detectable effect (this experiment does not refit it)."""
        _ = table
        return pl.read_csv(c510.WORK_DIR / "scores" / "mde_k2.csv")

    score470.mde = mde_510  # ty: ignore[invalid-assignment]
    score470.masks = masks  # ty: ignore[invalid-assignment]
    tails470.ARMS = (*ARMS, *EXTRA, *CONTROLS)
    curtailment470.ARMS = (*ARMS, *CONTROLS)
    curtailment470.CANDIDATES = (*ARMS, *CONTROLS)
    curtailment470.REFERENCES = ("manual_heuristic",)
    curtailment470.COMPARISONS = (
        *[(a, "K2-S1") for a in ARMS],
        *[(a, "manual_heuristic") for a in (*ARMS, "K2-S1")],
    )


def link_stored() -> None:
    """Symlink experiment 510's seed tables and experiment 470's stored arms that the scorer reads."""
    source = (
        ML_EXPERIMENTS_DATA_DIR
        / "00470-first-release-demand-forecaster-assembly-and-comparison"
        / "work"
        / "predictions"
    )
    (WORK_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    for arm in (*CONTROLS, *REFERENCES):
        link = WORK_DIR / "predictions" / arm
        if not link.exists():
            link.symlink_to(source / arm, target_is_directory=True)
    SEED_PREDS.mkdir(parents=True, exist_ok=True)
    for arm in STORED_510:
        link = SEED_PREDS / arm
        if not link.exists():
            link.symlink_to(c510.WORK_DIR / "seed_preds" / arm, target_is_directory=True)
    link = SEED_PREDS / "F1d"
    if not link.exists():
        link.symlink_to(fit556.E511 / "seed_preds" / "L-1d", target_is_directory=True)


def build() -> None:
    """Write each arm's pooled seeds on experiment 454's rows."""
    sites = build470.site_map()
    for month in FIVE:
        _, keys = build454.load_heads(month=month)
        rows = pl.DataFrame({key: keys[key] for key in build470.KEYS})
        jobs = {**{a: (a, seeds) for a, seeds in ARMS.items()}, **EXTRA}
        for arm, (source, seeds) in jobs.items():
            folder = SEED_PREDS / source
            if not all((folder / f"{month}_s{s}.parquet").exists() for s in seeds):
                print("skipped", arm, month, "(no forecasts)", flush=True)
                continue
            frame = build470.pool_stored(folder=folder, month=month, sites=sites, seeds=seeds)
            joined = build470.on_rows(rows=rows, frame=frame)
            print(
                "wrote",
                arm,
                month,
                joined.height,
                "rows; missing",
                int(joined["q50"].null_count()),
                flush=True,
            )
            build470.write_arm(arm=arm, month=month, cell=0, frame=joined)


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "rows", "stats", "tails", "curtailment"))
    args = parser.parse_args()
    configure()
    link_stored()
    {
        "build": build,
        "rows": score470.rows_command,
        "stats": score470.stats_command,
        "tails": tails470.main,
        "curtailment": curtailment470.main,
    }[args.command]()


if __name__ == "__main__":
    main()
