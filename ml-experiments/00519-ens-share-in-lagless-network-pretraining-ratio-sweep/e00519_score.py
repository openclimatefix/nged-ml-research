# ruff: noqa: E501
"""Score experiment 519's arms against the 0% flag-column control F0, L-Ww, and K2-S1 with experiment 470's scorer.

The pattern of experiment 511: experiment 470's scorer, tail script, and curtailment script are pointed at
this folder. Arms are the quantile averages of seeds 0 and 1 of the window forecasts that
`e00519_fit.py assemble` wrote, on experiment 454's rows. L-Ww is experiment 510's stored seed table and
L-mix (the 50% share) is experiment 511's, both linked unchanged. The minimum detectable effect is
experiment 511's two-seed `mde.csv`, not refitted. Counted tests: 16 contrasts x 2 groups for CRPS (32),
and the same for q99 pinball loss (64 in all).

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
import e00511_common as k511  # ty: ignore[unresolved-import]
import e00519_fit as fit  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = fit.W
SEED_PREDS: Final[Path] = WORK_DIR / "seed_preds"
REFITTED: Final[tuple[str, ...]] = tuple(fit.ARMS)
STORED_510: Final[tuple[str, ...]] = ("L-Ww",)
STORED_511: Final[tuple[str, ...]] = ("L-mix",)
ARMS: Final[dict[str, tuple[int, ...]]] = dict.fromkeys(
    (*STORED_510, *STORED_511, *REFITTED), (0, 1)
)
CONTROLS: Final[tuple[str, ...]] = ("K2-S1", "A-pool")
REFERENCES: Final[tuple[str, ...]] = ("BG2", "manual_heuristic", "climatology")
TESTS: Final[int] = 64
"""16 contrasts x 2 groups x [CRPS, q99 pinball loss]."""
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    # ENS share against the 0% flag-column control
    ("E10", "F0"), ("E25", "F0"), ("L-mix", "F0"), ("E75", "F0"), ("E100", "F0"),
    # the control and two shares against the 37-input anchor (L-mix minus L-Ww repeats experiment 511)
    ("F0", "L-Ww"), ("E25", "L-Ww"), ("L-mix", "L-Ww"),
    # flag ablation at 25% and 50%
    ("NF25", "E25"), ("NF50", "L-mix"), ("Z25", "E25"), ("Z50", "L-mix"),
    # shape of the sweep
    ("E25", "E10"), ("E75", "E25"), ("E100", "E25"), ("L-mix", "E25"),
)  # fmt: skip
TWO_STAGE_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (("TS", "L-Ww"), ("TS", "F0"))
"""The optional two-stage control is its own family: 2 contrasts x 2 groups x 2 metrics = 8 tests."""
FIVE: Final[tuple[str, ...]] = build470.MONTHS
assert len(CONTRASTS) * 4 == TESTS
TS_TESTS: Final[int] = len(TWO_STAGE_CONTRASTS) * 4


def configure(*, family: str = "primary") -> None:
    """Point experiment 470's scorer, tails, and curtailment scripts at this experiment.

    `family` is `primary` (the 16 contrasts) or `ts` (the optional two-stage control, scored in its own folder
    and its own Bonferroni family).
    """
    heads470.WORK_DIR = WORK_DIR
    score470.MONTHS = FIVE
    curtailment470.MONTHS = FIVE
    build454.MONTHS = FIVE
    score470.SCORE_DIR = WORK_DIR / ("scores" if family == "primary" else "scores_ts")
    score470.CANDIDATES = tuple(ARMS)
    score470.CONTROLS = CONTROLS
    score470.DESCRIPTIVE = ()
    score470.SINGLES = ()
    score470.REFERENCES = REFERENCES
    score470.MAIN_ARMS = (*ARMS, *CONTROLS)
    score470.CELLS = {}
    score470.CONTRASTS = CONTRASTS if family == "primary" else TWO_STAGE_CONTRASTS
    score470.GATING_CRPS = ("eight", "six_clean")
    score470.GATING_Q99 = ("eight", "six_clean")
    score470.TESTS = TESTS if family == "primary" else TS_TESTS

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

    def mde_511(*, table: pl.DataFrame) -> pl.DataFrame:
        """Return experiment 511's two-seed minimum detectable effect (this experiment does not refit it)."""
        _ = table
        return pl.read_csv(k511.WORK_DIR / "scores" / "mde.csv")

    score470.mde = mde_511  # ty: ignore[invalid-assignment]
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
    """Symlink experiments 510's and 511's seed tables and experiment 470's stored arms that the scorer reads."""
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
    for arm, root in [
        *[(a, c510.WORK_DIR) for a in STORED_510],
        *[(a, k511.WORK_DIR) for a in STORED_511],
    ]:
        link = SEED_PREDS / arm
        if not link.exists():
            link.symlink_to(root / "seed_preds" / arm, target_is_directory=True)


def build() -> None:
    """Write each arm's pooled seeds on experiment 454's rows."""
    sites = build470.site_map()
    for month in FIVE:
        _, keys = build454.load_heads(month=month)
        rows = pl.DataFrame({key: keys[key] for key in build470.KEYS})
        for arm, seeds in ARMS.items():
            folder = SEED_PREDS / arm
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
    parser.add_argument("--family", choices=("primary", "ts"), default="primary")
    args = parser.parse_args()
    configure(family=args.family)
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
