# ruff: noqa: E501
"""Score experiment 524's arms with experiment 510's scorer (experiment 470's machinery underneath).

The S2-0 gate: the stored L0PT weights with no fine-tune against L-Ww, L-W, L-M, and K2-S1, on
experiment 454's rows. Experiment 510's `configure` is reused, with the arm list, the contrasts, and
the folders replaced. Counted tests: 4 contrasts x 2 groups x [CRPS, q99 pinball loss] = 16.

Commands: `build`, `rows`, `baseline`, `stats`, `tails`.
"""

import argparse

import e00510_score as s510  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR = (
    ML_EXPERIMENTS_DATA_DIR
    / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator"
    / "work"
)


def setup() -> None:
    """Point experiment 510's scorer at this experiment's arms and folders."""
    s510.WORK_DIR = WORK_DIR
    s510.SEED_PREDS = WORK_DIR / "seed_preds"
    s510.ARMS = {"S2-0": (0, 1), "L-M": (0, 1), "L-W": (0, 1), "L-Ww": (0, 1)}
    s510.CONTRASTS = tuple(("S2-0", b) for b in ("L-Ww", "L-W", "L-M", "K2-S1"))
    s510.TESTS = 16
    s510.configure()
    s510.score470.CANDIDATES = tuple(s510.ARMS)
    s510.score470.MAIN_ARMS = (*s510.ARMS, *s510.CONTROLS)
    s510.score470.CONTRASTS = s510.CONTRASTS
    s510.score470.TESTS = 16
    s510.tails470.ARMS = s510.score470.MAIN_ARMS
    s510.link_stored()


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "rows", "baseline", "stats", "tails"))
    args = parser.parse_args()
    setup()
    {
        "build": s510.build,
        "rows": s510.score470.rows_command,
        "baseline": s510.baseline,
        "stats": s510.score470.stats_command,
        "tails": s510.tails470.main,
    }[args.command]()


if __name__ == "__main__":
    main()
