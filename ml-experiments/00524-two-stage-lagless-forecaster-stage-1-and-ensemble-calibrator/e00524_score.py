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
    new = (
        "S2-0F",
        "S2-F-A-nll",
        "S2-F-A-crps",
        "S2-F-B-nll",
        "S2-F-B-crps",
        "S2-J-0.4x",
        "S2-J-0.1x",
    )
    s510.ARMS = {
        "S2-0": (0, 1),
        **dict.fromkeys(new, (0, 1)),
        "L-M": (0, 1),
        "L-W": (0, 1),
        "L-Ww": (0, 1),
    }
    s510.CONTRASTS = (
        *[(a, "L-Ww") for a in ("S2-0", *new)],
        ("S2-F-B-nll", "S2-0F"), ("S2-F-A-nll", "S2-0F"), ("S2-F-A-nll", "S2-F-B-nll"),
        ("S2-F-B-nll", "S2-F-B-crps"), ("S2-J-0.4x", "S2-F-B-nll"), ("S2-J-0.1x", "S2-J-0.4x"),
        ("S2-0", "K2-S1"), ("S2-J-0.4x", "K2-S1"), ("S2-F-B-nll", "K2-S1"),
    )  # fmt: skip
    s510.TESTS = 4 * len(s510.CONTRASTS)
    s510.configure()
    s510.link_stored()
    source = (
        s510.ML_EXPERIMENTS_DATA_DIR
        / "00510-lagless-demand-network-weekly-finetune-and-recency-weights"
        / "work"
        / "seed_preds"
    )
    for arm in ("L-M", "L-W", "L-Ww"):
        link = WORK_DIR / "seed_preds" / arm
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(source / arm, target_is_directory=True)


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
