"""Experiment 713: choose `M*` from the screen's fit logs (design 1.6). It reads no forecast.

`M*` is the arm among `M1`, `M2`, and `M3` with the lowest mean held-out validation loss at the end
of stage 3 (unweighted mixture negative log-likelihood), averaged over the 4 screen folds and 2
seeds. An arm within 0.005 of the best loses to the simpler arm, in the order `M1`, `M2`, `M3`. The
choice is written to `work/selection.json`, and the per-arm losses are printed.
"""

import json
import sys
from pathlib import Path
from typing import Final

import e00713_arms as arms
from studies.sources import ML_EXPERIMENTS_DATA_DIR

TOLERANCE: Final = 0.005
ORDER: Final = ("M1", "M2", "M3")


def stage3_loss(*, data_dir: Path, arm: str, month: str, seed: int) -> float:
    """Return the stage 3 validation loss recorded in a fit log."""
    path = arms.log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit")
    assert path.exists(), f"missing fit log {path}"
    stages = json.loads(path.read_text())["stages"]
    return float(next(s for s in stages if s["stage"] == 3)["best_val"])


def choose(*, losses: dict[str, float]) -> str:
    """Return the simplest arm within `TOLERANCE` of the lowest loss."""
    best = min(losses.values())
    return next(a for a in ORDER if losses[a] <= best + TOLERANCE)


def main() -> int:
    """Write the selection."""
    data_dir = ML_EXPERIMENTS_DATA_DIR
    losses = {
        arm: sum(
            stage3_loss(data_dir=data_dir, arm=arm, month=m, seed=s)
            for m in arms.SCREEN_FOLDS
            for s in arms.SCREEN_SEEDS
        )
        / (len(arms.SCREEN_FOLDS) * len(arms.SCREEN_SEEDS))
        for arm in ORDER
    }
    star = choose(losses=losses)
    print("mean stage 3 validation loss:", {a: round(v, 4) for a, v in losses.items()}, "->", star)
    path = arms.selection_file(data_dir=data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"m_star": star, "losses": losses}, indent=1))
    tmp.rename(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
