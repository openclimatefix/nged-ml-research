"""Experiment 577: choose penalties, the posterior prior, and jump statistics on fold 1 only.

Fold 1 is the 2025-10 fold, seed 0. Reads the early-stopping negative log-likelihood (NLL) that each
sweep arm logged (no test row is read) and writes `picks.json`:

- `O1`, `O2`, `O3`: the largest penalty weight whose pretraining NLL is within 0.002 of the best
  (the steadier offset is preferred when accuracy ties), and the chosen arm's weights are copied to
  the arm's own pretrained folder for fold 1 seed 0.
- `prior_sd`: the prior standard deviation of the posterior with the lowest fine-tuning NLL.
- `q3`: the jump rate per Primary-day and the root-mean-square jump size (capacity fraction) of the
  steps of 1% or more in O2's pretrained daily offset path on fold 1.

Usage: python e00577_picks.py {penalties|prior|jumps|events}
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Final

import e00454_score as score454  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00573_steps as s573  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00577-static-between-steps-offsets-and-probabilistic-offset-widening"
    / "work"
)
PICKS: Final[Path] = W / "picks.json"
MONTH: Final = "2025-10"
TOLERANCE: Final = 0.002
GRIDS: Final = {
    "O1": (3000, 30000, 300000),
    "O2": (3000, 30000, 300000),
    "O3": (1000, 10000, 100000),
}


def load() -> dict:
    """Return the stored picks."""
    return json.loads(PICKS.read_text()) if PICKS.exists() else {}


def save(*, picks: dict) -> None:
    """Write the picks."""
    PICKS.write_text(json.dumps(picks, indent=1))


def penalties() -> None:
    """Pick each group's penalty weight from its pretraining NLL and copy the chosen weights."""
    picks = load()
    for group, grid in GRIDS.items():
        scores = {}
        for lam in grid:
            f = W / "logs" / "pretrain" / f"{group}-{lam}_{MONTH}_s0.json"
            if f.exists():
                scores[lam] = json.loads(f.read_text())["val_nll"]
        assert len(scores) == len(grid), (group, scores)
        best = min(scores.values())
        lam = max(a for a, v in scores.items() if v <= best + TOLERANCE)
        picks[group] = {"lam": float(lam), "val_nll": scores}
        dest = W / f"pretrained_{group}" / "real"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            W / f"pretrained_{group}-{lam}" / "real" / f"{MONTH}_s0.pt", dest / f"{MONTH}_s0.pt"
        )
        print(group, "scores", scores, "pick", lam)
    save(picks=picks)


def prior() -> None:
    """Pick the posterior's prior standard deviation from the fold-1 day-0 fine-tuning NLL."""
    picks = load()
    scores = {}
    for tau in (0.05, 0.1, 0.2):
        f = W / "logs" / f"Q2-t{tau:g}" / f"{MONTH}_d0_s0.json"
        scores[tau] = json.loads(f.read_text())["val_nll"]
    picks["prior_sd"] = min(scores, key=scores.__getitem__)
    picks["prior_scores"] = scores
    print("prior scores", scores, "pick", picks["prior_sd"])
    save(picks=picks)


def jumps() -> None:
    """Estimate the jump rate and size from O2's pretrained daily offset path on fold 1."""
    picks = load()
    knots = torch.load(W / "pretrained_O2" / "real" / f"{MONTH}_s0.pt", map_location="cpu")[
        "knots"
    ][:, :, 0].numpy()
    sd = np.nanstd(c.clean_stack(), axis=1)
    assert float(sd.max()) < 1.0, "target spreads must be capacity fractions"
    sizes = []
    for p in range(knots.shape[0]):
        sizes += [abs(z) for _, z in s573.extract(path=knots[p], std=float(sd[p]))]
    days = knots.shape[0] * knots.shape[1]
    picks["q3"] = {
        "rate_per_day": len(sizes) / days,
        "step_scale": float(np.sqrt(np.mean(np.square(sizes)))) if sizes else 0.0,
        "n_steps": len(sizes),
        "source": "O2 pretrained path, fold 1, steps of 1% of capacity or more",
    }
    print(picks["q3"])
    save(picks=picks)


def events() -> None:
    """Estimate the jump rate and size from labelled events that began before fold 1.

    The rate is the events per Primary-day from 2019-09-01 to the fold's first day. The step size
    of an event is its depth, the largest absolute value of experiment 435's step correction (a
    share of capacity) over the event. The quartiles of the depths are stored and the
    root-mean-square depth is the step scale.
    """
    picks = load()
    nra = np.load(score454.NRA_FILE)["f1.0"]
    grid, step = c.prep422.GRID_START_US, c.prep422.STEP_US
    fold_start_idx = (c.us(t=c.month_start(month=MONTH)) - grid) // step
    depths = []
    for p, spans in c.labelled_events().items():
        for onset, end in spans:
            if end < fold_start_idx:
                depths.append(float(np.abs(nra[p, onset : end + 1]).max()))
    days = 16 * fold_start_idx / 48
    d = np.array(depths)
    picks["q3"] = {
        "rate_per_day": len(d) / days,
        "step_scale": float(np.sqrt(np.mean(d**2))),
        "n_steps": len(d),
        "depth_quartiles": [float(x) for x in np.percentile(d, [25, 50, 75])],
        "source": "labelled events before fold 1 (O2 failed its static-ness check)",
    }
    print(picks["q3"])
    save(picks=picks)


if __name__ == "__main__":
    {"penalties": penalties, "prior": prior, "jumps": jumps, "events": events}[sys.argv[1]]()
