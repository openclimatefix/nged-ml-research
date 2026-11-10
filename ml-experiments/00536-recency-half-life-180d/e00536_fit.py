# ruff: noqa: ANN001, ANN201, E501
r"""Fit the daily recency half-life arms of experiment 534 with experiment 511's harness (outputs redirected).

Arms `R-3`, `R-7`, `R-14`, `R-30` (daily refit, pretrained start, half-life in days). Half-life 90 days
is 511's stored L-1d. A 6-hourly refit is not run: the harness's cutoffs, windows, and tiling are
day-based.
"""

import argparse

import e00510_common as c  # ty: ignore[unresolved-import]
import e00510_fit as f510  # ty: ignore[unresolved-import]
import e00511_fit as f511  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W = ML_EXPERIMENTS_DATA_DIR / "00536-recency-half-life-180d" / "work"
SEEDS = (0, 1)
f510.W = W
f511.W = W
_PRETRAINED_510 = f511.E510 / "pretrained" / "L0PT"


def pretrained_file(*, arm, month, seed):
    """Return 510's own pretrained weights when it has them, else 448's (the harness looks under W, which is redirected)."""
    own = _PRETRAINED_510 / f"{month}_s{seed}.pt"
    return own if own.exists() else c.E448_PRETRAINED / f"{month}_s{seed}.pt"


f511.pretrained_file = pretrained_file
ARMS = {f"R-{h}": f511.Arm(f"R-{h}", "d", float(h)) for h in (180,)}


def assemble(*, arm: f511.Arm) -> None:
    """Concatenate window forecasts into one table per month and seed."""
    for month in c.MONTHS:
        cuts = f511.k.cutoffs(month=month, cadence=arm.cadence)
        for seed in SEEDS:
            parts = [
                W
                / "fits"
                / arm.tag
                / f"{month}_d{f511.day_of(month=month, cut=cut)}_s{seed}.parquet"
                for cut in cuts
            ]
            if all(p.exists() or p.with_suffix(".empty").exists() for p in parts):
                frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
                f511.atomic_frame(
                    frame=frame, target=W / "seed_preds" / arm.tag / f"{month}_s{seed}.parquet"
                )
                print("wrote", arm.tag, month, seed, frame.height, flush=True)
            else:
                print("incomplete", arm.tag, month, seed, flush=True)


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit", "assemble"))
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--limit", type=int, default=None)
    a = parser.parse_args()
    arm = ARMS[a.arm]
    if a.command == "fit":
        f511.fit_arm(
            arm=arm,
            months=list(c.MONTHS),
            seeds=a.seeds,
            stress=a.stress,
            limit=a.limit,
            device="cuda",
        )
    else:
        assemble(arm=arm)


if __name__ == "__main__":
    main()
