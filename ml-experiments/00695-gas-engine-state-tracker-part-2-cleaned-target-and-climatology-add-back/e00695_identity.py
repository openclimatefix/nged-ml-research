# ruff: noqa: E501
"""Experiment 695: the network-level identity control, in two commands.

    python e00695_identity.py stack     # write stacks/A1id/all.npz: D8's raw row, sent through the cleaned-row route
    python e00695_identity.py compare   # compare arm A1id's forecasts with arm A0's, fold 2025-11, seeds 0 to 2

Arm A1id runs the A1 code path (`e00695_k1.py` replaces D8's row of the stack with a stored row) with
cleaning disabled: the stored row is the raw row (k = 0 everywhere, so the add-back's p_0 is 1 and
adds nothing). Its forecasts must equal arm A0's, which replaces nothing, within refit noise. The
seeds are the same, so the expected gap is floating-point noise or non-deterministic GPU kernels.
"""

import sys

import e00695_clean as clean
import numpy as np
import polars as pl
from e00695_common import DATA_DIR

MONTH = "2025-11"


def write_stack() -> None:
    """Write D8's raw power-over-capacity row as the A1id arm's stored row."""
    setup = clean.load_setup(arm="A1")
    out = DATA_DIR / "stacks" / "A1id"
    out.mkdir(parents=True, exist_ok=True)
    values = (setup.raw_row / setup.capacity).astype(np.float32)
    np.savez(out / "all.npz", row=setup.row, values=values)
    print(
        f"wrote {out / 'all.npz'}: row {setup.row}, {np.isfinite(values).sum()} finite half-hours"
    )


def compare() -> None:
    """Print the largest and mean absolute gap of A1id's quantiles from A0's, per seed."""
    for seed in (0, 1, 2):
        frames = {}
        for arm in ("A0", "A1id"):
            path = (
                DATA_DIR
                / "runs"
                / arm
                / "arms/k1_5w_sp1/work/fits/P5"
                / f"{MONTH}_d0_s{seed}.parquet"
            )
            frames[arm] = pl.read_parquet(path)
        a, b = frames["A0"], frames["A1id"]
        assert a.height == b.height, (a.height, b.height)
        numeric = [
            name for name, dtype in a.schema.items() if dtype.is_float() and name in b.columns
        ]
        key = [name for name in a.columns if name not in numeric]
        a, b = a.sort(key), b.sort(key)
        for name in numeric:
            diff = (a[name] - b[name]).abs()
            print(
                f"seed {seed} {name}: max abs gap {diff.max():.3e}, mean abs gap {diff.mean():.3e},"
                f" mean value {a[name].abs().mean():.4f}"
            )


if __name__ == "__main__":
    {"stack": write_stack, "compare": compare}[sys.argv[1]]()
