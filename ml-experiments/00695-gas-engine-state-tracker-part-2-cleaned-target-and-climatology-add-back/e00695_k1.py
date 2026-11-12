# ruff: noqa: E501
"""Experiment 695: run experiment 677's step-5 chain (K1, class 1) on a cleaned power stack (GPU).

    E00695_ARM=A1 E00677_MONTH=2026-02 ... python e00695_k1.py k1 pretrain --step 5 --months 2026-02 --seeds 0 1 2

This is a wrapper, not a copy. It replaces one row of the power-over-capacity stack that every stage
of the chain reads (`e00510_common.clean_stack`) with the row that `e00695_clean.py` wrote for the
arm and the fold, and then runs `e00677_k1.py` unchanged with the same arguments. Pre-training, the
fresh-mixer stage 2, and the fine-tune read training targets, lagged power, and synthetic-shift
sizes from that one stack, so all three stages see the cleaned series. `E00695_ARM=A0` replaces
nothing, which makes arm A0 the control that runs the same code path.

The replacement is installed before `e00677_fold.install`, which wraps `clean_stack` to hide the
held-out power in the training phase, so the hiding applies to the cleaned row as well.
"""

import os
import runpy
import sys

import e00510_common as c  # ty: ignore[unresolved-import]
import numpy as np
from e00695_common import DATA_DIR

ARM = os.environ["E00695_ARM"]
MONTH = os.environ["E00677_MONTH"]
ARMS_WITH_ROWS = {"A1", "A1n", "AH", "W0", "W1", "A1id"}  # A1id: the identity control
assert ARM == "A0" or ARM in ARMS_WITH_ROWS, f"unknown arm {ARM}"

if ARM != "A0":
    _folder = DATA_DIR / "stacks" / ARM
    _file = _folder / f"{MONTH}.npz" if (_folder / f"{MONTH}.npz").exists() else _folder / "all.npz"
    _row = np.load(_file)
    _original = c.clean_stack

    def clean_stack() -> np.ndarray:
        """Return the pipeline's stack with the arm's cleaned row in place of the raw one."""
        stack = _original().copy()
        values = _row["values"]
        assert stack.shape[1] == len(values), (stack.shape, len(values))
        stack[int(_row["row"])] = values
        return stack

    c.clean_stack = clean_stack
    print(f"E00695: arm {ARM}, fold {MONTH}, row {int(_row['row'])} from {_file}", flush=True)

sys.argv = ["e00677_k1.py", *sys.argv[1:]]
runpy.run_module("e00677_k1", run_name="__main__")
