# ruff: noqa: E501, E402
"""Experiment 696: experiment 677's step-5 chain (K1) with an engine state in the pre-training mixer.

    E00696_ARM=S1 E00677_MONTH=2026-02 ... python e00696_k1.py k1 pretrain --step 5 --months 2026-02 --seeds 0 1 2

This is a wrapper, not a copy. It changes four things and then runs `e00677_k1.py` unchanged with the
same arguments:

1. **The power stack.** For the host arms, the host's row of the stack (`e00510_common.clean_stack`)
   is replaced by experiment 695's row of the host with Gas A injected, in all three stages, as
   experiment 695's `e00695_k1.py` does.
2. **The network class.** `e00591_model.AuxLagMdn` is replaced by `e00696_model.StateAuxLagMdn`, which
   widens `mixer_pre` by 3 inputs at every stage. `e00591_arm.new_model` reads the class from its
   module when it builds a network, so the replacement takes effect there. Replacing
   `e00574_fit.new_model` instead would be overwritten when `e00591_arm` is imported.
3. **The pre-training rows.** A pass-through wrapper of `e00448_run.pretrain_rows` keeps the rows'
   Primary, target half-hour, origin half-hour, reading, and held-out flag. A wrapper of
   `e00574_fit.train` appends the 3 state columns to the pre-training tensor once per fold, fills them
   from the arm's state file, checks their alignment and liveness, and (arms W0, WST, W0p) runs the
   encoder probe M1 on the pre-trained network. Stage 2 and stage 3 pass through unchanged, after a
   check that their input has the original column layout.
4. **`E00696_ARM=A0` or `G0`** replaces nothing, so those arms run experiment 695's code path.

`e00591_arm` is imported here, before `e00677_k1`, with the arguments `e00677_k1` would give it, so that
`e00574_fit` can be patched. `e00677_k1` then imports the cached module.
"""

import argparse
import os
import runpy
import sys
from pathlib import Path

import e00510_common as c  # ty: ignore[unresolved-import]
import numpy as np
from e00696_common import ARMS, E695_DIR, STATE_FREE_ARMS
from e00696_state import state_path

ORIGINAL_ARGV = list(sys.argv[1:])
ARM = os.environ["E00696_ARM"]
MONTH = os.environ["E00677_MONTH"]
SPEC = ARMS[ARM]
PROBE_ONLY = ARM == "W0p"
RUN_ROOT = Path(os.environ["E00677_ROOT"])

_args = argparse.ArgumentParser()
_args.add_argument("name")
_args.add_argument("command", choices=("pretrain", "step2", "fit"))
_args.add_argument("--months", nargs="+")
_args.add_argument("--seeds", nargs="+", type=int)
MINE, _ = _args.parse_known_args(sys.argv[1:])
assert MINE.months == [MONTH], (MINE.months, MONTH)

RAW_STACK = c.clean_stack
HOST_ROW: int | None = None
if SPEC.stack is not None:
    _folder = E695_DIR / "stacks" / SPEC.stack
    _file = _folder / f"{MONTH}.npz" if (_folder / f"{MONTH}.npz").exists() else _folder / "all.npz"
    _row = np.load(_file)
    HOST_ROW = int(_row["row"])

    def clean_stack() -> np.ndarray:
        """Return the pipeline's stack with the host's row replaced by the row with Gas A injected."""
        stack = RAW_STACK().copy()
        values = _row["values"]
        assert stack.shape[1] == len(values), (stack.shape, len(values))
        stack[HOST_ROW] = values
        return stack

    c.clean_stack = clean_stack
    print(f"E00696: arm {ARM}, fold {MONTH}, host row {HOST_ROW} from {_file}", flush=True)

if SPEC.kind != "none":
    import e00696_model

    e00696_model.install()
    print(f"E00696: arm {ARM} builds e00696_model.StateAuxLagMdn", flush=True)

sys.argv = [
    "e00591_arm.py", "A1", MINE.command, "--months", *MINE.months, "--seeds", *map(str, MINE.seeds),
]  # fmt: skip
import e00591_arm as arm  # ty: ignore[unresolved-import]
from e00696_hooks import StateHooks

e = arm.e
HOOKS = StateHooks(
    spec=SPEC,
    month=MONTH,
    run_root=RUN_ROOT,
    state_file=state_path(arm=ARM, month=MONTH),
    host_row=HOST_ROW,
    raw_stack=RAW_STACK,
    stored_pretrained=lambda seed: (
        E695_DIR
        / "runs"
        / "W0"
        / "chains"
        / "5"
        / MONTH
        / "work"
        / f"pretrained_{e.SPEC.pretrain_from}"
        / f"{MONTH}_s{seed}.pt"
    ),
    probe_only=PROBE_ONLY,
)
if ARM not in STATE_FREE_ARMS or SPEC.probe:
    e.r448.pretrain_rows = HOOKS.wrap_pretrain_rows(e.r448.pretrain_rows)
    e.train = HOOKS.wrap_train(e.train)

sys.argv = ["e00677_k1.py", *ORIGINAL_ARGV]
runpy.run_module("e00677_k1", run_name="__main__")
