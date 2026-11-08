# ruff: noqa: E501, E402
"""Experiment 650, step 1: run K1's saved networks on the extended-shift bank (CPU).

Runs experiment 613's `k1_d5` inject path (K1's five seeds, the d0 and d14 daily fits) through
experiment 591's `run_inject`, with three patches and nothing else changed: `bank.contaminate`
returns this experiment's shifted power (`own` added to the Primary's lags; receivers' mirrored
shifts are not in K1's inputs, as in experiment 647), the output files go to this experiment's
`inject/` folder, and the device is the CPU. The saved weights are read only.

Usage: python e00650_inject.py --months 2025-10 2025-12 --seeds 0 1 2 3 4 --passes 0 1 2
"""

import argparse
import os
import runpy
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

import e00650_common as c650
import numpy as np

_p = argparse.ArgumentParser()
_p.add_argument("--months", nargs="+", required=True)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
_p.add_argument("--passes", nargs="+", type=int, required=True)
MINE = _p.parse_args()
assert max(MINE.months) <= "2026-06", "July 2026 onwards is never opened"
argv = [
    "e00613_arm.py", "k1_d5", "inject", "--months", *MINE.months, "--seeds", *map(str, MINE.seeds),
    "--passes", *map(str, MINE.passes),
]  # fmt: skip
sys.argv = ["e00591_arm.py", "A1", "inject", "--months", *MINE.months, "--seeds", *map(str, MINE.seeds),
            "--passes", *map(str, MINE.passes)]  # fmt: skip
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00591_arm as arm  # ty: ignore[unresolved-import]

month_of_start = {c650.month_start_idx(month=m): m for m in c650.MONTHS}


def contaminate(
    *, clean: np.ndarray, month_start_idx: int, r: int
) -> tuple[np.ndarray, list[dict]]:
    """Return `clean` plus pass `r`'s own shifts, and the event rows (experiment 581's interface)."""
    own, _mirror, events = c650.shifts_for(clean=clean, month=month_of_start[month_start_idx], r=r)
    return np.where(np.isfinite(clean), clean + own, clean), [e.row() for e in events]


def out_path(out_dir, month, r, day, seed):  # noqa: ANN001, ANN201
    """Return the output parquet of one cell in this experiment's folder."""
    return c650.INJECT / f"{month}_r{r:02d}_d{day}_s{seed}.parquet"


bank.contaminate = contaminate
_K1_S01 = (
    c650.ML_EXPERIMENTS_DATA_DIR
    / "00612-nra-combined-networks-three-class-roster-completion/arms/k1_d5/work/models"
)
_model_file = arm.e.model_file


def model_file(*, tag, month, day, seed, sfx=""):  # noqa: ANN001, ANN201
    """Return K1's weights: seeds 0 and 1 are experiment 612's stored fits, seeds 2 to 4 experiment 613's."""
    if seed < 2:
        return _K1_S01 / tag / f"{month}_d{day}_s{seed}{sfx}.pt"
    return _model_file(tag=tag, month=month, day=day, seed=seed, sfx=sfx)


arm.e.model_file = model_file


def attach_tables(*, standardiser) -> None:  # noqa: ANN001
    """Copy of experiment 574's `attach_tables` that keeps the expected-input table on the CPU."""
    e = arm.e
    cols, origin = e.d560.raw_table(source="ft", cache=e.W560 / "expected")
    e.CTX["origin_ft"] = np.asarray(origin)
    e.CTX["table_ft"] = e.torch.from_numpy(
        e.d560.standardised_table(cols=cols, standardiser=standardiser)
    )


arm.e.attach_tables = attach_tables
arm._out = out_path
_run = arm.run_inject
arm.run_inject = lambda: (c650.INJECT.mkdir(parents=True, exist_ok=True), _run(device="cpu"))
sys.argv = argv
runpy.run_path(
    str(
        __import__("pathlib").Path(arm.__file__).parent.parent
        / "00613-nra-roster-contrasts-with-intervals-five-seeds"
        / "e00613_arm.py"
    ),
    run_name="__main__",
)
