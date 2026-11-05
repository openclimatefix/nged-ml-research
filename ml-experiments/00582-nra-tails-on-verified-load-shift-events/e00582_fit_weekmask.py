# ruff: noqa: E402, ANN001, ANN201
"""Experiment 582: forecast every fold month with only the week lag masked (the day lag kept).

Predict-only: every fitted weight of experiments 572 and 578 and of this experiment is reused,
and the output goes to a separate work folder (`work_weekmask`), so no stored forecast is
touched. The twin `weekmasked` sets the week lag's value to 0, its missing flag to 1, and its age
code to the maximum, which is what `e00572_lags.absent` writes for a missing week lag and what
the network saw for the 24% of training rows whose week lag was missing. Usage: python
e00582_fit_weekmask.py {fit|assemble} --months ... --seeds 0 1.
"""

import sys
from pathlib import Path

CMD = {"fit": "fit", "assemble": "assemble"}[sys.argv[1]]
sys.argv = ["e00572_fit.py", CMD, "--arm", "V3D5", *sys.argv[2:]]

import e00572_fit as e  # ty: ignore[unresolved-import]
import e00572_lags as lg  # ty: ignore[unresolved-import]
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00582_common as k  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W572 = e.W
W578 = ML_EXPERIMENTS_DATA_DIR / k578.NAME / "work"
W582 = ML_EXPERIMENTS_DATA_DIR / k.NAME / "work"
OLD_572 = set(k578.MONTHS)
OLD_578 = {"2025-07", "2025-08"}
e.WORK = ML_EXPERIMENTS_DATA_DIR / k.NAME
e.W = e.WORK / "work_weekmask"
e.SIBLINGS[:] = ["weekmasked"]


def model_file(*, tag: str, month: str, day: int, seed: int, sfx: str = "") -> Path:
    """Return the stored weights (never trains: the file must exist)."""
    root = W572 if month in OLD_572 else W578 if month in OLD_578 else W582
    f = root / "models" / tag / f"{month}_d{day}_s{seed}{sfx}.pt"
    assert f.exists(), f
    return f


e.model_file = model_file
_build_test = e.data.build_test


def build_test(*, month, standardiser):
    """Replace the scoring twins by the single week-masked twin."""
    test = _build_test(month=month, standardiser=standardiser)
    x = test.x.copy()
    amax = lg.age_max(cap_steps=e.SPEC.cap_steps)
    x[:, 40], x[:, 41], x[:, 42] = 0.0, 1.0, amax[1]
    assert (x[:, 37:40] == test.x[:, 37:40]).all(), "the day lag changed"
    e.CTX["variants"] = {"weekmasked": x}
    return test


e.data.build_test = build_test

if __name__ == "__main__":
    if CMD == "assemble":
        e.assemble()
    else:
        e.fit_arm(months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda")
