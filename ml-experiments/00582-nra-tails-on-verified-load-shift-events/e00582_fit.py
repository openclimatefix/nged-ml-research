# ruff: noqa: E501, E402, ANN001, ANN201
"""Experiment 582: experiment 572's v3+D5 fit on the fold months that hold verified load-shift events, with a corrected-lag twin built from the verified transfers.

A thin wrapper around experiment 572's `e00572_fit.py` (imported, unchanged), in the manner of
experiment 578's `e00578_fit.py`: it re-points the output folder to this experiment's own work
folder, adds the fold month 2026-05 (three verified transfers end in May 2026), and replaces the
`corrected` scoring twin's shift with the verified one (observed minus the fitted depth inside each
verified transfer, no other correction). Months of experiments 572 and 578 reuse their saved fine-tuned
weights (predict only); 2026-05 pre-trains and fine-tunes here. Every lag is still at least 24 h before
the origin and the weather is the run at or before the origin minus 9 h, as in experiment 572.

Usage: python e00582_fit.py {pretrain|fit|assemble} [--months ...] [--seeds ...]
"""

import json
import sys
from pathlib import Path

CMD = {"pretrain": "pretrain-wt", "fit": "fit", "assemble": "assemble"}[sys.argv[1]]
rest = sys.argv[2:]
sys.argv = ["e00572_fit.py", CMD, "--arm", "V3D5", *rest]

import e00572_fit as e  # ty: ignore[unresolved-import]
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00582_common as k  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W572 = e.W
W578 = ML_EXPERIMENTS_DATA_DIR / k578.NAME / "work"
OLD_572 = set(k578.MONTHS)
OLD_578 = {"2025-07", "2025-08"}
e.WORK = ML_EXPERIMENTS_DATA_DIR / k.NAME
e.W = e.WORK / "work"
e.SIBLINGS.append("corrected")
_model_file = e.model_file


def model_file(*, tag: str, month: str, day: int, seed: int, sfx: str = "") -> Path:
    """Return saved weights: experiment 572's or 578's for months they fitted, this experiment's otherwise."""
    for months, root in ((OLD_572, W572), (OLD_578, W578)):
        if month in months:
            f = root / "models" / tag / f"{month}_d{day}_s{seed}{sfx}.pt"
            assert f.exists(), f
            return f
    return _model_file(tag=tag, month=month, day=day, seed=seed, sfx=sfx)


e.model_file = model_file
_build_test = e.data.build_test
SHIFT: np.ndarray | None = None


def build_test(*, month, standardiser):
    """Add the `corrected` scoring twin: the lag columns are rebuilt from the verified-event counterfactual."""
    global SHIFT  # noqa: PLW0603
    test = _build_test(month=month, standardiser=standardiser)
    clean = e.CTX["clean"]
    if SHIFT is None:
        SHIFT = k.verified_shift(shape=clean.shape)
    fixed = np.where(np.isfinite(clean), clean + SHIFT, clean).astype(np.float32)
    blk = e.block_of(
        clean=fixed, primary=test.primary, target_idx=test.target_idx, origin_idx=test.origin_idx
    )
    x = test.x.copy()
    assert x.shape[1] >= 45
    x[:, 37:45] = blk
    assert np.array_equal(x[:, 43:45], test.x[:, 43:45]), "the corrected lags moved a lag time"
    e.CTX["variants"]["corrected"] = x
    ch = float((np.abs(x[:, 37] - test.x[:, 37]) > 1e-6).mean())
    print(
        f"{month}: share of test rows whose day lag differs under the correction: {ch:.4f}",
        flush=True,
    )
    return test


e.data.build_test = build_test

if __name__ == "__main__":
    if CMD == "assemble":
        e.assemble()
    elif CMD == "pretrain-wt":
        e.pretrain_wt(device="cuda")
    else:
        e.fit_arm(months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda")
        out = (
            e.WORK
            / f"checks_fit_V3D5_{'-'.join(e.ARGS.months)}_s{'-'.join(map(str, e.ARGS.seeds))}.json"
        )
        out.write_text(json.dumps(e.CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
