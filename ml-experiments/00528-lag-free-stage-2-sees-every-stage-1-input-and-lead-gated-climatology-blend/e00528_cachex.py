# ruff: noqa: PT018
"""Experiment 528, `e00528_cachex.py`: kept for the record.

Experiment 528 (interim, unaudited): cache every stage-1 input column for the rows of 524's
stage-2 cache.
"""

import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00524_stage2 as s2  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = (
    ML_EXPERIMENTS_DATA_DIR
    / "00528-lag-free-stage-2-sees-every-stage-1-input-and-lead-gated-climatology-blend"
    / "work"
    / "xcache"
)
OUT.mkdir(parents=True, exist_ok=True)
clean = c.clean_stack()
for month in c.MONTHS:
    if (OUT / f"{month}.npz").exists():
        continue
    mr = c.build_month(month=month, clean=clean)
    test = data.build_test(month=month, standardiser=mr.base.standardiser)
    ws = c.us(t=c.month_start(month=month)) - c.WINDOW_DAYS * c.DAY_US
    rows = np.nonzero(mr.valid_us > ws)[0]
    ref = np.load(s2.CACHE / f"{month}_s0.npz")
    assert (ref["primary_train"] == mr.primary[rows]).all() and (
        ref["valid_train"] == mr.valid_us[rows]
    ).all()
    assert (ref["valid_test"] == test.valid).all() and (ref["primary_test"] == test.primary).all()
    np.savez(
        OUT / f"{month}.tmp.npz",
        x_train=mr.x[rows].astype(np.float32),
        x_test=test.x.astype(np.float32),
        columns=np.array(list(mr.base.columns)),
    )
    (OUT / f"{month}.tmp.npz").rename(OUT / f"{month}.npz")
    print(month, mr.x[rows].shape, test.x.shape, list(mr.base.columns), flush=True)
