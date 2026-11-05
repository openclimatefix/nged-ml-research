# ruff: noqa: E501
"""Experiment 573: cache the frozen M0 network's outputs on each fold's ERA5 and CAMS pre-training rows.

The network is experiment 554's pretrained M0 (stage `pre`, seed 0) and is never trained here. For
each fold month, the fold's pre-training rows are built exactly as experiment 565's `pretrain_wt` builds
them (experiment 448's `pretrain_rows`, on the real cleaned power), the nine mixture outputs of every
row are computed once, and the arrays that the offset fits need are saved. A world's injected
targets are then a function of the saved target index, so no world needs the network again.

Row-by-row assertions (the standing timing rule): the row's origin is not after the fold's refit, the
row's valid time is not after the refit minus `REFIT_CUTOFF_AGE`, the origin is before the valid time,
the input has 37 columns and no lag column. One example row is printed.
"""

import sys
from datetime import UTC, datetime

import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00573-stepwise-output-offset-known-answer-step-recovery"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
W554 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out/work"
)


def cache_path(*, month: str):  # noqa: ANN201
    """Return where the fold's cache lives."""
    return WORK / "cache" / f"{month}.npz"


def build(*, month: str, device: str) -> None:
    """Build and save one fold's cache."""
    clean = c.clean_stack()
    refit = c.month_start(month=month)
    rows = data.build_fold(refit=refit, clean=clean)
    pre = r448.pretrain_rows(rows=rows, refit=refit, clean=clean)
    grid_us, step = c.prep422.GRID_START_US, c.prep422.STEP_US
    valid_us = grid_us + pre.target_idx.astype(np.int64) * step
    origin_us = grid_us + pre.origin_idx.astype(np.int64) * step
    cut_valid = int((refit - c.regimes.REFIT_CUTOFF_AGE).timestamp() * 1e6)
    assert pre.x.shape[1] == m.N_INPUTS, "the base network is lag-free: 37 columns, no lag column"
    assert (valid_us <= cut_valid).all(), "a pre-training target is after the refit cut"
    assert (origin_us <= int(refit.timestamp() * 1e6)).all(), "a row's origin is after the refit"
    assert (origin_us < valid_us).all(), "a row's origin is not before its target"
    i = len(valid_us) // 2
    example = (
        f"target {datetime.fromtimestamp(valid_us[i] / 1e6, UTC)}, origin "
        f"{datetime.fromtimestamp(origin_us[i] / 1e6, UTC)} (weather issued at or before the origin; "
        f"all {len(valid_us):,} rows asserted origin < target <= refit {refit:%Y-%m-%d} minus "
        f"{c.regimes.REFIT_CUTOFF_AGE})"
    )
    print(month, "timing checks passed |", example, flush=True)
    model = m.ModularMdn(n_primaries=len(rows.target_mean), stage="pre")
    state = torch.load(W554 / "pretrained" / "M0" / f"{month}_s0.pt", map_location="cpu")
    model.load_state_dict({a: v for a, v in state.items() if not a.startswith("mixer_ft.")}, strict=False)
    model.to(device).eval()
    outs = []
    with torch.no_grad():
        for a in range(0, len(pre.x), 1_000_000):
            x = torch.from_numpy(pre.x[a : a + 1_000_000]).to(device)
            p = torch.from_numpy(pre.primary[a : a + 1_000_000]).to(device)
            outs.append(model(x, p).float().cpu().numpy())
    out = np.concatenate(outs)
    target = WORK / "cache"
    target.mkdir(parents=True, exist_ok=True)
    tmp = target / f"{month}.tmp.npz"
    np.savez(
        tmp, out=out, y=pre.y.astype(np.float32), primary=pre.primary.astype(np.int16),
        target_idx=pre.target_idx.astype(np.int32), is_val=pre.is_val,
        target_mean=rows.target_mean, target_std=rows.target_std, example=np.array(example),
    )  # fmt: skip
    tmp.rename(cache_path(month=month))
    print(month, "rows", len(out), "saved", flush=True)


if __name__ == "__main__":
    for month in sys.argv[1:] or list(c.MONTHS):
        if not cache_path(month=month).exists():
            build(month=month, device="cuda")
