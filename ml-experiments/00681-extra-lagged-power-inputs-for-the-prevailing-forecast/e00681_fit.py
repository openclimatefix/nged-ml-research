# ruff: noqa: D103, E501
"""Experiment 681, step 2: fit experiment 667's Rhaz head with extra lagged power inputs (CPU).

Experiment 667's `p667_fit.fit_rhaz` is called unchanged. Only its `load` is wrapped, so that each
origin's raw input row gets the arm's lag block (residual with NaN set to 0, then one missing flag
per lag) appended after the 667 columns, and its output folder is changed. The head's width follows
the input width. After the fit, the head's quantiles on the bank rows are also written (`bank_q`),
which experiment 667 does not store.

Arms: `L0` (no extra input), `L1` (first 8 lag columns), `L2` (11), `L3` (14), `P` and `P7` (the
placebo and the 7-day-shifted lags, at the width `--width`).

Usage: python e00681_fit.py ARM MONTH [--seeds 0 1] [--width 11]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Final

sys.path.insert(0, str(next(Path(__file__).resolve().parents[1].glob("00667-*/prevailing"))))

import e00681_lags as lags
import numpy as np
import p667_fit as base  # ty: ignore[unresolved-import]
import p667_head as head667  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]

WIDTH_OF: Final = {"L0": 0, "L1": 8, "L2": 11, "L3": 14}
BLOCK_OF: Final = {
    "L0": "true",
    "L1": "true",
    "L2": "true",
    "L3": "true",
    "P": "placebo",
    "P7": "shifted7",
}
OUT_ROOT: Final = lags.OUT.parent / "fits"
FIRST_TRAINED: Final = "2026-01"
"""Folds before this month have fewer than two earlier banks, so experiment 667 used the fallback head."""


def extra_columns(*, month: str, kind: str, arm: str, width: int) -> np.ndarray:
    """Return the (origins, 2 * width) block: residuals (NaN as 0) then missing flags."""
    z = np.load(lags.OUT / f"lags_{month}.npz")
    block = z[f"{kind}_{BLOCK_OF[arm]}"][:, :width]
    return np.concatenate(
        [np.nan_to_num(block), np.isnan(block).astype(np.float32)], axis=1
    ).astype(np.float32)


def install(*, arm: str, width: int) -> None:
    """Wrap `base.load` so every loaded head-data dict carries the arm's extra columns, once."""
    original = base.load

    def load(*, month: str, kind: str) -> dict:
        z = original(month=month, kind=kind)
        if width and not z.get("_e681"):
            extra = extra_columns(month=month, kind=kind, arm=arm, width=width)
            assert len(extra) == len(z["raw"]), (
                "lag block and head data disagree on the origin count"
            )
            z["raw"] = np.concatenate([z["raw"], extra], axis=1)
            z["_e681"] = True
        return z

    base.load = load
    base.OUT = OUT_ROOT / arm
    base._CACHE.clear()


def write_bank_q(*, month: str, seed: int) -> None:
    pt = torch.load(base.OUT / f"Rhaz_{month}_s{seed}.pt", weights_only=False)
    stats = pt["stats"]
    head = head667.HazardHead(n_in=stats[0].shape[0] + 3)
    head.load_state_dict(pt["state"])
    head.eval()
    bank = base.load(month=month, kind="bank")
    bank_q, bank_p = base.predict_rows(head=head, z=bank, stats=stats, seed=seed, with_q=True)
    np.savez(base.OUT / f"bank_{month}_s{seed}.npz", bank_q=bank_q, bank_p=bank_p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=sorted(BLOCK_OF))
    ap.add_argument("month")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--width", type=int, default=0)
    a = ap.parse_args()
    assert a.month >= FIRST_TRAINED, (
        "folds before 2026-01 use the fallback head, which no arm changes"
    )
    width = WIDTH_OF.get(a.arm) or a.width
    assert width or a.arm == "L0", "the placebo arms need --width"
    earlier = [m for m in base.MONTHS if m < a.month]
    for m in [*earlier, a.month]:
        assert (lags.OUT / f"lags_{m}.npz").exists(), f"run e00681_lags.py {m} first"
    install(arm=a.arm, width=width)
    for seed in a.seeds:
        t0 = time.time()
        print(base.fit_rhaz(month=a.month, seed=seed, train_months=earlier), flush=True)
        write_bank_q(month=a.month, seed=seed)
        if a.arm == "L0":  # gate: the refit must reproduce experiment 667's stored head
            stored = np.load(base.dat.ROOT / "prevailing" / "fits" / f"Rhaz_{a.month}_s{seed}.npz")
            mine = np.load(base.OUT / f"Rhaz_{a.month}_s{seed}.npz")
            gap = float(np.abs(stored["real_q"] - mine["real_q"]).max())
            print(f"L0 reproduces the stored head: max |difference| {gap:.2e}", flush=True)
            assert gap < 1e-4, "L0 does not reproduce experiment 667's stored head"
        print(f"{a.arm} {a.month} s{seed} width {width}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
