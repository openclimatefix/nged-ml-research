# ruff: noqa: E501
"""Experiment 616, check 1: do the stored stage-`pre` weights load and reproduce their logged validation NLL?

Each pretraining log line of experiment 565 (arm D5) records the best validation negative
log-likelihood (NLL) of one fold and seed. This script rebuilds the fold's pretraining rows, draws the
same 200,000-row validation subset that `train` draws (a generator seeded with the weight seed), and
recomputes the NLL of the stored weights. Experiment 554's arm M0 logged no pretraining NLL, so its
weights are only checked to load strictly and to give a finite NLL.

Usage: python e00616_reproduce.py [fold ...]
"""

import json
import re
import sys

import e00397_models as base  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00616_lib as lib
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = ML_EXPERIMENTS_DATA_DIR / "00616-lag-free-network-pv-capacity-probe-on-stored-weights"
LOG_D5 = sorted((lib.W565 / "logs" / "D5").glob("*"))
LOG_RE = re.compile(r"pretrain D5 (\d{4}-\d{2}) s(\d+): (\d+) epochs, val NLL ([0-9.]+)")


def logged() -> dict[tuple[str, int], float]:
    """Return the logged best validation NLL of each D5 fold and seed."""
    found: dict[tuple[str, int], float] = {}
    for path in [*LOG_D5, *(lib.W565 / "logs").glob("*.log")]:
        if path.is_file():
            for m in LOG_RE.finditer(path.read_text(errors="ignore")):
                found[(m[1], int(m[2]))] = float(m[4])
    return found


def main() -> None:
    """Recompute the validation NLL of every stored D5 and M0 pretrained network of the given folds."""
    torch.set_num_threads(4)
    folds = sys.argv[1:] or list(lib.FOLDS)
    log = logged()
    print("logged D5 entries:", len(log))
    result = []
    for month in folds:
        fold = lib.load_fold(month=month)
        pre = fold.pre
        z = (
            (pre.y - fold.rows.target_mean[pre.primary]) / fold.rows.target_std[pre.primary]
        ).astype(np.float32)
        val_all = np.nonzero(pre.is_val)[0]
        for arm in lib.ARMS:
            for seed in range(10):
                if not lib.weights_path(arm=arm, month=month, seed=seed).exists():
                    continue
                net = lib.load_net(arm=arm, month=month, seed=seed)
                gen = torch.Generator(device="cpu").manual_seed(seed)
                val = val_all
                if len(val) > c.VALIDATION_ROWS:
                    pick = torch.randperm(len(val), generator=gen)[: c.VALIDATION_ROWS]
                    val = val[pick.numpy()]
                x = lib.with_tau(
                    arm=arm, x=pre.x[val], valid_us=lib.valid_us_of(target_idx=pre.target_idx[val])
                )
                nll = []
                with torch.no_grad():
                    for a in range(0, len(val), 16_384):
                        sl = slice(a, a + 16_384)
                        out = net(torch.from_numpy(x[sl]), torch.from_numpy(pre.primary[val][sl]))
                        nll.append(
                            base.mixture_nll(out=out, z=torch.from_numpy(z[val][sl]), components=3)
                        )
                value = float(torch.cat(nll).mean())
                row = {
                    "fold": month,
                    "arm": arm,
                    "seed": seed,
                    "nll": value,
                    "logged": log.get((month, seed)) if arm == "D5" else None,
                }
                result.append(row)
                print(row, flush=True)
    (OUT / "reproduce_val_nll.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
