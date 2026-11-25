# ruff: noqa: E501, D103
"""Experiment 713: prove that the attention code changes no fit, forecast, or score. Reads only; no GPU.

Prints the code hash of the fit modules now and the code hash recorded in every stored stage 3 log of the
arms `M1`, `E`, and `P`, and the sha256 of the stored forecast files, and fails if any recorded hash differs from
the current one. The bit-identical forward pass is asserted inside `e00713_attention.py extract` on every case.

Usage: e00713_attention_proof.py
"""

import hashlib
import json
import sys

import e00713_arms as arms
from studies.sources import ML_EXPERIMENTS_DATA_DIR


def main() -> int:
    now = arms.code_hash()
    print("code hash now:", now)
    bad = 0
    for arm in ("M1", "E", "P"):
        for month in arms.ALL_FOLDS:
            for seed in arms.SCREEN_SEEDS:
                log = arms.log_file(
                    data_dir=ML_EXPERIMENTS_DATA_DIR, arm=arm, month=month, seed=seed, kind="fit"
                )
                path = log.with_name(f"{log.stem}_stage3.json")
                recorded = json.loads(path.read_text()).get("code_hash")
                fits = arms.fit_file(
                    data_dir=ML_EXPERIMENTS_DATA_DIR, arm=arm, month=month, seed=seed
                )
                digest = hashlib.sha256(fits.read_bytes()).hexdigest()[:12]
                flag = "" if recorded == now else "  DIFFERS"
                bad += recorded != now
                print(f"{arm} {month} s{seed}: recorded {recorded}, forecast sha256 {digest}{flag}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
