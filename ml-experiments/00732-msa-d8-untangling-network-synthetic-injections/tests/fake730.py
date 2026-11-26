# ruff: noqa: E501
"""An invented copy of experiment 730's data folder (banks, summary, stored predictions) for the stage tests."""

import json
from pathlib import Path
from typing import Any

import e00729_banks as banks729
import e00729_score as score729
import e00732_stages as stages
import numpy as np
from world732 import make_bank, make_base, make_neighbours


def build(*, root: Path, seeds: tuple[int, ...] = (0, 1)) -> dict:
    """Write banks, summary, margin, and stored predictions under `root` and return the pieces the tests need."""
    cal, segs = make_bank(draws_per_world=3)
    digests = {}
    kinds = [("train", s) for s in seeds] + [
        ("validation", None),
        ("test", None),
        ("shifted", None),
    ]
    for kind, seed in kinds:
        bank = banks729.Bank(
            kind=kind,
            seed=seed,
            root=0,
            none_share={},
            segments=segs,
            digest=banks729.bank_digest(segments=segs),
        )
        path = root / "banks" / (f"{kind}_s{seed}.pkl" if seed is not None else f"{kind}.pkl")
        banks729.save_bank(bank=bank, path=path)
        digests[f"{kind}_s{seed}" if seed is not None else kind] = bank.digest
    (root / "banks" / "summary.json").write_text(json.dumps({"digests": digests}))
    rows = score729.rows_from_segments(segments=segs, holidays=cal.holidays)
    rng = np.random.default_rng(0)
    p_true = (rows.x < 0).astype(float)
    for arm, arch in (("S", "unet"), ("S", "bilstm"), ("SD-R", "unet")):
        for s in (0, 1, 2):
            folder = root / "predictions" / f"{arm}_{arch}_s{s}"
            folder.mkdir(parents=True)
            p = np.clip(0.8 * p_true + 0.05 + 0.02 * rng.random(rows.n), 0, 1)
            arrays = {"p": p.astype(np.float32)}
            if arm == "SD-R":
                arrays |= {
                    "demand_mu": np.abs(rows.r).astype(np.float32),
                    "demand_s": np.ones(rows.n, np.float32),
                }
            np.savez(folder / "test.npz", **arrays)
    (root / "margin.json").write_text(
        json.dumps({"margin": 0.00122, "floor": 0.0007, "val_brier_s_by_seed": [0.1]})
    )
    return {
        "cal": cal, "segments": segs, "rows": rows, "n": rows.n, "base": make_base(cal=cal),
        "neighbours": make_neighbours(cal=cal),
    }  # fmt: skip


def patch_constants(monkeypatch: Any, *, root: Path, world: dict) -> None:
    """Point the plan's constants at the invented bank (row count, canary, pooled limit, limit range)."""
    import e00730_score as score730
    import e00732_core as core

    arrays = stages.load_stored(root730=root, arm="S", arch="unet")
    canary = score730.mean_brier(p=stages.seed_mean_p(arrays=arrays), x=world["rows"].x)
    pooled = core.band_limits(x=world["rows"].x, r=world["rows"].r)["pooled"].limit
    monkeypatch.setattr(core, "N_TEST_ROWS", world["n"])
    monkeypatch.setattr(core, "BRIER_CANARY", canary)
    monkeypatch.setattr(core, "POOLED_LIMIT_730", pooled)
    monkeypatch.setattr(core, "LIMIT_RANGE", (0.0, 1.0))
    monkeypatch.setattr(core, "MIN_TRANSFER_STEPS_TEST", 1)
    monkeypatch.setattr(core, "MIN_WINDOWS_PER_AMOUNT", 1)
