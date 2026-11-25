"""A tiny synthetic data folder: banks, gate, gallery check, choices, and invented predictions."""

import contextlib
import json
from pathlib import Path

import e00729_core as core729
import e00730_banks as banks
import e00730_core as core
import numpy as np
from world730 import make_calendar


def bank_inputs() -> banks.BankInputs:
    cal = make_calendar(n_days=62)
    keys = np.array([str(d)[:7] for d in cal.dates])
    months = {"training": ["2022-03"], "validation": ["2022-04"], "test": ["2022-04"]}
    use = {
        "training": {"2022-03": keys == "2022-03"},
        "validation": {"2022-04": keys == "2022-04"},
        "test": {"2022-04": keys == "2022-04"},
    }
    return banks.BankInputs(cal=cal, months=months, use=use)


def make_root(tmp_path: Path, monkeypatch) -> tuple[Path, np.ndarray, object]:  # noqa: ANN001
    """Build the banks, open the run and gate markers by hand; return root, holidays, calendar."""
    monkeypatch.setattr(core729, "SEGMENT_DRAWS_TRAIN", 4)
    monkeypatch.setattr(core729, "SEGMENT_DRAWS_EVAL", 4)
    root = tmp_path / "exp"
    root.mkdir()
    inputs = bank_inputs()
    with contextlib.suppress(banks.StoppedError):
        banks.stage_banks(root=root, inputs=inputs, train_seeds=[0], log=lambda _: None)
    for kind in ("shifted",):
        assert banks.bank_path(root=root, kind=kind).exists()
    marker = banks.run_marker_payload()
    banks.write_json(path=root / banks.RUN_MARKER, payload=marker)
    banks.write_json(path=root / "scores" / "gate.json", payload={"pass": True, "marker": marker})
    return root, inputs.cal.holidays, inputs.cal


def open_gallery(root: Path) -> None:
    folder = root / "PRIVATE" / "gallery"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "panel_check.json").write_text(
        json.dumps({"panels": dict.fromkeys(core.PANEL_REQUIREMENTS, True)})
    )


def write_choices(root: Path, *, style: str = "K", arch: str = "bilstm") -> None:
    x = np.where(np.random.default_rng(0).random(20_000) < 0.05, -1.0, 5.0)
    r = np.random.default_rng(1).uniform(0.5, 30.0, x.size)
    core.write_limits_file(path=root / "limits.json", limits=core.compute_limits(x=x, r=r))
    core.write_margin_file(path=root / "margin.json", floor=0.001, val_brier_s_by_seed=[0.02] * 5)
    banks.write_json(
        path=root / "selection.json", payload={"chosen": arch, "transformer_variant": "transformer"}
    )
    banks.write_json(path=root / "style.json", payload={"chosen": style})


def write_predictions(root: Path, rows, *, arch: str = "bilstm", style: str = "K") -> None:  # noqa: ANN001
    rng = np.random.default_rng(5)
    n = rows.n
    arms = [*core.ARMS_NEW, *core.REFIT_729_ARMS, *(f"SD-{s}" for s in core.STYLES)]
    for arm in dict.fromkeys(arms):
        for a in (arch, "transformer", "bilstm"):
            for seed in (*core.ARCH_SEEDS, 3, 4):
                folder = root / "predictions" / f"{arm.replace('*', 'star')}_{a}_s{seed}"
                for name in ("test", "shifted", "validation"):
                    size = n
                    arrays = {
                        "p": np.clip((rows.x < 0) * 0.7 + rng.uniform(0, 0.3, size), 0, 1)[:size],
                        "demand_mu": rows.x * 0 + 50.0 + rng.normal(size=size),
                    }
                    if arm in core.REFIT_729_ARMS:
                        m = np.sqrt(np.maximum(rows.r**2 - core729.QHAT_POINTS**2, 0.0))
                        arrays |= {
                            "mu_i": m + rng.normal(scale=0.2, size=size),
                            "s_i": np.full(size, 0.6),
                            "mu_r": -m + rng.normal(scale=0.2, size=size),
                            "s_r": np.full(size, 0.6),
                        }
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez(folder / f"{name}.npz", **arrays)
                for k in core.CAUSAL_K:
                    np.savez(folder / f"test_causal_k{k}.npz", p=rng.uniform(size=n))
                for real in ("r1", "k1", "k2", "eras"):
                    np.savez(
                        folder / f"real_{real}.npz",
                        p=rng.uniform(size=n) * 0.1,
                        demand_mu=rng.normal(size=n) + 50.0,
                    )
    del style


def write_real(root: Path, *, passed: bool = True) -> None:
    """Write a minimal `scores/real.json`: H2 read, so that the score stage may run."""
    banks.write_json(path=root / "scores" / "real.json", payload={"h2": {"pass": passed}})
