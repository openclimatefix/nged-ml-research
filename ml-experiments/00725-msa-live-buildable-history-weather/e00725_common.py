# ruff: noqa: E501
"""Experiment 725: the file layout, the arms, the frozen-checkout assertion, and the code hash.

Everything F8 writes goes under `ROOT`. Experiment 713's data folder is read only: the fit asserts that
the files it reads from there are unchanged when it ends. The module imports no experiment 713 module
at the top level, so the pure-index modules stay testable without the frozen checkout.
"""

import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00725-msa-live-buildable-history-weather"
NAME_713: Final = "00713-msa-axial-attention-demand-network-part-1"
ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
ROOT_713: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME_713
FROZEN_CODE_HASH: Final = "e85b84e2ea04b3cb"
"""The code hash that the stored `M1` states carry, as the frozen checkout `frozen-713-7c29114a8` computes it."""
M_STAR: Final = "M1"
"""Experiment 713's selected arm."""
ALL_FOLDS: Final = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
)  # fmt: skip
QUICK_FOLDS: Final = ("2025-11", "2026-01", "2026-03", "2026-06")
OTHER_FOLDS: Final = tuple(m for m in ALL_FOLDS if m not in QUICK_FOLDS)
TWO_SEEDS: Final = (0, 1)
ALL_SEEDS: Final = (0, 1, 2, 3, 4)
CAP_FIRST: Final = 60
CAP_RERUN: Final = 120
STAGES: Final = (2, 3)
CEILING_GPU_PROCESS_HOURS: Final = 12.0
HOURS_PER_FIT_SLOWEST: Final = 0.14
HOURS_PER_FIT_FASTEST: Final = 0.07
BATCH_SIZE: Final = 256
"""Experiment 713's training batch, for the derivation of optimiser steps from pairs."""


@dataclass(frozen=True)
class ArmSpec:
    """One fitted arm of F8.

    Attributes:
        name: `L`, `Z`, `E`, `L-P`, or `Z-P`.
        rule: `L`, `Z`, or `E` (see `e00725_rule`).
        patience: Early-stopping patience of stages 2 and 3.
        cap: Pass cap of the first run; the rerun cap is `CAP_RERUN` (a probe arm has no rerun).
        probe: True for the report-only budget probe.
    """

    name: str
    rule: str
    patience: int
    cap: int
    probe: bool


ARMS: Final[dict[str, ArmSpec]] = {
    "L": ArmSpec(name="L", rule="L", patience=8, cap=CAP_FIRST, probe=False),
    "Z": ArmSpec(name="Z", rule="Z", patience=8, cap=CAP_FIRST, probe=False),
    "E": ArmSpec(name="E", rule="E", patience=8, cap=CAP_FIRST, probe=False),
    "L-P": ArmSpec(name="L-P", rule="L", patience=30, cap=CAP_RERUN, probe=True),
    "Z-P": ArmSpec(name="Z-P", rule="Z", patience=30, cap=CAP_RERUN, probe=True),
}


@dataclass(frozen=True)
class PredictSpec:
    """A prediction-only reading from a stored state.

    Attributes:
        name: The arm name of the forecast file.
        state: `m_star` (experiment 713's stored stage 3 state of `M1`, read only) or `own:<arm>`
            (the stage 3 state of an F8 arm).
        rule: The history-weather rule the batch reads (`L`, `Z`, or `E`).
        target_summary_back_days: `T3`: the target row's CAMS day summary is that of UTC day D minus
            this many days; None leaves the target row as experiment 713 read it.
    """

    name: str
    state: str
    rule: str
    target_summary_back_days: int | None


PREDICT_SPECS: Final[dict[str, PredictSpec]] = {
    "S": PredictSpec(name="S", state="m_star", rule="L", target_summary_back_days=None),
    "T3-M": PredictSpec(name="T3-M", state="m_star", rule="Z", target_summary_back_days=3),
    "T3-L": PredictSpec(name="T3-L", state="own:L", rule="L", target_summary_back_days=3),
}
"""`S`: the stored `M*` on the live rule with no retraining. `T3-M` and `T3-L`: `M*` and `L` with the target row's summary moved."""


def resolve(*, arm: str) -> ArmSpec:
    """Return the specification of a fitted arm."""
    assert arm in ARMS, f"{arm!r} is not a fitted arm of F8: {sorted(ARMS)}"
    return ARMS[arm]


def work(*, data_dir: Path = ML_EXPERIMENTS_DATA_DIR) -> Path:
    """Return F8's `work` folder."""
    return data_dir / NAME / "work"


def smoke_work(*, data_dir: Path = ML_EXPERIMENTS_DATA_DIR) -> Path:
    """Return the folder that holds the outputs of the smoke fit, which are never scored."""
    return data_dir / NAME / "smoke"


def model_file(
    *, data_dir: Path, arm: str, month: str, seed: int, stage: int, smoke: bool = False
) -> Path:
    """Return the saved state of an F8 arm after `stage` (2 or 3)."""
    base = smoke_work(data_dir=data_dir) if smoke else work(data_dir=data_dir)
    return base / "models" / arm / f"{month}_s{seed}_stage{stage}.pt"


def fit_file(*, data_dir: Path, arm: str, month: str, seed: int) -> Path:
    """Return the per-seed forecast file of an F8 arm (`L`, `Z`, `E`, `S`, `T3-*`, ...)."""
    return work(data_dir=data_dir) / "fits" / arm / f"{month}_s{seed}.parquet"


def log_file(
    *, data_dir: Path, arm: str, month: str, seed: int, kind: str, smoke: bool = False
) -> Path:
    """Return a job's JSON log; `kind` is `fit` or `predict`."""
    base = smoke_work(data_dir=data_dir) if smoke else work(data_dir=data_dir)
    return base / "logs" / arm / f"{month}_s{seed}_{kind}.json"


def stage_log_file(
    *, data_dir: Path, arm: str, month: str, seed: int, stage: int, smoke: bool = False
) -> Path:
    """Return the log of one stage of an F8 fit."""
    path = log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit", smoke=smoke)
    return path.with_name(f"{path.stem}_stage{stage}.json")


def cache_dir(*, phase: str, data_dir: Path = ML_EXPERIMENTS_DATA_DIR) -> Path:
    """Return the folder of the ENS-array cache of a phase (`train` or `predict`)."""
    assert phase in {"train", "predict"}, phase
    return data_dir / NAME / "work" / "ens_cache" / phase


def own_dense_cache(*, data_dir: Path = ML_EXPERIMENTS_DATA_DIR) -> Path:
    """Return F8's own copy of the dense-array cache of experiment 713's loaders."""
    return data_dir / NAME / "cache"


def wiring_file(*, data_dir: Path = ML_EXPERIMENTS_DATA_DIR) -> Path:
    """Return the file that records the wiring checks KL1 to KL10 passing."""
    return work(data_dir=data_dir) / "wiring_checks_passed.json"


def phase0_file(*, data_dir: Path = ML_EXPERIMENTS_DATA_DIR) -> Path:
    """Return the file that records the phase 0 results: the reproduction check and the projection."""
    return work(data_dir=data_dir) / "phase0.json"


def write_json_atomic(*, obj: object, path: Path) -> None:
    """Write a JSON file under a temporary name that carries the process id, then rename it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.rename(path)


def stopping_record(*, entry: dict, stored: bool = False) -> dict:
    """Return the stopping-pass record of one stage log entry (design section 7).

    Args:
        entry: A stage entry as `e00713_fit` and `e00725_fit` write it: `pairs`, `first`, and `rerun`,
            each run holding `passes`, `best_val`, `cap_hit`, and `val_history`.
        stored: True for a stored experiment 713 record, which has no field for the optimiser steps.

    Returns:
        The passes run, the best pass (the first pass whose loss is the best under experiment 713's
        1e-5 rule), the optimiser steps at the best pass (`best pass * ceil(pairs / 256)`, an upper
        bound that counts the validation pairs), the validation loss at pass 1 and at the best pass,
        whether the cap was hit, and whether the 120-pass rerun ran.
    """
    run = entry["rerun"] or entry["first"]
    history = list(run["val_history"])
    best, best_pass = float("inf"), 0
    for i, value in enumerate(history, start=1):
        if value < best - 1e-5:
            best, best_pass = value, i
    pairs = int(entry["pairs"])
    return {
        "stage": entry["stage"],
        "passes": int(run["passes"]),
        "best_pass": best_pass,
        "steps_at_best_pass": best_pass * -(-pairs // BATCH_SIZE),
        "val_pass_1": history[0] if history else None,
        "val_best": best,
        "cap_hit_first_run": bool(entry["first"]["cap_hit"]),
        "rerun_ran": entry["rerun"] is not None,
        "formula_steps": stored,
    }


# --------------------------------------------------------------------- the frozen checkout and the code hash

OWN_FIT_FILES: Final = (
    "e00725_common.py", "e00725_rule.py", "e00725_sources.py", "e00725_batch.py", "e00725_fit.py",
    "e00725_predict.py", "e00725_frozen.py", "e00725_world.py",
)  # fmt: skip
"""The modules whose edits can change a fitted state or a prediction."""


CHECK_FILES: Final = ("e00725_checks.py", "e00725_poison.py", "e00725_weather_lookahead.py")
"""The modules of the wiring checks. They are in the gate's `checks_hash` and not in the fit hash."""


def checks_hash() -> str:
    """Return a hash of the wiring-check modules and the fit modules, which the wiring gate records."""
    here = Path(__file__).parent
    h = hashlib.sha256(own_files_hash().encode())
    for name in CHECK_FILES:
        h.update(name.encode())
        h.update((here / name).read_bytes())
    return h.hexdigest()[:16]


def write_check_json(*, obj: dict, path: Path) -> None:
    """Write a check result with the `checks_hash` of the code that produced it."""
    write_json_atomic(obj={**obj, "checks_hash": checks_hash()}, path=path)


def check_output_current(*, path: Path) -> bool:
    """Return whether a check result exists and carries the `checks_hash` of the code that is running."""
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("checks_hash") == checks_hash()
    except json.JSONDecodeError:
        return False


def assert_gate_checks_hash(*, gate: dict) -> None:
    """Refuse a wiring gate that was written by other check code than the code that is running."""
    assert gate.get("checks_hash") == checks_hash(), (
        f"the wiring gate carries checks hash {gate.get('checks_hash')}, the code has {checks_hash()}: "
        "rerun the checks (`e00725_checks.py collect` after the per-fold checks)"
    )


def _git(*args: str) -> str:
    """Return the output of a git command run in this folder, or `unavailable`."""
    try:
        return subprocess.run(
            ["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return "unavailable"


def own_files_hash() -> str:
    """Return a hash of F8's own fit and prediction modules (no experiment 713 module)."""
    here = Path(__file__).parent
    h = hashlib.sha256()
    for name in OWN_FIT_FILES:
        h.update(name.encode())
        h.update((here / name).read_bytes())
    return h.hexdigest()[:16]


def check_code(
    *, earlier: dict, accept: Sequence[str], record: dict, what: str, current: str, frozen: str
) -> None:
    """Refuse a state, log, or gate written by other code, unless an override names its exact old hash.

    No override is accepted when the frozen hash of experiment 713's code differs from the one the
    record holds.
    """
    old = earlier.get("code_hash")
    if old is None:
        msg = f"{what} carries no code record: refit it"
        raise AssertionError(msg)
    if earlier.get("e00713_code_hash") != frozen:
        msg = (
            f"{what} was written with experiment 713's code at hash {earlier.get('e00713_code_hash')}, "
            f"and the frozen checkout has hash {frozen}: no override is accepted"
        )
        raise AssertionError(msg)
    if old == current:
        return
    if old not in accept:
        msg = (
            f"{what} was written by code with hash {old}, and the code now has hash {current}. After "
            f"an edit that cannot change a fitted state, rerun with `--accept-code-change {old}`."
        )
        raise AssertionError(msg)
    entry = {
        "what": what,
        "old_hash": old,
        "new_hash": current,
        "new_commit": _git("rev-parse", "HEAD"),
    }
    if entry not in record.setdefault("code_overrides", []):
        record["code_overrides"].append(entry)
        print("code override accepted:", json.dumps(entry), flush=True)
