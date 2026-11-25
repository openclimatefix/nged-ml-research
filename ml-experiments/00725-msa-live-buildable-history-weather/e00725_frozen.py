# ruff: noqa: E501
"""Experiment 725: the frozen checkout of experiment 713, the code hash, and the read-only watch.

F8 imports experiment 713's modules from a frozen checkout that the run brief puts on `PYTHONPATH`
through `EXTRA_PYTHONPATH` (see `run.sh`). The stored stage 1 states of `M1` carry the code hash
`FROZEN_CODE_HASH`, and the checkout reproduces the hash. This module asserts both facts before any
state is loaded, and computes F8's own code hash, which covers F8's modules and every imported module
of experiment 713 (through experiment 713's hash).
"""

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import e00713_arms as arms713  # ty: ignore[unresolved-import]
import e00713_inputs as inp713  # ty: ignore[unresolved-import]
import e00725_common as cm

GATE_WAIVED_FAILURES: Final = ["KA12", "KA3"]
"""The checks of experiment 713's known-answer gate that the maintainer waived (design amendment 4)."""


def assert_frozen_checkout() -> str:
    """Assert that experiment 713's modules come from a checkout that reproduces the stored code hash.

    Returns:
        The path of the folder that holds experiment 713's modules.

    Raises:
        AssertionError: If the hash differs from `FROZEN_CODE_HASH`, or the modules are the ones this
            branch holds (the branch's own copy of experiment 713 must stay off the path).
    """
    folder = Path(arms713.__file__).resolve().parent
    here = Path(__file__).resolve().parent.parent
    assert here not in folder.parents, (
        f"experiment 713's modules come from this branch's own folder {folder}, not a frozen checkout"
    )
    now = arms713.code_hash()
    assert now == cm.FROZEN_CODE_HASH, (
        f"the checkout at {folder} computes code hash {now}, not the stored {cm.FROZEN_CODE_HASH}"
    )
    return str(folder)


def code_hash() -> str:
    """Return F8's code hash: F8's own modules and experiment 713's hash."""
    assert_frozen_checkout()
    h = hashlib.sha256()
    h.update(cm.own_files_hash().encode())
    h.update(f"e00713:{arms713.code_hash()}".encode())
    return h.hexdigest()[:16]


def code_record() -> dict:
    """Return what a log records about the code: both hashes, the commit, and whether the folder is dirty."""
    return {
        "code_hash": code_hash(),
        "e00713_code_hash": arms713.code_hash(),
        "git_commit": cm._git("rev-parse", "HEAD"),
        "git_dirty": bool(cm._git("status", "--porcelain", "--", ".")),
    }


def check_code(*, earlier: dict, accept: Sequence[str], record: dict, what: str) -> None:
    """Check a record of F8's own code (see `e00725_common.check_code`)."""
    cm.check_code(
        earlier=earlier, accept=accept, record=record, what=what, current=code_hash(),
        frozen=arms713.code_hash(),
    )  # fmt: skip


def check_warm_start(*, stage1_log: dict, what: str) -> None:
    """Assert that the stage 1 state of experiment 713 carries the stored code hash.

    No override exists: a state fitted by other code is never a warm start of an F8 fit.
    """
    old = stage1_log.get("code_hash")
    assert old == cm.FROZEN_CODE_HASH, (
        f"{what} was fitted by code with hash {old}, not {cm.FROZEN_CODE_HASH}"
    )
    assert arms713.code_hash() == old, (
        "the frozen checkout does not reproduce the state's code hash"
    )


def stage1_state_file(*, data_dir: Path, month: str, seed: int) -> Path:
    """Return experiment 713's stage 1 state of `M*` (read only)."""
    return arms713.model_file(data_dir=data_dir, arm=cm.M_STAR, month=month, seed=seed, stage=1)


def stage_log_713(*, data_dir: Path, month: str, seed: int, stage: int) -> Path:
    """Return the log of one stage of experiment 713's fit of `M*`."""
    path = arms713.log_file(data_dir=data_dir, arm=cm.M_STAR, month=month, seed=seed, kind="fit")
    return path.with_name(f"{path.stem}_stage{stage}.json")


def gate713_summary(*, data_dir: Path) -> dict:
    """Return a summary of experiment 713's known-answer gate record, and refuse a record at another code hash.

    The gate failed KA3 and KA12 and the maintainer waived both. The record is accepted when it
    carries the frozen code hash and either passes all five boolean checks or names exactly the two
    waived failures.
    """
    path = arms713.root(data_dir=data_dir) / "work" / "known_answers_passed.json"
    assert path.exists(), (
        f"{path} is missing: experiment 713's gate has neither passed nor been waived"
    )
    record = json.loads(path.read_text())
    code = record.get("code") or {}
    assert code.get("code_hash") == cm.FROZEN_CODE_HASH, (
        f"the gate record holds code hash {code.get('code_hash')}, not {cm.FROZEN_CODE_HASH}"
    )
    failed = sorted(
        name
        for name in ("KA2", "KA3", "KA4", "KA10", "KA12")
        if not (record.get(name) or {}).get("pass", False)
    )
    waived = bool(record.get("waived"))
    assert not failed or (waived and failed == GATE_WAIVED_FAILURES), (
        f"experiment 713's gate has failed checks {failed} and waived={waived}"
    )
    return {"code_hash": code["code_hash"], "failed_checks": failed, "waived": waived}


def install_own_cache(*, data_dir: Path) -> Path:
    """Point experiment 713's dense-array loaders at F8's own cache folder, seeded with copies of 713's files.

    The loaders of experiment 713 write the dense ERA5 and B1 arrays into `e00713_inputs.CACHE` the
    first time they build them. The redirect comes before the first call, so nothing is written into
    experiment 713's or 632's data. The files are copies and not hard links, so a rewrite cannot change
    experiment 713's file.
    """
    own = cm.own_dense_cache(data_dir=data_dir)
    own.mkdir(parents=True, exist_ok=True)
    old = Path(inp713.CACHE)
    if old.exists() and old != own:
        for src in sorted(old.glob("*.npy")):
            dest = own / src.name
            if not dest.exists():
                tmp = dest.with_name(dest.name + ".tmp")
                shutil.copy2(src, tmp)
                tmp.rename(dest)
    inp713.CACHE = own
    return own


def mtimes_of(*, folder: Path) -> dict[str, int]:
    """Return the modification time (ns) of every file and folder under `folder`."""
    out: dict[str, int] = {}
    if not folder.exists():
        return out
    for base, dirs, files in os.walk(folder):
        for name in (*dirs, *files):
            path = Path(base) / name
            out[str(path)] = path.stat().st_mtime_ns
    out[str(folder)] = folder.stat().st_mtime_ns
    return out


def work_snapshot_713(*, data_dir: Path) -> str:
    """Return a hash of the path, size, and modification time of every file under experiment 713's `work` folder.

    The design asserts that this hash is unchanged after the first F8 fit, so that F8 has written
    nothing into experiment 713's folder. The hash is recorded in F8's phase 0 file.
    """
    h = hashlib.sha256()
    folder = arms713.root(data_dir=data_dir) / "work"
    for base, dirs, files in os.walk(folder):
        dirs.sort()
        for name in sorted(files):
            path = Path(base) / name
            st = path.stat()
            h.update(f"{path} {st.st_size} {st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def read_files_713(*, data_dir: Path, month: str, seed: int) -> list[Path]:
    """Return the files of experiment 713's data that an F8 fit reads."""
    root = arms713.root(data_dir=data_dir)
    return [
        root / "work" / "standardiser" / f"{month}.sha256",
        arms713.selection_file(data_dir=data_dir),
        stage1_state_file(data_dir=data_dir, month=month, seed=seed),
        stage_log_713(data_dir=data_dir, month=month, seed=seed, stage=1),
    ]


def watched_mtimes(*, files: Sequence[Path]) -> dict[str, int]:
    """Return the modification times of `files`, which must exist."""
    return {str(f): f.stat().st_mtime_ns for f in files}


GPU_BUDGET_BYTES: Final = 6 * 1024**3
"""Each GPU process of F8 stays at or under 6 GB (design section 9)."""


def limit_gpu_memory(*, device: str) -> None:
    """Cap this process's CUDA allocator at 6 GB, so that an overrun raises an out-of-memory error at once."""
    import torch  # ty: ignore[unresolved-import]

    if device.startswith("cuda"):
        index = torch.device(device).index or 0
        total = torch.cuda.get_device_properties(index).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_BUDGET_BYTES / total), index)


def nvidia_smi_process_mb() -> int | None:
    """Return the GPU memory in MB that `nvidia-smi` reports for this process, or None."""
    import os
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
    except subprocess.CalledProcessError, FileNotFoundError:
        return None
    for line in out.splitlines():
        pid, _, mem = line.partition(",")
        if pid.strip() == str(os.getpid()):
            return int(mem.strip())
    return None
