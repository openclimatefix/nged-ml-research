"""Experiment 729: thread pinning, the data folders, and the folders of the experiments it imports.

Importing this module sets every numerical library to one thread (before any of them loads) and
appends, once, the experiment folders whose code experiment 729 imports to `sys.path`. Those are
experiment 721 (the reviewed score code: the Kish number, the widening, and the write-once file
pattern), experiment 694 (the M2 tracker), and experiment 700 (the dot chart). A script that needs
a sibling experiment skips cleanly when the folder is absent, so CI passes without them.
"""

import contextlib
import os
import sys
from pathlib import Path
from typing import Final

THREAD_NAMES: Final = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS",
    "POLARS_ASYNC_THREAD_COUNT",
)  # fmt: skip
"""Variables that every process of this experiment sets to 1 before any library is imported."""
THREAD_ENV: Final = {
    **dict.fromkeys(THREAD_NAMES, "1"),
    "_RJEM_MALLOC_CONF": "background_thread:false,narenas:1",
}
"""`THREAD_NAMES` at 1, and the allocator setting that stops polars starting idle threads."""
for _name, _value in THREAD_ENV.items():
    os.environ.setdefault(_name, _value)

HERE: Final = Path(__file__).resolve().parent
ROOT: Final = HERE.parent
EXPERIMENT_ID: Final = "00729"
SLUG: Final = "msa-d8-neural-untangler-synthetic-known-answers"
FOLDER: Final = f"{EXPERIMENT_ID}-{SLUG}"
"""The experiment's folder name under `ml-experiments/` and `data/ml-experiments/`."""
SIBLING_PREFIXES: Final = ("00721-", "00694-", "00700-", "00697-", "00039-", "00194-", "00722-")
"""The experiment folders whose modules this experiment imports by bare name."""


def install() -> list[str]:
    """Append this folder and the sibling experiment folders to `sys.path`, once."""
    wanted = [str(HERE)]
    for prefix in SIBLING_PREFIXES:
        wanted += [str(p) for p in sorted(ROOT.glob(f"{prefix}*")) if p.is_dir()]
    wanted.append(str(ROOT))  # `engine_constants` lives in `ml-experiments/`
    added = [p for p in wanted if p not in sys.path]
    sys.path.extend(added)
    return added


install()


def sibling_available(*, prefix: str) -> bool:
    """Return whether the experiment folder that starts with `prefix` exists in this checkout."""
    return any(p.is_dir() for p in ROOT.glob(f"{prefix}*"))


def data_root() -> Path:
    """Return this experiment's private data folder under `DATA_PATH_INTERNAL`."""
    return Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments" / FOLDER


def private_dir() -> Path:
    """Return (and create) the `PRIVATE` folder, which holds anything dated."""
    path = data_root() / "PRIVATE"
    path.mkdir(parents=True, exist_ok=True)
    return path


def thread_count() -> int:
    """Return the number of operating-system threads of this process (Linux `/proc`)."""
    return int(Path("/proc/self/status").read_text().split("Threads:")[1].split()[0])


def pin_threads() -> None:
    """Pin every numerical library of this process to one thread.

    Sets torch's intra-op and inter-op thread counts to 1 when torch is installed, limits the BLAS
    and OpenMP pools of `threadpoolctl` to 1 when it is installed, and, under `run.sh` (which sets
    `SLOT_CPU`), asserts that the process may run on one core only.
    """
    if os.environ.get("SLOT_CPU") is not None:
        cores = os.sched_getaffinity(0)
        assert len(cores) == 1, f"the process may run on {len(cores)} cores; run.sh pins one"
    try:
        import torch  # ty: ignore[unresolved-import]
    except ImportError:
        torch = None
    if torch is not None:
        torch.set_num_threads(1)
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(1)
        assert torch.get_num_threads() == 1, torch.get_num_threads()
    try:
        import threadpoolctl  # ty: ignore[unresolved-import]
    except ImportError:
        return
    threadpoolctl.threadpool_limits(limits=1)
