"""Experiment 730: thread pinning, the data folders, and the folders of the experiments it imports.

Importing this module sets every numerical library to one thread (before any of them loads) and
appends, once, the experiment folders whose code experiment 730 imports to `sys.path`: experiment
729 and the experiments that 729 imports, and experiment 545 and the experiments that 545 imports.
No file of those folders is edited. The data folder of this experiment is the only folder that any
stage writes to. The one file of experiment 729's data folder that this experiment reads is
`PRIVATE/splits.md`, through `splits_md_path`.
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
EXPERIMENT_ID: Final = "00730"
SLUG: Final = "msa-d8-sign-detector-with-demand-head-and-masking"
FOLDER: Final = f"{EXPERIMENT_ID}-{SLUG}"
"""The experiment's folder name under `ml-experiments/` and `data/ml-experiments/`."""
FOLDER_729: Final = "00729-msa-d8-neural-untangler-synthetic-known-answers"
"""Experiment 729's folder name; only its `PRIVATE/splits.md` is ever read from its data folder."""
SIBLING_PREFIXES: Final = (
    "00729-", "00721-", "00694-", "00700-", "00697-", "00039-", "00194-", "00722-",
    "00397-", "00087-", "00418-", "00422-", "00428-", "00437-", "00448-", "00510-", "00511-",
    "00545-",
)  # fmt: skip
"""The experiment folders whose modules this experiment imports by bare name: experiment 729 and
its imports (721, 694, 700, 697, 39, 194, and 722), and experiment 545's chain (397, 087, 418, 422,
428, 437, 448, 510, 511, and 545)."""


MAX_SLOT_CORES: Final = 6
"""Slot A2 holds cores 16 to 21: at most this many single-threaded workers run at once."""


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


def private_dir(*, root: Path | None = None) -> Path:
    """Return (and create) the `PRIVATE` folder of this experiment, which holds anything dated."""
    path = (data_root() if root is None else root) / "PRIVATE"
    path.mkdir(parents=True, exist_ok=True)
    return path


def splits_md_path() -> Path:
    """Return experiment 729's private `PRIVATE/splits.md`, the one file read from its folder."""
    return (
        Path(os.environ["DATA_PATH_INTERNAL"])
        / "ml-experiments"
        / FOLDER_729
        / "PRIVATE"
        / "splits.md"
    )


def thread_count() -> int:
    """Return the number of operating-system threads of this process (Linux `/proc`)."""
    return int(Path("/proc/self/status").read_text().split("Threads:")[1].split()[0])


def pin_threads() -> None:
    """Pin every numerical library of this process to one thread.

    Sets torch's intra-op and inter-op thread counts to 1 when torch is installed, limits the BLAS
    and OpenMP pools of `threadpoolctl` to 1 when it is installed, and, under `run.sh` (which sets
    `SLOT_CPUS`), asserts that the process may run on at most `MAX_SLOT_CORES` cores (slot A2 has
    cores 16 to 21).
    """
    if os.environ.get("SLOT_CPUS") is not None:
        cores = os.sched_getaffinity(0)
        assert len(cores) <= MAX_SLOT_CORES, f"the process may run on {len(cores)} cores"
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.set_num_threads(1)
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(1)
        assert torch.get_num_threads() == 1, torch.get_num_threads()
    try:
        import threadpoolctl
    except ImportError:
        return
    threadpoolctl.threadpool_limits(limits=1)
