# ruff: noqa: E501
"""Experiment 721: put the experiment folders that this experiment imports on `sys.path`.

Importing this module is the whole interface. It appends, once, the experiment folders of the frozen
experiment-713 worktree, this branch's experiment folders, and the folders of experiments 716 and
720, so that the bare imports of `e00713_*`, `e00716_*`, `e00720_*`, and the loaders they use
resolve.
One `PYTHONPATH` entry per folder would exceed the limit on the size of an environment variable.
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
"""`THREAD_NAMES` at 1, and the setting that stops the allocator inside polars from starting 4 idle background threads."""
for _name, _value in THREAD_ENV.items():
    os.environ.setdefault(_name, _value)

HERE: Final = Path(__file__).resolve().parent
ROOT: Final = HERE.parent
WORKTREES: Final = ROOT.parents[1]
EXTRA_FOLDERS: Final = (
    "msa-f7-pv-part-1/ml-experiments/00716-msa-unmetered-pv-part-1-history-estimate",
    "msa-mva-to-mw-rung-0/ml-experiments/00720-msa-mva-to-mw-rung-0-signed-power-from-mva-meters",
)


def install() -> list[str]:
    """Append the folders to `sys.path` (once) and return the ones that were added."""
    wanted = [str(HERE)]
    wanted += [
        str(p) for p in sorted((WORKTREES / "frozen-713-7c29114a8" / "ml-experiments").glob("00*"))
    ]
    wanted += [str(p) for p in sorted(ROOT.glob("00*")) if p != HERE]
    wanted += [str(WORKTREES / e) for e in EXTRA_FOLDERS if (WORKTREES / e).is_dir()]
    added = [p for p in wanted if p not in sys.path]
    sys.path.extend(added)
    return added


install()


def thread_count() -> int:
    """Return the number of operating-system threads of this process (Linux `/proc`)."""
    return int(Path("/proc/self/status").read_text().split("Threads:")[1].split()[0])


def pin_threads() -> None:
    """Pin every numerical library of this process to one thread, and assert it for torch.

    Sets torch's intra-op and inter-op thread counts to 1. The second call raises `RuntimeError` once
    parallel work has started, which is then fine because the count was already pinned. The function
    also limits `threadpoolctl`'s pools (BLAS, OpenMP) to 1 when `threadpoolctl` is installed.
    """
    try:
        import torch  # ty: ignore[unresolved-import]
    except ImportError:
        torch = None
    if torch is not None:
        torch.set_num_threads(1)
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(1)
        assert torch.get_num_threads() == 1, torch.get_num_threads()
        assert torch.get_num_interop_threads() == 1, torch.get_num_interop_threads()
    try:
        import threadpoolctl  # ty: ignore[unresolved-import]
    except ImportError:
        return
    threadpoolctl.threadpool_limits(limits=1)
