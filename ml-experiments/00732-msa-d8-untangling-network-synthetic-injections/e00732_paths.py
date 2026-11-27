"""Experiment 732: thread pinning, the data folders, and the folders of the experiments it imports.

Importing this module puts the folder of experiment 730 on `sys.path`, imports experiment 730's
`e00730_paths` (which pins every numerical library to one thread and puts experiments 729 and 730's
imports on `sys.path`), and adds this experiment's own folder. No file of an earlier experiment is
edited. The data folder of this experiment is the only folder that any stage writes to. Experiment
730's data folder is read only, through `root_730`, and experiment 729's `PRIVATE/splits.md` is read
only, through `splits_md_path`.
"""

import os
import sys
from pathlib import Path
from typing import Final

HERE: Final = Path(__file__).resolve().parent
ROOT: Final = HERE.parent
EXPERIMENT_ID: Final = "00732"
SLUG: Final = "msa-d8-untangling-network-synthetic-injections"
FOLDER: Final = f"{EXPERIMENT_ID}-{SLUG}"
"""The experiment's folder name under `ml-experiments/` and `data/ml-experiments/`."""
FOLDER_730: Final = "00730-msa-d8-sign-detector-with-demand-head-and-masking"
FOLDER_729: Final = "00729-msa-d8-neural-untangler-synthetic-known-answers"

for _folder in (ROOT / FOLDER_730, HERE):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import e00730_paths as paths730  # noqa: E402

THREAD_ENV: Final = paths730.THREAD_ENV
MAX_SLOT_CORES: Final = paths730.MAX_SLOT_CORES
pin_threads = paths730.pin_threads
thread_count = paths730.thread_count


def data_root() -> Path:
    """Return this experiment's private data folder under `DATA_PATH_INTERNAL`."""
    return Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments" / FOLDER


def root_730() -> Path:
    """Return experiment 730's data folder, which every stage reads and none writes."""
    return Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments" / FOLDER_730


def private_dir(*, root: Path | None = None) -> Path:
    """Return (and create) the `PRIVATE` folder of this experiment."""
    path = (data_root() if root is None else root) / "PRIVATE"
    path.mkdir(parents=True, exist_ok=True)
    return path


def splits_md_path() -> Path:
    """Return experiment 729's private `PRIVATE/splits.md`, read only."""
    return paths730.splits_md_path()
