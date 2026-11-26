"""Experiment 733: thread pinning, the data folders, and the folders of the experiments it imports.

Importing this module puts the folder of experiment 730 and its siblings on `sys.path` through
experiment 730's `e00730_paths` (which pins every numerical library to one thread), then the folders
of experiments 713, 716, 725, and 707 and of the experiments that experiment 713's recipe imports,
and last the folder of experiment 732. No file of an earlier experiment is edited.

Experiment 732 is not merged yet, so its code is found in the first of these places that holds a
folder named `00732-*`: this checkout's `ml-experiments/`, the folder that `E00733_732_CODE` names,
and the `exp-untangler-plan` worktree beside this one. Every stage reads experiment 732's data
folder (`root_732`) and writes nothing there. The data folder of this experiment is the only folder
that any stage writes to.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Final

HERE: Final = Path(__file__).resolve().parent
ROOT: Final = HERE.parent
EXPERIMENT_ID: Final = "00733"
SLUG: Final = "msa-forecaster-with-untangled-d8-history"
FOLDER: Final = f"{EXPERIMENT_ID}-{SLUG}"
"""The experiment's folder name under `ml-experiments/` and `data/ml-experiments/`."""
FOLDER_713: Final = "00713-msa-axial-attention-demand-network-part-1"
FOLDER_716: Final = "00716-msa-unmetered-pv-part-1-history-estimate"
FOLDER_730: Final = "00730-msa-d8-sign-detector-with-demand-head-and-masking"
FOLDER_732: Final = "00732-msa-d8-untangling-network-synthetic-injections"
FOLDER_729: Final = "00729-msa-d8-neural-untangler-synthetic-known-answers"
FOLDER_707: Final = "00707-ens-backfill-pretraining-gain-against-ens-history-length"
FOLDER_607: Final = "00607-bakeoff-extend-cached-inputs-to-2026-10-31"
CODE_732_ENV: Final = "E00733_732_CODE"
"""The environment variable that may name a folder holding experiment 732's code."""
UNTANGLER_WORKTREE: Final = "exp-untangler-plan"
"""The sibling worktree that holds experiment 732's code until the experiment merges."""
SIBLING_PREFIXES: Final = ("00713-", "00716-", "00717-", "00725-", "00707-", "00709-", "00667-")
"""Folders whose modules this experiment imports by bare name, beyond experiment 713's recipe."""

HERE_STR: Final = str(HERE)
if HERE_STR not in sys.path:
    sys.path.insert(0, HERE_STR)


def _extend_path(*, folders: list[Path]) -> list[str]:
    """Append each existing folder to `sys.path` once, and return the ones added."""
    added = []
    for folder in folders:
        text = str(folder)
        if folder.is_dir() and text not in sys.path:
            sys.path.append(text)
            added.append(text)
    return added


def recipe_numbers() -> tuple[int, ...]:
    """Return experiment 713's `RECIPE_EXPERIMENTS`, read from its source without importing it."""
    source = (ROOT / FOLDER_713 / "e00713_arms.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "RECIPE_EXPERIMENTS"
            and node.value is not None
        ):
            return tuple(ast.literal_eval(node.value))
    msg = "RECIPE_EXPERIMENTS was not found in experiment 713's e00713_arms.py"
    raise LookupError(msg)


def code_732_folder() -> Path | None:
    """Return the folder that holds experiment 732's code, or None when no checkout has it."""
    candidates = [ROOT / FOLDER_732]
    named = os.environ.get(CODE_732_ENV)
    if named:
        candidates.append(Path(named))
    worktrees = ROOT.parent.parent.parent / ".claude" / "worktrees"
    candidates.append(worktrees / UNTANGLER_WORKTREE / "ml-experiments" / FOLDER_732)
    candidates.append(ROOT.parent.parent / UNTANGLER_WORKTREE / "ml-experiments" / FOLDER_732)
    for path in candidates:
        if (path / "e00732_core.py").is_file():
            return path
    return None


def install() -> list[str]:
    """Put the folders of every experiment whose code this one imports on `sys.path`, once."""
    wanted: list[Path] = []
    for prefix in SIBLING_PREFIXES:
        wanted += sorted(p for p in ROOT.glob(f"{prefix}*") if p.is_dir())
    for number in recipe_numbers():
        wanted += sorted(p for p in ROOT.glob(f"{number:05d}-*") if p.is_dir())
    wanted.append(ROOT)  # `engine_constants` lives in `ml-experiments/`
    added = _extend_path(folders=wanted)
    code_732 = code_732_folder()
    if code_732 is not None:
        added += _extend_path(folders=[code_732])
    return added


_FOLDER_730_STR: Final = str(ROOT / FOLDER_730)
if _FOLDER_730_STR not in sys.path:
    sys.path.insert(0, _FOLDER_730_STR)
import e00730_paths as paths730  # noqa: E402  (pins threads, adds experiments 729 and 730)

install()

THREAD_ENV: Final = paths730.THREAD_ENV
MAX_SLOT_CORES: Final = paths730.MAX_SLOT_CORES
pin_threads = paths730.pin_threads
thread_count = paths730.thread_count


def _internal_root() -> Path:
    """Return the `ml-experiments` folder under `DATA_PATH_INTERNAL`."""
    return Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"


def data_root() -> Path:
    """Return this experiment's private data folder under `DATA_PATH_INTERNAL`."""
    return _internal_root() / FOLDER


def root_713() -> Path:
    """Return experiment 713's data folder, which every stage reads and none writes."""
    return _internal_root() / FOLDER_713


def ens_summary_dir() -> Path:
    """Return the folder of experiment 607's ensemble run summaries, read only."""
    return _internal_root() / FOLDER_607 / "work" / "ens_summary"


def ens_backfill_summary_dir() -> Path:
    """Return the folder of experiment 707's back-filled ensemble run summaries.

    (2020-10-15 to 2024-03-31), read only.

    Experiment 707 built these files from the `NWP` Delta table, in the format of experiment 607's
    summaries, so the synthetic world's base days (before 2024-04-01) read ensemble runs without a
    download.
    """
    return _internal_root() / FOLDER_707 / "work" / "backfill" / "ens_summary"


def ens_summary_dirs() -> tuple[Path, Path]:
    """Return the folders searched for a run summary: experiment 607's (from 2024-04-01).

    Then experiment 707's back-fill.
    """
    return (ens_summary_dir(), ens_backfill_summary_dir())


def root_716() -> Path:
    """Return experiment 716's data folder, read only."""
    return _internal_root() / FOLDER_716


def root_730() -> Path:
    """Return experiment 730's data folder, read only."""
    return _internal_root() / FOLDER_730


def root_732() -> Path:
    """Return experiment 732's data folder, read only."""
    return _internal_root() / FOLDER_732


def private_dir(*, root: Path | None = None) -> Path:
    """Return (and create) the `PRIVATE` folder of this experiment."""
    path = (data_root() if root is None else root) / "PRIVATE"
    path.mkdir(parents=True, exist_ok=True)
    return path


def channels_dir(*, root: Path | None = None) -> Path:
    """Return the folder of the channel table, `channels.json`'s neighbour, under `root`."""
    return (data_root() if root is None else root) / "channels"


def rerun_dir(*, root: Path | None = None) -> Path:
    """Return the folder of the re-run outputs per origin."""
    return channels_dir(root=root) / "rerun"


def channels_json(*, root: Path | None = None) -> Path:
    """Return the path of `channels.json` (the rule's choice)."""
    return (data_root() if root is None else root) / "channels.json"


def table_path(*, root: Path | None = None) -> Path:
    """Return the path of the channel table (untangled channels on the history grid)."""
    return channels_dir(root=root) / "table.parquet"


def splits_md_path() -> Path:
    """Return experiment 729's private `PRIVATE/splits.md`, read only."""
    return paths730.splits_md_path()


def forbidden_write_roots() -> tuple[Path, ...]:
    """Return the data folders of the earlier experiments, which no stage of this one writes to."""
    return tuple(
        _internal_root() / f
        for f in (FOLDER_713, FOLDER_716, FOLDER_729, FOLDER_730, FOLDER_732, FOLDER_707)
    )
