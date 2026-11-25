"""Experiment 721: put the experiment folders that this experiment imports on `sys.path`.

Importing this module is the whole interface. It appends, once, the experiment folders of the frozen
experiment-713 worktree, this branch's experiment folders, and the folders of experiments 716 and
720, so that the bare imports of `e00713_*`, `e00716_*`, `e00720_*`, and the loaders they use
resolve.
One `PYTHONPATH` entry per folder would exceed the limit on the size of an environment variable.
"""

import sys
from pathlib import Path
from typing import Final

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
