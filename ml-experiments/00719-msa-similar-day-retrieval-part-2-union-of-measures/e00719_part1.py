"""Experiment 719: put experiment 718's code folder on `sys.path`, then expose its modules.

Experiment 719 holds no fit, score, or figure code of its own. Experiment 718 (part 1) must be merged,
or checked out, before experiment 719 runs. This module finds experiment 718's folder in this order:
the environment variable `PART_1_FOLDER`, then a sibling folder named `00718-*`, then the same folder
under the main checkout of a git worktree layout.
"""

import os
import sys
from pathlib import Path


def find_part_1() -> Path:
    """Return experiment 718's code folder, or raise with the places that were tried."""
    tried: list[Path] = []
    env = os.environ.get("PART_1_FOLDER")
    if env:
        tried.append(Path(env))
    here = Path(__file__).resolve().parent
    tried.extend(sorted(here.parent.glob("00718-*")))
    for folder in tried:
        if (folder / "e00718_decision.py").is_file():
            return folder
    msg = f"experiment 718's code was not found; merge part 1 first or set PART_1_FOLDER. Tried: {tried}"
    raise FileNotFoundError(msg)


PART_1_FOLDER = find_part_1()
if str(PART_1_FOLDER) not in sys.path:
    sys.path.append(str(PART_1_FOLDER))
