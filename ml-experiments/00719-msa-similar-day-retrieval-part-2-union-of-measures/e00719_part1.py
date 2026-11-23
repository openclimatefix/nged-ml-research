"""Experiment 719: put experiment 718's code folder on `sys.path`, then expose its modules.

Experiment 719 holds no fit, score, or figure code of its own. Experiment 718 (part 1) must be
merged, or checked out, first. This module finds its folder in this order: the environment
variable `PART_1_FOLDER`, then a sibling folder named `00718-*`.
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
    msg = f"experiment 718 not found; merge part 1 or set PART_1_FOLDER. Tried: {tried}"
    raise FileNotFoundError(msg)


PART_1_FOLDER = find_part_1()
if str(PART_1_FOLDER) not in sys.path:
    sys.path.append(str(PART_1_FOLDER))


def run_part_1_script(*, module: str, takes_part: bool) -> int:
    """Run one of experiment 718's scripts as `__main__`, inserting part 2 after the command."""
    import runpy

    args = sys.argv[1:]
    if takes_part and args and args[0] != "self-test" and "2" not in args:
        args = [args[0], "2", *args[1:]]
    sys.argv = [f"{module}.py", *args]
    runpy.run_module(module, run_name="__main__")
    return 0
