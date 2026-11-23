# ruff: noqa: E501
"""Make part 2's scripts fail loudly when part 1's code is missing or is not the code that `run.sh` named.

Part 2 imports part 1's scoring, figure, and report code (the `e00716_*` modules), which lives in
another folder. `run.sh` puts the folder that `EXTRA_PYTHONPATH` names ahead of every other folder on
`PYTHONPATH`. If that folder is gone, Python would silently import a stale copy of `e00716_*` from an
older checkout, so every script of part 2 calls `require_part1` on each part 1 module it imports.

Merge order: part 1 (experiment 716) merges first, and part 2 is then rebased onto `main`. Before the
merge, `ty check` cannot resolve the `e00716_*` imports, and the imports carry `ty: ignore` comments.
"""

import os
from pathlib import Path
from types import ModuleType


def require_part1(*modules: ModuleType) -> None:
    """Assert that each part 1 module was imported from the folder that `EXTRA_PYTHONPATH` names."""
    extra = os.environ.get("EXTRA_PYTHONPATH")
    assert extra, (
        "set EXTRA_PYTHONPATH to the 00716 folder of the part 1 checkout (run.sh does this)"
    )
    folder = Path(extra).resolve()
    assert folder.is_dir(), f"EXTRA_PYTHONPATH names {folder}, which does not exist"
    for module in modules:
        where = Path(str(module.__file__)).resolve().parent
        assert where == folder, (
            f"{module.__name__} was imported from {where}, not from {folder}: refusing to use a stale copy"
        )
