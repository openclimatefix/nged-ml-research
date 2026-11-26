"""A tiny synthetic bank for the tests: two months of base days with invented values.

The calendar comes from experiment 730's `tests/world730.py`, loaded by path (this module imports
it, never copies it). The segments come from experiment 729's own simulator, so they carry the
fields, the reactive floor, and the BSP coupling that the overlays read.
"""

import importlib.util
from pathlib import Path

import e00729_sim as sim
import e00732_paths
import numpy as np

_PATH = e00732_paths.ROOT / e00732_paths.FOLDER_730 / "tests" / "world730.py"
_SPEC = importlib.util.spec_from_file_location("world730_for_732", _PATH)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
make_calendar = _MODULE.make_calendar

MONTHS = ("2022-03", "2022-04")
WORLDS = ("none", "engine", "solar", "both")


def make_bank(
    *, draws_per_world: int = 3, seed: int = 0, n_days: int = 62
) -> tuple[sim.Calendar, list[sim.Segment]]:
    """Return a calendar and `draws_per_world` segments of each world type in each of two months."""
    cal = make_calendar(n_days=n_days, seed=seed)
    segments = []
    for m, month in enumerate(MONTHS):
        for w, world in enumerate(WORLDS):
            for k in range(draws_per_world):
                rng = np.random.default_rng([seed, m, w, k])
                segments.append(
                    sim.simulate_segment(
                        cal=cal,
                        month=month,
                        use=np.ones(len(cal.dates), bool),
                        world=world,
                        rng=rng,
                    )
                )
    return cal, segments


def folder(*, name: str) -> Path:
    """Return a path under the experiment folder (for tests that read the code)."""
    return e00732_paths.HERE / name
