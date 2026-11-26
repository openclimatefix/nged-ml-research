# ruff: noqa: E501
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


def make_neighbours(*, cal: sim.Calendar, seed: int = 0) -> dict[str, np.ndarray]:
    """Return D3, D5, and D14 on the calendar, in the shape that `neighbours.build_arrays` returns.

    The series start on a master grid that begins a few days before the calendar and ends a few days
    after it, with invented values in points (D5 signed with one export day, D14 with one missing
    day), and are placed on the calendar by the real `place_on_calendar`, as the real loader does.
    """
    import e00732_neighbours as nb

    rng = np.random.default_rng([seed, 99])
    first = cal.dates[0] - np.timedelta64(3, "D")
    n_grid_days = len(cal.dates) + 8
    hh = np.tile(np.arange(48), n_grid_days)
    day = np.repeat(np.arange(n_grid_days), 48)
    shape = np.sin(np.pi * (hh - 10) / 48.0) ** 2
    grids = {
        "D3": 90.0 + 25.0 * shape + rng.normal(0, 1.0, hh.size),
        "D5": 70.0 + 20.0 * shape + rng.normal(0, 1.0, hh.size),
        "D14": 100.0 + 30.0 * shape + rng.normal(0, 1.0, hh.size),
    }
    grids["D5"][(day == 20)] = -15.0 + rng.normal(0, 1.0, (day == 20).sum())  # one export day
    grids["D14"][(day == 30)] = np.nan  # one missing day
    return {
        name: nb.place_on_calendar(grid=grid, dates=cal.dates, first_day=first)
        for name, grid in grids.items()
    }


def make_base(*, cal: sim.Calendar) -> object:
    """Return D8's base calendar as the stages read it: the first 60% of the days train, then
    validation, then test.
    """
    import e00732_stages as stages

    n = len(cal.dates)
    split = np.array(["training"] * n, dtype=object)
    split[int(0.6 * n) : int(0.8 * n)] = "validation"
    split[int(0.8 * n) :] = "test"
    return stages.BaseDays(
        dates=cal.dates, reading=cal.reading, bsp=cal.bsp, holidays=cal.holidays, split=split
    )


def seed_neighbour_files(*, root: Path, base: object, arrays: dict[str, np.ndarray]) -> None:
    """Write the neighbour table, the medians, and a passing check file, as a passed `neighbours`
    stage leaves them.

    The invented neighbours do not mirror the invented reading, so the stage's own gates are tested apart.
    """
    import e00732_neighbours as nb
    import e00732_stages as stages

    nb.write_table(root=root, dates=base.dates, arrays=arrays)  # ty: ignore[unresolved-attribute]
    medians = {
        n: nb.training_median(
            folded=nb.fold(signed=arrays[n], floor=0.0),
            train_rows=base.train_rows,  # ty: ignore[unresolved-attribute]
        )
        for n in nb.NEIGHBOURS
    }
    stages.write_json(path=root / stages.MEDIANS_FILE, payload=medians)
    stages.write_json(path=root / stages.NEIGHBOUR_CHECK, payload={"pass": True})
