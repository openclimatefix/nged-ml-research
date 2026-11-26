# ruff: noqa: E501
"""Smoke test of every real file that experiment 732 reads. Read-only and silent about values.

It is not a pytest module. Run it once by hand, only when the code review's checklist says so:

    CUDA_VISIBLE_DEVICES= python tests/smoke_real_files.py

Each check opens one input once and prints one line of flags and counts: a column-presence flag, a
row-count flag, or a status word. It prints no power value, no date, no site name, no engine size, and no
series identifier. Exit status 0 means every input opened with what the code reads.
"""

import sys
from collections.abc import Callable

import e00732_paths

# isort: split
import e00732_core as core
import e00732_neighbours as nb
import numpy as np

e00732_paths.pin_threads()


def check_experiment_730_files() -> str:
    """Check that experiment 730's banks, margin, digest record, and stored arms of experiment 730 are present."""
    import e00730_banks as banks730
    import e00732_stages as stages

    root = e00732_paths.root_730()
    banks = {k: banks730.bank_path(root=root, kind=k).exists() for k in ("validation", "test", "shifted")}
    banks["train seed 0"] = banks730.bank_path(root=root, kind="train", seed=0).exists()
    stored = {
        arm: (root / "predictions" / f"{stages.CONTROL_FOLDERS[arm][0]}_{stages.CONTROL_FOLDERS[arm][1]}_s0" / "test.npz").exists()
        for arm in ("S", "S-L", "SD*")
    }
    return (
        f"banks present: {banks}; margin file present: {(root / 'margin.json').exists()}; "
        f"banks digest id present: {banks730.banks_digest_id(root=root) is not None}; stored test predictions present: {stored}"
    )


def check_neighbour_sources() -> str:
    """Check that D3, D5, and D14 each have one time series id and a stored power column (counts only)."""
    import e00722_common as common722
    import e00722_data as data722
    import e00722_guard as guard

    ids = common722.load_topology().ids_of(labels=nb.NEIGHBOURS)
    grids = data722.load_power(ids=[ids[n] for n in nb.NEIGHBOURS], log=guard.ReadLog(), name="smoke")
    return (
        f"neighbours with an id: {len(ids)} of {len(nb.NEIGHBOURS)}; "
        f"each has power rows: {all(len(grids[ids[n]]) > 0 for n in nb.NEIGHBOURS)}; "
        f"each has a finite reading: {all(bool(np.isfinite(grids[ids[n]]).any()) for n in nb.NEIGHBOURS)}"
    )


def check_context_and_neighbour_table() -> str:
    """Build experiment 729's context and read the neighbour table on D8's base calendar, if the neighbours stage has run."""
    import e00732_cli as cli

    ctx = cli.real_context()
    path = e00732_paths.data_root() / nb.TABLE_NAME
    if not path.exists():
        return f"context built: True, base days present: {len(ctx.cal.dates) > 0}, neighbour table present: False"
    table = nb.read_table(root=e00732_paths.data_root(), dates=ctx.cal.dates)
    shapes = all(a.shape == (len(ctx.cal.dates), 48) for a in table.values())
    return (
        f"context built: True, base days present: {len(ctx.cal.dates) > 0}, neighbour table present: True, "
        f"table shape matches the base calendar: {shapes}, D3 is a magnitude (no negative): {bool(np.nanmin(table['D3']) >= 0.0)}, "
        f"D5 and D14 hold signed values: {all(bool(np.isfinite(table[n]).any()) for n in ('D5', 'D14'))}"
    )


def check_real_day_neighbours() -> str:
    """Check that the neighbours place onto D8's whole day table (the calendar of the real-day sets) and cover most of it."""
    import e00732_cli as cli

    ctx = cli.real_context()
    base = cli.real_base(ctx=ctx)
    del base
    import e00722_guard as guard

    arrays = nb.build_arrays(dates=ctx.table.calendar.dates, log=guard.ReadLog(), name="smoke_real_days")
    cover = nb.coverage(table=arrays, reading=ctx.table.calendar.reading)
    return f"coverage of the whole day table by neighbour at least {nb.MIN_COVERAGE}: {all(v >= nb.MIN_COVERAGE for v in cover.values())}, shape ok: {all(a.shape[0] == len(ctx.table.calendar.dates) for a in arrays.values())}"


def check_real_sets() -> str:
    """Build experiment 729's real-day sets (the pseudo-segments of the real checks) and print counts only."""
    import e00729_stages as stages729
    import e00732_cli as cli

    sets = stages729.real_sets(ctx=cli.real_context())
    return f"real sets built: {sorted(sets)}, each non-empty: {all(len(v) > 0 for v in sets.values())}"


def check_closure_edges() -> str:
    """Load the switching logs and print the status word and the number of edges (K3 is void unless `loaded`)."""
    import e00732_real as real

    edges, status = real.load_closure_edges()
    return f"switching logs status: {status}, closure edges read: {len(edges) > 0}"


def check_private_splits() -> str:
    """Check that experiment 729's private split table exists."""
    return f"private split table present: {e00732_paths.splits_md_path().exists()}"


def check_timing_inputs() -> str:
    """Check that the constants the timing gate reads are present."""
    return f"stop line {core.STOP_LINE_HOURS} GPU-process-hours, timing arms {len(core.TIMING_ARMS)}"


CHECKS: dict[str, Callable[[], str]] = {
    "private splits": check_private_splits,
    "experiment 730 files": check_experiment_730_files,
    "neighbour sources": check_neighbour_sources,
    "context and neighbour table": check_context_and_neighbour_table,
    "real-day neighbours": check_real_day_neighbours,
    "real sets": check_real_sets,
    "closure edges": check_closure_edges,
    "timing inputs": check_timing_inputs,
}


def main() -> int:
    """Run every check, print one line each, and return the exit status."""
    failed = 0
    for name, check in CHECKS.items():
        try:
            line = check()
            print(f"{name}: {line}")
            if "False" in line or "void" in line:
                failed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"{name}: FAILED ({type(exc).__name__})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
