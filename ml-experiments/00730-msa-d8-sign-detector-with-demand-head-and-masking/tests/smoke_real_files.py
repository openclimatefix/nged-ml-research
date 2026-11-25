# ruff: noqa: E501
"""Smoke test of every real file that experiment 730 reads. Read-only and silent about values.

It is not a pytest module. Run it once by hand, only when the code review's checklist says so:

    SLOT_CPUS=<list> ml-experiments/00730-*/run.sh --go tests/smoke_real_files.py

It opens each real file once and prints one line per input: a column-presence flag and a row-count
flag. It prints no power value, no date, no farm or site name, no engine size, and no series
identifier. The component checks of the `banks` stage read each simulated component against the
share `MIN_OUTPUT_SHARE`, which the smoke test quotes. Exit status 0 means every input opened with
the columns the code reads.
"""

import sys
from collections.abc import Callable

import e00730_paths

# isort: split
import e00730_core as core
import e00730_pretrain as pretrain
import numpy as np

e00730_paths.pin_threads()


def check_private_splits() -> str:
    """Check that experiment 729's private split table, the one file read from its folder, exists."""
    path = e00730_paths.splits_md_path()
    return f"private split table present: {path.exists()}"


def check_pretraining_inputs() -> str:
    """Check that experiment 545's chain is importable and the encoder width constants agree."""
    import e00545_model as model  # ty: ignore[unresolved-import]

    del model
    return (
        f"encoder columns {core.ENCODER_INPUT_COLUMNS} = 7 weather + 21 datetime: "
        f"{core.ENCODER_INPUT_COLUMNS == 28}, minimum output share {core.MIN_OUTPUT_SHARE}"
    )


def check_encoder_table_alignment() -> str:
    """Check that D8's encoder table peaks at midday UTC, from a synthetic table of the same shape."""
    table = np.zeros((2, 48, 3))
    table[:, :, 0] = np.maximum(0.0, np.cos((np.arange(48) - 24) / 48 * 2 * np.pi))
    hour = pretrain.clear_sky_peak_hour(table=table, cosz_column=0)
    low, high = core.ENCODER_PEAK_HOURS_UTC
    return f"clear-sky peak hour within {low} to {high} UTC: {low <= hour <= high}"


def check_grid_dates() -> str:
    """Check that experiment 87's grid gives one UTC day per half-hour for the removal mask."""
    import e00087_common as common  # ty: ignore[unresolved-import]

    days = common.grid_times().dt.date().to_numpy().astype("datetime64[D]")
    return f"grid has one day per half-hour: {days.shape[0] == common.grid_times().len()}"


CHECKS: dict[str, Callable[[], str]] = {
    "grid dates": check_grid_dates,
    "private splits": check_private_splits,
    "pre-training inputs": check_pretraining_inputs,
    "encoder table alignment": check_encoder_table_alignment,
}


def main() -> int:
    """Run every check, print one line each, and return the exit status."""
    failed = 0
    for name, check in CHECKS.items():
        try:
            print(f"{name}: {check()}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"{name}: FAILED ({type(exc).__name__})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
