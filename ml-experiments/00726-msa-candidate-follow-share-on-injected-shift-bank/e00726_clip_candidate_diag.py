"""Experiment 726: write the Candidate's clip share for a month whose jobs finished earlier.

**Why.** The first Candidate job of an all-shifted month writes
`diagnostics/clip_candidate_<month>.json` (`e00726_candidate_panel.write_clip_diagnostic`). Jobs
that finished before that function existed never wrote the file, and a rerun skips every finished
job, so it writes nothing either. This script builds the same inputs as `run_jobs` and calls the
same function, without forecasting.

**Reads and writes.** The script reads the cleaned power stack and the fold's training rows through
the stored pipeline, which sets the mean and standard deviation of the day lag. It writes one file,
`diagnostics/clip_candidate_<month>.json`, which holds shares by depth, sign, and group and no
time series. It skips the month if the file exists, and it uses the CPU only.

Usage: python e00726_clip_candidate_diag.py --month 2025-11|2025-12

Run it through `run.sh cpu`, which sets `E00667_PHASE=predict`, one BLAS thread, and the memory cap.
"""

import argparse
import os
import sys
from collections.abc import Sequence
from typing import Final

import e00726_common as com

MONTHS: Final = ("2025-11", "2025-12")
"""The fold months whose all-shifted Candidate jobs finished before the clip file existed."""


def clip_file_of(*, month: str) -> str:
    """Return the name of the file that holds the Candidate's clip share of `month`."""
    return f"clip_candidate_{month}.json"


def main(*, argv: Sequence[str]) -> str:
    """Write the Candidate's clip file of the month in `argv`, or skip it.

    Args:
        argv: The command-line arguments, which name the month.

    Returns:
        `written` or `skipped`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", choices=MONTHS, required=True)
    month = parser.parse_args(list(argv)).month
    target = com.data_root() / "diagnostics" / clip_file_of(month=month)
    if target.exists():
        print(f"{month}: {target.name} exists, skipped", flush=True)
        return "skipped"
    # no GPU is used, and the Candidate module reads its own arguments when it is imported
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.argv = [
        "e00726_candidate_panel.py", "e581_all", "--root", "e667", "--month", month,
        "--seeds", "0", "--device", "cpu",
    ]  # fmt: skip
    import e00726_candidate_panel as cp  # ty: ignore[unresolved-import]

    clean = cp.e.c.clean_stack()
    assert clean.shape == cp.GRID_SHAPE, (clean.shape, cp.GRID_SHAPE)
    cp.e.k.build_month = (
        cp.build_month_syn
    )  # as `run_jobs`: building the month sets the lag statistics
    mr = cp.e.k.build_month(month=month, clean=clean)
    cp.write_clip_diagnostic(mr=mr, clean=clean)
    assert target.exists(), target
    return "written"


if __name__ == "__main__":
    main(argv=sys.argv[1:])
