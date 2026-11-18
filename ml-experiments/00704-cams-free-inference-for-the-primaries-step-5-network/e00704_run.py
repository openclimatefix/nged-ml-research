"""Experiments 703 and 704: run one chain script of experiment 667 or 677 with an arm patched in.

Usage: python e00704_run.py SCRIPT_PATH [args]; `SCRIPT_PATH` is relative to `ml-experiments/`
(for example `00677-*/e00677_k1.py`, glob allowed). Environment: `E00704_ARM` is replay, proxy,
or drop (see `e00704_arm.py`). The patch is installed before the script imports the pipeline.
"""

import runpy
import sys
from pathlib import Path

import e00704_arm  # noqa: F401  (installs the patch)

script = next(Path(__file__).resolve().parents[1].glob(sys.argv[1]))
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(sys.argv[0], run_name="__main__")
