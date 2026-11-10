"""Experiment 684: run one of experiment 667's chain scripts with the output family patched in.

Usage: python e00684_run.py SCRIPT [args]; `SCRIPT` is `e00667_k1.py` or `e00667_cand.py` in
experiment 667's folder. The patch (`e00684_family`) is installed before the script imports the
pipeline, so every stage trains and reads out under `E00684_FAMILY`.
"""

import runpy
import sys
from pathlib import Path

import e00684_family  # noqa: F401  (installs the patch)

script = sys.argv[1]
folder = next(Path(__file__).resolve().parents[1].glob("00667-*"))
sys.argv = [str(folder / script), *sys.argv[2:]]
runpy.run_path(sys.argv[0], run_name="__main__")
