#!/usr/bin/env bash
# Run an experiment 507 script with every earlier experiment folder on PYTHONPATH.
# Usage: run.sh <script.py> [args...]   (from the repository root)
set -euo pipefail
export DATA_PATH_INTERNAL="${DATA_PATH_INTERNAL:-/home/jack/dev/nged-substation-forecast/data}"
export OMP_NUM_THREADS=1
PP="$(ls -d "$PWD"/ml-experiments/*/ | paste -sd:)"
export PYTHONPATH="$PP"
exec systemd-run --user --scope -q -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python "$@"
