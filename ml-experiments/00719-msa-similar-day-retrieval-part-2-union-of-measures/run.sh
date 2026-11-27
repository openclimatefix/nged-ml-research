#!/usr/bin/env bash
# Run one command of experiment 719 under the slot
# rules of experiment 713's `run.sh`: one thread for every numerical library, a memory cap with no
# swap, the venv's Python called directly (never `uv run` inside the scope), and the folders of every
# experiment whose code this one imports on PYTHONPATH.
#
# Experiment 719 holds no fit code of its own: its arms `U`, `U-A`, `U27`, `U27-A`, and `UX` are fitted by experiment 718's
# scripts, which write under experiment 719's data folder. Set `PART_1_FOLDER` to experiment 718's code folder when
# it is not in this checkout (the default is the sibling folder `00718-*`).
#
# Usage: run.sh ../00718-msa-similar-day-retrieval-part-1-weather-distance/e00718_fit.py U 2025-11 0 --smoke
#        SLOT_MEM=4G MIN_AVAILABLE_GB=25 run.sh ../00718-msa-similar-day-retrieval-part-1-weather-distance/e00718_check_fit.py
# Set DATA_PATH_INTERNAL first (CLAUDE.md, "Running an experiment"). `VENV_PYTHON` names the Python of
# a venv that has the `gpu` group (default: the checkout's `.venv`). Check `free -g` first: the script
# refuses to start with under 25 GB available.
set -euo pipefail
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
script="$1"
shift
available_gb=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${available_gb:-0}" -lt "${MIN_AVAILABLE_GB:-25}" ]; then
    echo "only ${available_gb} GB available; wait until ${MIN_AVAILABLE_GB:-25} GB is free (set MIN_AVAILABLE_GB to override)" >&2
    exit 1
fi
export PYTHONPATH="$here"
if [ -n "${PART_1_FOLDER:-}" ]; then
    export PYTHONPATH="$PYTHONPATH:$PART_1_FOLDER"
fi
for d in "$root"/00*/; do
    export PYTHONPATH="$PYTHONPATH:${d%/}"
done
export DATA_PATH_INTERNAL
export NGED_S3_BUCKET_URL="${NGED_S3_BUCKET_URL:-https://example.com}"
export NGED_S3_BUCKET_ACCESS_KEY="${NGED_S3_BUCKET_ACCESS_KEY:-dummy}"
export NGED_S3_BUCKET_SECRET="${NGED_S3_BUCKET_SECRET:-dummy}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1
exec systemd-run --user --scope -q -p MemoryMax="${SLOT_MEM:-12G}" -p MemorySwapMax=0 -p OOMPolicy=continue \
    "${VENV_PYTHON:-$(dirname "$root")/.venv/bin/python}" "$here/$script" "$@"
