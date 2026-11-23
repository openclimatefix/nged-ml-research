#!/usr/bin/env bash
# Run one command of experiment 720 under the slot rules of the design: one thread for every
# numerical library, a memory cap with no swap (default 8G), the venv's Python called directly
# (never `uv run` inside the scope), and the folders of the earlier experiments whose code this one
# imports on PYTHONPATH. Experiment 716's folder (branch `msa-f7-pv-part-1`, unmerged) is added
# through `EXTRA_PYTHONPATH`; nothing is merged from that branch.
#
# Usage: run.sh e00720_check_loss.py
#        SLOT_MEM=12G run.sh e00720_census.py --run --dense-raw <path to dense_raw.npy>
#        VENV_PYTHON=<python with torch> run.sh ...   # a script that calls `e00720_real.load_panel`
#                                                     # needs torch to import (CPU use only)
# Set DATA_PATH_INTERNAL first (CLAUDE.md, "Running an experiment"). The three NGED_S3_* variables are
# set to dummy values because the contracts package reads them at import. The script refuses to
# start with under 25 GB available.
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
for d in "$root"/00*/; do
    export PYTHONPATH="$PYTHONPATH:${d%/}"
done
if [ -n "${EXTRA_PYTHONPATH:-}" ]; then
    export PYTHONPATH="$PYTHONPATH:$EXTRA_PYTHONPATH"
fi
export DATA_PATH_INTERNAL
export NGED_S3_BUCKET_URL="${NGED_S3_BUCKET_URL:-https://example.com}"
export NGED_S3_BUCKET_ACCESS_KEY="${NGED_S3_BUCKET_ACCESS_KEY:-dummy}"
export NGED_S3_BUCKET_SECRET="${NGED_S3_BUCKET_SECRET:-dummy}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1
exec systemd-run --user --scope -q -p MemoryMax="${SLOT_MEM:-8G}" -p MemorySwapMax=0 -p OOMPolicy=continue \
    "${VENV_PYTHON:-$(dirname "$root")/.venv/bin/python}" "$here/$script" "$@"
