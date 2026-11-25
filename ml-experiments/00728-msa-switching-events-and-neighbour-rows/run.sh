#!/usr/bin/env bash
# Run one command of experiment 728 under the slot rules of the design (section 9): one thread for
# every numerical library, a memory cap with no swap (9G by default; set SLOT_MEM=12G only for a job
# whose measured peak needs it), the venv's Python called directly (never `uv run` inside the scope,
# because the snap build of uv moves its child out of the capped scope), and the folders of every
# experiment whose code this one imports on PYTHONPATH, with the frozen checkout of experiment 713
# first.
#
# Usage: run.sh e00728_gate.py gh                   # G-H, CPU only
#        GPU=1 run.sh e00728_fit.py Rb 2025-11 0     # a GPU job: one at a time, 6 GB cap
# Set DATA_PATH_INTERNAL first (CLAUDE.md, "Running an experiment"). The three NGED_S3_* variables
# are set to dummy values here because the contracts package reads them at import.
#
# Checks before every job: at least MIN_AVAILABLE_GB (default 20) of free memory, a one-minute load
# average below MAX_LOAD (default: 75% of the cores), and, with GPU=1, an exclusive lock so that this
# experiment runs at most one GPU process at a time. Python caps the GPU memory at GPU_CAP_GB (6)
# through `E00728_GPU_CAP_GB`; the fit script reads it.
set -euo pipefail
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
repo="$(dirname "$root")"
script="$1"
shift
frozen_713="${FROZEN_713:-$(dirname "$repo")/frozen-713-7c29114a8/ml-experiments/00713-msa-axial-attention-demand-network-part-1}"
if [ ! -d "$frozen_713" ]; then
    echo "the frozen checkout of experiment 713 is missing: $frozen_713 (set FROZEN_713)" >&2
    exit 1
fi
available_gb=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${available_gb:-0}" -lt "${MIN_AVAILABLE_GB:-20}" ]; then
    echo "only ${available_gb} GB available; wait until ${MIN_AVAILABLE_GB:-20} GB is free (set MIN_AVAILABLE_GB to override)" >&2
    exit 1
fi
cores=$(nproc)
load=$(awk '{print $1}' /proc/loadavg)
max_load="${MAX_LOAD:-$(awk -v c="$cores" 'BEGIN {printf "%.1f", 0.75 * c}')}"
if awk -v l="$load" -v m="$max_load" 'BEGIN {exit !(l > m)}'; then
    echo "load average $load is above $max_load on $cores cores; wait (set MAX_LOAD to override)" >&2
    exit 1
fi
export PYTHONPATH="$frozen_713:$here"
for d in "$root"/00*/; do
    export PYTHONPATH="$PYTHONPATH:${d%/}"
done
export DATA_PATH_INTERNAL
export NGED_S3_BUCKET_URL="${NGED_S3_BUCKET_URL:-https://example.com}"
export NGED_S3_BUCKET_ACCESS_KEY="${NGED_S3_BUCKET_ACCESS_KEY:-dummy}"
export NGED_S3_BUCKET_SECRET="${NGED_S3_BUCKET_SECRET:-dummy}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1
export E00728_GPU_CAP_GB="${GPU_CAP_GB:-6}"
python="$repo/.venv/bin/python"
scope=(systemd-run --user --scope -q -p MemoryMax="${SLOT_MEM:-9G}" -p MemorySwapMax=0 -p OOMPolicy=continue)
if [ "${GPU:-0}" = "1" ]; then
    lock="$DATA_PATH_INTERNAL/ml-experiments/00728-msa-switching-events-and-neighbour-rows/work/gpu.lock"
    mkdir -p "$(dirname "$lock")"
    exec flock --exclusive --nonblock "$lock" "${scope[@]}" "$python" "$here/$script" "$@"
fi
exec "${scope[@]}" "$python" "$here/$script" "$@"
