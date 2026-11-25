#!/usr/bin/env bash
# Run one command of experiment 725 under the slot rules of the design (section 9): one thread for every
# numerical library, a 9 GB memory cap with no swap, the venv's Python called directly (never `uv run`
# inside the scope, because the snap build of uv moves its child out of the capped scope), and experiment
# 713's frozen checkout first on PYTHONPATH.
#
# Usage: run.sh e00725_fit.py L 2025-11 0 [--device cuda]
#        SLOT_MEM=4G run.sh e00725_sources.py train        # the CPU cache step
#        run.sh e00725_phases.py run 1 --manager-notified  # the runner itself needs no scope: call it directly
# Set DATA_PATH_INTERNAL and EXTRA_PYTHONPATH first. EXTRA_PYTHONPATH names the `00713-*` folder of the
# frozen checkout that reproduces code hash e85b84e2ea04b3cb (the run brief gives its path). The three
# NGED_S3_* variables are set to dummy values because the contracts package reads them at import.
#
# Refuses to start when under 25 GB of memory is available (MIN_AVAILABLE_GB), when the 1-minute load
# average is 24 or more (MAX_LOAD), or when two GPU processes of F8 already run (a fit, a prediction, or
# a per-fold check; MAX_GPU_PROCESSES).
set -euo pipefail
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
: "${EXTRA_PYTHONPATH:?set EXTRA_PYTHONPATH to the 00713-* folder of the frozen checkout}"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
script="$1"
shift
available_gb=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${available_gb:-0}" -lt "${MIN_AVAILABLE_GB:-25}" ]; then
    echo "only ${available_gb} GB available; wait until ${MIN_AVAILABLE_GB:-25} GB is free" >&2
    exit 1
fi
load=$(awk '{print int($1)}' /proc/loadavg)
if [ "$load" -ge "${MAX_LOAD:-24}" ]; then
    echo "load average ${load} is at or above ${MAX_LOAD:-24}; wait" >&2
    exit 1
fi
case "$script" in
e00725_fit.py | e00725_predict.py | e00725_checks.py)
    running=$(pgrep -fc "e00725_(fit|predict|checks)\.py" || true)
    # this shell's own command line also matches, so count only processes besides it
    if [ "${running:-0}" -gt "${MAX_GPU_PROCESSES:-2}" ]; then
        echo "${running} F8 GPU processes already run (limit ${MAX_GPU_PROCESSES:-2})" >&2
        exit 1
    fi
    ;;
esac
# The frozen checkout's folder goes first, so that no other copy of experiment 713 is imported. The other
# experiment folders that F8 imports (437, 560, 562, 667, and the rest) come after it, minus any 00713-*.
export PYTHONPATH="$EXTRA_PYTHONPATH:$here"
for d in "$root"/00*/; do
    case "$d" in
    */00713-*/) ;;
    *) export PYTHONPATH="$PYTHONPATH:${d%/}" ;;
    esac
done
export DATA_PATH_INTERNAL
export NGED_S3_BUCKET_URL="${NGED_S3_BUCKET_URL:-https://example.com}"
export NGED_S3_BUCKET_ACCESS_KEY="${NGED_S3_BUCKET_ACCESS_KEY:-dummy}"
export NGED_S3_BUCKET_SECRET="${NGED_S3_BUCKET_SECRET:-dummy}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1
exec systemd-run --user --scope -q -p MemoryMax="${SLOT_MEM:-9G}" -p MemorySwapMax=0 -p OOMPolicy=continue \
    "$(dirname "$root")/.venv/bin/python" "$here/$script" "$@"
