#!/usr/bin/env bash
# Run one command of experiment 725 under the slot rules of the design (section 9): one thread for every
# numerical library, a memory cap with no swap (9 GB for a fit, a prediction, a per-fold check, and KL3r;
# 4 GB for the cache step and the other checks; SLOT_MEM overrides), the venv's Python called directly (never `uv run`
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
# Refuses to start when under 25 GB of memory is available (MIN_AVAILABLE_GB) or when two GPU processes
# of F8 already run (a fit, a prediction, or a per-fold check; MAX_GPU_PROCESSES). It waits, up to
# MAX_LOAD_WAIT_S seconds, while the 1-minute load average is at or above MAX_LOAD (24).
set -euo pipefail
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
: "${EXTRA_PYTHONPATH:?set EXTRA_PYTHONPATH to the 00713-* folder of the frozen checkout}"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
script="$1"
shift
# Wait (up to MAX_LOAD_WAIT_S, default 1 hour) until the 1-minute load average is below MAX_LOAD.
waited=0
while [ "$(awk '{print int($1)}' /proc/loadavg)" -ge "${MAX_LOAD:-24}" ]; do
    if [ "$waited" -ge "${MAX_LOAD_WAIT_S:-3600}" ]; then
        echo "load average still at or above ${MAX_LOAD:-24} after ${waited} s" >&2
        exit 1
    fi
    sleep 30
    waited=$((waited + 30))
done
# The memory check runs after the load wait, so that it reads the memory of the moment the job starts.
available_gb=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${available_gb:-0}" -lt "${MIN_AVAILABLE_GB:-25}" ]; then
    echo "only ${available_gb} GB available; wait until ${MIN_AVAILABLE_GB:-25} GB is free" >&2
    exit 1
fi
gpu_job=no
big_job=no
case "$script" in
e00725_fit.py | e00725_predict.py | e00725_t3_candidate.py) gpu_job=yes big_job=yes ;;
e00725_checks.py)
    [ "${1:-}" = fold ] && gpu_job=yes && big_job=yes
    [ "${1:-}" = kl3r ] && big_job=yes
    ;;
esac
if [ "$gpu_job" = yes ]; then
    # KL3r, the cache step, and the other checks run on the CPU outside the GPU slot. Only the
    # interpreter processes count (a command line that starts with a python binary), so this shell,
    # the shell of the caller, and `sh -c` wrappers never match.
    running=$(pgrep -fc "^[^ ]*/python[0-9.]* [^ ]*/e00725_(fit|predict|t3_candidate)[.]py|^[^ ]*/python[0-9.]* [^ ]*/e00725_checks[.]py fold" || true)
    if [ "${running:-0}" -ge "${MAX_GPU_PROCESSES:-2}" ]; then
        echo "${running} F8 GPU processes already run (limit ${MAX_GPU_PROCESSES:-2})" >&2
        exit 1
    fi
fi
# 9 GB for a fit, a prediction, a per-fold check, and KL3r (experiment 713's pipeline peaks at 6.4 to 7.4 GB
# on the host); 4 GB for the cache step and the other checks. SLOT_MEM overrides both.
if [ "$big_job" = yes ]; then default_mem=9G; else default_mem=4G; fi
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
python_bin="$(dirname "$root")/.venv/bin/python"
if [ -n "${MEMORY_PEAK_FILE:-}" ]; then
    # the scope's peak memory (page cache included) is written before the scope ends
    mkdir -p "$(dirname "$MEMORY_PEAK_FILE")"
    exec systemd-run --user --scope -q -p MemoryMax="${SLOT_MEM:-$default_mem}" -p MemorySwapMax=0 -p OOMPolicy=continue \
        sh -c '"$@"; status=$?; cat "/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.peak" > "$MEMORY_PEAK_FILE" 2>/dev/null; exit $status' \
        sh "$python_bin" "$here/$script" "$@"
fi
exec systemd-run --user --scope -q -p MemoryMax="${SLOT_MEM:-$default_mem}" -p MemorySwapMax=0 -p OOMPolicy=continue \
    "$python_bin" "$here/$script" "$@"
