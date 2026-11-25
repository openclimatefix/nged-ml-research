#!/usr/bin/env bash
# Launch one command of experiment 729 under the slot rules of the page's "Compute" section.
#
# The wrapper starts nothing by default: it needs `--go` before the script name, so that a stray
# invocation only prints this help.
#
# Usage:  SLOT_CPU=<one cpu number> run.sh --go e00729_cli.py <banks|gate|timing|screen|choose|ladder|margin|score|report|fit> <arguments>
#         SLOT_CPU=<n> FITS_STARTING=1 run.sh --go ...     prints the "fits starting" line for the manager
#         SLOT_CPU=<n> LAST_FIT_ENDED=1 run.sh --go ...    prints the "last fit ended" line for the manager
#
# Rules it sets:
#   - one scope at a time: `systemd-run --user --scope`, `MemoryMax=12G`, `MemorySwapMax=0`,
#     `OOMPolicy=continue`, `AllowedCPUs=$SLOT_CPU`;
#   - one thread for every numerical library (OMP, MKL, OpenBLAS, Polars, and Rayon);
#   - `uptime` load at most 24 and at least 25 GB available at launch, or the wrapper refuses;
#   - `taskset -c $SLOT_CPU` as well, and `e00729_paths.pin_threads` asserts that `os.sched_getaffinity` shows one core;
#   - the venv's Python called directly (never `uv run` inside the scope); `PYTHON` names a venv built
#     with `uv sync --group gpu` for the fits, and defaults to the `msa-f7-pv-part-1` worktree's venv,
#     which has torch; the wrapper stops with an error when the interpreter cannot import torch;
#   - for a command that uses the GPU (no `--device cpu`, and `CUDA_VISIBLE_DEVICES` not empty): the
#     `nvidia-smi` totals leave room for 6 GB under the total of 36 GB, at most one other GPU process
#     (of this experiment or another, counted together) holds the GPU, and a lock file allows one worker
#     of this experiment at a time. `banks`, `gate`, `choose`, `margin`, `score`, `report`, and `figures` run on the CPU and skip both checks;
#   - this folder on PYTHONPATH: `e00729_paths.py` adds the folders of experiments 721, 694, 700,
#     697, 39, 194, and 722 (the code this experiment imports) from this checkout.
# Set DATA_PATH_INTERNAL first (CLAUDE.md, "Running an experiment"). The three NGED_S3_* variables
# default to dummy values because the contracts package reads them at import. Each fit holds at most
# 6 GB of the GPU (`e00729_train.limit_gpu_memory`).
set -euo pipefail
if [ "${1:-}" != "--go" ] || [ -z "${2:-}" ]; then
    sed -n '2,30p' "$0" >&2
    echo "refusing to run: pass --go and a script name (nothing was started)" >&2
    exit 64
fi
shift
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
: "${SLOT_CPU:?set SLOT_CPU to the one cpu number the manager assigned}"
if ! [[ "$SLOT_CPU" =~ ^[0-9]+$ ]]; then
    echo "SLOT_CPU must be exactly one cpu number (got '$SLOT_CPU'); a range or a list pins several cores" >&2
    exit 64
fi
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
script="$1"
shift
available_gb=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${available_gb:-0}" -lt 25 ]; then
    echo "only ${available_gb} GB available; at least 25 GB must be free at launch" >&2
    exit 1
fi
load=$(awk '{print int($1)}' /proc/loadavg)
if [ "$load" -gt 24 ]; then
    echo "one-minute load ${load} is above 24; wait" >&2
    exit 1
fi
# `e00729_paths.py` puts the sibling experiment folders on sys.path.
export PYTHONPATH="$here"
export NGED_S3_BUCKET_URL="${NGED_S3_BUCKET_URL:-https://example.com}"
export NGED_S3_BUCKET_ACCESS_KEY="${NGED_S3_BUCKET_ACCESS_KEY:-dummy}"
export NGED_S3_BUCKET_SECRET="${NGED_S3_BUCKET_SECRET:-dummy}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1
export DATA_PATH_INTERNAL
if [ -n "${FITS_STARTING:-}" ]; then
    echo "fits starting: experiment 729, ${script} $*, core ${SLOT_CPU}, $(date -u +%FT%TZ)"
fi
default_python="$(dirname "$root")/.claude/worktrees/msa-f7-pv-part-1/.venv/bin/python"
[ -x "$default_python" ] || default_python="$HOME/dev/nged-ml-research/.claude/worktrees/msa-f7-pv-part-1/.venv/bin/python"
python_bin="${PYTHON:-$default_python}"
if ! [ -x "$python_bin" ] || ! "$python_bin" -c 'import torch' 2>/dev/null; then
    echo "PYTHON='$python_bin' cannot import torch: set PYTHON to a venv built with 'uv sync --group gpu'" >&2
    exit 1
fi
uses_gpu=1
[ -z "${CUDA_VISIBLE_DEVICES+x}" ] || [ -n "${CUDA_VISIBLE_DEVICES}" ] || uses_gpu=0
for arg in "$@"; do [ "$arg" = "cpu" ] && uses_gpu=0; done
# these stages read files and compute on the CPU only: no GPU check and no GPU lock
case "${1:-}" in banks | gate | choose | margin | score | report | figures) uses_gpu=0 ;; esac
lock_prefix=""
if [ "$uses_gpu" = 1 ] && command -v nvidia-smi >/dev/null; then
    used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
    if [ $((used_mb + 6144)) -gt 36864 ]; then
        echo "the GPU holds ${used_mb} MB; 6 GB more would pass the 36 GB total" >&2
        exit 1
    fi
    ours=0
    others=0
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
        if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'e00729'; then ours=$((ours + 1)); else others=$((others + 1)); fi
    done
    if [ $((ours + others)) -ge 2 ]; then
        echo "GPU processes: ${ours} of experiment 729 and ${others} of others; at most 2 GPU processes run at once across all agents, and this one would be the third" >&2
        exit 1
    fi
    mkdir -p "$HOME/.cache"
    lock_prefix="flock -n $HOME/.cache/e00729_gpu.lock"
fi
status=0
# shellcheck disable=SC2086
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -p OOMPolicy=continue \
    -p AllowedCPUs="$SLOT_CPU" taskset -c "$SLOT_CPU" \
    $lock_prefix "$python_bin" "$here/$script" "$@" || status=$?
if [ -n "${LAST_FIT_ENDED:-}" ]; then
    echo "last fit ended: experiment 729, ${script} exit ${status}, $(date -u +%FT%TZ)"
fi
exit "$status"
