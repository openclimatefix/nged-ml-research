#!/usr/bin/env bash
# Launch one stage of experiment 738 under the workstation rules (CPU stages, and GPU stages one
# process at a time).
#
# The wrapper starts nothing by default: it needs `--go` before the stage, so that a stray
# invocation only prints this help.
#
# Usage:  SLOT_CPUS=<cpu list> run.sh --go <stage> [arguments] [--workers N]
#         stages: phase0 review-mark smoke-fit smoke-predict smoke-collect screen-fits gate
#         gate-remedy screen-score full-fits predict score members report figures (see
#         `e00738_run.py --help`). `--workers N` starts N workers (`--worker i --of N`) inside the
#         one scope, through `run_workers.sh`, and exits with the worst worker's exit code. A worker
#         runs on one cpu of SLOT_CPUS when the list has one cpu per worker, and otherwise the
#         workers share the whole list. A GPU stage runs with one worker and `--device cuda`.
#         Exit codes: 3 a failed CPU stage, 5 a failed GPU chain or the gate's second failure, 6
#         the gate's first failure (run `gate-remedy`), 64 a bad call, 75 the GPU is busy.
#
# Rules it sets:
#   - one scope at a time: `systemd-run --user --scope`, `MemoryMax=$SLOT_MEM`, `MemorySwapMax=0`,
#     `OOMPolicy=continue`, `AllowedCPUs=$SLOT_CPUS`, and `taskset -c` on the same list. SLOT_MEM
#     defaults to 16G for the scope (the hard cap of the plan); set it lower for many small workers;
#   - one thread per worker for every numerical library (OMP, MKL, OpenBLAS, Polars, and Rayon);
#   - at least 25 GB available at launch, and one-minute load below 24, or the wrapper refuses;
#   - the venv's Python called directly (never `uv run` inside the scope), with this folder and the
#     folders of the recipe experiments, 647, 650, 652, 654, 713, and 726 on PYTHONPATH. Set
#     E00738_713_DIR to the frozen checkout of experiment 713's folder (commit 7c29114a8) and it
#     goes first, because the stored fits carry that checkout's code hash;
# Set DATA_PATH_INTERNAL first (CLAUDE.md, "Running an experiment"). The three NGED_S3_* variables
# default to dummy values because the contracts package reads them at import. The stages after
# gates need the review marker and code hash that `e00738_run.py` checks. Nothing is written
# outside `data/ml-experiments/00738-*` (or `E00738_OUT`, which needs E00738_SCRATCH=1 or a
# `scratch` path component).
set -euo pipefail
if [ "${1:-}" != "--go" ] || [ -z "${2:-}" ]; then
    sed -n '2,32p' "$0" >&2
    echo "refusing to run: pass --go and a stage (nothing was started)" >&2
    exit 64
fi
shift
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
: "${SLOT_CPUS:=8-15}"  # the plan's cores
if ! [[ "$SLOT_CPUS" =~ ^[0-9]+([-,][0-9]+)*$ ]]; then
    echo "SLOT_CPUS must be a cpu list such as 16-17 (got '$SLOT_CPUS')" >&2
    exit 64
fi
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
available_gb=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${available_gb:-0}" -lt 25 ]; then
    echo "only ${available_gb} GB available; at least 25 GB must be free at launch" >&2
    exit 1
fi
load=$(awk '{print int($1)}' /proc/loadavg)
if [ "$load" -ge 24 ]; then
    echo "one-minute load ${load} is 24 or above; wait" >&2
    exit 1
fi
paths="$here"
if [ -n "${E00738_713_DIR:-}" ]; then
    [ -d "$E00738_713_DIR" ] || { echo "E00738_713_DIR is not a folder" >&2; exit 64; }
    paths="$E00738_713_DIR:$paths"
fi
for n in 11 87 377 378 381 391 397 400 402 404 407 410 418 422 425 428 435 437 448 454 470 510 511 513 515 521 545 558 559 560 562 565 568 569 572 574 578 581 632 667 668 709 647 650 652 654 713 726; do
    for d in "$root"/${n}-* "$root"/0${n}-* "$root"/00${n}-* "$root"/000${n}-*; do
        [ -d "$d" ] && paths="$paths:$d"
    done
done
export PYTHONPATH="$paths"
export NGED_S3_BUCKET_URL="${NGED_S3_BUCKET_URL:-https://example.com}"
export NGED_S3_BUCKET_ACCESS_KEY="${NGED_S3_BUCKET_ACCESS_KEY:-dummy}"
export NGED_S3_BUCKET_SECRET="${NGED_S3_BUCKET_SECRET:-dummy}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1
export DATA_PATH_INTERNAL
python_bin="${PYTHON:-$(dirname "$root")/.venv/bin/python}"
[ -x "$python_bin" ] || { echo "no python at $python_bin; set PYTHON" >&2; exit 64; }
workers=0
args=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--workers" ]; then
        workers="${2:?--workers needs a number}"
        shift 2
    else
        args+=("$1")
        shift
    fi
done
if ! [[ "$workers" =~ ^[0-9]+$ ]]; then
    echo "--workers must be a number (got '$workers')" >&2
    exit 64
fi
# The plan fixes the memory cap of one scope at 16 GB, however many workers it holds.
: "${SLOT_MEM:=16G}"
if [ -n "${E00738_OUT:-}" ]; then
    echo "WARNING: E00738_OUT is set, so every output goes to ${E00738_OUT}, not to the data folder" >&2
fi
cpus=()
IFS=',' read -ra parts <<< "$SLOT_CPUS"
for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then
        for ((c = ${part%-*}; c <= ${part#*-}; c++)); do cpus+=("$c"); done
    else
        cpus+=("$part")
    fi
done
status=0
if [ "$workers" -le 1 ]; then
    systemd-run --user --scope -q -p MemoryMax="$SLOT_MEM" -p MemorySwapMax=0 -p OOMPolicy=continue \
        -p AllowedCPUs="$SLOT_CPUS" taskset -c "$SLOT_CPUS" \
        "$python_bin" "$here/e00738_run.py" "${args[@]}" || status=$?
    exit "$status"
fi
# One scope holds all workers; `run_workers.sh` pins each worker to one cpu when the list has one
# cpu per worker, and otherwise lets all workers share the whole list.
export WORKER_CPUS="${cpus[*]}" WORKERS="$workers" PYTHON_BIN="$python_bin" RUN_PY="$here/e00738_run.py"
systemd-run --user --scope -q -p MemoryMax="$SLOT_MEM" -p MemorySwapMax=0 -p OOMPolicy=continue \
    -p AllowedCPUs="$SLOT_CPUS" bash "$here/run_workers.sh" "${args[@]}" || status=$?
exit "$status"
