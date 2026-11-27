#!/usr/bin/env bash
# Launch one stage of experiment 734 under the workstation rules.
#
# The wrapper starts nothing by default: it needs `--go` before the stage, so that a stray
# invocation only prints this help.
#
# Usage:  SLOT_CPUS=<cpu list> run.sh --go <stage> [arguments] [--workers N]
#         stages: stage0 gallery gates tune choose score real (see `e00734_run.py --help`)
#         `--workers N` starts N workers (`--worker i --of N`, one per cpu of SLOT_CPUS when the list
#         has N cpus) inside the one scope, and exits with the worst worker's exit code. The scope's
#         limit is shared by the workers, so SLOT_MEM is N times the peak of one worker: 6 workers
#         at about 2 GB each need SLOT_MEM=12G (the default).
#
# Rules it sets:
#   - one scope at a time: `systemd-run --user --scope`, `MemoryMax=$SLOT_MEM`, `MemorySwapMax=0`,
#     `OOMPolicy=continue`, `AllowedCPUs=$SLOT_CPUS`, and `taskset -c $SLOT_CPUS` as well;
#   - one thread per worker for every numerical library (OMP, MKL, OpenBLAS, Polars, and Rayon);
#   - at least 10 GB available at launch, and one-minute load below 24, or the wrapper refuses;
#   - the venv's Python called directly (never `uv run` inside the scope), with this folder and the
#     folders of experiments 653, 657, 660, 665, 670, 680, and 686 on PYTHONPATH.
# Set DATA_PATH_INTERNAL first (CLAUDE.md, "Running an experiment"). The three NGED_S3_* variables
# default to dummy values because the contracts package reads them at import. The stages tune,
# score, and real need the review markers that `e00734_run.py` checks.
set -euo pipefail
if [ "${1:-}" != "--go" ] || [ -z "${2:-}" ]; then
    sed -n '2,23p' "$0" >&2
    echo "refusing to run: pass --go and a stage (nothing was started)" >&2
    exit 64
fi
shift
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
: "${SLOT_CPUS:?set SLOT_CPUS to the cpu list the manager assigned}"
: "${SLOT_MEM:=12G}"
if ! [[ "$SLOT_CPUS" =~ ^[0-9]+([-,][0-9]+)*$ ]]; then
    echo "SLOT_CPUS must be a cpu list such as 16-17 (got '$SLOT_CPUS')" >&2
    exit 64
fi
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
available_gb=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${available_gb:-0}" -lt 10 ]; then
    echo "only ${available_gb} GB available; at least 10 GB must be free at launch" >&2
    exit 1
fi
load=$(awk '{print int($1)}' /proc/loadavg)
if [ "$load" -ge 24 ]; then
    echo "one-minute load ${load} is 24 or above; wait" >&2
    exit 1
fi
paths="$here"
for n in 653 657 660 665 670 680 686; do
    for d in "$root"/${n}-* "$root"/0${n}-* "$root"/00${n}-*; do
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
[ -x "$python_bin" ] || python_bin="$(command -v python3)"
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
        "$python_bin" "$here/e00734_run.py" "${args[@]}" || status=$?
    exit "$status"
fi
# One scope holds all workers; each worker runs on one cpu when the list has one cpu per worker.
export WORKER_CPUS="${cpus[*]}" WORKERS="$workers" PYTHON_BIN="$python_bin" RUN_PY="$here/e00734_run.py"
systemd-run --user --scope -q -p MemoryMax="$SLOT_MEM" -p MemorySwapMax=0 -p OOMPolicy=continue \
    -p AllowedCPUs="$SLOT_CPUS" bash "$here/run_workers.sh" "${args[@]}" || status=$?
exit "$status"
