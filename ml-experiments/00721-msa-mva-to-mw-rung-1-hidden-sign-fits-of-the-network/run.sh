#!/usr/bin/env bash
# Launch one command of experiment 721 under the slot rules of the page's "Compute" section.
#
# The wrapper starts nothing by default: it needs `--go` before the script name, so that a stray
# invocation only prints this help.
#
# Usage:  SLOT_CPU=<one cpu number> run.sh --go e00721_fit.py <arguments>
#         SLOT_CPU=<n> FITS_STARTING=1 run.sh --go ...     prints the "fits starting" line for the manager
#         SLOT_CPU=<n> LAST_FIT_ENDED=1 run.sh --go ...    prints the "last fit ended" line for the manager
#
# Rules it sets:
#   - one scope at a time: `systemd-run --user --scope`, `MemoryMax=12G`, `MemorySwapMax=0`,
#     `OOMPolicy=continue`, `AllowedCPUs=$SLOT_CPU`;
#   - one thread for every numerical library;
#   - `uptime` load at most 24 and at least 25 GB available at launch, or the wrapper refuses;
#   - `taskset -c $SLOT_CPU` as well, and the job fails unless `os.sched_getaffinity` shows one core;
#   - the venv's Python called directly (never `uv run` inside the scope); `PYTHON` names a venv built
#     with `uv sync --group gpu` for the fits;
#   - this folder, and the folders of the experiments whose code this one imports, on PYTHONPATH:
#     713 (network, loop, scorer), 716 and 717 (F7's `HT` inputs), 720 (injection and folding), and 87 and 506 (the grid and the loaders of power, sky, and farms).
#     `WORKTREES` names the folder that holds those worktrees.
# Set DATA_PATH_INTERNAL first (CLAUDE.md, "Running an experiment"). The three NGED_S3_* variables
# default to dummy values because the contracts package reads them at import. Each fit holds at most
# 6 GB of the GPU (`e00721_fit.limit_gpu_memory`).
set -euo pipefail
if [ "${1:-}" != "--go" ] || [ -z "${2:-}" ]; then
    sed -n '2,22p' "$0" >&2
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
worktrees="${WORKTREES:-$(dirname "$(dirname "$root")")}"
paths="$here"
for d in \
    "$worktrees"/frozen-713-7c29114a8/ml-experiments/00713-msa-axial-attention-demand-network-part-1 \
    "$worktrees"/frozen-713-7c29114a8/ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand \
    "$worktrees"/frozen-713-7c29114a8/ml-experiments/00559-* \
    "$worktrees"/msa-f7-pv-part-1/ml-experiments/00716-msa-unmetered-pv-part-1-history-estimate \
    "$worktrees"/msa-f7-pv-part-1/ml-experiments/00717-* \
    "$worktrees"/msa-mva-to-mw-rung-0/ml-experiments/00720-msa-mva-to-mw-rung-0-signed-power-from-mva-meters \
    "$root"/00087-* "$root"/00506-*; do
    [ -d "$d" ] && paths="$paths:$d"
done
export PYTHONPATH="$paths${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}"
export NGED_S3_BUCKET_URL="${NGED_S3_BUCKET_URL:-https://example.com}"
export NGED_S3_BUCKET_ACCESS_KEY="${NGED_S3_BUCKET_ACCESS_KEY:-dummy}"
export NGED_S3_BUCKET_SECRET="${NGED_S3_BUCKET_SECRET:-dummy}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=1
export DATA_PATH_INTERNAL
if [ -n "${FITS_STARTING:-}" ]; then
    echo "fits starting: experiment 721, ${script} $*, core ${SLOT_CPU}, $(date -u +%FT%TZ)"
fi
status=0
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -p OOMPolicy=continue \
    -p AllowedCPUs="$SLOT_CPU" taskset -c "$SLOT_CPU" \
    "${PYTHON:-$(dirname "$root")/.venv/bin/python}" "$here/$script" "$@" || status=$?
if [ -n "${LAST_FIT_ENDED:-}" ]; then
    echo "last fit ended: experiment 721, ${script} exit ${status}, $(date -u +%FT%TZ)"
fi
exit "$status"
