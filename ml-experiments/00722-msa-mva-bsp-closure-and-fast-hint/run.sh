#!/usr/bin/env bash
# Launch one job of experiment 722 under the slot rules of section 8.1 of the design review.
#
# The wrapper never runs a job by default: it needs the explicit argument `--go` before the script
# name, so that a stray invocation only prints this help.
#
# Usage:  SLOT_CPU=<one cpu number> run.sh --go e00722_u0.py
#
# Rules it sets:
#   - one scope at a time: `systemd-run --user --scope`, with `MemoryMax=12G`, `MemorySwapMax=0`,
#     `OOMPolicy=continue`, and `AllowedCPUs=$SLOT_CPU` (the one core the manager assigned);
#   - one thread for every numerical library (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
#     `OPENBLAS_NUM_THREADS`, `POLARS_MAX_THREADS`);
#   - the GPU untouched: `CUDA_VISIBLE_DEVICES=""`, and no job imports `torch`;
#   - at least 25 GB of memory available at launch, or the wrapper refuses;
#   - the venv's Python called directly (never `uv run` inside the scope);
#   - only this experiment's folder on PYTHONPATH, plus `EXTRA_PYTHONPATH` if set.
# Set DATA_PATH_INTERNAL first (CLAUDE.md, "Running an experiment"). The three NGED_S3_* variables
# default to dummy values because the contracts package reads them at import. A job's first fit is
# timed inside the job, which stops when the projection exceeds 1 core-hour.
set -euo pipefail
if [ "${1:-}" != "--go" ] || [ -z "${2:-}" ]; then
    sed -n '2,20p' "$0" >&2
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
export PYTHONPATH="$here${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}"
export NGED_S3_BUCKET_URL="${NGED_S3_BUCKET_URL:-https://example.com}"
export NGED_S3_BUCKET_ACCESS_KEY="${NGED_S3_BUCKET_ACCESS_KEY:-dummy}"
export NGED_S3_BUCKET_SECRET="${NGED_S3_BUCKET_SECRET:-dummy}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=1
export CUDA_VISIBLE_DEVICES=""
exec systemd-run --user --scope -q \
    -p MemoryMax=12G -p MemorySwapMax=0 -p OOMPolicy=continue -p AllowedCPUs="$SLOT_CPU" \
    "$(dirname "$root")/.venv/bin/python" "$here/$script" "$@"
