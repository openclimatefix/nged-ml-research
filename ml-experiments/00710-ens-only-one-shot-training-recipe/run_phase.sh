#!/usr/bin/env bash
# Run one command of experiment 710 under the compute-slot rules: SLOT_CORES BLAS and Polars threads (default 1), a 16 GB
# memory cap with no swap, and the folders of the experiments whose code this one imports.
#
# Usage: run_phase.sh e00710_fit.py --phase T --minimal --device cuda
#        run_phase.sh e00710_known_answer.py --tests 1 4 6
# Set DATA_PATH_INTERNAL, NGED_S3_BUCKET_URL, NGED_S3_BUCKET_ACCESS_KEY, and NGED_S3_BUCKET_SECRET
# first (CLAUDE.md, "Running an experiment").
set -euo pipefail
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
script="$1"
shift
export PYTHONPATH="$here"
for d in "$root"/00*/; do
    export PYTHONPATH="$PYTHONPATH:${d%/}"
done
export OMP_NUM_THREADS="${SLOT_CORES:-1}" MKL_NUM_THREADS="${SLOT_CORES:-1}" OPENBLAS_NUM_THREADS="${SLOT_CORES:-1}"
export POLARS_MAX_THREADS="${SLOT_CORES:-1}" RAYON_NUM_THREADS="${SLOT_CORES:-1}"
exec systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=continue \
    uv run --no-sync python "$here/$script" "$@"
