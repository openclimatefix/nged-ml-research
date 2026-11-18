#!/bin/bash
# usage: gpu.sh SCRIPT [args]: one process in its own scope, 16 GB host cap, no swap, 1 thread per maths library.
# SCRIPT is a file name in this folder, or an absolute path (the global fits run experiment 677's script).
D="$(cd "$(dirname "$0")" && pwd)"
R=$(cd "$D/../.." && pwd)
cd "$R" || exit 1
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
export DATA_PATH_INTERNAL
# env.sh prepends every experiment folder to PYTHONPATH (about 107 KB). Start from an empty PYTHONPATH so
# that a caller who sourced env.sh already does not double it past Linux's 128 KB limit on one
# environment string, at which exec fails with "Argument list too long".
unset PYTHONPATH
source ml-experiments/00524*/env.sh
SCRIPT=$1; shift
case $SCRIPT in /*) ;; *) SCRIPT=$D/$SCRIPT ;; esac
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=continue env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DATA_PATH_INTERNAL="$DATA_PATH_INTERNAL" NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x "$R/.venv/bin/python" "$SCRIPT" "$@"
