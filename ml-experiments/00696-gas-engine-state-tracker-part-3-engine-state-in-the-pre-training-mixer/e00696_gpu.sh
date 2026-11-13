#!/bin/bash
# usage: e00696_gpu.sh SCRIPT [args]; one GPU process under a host-memory cap, 1 thread per math library.
# E00696_MEM sets the cap (default 16G; raise it from the measured peak of the first job). Swap is off.
D="$(cd "$(dirname "$0")" && pwd)"
R=$(cd "$D/../.." && pwd)
cd $R
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
export DATA_PATH_INTERNAL
source ml-experiments/00524*/env.sh
exec systemd-run --user --scope -q -p MemoryMax=${E00696_MEM:-16G} -p MemorySwapMax=0 -p OOMPolicy=continue env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="src:$PYTHONPATH" DATA_PATH_INTERNAL=$DATA_PATH_INTERNAL E00696_ARM=$E00696_ARM NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x $R/.venv/bin/python $D/"$@"
