#!/bin/bash
# usage: cpu_p.sh SCRIPT [args]: run a CPU script of this folder with the experiment folders on PYTHONPATH
D="$(cd "$(dirname "$0")" && pwd)"
R=$(cd "$D/../../.." && pwd)
cd $R
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
source ml-experiments/00524*/env.sh
export CUDA_VISIBLE_DEVICES=""
exec env OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} $R/.venv/bin/python $D/"$@"
