#!/bin/bash
# usage: cpu.sh SCRIPT [args]: run a CPU script of this folder with every experiment folder, and experiment 667's prevailing folder, on PYTHONPATH
D="$(cd "$(dirname "$0")" && pwd)"
R=$(cd "$D/../.." && pwd)
cd $R
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
source ml-experiments/00524*/env.sh
export PYTHONPATH="$(printf '%s:' $R/ml-experiments/00667-*/prevailing):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=""
exec env OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} $R/.venv/bin/python $D/"$@"
