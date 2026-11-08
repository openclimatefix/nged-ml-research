#!/bin/bash
# usage: cpu.sh SCRIPT [args]; CPU-only run in this folder with every experiment folder on PYTHONPATH
D="$(cd "$(dirname "$0")" && pwd)"
R=$(cd "$D/../.." && pwd)
cd $R
export DATA_PATH_INTERNAL=${DATA_PATH_INTERNAL:-/home/jack/dev/nged-substation-forecast/data}
source ml-experiments/00524*/env.sh
[ -n "$USE_GPU" ] || export CUDA_VISIBLE_DEVICES=
exec $R/.venv/bin/python -W ignore $D/"$@"
