# Source from the repository root with DATA_PATH_INTERNAL already exported: experiment 00429's paths plus 00230 and this folder.
: "${DATA_PATH_INTERNAL:?export DATA_PATH_INTERNAL first}"
source ml-experiments/00429-unified-forecaster-wind-ablations/env.sh
export PYTHONPATH=$PYTHONPATH:$PWD/ml-experiments/00450-wind-upper-tail-check-with-table-p-and-sweep:$PWD/ml-experiments/00230-wind-physics-core-part-1
