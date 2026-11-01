# Source from the repository root with DATA_PATH_INTERNAL already exported: experiment 00450's paths (which include 00429's) plus this folder.
: "${DATA_PATH_INTERNAL:?export DATA_PATH_INTERNAL first}"
source ml-experiments/00450-wind-upper-tail-check-with-table-p-and-sweep/env.sh
export PYTHONPATH=$PYTHONPATH:$PWD/ml-experiments/00450-wind-upper-tail-check-with-table-p-and-sweep:$PWD/ml-experiments/00451-farm-a-upper-tail-exceedances-and-capacity-state
