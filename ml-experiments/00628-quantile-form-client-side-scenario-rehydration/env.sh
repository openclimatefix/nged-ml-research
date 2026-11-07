# Source from the repository root with DATA_PATH_INTERNAL already exported: sets the paths experiment 00628's scripts need.
: "${DATA_PATH_INTERNAL:?export DATA_PATH_INTERNAL first}"
source ml-experiments/00625-coherent-members-from-quantiles-k-sweep-and-path-checks/env.sh
export PYTHONPATH=$PWD/ml-experiments/00628-quantile-form-client-side-scenario-rehydration:$PWD/ml-experiments/00625-coherent-members-from-quantiles-k-sweep-and-path-checks:$PYTHONPATH
