# Source from the repository root with DATA_PATH_INTERNAL already exported: sets the paths experiment 00629's scripts need.
: "${DATA_PATH_INTERNAL:?export DATA_PATH_INTERNAL first}"
export DATA_PATH_INTERNAL
E=ml-experiments
export PYTHONPATH=$PWD/$E/00629-bsp-group-sum-of-nodes-against-direct-bsp-forecast:$PWD/$E/00434-demand-ensemble-copula-coupling-and-schaake-reordering-of-stored-forecasts:$PWD/$E/00436-joint-scenarios-across-demand-wind-solar-and-leads:$PWD/$E/00087-bg2-best-guess-assembled-with-drop-one-ablations:$PWD/$E/00377-demand-v1-first-run-feasibility:$PWD/$E/00381-simple-weekly-xgboost-demand-against-gam-stack
export NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x
export CUDA_VISIBLE_DEVICES=
