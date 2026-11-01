# Source from the repository root with DATA_PATH_INTERNAL already exported: sets the paths experiment 00436's scripts need.
: "${DATA_PATH_INTERNAL:?export DATA_PATH_INTERNAL first}"
export DATA_PATH_INTERNAL
E=ml-experiments
export PYTHONPATH=$PWD/$E/00436-joint-scenarios-across-demand-wind-solar-and-leads:$PWD/$E/00434-demand-ensemble-copula-coupling-and-schaake-reordering-of-stored-forecasts:$PWD/$E/00429-unified-forecaster-wind-ablations:$PWD/$E/00347-wind-physics-hybrid-plant-then-mdn:$PWD/$E/00230-wind-physics-core-part-1:$PWD/$E/00303-wind-adapter-in-production-setting:$PWD/$E/00310-metered-farm-long-lead-scoring:$PWD/$E/00326-wind-research-default-1-change-arms:$PWD/$E/00336-wind-research-default-1-below-cut-in-zeroing:$PWD/$E/00087-bg2-best-guess-assembled-with-drop-one-ablations:$PWD/$E/00377-demand-v1-first-run-feasibility:$PWD/$E/00381-simple-weekly-xgboost-demand-against-gam-stack
export NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x
export CUDA_VISIBLE_DEVICES=
