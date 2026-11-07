# Source from the repository root with DATA_PATH_INTERNAL already exported: sets the paths experiment 00627's scripts need.
: "${DATA_PATH_INTERNAL:?export DATA_PATH_INTERNAL first}"
export DATA_PATH_INTERNAL
X=ml-experiments
export PYTHONPATH=$PWD/$X/00627-scenario-subset-selection-from-51-coherent-paths:$PWD/$X/00436-joint-scenarios-across-demand-wind-solar-and-leads:$PWD/$X/00434-demand-ensemble-copula-coupling-and-schaake-reordering-of-stored-forecasts:$PWD/$X/00377-demand-v1-first-run-feasibility:$PWD/$X/00381-simple-weekly-xgboost-demand-against-gam-stack:$PWD/$X/00087-bg2-best-guess-assembled-with-drop-one-ablations:$PWD/$X/00429-unified-forecaster-wind-ablations:$PWD/$X/00347-wind-physics-hybrid-plant-then-mdn:$PWD/$X/00310-metered-farm-long-lead-scoring:$PWD/$X/00230-wind-physics-core-part-1:$PWD/$X/00303-wind-adapter-in-production-setting:$PWD/$X/00326-wind-research-default-1-change-arms:$PWD/$X/00336-wind-research-default-1-below-cut-in-zeroing
export NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x
export CUDA_VISIBLE_DEVICES=
