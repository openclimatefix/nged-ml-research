# Source from the repository root with DATA_PATH_INTERNAL already exported: sets the paths experiment 00429's scripts need.
: "${DATA_PATH_INTERNAL:?export DATA_PATH_INTERNAL first}"
export DATA_PATH_INTERNAL
E=ml-experiments
export PYTHONPATH=$PWD/$E/00429-unified-forecaster-wind-ablations:$PWD/$E/00347-wind-physics-hybrid-plant-then-mdn:$PWD/$E/00230-wind-physics-core-part-1:$PWD/$E/00303-wind-adapter-in-production-setting:$PWD/$E/00310-metered-farm-long-lead-scoring:$PWD/$E/00326-wind-research-default-1-change-arms:$PWD/$E/00336-wind-research-default-1-below-cut-in-zeroing
export NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x
