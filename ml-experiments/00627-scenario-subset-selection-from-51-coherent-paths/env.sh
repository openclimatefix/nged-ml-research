# Source from the repository root with DATA_PATH_INTERNAL already exported: sets the paths experiment 00627's scripts need.
: "${DATA_PATH_INTERNAL:?export DATA_PATH_INTERNAL first}"
export DATA_PATH_INTERNAL
export PYTHONPATH=$PWD/ml-experiments/00627-scenario-subset-selection-from-51-coherent-paths
export NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x
export CUDA_VISIBLE_DEVICES=
