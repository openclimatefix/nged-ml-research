# source from the worktree root: puts every experiment folder on PYTHONPATH
export PYTHONPATH="$(printf '%s:' "$PWD"/ml-experiments/*/ )${PYTHONPATH:-}"
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
export NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x
export OMP_NUM_THREADS=1
