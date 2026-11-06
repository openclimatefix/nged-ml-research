#!/bin/bash
# usage: score.sh ARM [ARM ...]  (a0, c0, a1 .. a5); CPU only, 12 GB host cap
D="$(cd "$(dirname "$0")" && pwd)"
R=$(cd "$D/../.." && pwd)
cd $R
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
source ml-experiments/00524*/env.sh
for a in "$@"; do
  systemd-run --user --scope -q -p MemoryMax=12G -p OOMPolicy=continue env OMP_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" DATA_PATH_INTERNAL=$DATA_PATH_INTERNAL NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x $R/.venv/bin/python $D/e00591_score.py $a
done
