#!/bin/bash
# usage: e00695_arm.sh ARM [MONTH...]: run the step-5 chain of e00695_chain.sh for each month in turn
# (default: the 8 test months). Needs DATA_PATH_INTERNAL; the memory cap and OMP_NUM_THREADS=1 are in e00695_gpu.sh.
D="$(cd "$(dirname "$0")" && pwd)"
export NGED_S3_BUCKET_URL=https://example.com NGED_S3_BUCKET_ACCESS_KEY=d NGED_S3_BUCKET_SECRET=d
ARM=$1; shift
MONTHS=${*:-2025-11 2025-12 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06}
for m in $MONTHS; do
  s=$(date +%s)
  "$D/e00695_chain.sh" "$ARM" "$m" || { echo "FAILED $ARM $m"; continue; }
  echo "$ARM $m total $(( $(date +%s)-s ))s"
done
echo "$ARM ALLDONE"
