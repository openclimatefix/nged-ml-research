#!/bin/bash
# Stage 2 pipeline, resumable. Usage: pipeline_wide.sh KIND [KIND ...]
set -e
D="$(cd "$(dirname "$0")" && pwd)"
for k in "$@"; do
  $D/run_wide.sh $k pretrain
  $D/run_wide.sh $k fit
  $D/run_wide.sh $k assemble
  $D/run_wide.sh $k stress --seeds 0
  $D/run_wide.sh $k inject
done
echo PIPELINE DONE
