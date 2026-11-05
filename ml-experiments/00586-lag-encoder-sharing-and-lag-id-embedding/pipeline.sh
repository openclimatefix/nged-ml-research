#!/bin/bash
# Stage 1 pipeline, resumable (every command skips finished cells). Usage: pipeline.sh KIND [KIND ...]
set -e
D="$(cd "$(dirname "$0")" && pwd)"
for k in "$@"; do
  $D/run.sh $k pretrain
  $D/run.sh $k fit
  $D/run.sh $k assemble
  $D/run.sh $k stress --seeds 0
  $D/run.sh $k inject
done
echo PIPELINE DONE
