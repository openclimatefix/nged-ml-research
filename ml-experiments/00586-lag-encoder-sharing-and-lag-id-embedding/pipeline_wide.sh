#!/bin/bash
# Stage 2 pipeline, resumable (no stress step: it would shift only the day and week lags; leak comes from the injected bank). Usage: pipeline_wide.sh KIND [KIND ...]
set -e
D="$(cd "$(dirname "$0")" && pwd)"
for k in "$@"; do
  $D/run_wide.sh $k pretrain
  $D/run_wide.sh $k fit
  $D/run_wide.sh $k assemble
  $D/run_wide.sh $k inject
done
echo PIPELINE DONE
