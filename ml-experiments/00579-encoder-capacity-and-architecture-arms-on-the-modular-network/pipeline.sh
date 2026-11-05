#!/bin/bash
# Stage 1 pipeline (seed 0), in the brief's priority order; every step is resumable (finished cells are reused).
# usage: pipeline.sh ARM [ARM ...]
D="$(cd "$(dirname "$0")" && pwd)"
for arm in "$@"; do
  case $arm in B|ML|K5|K8|A2|U) ;; *) $D/run.sh pretrain-wt --arm $arm --seeds 0 || exit 1 ;; esac
  $D/run.sh fit --arm $arm --seeds 0 || exit 1
  $D/run.sh assemble --arm $arm --seeds 0 || exit 1
  echo "ARM DONE $arm $(date -u +%FT%TZ)"
done
echo PIPELINE DONE
