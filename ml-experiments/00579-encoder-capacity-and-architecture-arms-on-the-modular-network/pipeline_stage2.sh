#!/bin/bash
# Stage 2 (passers): seed 1, the injected-shift stress fits (seed 0), and assembly. Resumable.
# usage: pipeline_stage2.sh ARM [ARM ...]
D="$(cd "$(dirname "$0")" && pwd)"
for arm in "$@"; do
  case $arm in B|ML|K5|K8|A2|U) ;; *) $D/run.sh pretrain-wt --arm $arm --seeds 1 || exit 1 ;; esac
  $D/run.sh fit --arm $arm --seeds 1 || exit 1
  $D/run.sh assemble --arm $arm --seeds 1 || exit 1
  $D/run.sh fit --arm $arm --seeds 0 --stress || exit 1
  echo "ARM DONE $arm $(date -u +%FT%TZ)"
done
echo PIPELINE DONE
