#!/bin/bash
# Resumable (every command skips finished cells). Usage: pipeline.sh ARM [ARM ...]; A4 runs its own chain first.
set -e
D="$(cd "$(dirname "$0")" && pwd)"
for a in "$@"; do
  if [ "$a" = A4 ]; then
    $D/run.sh A4 pretrain
    $D/run.sh A4 step2
  fi
  $D/run.sh $a fit
  $D/run.sh $a assemble
  $D/run.sh $a inject
done
echo PIPELINE DONE
