#!/bin/bash
# Resumable: every command skips finished cells. One arm after another (one GPU process at a time).
set -e
D="$(cd "$(dirname "$0")" && pwd)"
for k in ${KINDS:-y1 y1c n1 n3 y3}; do
  $D/run.sh $k pretrain
  $D/run.sh $k fit
  $D/run.sh $k assemble
  $D/run.sh $k inject
done
echo PIPELINE DONE
