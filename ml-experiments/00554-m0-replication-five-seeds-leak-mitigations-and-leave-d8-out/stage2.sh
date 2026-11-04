#!/bin/bash
# Stage 2: diagnostic and mitigation arms, 3 seeds (0-2) each, then seed-0 stress runs; then assemble.
D="$(cd "$(dirname "$0")" && pwd)"
for arm in M0-hl180 M0-c3 M0-sep M0-mix; do
  $D/run.sh fit --arm $arm --seeds 0 1 2 || exit 1
  $D/run.sh fit --arm $arm --seeds 0 --stress || exit 1
  $D/run.sh assemble --arm $arm || exit 1
done
echo STAGE 2 DONE
