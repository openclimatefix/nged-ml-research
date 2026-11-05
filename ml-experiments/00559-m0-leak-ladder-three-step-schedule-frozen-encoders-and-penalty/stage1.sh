#!/bin/bash
# Stage 1: scored fits (seeds 0-1) of A1-A3 and A4, A5 (average), then assemble.
D="$(cd "$(dirname "$0")" && pwd)"
for arm in A1 A2 A3; do
  $D/run.sh fit --arm $arm --seeds 0 1 || exit 1
  $D/run.sh assemble --arm $arm || exit 1
done
$D/run.sh average --arm A5 --seeds 0 1 || exit 1
$D/run.sh average --arm A5 --seeds 0 --stress || exit 1
$D/run.sh assemble --arm A5 || exit 1
echo STAGE 1 DONE
