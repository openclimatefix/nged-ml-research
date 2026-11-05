#!/bin/bash
# Stage 2 (after the stop rule passes): A4 (gentle fine-tune), A6 (180-day half-life), A7 (3-day cadence).
# Seed-0 stress first so the leak is known early; then the 2-seed scored fits; then assemble.
D="$(cd "$(dirname "$0")" && pwd)"
for arm in A4 A6 A7; do
  $D/run.sh fit --arm $arm --seeds 0 --stress || exit 1
  $D/run.sh fit --arm $arm --seeds 0 1 || exit 1
  $D/run.sh assemble --arm $arm || exit 1
done
echo STAGE 2 DONE
