#!/bin/bash
# Fits, assembles, and stress-fits every arm; ARMS overrides the arm list. Needs DATA_PATH_INTERNAL.
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for arm in ${ARMS:-F1 F3 F2}; do
  "$D/run.sh" fit --arm "$arm" && "$D/run.sh" assemble --arm "$arm" || exit 1
done
for arm in ${ARMS:-F1 F3 F2}; do "$D/run.sh" fit --arm "$arm" --stress --seeds 0 || exit 1; done
echo ALL DONE
