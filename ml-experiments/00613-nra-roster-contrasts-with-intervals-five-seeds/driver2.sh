#!/bin/bash
# usage: driver2.sh: queue the class 2 candidate (tables, fit, inject for seeds 2-4) and the 3-day-embargo injected bank for seeds 0 and 1
D="$(cd "$(dirname "$0")" && pwd)"
R=$DATA_PATH_INTERNAL/ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds
M="2025-10 2025-12 2026-01 2026-03 2026-06"
L=$R/logs
until [ -e $R/chains/d5/work/step2/P5/2026-06_s4.pt ]; do sleep 30; done
export CHAIN613=$R/chains/d5/work
for c in table fit inject; do
  $D/queue.sh $L/c2_$c.log $D/gpu.sh e00613_nbarm.py P7+D50 $c --work-suffix _e3 --months $M --seeds 2 3 4 || exit 1
done &
(until grep -q "CHECKS PASSED" $L/noe3_a.log 2>/dev/null; do sleep 30; done; $D/queue.sh $L/noe3_a_inject.log $D/gpu.sh e00613_arm.py noe3_d5 inject --e3 0 --months 2026-01 2026-03 2026-06 --seeds 0 1) &
wait
