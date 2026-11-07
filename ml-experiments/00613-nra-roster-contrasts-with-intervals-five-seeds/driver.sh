#!/bin/bash
# usage: driver.sh: queue every remaining GPU job of experiment 613 (waits for the d5 chain's step 2 where needed)
D="$(cd "$(dirname "$0")" && pwd)"
R=$DATA_PATH_INTERNAL/ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds
M="2025-10 2025-12 2026-01 2026-03 2026-06"
L=$R/logs
sleep 5
$D/queue.sh $L/k3_o2.log $D/gpu.sh e00613_free_o2.py k3_o2 --months $M --seeds 5 6 7 &
sleep 5
$D/queue.sh $L/k3_d5b.log $D/gpu.sh e00613_free.py k3_d5b --months $M --seeds 5 6 7 &
sleep 5
sleep 5
until [ -e $R/chains/d5/work/step2/P5/2026-06_s4.pt ]; do sleep 30; done
export CHAIN613=$R/chains/d5/work
$D/queue.sh $L/noe3_a.log $D/gpu.sh e00613_arm.py noe3_d5 fit --e3 0 --months 2026-01 2026-03 2026-06 --seeds 0 1 &
sleep 5
(for c in fit inject; do $D/queue.sh $L/noe3_b_$c.log $D/gpu.sh e00613_arm.py noe3_d5 $c --e3 0 --months $M --seeds 2 3 4; done) &
sleep 5
wait
