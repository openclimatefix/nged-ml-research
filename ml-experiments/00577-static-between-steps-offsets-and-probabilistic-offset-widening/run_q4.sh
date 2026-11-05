#!/bin/bash
# Q3 re-forecast with the labelled-event jump statistics, then Q4 with O2; waits for run_all.sh and run_q5.sh.
cd "$(dirname "$0")/../.."
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
export DATA_PATH_INTERNAL
while pgrep -f "run_all.sh|run_q5.sh" >/dev/null; do sleep 20; done
R=ml-experiments/00577-static-between-steps-offsets-and-probabilistic-offset-widening/run.sh
rm -rf $DATA_PATH_INTERNAL/ml-experiments/00577-static-between-steps-offsets-and-probabilistic-offset-widening/work/fits/Q3 $DATA_PATH_INTERNAL/ml-experiments/00577-static-between-steps-offsets-and-probabilistic-offset-widening/work/seed_preds/Q3
for a in Q3 Q4-O2 Q4-O2q3; do
  $R e00577_fit.py fit --arm $a --seeds 0 1 || exit 1
  $R e00577_fit.py assemble --arm $a --seeds 0 1 || exit 1
done
echo Q4 DONE
