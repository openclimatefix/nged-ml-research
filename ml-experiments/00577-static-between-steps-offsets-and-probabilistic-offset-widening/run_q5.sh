#!/bin/bash
# Q5 (scale offset) and its shuffled placebo; waits for run_all.sh.
cd /home/jack/dev/nged-ml-research/.claude/worktrees/offset-tv-probabilistic
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
while pgrep -f run_all.sh >/dev/null; do sleep 20; done
R=ml-experiments/00577-static-between-steps-offsets-and-probabilistic-offset-widening/run.sh
$R e00577_fit.py pretrain --arm Q5 --seeds 0 1 || exit 1
for a in Q5 Q5shuf; do
  $R e00577_fit.py fit --arm $a --seeds 0 1 || exit 1
  $R e00577_fit.py assemble --arm $a --seeds 0 1 || exit 1
done
echo Q5 DONE
