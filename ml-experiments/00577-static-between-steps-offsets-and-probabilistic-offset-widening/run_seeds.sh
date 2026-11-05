#!/bin/bash
# Post hoc Q3b, then fresh seeds 5-9 for the arms with a point gain (O2 and Q4-O2).
cd /home/jack/dev/nged-ml-research/.claude/worktrees/offset-tv-probabilistic
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
R=ml-experiments/00577-static-between-steps-offsets-and-probabilistic-offset-widening/run.sh
S="5 6 7 8 9"
$R e00577_fit.py fit --arm Q3b --seeds 0 1 || exit 1
$R e00577_fit.py assemble --arm Q3b --seeds 0 1 || exit 1
$R e00577_fit.py pretrain --arm O2 --seeds $S || exit 1
for a in O2 Q4-O2; do
  $R e00577_fit.py fit --arm $a --seeds $S || exit 1
  $R e00577_fit.py assemble --arm $a --seeds $S || exit 1
done
echo SEEDS DONE
