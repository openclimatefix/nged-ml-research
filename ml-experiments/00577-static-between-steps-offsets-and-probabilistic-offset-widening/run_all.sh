#!/bin/bash
# Resumable pipeline: finished outputs are skipped. Log to the private data folder.
cd "$(dirname "$0")/../.."
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
export DATA_PATH_INTERNAL
E=ml-experiments/00577-static-between-steps-offsets-and-probabilistic-offset-widening
R=$E/run.sh
P="( source ml-experiments/00524*/env.sh; .venv/bin/python $E/e00577_picks.py"
for a in O1 O2 O3; do
  $R e00577_fit.py pretrain --arm $a --seeds 0 1 || exit 1
  $R e00577_fit.py fit --arm $a --seeds 0 1 || exit 1
  $R e00577_fit.py assemble --arm $a --seeds 0 1 || exit 1
done
( source ml-experiments/00524*/env.sh; .venv/bin/python $E/e00577_picks.py jumps ) || exit 1
$R e00577_fit.py fit --arm Q2 --seeds 0 1 || exit 1
$R e00577_fit.py assemble --arm Q2 --seeds 0 1 || exit 1
$R e00577_fit.py calibrate --arm Q2 || exit 1
for a in Q1 Qconst Q3; do
  $R e00577_fit.py fit --arm $a --seeds 0 1 || exit 1
  $R e00577_fit.py assemble --arm $a --seeds 0 1 || exit 1
done
$R e00577_known_answer.py || exit 1
echo ALL DONE
