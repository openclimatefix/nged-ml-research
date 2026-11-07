#!/bin/bash
# usage: score_rest.sh SFX SEEDS_K1 SEEDS_K3: score the remaining arms for one seed set (SFX may be empty)
D="$(cd "$(dirname "$0")" && pwd)"
S=$1; SEEDS=($2); K3=($3)
L=$DATA_PATH_INTERNAL/ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds/logs
SC=$DATA_PATH_INTERNAL/ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds/score
set -e
if [ -n "$S" ]; then
  $D/run.sh e00613_score.py k1_d5$S --arm k1_d5 --seeds ${SEEDS[@]} > $L/score_k1_d5$S.log 2>&1
  $D/run.sh e00613_score.py k1_o2$S --arm k1_o2 --seeds ${SEEDS[@]} > $L/score_k1_o2$S.log 2>&1
  $D/run.sh e00613_score.py c2_e3$S --arm c2_e3 --seeds ${SEEDS[@]} > $L/score_c2_e3$S.log 2>&1
  $D/run.sh e00613_score.py k3_d5b$S --arm k1_d5 --seeds 0 1 --twin k3_d5b --lagfree k3_d5b --lagfree-tag D5 --lagfree-seeds ${K3[@]} > $L/score_k3_d5b$S.log 2>&1
  $D/run.sh e00613_score.py k3_o2$S --arm k1_d5 --seeds 0 1 --twin k3_d5b --lagfree k3_o2 --lagfree-tag O2 --lagfree-seeds ${K3[@]} > $L/score_k3_o2$S.log 2>&1
fi
$D/run.sh e00613_score.py noe3_d5$S --arm noe3_d5 --seeds ${SEEDS[@]} > $L/score_noe3_d5$S.log 2>&1
$D/run.sh e00613_widths.py --suffix "$S" > $L/widths$S.log 2>&1
$D/run.sh e00613_score.py c2_e3w$S --arm c2_e3 --seeds ${SEEDS[@]} --width-factors $SC/c2_width_factors$S.csv > $L/score_c2_e3w$S.log 2>&1
$D/run.sh e00613_score.py c2_e3wo$S --arm c2_e3 --seeds ${SEEDS[@]} --width-factors $SC/c2_width_factors_loo$S.csv > $L/score_c2_e3wo$S.log 2>&1
