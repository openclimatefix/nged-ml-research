#!/bin/bash
# usage: score_all.sh: score every arm of experiment 613 (CPU, one process at a time, 16 GB cap)
D="$(cd "$(dirname "$0")" && pwd)"
L=$DATA_PATH_INTERNAL/ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds/logs
set -e
$D/run.sh e00613_score.py k1_d5 --arm k1_d5 > $L/score_k1_d5.log 2>&1
$D/run.sh e00613_score.py k1_o2 --arm k1_o2 > $L/score_k1_o2.log 2>&1
$D/run.sh e00613_score.py k3_d5 --arm k1_d5 --seeds 0 1 --lagfree k3_d5 --lagfree-tag D5 --lagfree-seeds 0 1 2 3 4 > $L/score_k3_d5.log 2>&1
$D/run.sh e00613_score.py k3_d5b --arm k1_d5 --seeds 0 1 --twin k3_d5b --lagfree k3_d5b --lagfree-tag D5 --lagfree-seeds 0 1 5 6 7 > $L/score_k3_d5b.log 2>&1
$D/run.sh e00613_score.py k3_o2 --arm k1_d5 --seeds 0 1 --twin k3_d5b --lagfree k3_o2 --lagfree-tag O2 --lagfree-seeds 0 1 5 6 7 > $L/score_k3_o2.log 2>&1
