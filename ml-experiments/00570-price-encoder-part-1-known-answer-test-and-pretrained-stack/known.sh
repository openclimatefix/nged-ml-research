#!/bin/bash
# Known-answer runs, sequential: R1 on all five folds, the R1 contamination run and rules R2, R3 on two folds.
HERE=$(cd "$(dirname "$0")" && pwd)
L=$DATA_PATH_INTERNAL/ml-experiments/00570-price-encoder-part-1-known-answer-test-and-pretrained-stack/logs
$HERE/run.sh known --arm known --rules R1 --variants all --months 2025-10 2025-12 2026-01 2026-03 2026-06 --seeds 0 > $L/known_R1.log 2>&1
$HERE/run.sh known --arm known --rules R1 --variants train --months 2025-10 2026-01 --seeds 0 > $L/known_K4.log 2>&1
$HERE/run.sh known --arm known --rules R2 R3 --variants all --months 2025-10 2026-01 --seeds 0 > $L/known_R23.log 2>&1
echo "KNOWN DONE $(date -u +%H:%MZ)" >> $L/batch.log
