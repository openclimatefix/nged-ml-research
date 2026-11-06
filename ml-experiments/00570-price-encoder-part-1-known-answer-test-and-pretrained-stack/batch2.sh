#!/bin/bash
# Rerun of the arms that failed in batch.sh (M0 pretraining was built as a price network by mistake).
HERE=$(cd "$(dirname "$0")" && pwd)
L=$DATA_PATH_INTERNAL/ml-experiments/00570-price-encoder-part-1-known-answer-test-and-pretrained-stack/logs
S="0 1"
$HERE/run.sh pretrain --arm M0 --seeds $S > $L/pre_M0.log 2>&1 || echo "FAILED pretrain M0 (2)" >> $L/batch.log
echo "pretrain M0 (2) done $(date -u +%H:%MZ)" >> $L/batch.log
for a in M0 A3-res A3-noise A3-res-oracle; do
  $HERE/run.sh fit --arm $a --seeds $S > $L/fit_$a.log 2>&1 || echo "FAILED fit $a (2)" >> $L/batch.log
  $HERE/run.sh assemble --arm $a --seeds $S >> $L/fit_$a.log 2>&1 || echo "FAILED assemble $a (2)" >> $L/batch.log
  echo "fit $a (2) done $(date -u +%H:%MZ)" >> $L/batch.log
done
echo "BATCH2 DONE $(date -u +%H:%MZ)" >> $L/batch.log
