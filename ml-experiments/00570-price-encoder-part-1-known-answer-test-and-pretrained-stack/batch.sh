#!/bin/bash
# Sequential batch for experiment 570 (one GPU process at a time). usage: batch.sh
HERE=$(cd "$(dirname "$0")" && pwd)
L=$DATA_PATH_INTERNAL/ml-experiments/00570-price-encoder-part-1-known-answer-test-and-pretrained-stack/logs
S="0 1"
for a in M0 A6-res A6-raw A6-p1 A6-p2 A6-noise A6-const; do
  $HERE/run.sh pretrain --arm $a --seeds $S > $L/pre_$a.log 2>&1 || echo "FAILED pretrain $a" >> $L/batch.log
  echo "pretrain $a done $(date -u +%H:%MZ)" >> $L/batch.log
done
for a in M0 A6-res A6-raw A6-p1 A6-p2 A6-noise A6-const A1-res A1-p2 A1-noise A1-const A3-res A3-noise A1-res-oracle A1-p2-oracle A3-res-oracle; do
  $HERE/run.sh fit --arm $a --seeds $S > $L/fit_$a.log 2>&1 || echo "FAILED fit $a" >> $L/batch.log
  $HERE/run.sh assemble --arm $a --seeds $S >> $L/fit_$a.log 2>&1 || echo "FAILED assemble $a" >> $L/batch.log
  echo "fit $a done $(date -u +%H:%MZ)" >> $L/batch.log
done
echo "BATCH DONE $(date -u +%H:%MZ)" >> $L/batch.log
