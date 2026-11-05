#!/bin/bash
# Stage 5: A10 (per-Primary drift embedding, encoders frozen in fine-tuning): pretrain, seed-0 stress, scored fits, assemble.
D="$(cd "$(dirname "$0")" && pwd)"
$D/run.sh pretrain-wt --arm A10 || exit 1
$D/run.sh fit --arm A10 --seeds 0 --stress || exit 1
$D/run.sh fit --arm A10 --seeds 0 1 || exit 1
$D/run.sh assemble --arm A10 || exit 1
echo STAGE 5 DONE
