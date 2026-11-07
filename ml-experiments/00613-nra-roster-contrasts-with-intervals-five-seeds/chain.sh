#!/bin/bash
# usage: chain.sh NAME CHAIN_DIR [extra arm args]: pre-training, step 2, fit, and the injected bank of one K1 arm on seeds 2 to 4
D="$(cd "$(dirname "$0")" && pwd)"
N=$1; export CHAIN613=$2; shift 2
M="2025-10 2025-12 2026-01 2026-03 2026-06"
set -e
for c in pretrain step2 fit inject; do
  $D/gpu.sh e00613_arm.py $N $c --months $M --seeds 2 3 4 "$@"
done
