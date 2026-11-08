#!/bin/bash
# usage: enc_queue6.sh; the encoder arms with the reviewer's changes (valid-time block validation, E3 branch at lr 3e-5 for at most 3 epochs); one GPU process at a time
D="$(cd "$(dirname "$0")" && pwd)"
run() { nvidia-smi --query-gpu=memory.used --format=csv,noheader; $D/gpu.sh e00647_enc.py "$@" 2>&1 | grep -v mlflow || echo "FAIL $*"; }
for arm in E2 E1; do for m in 2025-10 2025-12 2026-01 2026-03 2026-06; do run $arm $m lomo --seeds 0 1 2 3 4; done; done
for arm in E3 E3p E2 E1; do run $arm 2025-12 fcb --seeds 0 1; done
echo QUEUEDONE
