#!/bin/bash
# usage: predict_part1.sh: open the final-test window ONCE for BG2. Run only after the manager's "go".
# Verifies the freeze, creates the lock (a second run is refused), forecasts all 123 runs, then scores.
D="$(cd "$(dirname "$0")" && pwd)"
R=$DATA_PATH_INTERNAL/ml-experiments/00633-bakeoff-final-test-bg2-and-lag-enabled-bsp-gsp-network
set -e
$D/gpu.sh e00633_freeze.py --verify
set -o noclobber
echo "started $(date -u +%FT%TZ)" > $R/final_test_633_part1.lock
echo "started $(date -u +%FT%TZ)" > $D/final_test_633_part1.lock
set +o noclobber
export E00633_PHASE=predict
$D/gpu.sh e00633_bg2.py predict > $R/logs/predict_bg2.log 2>&1
for s in D2 D3 D4 D5 D8 D9 D10 D11 D12 D14 D15 D16 D18 D19 D21 D22; do [ -s $R/preds/bg2_single_fit/$s.parquet ] || { echo "missing forecast $s" >&2; exit 3; }; done
{ echo "finished $(date -u +%FT%TZ)"; sha256sum $R/preds/bg2_single_fit/*.parquet; } | tee -a $D/final_test_633_part1.lock >> $R/final_test_633_part1.lock
