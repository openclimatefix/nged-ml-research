#!/bin/bash
# usage: predict_part2.sh: open the Part 2 window ONCE for the lag-enabled network at the BSP/GSP series. Run only after "go".
# Verifies the freeze, creates the lock (a second run is refused), forecasts both groups, assembles. It does not score.
D="$(cd "$(dirname "$0")" && pwd)"
R=$DATA_PATH_INTERNAL/ml-experiments/00633-bakeoff-final-test-bg2-and-lag-enabled-bsp-gsp-network
set -e
[ -z "$(ls $R/work/bsp/fits/J_LG_B1/2026-07_d0_s?.parquet $R/work/gsp/fits/J_LG_B1/2026-07_d0_s?.parquet 2>/dev/null)" ] || { echo "window forecasts exist already" >&2; exit 4; }
$D/gpu.sh e00633_freeze2.py --verify
set -o noclobber
echo "started $(date -u +%FT%TZ)" > $R/final_test_633_part2.lock
echo "started $(date -u +%FT%TZ)" > $D/final_test_633_part2.lock
set +o noclobber
# the train phase left .empty markers (no forecast window); fit_arm would skip on them
rm -f $R/work/bsp/fits/J_LG_B1*/2026-07_d0_s?.empty $R/work/gsp/fits/J_LG_B1*/2026-07_d0_s?.empty
export E00633_PHASE=predict
for g in bsp gsp; do
  $D/gpu.sh e00633_node_fit.py fit --group $g --months 2026-07 --seeds 0 1 > $R/logs/predict_part2_fit_$g.log 2>&1
  [ -s $R/work/$g/fits/J_LG_B1/2026-07_d0_s0.parquet ] && [ -s $R/work/$g/fits/J_LG_B1/2026-07_d0_s1.parquet ] || { echo "missing forecast $g" >&2; exit 3; }
  $D/gpu.sh e00633_node_fit.py assemble --group $g --months 2026-07 --seeds 0 1 > $R/logs/predict_part2_assemble_$g.log 2>&1
done
{ echo "finished $(date -u +%FT%TZ)"; sha256sum $R/work/*/seed_preds/J_LG_B1/2026-07_s?.parquet; } | tee -a $D/final_test_633_part2.lock >> $R/final_test_633_part2.lock
