#!/bin/bash
# usage: predict.sh: open the final-test window ONCE. Run only after the manager's "go".
# Verifies the freeze, creates the lock (a second run is refused), forecasts all three networks, assembles.
D="$(cd "$(dirname "$0")" && pwd)"
R=$DATA_PATH_INTERNAL/ml-experiments/00620-bakeoff-final-test-demand-networks-at-primaries
set -e
# a dry-run or bisection switch left set would use up the single run
[ -z "${E00620_DRY:-}" ] && [ -z "${E00620_SKIP:-}" ] || { echo "E00620_DRY / E00620_SKIP must be unset" >&2; exit 4; }
$D/gpu.sh e00620_freeze.py --verify
# a second lock in the data folder survives a fresh checkout
set -o noclobber
echo "started $(date -u +%FT%TZ)" > $R/final_test_620.lock
echo "started $(date -u +%FT%TZ)" > $D/final_test_620.lock
set +o noclobber
# the train phase left .empty markers (no forecast window); fit_arm would skip on them
rm -f $R/arms/k1/work/fits/P5*/2026-07_d0_s?.empty $R/arms/k3/work/fits/D5*/2026-07_d0_s?.empty $R/P7_D50_e3/work/fits/P5*/2026-07_d0_s?.empty
need() { for f in "$@"; do [ -s "$f" ] || { echo "missing forecast $f" >&2; exit 3; }; done; }
export E00620_PHASE=predict
M="--months 2026-07 --seeds 0 1 2 3 4"
CHAIN620=$R/chains/k1/work $D/gpu.sh e00620_k1.py k1 fit $M > $R/logs/final_k1.log 2>&1
need $R/arms/k1/work/fits/P5/2026-07_d0_s{0..4}.parquet $R/arms/k1/work/fits/P5-removed/2026-07_d0_s{0..4}.parquet
CHAIN620=$R/chains/k3/work $D/gpu.sh e00620_k3.py k3 fit $M > $R/logs/final_k3.log 2>&1
need $R/arms/k3/work/fits/D5/2026-07_d0_s{0..4}.parquet
export CHAIN620=$R/chains/k1/work
$D/gpu.sh e00620_cand.py P7+D50 table --work-suffix _e3 $M > $R/logs/final_cand_table.log 2>&1
$D/gpu.sh e00620_cand.py P7+D50 fit --work-suffix _e3 $M > $R/logs/final_cand_fit.log 2>&1
need $R/P7_D50_e3/work/fits/P5/2026-07_d0_s{0..4}.parquet
$D/gpu.sh e00620_assemble.py 2026-07 > $R/logs/final_assemble.log 2>&1
need $R/export/{K1,Candidate,K3}_2026-07.parquet
{ echo "finished $(date -u +%FT%TZ)"; sha256sum $R/export/*_2026-07.parquet $R/tables/predict/T_2026-07_s?.npz $R/arms/*/work/fits/*/2026-07_d0_s?.parquet $R/P7_D50_e3/work/fits/*/2026-07_d0_s?.parquet; } | tee -a $D/final_test_620.lock >> $R/final_test_620.lock
