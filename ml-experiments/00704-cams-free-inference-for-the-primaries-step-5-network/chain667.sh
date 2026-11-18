#!/bin/bash
# usage: chain667.sh ARM MONTH [STEP...]   ARM is replay, proxy, or drop (E00704_ARM, see e00704_arm.py).
# Experiment 667's chain for one fold (the Candidate of experiment 703), every python step through
# e00704_run.py with 667's gpu.sh. Steps (default all, in order): pretrain, step2 (K1), table, fit
# (Candidate, train phase), ptable, pfit (predict phase). Seeds 0 and 1. Output root: arms/<arm>/ of
# experiment 704's data folder. Every output path holds the arm and the month (the Candidate's
# checks_fit file is in P7_D50_<month>/), so any mix of up to three arms and months can run at once.
D="$(cd "$(dirname "$0")" && pwd)"
D667=$(echo "$D"/../00667-*/)
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
BASE=$DATA_PATH_INTERNAL/ml-experiments/00704-cams-free-inference-for-the-primaries-step-5-network
ARM=$1; M=$2; shift 2
case $ARM in replay|proxy|drop) ;; *) echo "bad arm $ARM"; exit 1 ;; esac
R=$BASE/arms/$ARM
export E00704_ARM=$ARM E00667_ROOT=$R E00667_MONTH=$M CHAIN667=$R/chains/$M/work
STEPS=${*:-pretrain step2 table fit ptable pfit}
A="--months $M --seeds 0 1"
PY=$D/../../.venv/bin/python
set -e
mkdir -p $R/logs $CHAIN667
run() { "$D667"/gpu.sh ../$(basename "$D")/e00704_run.py "$@"; }
touch $R/logs/.start_${M}_$$
checks() {  # the checks of e00704_checks.py; any failure ends the chain
  (cd "$D/../.." && source ml-experiments/00524*/env.sh && "$PY" "$D/e00704_checks.py" "$@")
}
stats_check() { checks stats $ARM $CHAIN667/stats/B1/$M.npz $(echo $DATA_PATH_INTERNAL/ml-experiments/00667-*/chains/$M/work/stats/B1/$M.npz); }
if [ "$ARM" = proxy ]; then checks proxy-mean; fi  # before the first step
case " $STEPS " in
  " pretrain "*) ;;
  *) stats_check ;;  # the chain starts after pretrain: the stored standardiser must already be right
esac
for s in $STEPS; do
  case $s in
    pretrain|step2) E00667_PHASE=train run "00667-*/e00667_k1.py" k1 $s $A > $R/logs/${M}_$s.log 2>&1 ;;
    table|fit) E00667_PHASE=train run "00667-*/e00667_cand.py" P7+D50 $s --work-suffix _$M $A > $R/logs/${M}_c$s.log 2>&1 ;;
    ptable) E00667_PHASE=predict run "00667-*/e00667_cand.py" P7+D50 table --work-suffix _$M $A > $R/logs/${M}_c$s.log 2>&1 ;;
    pfit) rm -f $R/P7_D50_$M/work/fits/P5/${M}_d0_s?.empty  # the train phase's markers; fit_arm would skip on them
      E00667_PHASE=predict run "00667-*/e00667_cand.py" P7+D50 fit --work-suffix _$M $A > $R/logs/${M}_c$s.log 2>&1 ;;
    *) echo "bad step $s"; exit 1 ;;
  esac
  if [ "$s" = pretrain ]; then stats_check; fi
  echo "$ARM $M $s done $(date -u +%T)"
done
# nothing may be written under the original experiments' data folders (other sessions may also write there: a warning, not a failure)
for o in 00667 00677; do
  find $DATA_PATH_INTERNAL/ml-experiments/$o-* -newer $R/logs/.start_${M}_$$ -type f 2>/dev/null | head -5 | sed "s/^/WARNING: file written under experiment $o: /"
done
