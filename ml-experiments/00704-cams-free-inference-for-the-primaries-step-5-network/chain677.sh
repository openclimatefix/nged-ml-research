#!/bin/bash
# usage: chain677.sh ARM MONTH [STEP...]   ARM is replay, proxy, or drop (E00704_ARM, see e00704_arm.py).
# Experiment 677's step 5w with monthly refits (span 1): steps (default all, in order) pretrain, step2,
# fit (train phase: held-out power hidden), pfit (predict phase: the month's window opened). The branch
# mirrors `1|2|3a|3b|4|5w|5wo` of experiment 677's chain.sh, except that every python step runs through
# e00704_run.py, the output root is arms/<arm>/ of this experiment's data folder, and the B1 statistics
# that pretrain writes stay in the chain's own folder (STATS677 = CHAIN677). Seeds: SEEDS="0 1 2" by default.
# fit and pfit write one checks_fit.json per arm folder, so they run under a lock (one fold of an arm at a
# time) and the file is copied to checks/<month>_<step>.json after each. Safe to run different arms or
# months at once otherwise: every other output path holds the month or the arm.
D="$(cd "$(dirname "$0")" && pwd)"
D677=$(echo "$D"/../00677-*/)
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
BASE=$DATA_PATH_INTERNAL/ml-experiments/00704-cams-free-inference-for-the-primaries-step-5-network
ARM=$1; M=$2; shift 2
case $ARM in replay|proxy|drop) ;; *) echo "bad arm $ARM"; exit 1 ;; esac
R=$BASE/arms/$ARM
export E00704_ARM=$ARM E00677_ROOT=$R E00677_SPAN=1 E00677_MONTH=$M
export CHAIN677=$R/chains/5/$M/work STATS677=$R/chains/5/$M/work
STEPS=${*:-pretrain step2 fit pfit}
A="--months $M --seeds ${SEEDS:-0 1 2}"
PY=$D/../../.venv/bin/python
set -e
mkdir -p $R/logs $CHAIN677 $R/checks
run() { "$D677"/gpu.sh ../$(basename "$D")/e00704_run.py "$@"; }
keep_checks() {  # STEP: copy checks_fit.json of this month, refusing a file that holds another month
  local src=$R/arms/k1_5w_sp1/checks_fit.json
  [ -f "$src" ] && [ "$(stat -c %Y "$src")" -ge "$2" ] || { echo "no fresh $src"; exit 1; }
  "$PY" -I - "$src" "$M" <<'PYEOF'
import json
import sys

months = set(json.load(open(sys.argv[1])).get("months", {}))
if months != {sys.argv[2]}:
    sys.exit(f"{sys.argv[1]} holds months {sorted(months)}, not {sys.argv[2]}: another fold wrote it")
PYEOF
  cp "$src" "$R/checks/${M}_$1.json"
}
touch $R/logs/.start_${M}_$$
checks() {  # the checks of e00704_checks.py; any failure ends the chain
  (cd "$D/../.." && source ml-experiments/00524*/env.sh && "$PY" "$D/e00704_checks.py" "$@")
}
stats_check() { checks stats $ARM $CHAIN677/stats/B1/$M.npz $(echo $DATA_PATH_INTERNAL/ml-experiments/00677-*/chains/3b/$M/work/stats/B1/$M.npz); }
if [ "$ARM" = proxy ]; then checks proxy-mean; fi  # before the first step
case " $STEPS " in
  " pretrain "*) ;;
  *) stats_check ;;  # the chain starts after pretrain: the stored standardiser must already be right
esac
for s in $STEPS; do
  case $s in
    pretrain|step2) E00677_PHASE=train run "00677-*/e00677_k1.py" k1 $s --step 5 $A > $R/logs/${M}_5w_$s.log 2>&1 ;;
    fit) start=$(date +%s)
      exec 9>$R/.checks.lock; flock 9
      E00677_PHASE=train run "00677-*/e00677_k1.py" k1_5w fit --step 5 --synthetic 1 $A > $R/logs/${M}_5w_fit.log 2>&1
      keep_checks fit $start
      mkdir -p $R/checks/logs  # the fit's own logs (epochs); the pfit that follows overwrites them
      for f in $R/arms/k1_5w_sp1/work/logs/P5/${M}_d0_s*.json; do
        if grep -q '"epochs"' "$f"; then cp "$f" $R/checks/logs/; fi
      done
      flock -u 9 ;;
    pfit) start=$(date +%s)
      exec 9>$R/.checks.lock; flock 9
      rm -f $R/arms/k1_5w_sp1/work/fits/P5/${M}_d0_s?.empty  # the train phase's markers; fit_arm would skip on them
      E00677_PHASE=predict run "00677-*/e00677_k1.py" k1_5w fit --step 5 --synthetic 1 $A > $R/logs/${M}_5w_pfit.log 2>&1
      keep_checks pfit $start
      flock -u 9 ;;
    *) echo "bad step $s"; exit 1 ;;
  esac
  if [ "$s" = pretrain ]; then stats_check; fi
  echo "$ARM $M $s done $(date -u +%T)"
done
# nothing may be written under the original experiments' data folders (other sessions may also write there: a warning, not a failure)
for o in 00667 00677; do
  find $DATA_PATH_INTERNAL/ml-experiments/$o-* -newer $R/logs/.start_${M}_$$ -type f 2>/dev/null | head -5 | sed "s/^/WARNING: file written under experiment $o: /"
done
