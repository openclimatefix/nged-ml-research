#!/bin/bash
# usage: e00695_chain.sh ARM MONTH [COMMAND...]: experiment 677's step 9a (monthly refit of step 5 with injected
# shifts) on the power rows of ARM, for one fold month. ARM is A0 (the raw control), A1, A1n, AH, W0, or W1.
# Commands, in order: clean pretrain step2 fit pfit (default: pretrain step2 fit pfit; `clean` runs
# e00695_clean.py first and needs no GPU). Seeds: SEEDS="0 1 2" (default).
# Everything is written under this experiment's data folder; experiment 677's folders are read only.
D="$(cd "$(dirname "$0")" && pwd)"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
NAME=00695-gas-engine-state-tracker-part-2-cleaned-target-and-climatology-add-back
R=${E00695_ROOT:-$DATA_PATH_INTERNAL/ml-experiments/$NAME}
R677=$DATA_PATH_INTERNAL/ml-experiments/00677-building-up-the-candidate-demand-network-one-component-at-a-time
ARM=$1; M=$2; shift 2
export E00695_ARM=$ARM E00677_ROOT=$R/runs/$ARM E00677_SPAN=1 E00677_MONTH=$M
export CHAIN677=$R/runs/$ARM/chains/5/$M/work STATS677=$R/runs/$ARM/chains/stats/$M/work
A="--months $M --seeds ${SEEDS:-0 1 2}"
set -e
mkdir -p $R/runs/$ARM/logs $CHAIN677 $STATS677/stats
# the B1 columns are weather statistics, the same for every arm: start from experiment 677's file
[ -f $STATS677/stats/B1/$M.npz ] || cp -r $R677/chains/3b/$M/work/stats/. $STATS677/stats/
CMDS=${*:-pretrain step2 fit pfit}
# no network is fitted before gate G1 has passed (set E00695_SKIP_G1=1 only for a deliberate exception)
case " $CMDS " in *" pretrain "*|*" step2 "*|*" fit "*|*" pfit "*)
  if [ -z "$E00695_SKIP_G1" ] && ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$R/checks.json')).get('g1_passed') is True else 1)" 2>/dev/null; then
    echo "gate G1 has not passed: run e00695_checks.py --g1 --write first"; exit 1
  fi ;;
esac
for c in $CMDS; do
  case $c in
    clean) [ "$ARM" = A0 ] || ( cd $D/../.. && source ml-experiments/00524*/env.sh && PYTHONPATH="src:$PYTHONPATH" .venv/bin/python $D/e00695_clean.py $ARM --months $M --workers 1 ) ;;
    pretrain|step2)
      E00677_PHASE=train $D/e00695_gpu.sh e00695_k1.py k1 $c --step 5 $A > $R/runs/$ARM/logs/${M}_$c.log 2>&1 ;;
    fit)
      E00677_PHASE=train $D/e00695_gpu.sh e00695_k1.py k1_5w fit --step 5 --synthetic 1 $A > $R/runs/$ARM/logs/${M}_fit.log 2>&1 ;;
    pfit)
      rm -f $R/runs/$ARM/arms/k1_5w_sp1/work/fits/P5/${M}_d0_s?.empty  # the train phase's markers; fit_arm would skip on them
      E00677_PHASE=predict $D/e00695_gpu.sh e00695_k1.py k1_5w fit --step 5 --synthetic 1 $A > $R/runs/$ARM/logs/${M}_pfit.log 2>&1 ;;
    *) echo "bad command $c"; exit 1 ;;
  esac
  echo "$ARM $M $c done $(date -u +%T)"
done
