#!/bin/bash
# usage: e00696_chain.sh ARM MONTH [COMMAND...]: experiment 677's step 9a (monthly refit of step 5 with injected
# shifts), run by experiment 695's route (e00695_k1.py's stack replacement and chain) with the engine state in
# the pre-training mixer, for one fold month. ARM is one of the arms of e00696_common.ARMS (A0, G0, S1, SP, SZ,
# S1d, W0, W0p, WST, WS1). Commands, in order: state pretrain step2 fit pfit (default: pretrain step2 fit pfit;
# `state` builds the arm's per-fold state file on the CPU; W0p takes only `pretrain`).
# Seeds: SEEDS="0 1 2" (default; A0 and W0 are run for seeds 3 to 9 only). Seed groups of one arm and fold share
# folders, so each group's log carries its seeds in its name.
# Everything is written under this experiment's data folder; experiments 677 and 695 are read only.
D="$(cd "$(dirname "$0")" && pwd)"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
NAME=00696-gas-engine-state-tracker-part-3-engine-state-in-the-pre-training-mixer
R=${E00696_ROOT:-$DATA_PATH_INTERNAL/ml-experiments/$NAME}
R677=$DATA_PATH_INTERNAL/ml-experiments/00677-building-up-the-candidate-demand-network-one-component-at-a-time
ARM=$1; M=$2; shift 2
SEEDS=${SEEDS:-0 1 2}
TAG=$(echo $SEEDS | tr ' ' '-')
export E00696_ARM=$ARM E00677_ROOT=$R/runs/$ARM E00677_SPAN=1 E00677_MONTH=$M
export CHAIN677=$R/runs/$ARM/chains/5/$M/work STATS677=$R/runs/$ARM/chains/stats/$M/work
A="--months $M --seeds $SEEDS"
set -e
mkdir -p $R/runs/$ARM/logs $CHAIN677 $STATS677/stats
# the B1 columns are weather statistics, the same for every arm: start from experiment 677's file
[ -f $STATS677/stats/B1/$M.npz ] || cp -r $R677/chains/3b/$M/work/stats/. $STATS677/stats/
CMDS=${*:-pretrain step2 fit pfit}
# no network is fitted before the synthetic checks have passed (e00696_checks.py --write)
case " $CMDS " in *" pretrain "*|*" step2 "*|*" fit "*|*" pfit "*)
  if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$R/checks.json')).get('synthetic_passed') is True else 1)" 2>/dev/null; then
    echo "the synthetic checks have not passed: run e00696_checks.py --write first"; exit 1
  fi ;;
esac
for c in $CMDS; do
  case $c in
    state) ( cd $D/../.. && source ml-experiments/00524*/env.sh && PYTHONPATH="src:$PYTHONPATH" .venv/bin/python $D/e00696_state.py $ARM --months $M ) ;;
    pretrain|step2)
      E00677_PHASE=train $D/e00696_gpu.sh e00696_k1.py k1 $c --step 5 $A > $R/runs/$ARM/logs/${M}_${c}_$TAG.log 2>&1 ;;
    fit)
      E00677_PHASE=train $D/e00696_gpu.sh e00696_k1.py k1_5w fit --step 5 --synthetic 1 $A > $R/runs/$ARM/logs/${M}_fit_$TAG.log 2>&1 ;;
    pfit)
      for s in $SEEDS; do rm -f $R/runs/$ARM/arms/k1_5w_sp1/work/fits/P5/${M}_d0_s${s}.empty; done  # the train phase's markers; fit_arm would skip on them
      E00677_PHASE=predict $D/e00696_gpu.sh e00696_k1.py k1_5w fit --step 5 --synthetic 1 $A > $R/runs/$ARM/logs/${M}_pfit_$TAG.log 2>&1 ;;
    *) echo "bad command $c"; exit 1 ;;
  esac
  echo "$ARM $M $c done $(date -u +%T)"
done
