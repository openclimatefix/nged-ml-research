#!/bin/bash
# usage: chain.sh MONTH SPAN STEP [COMMAND...]: one ladder step (1 to 5) at one refit.
# Commands (default all, in order): pretrain (steps 3b, 4, 5 only), step2, fit (train phase), pfit (predict phase).
# Steps 1, 2, and 3a have no pre-training; they read the B1 statistics that step 3b's pre-training writes, so run 3b first.
# STEP is 1, 2, 3a, 3b, 4, 5w, or 5wo. SPAN is the months scored per refit (3 quarterly, 1 monthly pilot).
D="$(cd "$(dirname "$0")" && pwd)"
R=${E00677_ROOT:-$DATA_PATH_INTERNAL/ml-experiments/00677-building-up-the-candidate-demand-network-one-component-at-a-time}
export E00677_ROOT=$R E00677_SPAN=$2
M=$1; SPAN=$2; S=$3; shift 3
case $S in
  1|2|3a|3b|4|5w|5wo) ;;
  *) echo "bad step $S"; exit 1 ;;
esac
LADDER=${S%%[a-z]*}; [ "$S" = 5w ] && LADDER=5; [ "$S" = 5wo ] && LADDER=5
EXTRA=""; SYN=1
case $S in 3a|1|2) EXTRA="--ens-only 1" ;; 5wo) SYN=0 ;; esac
# the chain holds pre-training and stage-2 weights; the two-lag chain is shared by 5w, 5wo and steps 6 to 8
case $S in 5w|5wo) CH=5 ;; *) CH=$S ;; esac
export E00677_MONTH=$M CHAIN677=$R/chains/$CH/$M/work STATS677=$R/chains/3b/$M/work
if [ -n "$EXTRA" ]; then DEFAULT="step2 fit pfit"; else DEFAULT="pretrain step2 fit pfit"; fi
CMDS=${*:-$DEFAULT}
mkdir -p $R/logs $CHAIN677
A="--months $M --seeds ${SEEDS:-0 1 2}"
set -e
for c in $CMDS; do
  case $c in
    pretrain|step2) [ "$S" = 5wo ] && continue
      E00677_PHASE=train $D/gpu.sh e00677_k1.py k1 $c --step $LADDER $EXTRA $A > $R/logs/${M}_${S}_$c.log 2>&1 ;;
    fit) [ "$S" = 5wo ] && [ ! -f $CHAIN677/step2/P5/${M}_s0.pt ] && { echo "run 5w first"; exit 1; }
      E00677_PHASE=train $D/gpu.sh e00677_k1.py k1_$S fit --step $LADDER --synthetic $SYN $EXTRA $A > $R/logs/${M}_${S}_fit.log 2>&1 ;;
    pfit) rm -f $R/arms/k1_${S}_sp$SPAN/work/fits/P5/${M}_d0_s?.empty  # the train phase's markers; fit_arm would skip on them
      E00677_PHASE=predict $D/gpu.sh e00677_k1.py k1_$S fit --step $LADDER --synthetic $SYN $EXTRA $A > $R/logs/${M}_${S}_pfit.log 2>&1 ;;
  esac
  echo "$M $S $c done $(date -u +%T)"
done
