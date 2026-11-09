#!/bin/bash
# usage: chain.sh MONTH SPAN STEP [COMMAND...]: one ladder step at one refit. SPAN is the months scored per
# refit (3 quarterly, 1 monthly pilot). Seeds: SEEDS="0 1 2" (default).
# STEP is 1, 2, 3a, 3b, 4, 5w, 5wo (K1-type networks: pretrain, step2, fit, pfit) or 6w, 6wo, 7w, 7wo,
# 8w, 8wo (Candidate-type networks: table, fit, ptable, pfit; they read the two-lag chain of step 5).
# Steps 1, 2, and 3a have no pre-training; they read the B1 statistics that step 3b's pre-training writes,
# so run 3b first. "w" is with the experiment 581 injected events, "wo" without.
D="$(cd "$(dirname "$0")" && pwd)"
R=${E00677_ROOT:-$DATA_PATH_INTERNAL/ml-experiments/00677-building-up-the-candidate-demand-network-one-component-at-a-time}
export E00677_ROOT=$R E00677_SPAN=$2
M=$1; SPAN=$2; S=$3; shift 3
SYN=1; case $S in 5wo|6wo|7wo|8wo) SYN=0 ;; esac
export E00677_MONTH=$M STATS677=$R/chains/3b/$M/work
A="--months $M --seeds ${SEEDS:-0 1 2}"
set -e
mkdir -p $R/logs
case $S in
  6w|6wo|7w|7wo|8w|8wo)
    ARM=$([ "${S:0:1}" = 6 ] && echo "P3+D50" || { [ "${S:0:1}" = 7 ] && echo "P5+D50" || echo "P7+D50"; })
    DIRNAME=${ARM//+/_}_${M}_${S}_sp$SPAN
    export CHAIN677=$R/chains/5/$M/work
    CMDS=${*:-table fit ptable pfit}
    [ -f $CHAIN677/step2/P5/${M}_s0.pt ] || { echo "run step 5w pretrain and step2 first"; exit 1; }
    for c in $CMDS; do
      case $c in
        table|fit) E00677_PHASE=train $D/gpu.sh e00677_cand.py $ARM $c --work-suffix _${M}_${S}_sp$SPAN --synthetic $SYN $A > $R/logs/${M}_${S}_sp${SPAN}_$c.log 2>&1 ;;
        ptable) E00677_PHASE=predict $D/gpu.sh e00677_cand.py $ARM table --work-suffix _${M}_${S}_sp$SPAN --synthetic $SYN $A > $R/logs/${M}_${S}_sp${SPAN}_$c.log 2>&1 ;;
        pfit) rm -f $R/$DIRNAME/work/fits/P5/${M}_d0_s?.empty  # the train phase's markers; fit_arm would skip on them
          E00677_PHASE=predict $D/gpu.sh e00677_cand.py $ARM fit --work-suffix _${M}_${S}_sp$SPAN --synthetic $SYN $A > $R/logs/${M}_${S}_sp${SPAN}_$c.log 2>&1 ;;
      esac
      echo "$M $S $c done $(date -u +%T)"
    done ;;
  1|2|3a|3b|4|5w|5wo)
    LADDER=${S%%[a-z]*}
    EXTRA=""; case $S in 3a|1|2) EXTRA="--ens-only 1" ;; esac
    # the chain holds pre-training and stage-2 weights; the two-lag chain is shared by 5w, 5wo and steps 6 to 8
    case $S in 5w|5wo) CH=5 ;; *) CH=$S ;; esac
    export CHAIN677=$R/chains/$CH/$M/work
    mkdir -p $CHAIN677
    if [ -n "$EXTRA" ]; then DEFAULT="step2 fit pfit"; else DEFAULT="pretrain step2 fit pfit"; fi
    CMDS=${*:-$DEFAULT}
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
    done ;;
  *) echo "bad step $S"; exit 1 ;;
esac
