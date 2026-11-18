#!/bin/bash
# usage: chain_global.sh MONTH [COMMAND...]   (default: pretrain step2 fit pfit bank)
# Fits the control arm of this experiment (`e00702_fit.py global`: all 16 Primaries at once, batch 8,192, step 5w of
# experiment 677 at monthly refits) for one fold. It reads and writes only this experiment's data folder; experiment
# 677's B1 statistics are read to check that the refit's own statistics equal them.
#   Control pair (seeds 3 to 5), per test month:  chain_global.sh 2025-11
#   First run (seed 0, the pilot fold):           SEEDS="0" SUFFIX=_pilot chain_global.sh 2026-05
# Output: arms/global<SUFFIX>/<MONTH>/pooled/work/fits/P5/<MONTH>_d0_s<seed>.parquet. Export it with
#   e00702_export.py global_pair MONTH --seeds 3 4 5            (the control pair)
#   e00702_export.py global_pilot 2026-05 --seeds 0 --name global_pilot     (the match-refit check; the pilot fit sits in arms/global_pilot)
# The command `bank` also runs the bank on the refit networks; `bank-stored` runs it on experiment 677's stored step-3
# networks (output arms/global_stored/...), for `e00702_score.py match-bank --months MONTH --seed S`.
D="$(cd "$(dirname "$0")" && pwd)"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
R=$DATA_PATH_INTERNAL/ml-experiments/00702-per-substation-versus-global-training-of-the-modular-demand-network
export E00677_ROOT=$R E00677_SPAN=1
M=$1; shift
export E00677_MONTH=$M
SFX=${SUFFIX:-}
SEEDARGS="--seeds ${SEEDS:-3 4 5}"
set -e
mkdir -p "$R/logs"
run() {  # PHASE COMMAND LOGNAME [EXTRA...]
  local phase=$1 cmd=$2 log=$3; shift 3
  E00677_PHASE=$phase "$D/gpu.sh" e00702_fit.py global "$cmd" --month "$M" $SEEDARGS --suffix "$SFX" "$@" > "$R/logs/global${SFX}_${M}_$log.log" 2>&1
}
CMDS=${*:-pretrain step2 fit pfit bank}
# A run that starts with pre-training refuses a folder left by an earlier run (its files would be skipped and reused).
# FRESH=clean deletes the folder first; FRESH=resume reuses it on purpose.
F="$R/arms/global${SFX}/$M/pooled"
case " $CMDS " in
  *" pretrain "*)
    if [ -e "$F" ]; then
      case "${FRESH:-}" in
        clean) rm -rf "$F"; echo "deleted $F" ;;
        resume) echo "reusing $F" ;;
        *) echo "outputs of an earlier run in $F: set FRESH=clean to delete them, or FRESH=resume to reuse them"; exit 1 ;;
      esac
    fi ;;
esac
for c in $CMDS; do
  case $c in
    pretrain|step2|fit) run train "$c" "$c" ;;
    pfit)
      rm -f "$R"/arms/global${SFX}/$M/pooled/work/fits/P5/${M}_d0_s?.empty
      run predict fit pfit ;;
    bank) run predict bank bank ;;
    bank-stored) SFX=_stored run predict bank bank_stored --stored-models ;;
    *) echo "bad command $c"; exit 1 ;;
  esac
  echo "$M global $c done $(date -u +%T)"
done
