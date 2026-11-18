#!/bin/bash
# usage: chain_arm.sh MONTH ARM [PRIMARIES] [EXTRA ARGS...]
# Runs one per-substation arm for one fold, in order: [pretrain] [step2] fit (train phase), then pfit (predict
# phase, after deleting the train phase's `.empty` markers), then bank (predict phase).
#   ARM split1 : pretrain step2 fit pfit bank, one process per Primary (PRIMARIES is a space-free list, default all 16)
#   ARM split2 : step2 fit pfit bank, one process for the listed Primaries (default all)
#   ARM hybrid3: as split2
#   ARM split3 : fit pfit bank
# The fold's `fold_spans/<MONTH>.json` is written first when missing; split1 also runs `isolation` (predict phase) per Primary.
# EXTRA ARGS go to every e00702_fit.py call, for example `--batch 8192 --suffix _b8192` (the diagnostic).
# FRESH=clean or FRESH=resume: what to do with a Primary's folder left by an earlier run (default: stop).
# Seeds: SEEDS="0 1 2" (default). Output: $DATA_PATH_INTERNAL/ml-experiments/00702-*/arms/<ARM><suffix>/<MONTH>/<label>/.
D="$(cd "$(dirname "$0")" && pwd)"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
R=$DATA_PATH_INTERNAL/ml-experiments/00702-per-substation-versus-global-training-of-the-modular-demand-network
export E00677_ROOT=$R E00677_SPAN=1 E00677_MONTH=$1
M=$1; ARM=$2; PRIM=${3:-all}; shift 3 2>/dev/null || shift $#
EXTRA="$*"
SFX=""; case "$EXTRA" in *--suffix*) SFX=$(echo "$EXTRA" | sed -n 's/.*--suffix \([^ ]*\).*/\1/p') ;; esac
ALL="D2,D3,D4,D5,D8,D9,D10,D11,D12,D14,D15,D16,D18,D19,D21,D22"
[ "$PRIM" = all ] && PRIM=$ALL
set -e
mkdir -p "$R/logs"
# A Primary's folder left by an earlier run would be reused silently (existing pre-training, step-2, and fit files are
# skipped), so a fresh run refuses it. FRESH=clean deletes the folder first; FRESH=resume reuses it on purpose.
for P in ${PRIM//,/ }; do
  F="$R/arms/${ARM}${SFX}/$M/$P"
  if [ -e "$F" ]; then
    case "${FRESH:-}" in
      clean) rm -rf "$F"; echo "deleted $F" ;;
      resume) echo "reusing $F" ;;
      *) echo "outputs of an earlier run in $F: set FRESH=clean to delete them, or FRESH=resume to reuse them"; exit 1 ;;
    esac
  fi
done
# the pooled rows' spans and shifted-row hashes of the fold, which a per-substation arm checks itself against
if [ ! -f "$R/fold_spans/$M.json" ]; then
  E00677_PHASE=predict "$D/gpu.sh" e00702_fit.py split3 spans --month "$M" > "$R/logs/spans_${M}.log" 2>&1
fi
SEEDARGS="--seeds ${SEEDS:-0 1 2}"
run() {  # PHASE COMMAND PRIMARIES LOGNAME
  E00677_PHASE=$1 "$D/gpu.sh" e00702_fit.py "$ARM" "$2" --month "$M" --primaries "$3" $SEEDARGS $EXTRA > "$R/logs/${ARM}${SFX}_${M}_$4.log" 2>&1
  echo "$ARM $M $4 done $(date -u +%T)"
}
pfit() {  # PRIMARIES LOGNAME
  rm -f "$R"/arms/${ARM}${SFX}/$M/*/work/fits/P5/${M}_d0_s?.empty
  run predict fit "$1" "$2"
}
case $ARM in
  split1)
    for P in ${PRIM//,/ }; do
      run predict isolation "$P" "${P}_isolation"
      run train pretrain "$P" "${P}_pretrain"; run train step2 "$P" "${P}_step2"; run train fit "$P" "${P}_fit"
    done
    for P in ${PRIM//,/ }; do pfit "$P" "${P}_pfit"; run predict bank "$P" "${P}_bank"; done ;;
  split2|hybrid3)
    run train step2 "$PRIM" step2; run train fit "$PRIM" fit; pfit "$PRIM" pfit; run predict bank "$PRIM" bank ;;
  split3)
    run train fit "$PRIM" fit; pfit "$PRIM" pfit; run predict bank "$PRIM" bank ;;
  *) echo "bad arm $ARM"; exit 1 ;;
esac
