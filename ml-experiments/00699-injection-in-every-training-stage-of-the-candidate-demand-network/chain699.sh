#!/bin/bash
# usage: chain699.sh CHAIN MONTH [PART]
#   CHAIN is clean, stage2, or early (a sub-folder of roots/ in this experiment's data folder).
#   PART is all (default), fits (no bank), a0 (the stage-3-only arm alone, for the reproduction check),
#   bank581 (the bank of experiment 677), or bank660 (the second bank).
# One fold of one chain, one command of experiment 677's chain.sh at a time. After each fit and pfit the arm's
# checks_fit.json is copied to checks/<chain>/<arm>/<month>_<command>.json, because the next call overwrites it.
# Seeds: SEEDS="0 1 2 3 4 5 6 7" by default.
D="$(cd "$(dirname "$0")" && pwd)"
R=$(cd "$D/../.." && pwd)
D677=$(echo "$D"/../00677-*/)
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
BASE=$DATA_PATH_INTERNAL/ml-experiments/00699-injection-in-every-training-stage-of-the-candidate-demand-network
C=$1; M=$2; PART=${3:-all}
case $C in clean|stage2|early) ;; *) echo "bad chain $C"; exit 1 ;; esac
export E00677_ROOT=$BASE/roots/$C E00677_SPAN=1 E00677_MONTH=$M
export CHAIN677=$E00677_ROOT/chains/5/$M/work STATS677=$E00677_ROOT/chains/3b/$M/work
export SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
CLEAN=$BASE/roots/clean
PY=$R/.venv/bin/python
set -e
mkdir -p "$E00677_ROOT/logs" "$CHAIN677" "$BASE/checks/$C"

run677() {  # STEP COMMAND: one command of experiment 677's chain.sh, then keep its checks file
  local step=$1 cmd=$2 start
  start=$(date +%s)
  "$D677"/chain.sh "$M" 1 "$step" $cmd
  case $cmd in
    fit|pfit)
      local src=$E00677_ROOT/arms/k1_${step}_sp1/checks_fit.json
      [ -f "$src" ] && [ "$(stat -c %Y "$src")" -ge "$start" ] || { echo "no fresh $src"; exit 1; }
      mkdir -p "$BASE/checks/$C/$step"
      cp "$src" "$BASE/checks/$C/$step/${M}_$cmd.json"
      if [ "$cmd" = fit ]; then  # the fit's own logs (rows, epochs); the pfit that follows overwrites them
        mkdir -p "$BASE/checks/$C/$step/logs"
        cp "$E00677_ROOT"/arms/k1_${step}_sp1/work/logs/P5/${M}_d0_s*.json "$BASE/checks/$C/$step/logs/"
      fi ;;
  esac
}

k1() {  # COMMAND OPTION...: stage 1 or stage 2 with shifted lags, through this experiment's gpu script
  local cmd=$1; shift
  E00677_PHASE=train "$D"/gpu699.sh e00699_k1.py k1 "$cmd" --step 5 "$@" --months "$M" --seeds $SEEDS \
    > "$E00677_ROOT/logs/${M}_5w_$cmd.log" 2>&1
  echo "$M $C $cmd done $(date -u +%T)"
}

guard() {  # refuse a stage-3 fit unless injection.json names exactly the stages that this chain injects
  "$PY" -I - "$CHAIN677/injection.json" "$C" <<'PYEOF'
import json
import pathlib
import sys

path, chain = pathlib.Path(sys.argv[1]), sys.argv[2]
found = json.loads(path.read_text()) if path.exists() else {}
have = {k for k in found if k in ("stage1", "stage2")}
want = {"clean": set(), "stage2": {"stage2"}, "early": {"stage1", "stage2"}}[chain]
if have != want:
    sys.exit(f"injection.json names {sorted(have)} but chain {chain} must inject {sorted(want)}")
PYEOF
}

fits_clean() {
  run677 5w pretrain; run677 5w step2
  guard; run677 5w fit; run677 5w pfit
  if [ "$PART" != a0 ]; then
    guard; run677 5wo fit; run677 5wo pfit
  fi
}
fits_stage2() {
  [ -d "$CLEAN/chains/5/$M/work/pretrained_P2" ] || { echo "run the clean chain of $M first"; exit 1; }
  ln -sfn "$CLEAN/chains/5/$M/work/pretrained_P2" "$CHAIN677/pretrained_P2"
  mkdir -p "$STATS677"
  ln -sfn "$CLEAN/chains/3b/$M/work/stats" "$STATS677/stats"
  k1 step2 --inject-stage2 1
  guard; run677 5w fit; run677 5w pfit
}
fits_early() {
  k1 pretrain --inject-stage1 1
  k1 step2 --inject-stage2 1
  guard; run677 5w fit; run677 5w pfit
  guard; run677 5wo fit; run677 5wo pfit
}

case $PART in
  all|fits|a0)
    case $C in clean) fits_clean ;; stage2) fits_stage2 ;; early) fits_early ;; esac ;;
esac
case $PART in
  all|bank581)
    case $C in
      clean|early) run677 5wbank bank; run677 5wobank bank ;;
      stage2) run677 5wbank bank ;;
    esac ;;
  bank660)
    # the second bank: experiment 660's generator, forecast from the same saved models
    steps=$([ "$C" = stage2 ] && echo 5w || echo "5w 5wo")
    for S in $steps; do
      SYN=1; [ "$S" = 5wo ] && SYN=0
      E00677_PHASE=predict "$D"/gpu699.sh e00699_bank660.py P0+D50 bank \
        --models-from "$E00677_ROOT/arms/k1_${S}_sp1/work/models/P5" --work-suffix "_${M}_${S}_sp1_b660" \
        --synthetic $SYN --months "$M" --seeds $SEEDS > "$E00677_ROOT/logs/${M}_${S}_sp1_bank660.log" 2>&1
      echo "$M $C $S bank660 done $(date -u +%T)"
    done ;;
esac
echo "$M $C $PART finished $(date -u +%T)"
