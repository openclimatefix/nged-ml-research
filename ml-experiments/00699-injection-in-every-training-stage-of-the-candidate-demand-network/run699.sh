#!/bin/bash
# usage: run699.sh PHASE
#   PHASE is stored, a0, a1, a2, b, bank660, or bank660. Each phase runs its fits through chain699.sh,
#   then exports each arm with experiment 677's e00677_export.py, cuts the exported files at the holdout with
#   e00699_filter.py, and only then calls experiment 677's e00677_score.py once per seed and once for all seeds.
#   stored   cut copies of experiment 677's stored forecast and bank files, scored (the known-answer rows)
#   a0       clean chain, fold 2026-01, seeds 0 to 2, stage-3-only arm: the reproduction check
#            (run `e00699_contrasts.py a0` by hand afterwards)
#   a1       clean chain, the four phase-A folds, seeds 0 to 7
#   a2       early chain and stage-2 arm, the four phase-A folds
#   b        all three chains, the four other folds
#   bank660  the second bank for every new arm, every fold that the chains have fitted
# Folds of one chain run one at a time (677 keeps one checks_fit.json per arm folder). Chains run side by side
# (14 GB of host memory each); the stage2 chain starts after the clean chain of the same month.
D="$(cd "$(dirname "$0")" && pwd)"
R=$(cd "$D/../.." && pwd)
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL}"
BASE=$DATA_PATH_INTERNAL/ml-experiments/00699-injection-in-every-training-stage-of-the-candidate-demand-network
D677=$(echo "$D"/../00677-*/)
cd "$R"
source "$R"/ml-experiments/00524*/env.sh
export PYTHONPATH="$D:$PYTHONPATH"
PY=$R/.venv/bin/python
PHASE=$1
PARALLEL=1
A_FOLDS="2025-11 2026-01 2026-04 2026-06"
B_FOLDS="2025-12 2026-02 2026-03 2026-05"
ALL_SEEDS="0 1 2 3 4 5 6 7"
set -e

score_file() {  # ROOTNAME FILESTEM SEEDS: score one exported file for each seed and for all seeds
  local root=$1 stem=$2; shift 2
  for s in "$@"; do echo "$s"; done | xargs -P 6 -I{} env E00677_ROOT="$BASE/roots/$root" \
    "$PY" "$D677"/e00677_score.py "$stem" {}
  E00677_ROOT="$BASE/roots/$root" "$PY" "$D677"/e00677_score.py "$stem" "$@"
}

export_score() {  # ROOTNAME STEP "MONTHS" "SEEDS"
  local root=$1 step=$2 months=$3 seeds=$4 version stem
  case $step in 5w) version=with_injected ;; 5wo) version=without_injected ;; esac
  stem=step${step}_${version}_sp1
  E00677_ROOT="$BASE/roots/$root" "$PY" "$D677"/e00677_export.py "$step" 1 $months --seeds $seeds
  "$PY" "$D"/e00699_filter.py new "$root"
  score_file "$root" "$stem" $seeds
}

run_folds() {  # CHAIN "MONTHS" [PART]: fit the folds, PARALLEL at a time
  local chain=$1 months=$2 part=${3:-fits}
  for m in $months; do echo "$m"; done | xargs -P "$PARALLEL" -I{} "$D"/chain699.sh "$chain" {} "$part"
}

run_banks() {  # CHAIN "MONTHS": the bank of experiment 677 for the folds, then the cut at the holdout
  run_folds "$1" "$2" bank581
  "$PY" "$D"/e00699_filter.py new "$1"
}

case $PHASE in
  stored)
    "$PY" "$D"/e00699_filter.py stored
    score_file stored step5w_with_injected_sp1 0 1 2
    score_file stored step5wo_without_injected_sp1 0 1 2 ;;
  a0)
    SEEDS="0 1 2" "$D"/chain699.sh clean 2026-01 a0
    export_score clean 5w 2026-01 "0 1 2" ;;
  a1)
    run_folds clean "$A_FOLDS" fits
    export_score clean 5w "$A_FOLDS" "$ALL_SEEDS"
    export_score clean 5wo "$A_FOLDS" "$ALL_SEEDS"
    run_banks clean "$A_FOLDS" ;;
  a2)
    run_folds early "$A_FOLDS" fits &
    run_folds stage2 "$A_FOLDS" fits
    wait
    export_score early 5w "$A_FOLDS" "$ALL_SEEDS"
    export_score early 5wo "$A_FOLDS" "$ALL_SEEDS"
    export_score stage2 5w "$A_FOLDS" "$ALL_SEEDS"
    run_banks early "$A_FOLDS" &
    run_banks stage2 "$A_FOLDS"
    wait ;;
  b)
    run_folds early "$B_FOLDS" fits &
    run_folds clean "$B_FOLDS" fits
    wait
    run_folds stage2 "$B_FOLDS" fits
    ALL="2025-11 2025-12 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06"
    export_score clean 5w "$ALL" "$ALL_SEEDS"
    export_score clean 5wo "$ALL" "$ALL_SEEDS"
    export_score early 5w "$ALL" "$ALL_SEEDS"
    export_score early 5wo "$ALL" "$ALL_SEEDS"
    export_score stage2 5w "$ALL" "$ALL_SEEDS"
    run_banks early "$B_FOLDS" &
    run_banks clean "$B_FOLDS"
    wait
    run_banks stage2 "$B_FOLDS" ;;
  bank660)
    FOLDS_DONE=${BANK660_FOLDS:-"$A_FOLDS $B_FOLDS"}
    for chain in clean early stage2; do
      run_folds "$chain" "$FOLDS_DONE" bank660
      "$PY" "$D"/e00699_filter.py new "$chain"
    done ;;
  *) echo "bad phase $PHASE"; exit 1 ;;
esac
echo "phase $PHASE finished $(date -u +%T)"
