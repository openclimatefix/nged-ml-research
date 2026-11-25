#!/bin/bash
# Experiment 731, driver: stage 3 ("fit": train phase, "pfit": predict phase) or the held-out bank ("bank")
# of one arm at one refit month. Stages 1 and 2 are experiment 677's stored weights, read only.
#
# README. Runs the fit, predict, and bank steps of experiment 731 through `e00731_k1.py` (a copy of
# 677's `e00677_k1.py`; relies on 677, 574, 581, 591, 710) and `e00677_cand.py bank` (677). PYTHONPATH:
# this script sources `ml-experiments/00524*/env.sh`, which puts every experiment folder on it.
#
# usage: e00731_run.sh ARM MONTH PHASE     ARM: R0 W0 W30 W270 WP IP B0; MONTH: 2025-11 .. 2026-10;
#                                          PHASE: fit | pfit | bank | all (fit then pfit)
# env:   DATA_PATH_INTERNAL (required), SEEDS ("0 1 2"), CORES (default 22-29), CAP (default 10G),
#        R731 and R677 (data folders; R731 must contain 00731)
# Every fit and forecast is skipped when its output exists. Each Python command runs in a systemd scope with
# a memory cap and no swap, one BLAS thread, pinned to CORES. Phase times go to $R731/logs/phase_times.csv.
set -euo pipefail
ARM=$1; M=$2; PHASE=$3
D="$(cd "$(dirname "$0")" && pwd)"
REPO=$(cd "$D/../.." && pwd)
cd "$REPO"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
R731=${R731:-$DATA_PATH_INTERNAL/ml-experiments/00731-step-5-recency-and-injection-ablations-monthly-refits}
R677=${R677:-$DATA_PATH_INTERNAL/ml-experiments/00677-building-up-the-candidate-demand-network-one-component-at-a-time}
[[ "$R731" == *00731* ]] || { echo "R731 must be experiment 731's data folder: $R731"; exit 1; }
[[ "$R731" != *00677* ]] || { echo "never write under experiment 677's folder"; exit 1; }
case $ARM in R0|W0|W30|W270|WP|IP|B0) ;; *) echo "bad arm $ARM"; exit 1 ;; esac
SEEDS=${SEEDS:-0 1 2}
CORES=${CORES:-22-29}
SYN=1; [ "$ARM" = B0 ] && SYN=0
source "$(ls -d ml-experiments/00524*)/env.sh"   # PYTHONPATH, OMP_NUM_THREADS=1
mkdir -p "$R731/logs"
export E00677_ROOT=$R731 E00677_SPAN=1 E00677_MONTH=$M E00731_ARM=$ARM
export CHAIN677=$R677/chains/5/$M/work STATS677=$R677/chains/3b/$M/work   # 677's frozen stage 1 and 2, read only
[ -d "$CHAIN677/step2/P5" ] || { echo "missing stage-2 weights $CHAIN677"; exit 1; }
[ -f "$STATS677/stats/B1/$M.npz" ] || { echo "missing statistics $STATS677"; exit 1; }
A="--months $M --seeds $SEEDS"
ARMDIR=$R731/arms/k1_${ARM}_sp1/work
TRAINLOG=$R731/logs/${M}_${ARM}_fit.log

py() {  # one Python process: memory cap, no swap, one BLAS thread, pinned cores
  systemd-run --user --scope -q -p MemoryMax=${CAP:-10G} -p MemorySwapMax=0 -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" \
    DATA_PATH_INTERNAL="$DATA_PATH_INTERNAL" NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x \
    E00677_ROOT="$R731" E00677_SPAN=1 E00677_MONTH="$M" E00731_ARM="$ARM" CHAIN677="$CHAIN677" STATS677="$STATS677" \
    E00677_PHASE="$E00677_PHASE" \
    taskset -c "$CORES" "$REPO/.venv/bin/python" "$@"
}

phase_time() {  # phase_time PHASE DEVICE START END: one CSV row per run phase
  local f=$R731/logs/phase_times.csv
  ( flock 9
    [ -s "$f" ] || echo "arm,month,phase,device,start_utc,end_utc" >&9
    echo "$ARM,$M,$1,$2,$3,$4" >&9 ) 9>>"$f"
}

have_models() { for s in $SEEDS; do [ -f "$ARMDIR/models/P5/${M}_d0_s$s.pt" ] || return 1; done; }
have_fits() { for s in $SEEDS; do [ -f "$ARMDIR/fits/P5/${M}_d0_s$s.parquet" ] || return 1; done; }
have_bank() { for s in $SEEDS; do [ -f "$R731/P0_D50_${M}_${ARM}_sp1/work/bank/${M}_d0_s$s.parquet" ] || return 1; done; }

run_fit() {
  if have_models && [ -f "$TRAINLOG" ] && grep -q "CHECKS PASSED" "$TRAINLOG"; then echo "$M $ARM fit exists"; return; fi
  local t0=$(date -u +%FT%TZ)
  E00677_PHASE=train py "$D/e00731_k1.py" "k1_$ARM" fit --step 5 --synthetic $SYN $A > "$TRAINLOG" 2>&1
  phase_time fit cuda "$t0" "$(date -u +%FT%TZ)"
  echo "$M $ARM fit done $(date -u +%T)"
}

check_before_predict() {  # a failed fit must never be refitted with the scored month open
  have_models || { echo "missing .pt files for $ARM $M: $SEEDS"; exit 1; }
  grep -q "CHECKS PASSED" "$TRAINLOG" || { echo "train log of $ARM $M lacks CHECKS PASSED"; exit 1; }
  for s in $SEEDS; do [ -f "$ARMDIR/history/${M}_d0_s$s.json" ] || echo "note: no loss history for seed $s (fit reused?)"; done
}

run_pfit() {
  check_before_predict
  if have_fits; then echo "$M $ARM pfit exists"; return; fi
  rm -f "$ARMDIR"/fits/P5/${M}_d0_s?.empty  # the train phase's markers; fit_arm would skip on them
  local t0=$(date -u +%FT%TZ)
  E00677_PHASE=predict py "$D/e00731_k1.py" "k1_$ARM" fit --step 5 --synthetic $SYN $A > "$R731/logs/${M}_${ARM}_pfit.log" 2>&1
  phase_time pfit cuda "$t0" "$(date -u +%FT%TZ)"
  echo "$M $ARM pfit done $(date -u +%T)"
}

run_bank() {  # the held-out injected bank on the K1-type step-5 models, through 677's Candidate code with no signal columns
  check_before_predict
  if have_bank; then echo "$M $ARM bank exists"; return; fi
  local t0=$(date -u +%FT%TZ) CAND
  CAND=$(ls "$REPO"/ml-experiments/00677-*/e00677_cand.py)
  E00677_PHASE=predict py "$CAND" P0+D50 bank --models-from "$ARMDIR/models/P5" --work-suffix "_${M}_${ARM}_sp1" \
    --synthetic $SYN $A > "$R731/logs/${M}_${ARM}_sp1_bank.log" 2>&1
  phase_time bank cuda "$t0" "$(date -u +%FT%TZ)"
  echo "$M $ARM bank done $(date -u +%T)"
}

case $PHASE in
  fit) run_fit ;;
  pfit) run_pfit ;;
  bank) run_bank ;;
  all) run_fit; run_pfit ;;
  *) echo "bad phase $PHASE"; exit 1 ;;
esac
