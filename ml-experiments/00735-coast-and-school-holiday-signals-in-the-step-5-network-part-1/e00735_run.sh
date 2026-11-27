#!/bin/bash
# Experiment 735, driver: stage 2 ("step2"), stage 3 ("fit": train phase), and the prediction pass ("pfit")
# of one arm at one refit month, or the smoke test's validation-row forecast ("valpred"). Stage 1 is
# experiment 677's stored weights (`pretrained_P2`), read only.
#
# README. Runs `e00735_k1.py` (a copy of 731's runner; relies on experiments 677, 574, 581, 591, 510 and
# on `e00735_columns.py` and `e00735_model.py`). PYTHONPATH: this script sources
# `ml-experiments/00524*/env.sh`, which puts every experiment folder on it. Needs `uv sync --group gpu`
# and runs `.venv/bin/python` directly (never `uv run`, which would move the job out of its scope).
#
# usage: e00735_run.sh ARM MONTH PHASE   ARM: R0 S SP SKW SKWP SE SEP; MONTH: 2025-11 .. 2026-10;
#                                        PHASE: step2 | fit | pfit | valpred | all (step2, fit, pfit)
# env:   DATA_PATH_INTERNAL (required), SEEDS ("0 1 2"), CORES (default 22,23), CAP (default 12G),
#        R735 and R677 (data folders; R735 must contain 00735), SMOKE=1 (one epoch per stage, output
#        to .claude/scratch/e735_smoke/00735-smoke, never the experiment's data folder)
# Every stage is skipped when its output exists. Each Python command runs in a systemd scope with a memory
# cap and no swap, one BLAS thread, pinned to CORES. Phase times go to $R735/logs/phase_times.csv.
# Exit codes: 0 done, 2 bad usage or missing input, 20 a stage of the chain failed (the launcher that
# refuses to start a job while the GPU is busy has its own code, which is never 20).
set -uo pipefail
ARM=$1; M=$2; PHASE=$3
D="$(cd "$(dirname "$0")" && pwd)"
REPO=$(cd "$D/../.." && pwd)
cd "$REPO"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
R677=${R677:-$DATA_PATH_INTERNAL/ml-experiments/00677-building-up-the-candidate-demand-network-one-component-at-a-time}
if [ "${SMOKE:-0}" = 1 ]; then
  R735=$REPO/.claude/scratch/e735_smoke/00735-smoke
  export E00735_SMOKE=1
else
  R735=${R735:-$DATA_PATH_INTERNAL/ml-experiments/00735-coast-and-school-holiday-signals-in-the-step-5-network-part-1}
  export E00735_SMOKE=0
fi
[[ "$R735" == *00735* ]] || { echo "R735 must be experiment 735's data folder: $R735"; exit 2; }
[[ "$R735" != *00677* ]] || { echo "never write under experiment 677's folder"; exit 2; }
case $ARM in R0|S|SP|SKW|SKWP|SE|SEP) ;; *) echo "bad arm $ARM"; exit 2 ;; esac
SEEDS=${SEEDS:-0 1 2}
CORES=${CORES:-22,23}
CAP=${CAP:-12G}
source "$(ls -d ml-experiments/00524*)/env.sh"   # PYTHONPATH, OMP_NUM_THREADS=1
mkdir -p "$R735/logs"
export E00677_ROOT=$R735 E00677_SPAN=1 E00677_MONTH=$M E00735_ARM=$ARM
export CHAIN677=$R677/chains/5/$M/work STATS677=$R677/chains/3b/$M/work   # 677's frozen stage 1, read only
for s in $SEEDS; do
  [ -f "$CHAIN677/pretrained_P2/${M}_s$s.pt" ] || { echo "missing stage-1 weights $CHAIN677/pretrained_P2/${M}_s$s.pt"; exit 2; }
done
[ -f "$STATS677/stats/B1/$M.npz" ] || { echo "missing statistics $STATS677"; exit 2; }
A="--months $M --seeds $SEEDS"
ARMDIR=$R735/arms/k1_${ARM}_sp1/work
TAG=${M}_${ARM}
[ "${SMOKE:-0}" = 1 ] && TAG=smoke_${M}_${ARM}

py() {  # one Python process: memory cap, no swap, one BLAS thread, pinned cores
  systemd-run --user --scope -q -p MemoryMax="$CAP" -p MemorySwapMax=0 -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" \
    DATA_PATH_INTERNAL="$DATA_PATH_INTERNAL" NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x \
    E00677_ROOT="$R735" E00677_SPAN=1 E00677_MONTH="$M" E00735_ARM="$ARM" E00735_SMOKE="$E00735_SMOKE" \
    CHAIN677="$CHAIN677" STATS677="$STATS677" E00677_PHASE="$E00677_PHASE" \
    taskset -c "$CORES" "$REPO/.venv/bin/python" "$@"
}

phase_time() {  # phase_time PHASE DEVICE START END: one CSV row per run phase
  local f=$R735/logs/phase_times.csv
  ( flock 9
    [ -s "$f" ] || echo "arm,month,phase,device,start_utc,end_utc" >&9
    echo "$ARM,$M,$1,$2,$3,$4" >&9 ) 9>>"$f"
}

CODE_HASH=$(cat "$D/e00735_k1.py" "$D/e00735_columns.py" "$D/e00735_model.py" | sha256sum | cut -d' ' -f1)
check_code_hash() {  # a changed code state must not reuse earlier outputs (the smoke fits, for example)
  local marker=$ARMDIR/code_hash_$M.txt
  if [ -f "$marker" ] && [ "$(tr -d '[:space:]' < "$marker")" != "$CODE_HASH" ]; then
    echo "the experiment 735 code changed since $ARM $M was fitted: delete $ARMDIR outputs for $M first"; return 1
  fi
  return 0
}
have_step2() { for s in $SEEDS; do [ -f "$ARMDIR/step2/P5/${M}_s$s.pt" ] || return 1; done; }
have_models() { for s in $SEEDS; do [ -f "$ARMDIR/models/P5/${M}_d0_s$s.pt" ] || return 1; done; }
have_fits() { for s in $SEEDS; do [ -f "$ARMDIR/fits/P5/${M}_d0_s$s.parquet" ] || return 1; done; }
have_valpred() { for s in $SEEDS; do [ -f "$ARMDIR/valpred/${M}_d0_s$s.parquet" ] || return 1; done; }

run_stage() {  # run_stage NAME PHASE-ENV COMMAND ARGS...: time one Python stage; the log keeps the traceback
  local name=$1 phase=$2; shift 2
  local t0; t0=$(date -u +%FT%TZ)
  E00677_PHASE=$phase py "$D/e00735_k1.py" "k1_$ARM" "$@" --step 5 --synthetic 1 $A > "$R735/logs/${TAG}_$name.log" 2>&1
  local code=$?
  phase_time "$name" cuda "$t0" "$(date -u +%FT%TZ)"
  if [ $code -ne 0 ]; then echo "$M $ARM $name FAILED with exit code $code (see $R735/logs/${TAG}_$name.log)"; return 20; fi
  echo "$M $ARM $name done $(date -u +%T)"
}

run_step2() {
  check_code_hash || return 20
  if have_step2; then echo "$M $ARM step2 exists"; return 0; fi
  run_stage step2 train step2
}

run_fit() {
  check_code_hash || return 20
  have_step2 || { echo "missing step-2 weights for $ARM $M: run step2 first"; return 20; }
  if have_models && [ -f "$R735/logs/${TAG}_fit.log" ] && grep -q "CHECKS PASSED" "$R735/logs/${TAG}_fit.log"; then echo "$M $ARM fit exists"; return 0; fi
  run_stage fit train fit
}

run_pfit() {
  check_code_hash || return 20
  have_models || { echo "missing .pt files for $ARM $M: $SEEDS"; return 20; }
  grep -q "CHECKS PASSED" "$R735/logs/${TAG}_fit.log" || { echo "train log of $ARM $M lacks CHECKS PASSED"; return 20; }
  if have_fits; then echo "$M $ARM pfit exists"; return 0; fi
  rm -f "$ARMDIR"/fits/P5/${M}_d0_s?.empty  # the train phase's markers; fit_arm would skip on them
  run_stage pfit predict fit
}

run_valpred() {
  check_code_hash || return 20
  have_models || { echo "missing .pt files for $ARM $M: $SEEDS"; return 20; }
  if have_valpred; then echo "$M $ARM valpred exists"; return 0; fi
  run_stage valpred train valpred
}

case $PHASE in
  step2) run_step2 ;;
  fit) run_fit ;;
  pfit) run_pfit ;;
  valpred) run_valpred ;;
  all) run_step2 && run_fit && run_pfit ;;
  *) echo "bad phase $PHASE"; exit 2 ;;
esac
