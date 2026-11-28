#!/bin/bash
# Experiment 735 part 2, driver: stage 1 ("pretrain"), stage 2 ("step2"), stage 3 ("fit": train phase),
# and the prediction pass ("pfit") of one arm at one refit month, or the smoke test's validation-row
# forecast ("valpred"). Every arm trains its own stage 1 (the encoder has more inputs than experiment
# 677's stored one), under this experiment's own data folder; experiment 677's folder is read only for
# the stored B1 statistics.
#
# README. Runs `e00735b_k1.py` (relies on experiments 677, 574, 581, 591, 510, on part 1's
# `e00735_columns.py` and `e00735_diagnostic.py`, and on `e00735b_columns.py`, `e00735b_model.py`, and
# `e00735b_stage1.py`). PYTHONPATH: this script sources `ml-experiments/00524*/env.sh`, which puts every
# experiment folder on it. Needs `uv sync --group gpu` and runs `.venv/bin/python` directly (never `uv run`,
# which would move the job out of its scope).
#
# usage: e00735b_run.sh ARM MONTH PHASE   ARM: R1 SKE SKE-S21 SKE-CP; MONTH: 2025-11 .. 2026-06;
#                                         PHASE: pretrain | step2 | fit | pfit | valpred | all (pretrain, step2, fit, pfit)
# env:   DATA_PATH_INTERNAL (required), SEEDS ("0 1 2"), CORES (default 22,23), CAP (default 12G),
#        R735 and R677 (data folders; R735 must contain school-columns-in-the-date-time-encoder-part-2
#        or 00735-smoke, and neither coast-and-school nor 00677), SMOKE=1 (one epoch per stage, output
#        to .claude/scratch/e735b_smoke/00735-smoke, never the experiment's data folder)
# This launcher checks no review markers: only `e00735b_phases.py` does, so run phases through it.
# `STOP_HOURS` from the environment is read by `e00735b_phases.py` only, and is clamped to at most 6.5.
# Every stage is skipped when its output exists. Each Python command runs in a systemd scope with a memory
# cap and no swap, one BLAS thread, pinned to CORES. Phase times go to $R735/logs/phase_times.csv.
# EXIT CODES of this launcher and of `e00735b_phases.py` (the only place that lists them; the Python
# scripts point here):
#   0   done
#   2   bad usage, missing input, or a phase without its review marker (run.sh and phases.py)
#   20  a stage of the chain failed (run.sh and phases.py)
#   21  phases.py only: the GPU-hours stop line was reached
#   22  e00735b_score.py --gate (passed through by phases.py): the replay gate FAILED
#   75  the GPU is busy (run.sh, passed through by phases.py): nothing was started and nothing failed;
#       wait and run the same command again. The guard is the one of part 1 (`nvidia-smi` totals leave
#       room for 6 GB under 36 GB, at most 2 GPU processes across all agents, one process of experiment
#       735, a lock file shared with part 1).
#   1   is never used on purpose: it is a Python crash.
set -uo pipefail
ARM=$1; M=$2; PHASE=$3
D="$(cd "$(dirname "$0")" && pwd)"
REPO=$(cd "$D/../.." && pwd)
cd "$REPO"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
R677=${R677:-$DATA_PATH_INTERNAL/ml-experiments/00677-building-up-the-candidate-demand-network-one-component-at-a-time}
if [ "${SMOKE:-0}" = 1 ]; then
  R735=$REPO/.claude/scratch/e735b_smoke/00735-smoke
  export E00735B_SMOKE=1
else
  R735=${R735:-$DATA_PATH_INTERNAL/ml-experiments/00735-school-columns-in-the-date-time-encoder-part-2}
  export E00735B_SMOKE=0
fi
[[ "$R735" == *school-columns-in-the-date-time-encoder-part-2* || "$R735" == *00735-smoke* ]] || { echo "R735 must be experiment 735 part 2's data folder: $R735"; exit 2; }
[[ "$R735" != *coast-and-school* ]] || { echo "never write under part 1's folder"; exit 2; }
[[ "$R735" != *00677* ]] || { echo "never write under experiment 677's folder"; exit 2; }
case $ARM in R1|SKE|SKE-S21|SKE-CP) ;; *) echo "bad arm $ARM"; exit 2 ;; esac
SEEDS=${SEEDS:-0 1 2}
CORES=${CORES:-22,23}
CAP=${CAP:-12G}
source "$(ls -d ml-experiments/00524*)/env.sh"   # PYTHONPATH, OMP_NUM_THREADS=1
mkdir -p "$R735/logs"
export E00677_ROOT=$R735 E00677_SPAN=1 E00677_MONTH=$M E00735B_ARM=$ARM
export STATS677=$R677/chains/3b/$M/work   # experiment 677's stored B1 statistics, read only
[ -f "$STATS677/stats/B1/$M.npz" ] || { echo "missing statistics $STATS677"; exit 2; }
A="--months $M --seeds $SEEDS"
ARMDIR=$R735/arms/k1_${ARM}_sp1/work
TAG=${M}_${ARM}
[ "${SMOKE:-0}" = 1 ] && TAG=smoke_${M}_${ARM}

py() {  # one Python process: memory cap, no swap, one BLAS thread, pinned cores
  systemd-run --user --scope -q -p MemoryMax="$CAP" -p MemorySwapMax=0 -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" \
    DATA_PATH_INTERNAL="$DATA_PATH_INTERNAL" NGED_S3_BUCKET_URL=x NGED_S3_BUCKET_ACCESS_KEY=x NGED_S3_BUCKET_SECRET=x \
    E00677_ROOT="$R735" E00677_SPAN=1 E00677_MONTH="$M" E00735B_ARM="$ARM" E00735B_SMOKE="$E00735B_SMOKE" \
    STATS677="$STATS677" E00677_PHASE="$E00677_PHASE" \
    taskset -c "$CORES" "$REPO/.venv/bin/python" "$@"
}

phase_time() {  # phase_time PHASE DEVICE START END: one CSV row per run phase
  local f=$R735/logs/phase_times.csv
  ( flock 9
    [ -s "$f" ] || echo "arm,month,phase,device,start_utc,end_utc" >&9
    echo "$ARM,$M,$1,$2,$3,$4" >&9 ) 9>>"$f"
}

P1=$(ls -d "$REPO"/ml-experiments/00735-coast*)
CODE_HASH=$(cat "$D/e00735b_k1.py" "$D/e00735b_columns.py" "$D/e00735b_model.py" "$D/e00735b_stage1.py" "$P1/e00735_columns.py" "$P1/e00735_diagnostic.py" | sha256sum | cut -d' ' -f1)  # the same files in the same order as HASHED_FILES in e00735b_k1.py
check_code_hash() {  # a changed code state must not reuse earlier outputs (the smoke fits, for example)
  local marker=$ARMDIR/code_hash_$M.txt
  if [ -f "$marker" ] && [ "$(tr -d '[:space:]' < "$marker")" != "$CODE_HASH" ]; then
    echo "the experiment 735 part 2 code changed since $ARM $M was fitted: delete $ARMDIR outputs for $M first"; return 1
  fi
  return 0
}
have_pretrain() { for s in $SEEDS; do [ -f "$ARMDIR/pretrained_P2/${M}_s$s.pt" ] || return 1; done; }
have_step2() { for s in $SEEDS; do [ -f "$ARMDIR/step2/P5/${M}_s$s.pt" ] || return 1; done; }
have_models() { for s in $SEEDS; do [ -f "$ARMDIR/models/P5/${M}_d0_s$s.pt" ] || return 1; done; }
have_fits() { for s in $SEEDS; do [ -f "$ARMDIR/fits/P5/${M}_d0_s$s.parquet" ] || return 1; done; }
have_valpred() { for s in $SEEDS; do [ -f "$ARMDIR/valpred/${M}_d0_s$s.parquet" ] || return 1; done; }

EXIT_BUSY=75
gpu_guard() {  # refuse to start while the GPU is busy; the lock is held until this script exits
  command -v nvidia-smi >/dev/null || return 0
  local used_mb ours=0 others=0 pid
  used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
  if [ $((used_mb + 6144)) -gt 36864 ]; then
    echo "the GPU holds ${used_mb} MB; 6 GB more would pass the 36 GB total" >&2; return 1
  fi
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'e00735'; then ours=$((ours + 1)); else others=$((others + 1)); fi
  done
  if [ $((ours + others)) -ge 2 ]; then
    echo "GPU processes: ${ours} of experiment 735 and ${others} of others; at most 2 GPU processes run at once across all agents" >&2; return 1
  fi
  if [ "$ours" -ge 1 ]; then echo "a process of experiment 735 already holds the GPU" >&2; return 1; fi
  mkdir -p "$HOME/.cache"
  exec 8>"$HOME/.cache/e00735_gpu.lock"
  flock -n 8 || { echo "another worker of experiment 735 holds the lock" >&2; return 1; }
  return 0
}

run_stage() {  # run_stage NAME PHASE-ENV COMMAND ARGS...: time one Python stage; the log keeps the traceback
  local name=$1 phase=$2; shift 2
  gpu_guard || { echo "$M $ARM $name not started: GPU busy (exit $EXIT_BUSY)"; return $EXIT_BUSY; }
  local t0; t0=$(date -u +%FT%TZ)
  E00677_PHASE=$phase py "$D/e00735b_k1.py" "k1_$ARM" "$@" $A > "$R735/logs/${TAG}_$name.log" 2>&1
  local code=$?
  phase_time "$name" cuda "$t0" "$(date -u +%FT%TZ)"
  if [ $code -ne 0 ]; then echo "$M $ARM $name FAILED with exit code $code (see $R735/logs/${TAG}_$name.log)"; return 20; fi
  echo "$M $ARM $name done $(date -u +%T)"
}

run_pretrain() {
  check_code_hash || return 20
  if have_pretrain; then echo "$M $ARM pretrain exists"; return 0; fi
  run_stage pretrain train pretrain
}

run_step2() {
  check_code_hash || return 20
  have_pretrain || { echo "missing stage-1 weights for $ARM $M: run pretrain first"; return 20; }
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
  pretrain) run_pretrain ;;
  step2) run_step2 ;;
  fit) run_fit ;;
  pfit) run_pfit ;;
  valpred) run_valpred ;;
  all) run_pretrain && run_step2 && run_fit && run_pfit ;;
  *) echo "bad phase $PHASE"; exit 2 ;;
esac
