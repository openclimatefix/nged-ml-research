#!/bin/bash
# usage: e00709_chain.sh --era "ARM ..." --folds "MONTH ..." [--seeds "0 1"] [--max-jobs N] [--mem-cap 16G]
#                        [--bank] [--pilot]
# Experiment 709, phase B: stage 3 of the Candidate with an era signal, fold by fold. For each fold, seed,
# and arm the chain runs `e00709_cand.py fit` in the train phase (the scored month is hidden; the network
# is fitted and saved), then in the predict phase (the saved network forecasts the month and the forecasts
# are stored). With --bank it then runs `e00709_cand.py bank` (inference on the injected bank, 16 passes).
# With --pilot it runs ONE fold, ONE seed, ONE arm on a small slice, for timing, into a `pilot` folder.
# Experiment 667's chains, tables, and inputs are read only; everything is written under experiment 709's
# data folder. No fit starts unless checks.json holds synthetic_passed, signals_passed, and
# alignment_passed, and every arm's signal table exists.
# Each job runs under systemd-run with a memory cap (--mem-cap, default 16G, no swap) and one BLAS thread.
# Experiment folders are put on PYTHONPATH by ml-experiments/00524*/env.sh. Seeds 0 and 1 only (667's).
D="$(cd "$(dirname "$0")" && pwd)"
R=$(cd "$D/../.." && pwd)
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
NAME=00709-era-signal-in-the-candidate-part-1
OUT=$DATA_PATH_INTERNAL/ml-experiments/$NAME
ARMS=""; FOLDS=""; SEEDS="0 1"; MAXJOBS=1; MEMCAP=16G; BANK=0; PILOT=0
while [ $# -gt 0 ]; do
  case $1 in
    --era) ARMS=$2; shift 2 ;;
    --folds) FOLDS=$2; shift 2 ;;
    --seeds) SEEDS=$2; shift 2 ;;
    --max-jobs) MAXJOBS=$2; shift 2 ;;
    --mem-cap) MEMCAP=$2; shift 2 ;;
    --bank) BANK=1; shift ;;
    --pilot) PILOT=1; shift ;;
    *) echo "unknown argument $1"; exit 1 ;;
  esac
done
[ -n "$ARMS" ] && [ -n "$FOLDS" ] || { echo "give --era and --folds"; exit 1; }
for a in $ARMS; do
  case $a in none|aug|oracle|placebo|oracle-enc|all|rule|soft|hmm|stat) ;; *) echo "bad arm $a"; exit 1 ;; esac
done
for f in $FOLDS; do
  case $f in 2025-11|2025-12|2026-01|2026-02|2026-03|2026-04|2026-05|2026-06) ;; *) echo "bad fold $f"; exit 1 ;; esac
done
for s in $SEEDS; do case $s in 0|1) ;; *) echo "bad seed $s (experiment 667 has seeds 0 and 1)"; exit 1 ;; esac; done
if [ $PILOT = 1 ]; then
  ARMS=${ARMS%% *}; FOLDS=${FOLDS%% *}; SEEDS=${SEEDS%% *}; MAXJOBS=1
fi

# no fit before the checks have passed
if ! python3 -c "import json,sys; c=json.load(open('$OUT/checks.json')); sys.exit(0 if all(c.get(k) is True for k in ('synthetic_passed','signals_passed','alignment_passed')) else 1)" 2>/dev/null; then
  echo "synthetic_passed, signals_passed, and alignment_passed are not all true in $OUT/checks.json: run e00709_checks.py --synthetic, --signals, and --alignment FOLD first"; exit 1
fi
for a in $ARMS; do
  b=$a; [ $a = oracle-enc ] && b=oracle
  case $a in none|aug) continue ;; esac
  [ -f $OUT/signals/$b.parquet ] || { echo "missing $OUT/signals/$b.parquet: run e00709_signal.py first"; exit 1; }
done

# memory: every parallel job may use up to its cap, and the host must hold all of them with 8 GB spare
CAP_GB=${MEMCAP%G}
NEED=$(( MAXJOBS * CAP_GB + 8 ))
AVAIL=$(free -g | awk '/^Mem:/ {print $7}')
if [ "$AVAIL" -lt "$NEED" ]; then
  echo "free -g shows $AVAIL GB available, $NEED GB needed for $MAXJOBS job(s) of $MEMCAP plus 8 GB spare"; exit 1
fi

# the GPU must hold every parallel job: each may use up to GPU_NEED_MIB (the pilot's peak_gpu_gib sets the real figure)
GPU_NEED_MIB=${GPU_NEED_MIB:-6000}
if command -v nvidia-smi >/dev/null; then
  GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$GPU_FREE" -lt $(( MAXJOBS * GPU_NEED_MIB )) ]; then
    echo "nvidia-smi shows $GPU_FREE MiB free, $(( MAXJOBS * GPU_NEED_MIB )) MiB needed for $MAXJOBS job(s) of $GPU_NEED_MIB MiB"; exit 1
  fi
else
  echo "nvidia-smi not found: cannot check the free GPU memory"; exit 1
fi
cd "$R"  # the env file uses paths relative to the repository root
source $R/ml-experiments/00524*/env.sh
export PYTHONPATH=$R/src:$PYTHONPATH
mkdir -p $OUT/logs
PILOT_FLAG=""; [ $PILOT = 1 ] && PILOT_FLAG="--pilot"

run_job() {  # run_job LOGNAME ARGS...
  local log=$OUT/logs/$1.log; shift
  systemd-run --user --scope -q -p MemoryMax=$MEMCAP -p MemorySwapMax=0 -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" \
    DATA_PATH_INTERNAL=$DATA_PATH_INTERNAL NGED_S3_BUCKET_URL=https://example.com \
    NGED_S3_BUCKET_ACCESS_KEY=dummy NGED_S3_BUCKET_SECRET=dummy \
    $R/.venv/bin/python $D/e00709_cand.py "$@" > $log 2>&1
}

one() {  # one FOLD SEED ARM
  local f=$1 s=$2 a=$3 t=${3//-/_}_${1}_s$2
  [ $PILOT = 1 ] && t=${t}_pilot
  run_job ${t}_train fit --era $a --fold $f --seed $s --phase train $PILOT_FLAG || return 1
  run_job ${t}_predict fit --era $a --fold $f --seed $s --phase predict $PILOT_FLAG || return 1
  if [ $BANK = 1 ]; then
    run_job ${t}_bank bank --era $a --fold $f --seed $s --phase predict $PILOT_FLAG || return 1
  fi
  echo "$a $f s$s done $(date -u +%T)"
}

FAILED=0
for f in $FOLDS; do
  for s in $SEEDS; do
    for a in $ARMS; do
      while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n || FAILED=1; done
      one $f $s $a &
    done
  done
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || FAILED=1; done
[ $FAILED = 0 ] || { echo "at least one job failed: see $OUT/logs"; exit 1; }
