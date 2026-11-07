#!/bin/bash
# usage: chain.sh [k3|k1]...; the 25 (month, fit age) cutoffs for K3 then K1, one GPU process at a time.
# Per cutoff: train phase (pre-training, step 2 for K1, one fit with the power hidden from the
# cutoff), then predict phase (forecast the real month with the stored models, nothing refitted).
# A cutoff whose outputs exist is skipped. Logs are under the data folder.
D="$(cd "$(dirname "$0")" && pwd)"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
R=$DATA_PATH_INTERNAL/ml-experiments/00640-demand-fit-age-refit-k1-k3
mkdir -p "$R/logs"
SEEDS="0 1 2"
NETS="${*:-k3 k1}"
set -u
exec 9>"$R/chain.lock"; flock -n 9 || { echo "another chain.sh holds the lock" >&2; exit 4; }

have_all() { for f in "$@"; do [ -s "$f" ] || return 1; done; }  # every file exists and is non-empty

run() {  # run LOGNAME ARGS...: one GPU process, log under the data folder, stop the chain on failure
  local log=$R/logs/$1.log; shift
  "$D/gpu.sh" "$@" > "$log" 2>&1 || { echo "FAILED: $* (see $log)" >&2; exit 1; }
}

for net in $NETS; do
  case $net in
    k3) tag=D5;  tags="D5";;
    k1) tag=P5;  tags="P5 P5-removed";;
    *) echo "unknown network $net" >&2; exit 2;;
  esac
  export CHAIN640=$R/chains/$net/work
  for month in 2025-10 2025-12 2026-01 2026-03 2026-06; do
    for age in 0 4 8 16 26; do
      fold=$(printf '%s-%02d' "$month" "$age")
      models=(); fits=()
      for s in $SEEDS; do
        models+=("$R/arms/$net/work/models/$tag/${fold}_d0_s$s.pt")
        for t in $tags; do fits+=("$R/arms/$net/work/fits/$t/${fold}_d0_s$s.parquet"); done
      done
      if have_all "${fits[@]}"; then echo "skip $net $fold: forecasts exist"; continue; fi
      if ! have_all "${models[@]}"; then
        echo "train $net $fold"
        export E00640_PHASE=train
        if [ "$net" = k3 ]; then steps="pretrain fit"; else steps="pretrain step2 fit"; fi
        for step in $steps; do
          run "${net}_${fold}_train_$step" "e00640_$net.py" "$net" "$step" --folds "$fold" --seeds $SEEDS
        done
      fi
      echo "predict $net $fold"
      export E00640_PHASE=predict
      # the train phase left .empty markers (no forecast window); fit_arm would skip on them
      rm -f "$R"/arms/$net/work/fits/*/"${fold}"_d0_s?.empty
      run "${net}_${fold}_predict" "e00640_$net.py" "$net" fit --folds "$fold" --seeds $SEEDS
      have_all "${fits[@]}" || { echo "missing forecasts for $net $fold" >&2; exit 3; }
    done
  done
done
