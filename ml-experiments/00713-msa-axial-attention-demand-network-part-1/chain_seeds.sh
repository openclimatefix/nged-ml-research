#!/usr/bin/env bash
# usage: chain_seeds.sh MONTH [STEP...]: the Candidate chain of experiment 667 for seeds 2, 3, and 4.
#
# SOURCE: this file is a copy of `00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/chain.sh`
# and its `gpu.sh`, because a merged script is never edited and `chain.sh` hard-codes `--seeds 0 1`
# (line 12). Lines changed from the source:
#   1. `A="--months $M --seeds 0 1"` became `A="--months $M --seeds $SEEDS"` with SEEDS=2 3 4
#      (SEEDS=0 and REPRO=1 reruns seed 0 for the reproduction check below).
#   2. `E00667_ROOT` defaults to `$DATA_PATH_INTERNAL/ml-experiments/00713-msa-axial-attention-demand-network-part-1/candidate_seeds`
#      (the source defaults to experiment 667's own folder), so nothing is written under `00667-*/`.
#   3. The scope's cap is `MemoryMax=12G` with `MemorySwapMax=0` (the source's `gpu.sh` caps at 8G), the
#      venv's Python is called directly, and the scripts are called from experiment 667's folder.
#   4. The unchanged-folder assertion and the seed 0 reproduction were added.
#
# Steps (default all, in order): pretrain, step2 (K1's chain), table, fit (Candidate, train phase: held-out
# power hidden), then ptable, pfit (Candidate, predict phase: the month's window opened, forecasts written).
#
# Unchanged-folder assertion: the modification times of experiment 667's `chains/` and `P7_D50_*` folders
# (every file and folder, two levels deep) are listed before and after the chain, and the script fails
# if they differ. Seed 0 reproduction: `REPRO=1 SEEDS=0 chain_seeds.sh MONTH` reruns seed 0 into the new
# root and prints the maximum absolute difference of the forecast against the stored seed 0 (a difference
# above 1e-3 of capacity fires audit trigger 1 of the design).
#
# Run `free -g` first and wait when under 25 GB is available. At most 3 GPU processes at once.
set -euo pipefail
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
D="$(cd "$(dirname "$0")" && pwd)"
EXPS="$(dirname "$D")"
WT="$(dirname "$EXPS")"
D667="$(echo "$EXPS"/00667-*/)"
D667="${D667%/}"
NAME667="$(basename "$D667")"
NAME713="$(basename "$D")"
STORED="$DATA_PATH_INTERNAL/ml-experiments/$NAME667"
R=${E00667_ROOT:-$DATA_PATH_INTERNAL/ml-experiments/$NAME713/candidate_seeds}
export E00667_ROOT=$R
case "$R" in "$STORED"*) echo "E00667_ROOT must not be experiment 667's folder" >&2; exit 1 ;; esac
SEEDS=${SEEDS:-2 3 4}
M=$1; shift
STEPS=${*:-pretrain step2 table fit ptable pfit}
export E00667_MONTH=$M CHAIN667=$R/chains/$M/work
mkdir -p "$R/logs" "$CHAIN667"

snapshot() {  # every path, type, size, and mtime below chains/, P7_D50_*, tables/, and exports/, at any depth, plus hashes of the stored fits and exports
  (cd "$STORED" && find chains P7_D50_* tables exports -printf '%p %y %s %T@\n' 2>/dev/null | sort
    { find P7_D50_*/work/fits/P5 -name '*_s[01].parquet' -print0 2>/dev/null
      find exports -name 'NRA_*.parquet' -print0 2>/dev/null; } | sort -z | xargs -0 -r sha256sum)
}
frames_snapshot() { (cd "$STORED" && find inputs/frames -printf '%p %s %T@\n' 2>/dev/null | sort); }
FRAMES_BEFORE="$(frames_snapshot)"
BEFORE="$(snapshot)"

PYTHONPATH_ALL="$D"
for d in "$EXPS"/00*/; do PYTHONPATH_ALL="$PYTHONPATH_ALL:${d%/}"; done
gpu() {  # usage: gpu SCRIPT [args]; one GPU process, 12 GB host cap, no swap, 1 thread
  systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 \
    PYTHONPATH="$PYTHONPATH_ALL" DATA_PATH_INTERNAL="$DATA_PATH_INTERNAL" \
    NGED_S3_BUCKET_URL=https://example.com NGED_S3_BUCKET_ACCESS_KEY=dummy NGED_S3_BUCKET_SECRET=dummy \
    "$WT/.venv/bin/python" "$D667/$@"
}

A="--months $M --seeds $SEEDS"
for s in $STEPS; do
  case $s in
    pretrain|step2) E00667_PHASE=train gpu e00667_k1.py k1 $s $A > "$R/logs/${M}_$s.log" 2>&1 ;;
    table|fit) E00667_PHASE=train gpu e00667_cand.py P7+D50 $s --work-suffix _$M $A > "$R/logs/${M}_c$s.log" 2>&1 ;;
    ptable) E00667_PHASE=predict gpu e00667_cand.py P7+D50 table --work-suffix _$M $A > "$R/logs/${M}_c$s.log" 2>&1 ;;
    pfit) for sd in $SEEDS; do rm -f "$R/P7_D50_$M/work/fits/P5/${M}_d0_s$sd.empty"; done  # the train phase's markers; fit_arm would skip on them
      E00667_PHASE=predict gpu e00667_cand.py P7+D50 fit --work-suffix _$M $A > "$R/logs/${M}_c$s.log" 2>&1 ;;
  esac
  echo "$M $s done $(date -u +%T)"
done

AFTER="$(snapshot)"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "experiment 667's chains/, P7_D50_*, tables/, or exports/ changed during the chain" >&2
  diff <(echo "$BEFORE") <(echo "$AFTER") >&2 || true
  exit 1
fi
[ "$FRAMES_BEFORE" = "$(frames_snapshot)" ] || echo "warning: inputs/frames of experiment 667 changed during the chain (stub frames; any process may write them)" >&2
echo "unchanged-folder assertion passed for $STORED"

if [ "${REPRO:-0}" = 1 ]; then
  "$WT/.venv/bin/python" - "$M" "$R" "$STORED" <<'PY'
import sys

import numpy as np
import polars as pl

month, new_root, stored_root = sys.argv[1:4]
tail = f"P7_D50_{month}/work/fits/P5/{month}_d0_s0.parquet"
cols = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
a = pl.read_parquet(f"{new_root}/{tail}").sort(keys)
b = pl.read_parquet(f"{stored_root}/{tail}").sort(keys)
assert a.select(keys).equals(b.select(keys)), "the reproduced seed 0 holds other keys"
worst = float(np.abs(a.select(cols).to_numpy() - b.select(cols).to_numpy()).max())
print(f"seed 0 reproduction, {month}: maximum absolute difference {worst:.2e} of capacity")
if worst > 1e-3:
    print("AUDIT TRIGGER 1: the copied wrapper does not reproduce the stored recipe")
    sys.exit(2)
PY
fi
