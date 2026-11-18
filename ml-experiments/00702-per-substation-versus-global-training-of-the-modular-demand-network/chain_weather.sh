#!/bin/bash
# usage: chain_weather.sh MONTH [PRIMARIES]
# The truncated-archive weather test (`e00702_fit.py ... weather`), run before the first scored fit of a fold. It runs in the
# one-Primary world (`split1`, one process per Primary) and in the pooled world (`split3`, one process for the list).
# PRIMARIES is a comma-separated list (default: all 16; the run time per Primary and origin is unmeasured, so a first run can pass
# "D15,D18"). Both worlds need E00677_PHASE=predict, because the test rows of the fold are the thing under test.
D="$(cd "$(dirname "$0")" && pwd)"
: "${DATA_PATH_INTERNAL:?set DATA_PATH_INTERNAL to the private data folder}"
R=$DATA_PATH_INTERNAL/ml-experiments/00702-per-substation-versus-global-training-of-the-modular-demand-network
export E00677_ROOT=$R E00677_SPAN=1 E00677_MONTH=$1 E00677_PHASE=predict
M=$1
ALL="D2,D3,D4,D5,D8,D9,D10,D11,D12,D14,D15,D16,D18,D19,D21,D22"
PRIM=${2:-$ALL}; [ "$PRIM" = all ] && PRIM=$ALL
set -e
mkdir -p "$R/logs"
if [ ! -f "$R/fold_spans/$M.json" ]; then
  "$D/gpu.sh" e00702_fit.py split3 spans --month "$M" > "$R/logs/spans_${M}.log" 2>&1
fi
for P in ${PRIM//,/ }; do
  "$D/gpu.sh" e00702_fit.py split1 weather --month "$M" --primaries "$P" > "$R/logs/weather_split1_${M}_${P}.log" 2>&1
  echo "$M split1 weather $P done $(date -u +%T)"
done
"$D/gpu.sh" e00702_fit.py split3 weather --month "$M" --primaries "$PRIM" > "$R/logs/weather_split3_${M}.log" 2>&1
echo "$M split3 weather done $(date -u +%T)"
