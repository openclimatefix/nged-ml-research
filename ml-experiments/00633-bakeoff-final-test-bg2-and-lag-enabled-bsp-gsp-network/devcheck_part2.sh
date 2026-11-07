#!/bin/bash
# Dev-check: forecast dev month 2026-06 with experiment 622's stored weights on the extended inputs; compare with 622's stored forecast.
D="$(cd "$(dirname "$0")" && pwd)"
E=$DATA_PATH_INTERNAL/ml-experiments
S=$(ls -d $E/00622-*)/work
R=$(ls -d $E/00633-*)/work
export E00633_PHASE=predict
for g in bsp gsp; do
  mkdir -p $R/$g/models/J_LG_B1 $R/$g/stats/B1
  cp -n $S/$g/models/J_LG_B1/2026-06_d*_s?.pt $R/$g/models/J_LG_B1/
  cp -n $S/$g/stats/B1/2026-06*.npz $R/$g/stats/B1/ 2>/dev/null || true
  $D/gpu.sh e00633_node_fit.py fit --group $g --months 2026-06 --seeds 0 1 || exit 1
done
echo DEVCHECK RAN
