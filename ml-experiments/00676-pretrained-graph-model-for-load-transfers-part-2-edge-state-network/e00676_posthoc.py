"""Experiment 676, Part A (post hoc): experiment 678's hybrids with donors that may end sooner.

Experiment 678 searched an added donor's end from 2 days after its start, so the short cells equalled
E1-J. Here the earliest end is 6 hours after the start and the running-median window of the end
search is 6 hours. Everything else is experiment 678's, imported and patched; only the F and R
probes are run.

    export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
    P=ml-experiments
    export PYTHONPATH=$P/00676-pretrained-graph-model-for-load-transfers-part-2-edge-state-network:\
$P/00678-donor-posterior-hybrid-for-load-transfer-pairing:\
$P/00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe:\
$P/00670-event-table-recipient-search-on-six-cell-grid:\
$P/00660-shared-generator-of-injected-load-transfer-events:\
$P/00653-edge-state-event-table-penalised-segmentation:\
$P/00665-event-table-recall-by-length-and-depth
    OMP_NUM_THREADS=1 uv run --no-sync python e00676_posthoc.py <first_seed> <n_worlds>
"""

from __future__ import annotations

import e00678_run as base  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

base.OUT = (
    ML_EXPERIMENTS_DATA_DIR
    / "00676-pretrained-graph-model-for-load-transfers-part-2-edge-state-network"
    / "posthoc"
)
base.MIN_STAY_HH = 12
base.LEVEL_WINDOW_HH = 12
base.ARMS = (("F", 0), *(("R", k) for k in (0, 1, 2)))

if __name__ == "__main__":
    base.main()
