# ruff: noqa: E501
"""Experiment 585: tail metrics by phase on the lag-only injected bank, for this experiment's arms.

Runs experiment 581's `e00581_tails` with its work folder pointed at this experiment's folder, so the
same event tables (follow ratio, safe-side share, width, pinball loss) are built for N1, N3, Y1,
Y1c, and Y3. The control (v3+D5) and N5 (X3-med) tables are experiment 581's own, in
`00581-robust-multiweek-lag-encoder-x3/tails/`.
"""

import e00581_tails as tails  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

tails.WORK = ML_EXPERIMENTS_DATA_DIR / "00585-lag-count-and-year-ago-lags-through-the-lag-encoder"

if __name__ == "__main__":
    tails.main()
