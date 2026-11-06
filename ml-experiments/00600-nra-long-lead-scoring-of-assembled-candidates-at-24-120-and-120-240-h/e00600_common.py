# ruff: noqa: E501, D107
"""Experiment 600: shared set-up for scoring the assembled candidates at 0-24, 24-120, and 120-240 h.

No fit. Experiment 597's `Inputs` builds the row table, the lags-masked twin, and the G4 gate state
from stored forecasts. Those forecasts reach 240 h, and experiment 597 cut them at 120 h only
through `e00582_arms.MAX_LEAD_H`, so this module raises that constant to 240 before the table is
built and then labels every row with one of three lead bands. The G4 gate reads only leads below
`FRESH_LEAD_H` and 24 to 48 h, so its state must equal experiment 592's stored state exactly; the
check is `Inputs.check_gate_state`.
"""

from typing import Final

import e00582_arms as a  # ty: ignore[unresolved-import]
import e00597_common as k597  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00600-nra-long-lead-scoring-of-assembled-candidates-at-24-120-and-120-240-h"
WORK: Final = ML_EXPERIMENTS_DATA_DIR / NAME
a.MAX_LEAD_H = 240
BANDS: Final[dict[int, str]] = {0: "0-24 h", 1: "24-120 h", 2: "120-240 h"}
GROUPS_SCORED: Final = k597.GROUPS_SCORED
GROUP_NAME: Final = k597.GROUP_NAME
PHASES: Final = k597.PHASES
N_COUNTED: Final[int] = 8
"""Counted tests: (1) the day-4-to-end coverage of C1 + G4 and of A1 + G4 at 120-240 h against 0.90 (2); (2) the gate's value at 120-240 h against 24-120 h, for C1 and for A1 (2); (3) the extra pinball loss at q05 and q95 of C1 + G4 over C1 and of A1 + G4 over A1 on event-free rows at 120-240 h (4)."""


def band_of(*, lead: np.ndarray) -> np.ndarray:
    """Return 0 for leads below 24 h, 1 for 24 to below 120 h, 2 for 120 h and above."""
    return np.where(lead < 24, 0, np.where(lead < 120, 1, 2))


class Inputs(k597.Inputs):
    """Experiment 597's inputs with leads to 240 h and three lead bands."""

    def __init__(self, *, arm: str) -> None:
        super().__init__(arm=arm)
        self.t.band = band_of(lead=self.t.lead)


def lag_free_quantiles(*, t) -> np.ndarray:  # noqa: ANN001
    """Return, per table row, experiment 565's lag-free D5 network (E2) quantiles, NaN where absent."""
    e = a._load(root=a.K565_W, tag="D5", months=k597.c.MONTHS)
    sm = k597.c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    e = e.with_columns(
        prim=pl.col("time_series_id").replace_strict(prim_of, return_dtype=pl.Int64),
        idx=(
            (pl.col("valid_time").dt.epoch("us") - k597.c.k578.prep.GRID_START_US)
            // k597.c.k578.prep.STEP_US
        ).cast(pl.Int64),
        oidx=(
            (pl.col("power_fcst_init_time").dt.epoch("us") - k597.c.k578.prep.GRID_START_US)
            // k597.c.k578.prep.STEP_US
        ).cast(pl.Int64),
    )
    kk = k597.k584.s.key(e["prim"].to_numpy(), e["oidx"].to_numpy(), e["idx"].to_numpy())
    order = np.argsort(kk)
    kk_sorted = kk[order]
    mine = k597.k584.s.key(t.prim, t.origin_idx, t.idx)
    pos = np.clip(np.searchsorted(kk_sorted, mine), 0, len(kk_sorted) - 1)
    ok = kk_sorted[pos] == mine
    q = np.sort(e.select(a.QCOLS).to_numpy().astype(np.float32), axis=1)[order[pos]]
    return np.where(ok[:, None], q, np.nan).astype(np.float32)
