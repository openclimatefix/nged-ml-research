# ruff: noqa: E501, D107
"""Experiment 597: shared set-up for assembling C1 + G4 and A1 + G4 from stored forecasts.

Nothing is refitted. The assembled forecast of a network is, on every row, that network's own
quantiles, except that on origin days where the G4 gate is on or held (on, and the 7 days after it
turns off) the q05-q95 band is the widest of the network's band and its lags-masked twin's band at
every level (experiment 584's union band; the median stays the network's median).

G4 (experiment 592) reads observed power and the lag-free network E2, and the neighbours' readings,
so its state does not depend on which network is being assembled. `Inputs` rebuilds it from
the stored files for each arm and `check_gate_state` compares it with experiment 592's stored state.
"""

from typing import Final

import e00582_arms as a  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00584_score as k584  # ty: ignore[unresolved-import]
import e00592_common as g  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00597-nra-assemble-c1-and-a1-with-g4-gate-as-single-forecasts"
WORK: Final = ML_EXPERIMENTS_DATA_DIR / NAME
EXPORT_DATE: Final[str] = "2026-11-06"
EXPORT: Final = ML_EXPERIMENTS_DATA_DIR / f"backtest_export_candidates_{EXPORT_DATE}"
C1_WORK: Final = ML_EXPERIMENTS_DATA_DIR / "00584-nra-candidate-pack-x6/c1/work"
A1_WORK: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00591-shared-lag-encoder-auxiliary-shift-depth-loss/A1/work"
)
V592: Final = ML_EXPERIMENTS_DATA_DIR / g.NAME
ARM_WORK: Final = {"c1": C1_WORK, "a1": A1_WORK}
ARM_LABEL: Final = {"c1": "C1", "a1": "A1"}
GROUPS_SCORED: Final = ("MVA-only eight", "signed-MW six")
GROUP_NAME: Final = {
    "MVA-only eight": "eight MVA Primaries",
    "signed-MW six": "six MW Primaries",
    "D15 and D18": "report-only (D15 and D18)",
}
PHASES: Final = ("before", "days0-3", "days4-end", "after0-7", "after7-14")
"""Phases by forecast valid time relative to the event: before (7 days to onset), days 0-3, day 4 to the end, 0-7 days after the end, 7-14 days after."""

for _arm, _work in ARM_WORK.items():
    k584.ARMS[_arm] = {"fits": _work / "fits", "fit_tag": "P5", "inject": _work / "inject"}


class Inputs:
    """The stored table of one arm, the real-data gap arrays, and the G4 gate state."""

    def __init__(self, *, arm: str) -> None:
        k584.setup(arm)
        self.arm = arm
        self.t = c.X4Table()
        gi = c.gate_inputs(t=self.t)
        self.gi = gi
        self.origin_days = gi["origin_days"]
        self.n_days = gi["gap"].shape[1]
        e2 = g.lag_free_median(t=self.t)
        self.fresh = g.FreshGap(t=self.t, e2=e2, n_days=self.n_days)
        self.fresh_gap = self.fresh.gap()
        self.fresh_thr = g.threshold(gap=self.fresh_gap, near=gi["near"])
        self.nb = self.t.neighbours()
        self.states = g.all_gates(
            old_gap=gi["gap"], old_thr=gi["thr"], fresh_gap=self.fresh_gap,
            fresh_thr=self.fresh_thr, origin_days=self.origin_days, neighbours=self.nb,
        )  # fmt: skip
        self.masks = g.arm_masks(states=self.states)
        self.mode = self.masks["G4"][0]
        """(16, origin days): True where the G4 band is wide (gate on, or within 7 days after it turned off)."""

    def check_gate_state(self) -> dict[str, bool]:
        """Compare the rebuilt G4 state with experiment 592's stored `gate_state.npz`."""
        z = np.load(V592 / "gate_state.npz")
        return {
            "fresh_gap": bool(np.allclose(z["fresh_gap"], self.fresh_gap, equal_nan=True)),
            "G4_on": bool((z["G4_on"] == self.states["G4"][0]).all()),
            "G4_snap": bool((z["G4_snap"] == self.states["G4"][1]).all()),
        }


def assemble(*, v: np.ndarray, m: np.ndarray, wide: np.ndarray) -> np.ndarray:
    """Return the assembled quantiles: the union band where `wide`, the network's own elsewhere."""
    return np.where(wide[:, None], a.union_band(lag=v, masked=m), v)
