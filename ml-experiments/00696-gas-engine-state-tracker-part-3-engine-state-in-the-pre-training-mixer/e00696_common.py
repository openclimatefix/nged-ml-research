# ruff: noqa: E501
"""Paths, constants, and the arm table shared by the scripts of experiment 696.

Experiment 696 reuses experiment 695's cleaned stacks, tracker states, engine injection, and scorer by
import, and reads experiment 695's stored fits for seeds 0 to 2 of arms A0 and W0. D8 is a substation
and may be named. The engine record is called "Gas A" everywhere that is printed or drawn, never with
dates.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from e00695_common import DATA_DIR as E695_DIR  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 696
SLUG: Final[str] = "gas-engine-state-tracker-part-3-engine-state-in-the-pre-training-mixer"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"{EXPERIMENT_ID:05d}-{SLUG}"
)
ROW_SET_ID: Final[str] = "backtest_year_2025_11_to_2026_06_daily_12utc"
"""The decision rows: origins 2025-11-01 to 2026-06-30 of the shared year's row set."""

N_STATE: Final[int] = 3
"""State columns appended to the pre-training rows: expected output share, probability any unit runs, hidden flag."""
SEEDS: Final[tuple[int, ...]] = tuple(range(10))
REUSED_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
"""Seeds of arms A0 and W0 that are read from experiment 695's stored fits."""
DESCRIPTIVE_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
"""Seeds of the descriptive arm S1d."""
FOLD_MONTHS: Final[tuple[str, ...]] = (
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
N_CONTRASTS: Final[int] = 4
FAMILY_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / N_CONTRASTS)
"""The coverage of each planned contrast: a Bonferroni split of 5% across four contrasts (98.75%)."""
FAMILY_RESAMPLES: Final[int] = 40_000
"""Bootstrap resamples; at 98.75% each tail holds 250 draws."""
DESCRIPTIVE_LEVEL: Final[float] = 95.0
THRESHOLD_POINTS: Final[float] = 0.16
"""The smallest gain worth porting at D8, in points of capacity (8 times demand-v2's 0.02)."""
PROBE_LEVEL: Final[float] = 95.0

StateKindType = Literal["none", "tracked", "placebo", "zero", "hidden", "true"]
"""Which state the pre-training mixer reads: none (the network is experiment 695's), the tracked state, a day-permuted tracked state, zeros, the tracked state hidden on half the origin days, or the true injected state."""


@dataclass(frozen=True)
class Arm:
    """One arm: whose rows, which state, and where its state file comes from."""

    name: str
    series: Literal["d8", "host"]
    kind: StateKindType
    source: str | None
    """The experiment 695 arm whose per-fold tracker states the state is built from (`A1` or `W1`)."""
    stack: str | None
    """The experiment 695 stack arm whose row replaces the power row of every stage (`W0` at the host)."""
    probe: bool = False
    """Whether the encoder probe M1 runs after each seed's pre-training."""


ARMS: Final[dict[str, Arm]] = {
    a.name: a
    for a in (
        Arm("A0", "d8", "none", None, None),
        Arm("G0", "d8", "none", None, None),
        Arm("S1", "d8", "tracked", "A1", None),
        Arm("SP", "d8", "placebo", "A1", None),
        Arm("SZ", "d8", "zero", None, None),
        Arm("S1d", "d8", "hidden", "A1", None),
        Arm("W0", "host", "none", None, "W0", probe=True),
        Arm("G0W", "host", "none", None, "W0", probe=True),
        Arm("WST", "host", "true", None, "W0", probe=True),
        Arm("WS1", "host", "tracked", "W1", "W0"),
        Arm("W0p", "host", "none", None, "W0", probe=True),
    )
}
"""Arm G0 is A0 refitted for seed 0 of the first fold only, in a folder of its own, to check the reproduction. Arm G0W does the same for W0, through the code path that produces W0 seeds 3 to 9. Arm W0p runs no fit: it loads experiment 695's stored pre-trained weights of W0 for seeds 0 to 2 and runs the encoder probe on them."""
STATE_FREE_ARMS: Final[frozenset[str]] = frozenset({"A0", "G0", "G0W", "W0"})


def runs_dir(*, arm: str) -> Path:
    """Return the folder that holds an arm's chains, logs, and fits."""
    return DATA_DIR / "runs" / arm


def fits_dir(*, arm: str, root: Path = DATA_DIR) -> Path:
    """Return the folder of an arm's prediction fits under `root` (this experiment's, or 695's)."""
    return root / "runs" / arm / "arms" / "k1_5w_sp1" / "work" / "fits" / "P5"


def fits_roots(*, arm: str) -> dict[int, Path]:
    """Return the folder each seed's fits live in: 695's for seeds 0 to 2 of A0 and W0, else this one's."""
    return {
        seed: fits_dir(
            arm=arm, root=E695_DIR if arm in {"A0", "W0"} and seed in REUSED_SEEDS else DATA_DIR
        )
        for seed in (DESCRIPTIVE_SEEDS if arm == "S1d" else SEEDS)
    }
