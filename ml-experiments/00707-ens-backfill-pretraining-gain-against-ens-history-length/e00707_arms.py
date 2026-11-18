"""The arms of experiment 707: stages, ENS windows, flags, seeds, and the folds they run on.

This module is the one table that the fit, placebo, gate, and score scripts read. It holds data
only, so importing it reads no file. The plan is in
`docs/ml-experiments/00707-ens-backfill-pretraining-gain-against-ens-history-length/result.md`.

**An arm is a chain of up to three training stages on the pooled mixture density network of
experiment 425.**

- `S_era`: pre-training on ERA5 and CAMS pseudo-runs from 2019-09-01, unweighted (experiment 418's
  recipe, which keeps lockdown rows with the `covid` input set).
- `S_ens(W)`: an unweighted stage on the ENS rows in window `W`, with `S_era`'s optimiser.
- `FT(W)`: experiment 425's fine-tune on the ENS rows of window `W`, weighted by a 0.5-year
  half-life.

**A window is a start rule on the ENS rows, applied to valid times.** `days` counts back from the
fold's refit time. `floor` is the earliest valid time that a row may have. With `floor_lags` the
floor also applies to the row's origin and to the time of each of its three lags, so that no
pre-floor power reading (for instance a lockdown reading) reaches a row.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR_NAME: Final[str] = "00707-ens-backfill-pretraining-gain-against-ens-history-length"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_DIR_NAME
"""This experiment's private data folder; every other experiment's folder is only read."""
STORED_425_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00425-best-mdn-architecture-assembly-with-ablations" / "work"
)
"""Experiment 425's work folder, read only: its fitted states and per-seed raw forecasts."""

STORED_ENS_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
"""The first run of the ENS rows that experiment 425 trained on (the table's old start)."""
LONG_WINDOW_FLOOR: Final[datetime] = datetime(2021, 7, 20, tzinfo=UTC)
"""The day after the lockdown indicator ends: window F starts here."""
LOCKDOWN_WINDOW_FLOOR: Final[datetime] = datetime(2020, 10, 15, tzinfo=UTC)
"""The first run of the ENS back-fill: window C starts here and keeps the lockdown ENS rows."""

ALL_MONTHS: Final[tuple[str, ...]] = (
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
"""The 12 monthly folds, 2025-07 to 2026-06, each scored on its own month."""
SCREEN_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2026-01", "2026-03", "2026-06")
"""The four screen folds; every number from them is interim."""
GATE_MONTHS: Final[tuple[str, ...]] = ("2026-06", "2026-01")
"""The two folds of the refit gate."""
SCREEN_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
FIVE_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
THREE_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)


@dataclass(frozen=True)
class Window:
    """A start rule on the ENS rows of one stage.

    Attributes:
        name: Short label used in file names and logs.
        days: Keep rows whose valid time is later than the refit minus this many days; `None` for no
            trailing limit.
        floor: Keep rows whose valid time is at or after this instant; `None` for no floor.
        floor_lags: Whether the floor also applies to the row's origin and to each lag's time.
        note: One line saying what the window is.
    """

    name: str
    days: int | None
    floor: datetime | None = None
    floor_lags: bool = False
    note: str = ""


W24: Final[Window] = Window(
    name="w730",
    days=730,
    note="24 months of valid time, back-filled runs included; the filter is on valid time only",
)
W24_STORED: Final[Window] = Window(
    name="w730s",
    days=730,
    floor=STORED_ENS_START,
    note="24 months, but no run before 2024-04-01: the rows that experiment 425 trained on",
)
W12: Final[Window] = Window(name="w365", days=365, note="12 months of valid time")
W_F: Final[Window] = Window(
    name="F",
    days=None,
    floor=LONG_WINDOW_FLOOR,
    floor_lags=True,
    note="2021-07-20 to the cut: 47 to 58 months, no lockdown ENS rows",
)
W_C: Final[Window] = Window(
    name="C",
    days=None,
    floor=LOCKDOWN_WINDOW_FLOOR,
    floor_lags=True,
    note="2020-10-15 to the cut: 56 to 68 months, lockdown ENS rows kept with the covid input",
)

FLAG_DATES: Final[tuple[datetime, datetime]] = (
    datetime(2023, 6, 27, tzinfo=UTC),
    datetime(2024, 4, 1, tzinfo=UTC),
)
"""The resolution upgrade and the source change; `R_F+flags` carries one indicator for each.

An indicator is 1 for a row whose ENS run was initialised on or after the date."""


@dataclass(frozen=True)
class Arm:
    """One arm of the experiment.

    Attributes:
        name: The arm's name on the plan page.
        era: Whether the chain starts with the `S_era` stage.
        ens: The `S_ens` stage's window, or `None` if the arm has no such stage.
        ft: The `FT` stage's window; it also sets the fold's standardiser and target statistics.
        placebo: Whether `S_era` uses the weather-day permutation of `e00707_placebo`.
        flags: Whether two resolution and source indicators are appended to the inputs.
        seeds: The seeds of the full run.
        in_full_run: Whether the full run fits the arm; the `A_24` and `R_24` refits are fitted for
            the refit gate only, and the scorer reads experiment 425's stored arms for them.
        stored_folder: The folder of experiment 425's `seed_preds` that holds the stored arm, or
            `None` for an arm with no stored forecasts.
    """

    name: str
    era: bool
    ens: Window | None
    ft: Window
    placebo: bool = False
    flags: bool = False
    seeds: tuple[int, ...] = FIVE_SEEDS
    in_full_run: bool = True
    stored_folder: str | None = None

    @property
    def folder(self) -> str:
        """Return the arm's name as a folder name (`R_F+flags` becomes `R_F_flags`)."""
        return self.name.replace("+", "_")

    @property
    def stages(self) -> str:
        """Return the stages as the plan page writes them, such as `S_era→S_ens→FT`."""
        parts = []
        if self.era:
            parts.append("S_era (shuffled weather)" if self.placebo else "S_era")
        if self.ens is not None:
            parts.append("S_ens")
        parts.append("FT")
        return "→".join(parts)

    @property
    def pretrained(self) -> bool:
        """Return whether the fine-tune follows an earlier stage (so it uses the lower rate)."""
        return self.era or self.ens is not None

    def stage_key(self, *, stage: str) -> str:
        """Return the file-name key of the network state after a stage.

        The key names every choice that changes the stage's weights, so that two arms with the same
        key share one stored state: the stages so far, the `S_ens` window, the fine-tune window
        (which sets the standardiser), and the extra inputs.

        Args:
            stage: `era` for the state after `S_era`, or `ens` for the state after `S_ens`.

        Returns:
            The key, such as `era_ftw730` or `era+ensF_ftw730`.

        Raises:
            ValueError: If `stage` is not `era` or `ens`, or the arm has no such stage.
        """
        if stage not in ("era", "ens"):
            msg = f"unknown stage {stage!r}"
            raise ValueError(msg)
        if (stage == "era" and not self.era) or (stage == "ens" and self.ens is None):
            msg = f"arm {self.name} has no {stage} stage"
            raise ValueError(msg)
        pieces = []
        if self.era:
            pieces.append("eraP" if self.placebo else "era")
        if stage == "ens" and self.ens is not None:
            pieces.append(f"ens{self.ens.name}")
        flag = "_flags" if self.flags else ""
        return "+".join(pieces) + f"_ft{self.ft.name}{flag}"


ARMS: Final[dict[str, Arm]] = {
    a.name: a
    for a in (
        Arm("A_24", era=True, ens=None, ft=W24_STORED, in_full_run=False, stored_folder="A"),
        Arm("R_24", era=False, ens=None, ft=W24_STORED, in_full_run=False, stored_folder="R"),
        Arm("A_E24", era=True, ens=W24, ft=W24),
        Arm("R_E24", era=False, ens=W24, ft=W24),
        Arm("A_F", era=True, ens=W_F, ft=W24),
        Arm("R_F", era=False, ens=W_F, ft=W24),
        Arm("A_12", era=True, ens=None, ft=W12, seeds=THREE_SEEDS),
        Arm("R_12", era=False, ens=None, ft=W12, seeds=THREE_SEEDS),
        Arm("A_C", era=True, ens=W_C, ft=W24, seeds=THREE_SEEDS),
        Arm("R_C", era=False, ens=W_C, ft=W24, seeds=THREE_SEEDS),
        Arm("A1x_24", era=True, ens=None, ft=W24_STORED, placebo=True, seeds=THREE_SEEDS),
        Arm("R_F+flags", era=False, ens=W_F, ft=W24, flags=True, seeds=THREE_SEEDS),
    )
}
"""Every arm, keyed by name. Seeds and windows follow the plan's arm table.

`A_24` and `R_24` are the refits of experiment 425's `A` and `R` (window `W24_STORED` reproduces
its rows); they are fitted on the two gate folds for the refit gate, and the scorer uses the
stored `A` and `R` as `A_24` and `R_24`. `A1x_24` shares `A_24`'s window so that contrast C5
differs from it by the weather shuffle alone.
"""
LONG_WINDOW_ARMS: Final[tuple[str, ...]] = tuple(
    name for name, arm in ARMS.items() if arm.ens is not None and arm.ens.floor_lags
)
"""The arms whose `S_ens` window reaches into the back-filled runs before 2024-04-01."""


def seeds_for(*, arm: Arm, screen: bool) -> tuple[int, ...]:
    """Return the seeds to fit for an arm: its own, or at most the first three on the screen."""
    return tuple(s for s in arm.seeds if s in SCREEN_SEEDS) if screen else arm.seeds


def print_table() -> None:
    """Print every arm with its stages, windows, flags, seeds, and the folds."""
    print(f"{'arm':10s} {'stages':26s} {'S_ens window':14s} {'FT window':8s} seeds  flags/placebo")
    for arm in ARMS.values():
        ens = "-" if arm.ens is None else arm.ens.name
        extras = ", ".join(x for x, on in (("flags", arm.flags), ("placebo", arm.placebo)) if on)
        full = "" if arm.in_full_run else " (gate refit only)"
        print(
            f"{arm.name:10s} {arm.stages:26s} {ens:14s} {arm.ft.name:8s} "
            f"{list(arm.seeds)} {extras}{full}"
        )
    print(f"folds: {len(ALL_MONTHS)} months {ALL_MONTHS[0]} to {ALL_MONTHS[-1]}")
    print(f"screen months: {list(SCREEN_MONTHS)}, seeds {list(SCREEN_SEEDS)}")
    print(f"gate months: {list(GATE_MONTHS)}")


if __name__ == "__main__":
    print_table()
