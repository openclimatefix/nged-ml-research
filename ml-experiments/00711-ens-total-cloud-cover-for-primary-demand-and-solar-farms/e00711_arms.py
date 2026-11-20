"""The arms, folds, seeds, and prediction-set names of experiment 711 (plan sections 5 and 8).

This module is the one table that the fit driver, the job-list generator, and the scorer read. It
holds data and small pure functions only, imports no `torch`, and reads no file.

**An arm is a choice of extra inputs on one of two base networks.** The Primary demand arms (names
starting `D`, `K`, or `R`) extend experiment 707's arm `R_F`. The solar farm arms (names starting
`P`) extend experiment 659's `PLP`. The extra inputs are a summary of the 51 ENS members' total
cloud cover (`t1`, `t2`, or `t3`) or, for the positive control, of 2 m temperature and dew point
(`temperature`), a placebo that replaces the summary with another run's, the `cloud_missing` flag,
and whether the rows of the 19-month gap are dropped.

The `Stored` arms hold no fit: `D0` is experiment 707's stored `R_F`, and `P0` is experiment 666's
stored PLP.
"""

from dataclasses import dataclass
from typing import Final, Literal

import e00707_arms as arms707  # ty: ignore[unresolved-import]

TargetType = Literal["primary", "solar"]
SummaryType = Literal["none", "t1", "t2", "t3", "temperature"]
PlaceboType = Literal["none", "shuffled", "shifted"]
FoldSetType = Literal["recent12", "cloud7", "gate2", "replicate12", "solar7"]

SUMMARY_COLUMNS: Final[dict[SummaryType, tuple[str, ...]]] = {
    "none": (),
    "t1": ("cloud_mean", "cloud_spread"),
    "t2": ("cloud_mean", "cloud_spread", "cloud_clear_share", "cloud_overcast_share"),
    "t3": ("cloud_mean", "cloud_spread", "cloud_daylight_mean"),
    "temperature": ("temp_mean", "temp_std", "dew_mean", "dew_std"),
}
"""The raw extra columns of each summary, in input order."""
FLAG_COLUMN: Final[str] = "cloud_missing"
TEMPERATURE_BASE_COLUMNS: Final[tuple[str, ...]] = (
    "mean_temperature_2m",
    "std_temperature_2m",
    "mean_dew_point_temperature_2m",
    "std_dew_point_temperature_2m",
)
"""The four base inputs that the positive control removes (they come back as `temperature`)."""

ALL_MONTHS: Final[tuple[str, ...]] = arms707.ALL_MONTHS
"""The 12 monthly folds 2025-07 to 2026-06 of experiment 707."""
GAP_SCORED_MONTHS: Final[tuple[str, ...]] = ALL_MONTHS[:4]
"""2025-07 to 2025-10: the scored runs carry no cloud, so the folds are a negative control."""
MIXED_MONTH: Final[str] = "2025-11"
"""Cloud from 2025-11-21 only: reported on its own and in no contrast."""
CLOUD_SCORED_MONTHS: Final[tuple[str, ...]] = ALL_MONTHS[5:]
"""2025-12 to 2026-06: every scored run carries cloud."""
GATE_MONTHS: Final[tuple[str, ...]] = arms707.GATE_MONTHS
REPLICATE_MONTHS: Final[tuple[str, ...]] = (
    "2023-04",
    "2023-05",
    "2023-06",
    "2023-07",
    "2023-08",
    "2023-09",
    "2023-10",
    "2023-11",
    "2023-12",
    "2024-01",
    "2024-02",
    "2024-03",
)
SOLAR_MONTHS: Final[tuple[str, ...]] = CLOUD_SCORED_MONTHS

FOLDS: Final[dict[FoldSetType, tuple[str, ...]]] = {
    "recent12": ALL_MONTHS,
    "cloud7": CLOUD_SCORED_MONTHS,
    "gate2": GATE_MONTHS,
    "replicate12": REPLICATE_MONTHS,
    "solar7": SOLAR_MONTHS,
}

PRIMARY_ROW_SET: Final[str] = "fold_a_daily_gap_v1"
REPLICATE_ROW_SET: Final[str] = "backfill_daily_2023_04_to_2024_03_v1"
SOLAR_ROW_SET: Final[str] = "solar_daily_1200_2025_12_to_2026_06_v1"
EXPERIMENT_ID: Final[int] = 711

SEEDS_3: Final[tuple[int, ...]] = (0, 1, 2)
SEEDS_5: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
SEEDS_6: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5)


@dataclass(frozen=True)
class Arm:
    """One arm of the experiment.

    Attributes:
        name: The arm's name on the plan page.
        target: `primary` (the network of experiment 707's `R_F`) or `solar` (PLP).
        summary: Which summary of the members' cloud (or, for `temperature`, of temperature) the
            arm adds.
        placebo: Whether the summary is replaced by another run's: `shuffled` (a donor in another
            year within 15 days of the day of year) or `shifted` (the run issued about 14 days
            earlier).
        flag: Whether the `cloud_missing` column is an input.
        drop_gap: Whether the training rows of the 19-month gap are dropped (the same-row pair).
        folds: The fold set the arm runs on.
        seeds: The fitting seeds.
        stored: Whether the arm is a stored forecast with no fit.
        note: One line saying what the arm is.
    """

    name: str
    target: TargetType
    summary: SummaryType
    placebo: PlaceboType
    flag: bool
    drop_gap: bool
    folds: FoldSetType
    seeds: tuple[int, ...]
    stored: bool = False
    note: str = ""

    @property
    def extra_columns(self) -> tuple[str, ...]:
        """Return the raw extra input columns in input order, the flag last."""
        flag = (FLAG_COLUMN,) if self.flag else ()
        return (*SUMMARY_COLUMNS[self.summary], *flag)

    @property
    def months(self) -> tuple[str, ...]:
        """Return the arm's folds."""
        return FOLDS[self.folds]

    @property
    def row_set_id(self) -> str:
        """Return the id of the row set the arm's predictions cover."""
        if self.target == "solar":
            return SOLAR_ROW_SET
        return REPLICATE_ROW_SET if self.folds == "replicate12" else PRIMARY_ROW_SET

    @property
    def removes_temperature(self) -> bool:
        """Return whether the arm takes the four temperature inputs out of the base inputs."""
        return self.summary == "temperature"

    @property
    def n_fits(self) -> int:
        """Return how many fits the arm needs (zero for a stored arm)."""
        return 0 if self.stored else len(self.months) * len(self.seeds)


def _arm(
    name: str,
    target: TargetType,
    summary: SummaryType,
    folds: FoldSetType,
    seeds: tuple[int, ...],
    *,
    placebo: PlaceboType = "none",
    flag: bool = True,
    drop_gap: bool = False,
    stored: bool = False,
    note: str = "",
) -> Arm:
    """Return an arm; the flag defaults to on, as in every masked arm."""
    return Arm(
        name=name,
        target=target,
        summary=summary,
        placebo=placebo,
        flag=flag,
        drop_gap=drop_gap,
        folds=folds,
        seeds=seeds,
        stored=stored,
        note=note,
    )


ARM_LIST: Final[tuple[Arm, ...]] = (
    # Primary demand, experiment 707's R_F as the base.
    _arm("D0", "primary", "none", "recent12", SEEDS_5, flag=False, stored=True, note="stored R_F"),
    _arm(
        "D0n", "primary", "none", "gate2", SEEDS_3, flag=False, note="gate: R_F refit by new code"
    ),
    _arm("DF", "primary", "none", "recent12", SEEDS_6, note="cloud_missing only"),
    _arm("D1", "primary", "t1", "recent12", SEEDS_6, note="T1 real, masked"),
    _arm("D1p", "primary", "t1", "recent12", SEEDS_6, placebo="shuffled", note="T1 shuffled run"),
    _arm("D1t", "primary", "t1", "recent12", SEEDS_3, placebo="shifted", note="T1 time-shifted"),
    _arm("D2", "primary", "t2", "recent12", SEEDS_6, note="T2 real, masked"),
    _arm("D2p", "primary", "t2", "recent12", SEEDS_6, placebo="shuffled", note="T2 shuffled run"),
    _arm("D3", "primary", "t3", "cloud7", SEEDS_3, note="T3 real, masked (exploratory)"),
    _arm(
        "D0g", "primary", "none", "cloud7", SEEDS_3, flag=False, drop_gap=True, note="no gap rows"
    ),
    _arm(
        "D1g", "primary", "t1", "cloud7", SEEDS_3, flag=False, drop_gap=True, note="T1, no gap rows"
    ),
    _arm("K1", "primary", "temperature", "cloud7", SEEDS_3, note="positive control"),
    _arm(
        "K1p",
        "primary",
        "temperature",
        "cloud7",
        SEEDS_3,
        placebo="shuffled",
        note="positive control placebo",
    ),
    _arm("R0", "primary", "none", "replicate12", SEEDS_3, flag=False, note="replicate reference"),
    _arm("R1", "primary", "t1", "replicate12", SEEDS_6, flag=False, note="replicate T1 real"),
    _arm(
        "R1p",
        "primary",
        "t1",
        "replicate12",
        SEEDS_6,
        placebo="shuffled",
        flag=False,
        note="replicate T1 placebo",
    ),
    # Solar farms, PLP as the base.
    _arm("P0", "solar", "none", "solar7", SEEDS_3, flag=False, stored=True, note="stored PLP"),
    _arm("P0L", "solar", "none", "solar7", SEEDS_3, flag=False, note="monthly refit, long window"),
    _arm("PF", "solar", "none", "solar7", SEEDS_5, note="cloud_missing only"),
    _arm("P1", "solar", "t1", "solar7", SEEDS_5, note="T1 real, masked"),
    _arm("P1p", "solar", "t1", "solar7", SEEDS_5, placebo="shuffled", note="T1 shuffled run"),
    _arm("P1t", "solar", "t1", "solar7", SEEDS_3, placebo="shifted", note="T1 time-shifted"),
    _arm("P2", "solar", "t2", "solar7", SEEDS_5, note="T2 real, masked"),
    _arm("P2p", "solar", "t2", "solar7", SEEDS_5, placebo="shuffled", note="T2 shuffled run"),
    _arm("P0g", "solar", "none", "solar7", SEEDS_3, flag=False, drop_gap=True, note="no gap rows"),
    _arm(
        "P1g", "solar", "t1", "solar7", SEEDS_3, flag=False, drop_gap=True, note="T1, no gap rows"
    ),
)
ARMS: Final[dict[str, Arm]] = {a.name: a for a in ARM_LIST}
"""Every arm, keyed by name. Seeds and folds follow plan section 5."""


def set_name(*, arm: str, month: str, seed: int) -> str:
    """Return the name of an (arm, fold, seed) prediction set, such as `d1p-2026-06-s0`.

    The name matches `experiment_store.SET_NAME_PATTERN` (lower-case words joined by hyphens).
    """
    return f"{arm.lower()}-{month}-s{seed}"


def fitted_arms(*, target: TargetType | None = None) -> list[Arm]:
    """Return the arms that need fits, optionally of one target."""
    return [a for a in ARM_LIST if not a.stored and (target is None or a.target == target)]


def check_table() -> None:
    """Raise if an arm breaks a rule that the plan states about the table.

    A placebo arm has the real arm's columns in the same order, and a stored arm has no extra input.

    Raises:
        ValueError: If a placebo arm's extra columns differ from its real arm's, or a name is not a
            valid prediction-set name.
    """
    twins = {"D1p": "D1", "D1t": "D1", "D2p": "D2", "K1p": "K1", "P1p": "P1", "P1t": "P1"}
    twins |= {"P2p": "P2", "R1p": "R1"}
    for placebo_name, real_name in twins.items():
        placebo, real = ARMS[placebo_name], ARMS[real_name]
        if placebo.extra_columns != real.extra_columns or placebo.target != real.target:
            msg = f"{placebo_name} does not have {real_name}'s columns in the same order"
            raise ValueError(msg)
    for arm in ARM_LIST:
        if arm.placebo != "none" and arm.summary == "none":
            msg = f"{arm.name} has a placebo of nothing"
            raise ValueError(msg)
        if arm.drop_gap and arm.flag:
            msg = f"{arm.name} drops the gap rows, so it has no use for the cloud_missing flag"
            raise ValueError(msg)


if __name__ == "__main__":
    check_table()
    for a in ARM_LIST:
        print(f"{a.name:4s} {a.target:8s} {a.extra_columns!s:90s} {a.folds:12s} {a.n_fits:4d} fits")
