"""The weather-day permutation of the placebo arm `A1x_24` of experiment 707.

**The placebo gives `S_era` real weather on the wrong days.** Each Primary's ERA5-and-CAMS rows keep
their power, calendar columns, and lags, and the seven weather means of each row come from another
day of the same Primary: a block of 48 consecutive half-hours (a UTC day, with its 48 period-ending
stamps from 00:00 inclusive). If the placebo gains as much from pre-training as `A_24` does, the
gain does not come from the weather link, so contrast C5 (`A_24` minus `A1x_24`) must lie wholly
below zero for the experiment to pass.

**The donor of a day follows three rules from the plan.**

- It is drawn only from days whose 48 stamps are all at least 3.5 hours before the fold's cut, so
  the placebo reads no weather that the fold's real rows may not read.
- Its day of year is within 15 days of the target day's, wrapping at the end of the year.
- It lies in a different calendar year ("across years"), so no day keeps its own weather.

A day draws its donor uniformly from the eligible days, with replacement; the map is a random
draw, not a bijection of days. Every stamp of a target day reads the donor day's stamp at the same
time of day. The donor days of a Primary, fold, and seed are fixed by `numpy`'s generator seeded
with the three, and are stored under `work/placebo/<month>_s<seed>/<site>.npz` so that they can be
audited. The standard deviations `std_*` of `S_era` rows are not permuted: they come from the fold's
spread table for every arm.

```bash
# environment as in e00707_fit.py
uv run --no-sync python ml-experiments/00707-*/e00707_placebo.py check --months 2026-01 --seeds 0
uv run --no-sync python ml-experiments/00707-*/e00707_placebo.py write --months 2025-10 2026-01
```
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00707_arms as arms  # ty: ignore[unresolved-import]
import e00707_common as common  # ty: ignore[unresolved-import]
import e00707_prep as prep  # ty: ignore[unresolved-import]
import numpy as np

PLACEBO_DIR: Final[Path] = arms.WORK_DIR / "work" / "placebo"
BLOCK_STEPS: Final[int] = common.STEPS_PER_DAY
"""Half-hours in a weather day."""
DAY_OF_YEAR_WINDOW: Final[int] = 15
YEAR_DAYS: Final[int] = 365
STEP_US: Final[int] = 1_800_000_000
UNMAPPED: Final[int] = -1


@dataclass(frozen=True)
class DonorPool:
    """The weather days of one Primary that may serve as donors at one fold.

    Attributes:
        eligible: Per day block, whether all 48 stamps are before the cut and all weather values
            are finite.
        day_of_year: Per day block, the day of the year of its first stamp (1 to 366).
        year: Per day block, the calendar year of its first stamp.
    """

    eligible: np.ndarray
    day_of_year: np.ndarray
    year: np.ndarray


def block_calendar() -> tuple[np.ndarray, np.ndarray]:
    """Return the day of year and the year of every day block of the dense grid."""
    n_blocks = common.grid_times().len() // BLOCK_STEPS
    start = np.datetime64(common.GRID_START.replace(tzinfo=None), "D")
    days = start + np.arange(n_blocks)
    years = days.astype("datetime64[Y]")
    day_of_year = (days - years.astype("datetime64[D]")).astype(int) + 1
    return day_of_year, years.astype(int) + 1970


def donor_pool(*, dense: np.ndarray, refit: datetime) -> DonorPool:
    """Return the days of a Primary that may donate weather at a fold.

    Args:
        dense: The Primary's `prep.dense_weather`, shape (grid, 7).
        refit: The fold's cut.

    Returns:
        The pool: a day is eligible if every stamp is at or before `refit` minus 3.5 hours and
        every weather value of the day is finite.
    """
    day_of_year, year = block_calendar()
    n_blocks = len(day_of_year)
    last_stamp = int(common.grid_index(times=refit - common.REFIT_CUTOFF_AGE))
    blocks = dense[: n_blocks * BLOCK_STEPS].reshape(n_blocks, BLOCK_STEPS, dense.shape[1])
    finite = np.isfinite(blocks).all(axis=(1, 2))
    last_in_block = (np.arange(n_blocks) + 1) * BLOCK_STEPS - 1
    return DonorPool(
        eligible=finite & (last_in_block <= last_stamp), day_of_year=day_of_year, year=year
    )


def circular_gap(*, a: int, b: np.ndarray) -> np.ndarray:
    """Return the gap in days between a day of year and an array of them, wrapping the year."""
    gap = np.abs(b - a)
    return np.minimum(gap, YEAR_DAYS - gap)


def donor_blocks(*, pool: DonorPool, targets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a donor day block for each target day block.

    Args:
        pool: The eligible donors.
        targets: The target day blocks (indices into the grid's day blocks).
        rng: The generator that draws each donor.

    Returns:
        An array shaped like `targets`: the donor of each target block.

    Raises:
        ValueError: If a target day has no eligible donor in another year within 15 days of the
            day of year.
    """
    donors = np.empty(len(targets), dtype=np.int64)
    for i, block in enumerate(targets):
        near = (
            circular_gap(a=int(pool.day_of_year[block]), b=pool.day_of_year) <= DAY_OF_YEAR_WINDOW
        )
        candidates = np.nonzero(pool.eligible & near & (pool.year != pool.year[block]))[0]
        if len(candidates) == 0:
            msg = f"day block {block} has no donor in another year within 15 days of its day"
            raise ValueError(msg)
        donors[i] = candidates[rng.integers(len(candidates))]
    return donors


def donor_map(*, site: str, refit: datetime, seed: int, dense: np.ndarray) -> np.ndarray:
    """Return a Primary's donor block for every day block before the cut, `UNMAPPED` elsewhere.

    The map is read from `PLACEBO_DIR` if a file for the fold and seed exists, and otherwise
    drawn and written atomically.
    """
    month = f"{refit:%Y-%m}"
    path = PLACEBO_DIR / f"{month}_s{seed}" / f"{site}.npz"
    if path.exists():
        return np.load(path)["donor"]
    pool = donor_pool(dense=dense, refit=refit)
    n_blocks = len(pool.eligible)
    last_stamp = int(common.grid_index(times=refit - common.REFIT_CUTOFF_AGE))
    targets = np.arange(min(n_blocks, last_stamp // BLOCK_STEPS + 1))
    rng = np.random.default_rng(
        [seed, int(site.removeprefix("D")), int(refit.timestamp()) // 86_400]
    )
    donor = np.full(n_blocks, UNMAPPED, dtype=np.int64)
    donor[targets] = donor_blocks(pool=pool, targets=targets, rng=rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, donor=donor)
    temporary.rename(path)
    return donor


def permute_weather(
    *, x: np.ndarray, valid: np.ndarray, donor: np.ndarray, dense: np.ndarray
) -> np.ndarray:
    """Return raw rows with each row's seven weather means taken from its donor day.

    Args:
        x: Raw (unstandardised) `S_era` rows, shape (rows, 46).
        valid: The rows' valid times, microseconds since the epoch.
        donor: `donor_map`.
        dense: The Primary's `prep.dense_weather`.

    Returns:
        A copy of `x` whose `mean_<w>` columns hold the donor day's weather at the same time of
        day, with precipitation through `log1p` as in the feature matrix.

    Raises:
        ValueError: If a row's day has no donor, or a donor value is not finite.
    """
    names, _, _ = prep.feature_names()
    mean_cols = [names.index(f"mean_{w}") for w in common.WEATHER]
    stamp = (valid - int(common.GRID_START.timestamp() * 1_000_000)) // STEP_US
    block, within = stamp // BLOCK_STEPS, stamp % BLOCK_STEPS
    chosen = donor[block]
    if (chosen == UNMAPPED).any():
        msg = "a row's day has no donor: the row is not before the fold's cut"
        raise ValueError(msg)
    weather = dense[chosen * BLOCK_STEPS + within].astype(np.float64)
    if not np.isfinite(weather).all():
        msg = "a donor weather value is not finite"
        raise ValueError(msg)
    precipitation = common.WEATHER.index("precipitation")
    weather[:, precipitation] = np.log1p(np.clip(weather[:, precipitation], 0, None))
    out = x.copy()
    out[:, mean_cols] = weather.astype(np.float32)
    return out


class Permuter:
    """A function from raw `S_era` rows to rows with permuted weather, for one seed.

    It has the signature that `e00707_folds.fold_pretrain` expects of `permute`, and it loads each
    Primary's dense ERA5-and-CAMS weather once.
    """

    def __init__(self, *, seed: int) -> None:
        """Remember the seed; the Primaries' weather loads on first use."""
        self.seed = seed
        self._dense: dict[str, np.ndarray] = {}

    def dense(self, *, site: str) -> np.ndarray:
        """Return a Primary's dense weather, loaded once."""
        if site not in self._dense:
            row = next(r for r in prep.primaries().iter_rows(named=True) if r["site"] == site)
            self._dense[site] = prep.dense_weather(series_id=row["time_series_id"], site=site)
        return self._dense[site]

    def __call__(
        self, *, site: str, x: np.ndarray, valid: np.ndarray, refit: datetime
    ) -> np.ndarray:
        """Return `x` with the seven weather means of each row replaced by a donor day's."""
        dense = self.dense(site=site)
        donor = donor_map(site=site, refit=refit, seed=self.seed, dense=dense)
        return permute_weather(x=x, valid=valid, donor=donor, dense=dense)


def check(*, month: str, seed: int, sites: list[str]) -> None:
    """Draw the donors of some Primaries and assert the three rules; print one example."""
    refit = prep.refit_of(month=month)
    permuter = Permuter(seed=seed)
    day_of_year, year = block_calendar()
    last_stamp = int(common.grid_index(times=refit - common.REFIT_CUTOFF_AGE))
    for site in sites:
        dense = permuter.dense(site=site)
        donor = donor_map(site=site, refit=refit, seed=seed, dense=dense)
        mapped = np.nonzero(donor != UNMAPPED)[0]
        chosen = donor[mapped]
        assert (((chosen + 1) * BLOCK_STEPS - 1) <= last_stamp).all(), (
            "a donor day is after the cut"
        )
        assert (year[chosen] != year[mapped]).all(), "a donor day is in the target day's year"
        gap = np.minimum(
            np.abs(day_of_year[chosen] - day_of_year[mapped]),
            YEAR_DAYS - np.abs(day_of_year[chosen] - day_of_year[mapped]),
        )
        assert (gap <= DAY_OF_YEAR_WINDOW).all(), "a donor day is more than 15 days away"
        example = int(mapped[len(mapped) // 2])
        day = common.GRID_START + timedelta(days=example)
        donor_day = common.GRID_START + timedelta(days=int(donor[example]))
        print(
            f"{site} {month} seed {seed}: {len(mapped)} days mapped; all donors before the cut, in "
            f"another year, within 15 days of the day of year; e.g. {day:%Y-%m-%d} takes the "
            f"weather of {donor_day:%Y-%m-%d}"
        )


def write(*, months: list[str], seeds: list[int]) -> None:
    """Write the donor map of every Primary for the folds and seeds, skipping existing files."""
    permuter = Permuter(seed=0)
    for month in months:
        refit = prep.refit_of(month=month)
        for seed in seeds:
            for row in prep.primaries().iter_rows(named=True):
                dense = permuter.dense(site=row["site"])
                donor_map(site=row["site"], refit=refit, seed=seed, dense=dense)
            print(f"{month} seed {seed}: donor maps in {PLACEBO_DIR / f'{month}_s{seed}'}")


def main() -> None:
    """Check the rules on one fold, or write donor maps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "write"))
    parser.add_argument("--months", nargs="+", default=list(arms.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(arms.SCREEN_SEEDS))
    parser.add_argument("--sites", nargs="+", default=["D5", "D2"])
    args = parser.parse_args()
    if args.command == "check":
        for month in args.months:
            for seed in args.seeds:
                check(month=month, seed=seed, sites=args.sites)
    else:
        write(months=args.months, seeds=args.seeds)


if __name__ == "__main__":
    main()
