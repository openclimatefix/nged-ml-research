"""The placebo cloud series of experiment 711: the shuffled-run map and the time-shifted map.

**Both placebos replace the cloud columns of a run with those of a donor run at the same leads, and
copy the real missing pattern exactly.** A run in the 19-month gap, and run 2024-01-03, have no real
cloud, so they have no donor and stay missing. Every other run has a donor, and the code asserts
before any fit that the two missing patterns are equal.

- **Shuffled-run placebo (in the planned contrasts).** The donor of run `r` is drawn uniformly, with
  replacement, among the runs that carry cloud, lie in another calendar year, fall within 15 days
  of `r`'s day of the year (wrapping at the end of the year), and were published before the fold's
  cut (`donor + 9 h <= cut`). This is the donor rule of experiment 707's `e00707_placebo.py`
  applied to forecast runs. The donor need not precede `r`, because a donor published before the
  cut has no bearing on `r`'s origin. The generator is seeded by series, fold cut, and fitting
  seed, and the map is stored under `work/placebo/shuffled/<month>_s<seed>/<series>.npz`.
- **Time-shifted placebo (second control).** The donor of run `r` is the run issued 14 days
  earlier. Where that run carries no cloud, the run 21 days earlier is used, then the run 7 days
  earlier, then the run 744 days earlier. The donor is always issued before `r`, and the
  map does not depend on a seed or a cut. It is stored under `work/placebo/shifted/<series>.npz`.

**The two maps are independent of the targets' weather.** A map holds run dates only, as days since
the epoch, with `UNMAPPED` for a run that carries no cloud. The cloud summary tables, and the
solar tables that the later stages build, apply a map with `apply_donor_map`.

`check` and `write` both store the maps that they draw.

```bash
# environment as in e00711_common.py
uv run --no-sync python ml-experiments/00711-*/e00711_placebo.py check --months 2026-01 --seeds 0
uv run --no-sync python ml-experiments/00711-*/e00711_placebo.py write --months 2025-07 2026-06
```
"""

import argparse
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import e00711_common as common
import numpy as np
import polars as pl

UNMAPPED: Final[int] = -1
"""The donor of a run that carries no cloud."""
DAY_OF_YEAR_WINDOW: Final[int] = 15
YEAR_DAYS: Final[int] = 365
SHIFT_DAYS: Final[tuple[int, ...]] = (14, 21, 7)
"""The time-shifted donor offsets in the order that they are tried."""
YEAR_SHIFT_DAYS: Final[int] = 2 * 365 + 14
"""The last resort of the time-shifted rule: 744 days earlier (2025-11-21 maps to 2023-11-08)."""
EPOCH: Final[date] = date(1970, 1, 1)
SHUFFLED_DIR: Final[Path] = common.PLACEBO_DIR / "shuffled"
SHIFTED_DIR: Final[Path] = common.PLACEBO_DIR / "shifted"
TARGET_FIRST_RUN: Final[datetime] = common.ROW_FIRST_RUN
"""The first run that a row reads; earlier cached runs are donors only."""


def day_number(*, run: datetime) -> int:
    """Return a run's date as days since the epoch."""
    return (run.date() - EPOCH).days


def run_of(*, day: int) -> datetime:
    """Return the 00:00 UTC run of a day number."""
    return datetime.combine(EPOCH + timedelta(days=int(day)), datetime.min.time(), tzinfo=UTC)


def circular_gap(*, a: int, b: np.ndarray) -> np.ndarray:
    """Return the gap in days between a day of year and an array of them, wrapping the year."""
    gap = np.abs(b - a)
    return np.minimum(gap, YEAR_DAYS - gap)


def calendar_of(*, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the day of year (1 to 366) and the calendar year of each day number."""
    dates = np.datetime64("1970-01-01") + days.astype("timedelta64[D]")
    years = dates.astype("datetime64[Y]")
    day_of_year = (dates - years.astype("datetime64[D]")).astype(int) + 1
    return day_of_year, years.astype(int) + 1970


def donor_pool(*, available: list[datetime], cut: datetime) -> np.ndarray:
    """Return the day numbers of the runs that may donate at a fold: published before the cut.

    Args:
        available: The runs that carry cloud and have a cache file.
        cut: The fold's cut.

    Returns:
        The sorted day numbers of the available runs with `run + 9 h <= cut`.
    """
    return np.array(
        sorted(day_number(run=r) for r in available if r + common.PUBLICATION_DELAY <= cut),
        dtype=np.int64,
    )


def shuffled_donor_map(
    *, targets: list[datetime], available: list[datetime], cut: datetime, seed: int, series_key: int
) -> np.ndarray:
    """Return the shuffled-run donor day of each target run, `UNMAPPED` for a missing run.

    Args:
        targets: The runs to map, in order (the runs that a row can read).
        available: The runs that carry cloud and have a cache file.
        cut: The fold's cut; every donor satisfies `donor + 9 h <= cut`.
        seed: The fitting seed.
        series_key: An integer that identifies the series, so each series draws its own donors.

    Returns:
        An array shaped like `targets`: the donor's day number, or `UNMAPPED` where the target run
        carries no cloud.

    Raises:
        ValueError: If a target that carries cloud has no donor in another calendar year within 15
            days of its day of year.
    """
    pool = donor_pool(available=available, cut=cut)
    pool_doy, pool_year = calendar_of(days=pool)
    rng = np.random.default_rng([seed, series_key, int(cut.timestamp()) // 86_400])
    donors = np.full(len(targets), UNMAPPED, dtype=np.int64)
    for i, run in enumerate(targets):
        if common.is_missing_run(run=run):
            continue
        day = np.array([day_number(run=run)])
        doy, year = calendar_of(days=day)
        near = circular_gap(a=int(doy[0]), b=pool_doy) <= DAY_OF_YEAR_WINDOW
        candidates = pool[near & (pool_year != year[0])]
        if len(candidates) == 0:
            msg = f"run {run:%Y-%m-%d} has no donor in another year within 15 days of its day"
            raise ValueError(msg)
        donors[i] = candidates[rng.integers(len(candidates))]
    return donors


def shifted_donor_map(*, targets: list[datetime], available: list[datetime]) -> np.ndarray:
    """Return the time-shifted donor day of each target run, `UNMAPPED` for a missing run.

    Args:
        targets: The runs to map.
        available: The runs that carry cloud and have a cache file.

    Returns:
        The day number of the run issued 14 days earlier, or 21 days earlier, or 7 days earlier, or
        `YEAR_SHIFT_DAYS` (744 days) earlier, taking the first of those that is available.

    Raises:
        ValueError: If a target that carries cloud has none of the four donors.
    """
    held = {day_number(run=r) for r in available}
    donors = np.full(len(targets), UNMAPPED, dtype=np.int64)
    for i, run in enumerate(targets):
        if common.is_missing_run(run=run):
            continue
        day = day_number(run=run)
        for shift in (*SHIFT_DAYS, YEAR_SHIFT_DAYS):
            if day - shift in held:
                donors[i] = day - shift
                break
        else:
            msg = f"run {run:%Y-%m-%d} has no time-shifted donor"
            raise ValueError(msg)
    return donors


# --- assertions ---------------------------------------------------------------------------------


def assert_same_missing_pattern(*, targets: list[datetime], donors: np.ndarray) -> None:
    """Assert that a placebo row is missing if and only if the real row is missing.

    Args:
        targets: The runs of the map.
        donors: The donor day of each target, `UNMAPPED` where the placebo is missing.

    Raises:
        ValueError: If a run's two missing flags differ.
    """
    real = np.array(common.cloud_missing_flags(runs=targets))
    placebo = donors == UNMAPPED
    if (real != placebo).any():
        msg = f"{int((real != placebo).sum())} runs differ in their real and placebo missing flag"
        raise ValueError(msg)


def assert_donors_carry_cloud(*, donors: np.ndarray) -> None:
    """Assert that no mapped donor is a run in the gap or the upstream gap run.

    Args:
        donors: The donor day number of each target run, `UNMAPPED` where there is none.

    Raises:
        ValueError: If a donor run carries no cloud.
    """
    for day in donors[donors != UNMAPPED]:
        if common.is_missing_run(run=run_of(day=int(day))):
            msg = f"donor {run_of(day=int(day)):%Y-%m-%d} carries no cloud"
            raise ValueError(msg)


def assert_shuffled_rules(*, targets: list[datetime], donors: np.ndarray, cut: datetime) -> None:
    """Assert the three rules of the shuffled-run donor on every mapped run.

    The donor is published before the cut, lies in another calendar year, and is within 15 days of
    the target's day of year.

    Args:
        targets: The runs of the map.
        donors: The donor day of each target.
        cut: The fold's cut.

    Raises:
        ValueError: If any mapped donor breaks a rule.
    """
    assert_same_missing_pattern(targets=targets, donors=donors)
    assert_donors_carry_cloud(donors=donors)
    mapped = donors != UNMAPPED
    target_days = np.array([day_number(run=r) for r in targets], dtype=np.int64)[mapped]
    chosen = donors[mapped]
    cut_day_limit = (cut - common.PUBLICATION_DELAY).timestamp() / 86_400
    if (chosen > cut_day_limit).any():
        msg = "a donor run is published after the fold's cut"
        raise ValueError(msg)
    target_doy, target_year = calendar_of(days=target_days)
    donor_doy, donor_year = calendar_of(days=chosen)
    if (donor_year == target_year).any():
        msg = "a donor run is in its target run's calendar year"
        raise ValueError(msg)
    gap = np.minimum(np.abs(donor_doy - target_doy), YEAR_DAYS - np.abs(donor_doy - target_doy))
    if (gap > DAY_OF_YEAR_WINDOW).any():
        msg = "a donor run is more than 15 days of the year away from its target run"
        raise ValueError(msg)


def assert_shifted_rules(*, targets: list[datetime], donors: np.ndarray) -> None:
    """Assert that every time-shifted donor is issued before its target run, and carries cloud.

    Args:
        targets: The target runs.
        donors: The donor day number of each target run.

    Raises:
        ValueError: If a donor is not before its target, or the missing patterns differ.
    """
    assert_same_missing_pattern(targets=targets, donors=donors)
    assert_donors_carry_cloud(donors=donors)
    mapped = donors != UNMAPPED
    target_days = np.array([day_number(run=r) for r in targets], dtype=np.int64)[mapped]
    if (donors[mapped] >= target_days).any():
        msg = "a time-shifted donor is not issued before its target run"
        raise ValueError(msg)


# --- applying a map to cloud rows -----------------------------------------------------------------


def apply_donor_map(
    *, rows: pl.DataFrame, targets: list[datetime], donors: np.ndarray, columns: tuple[str, ...]
) -> pl.DataFrame:
    """Return the rows with each run's `columns` replaced by its donor run's at the same leads.

    Args:
        rows: Cloud rows of one series for every run that carries cloud, with `nwp_init_time`,
            `power_fcst_init_time`, `valid_time`, and `columns`.
        targets: The runs of the map.
        donors: The donor day of each target; a target with `UNMAPPED` is dropped, because a
            missing run has no row to replace.
        columns: The cloud columns to replace.

    Returns:
        The rows of the mapped targets, with the donor's values under the same names.

    Raises:
        ValueError: If a donor run or a lead is absent from `rows`, or the donor's lead grid
            differs from the target's.
    """
    if rows["time_series_id"].n_unique() != 1:
        msg = "apply_donor_map takes the rows of one series: the maps are per series"
        raise ValueError(msg)
    keep = donors != UNMAPPED
    link = pl.DataFrame(
        {
            "nwp_init_time": [t for t, k in zip(targets, keep, strict=True) if k],
            "donor_init_time": [run_of(day=int(d)) for d in donors[keep]],
        },
        schema={
            "nwp_init_time": rows.schema["nwp_init_time"],
            "donor_init_time": rows.schema["nwp_init_time"],
        },
    )
    with_lead = rows.with_columns(lead=pl.col("valid_time") - pl.col("power_fcst_init_time"))
    own = with_lead.drop(*columns).join(link, on="nwp_init_time", how="inner")
    donor_values = with_lead.select(
        pl.col("nwp_init_time").alias("donor_init_time"), "lead", *columns
    )
    out = own.join(donor_values, on=["donor_init_time", "lead"], how="left")
    if out.select(pl.col(columns[0]).is_null().any()).item():
        msg = "a donor run is absent from the rows, or its leads differ from its target's"
        raise ValueError(msg)
    return out.drop("donor_init_time", "lead").sort(*common.KEYS)


# --- stored maps --------------------------------------------------------------------------------


def _atomic_save(*, path: Path, donor: np.ndarray, target: np.ndarray) -> None:
    """Write an `.npz` file through a temporary name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, donor=donor, target=target)
    temporary.rename(path)


def _assert_stored_calendar(*, stored_target: np.ndarray, targets: list[datetime]) -> None:
    """Raise if a stored map was drawn for a different list of target runs."""
    current = np.array([day_number(run=r) for r in targets])
    if stored_target.shape != current.shape or (stored_target != current).any():
        msg = "a stored placebo map was drawn for a different list of target runs: delete it"
        raise ValueError(msg)


def target_runs(*, last: datetime = common.SCORED_LAST_RUN) -> list[datetime]:
    """Return every run that a row can read: from 2021-07-20 to `last`, gap runs included."""
    return common.runs_between(first=TARGET_FIRST_RUN, last=last)


def stored_shuffled_map(*, month: str, seed: int, series_key: int) -> np.ndarray:
    """Return a series' shuffled-run donor map for a fold, drawing and storing it if absent.

    Args:
        month: The fold's month, such as `2026-01`; the cut is 00:00 UTC on its first day.
        seed: The fitting seed.
        series_key: An integer that identifies the series.

    Returns:
        The donor days of `target_runs()`.
    """
    cut = datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC)
    path = SHUFFLED_DIR / f"{month}_s{seed}" / f"{series_key}.npz"
    targets = target_runs()
    if path.exists():
        stored = np.load(path)
        _assert_stored_calendar(stored_target=stored["target"], targets=targets)
        donors = stored["donor"]
        assert_shuffled_rules(targets=targets, donors=donors, cut=cut)
    else:
        donors = shuffled_donor_map(
            targets=targets,
            available=common.cache_runs(),
            cut=cut,
            seed=seed,
            series_key=series_key,
        )
        assert_shuffled_rules(targets=targets, donors=donors, cut=cut)
        _atomic_save(path=path, donor=donors, target=np.array([day_number(run=r) for r in targets]))
    return donors


def stored_shifted_map(*, series_key: int) -> np.ndarray:
    """Return a series' time-shifted donor map, drawing and storing it if absent."""
    path = SHIFTED_DIR / f"{series_key}.npz"
    targets = target_runs()
    if path.exists():
        stored = np.load(path)
        _assert_stored_calendar(stored_target=stored["target"], targets=targets)
        assert_shifted_rules(targets=targets, donors=stored["donor"])
        return stored["donor"]
    donors = shifted_donor_map(targets=targets, available=common.cache_runs())
    assert_shifted_rules(targets=targets, donors=donors)
    _atomic_save(path=path, donor=donors, target=np.array([day_number(run=r) for r in targets]))
    return donors


def describe(*, month: str, seed: int, series_key: int) -> None:
    """Draw both maps of a series for a fold and print the rules that hold and one example."""
    cut = datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC)
    targets = target_runs()
    shuffled = stored_shuffled_map(month=month, seed=seed, series_key=series_key)
    shifted = stored_shifted_map(series_key=series_key)
    assert_shuffled_rules(targets=targets, donors=shuffled, cut=cut)
    assert_shifted_rules(targets=targets, donors=shifted)
    mapped = np.nonzero(shuffled != UNMAPPED)[0]
    example = int(mapped[len(mapped) // 2])
    print(
        f"series {series_key} {month} seed {seed}: {len(mapped)} of {len(targets)} runs mapped, "
        f"{len(targets) - len(mapped)} missing in both placebos; all shuffled donors published "
        f"before {cut:%Y-%m-%d}, in another year, within 15 days of the day of year; all shifted "
        f"donors issued before their runs; e.g. run {targets[example]:%Y-%m-%d} takes the shuffled "
        f"cloud of {run_of(day=int(shuffled[example])):%Y-%m-%d} and the shifted cloud of "
        f"{run_of(day=int(shifted[example])):%Y-%m-%d}"
    )


def main() -> None:
    """Check the rules for some series and one fold, or write every Primary's maps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "write"))
    parser.add_argument("--months", nargs="+", default=["2026-01"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--series-keys", nargs="+", type=int, default=None)
    args = parser.parse_args()
    from e00711_cloud import primaries

    keys = args.series_keys or [int(s.removeprefix("D")) for s in primaries()["site"]]
    for month in args.months:
        for seed in args.seeds:
            for key in keys[: 2 if args.command == "check" else None]:
                describe(month=month, seed=seed, series_key=key)


if __name__ == "__main__":
    main()
