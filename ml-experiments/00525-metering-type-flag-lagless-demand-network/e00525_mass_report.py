"""Summarise the mixture's probability mass below zero per Primary, fold, and seed (experiment 525).

Writes `scores/mass_by_primary.csv` (mean mass, share of half-hours whose q05 is below zero, share
whose median is below zero), `scores/mass_ratio.csv` (M over C per fold and seed at D8, D12, and D22),
and `scores/d10_reverse_flow.csv` (the same shares on the days experiment 506 stored as having a
negative reading at D10, from 2025-10-01). A half-hour has its q05 below zero when its mass exceeds
0.05, and its median below zero when its mass exceeds 0.5.
"""

from datetime import date
from typing import Final

import e00525_fit as fit  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final = fit.W / "scores"
FOCUS: Final[tuple[str, ...]] = ("D8", "D12", "D22", "D10")


def load(*, arms: tuple[str, ...]) -> pl.DataFrame:
    """Return every row's mass with its arm, fold, seed, and site."""
    sites = pl.DataFrame({"primary": list(range(16)), "site": fit.data.sites()})
    parts = []
    for arm in arms:
        for path in sorted((fit.W / "mass" / arm).glob("*.parquet")):
            month, seed = path.stem.split("_s")
            parts.append(
                pl.read_parquet(path).with_columns(
                    arm=pl.lit(arm), month=pl.lit(month), seed=pl.lit(int(seed))
                )
            )
    return pl.concat(parts).join(sites, on="primary", how="left")


def main() -> None:
    """Write the three tables."""
    arms = tuple(a for a in fit.MASS_ARMS if (fit.W / "mass" / a).exists())
    frame = load(arms=arms).with_columns(
        q05_below=(pl.col("mass") > 0.05).cast(pl.Float64),
        median_below=(pl.col("mass") > 0.5).cast(pl.Float64),
    )
    cell = (
        frame.group_by("arm", "site", "month", "seed")
        .agg(
            n=pl.len(),
            mean_mass=pl.col("mass").mean(),
            share_q05_below=pl.col("q05_below").mean(),
            share_median_below=pl.col("median_below").mean(),
        )
        .sort("arm", "site", "month", "seed")
    )
    OUT.mkdir(parents=True, exist_ok=True)
    by_primary = (
        cell.group_by("arm", "site")
        .agg(
            mean_mass=pl.col("mean_mass").mean(),
            share_q05_below=pl.col("share_q05_below").mean(),
            share_median_below=pl.col("share_median_below").mean(),
        )
        .sort("site", "arm")
    )
    by_primary.write_csv(OUT / "mass_by_primary.csv")
    focus = cell.filter(pl.col("site").is_in(["D8", "D12", "D22"]))
    ratio = (
        focus.filter(pl.col("arm") == "M")
        .join(
            focus.filter(pl.col("arm") == "C"),
            on=["site", "month", "seed"],
            suffix="_c",
        )
        .select(
            "site", "month", "seed",
            mass_m=pl.col("mean_mass"), mass_c=pl.col("mean_mass_c"),
            ratio=pl.col("mean_mass") / pl.col("mean_mass_c"),
        )
        .sort("site", "month", "seed")
    )  # fmt: skip
    ratio.write_csv(OUT / "mass_ratio.csv")
    days = pl.read_parquet(
        next(ML_EXPERIMENTS_DATA_DIR.glob("00506-*/work/part1_days_six.parquet"))
    ).filter(
        (pl.col("site") == "D10")
        & (pl.col("level") == 0)
        & pl.col("any_negative")
        & (pl.col("date") >= date(2025, 10, 1))
    )
    d10 = frame.filter(pl.col("site") == "D10").with_columns(
        day=pl.from_epoch("valid_us", time_unit="us").dt.date()
    )
    rev = d10.join(days.select(date=pl.col("date")), left_on="day", right_on="date", how="semi")
    out = (
        rev.group_by("arm")
        .agg(
            days=pl.col("day").n_unique(),
            rows=pl.len(),
            share_q05_below=pl.col("q05_below").mean(),
            share_median_below=pl.col("median_below").mean(),
        )
        .sort("arm")
    )
    out.write_csv(OUT / "d10_reverse_flow.csv")
    print(by_primary.filter(pl.col("site").is_in(list(FOCUS))))
    print(ratio)
    print(out, "stored reverse-flow days:", days.height)


if __name__ == "__main__":
    main()
