"""Experiment 431: the private gallery of 60 farm-days per farm.

    nice -n 19 uv run --no-sync python -I ml-experiments/00431-wind-idle-data-look/e00431_gallery.py

Reads `features_private.parquet` (from `e00431_analysis.py`) and experiment 210's `mask.parquet`.
Writes one PNG per farm and category (20 farm-days each) under
`data/ml-experiments/00431-wind-idle-data-look/gallery_private/`. The gallery shows raw farm-days
and is never published, never copied into `docs/`, and never named in a pull request.

A day (UTC) is "sure-idle" when at least 80% of its half-hours are in the sure-idle set,
"sure-generating" when at least 50% of its half-hours are in the sure-generating set (the set is
rarer), and a "disagreement" day when the 210 mask and the texture rule disagree on at least 40% of
its half-hours. The texture rule calls a half-hour idle when the 24-hour centred share of repeated
values is above the midpoint of that statistic's median in the two sure sets.
"""

import os
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

DATA_ROOT: Final[Path] = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
MASK_PATH: Final[Path] = DATA_ROOT / "00210-wind-idle-noise-mva-artefact" / "mask.parquet"
OUT_DIR: Final[Path] = DATA_ROOT / "00431-wind-idle-data-look"
GALLERY_DIR: Final[Path] = OUT_DIR / "gallery_private"
SITES: Final[dict[str, str]] = {"W1": "Farm A", "W2": "Farm B", "W3": "Farm C"}
N_DAYS: Final[int] = 20
MIN_HALF_HOURS: Final[int] = 40
SEED: Final[int] = 431
TEXTURE_STAT: Final[str] = "repeat_share_24h_centre"


def day_table(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return per-farm-day shares of sure-idle, sure-generating, and rule disagreement."""
    return (
        frame.with_columns(day=pl.col("valid_time").dt.date())
        .group_by("site", "day")
        .agg(
            n=pl.len(),
            idle_share=(pl.col("sure") == "idle").mean(),
            gen_share=(pl.col("sure") == "gen").mean(),
            disagree=(pl.col("masked") != pl.col("texture_idle")).mean(),
        )
        .filter(pl.col("n") >= MIN_HALF_HOURS)
    )


def pick(*, days: pl.DataFrame, site: str, category: str) -> pl.DataFrame:
    """Pick up to `N_DAYS` days of one category for one farm, at random with a fixed seed."""
    sub = days.filter(pl.col("site") == site)
    if category == "sure_idle":
        sub = sub.filter(pl.col("idle_share") >= 0.8)
    elif category == "sure_generating_low":
        sub = sub.filter(pl.col("gen_share") >= 0.5)
    else:
        sub = sub.filter(pl.col("disagree") >= 0.4)
    return sub.sample(min(N_DAYS, sub.height), seed=SEED).sort("day")


def draw(*, frame: pl.DataFrame, chosen: pl.DataFrame, site: str, category: str) -> None:
    """Draw one 4 by 5 grid: power as a share of capacity (blue) and ERA5 wind (grey)."""
    fig, axes = plt.subplots(4, 5, figsize=(20, 12), sharex=True)
    for ax, day in zip(axes.flat, chosen["day"].to_list(), strict=False):
        part = frame.filter((pl.col("site") == site) & (pl.col("valid_time").dt.date() == day))
        hours = part["valid_time"].dt.hour() + part["valid_time"].dt.minute() / 60
        ax.plot(hours, part["power"] / part["capacity_mw"] * 100, color="tab:blue", marker=".")
        masked = part.filter(pl.col("masked"))
        ax.scatter(
            masked["valid_time"].dt.hour() + masked["valid_time"].dt.minute() / 60,
            masked["power"] / masked["capacity_mw"] * 100,
            color="tab:red",
            s=12,
            zorder=3,
        )
        twin = ax.twinx()
        twin.plot(hours, part["wind_speed"], color="grey", alpha=0.6)
        ax.set_title(f"{day}", fontsize=9)
    fig.suptitle(
        f"{SITES[site]}, {category}: power (% of capacity, blue; red = in experiment 210 mask), "
        "ERA5 wind (m/s, grey)"
    )
    fig.tight_layout()
    fig.savefig(GALLERY_DIR / f"{SITES[site].replace(' ', '_')}_{category}.png", dpi=80)
    plt.close(fig)


def main() -> None:
    """Pick the days and draw the grids."""
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(OUT_DIR / "features_private.parquet")
    mask = (
        pl.read_parquet(MASK_PATH)
        .select("time_series_id", "valid_time")
        .with_columns(masked=pl.lit(value=True))
    )
    frame = frame.join(mask, on=["time_series_id", "valid_time"], how="left").with_columns(
        masked=pl.col("masked").fill_null(value=False)
    )
    medians = (
        frame.filter(pl.col("sure").is_not_null())
        .group_by("site", "sure")
        .agg(med=pl.col(TEXTURE_STAT).median())
        .pivot(on="sure", index="site", values="med")
    )
    cut = medians.with_columns(cut=(pl.col("idle") + pl.col("gen")) / 2).select("site", "cut")
    frame = frame.join(cut, on="site").with_columns(
        texture_idle=pl.col(TEXTURE_STAT) > pl.col("cut")
    )
    days = day_table(frame=frame.drop_nulls(TEXTURE_STAT))
    counts = []
    for site in SITES:
        for category in ("sure_idle", "sure_generating_low", "disagreement"):
            chosen = pick(days=days, site=site, category=category)
            counts.append({"site": site, "category": category, "n_days": chosen.height})
            if chosen.height:
                draw(frame=frame, chosen=chosen, site=site, category=category)
    pl.DataFrame(counts).write_parquet(OUT_DIR / "gallery_counts.parquet")
    print(pl.DataFrame(counts))


if __name__ == "__main__":
    main()
