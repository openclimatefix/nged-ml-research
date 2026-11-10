# ruff: noqa: D103, E501
"""Experiment 677: follow of each ladder step on the held-out injected bank (private CSV).

Usage: python e00677_bank_score.py

Reads the bank files written by `e00677_cand.py bank` (one per month window, fold, and seed), which hold,
for every scored row whose 1-day lag falls inside an injected shift, the shift (a share of capacity) and the
median forecast from the observed lags and from the shifted lags. The follow of one group of rows is the
least-squares slope of the median's change on the shift, sum(change * shift) / sum(shift ** 2): 0 means the
median ignores the shift, 1 means it copies the shift in full. Writes `scores/bank_follow.csv` (by step,
version, seed, and lead band) and `scores/bank_follow_summary.csv` (by step and version, with the spread over
seeds and folds).
"""

import os
from pathlib import Path
from typing import Final

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
ROOT: Final[Path] = Path(os.environ["E00677_ROOT"]) if os.environ.get("E00677_ROOT") else ML_EXPERIMENTS_DATA_DIR / NAME
FOLDS: Final = ("2025-11", "2026-02", "2026-05", "2026-08")
ARM: Final = {"5": "P0_D50", "6": "P3_D50", "7": "P5_D50", "8": "P7_D50"}
BANDS: Final = [(0, 24), (24, 72), (72, 168), (168, 336)]


def load(*, span: int = 3) -> pl.DataFrame:
    frames = []
    for step in ("5w", "5wo", "6w", "6wo", "7w", "7wo", "8w", "8wo"):
        for month in FOLDS:
            folder = ROOT / f"{ARM[step[0]]}_{month}_{step}_sp{span}" / "work" / "bank"
            files = sorted(folder.glob("*.parquet"))
            assert files, folder
            for f in files:
                seed = int(f.stem.rsplit("_s", 1)[1])
                frames.append(
                    pl.read_parquet(f, columns=["power_fcst_init_time", "valid_time", "median_base", "median_bank", "lag_shift"])
                    .filter(pl.col("lag_shift") != 0)
                    .with_columns(
                        step=pl.lit(step), fold=pl.lit(month), seed=pl.lit(seed),
                        lead_h=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
                        change=pl.col("median_bank") - pl.col("median_base"),
                    )
                    .select("step", "fold", "seed", "lead_h", "change", "lag_shift")
                )
    return pl.concat(frames)


def slope(frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return frame.group_by(by).agg(
        follow=(pl.col("change") * pl.col("lag_shift")).sum() / (pl.col("lag_shift") ** 2).sum(),
        rows=pl.len(),
        mean_abs_shift=pl.col("lag_shift").abs().mean(),
    )


def main() -> None:
    rows = load()
    out = ROOT / "scores"
    bands = pl.concat(
        [slope(rows.filter((pl.col("lead_h") >= lo) & (pl.col("lead_h") < hi)).with_columns(band=pl.lit(f"{lo // 24}-{hi // 24} d")), ["step", "seed", "fold", "band"]) for lo, hi in BANDS]
    )
    allb = slope(rows.with_columns(band=pl.lit("0-14 d")), ["step", "seed", "fold", "band"])
    per = pl.concat([allb, bands]).sort("step", "band", "fold", "seed")
    per.write_csv(out / "bank_follow.csv")
    # Pool folds with weights equal to the number of shifted rows; summarise the spread over seeds.
    pooled = per.group_by("step", "band", "seed").agg(
        follow=(pl.col("follow") * pl.col("rows")).sum() / pl.col("rows").sum(), rows=pl.col("rows").sum()
    )
    summary = pooled.group_by("step", "band").agg(
        follow_mean=pl.col("follow").mean(), follow_sd_over_seeds=pl.col("follow").std(), seeds=pl.len(), rows_per_seed=pl.col("rows").mean()
    ).sort("step", "band")
    summary.write_csv(out / "bank_follow_summary.csv")
    print(summary.filter(pl.col("band") == "0-14 d"))
    print(summary)


if __name__ == "__main__":
    main()
