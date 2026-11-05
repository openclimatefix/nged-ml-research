# ruff: noqa: E501
"""Experiment 565, step R: injected-shift leak of A10 and M0 on seeds 0 to 4 (experiment 511's stress world).

The leak is `sign x 100 x (median with the block injected - median of the control)`, in points of
capacity, over the forecast windows of the daily fits on days 14 to 20 (block ages 0 to 6), both
signs, mean of the five folds. Files are looked up in every folder that holds the arm's fits.
"""

from pathlib import Path

import polars as pl
from e00565_design6 import MONTHS, W545, W559
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W554 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out/work"
)
FOLDERS = {"M0": [W545 / "fits/M0", W554 / "fits/M0"], "A10": [W559 / "fits/A10"]}
KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]
SEEDS = (0, 1, 2, 3, 4)


def find(*, arm: str, name: str) -> Path:
    """Return the one existing file called `name` in the arm's folders."""
    hits = [f / name for f in FOLDERS[arm] if (f / name).exists()]
    assert len(hits) >= 1, (arm, name)
    return hits[0]


def main() -> None:
    """Write the per-seed leak table and print the arm summaries."""
    rows = []
    for arm in FOLDERS:
        for seed in SEEDS:
            for month in MONTHS:
                for day in range(14, 21):
                    ctrl = pl.read_parquet(
                        find(arm=arm, name=f"{month}_d{day}_s{seed}.parquet")
                    ).select(*KEYS, c="q50")
                    for sign, tag in ((1, "p"), (-1, "n")):
                        st = pl.read_parquet(
                            find(arm=arm, name=f"{month}_d{day}_s{seed}_stress{tag}.parquet")
                        ).select(*KEYS, s="q50")
                        j = ctrl.join(st, on=KEYS)
                        assert j.height == ctrl.height
                        rows.append(
                            {
                                "arm": arm,
                                "seed": seed,
                                "month": month,
                                "block_age_days": day - 14,
                                "sign": sign,
                                "leak": float((sign * 100.0 * (j["s"] - j["c"])).mean()),
                            }
                        )
    t = pl.DataFrame(rows)
    out = W545.parent.parent / "00565-drift-embedding-variants-six-designs/work/scores"
    out.mkdir(parents=True, exist_ok=True)
    t.write_csv(out / "stepr_leak_long.csv")
    fold = t.group_by("arm", "seed", "month").agg(pl.col("leak").mean())
    per_seed = fold.group_by("arm", "seed").agg(leak=pl.col("leak").mean()).sort("arm", "seed")
    per_seed.write_csv(out / "stepr_leak_by_seed.csv")
    summary = per_seed.group_by("arm").agg(
        mean=pl.col("leak").mean(),
        sd=pl.col("leak").std(),
        lo=pl.col("leak").min(),
        hi=pl.col("leak").max(),
    )
    diff = per_seed.pivot(on="arm", index="seed", values="leak").with_columns(
        diff=pl.col("A10") - pl.col("M0")
    )
    summary.write_csv(out / "stepr_leak.csv")
    diff.write_csv(out / "stepr_leak_diff_by_seed.csv")
    with pl.Config(tbl_rows=30):
        print(per_seed.pivot(on="arm", index="seed", values="leak"), summary, diff, sep="\n")


if __name__ == "__main__":
    main()
