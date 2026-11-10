# ruff: noqa: D103
"""Experiment 534, `e00534_leak.py`: kept for the record.

Injected-shift leak (511's stress world, seed 0) of the 534 arms: median shift in points of
capacity, signed by the injected sign.

Ensembles: the per-row mean of the members' q50 shifts (the quantile average is linear in q50).
R-90 is 511's L-1d.
"""

import e00511_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W = ML_EXPERIMENTS_DATA_DIR / "00534-lag-free-recency-sweep-and-member-ensemble" / "work"
E511 = k.WORK_DIR
MONTHS = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]
FOLDER = {"R-3": W, "R-7": W, "R-14": W, "R-30": W, "R-90": E511}
TAG = {"R-90": "L-1d"}
ENS = {"ENS-3": ("R-7", "R-30", "R-90"), "ENS-5": ("R-3", "R-7", "R-14", "R-30", "R-90")}
BANDS = {"all": (0, 1e9), "day1": (0, 24), "days2_3": (24, 72), "days4_10": (72, 1e9)}


def shift(*, arm: str, month: str, day: int, name: str, sign: int) -> pl.DataFrame:
    folder, tag = FOLDER[arm], TAG.get(arm, arm)
    c = pl.read_parquet(folder / "fits" / tag / f"{month}_d{day}_s0.parquet").select(
        *KEYS, c=pl.col("q50")
    )
    s = pl.read_parquet(folder / "fits" / tag / f"{month}_d{day}_s0_stress{name}.parquet").select(
        *KEYS, s=pl.col("q50")
    )
    return (
        c.join(s, on=KEYS).select(*KEYS, sh=sign * 100.0 * (pl.col("s") - pl.col("c"))).sort(KEYS)
    )


def main() -> None:
    records = []
    for arm in (*FOLDER, *ENS):
        members = ENS.get(arm, (arm,))
        for month in MONTHS:
            for day in range(14, 21):
                signed = []
                for name, sign in (("p", 1), ("n", -1)):
                    frames = [
                        shift(arm=m, month=month, day=day, name=name, sign=sign) for m in members
                    ]
                    for f in frames[1:]:
                        assert f.select(KEYS).equals(frames[0].select(KEYS))
                    sh = np.mean([f["sh"].to_numpy() for f in frames], axis=0)
                    lead = (
                        (frames[0]["valid_time"] - frames[0]["power_fcst_init_time"])
                        .dt.total_hours()
                        .to_numpy()
                    )
                    signed.append((sh, lead))
                row = {"arm": arm, "month": month, "block_age_days": day - 14}
                for lab, (lo, hi) in BANDS.items():
                    row[lab] = float(
                        np.mean(
                            [
                                sh[(lead > lo if lo else lead >= 0) & (lead <= hi)].mean()
                                for sh, lead in signed
                            ]
                        )
                    )
                records.append(row)
    t = pl.DataFrame(records)
    t.write_csv(W / "scores" / "stress_leak_by_fold.csv")
    summary = t.group_by("arm", "block_age_days", maintain_order=True).agg(
        pl.col("all").mean(),
        pl.col("all").min().alias("min"),
        pl.col("all").max().alias("max"),
        pl.col("day1").mean(),
        pl.col("days2_3").mean(),
        pl.col("days4_10").mean(),
    )
    summary.write_csv(W / "scores" / "stress_leak.csv")
    with pl.Config(tbl_rows=100):
        print(
            summary.filter(pl.col("block_age_days").is_in([0, 3, 6])).with_columns(
                pl.col(pl.Float64).round(3)
            )
        )


if __name__ == "__main__":
    main()
