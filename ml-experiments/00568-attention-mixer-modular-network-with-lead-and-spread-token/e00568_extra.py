# ruff: noqa: E501
"""Experiment 568: injected-shift leak, CRPS by spread tercile, and attention weights.

Each table is printed and stored as CSV.
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00568-attention-mixer-modular-network-with-lead-and-spread-token"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
W: Final[Path] = WORK / "work"
MONTHS: Final = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
ARMS: Final = ("T0", "T0z", "T1", "T2", "T3", "T4", "T5")
KEYS: Final = ["time_series_id", "power_fcst_init_time", "valid_time"]


def leak() -> pl.DataFrame:
    """Sign times the change in the median forecast over the 7 days after the block.

    The change is in points of capacity, and the fine-tune happens at day 14.
    """
    rows = []
    for arm in ARMS:
        for m in MONTHS:
            ctrl = pl.read_parquet(W / "stress" / arm / f"{m}_ctrl.parquet").select(*KEYS, c="q50")
            for sign, tag in ((1, "p"), (-1, "n")):
                s = pl.read_parquet(W / "stress" / arm / f"{m}_{tag}.parquet").select(
                    *KEYS, s="q50"
                )
                j = ctrl.join(s, on=KEYS).with_columns(
                    leak=sign * 100.0 * (pl.col("s") - pl.col("c"))
                )
                rows.append({"arm": arm, "month": m, "sign": sign, "leak": j["leak"].mean()})
    t = pl.DataFrame(rows)
    out = (
        t.group_by("arm", "month")
        .agg(pl.col("leak").mean())
        .group_by("arm")
        .agg(leak=pl.col("leak").mean(), low=pl.col("leak").min(), high=pl.col("leak").max())
    )
    out = out.sort("arm")
    out.write_csv(WORK / "scores" / "stress_leak.csv")
    return out


def slices() -> pl.DataFrame:
    """Contrasts at 24-120 h by slice: cold nights, Christmas week, and spread tercile.

    The Bonferroni count is 6 per family.
    """
    import e00521_stats as s  # ty: ignore[unresolved-import]
    import e00559_extra as extra  # ty: ignore[unresolved-import]

    base = s.load()
    keys = ["primary", "origin", "target"]
    own = pl.concat(
        [
            pl.read_parquet(
                W / "scores" / f"rows_{m}.parquet", columns=[*keys, *[f"{a}__crps" for a in ARMS]]
            )
            for m in MONTHS
        ]
    )
    tert = pl.concat(
        [
            pl.read_parquet(W / "terciles" / f"{m}.parquet")
            .with_columns(
                primary=pl.col("primary").cast(pl.Int16),
                origin=((pl.col("power_fcst_init_time") - extra.GRID) // extra.STEP).cast(pl.Int32),
                target=((pl.col("valid_time") - extra.GRID) // extra.STEP).cast(pl.Int32),
            )
            .select(*keys, "tercile")
            for m in MONTHS
        ]
    )
    t = base.join(own, on=keys, how="left", validate="1:1").join(
        tert, on=keys, how="left", validate="1:1"
    )
    t = t.filter(pl.all_horizontal([pl.col(f"{a}__crps").is_finite() for a in ARMS]))
    rng = np.random.default_rng(0)
    rows = []
    for setname in ("eight", "six"):
        part = s.row_set(table=t, name=setname).filter(
            pl.col("lead_hours").is_between(24, 120, closed="left")
        )
        slc = {
            "all": part,
            "cold nights": part.filter(pl.col("cold")),
            "Christmas week": part.filter(pl.col("xmas")),
            **{f"spread tercile {i}": part.filter(pl.col("tercile") == i) for i in (1, 2, 3)},
        }
        for name, sub in slc.items():
            for a, b in (("T1", "T0"), ("T4", "T0"), ("T1", "T2"), ("T1", "T4"), ("T5", "T2")):
                dd = sub.select(
                    "month", "day", d=pl.col(f"{a}__crps") - pl.col(f"{b}__crps")
                ).filter(pl.col("d").is_finite())
                if dd.height == 0:
                    continue
                st = s.block_stats(diff=dd, rng=rng, tests=6)
                rows.append({"set": setname, "slice": name, "contrast": f"{a} - {b}", "n": dd.height,
                             "difference": st["difference"], "low": st["low_bonferroni_month"], "high": st["high_bonferroni_month"],
                             "folds_positive": st["folds_positive"]})  # fmt: skip
    out = pl.DataFrame(rows)
    out.write_csv(WORK / "scores" / "slices.csv")
    return out


if __name__ == "__main__":
    with pl.Config(tbl_rows=50, fmt_str_lengths=40):
        print(leak().with_columns(pl.col(pl.Float64).round(2)))
        print(slices().with_columns(pl.col(pl.Float64).round(3)))
