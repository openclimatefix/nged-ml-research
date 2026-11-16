# ruff: noqa: E501
"""Experiment 699: the copied share of an injected lag-only shift, by arm, seed, fold, lead band, and depth.

The copied share of a group of rows is the least-squares slope of the median's change on the shift,
sum(change * shift) / sum(shift ** 2) (`slope` of experiment 677's `e00677_bank_score.py`). A bank
file holds, for every scored row, the shift of the row's 1-day lag (a share of capacity) and the
median forecast from the observed lags and from the shifted lags. This script reduces each file to
the numerator and the denominator of the slope for each arm, seed, and origin month, prints the
tables, and prints the three families' contrasts of the copied share with both intervals.

Usage: python e00699_bank.py [FOLD ...]   (the folds that every arm has; default: all eight)
Two banks: `581` (experiment 581's generator, the training generator) and `660` (experiment 660's
generator, files in the `_b660` folders).
"""

import sys
from functools import cache

import e00677_bank_score as bank677  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00699_common import (
    ARMS,
    FOLDS,
    HOLDOUT,
    LEVEL_FAMILY,
    MARGIN_COPIED_SHARE,
    bank_folder,
    seeds_of,
)
from e00699_stats import Interval, contrast

BANDS = tuple((f"{lo // 24} to {hi // 24} days", lo, hi) for lo, hi in bank677.BANDS)
SUFFIX = {"581": "", "660": "_b660"}
DEPTHS = (0.15, 0.30, 0.50)


@cache
def sums(*, arm_key: str, bank: str) -> pl.DataFrame:
    """Return the numerator, denominator, and row count of the slope by seed, fold, lead band, and depth."""
    arm = ARMS[arm_key]
    parts = []
    for month in FOLDS:
        folder = bank_folder(arm=arm, month=month, suffix=SUFFIX[bank])
        for seed in seeds_of(arm=arm):
            path = folder / f"{month}_d0_s{seed}.parquet"
            if not path.exists():
                continue
            frame = pl.read_parquet(path)
            assert frame["valid_time"].max() < HOLDOUT, (  # ty: ignore[unsupported-operator]
                f"a bank row at or after the holdout: {path}"
            )
            frame = frame.filter(pl.col("lag_shift") != 0).with_columns(
                lead_h=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
                change=(pl.col("median_bank") - pl.col("median_base")).cast(pl.Float64),
                shift=pl.col("lag_shift").cast(pl.Float64),
            )
            frame = frame.with_columns(
                band=pl.when(pl.col("lead_h") < 24).then(pl.lit(BANDS[0][0]))
                .when(pl.col("lead_h") < 72).then(pl.lit(BANDS[1][0]))
                .when(pl.col("lead_h") < 168).then(pl.lit(BANDS[2][0]))
                .otherwise(pl.lit(BANDS[3][0])),
                depth=(pl.col("shift").abs() * 100).round(0).cast(pl.Int32),
            )  # fmt: skip
            parts.append(
                frame.group_by("band", "depth").agg(
                    num=(pl.col("change") * pl.col("shift")).sum(),
                    den=(pl.col("shift") ** 2).sum(),
                    rows=pl.len(),
                ).with_columns(arm=pl.lit(arm_key), bank=pl.lit(bank), month=pl.lit(month), seed=pl.lit(seed))
            )  # fmt: skip
    if not parts:
        raise FileNotFoundError(f"no bank files for {arm_key} on bank {bank}")
    return pl.concat(parts)


def arrays(*, arm_key: str, bank: str, months: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return the numerators and denominators over all rows, shape (months, seeds)."""
    arm = ARMS[arm_key]
    table = (
        sums(arm_key=arm_key, bank=bank)
        .group_by("month", "seed")
        .agg(pl.col("num").sum(), pl.col("den").sum())
    )
    num = np.zeros((len(months), len(seeds_of(arm=arm))))
    den = np.zeros_like(num)
    for row in table.iter_rows(named=True):
        if row["month"] in months:
            i, j = months.index(row["month"]), list(seeds_of(arm=arm)).index(row["seed"])
            num[i, j], den[i, j] = row["num"], row["den"]
    empty = np.argwhere(den <= 0)
    assert not empty.size, (
        arm_key,
        bank,
        "no shifted rows in",
        [(months[i], list(seeds_of(arm=arm))[j]) for i, j in empty],
    )
    return num, den


def copied_share(*, arm_key: str, bank: str, months: tuple[str, ...]) -> float:
    """Return the pooled copied share of an arm over the folds and seeds."""
    num, den = arrays(arm_key=arm_key, bank=bank, months=months)
    return float(num.sum() / den.sum())


def family_contrast(
    *,
    a: str,
    b: str,
    bank: str,
    months: tuple[str, ...],
    paired_seeds: bool,
    level: float = LEVEL_FAMILY,
) -> Interval:
    """Return the copied share of arm b minus arm a, with both intervals."""
    na, da = arrays(arm_key=a, bank=bank, months=months)
    nb, db = arrays(arm_key=b, bank=bank, months=months)
    if na.shape[1] != nb.shape[1]:  # a stored arm holds three seeds: use seeds 0 to 2 of the other
        k = min(na.shape[1], nb.shape[1])
        na, da, nb, db = na[:, :k], da[:, :k], nb[:, :k], db[:, :k]
    return contrast(num_a=na, den_a=da, num_b=nb, den_b=db, level=level, paired_seeds=paired_seeds)


def non_inferior(*, interval: Interval) -> bool:
    """Return whether both upper bounds are below the margin."""
    return interval.upper_below(bound=MARGIN_COPIED_SHARE)


def main() -> None:
    """Print the copied shares and the three families' contrasts for both banks."""
    months = tuple(sys.argv[1:]) or FOLDS
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    for bank in ("581", "660"):
        print(f"\n=== bank {bank}: shifted rows per seed over the folds {months}")
        available = [
            k
            for k in ("none", "s3", "s23", "s12", "s123", "none_stored", "s3_stored")
            if _has(k, bank)
        ]
        for key in available:
            table = sums(arm_key=key, bank=bank).filter(pl.col("month").is_in(list(months)))
            per_seed = table.group_by("seed").agg(rows=pl.col("rows").sum())
            print(ARMS[key].label, "rows per seed:", sorted(per_seed["rows"].unique().to_list()))
            share = copied_share(arm_key=key, bank=bank, months=months)
            num, den = arrays(arm_key=key, bank=bank, months=months)
            by_seed = num.sum(axis=0) / den.sum(axis=0)
            print(
                f"  copied share {share:.4f}; by seed {np.round(by_seed, 4).tolist()}; SD over seeds {by_seed.std(ddof=1):.4f}"
            )
            for name, _lo, _hi in BANDS:
                t = (
                    table.filter(pl.col("band") == name)
                    .group_by("seed")
                    .agg(s=pl.col("num").sum() / pl.col("den").sum())
                )
                print(f"    lead {name}: mean over seeds {t['s'].mean():.4f}")
            for depth in DEPTHS if bank == "581" else ():  # 660's shifts are not at these depths
                t = (
                    table.filter(pl.col("depth") == round(depth * 100))
                    .group_by("seed")
                    .agg(s=pl.col("num").sum() / pl.col("den").sum(), rows=pl.col("rows").sum())
                )
                if t.height:
                    print(
                        f"    depth {depth:.2f}: mean over seeds {t['s'].mean():.4f}, rows per seed {t['rows'].mean():.0f}"
                    )
        if bank == "660":
            for key in available:
                rows = (
                    sums(arm_key=key, bank=bank)
                    .group_by("seed")
                    .agg(rows=pl.col("rows").sum())["rows"]
                )
                print(
                    f"  660 bank, {key}: shifted rows per seed over all folds, minimum {rows.min():,}"
                )
        for name, a, b, paired in (
            ("H: stages 1 to 3 minus stage 3 only", "s3", "s123", False),
            ("S2: stages 2 and 3 minus stage 3 only", "s3", "s23", True),
            ("S3: stages 1 to 3 minus stages 1 and 2", "s12", "s123", True),
            ("clean chain: stage 3 only minus no injection", "none", "s3", True),
        ):
            if not (_has(a, bank) and _has(b, bank)):
                continue
            iv = family_contrast(a=a, b=b, bank=bank, months=months, paired_seeds=paired)
            print(
                f"{name} [{bank}] {iv.level:.2f}%: {iv.point:+.4f}; bootstrap {iv.boot_lo:+.4f} to {iv.boot_hi:+.4f}; "
                f"t {iv.t_lo:+.4f} to {iv.t_hi:+.4f}; non-inferior at {MARGIN_COPIED_SHARE}: {non_inferior(interval=iv)}; "
                f"higher: {iv.lower_above(bound=0.0)}"
            )


def _has(arm_key: str, bank: str) -> bool:
    arm = ARMS[arm_key]
    if arm.stored and bank == "660":
        return False
    return any(bank_folder(arm=arm, month=m, suffix=SUFFIX[bank]).exists() for m in FOLDS)


if __name__ == "__main__":
    main()
