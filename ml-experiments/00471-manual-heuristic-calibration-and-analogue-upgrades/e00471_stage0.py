# ruff: noqa: E501, PTH207
"""Stage 0 of experiment 471: reproduce the stored heuristic, then measure the analogues' calibration.

Step 0 recomputes the 11 quantiles of every stored member set (linear interpolation) and compares
them with experiment 87's `ref_manual_heuristic`. Then, on the cleaned actuals, it prints the rank
histogram of the actual among the members, the correlation of the members' spread with the absolute
error of their mean, and the exceedance of every output level, per lead band and Primary group.
Tables go to `work/stage0/`.

```bash
source /tmp/e471_env.sh
uv run --no-sync python ml-experiments/00471-*/e00471_stage0.py
```
"""

import glob

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00448_refs as base  # ty: ignore[unresolved-import]
import e00471_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = common.WORK_DIR / "stage0"


def quantiles_of(*, members: np.ndarray) -> np.ndarray:
    """Return the 11 linear-interpolation quantiles of each row's non-NaN members."""
    return np.nanquantile(members, common.LEVEL_VALUES, axis=1, method="linear").T


def reproduce(*, table: pl.DataFrame, series: pl.DataFrame) -> pl.DataFrame:
    """Compare recomputed quantiles with experiment 87's stored reference; return the gap per Primary."""
    path = glob.glob(
        str(ML_EXPERIMENTS_DATA_DIR / "00087*/predictions/ref_manual_heuristic/predictions.parquet")
    )[0]
    stored = pl.read_parquet(path).select(*common.KEYS, *common.LEVELS)
    stored = stored.join(series.select("time_series_id", "capacity_mw"), on="time_series_id")
    q = quantiles_of(members=table.select(common.MEMBER_COLUMNS).to_numpy())
    mine = table.select(*common.KEYS).with_columns(
        *[pl.Series(f"r_{k}", q[:, i]) for i, k in enumerate(common.LEVELS)]
    )
    joined = mine.join(stored, on=common.KEYS)
    gap = joined.select(
        "time_series_id",
        *[
            ((pl.col(f"r_{k}") - pl.col(k) * pl.col("capacity_mw")).abs()).alias(f"gap_{k}")
            for k in common.LEVELS
        ],
    )
    summary = gap.group_by("time_series_id").agg(
        pl.len().alias("rows"), *[pl.col(f"gap_{k}").max() for k in common.LEVELS]
    )
    print(f"rows in the member table {table.height:,}; matched to experiment 87 {joined.height:,}")
    print(
        f"largest gap in MW over every row, level, and series: {gap.drop('time_series_id').max_horizontal().max():.3e}"
    )
    return summary


def calibration(*, table: pl.DataFrame, actuals: pl.DataFrame, groups: pl.DataFrame) -> None:
    """Print and write the rank histogram, spread-error correlation, and exceedance tables."""
    joined = table.join(actuals, on=["time_series_id", "valid_time"]).join(
        groups, on="time_series_id"
    )
    joined = joined.filter(pl.col("actual").is_finite())
    m = joined.select(common.MEMBER_COLUMNS).to_numpy()
    actual = joined["actual"].to_numpy()
    n = joined["n"].to_numpy()
    band = common.band_of(lead_hours=joined["lead_hours"].to_numpy())
    rank = np.nansum(m < actual[:, None], axis=1)
    mean = np.nanmean(m, axis=1)
    sd = np.nanstd(m, axis=1, ddof=1)
    q = quantiles_of(members=m)
    group = joined["group"].to_numpy()
    records_rank, records_corr, records_exc = [], [], []
    for g in ("MVA-only eight", "signed-MW six", "D15 and D18"):
        for b in range(5):
            sel = (group == g) & (band == b)
            for count in (13, 12):
                s2 = sel & (n == count)
                if s2.sum() < 100:
                    continue
                hist = np.bincount(rank[s2], minlength=count + 1) / s2.sum()
                records_rank.append({"group": g, "band": b, "members": count, "rows": int(s2.sum()),
                                     **{f"r{i}": float(h) for i, h in enumerate(hist)}})  # fmt: skip
            s3 = sel & (n >= 12)
            err = np.abs(actual - mean)
            rho = spearmanr(sd[s3], err[s3]).statistic
            records_corr.append({"group": g, "band": b, "rows": int(s3.sum()),
                                 "spearman_spread_vs_abs_error": float(rho),
                                 "mean_bias_mw": float((actual - mean)[s3].mean()),
                                 "mean_abs_error_mw": float(err[s3].mean())})  # fmt: skip
            exc = {
                f"exceed_{k}": float((actual[s3] > q[s3, i]).mean())
                for i, k in enumerate(common.LEVELS)
            }
            records_exc.append({"group": g, "band": b, "rows": int(s3.sum()), **exc})
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = pl.Config(tbl_rows=100, tbl_cols=30, tbl_width_chars=250)
    for name, rec in (
        ("rank_histogram", records_rank),
        ("spread_error", records_corr),
        ("exceedance", records_exc),
    ):
        frame = pl.DataFrame(rec)
        frame.write_csv(OUT / f"{name}.csv")
        with cfg:
            print(f"== {name} ==")
            print(frame.with_columns(pl.col(pl.Float64).round(3)))
    flagged = float(((m == 0.0).sum(axis=1) > 0).mean())
    print(f"share of rows with an exactly-zero member: {flagged:.5f}")
    print("member-count distribution in the joined rows:")
    print({int(k): int(v) for k, v in zip(*np.unique(n, return_counts=True), strict=True)})


def main() -> None:
    """Run step 0 and the calibration measurements."""
    series = adapter.series_table()
    groups = base.sites_table().select("time_series_id", "group")
    table = common.load_members()
    OUT.mkdir(parents=True, exist_ok=True)
    reproduce(table=table, series=series).write_csv(OUT / "reproduction.csv")
    actuals = common.cleaned_actuals(series=series)
    calibration(table=table, actuals=actuals, groups=groups)


if __name__ == "__main__":
    main()
