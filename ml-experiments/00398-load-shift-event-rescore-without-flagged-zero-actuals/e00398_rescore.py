"""Re-aggregate stored per-row scores with and without the flagged exact-zero half-hours.

One streaming pass per file computes, for every arm, Primary, slice, and metric, the mean over all
stored rows (old) and over the rows whose (Primary, valid time) is not flagged (new). Counts of the
dropped rows are kept so the table shows how many rows each file loses.

    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00398-*/e00398_rescore.py <name>
"""

import sys
from pathlib import Path
from typing import Final

import polars as pl
from e00398_flagged_keys import OUT, ROW_FILES
from studies.eval_protocol.metrics import METRIC_NAMES
from studies.eval_protocol.slices import SLICE_NAMES
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARM_FILES_395: Final[list[str]] = [
    "m_control", "m_e7", "m_hl025", "m_hl1", "m_leadw", "m_solar", "m_sunny", "m_win12",
    "S0M_raw", "S1M_raw", "stack",
]  # fmt: skip


def aggregate(*, path: Path, arm_literal: str | None) -> pl.DataFrame:
    """Return old and new means per arm, Primary, group, slice, and metric for one row file."""
    flagged = pl.scan_parquet(OUT / "flagged_keys.parquet").select(
        "time_series_id", "valid_time", flagged=pl.lit(value=True)
    )
    lf = pl.scan_parquet(path)
    names = lf.collect_schema().names()
    slices = [s for s in SLICE_NAMES if s in names]
    metrics = [m for m in METRIC_NAMES if m in names]
    if arm_literal is not None:
        lf = lf.with_columns(arm=pl.lit(arm_literal))
    lf = lf.join(flagged, on=["time_series_id", "valid_time"], how="left").with_columns(
        keep=pl.col("flagged").is_null()
    )
    pieces = []
    for s in slices:
        aggs = [pl.col(s).sum().alias("n_old"), (pl.col(s) & pl.col("keep")).sum().alias("n_new")]
        for m in metrics:
            aggs.append(pl.col(m).filter(pl.col(s)).sum().alias(f"old__{m}"))
            aggs.append(pl.col(m).filter(pl.col(s) & pl.col("keep")).sum().alias(f"new__{m}"))
        pieces.append(
            lf.group_by("arm", "time_series_id", "label", "group", "capacity_mw")
            .agg(aggs)
            .with_columns(slice=pl.lit(s))
        )
    return pl.concat(pieces).collect(engine="streaming")


def main() -> None:
    """Aggregate the file named on the command line."""
    name = sys.argv[1]
    if name == "395":
        base = ML_EXPERIMENTS_DATA_DIR / (ROW_FILES["395_control"].rsplit("/", 1)[0])
        res = pl.concat(
            [
                aggregate(path=base / f"b_rows_{arm}.parquet", arm_literal=arm)
                for arm in ARM_FILES_395
            ]
        )
    else:
        res = aggregate(path=ML_EXPERIMENTS_DATA_DIR / ROW_FILES[name], arm_literal=None)
    res.write_parquet(OUT / f"sums_{name}.parquet")
    print(name, res.height, "rows written")


if __name__ == "__main__":
    main()
