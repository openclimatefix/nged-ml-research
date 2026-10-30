"""Experiment 401, Job 2, score step: per-row losses of every arm, as in experiment 400.

Reads `arm_<name>.parquet` from the arms step, scores each arm with experiment 390's
`trimmed_rows` on experiment 395's Part F slice table, and writes `rows_<name>.parquet` with the
flagged half-hours tagged (`flagged`), the median, q05, q95, q99, and actual in MW.

```bash
PYTHONPATH=<as in e00401_arms.py, plus experiment 390 and 395's folders> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00401-*/e00401_score.py
```
"""

import gc
import sys
from pathlib import Path

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00390_build as b390  # ty: ignore[unresolved-import]
import e00400_partf as pf  # ty: ignore[unresolved-import]
import e00401_arms as arms
import polars as pl

KEYS = arms.KEYS
WORK = arms.WORK
BASE = ("lags", "lagless")
KINDS = ("mix", "avg")
WEIGHTS = ("w30", "w50", "w70", "lead")
NAMES: tuple[str, ...] = tuple(
    f"{stem}{suffix}"
    for suffix in ("", "_x115")
    for stem in (*BASE, *(f"{k}_{w}" for k in KINDS for w in WEIGHTS))
)


def main() -> None:
    """Score every arm, one at a time."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    table = pl.read_parquet(pf.WORK / "slice_table.parquet")
    flags = pf.flagged_keys(series=series)
    for name in NAMES:
        target = WORK / f"rows_{name}.parquet"
        if target.exists():
            continue
        frame = pl.read_parquet(WORK / f"arm_{name}.parquet").with_columns(
            pl.col("time_series_id").cast(series.schema["time_series_id"])
        )
        use = table.join(frame.select(*KEYS), on=KEYS, how="semi")
        rows = b390.trimmed_rows(name=name, frame=frame, table=use, ctx=ctx, series=series)
        rows = rows.join(flags, on=["time_series_id", "valid_time"], how="left").with_columns(
            pl.col("flagged").fill_null(value=False)
        )
        extra = frame.join(ctx["actuals"], on=["time_series_id", "valid_time"]).select(
            *KEYS,
            q50_mw=pl.col("q50"),
            actual_mw=pl.col("actual"),
            q95_mw=pl.col("q95"),
            q99_mw=pl.col("q99"),
            q05_mw=pl.col("q05"),
        )
        rows.join(extra, on=KEYS, how="left", validate="1:1").write_parquet(
            target.with_suffix(".tmp")
        )
        target.with_suffix(".tmp").rename(target)
        print("scored", name, rows.height, flush=True)
        del frame, rows, extra, use
        gc.collect()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
