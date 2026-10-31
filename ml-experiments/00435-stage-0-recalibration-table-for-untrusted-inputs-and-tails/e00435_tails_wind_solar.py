"""Experiment 435: tail coverage on stored wind (429) and solar (430) forecasts.

Wind: each arm's stored quantiles (5 seeds, rows in the order of `data347.scored_frame()`) are
averaged over seeds in quantile space, then scored raw and widened by 1.15 per lead band, and per
band of the forecast median (a stand-in for a wind-speed band, because the scored frame carries no
wind speed; the median rises with wind speed up to the rated power, so the top band holds the
windiest half-hours). Solar: each arm's stored parquet files (seeds 0 to 2) are averaged the same
way. Quantile levels are crossed-sorted after averaging. Output: `work/tails_429.parquet` and
`work/tails_430.parquet`.

```bash
# PYTHONPATH: the folders of e00429_score's transitive imports (experiments 303, 326, 347, 429)
uv run --no-sync python ml-experiments/00435-*/e00435_tails_wind_solar.py
```
"""

from pathlib import Path
from typing import Final

import e00347_s4_data as data347  # ty: ignore[unresolved-import]
import e00429_run as run429  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00435_tailcore import (  # ty: ignore[unresolved-import]
    TAIL_LEVELS,
    WIDENING,
    lead_band_expr,
    tail_table,
    widen,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails"
    / "work"
)
SOLAR_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00430-unified-forecaster-solar-and-pretraining-ablations"
    / "predictions"
)
LEVELS: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
MEDIAN_BANDS: Final[tuple[float, ...]] = (0.1, 0.4, 0.7)
"""Edges of the forecast-median bands, as shares of capacity."""


def median_band_expr() -> pl.Expr:
    """Return the name of the forecast-median band."""
    q = pl.col("q50")
    return (
        pl.when(q < MEDIAN_BANDS[0]).then(pl.lit("med<0.1"))
        .when(q < MEDIAN_BANDS[1]).then(pl.lit("med0.1-0.4"))
        .when(q < MEDIAN_BANDS[2]).then(pl.lit("med0.4-0.7"))
        .otherwise(pl.lit("med>=0.7"))
    )  # fmt: skip


def score_frame(
    *, frame: pl.DataFrame, source: str, arm: str, by_sets: list[list[str]]
) -> pl.DataFrame:
    """Return the raw and widened tail tables of one arm for each key set in `by_sets`."""
    out: list[pl.DataFrame] = []
    scored = frame.filter(pl.col("actual").is_not_nan() & pl.col("actual").is_not_null())
    scored = scored.with_columns(lead_band=lead_band_expr(), median_band=median_band_expr())
    scored = scored.filter(pl.col("lead_band").is_not_null())
    for variant, f in (("raw", scored), ("w1.15", widen(frame=scored, factor=WIDENING))):
        for by in by_sets:
            table = tail_table(frame=f, by=tuple(by))
            out.append(
                table.with_columns(
                    source=pl.lit(source), arm=pl.lit(arm), variant=pl.lit(variant),
                    by=pl.lit("+".join(by)), key=pl.concat_str(by, separator="|"),
                ).drop(by)
            )  # fmt: skip
    return pl.concat(out)


def wind_tables() -> pl.DataFrame:
    """Score every stored experiment 429 arm."""
    cases = data347.scored_frame()
    base = cases.select("site", "month", "masked", "lead_hours", actual=pl.col("actual_norm"))
    out: list[pl.DataFrame] = []
    for name in run429.ARMS:
        paths = [run429.PARTS_DIR / f"{name}_h{s}.npy" for s in run429.SEEDS]
        present = [np.load(p) for p in paths if p.exists()]
        if not present:
            continue
        q = np.sort(np.mean(np.stack(present), axis=0), axis=1)
        frame = base.with_columns(*[pl.Series(c, q[:, i]) for i, c in enumerate(LEVELS)])
        print(f"429 {name}: {len(present)} seeds, {frame.height} rows")
        out.append(
            score_frame(
                frame=frame, source="429", arm=name,
                by_sets=[["lead_band"], ["site", "lead_band"], ["median_band", "lead_band"]],
            )
        )  # fmt: skip
    return pl.concat(out)


def solar_tables() -> pl.DataFrame:
    """Score every stored experiment 430 arm."""
    out: list[pl.DataFrame] = []
    for arm_dir in sorted(p for p in SOLAR_DIR.iterdir() if p.is_dir()):
        files = sorted(arm_dir.glob("s*_*.parquet"))
        if not files:
            continue
        frame = pl.concat([pl.read_parquet(f) for f in files])
        keys = ["time", "init_time", "time_series_id", "lead_hours", "actual"]
        mean = frame.group_by(keys).agg(*[pl.col(c).mean() for c in LEVELS]).sort(keys)
        q = np.sort(mean.select(list(LEVELS)).to_numpy(), axis=1)
        mean = mean.with_columns(*[pl.Series(c, q[:, i]) for i, c in enumerate(LEVELS)])
        print(f"430 {arm_dir.name}: {len(files)} files, {mean.height} rows")
        out.append(
            score_frame(
                frame=mean, source="430", arm=arm_dir.name,
                by_sets=[["lead_band"], ["time_series_id", "lead_band"]],
            )
        )  # fmt: skip
    return pl.concat(out)


def main() -> None:
    """Write the wind and solar tail tables."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    wind = wind_tables()
    wind.write_parquet(WORK_DIR / "tails_429.parquet")
    solar = solar_tables()
    solar.write_parquet(WORK_DIR / "tails_430.parquet")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(20)
    cols = ["arm", "variant", "key", "n", *[f"cov_{q}" for q in TAIL_LEVELS]]
    for table in (wind, solar):
        print(table.filter(pl.col("by") == "lead_band").select(cols).sort("arm", "variant", "key"))


if __name__ == "__main__":
    main()
