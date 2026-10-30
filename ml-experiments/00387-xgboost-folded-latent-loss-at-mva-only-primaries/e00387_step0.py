"""Experiment 387, Step 0: how close do the apparent-power readings get to zero?

Prints, for D12 and reference Primaries, the minimum, low quantiles, and counts of half-hours below
1%, 2%, and 5% of effective capacity, by period (training: before 2025-07-01; full record), and
when they occur. Writes `work/step0.parquet` of daily-grid readings for the charts.

    PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
        uv run --no-sync python ml-experiments/00387-*/e00387_step0.py
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00387-xgboost-folded-latent-loss-at-mva-only-primaries" / "work"
)
SITES: Final[tuple[str, ...]] = ("D12", "D2", "D3", "D5", "D10")
TRAIN_END: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
QUANTILES: Final[tuple[float, ...]] = (0.0, 0.001, 0.01, 0.05)
THRESHOLDS: Final[tuple[float, ...]] = (1.0, 2.0, 5.0)
SUNNY_SHORTWAVE: Final[float] = 400.0


def main() -> None:
    """Print the Step 0 statistics and store the readings."""
    WORK.mkdir(parents=True, exist_ok=True)
    series = common.series_table().filter(pl.col("site").is_in(SITES))
    arrays = common.load_series_arrays(series=series)
    columns = common.feature_columns()
    shortwave_index = columns.index("shortwave")
    frames = []
    for row in series.iter_rows(named=True):
        site = row["site"]
        a = arrays[site]
        frame = pl.DataFrame(
            {
                "site": site,
                "units": row["units"],
                "valid_time": a.valid_time,
                "reading": a.power * 100,
                "shortwave": a.features[:, shortwave_index],
            }
        ).filter(pl.col("reading").is_finite())
        for label, sub in (
            ("train", frame.filter(pl.col("valid_time") < TRAIN_END)),
            ("full", frame),
        ):
            for kind, mask in (
                ("all", sub),
                ("abs", sub.with_columns(reading=pl.col("reading").abs())),
            ):
                if site in ("D5", "D10") and kind == "all":
                    continue
                r = kind == "abs" and site in ("D5", "D10")
                values = mask["reading"]
                qs = {q: float(np.quantile(values.to_numpy(), q)) for q in QUANTILES}
                counts = {t: int((values < t).sum()) for t in THRESHOLDS}
                print(
                    site,
                    row["units"],
                    label,
                    "abs-of-signed" if r else "metered",
                    values.len(),
                    "min/q0.1/q1/q5 =",
                    [round(v, 3) for v in qs.values()],
                    "counts<1,2,5% =",
                    counts,
                )
        low = frame.filter(pl.col("reading") < 2.0)
        local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
        if low.height:
            print(
                site,
                "low (<2%) by month",
                low.group_by(local.dt.month().alias("m")).len().sort("m").rows(),
                "\n  by hour",
                low.group_by(local.dt.hour().alias("h")).len().sort("h").rows(),
                "\n  sunny share",
                round(float(np.mean((low["shortwave"] > SUNNY_SHORTWAVE).to_numpy())), 3),
            )
        frames.append(frame)
    pl.concat(frames).write_parquet(WORK / "step0.parquet")


if __name__ == "__main__":
    main()
