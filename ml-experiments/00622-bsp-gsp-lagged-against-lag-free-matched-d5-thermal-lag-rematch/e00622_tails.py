# ruff: noqa: E501
"""Experiment 622: tail calibration of the lag-enabled and lag-free networks (both with B1) from the Parquet exports.

For q01, q05, q95, and q99: the share of rows with the observed power at or below the quantile (target
equal to the level) and the pinball loss in points of effective capacity, by group and lead set, on
the rows both exports have.

Usage: python e00622_tails.py
"""

from pathlib import Path
from typing import Final

import e00611_pool as pool  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00622-bsp-gsp-lagged-against-lag-free-matched-d5-thermal-lag-rematch"
LEVELS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
FOLDERS: Final[dict[str, tuple[Path, str]]] = {
    "lagged_b1": (ROOT / "export", "lagged_b1"),
    "lagfree_b1": (ROOT / "export", "lagfree_b1"),
}
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]


def main() -> None:
    """Write `scores/tails.csv`."""
    caps = pl.read_csv(ROOT / "export" / "substations.csv")
    frames = {}
    for arm, (folder, prefix) in FOLDERS.items():
        frames[arm] = (
            pl.concat([pl.read_parquet(p) for p in sorted(folder.glob(f"{prefix}_*.parquet"))])
            .filter(pl.col("observed_power").is_not_null(), pl.col("lead_hours") < 240)
            .join(caps.select("time_series_id", "kind", "capacity_mw"), on="time_series_id")
        )
    shared = frames["lagged_b1"].select(KEYS).join(frames["lagfree_b1"].select(KEYS), on=KEYS)
    out = []
    for arm, frame in frames.items():
        f = frame.join(shared, on=KEYS).with_columns(
            set=pl.when(pl.col("lead_hours") < 72)
            .then(pl.lit("days1-3"))
            .otherwise(pl.lit("days4-10"))
        )
        for name, tau in LEVELS.items():
            diff = pl.col("observed_power") - pl.col(name)
            pin = (
                pl.when(diff >= 0).then(diff * tau).otherwise(-diff * (1 - tau))
                / pl.col("capacity_mw")
                * 100
            )
            out.append(
                f.group_by("kind", "set").agg(
                    pl.lit(arm).alias("arm"), pl.lit(name).alias("quantile"),
                    (pl.col("observed_power") <= pl.col(name)).mean().alias("share_at_or_below"),
                    pin.mean().alias("pinball_points"), pl.len().alias("n"),
                )
            )  # fmt: skip
    pl.concat(out).sort("kind", "set", "quantile", "arm").write_csv(ROOT / "scores" / "tails.csv")
    print(
        pl.read_csv(ROOT / "scores" / "tails.csv").to_pandas().to_string()
        if False
        else pl.read_csv(ROOT / "scores" / "tails.csv")
    )


if __name__ == "__main__":
    main()
