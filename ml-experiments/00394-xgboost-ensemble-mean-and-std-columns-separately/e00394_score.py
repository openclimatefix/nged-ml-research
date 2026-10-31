"""Score the mean-only and std-only arms against the stored monthly member and experiment 389's S1.

Reuses experiment 389's scorer with this experiment's arms folder and contrasts. Run
`e00394_arms.py` for each arm first.

    PYTHONPATH=ml-experiments/00379-*:ml-experiments/00087-*:ml-experiments/00389-* \
        DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00394-*/e00394_score.py MEAN
"""

import sys
from typing import Final

import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00379_score as base  # ty: ignore[unresolved-import]
import e00389_score as sc  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARMS: Final[list[str]] = sys.argv[1:]


def arm_frames(*, series: pl.DataFrame, extra: list[str]) -> dict[str, pl.DataFrame]:
    """Return the forecast in MW of every arm and of BG2, leads to 336 hours.

    The `manual_heuristic` and `climatology` references are left out to keep the memory of the
    scoring process low; experiment 389's page scores them on the same rows.
    """
    out = {
        name: adapter.forecast_frame(
            stored=pl.read_parquet(sc.WORK / "arms" / f"{name}.parquet"),
            series=series,
            unit="fraction",
        )
        for family in sc.FAMILIES.values()
        for name in family
    }
    for name in sc.BASES.values():
        out[name] = adapter.forecast_frame(
            stored=pl.read_parquet(sc.WORK / "arms" / f"{name}.parquet"),
            series=series,
            unit="fraction",
        )
    stored = pl.read_parquet(base.STORED_87 / f"{base.STORED['bg2']}.parquet").filter(
        pl.col("time_series_id").is_in(series["time_series_id"].to_list()),
        pl.col("lead_hours") <= base.MAX_LEAD_HOURS,
    )
    out["bg2"] = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    return out


def main() -> None:
    """Point the experiment 389 scorer at this experiment's arms and run it."""
    sc.WORK = (
        ML_EXPERIMENTS_DATA_DIR / "00394-xgboost-ensemble-mean-and-std-columns-separately" / "work"
    )
    # Only the widened (factor 1.15) arms are scored here, to keep the memory of the process low.
    sc.FAMILIES = dict.fromkeys(("S0M", "S1M", *ARMS), ())
    sc.BASES = {a: f"{a}_base" for a in ("S0M", "S1M", *ARMS)}
    sc.CONTRASTS = {
        "S0M_base": tuple(f"{a}_base" for a in ("S1M", *ARMS)),
        "S1M_base": tuple(f"{a}_base" for a in ARMS),
    }
    sc.arm_frames = arm_frames
    sys.argv = [sys.argv[0]]
    sc.main()


if __name__ == "__main__":
    main()
