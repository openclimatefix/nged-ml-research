# ruff: noqa: E501
"""Experiment C: refit the two current defaults with a 730-day capacity.

The capacity is the 99.9th percentile of the previous 730 days.

Wind default 2 is experiment 615's refit (`e00615_wind_fit.py`) and solar `a2_member` is experiment
430's job (`e00430_fit.run_job`), each run unchanged except for the capacity rule: the target is the
output over the 730-day capacity, and the stored quantiles are scaled back to the units of
experiment 615's scoring capacity. Scoring capacity itself is not changed, so the scores compare with
the stored defaults. Outputs go to this experiment's folder; experiment 615's and 430's folders are
not written.

Commands (first argument):

- `gate-wind`: refit experiment 347's own head and compare with its stored file (615's gate).
- `gate-solar`: refit `a2_member` seed 0 for the 2026-06 refit with the old capacity rule (ratio 1)
  and compare with experiment 430's stored forecasts.
- `wind`: refit wind default 2 with the 730-day capacity, seeds 0 and 1.
- `solar`: refit `a2_member` with the 730-day capacity, seeds 0 to 2.
"""

import os
import sys
from datetime import UTC, datetime, timedelta

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_fit as f310  # ty: ignore[unresolved-import]
import e00430_common as common430  # ty: ignore[unresolved-import]
import e00430_fit as f430  # ty: ignore[unresolved-import]
import e00461_common as c461  # ty: ignore[unresolved-import]
import e00615_wind_fit as w615  # ty: ignore[unresolved-import]
import e00659_common as common
import e00659_solar_fit as sfit
import numpy as np
import polars as pl

_ORIGINAL_CAPACITY = c461.capacity
_ORIGINAL_ENS_PIECE = f430.ens_piece
_ORIGINAL_PREDICT = f430.predict_rows
_ORIGINAL_CEILING = f430.farm_ceiling
SOLAR_FORCE_ONE = False


def wind_capacity(*, site: str, cutoff: datetime) -> dict[str, float | int]:
    """Use the 730-day rule at refits and experiment 615's all-history rule for scoring capacity."""
    if cutoff == w615.SCORING_CAPACITY_CUTOFF:
        return _ORIGINAL_CAPACITY(site=site, cutoff=cutoff)
    start = cutoff - timedelta(days=common.CAPACITY_DAYS)
    series = pl.read_parquet(c461.SERIES_PATH).filter(pl.col("site") == site)
    values = (
        series.filter((pl.col("time") >= start) & (pl.col("time") < cutoff))["power"]
        .drop_nulls()
        .to_numpy()
    )
    old, new = float(series["cap"][0]), float(np.quantile(values, common.CAPACITY_QUANTILE))
    return {"old_mw": old, "new_mw": new, "ratio": new / old, "n_history": len(values)}


def ratio(*, farm: str, cutoff: datetime) -> float:
    """Return new over old capacity for a solar farm at the refit one day after `cutoff`."""
    if SOLAR_FORCE_ONE:
        return 1.0
    return sfit.capacity_730(farm=farm, refit=cutoff + timedelta(days=f310.REFIT_GAP_DAYS))["ratio"]


def ens_piece(*, farm: str, band: str, cutoff: datetime, need_members: bool):  # noqa: ANN201
    """Experiment 430's training piece with the target and ceiling over the new capacity."""
    piece = _ORIGINAL_ENS_PIECE(farm=farm, band=band, cutoff=cutoff, need_members=need_members)
    if SOLAR_FORCE_ONE:
        return piece
    piece.y = (piece.y / ratio(farm=farm, cutoff=cutoff)).astype(np.float32)
    piece.ceiling = np.ones_like(piece.ceiling)
    return piece


def farm_ceiling(farm: str, band: str, cutoff: datetime) -> float:
    """Ceiling in normalised units: 1.0 under the new rule, experiment 430's own otherwise."""
    return _ORIGINAL_CEILING(farm, band, cutoff) if SOLAR_FORCE_ONE else 1.0


def predict_rows(**kwargs):  # noqa: ANN003, ANN201
    """Experiment 430's prediction table, with quantiles scaled back to fractions of old capacity."""
    table = _ORIGINAL_PREDICT(**kwargs)
    if SOLAR_FORCE_ONE:
        return table
    cutoff = kwargs["cutoff"]
    scale = table["time_series_id"].replace_strict(
        {f: ratio(farm=f, cutoff=cutoff) for f in c310.FARMS}, return_dtype=pl.Float64
    )
    return table.with_columns(
        [
            (pl.col(c) * scale).alias(c)
            for c in table.columns
            if c.startswith("q") and c[1:].isdigit()
        ]
    )


def main() -> None:
    """Run the named command."""
    command = sys.argv[1]
    if command == "gate-wind":
        w615.gate()
    elif command == "wind":
        c461.capacity = wind_capacity
        w615.common461.capacity = wind_capacity
        w615.OUT_DIR = common.DATA_DIR / "wind_default2_730"
        sys.argv = [sys.argv[0], "--seeds", "0", "1"]
        w615.main()
    elif command in ("gate-solar", "solar"):
        global SOLAR_FORCE_ONE  # noqa: PLW0603
        SOLAR_FORCE_ONE = command == "gate-solar"
        out = common.DATA_DIR / ("solar_gate" if SOLAR_FORCE_ONE else "solar_a2_member_730")
        f430.common.PATH_ARMS = frozenset()
        f430.ens_piece, f430.predict_rows, f430.farm_ceiling = ens_piece, predict_rows, farm_ceiling
        f430.pred_path = lambda *, arm, seed, start: out / arm / f"s{seed}_{start:%Y-%m}.parquet"
        seeds = [0] if SOLAR_FORCE_ONE else [0, 1, 2]
        refits = [datetime(2026, 6, 1, tzinfo=UTC)] if SOLAR_FORCE_ONE else list(f310.REFIT_STARTS)
        for seed in seeds:
            for start in refits:
                print(
                    f430.run_job(
                        arm="a2_member", config=common430.ARMS["a2_member"], seed=seed, start=start
                    ),
                    flush=True,
                )
        if SOLAR_FORCE_ONE:
            new = pl.read_parquet(out / "a2_member" / "s0_2026-06.parquet")
            old = pl.read_parquet(common430.PRED_DIR / "a2_member" / "s0_2026-06.parquet")
            keys = ["time_series_id", "init_time", "time"]
            joined = new.join(old, on=keys, suffix="_old", validate="1:1")
            diffs = [
                joined.select((pl.col(q) - pl.col(f"{q}_old")).abs().max()).item()
                for q in ("q01", "q50", "q99")
            ]
            print("gate-solar: rows", new.height, old.height, "max abs diffs q01 q50 q99", diffs)


if __name__ == "__main__":
    main()
