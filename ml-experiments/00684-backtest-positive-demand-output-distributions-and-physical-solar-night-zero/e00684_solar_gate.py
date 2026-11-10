r"""Experiment 684, solar arm a: gate the stored experiment 666 networks' quantiles at night.

The networks are unchanged. Every quantile is set to zero where the sun's elevation at the middle
of the half-hour (valid time minus 15 minutes, the rule the scores use) is at or below zero. The
control is experiment 666's own export (`e00666_export.solar_frame`, zero only where the 3-hour or
6-hour period has no daylight). Both are scored at Solar B and Solar E on the origins of the
2026-02-01 refit (origins 2026-02-01 to 2026-04-30), in four sun classes: night (elevation at or
below 0), dawn and dusk (above 0 and under 5 degrees), and daylight (5 degrees or more), and all.

    export DATA_PATH_INTERNAL=<path to the internal data folder>
    PYTHONPATH=ml-experiments/00666-*:ml-experiments/00684-* \
        uv run --no-sync python ml-experiments/00684-*/e00684_solar_gate.py
"""

import os
from datetime import UTC, datetime
from typing import Final

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00666_common as common  # ty: ignore[unresolved-import]
import e00666_export as export  # ty: ignore[unresolved-import]
import e00666_score as score666  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.pv_dataset import pv_sites
from studies.solar import zenith

REFIT_START: Final = datetime(2026, 2, 1, tzinfo=UTC)
REFIT_STOP: Final = datetime(2026, 5, 1, tzinfo=UTC)
FARMS: Final = ("Solar B", "Solar E")
"""Anonymised labels in the order of `pv_sites()` sorted by series id (A to F)."""
DAWN_DEGREES: Final = 5.0


def labelled_sites() -> pl.DataFrame:
    """Return the two farms' site rows with their anonymised label."""
    sites = pv_sites().sort("time_series_id").with_row_index("rank")
    labels = pl.DataFrame(
        {"rank": [1, 4], "farm": list(FARMS)}, schema={"rank": pl.UInt32, "farm": pl.String}
    )
    return sites.join(labels, on="rank")


def elevation(*, frame: pl.DataFrame, sites: pl.DataFrame) -> np.ndarray:
    """Return the sun's elevation in degrees at the middle of each row's half-hour."""
    out = np.full(frame.height, np.nan)
    ids = frame["time_series_id"].to_numpy()
    for rec in sites.iter_rows(named=True):
        sel = np.flatnonzero(ids == rec["time_series_id"])
        if len(sel) == 0:
            continue
        mid = frame["valid_time"].gather(sel.tolist()).dt.offset_by("-15m")
        out[sel] = 90.0 - np.asarray(
            zenith(stamps=mid, latitude=rec["latitude"], longitude=rec["longitude"])
        )
    return out


def sun_class(elev: np.ndarray) -> np.ndarray:
    """Return `night`, `dawn_dusk`, or `daylight` for each elevation."""
    return np.where(elev <= 0.0, "night", np.where(elev < DAWN_DEGREES, "dawn_dusk", "daylight"))


def main() -> None:
    """Score the control and the gate and print the table."""
    sites = labelled_sites()
    control = export.solar_frame().filter(
        pl.col("time_series_id").is_in(sites["time_series_id"].to_list())
        & (pl.col("nwp_init_time") >= REFIT_START)
        & (pl.col("nwp_init_time") < REFIT_STOP)
    )
    elev = elevation(frame=control, sites=sites)
    night = elev <= 0.0
    keep = pl.Series("night", night)
    gate = control.with_columns(
        pl.when(keep).then(0.0).otherwise(pl.col(q)).cast(pl.Float32).alias(q)
        for q in common.QUANTILES
    )
    control = control.with_columns(pl.col(q).cast(pl.Float32) for q in common.QUANTILES)
    classes = pl.Series("sun_class", sun_class(elev))
    label = sites.select("time_series_id", "farm").with_columns(
        pl.col("time_series_id").cast(pl.Int8)
    )
    rows = []
    for name, frame in (("control_666", control), ("gate", gate)):
        prepared = score666.prepare(
            forecast=frame.with_columns(classes), sites=pv_sites(), mask=False
        ).join(label, on="time_series_id")
        scored = score666.losses(frame=prepared).hstack(
            prepared.select("sun_class", "farm", "days", "actual")
        )
        nonzero_night = int(
            (frame.filter(keep).select(pl.any_horizontal(pl.col(q) != 0 for q in common.QUANTILES)))
            .to_series()
            .sum()
        )
        print(
            f"{name}: night rows with a non-zero quantile {nonzero_night:,} of {int(night.sum()):,}"
        )
        for by in (["farm", "sun_class"], ["sun_class"]):
            table = scored.group_by(by).agg(
                rows=pl.len(),
                score=pl.col("score").mean(),
                mae_median=pl.col("mae_median").mean(),
                inside_q05_q95=pl.col("inside_q05_q95").mean(),
                above_q99=pl.col("above_q99").mean(),
                below_q01=pl.col("below_q01").mean(),
            )
            rows.append(table.with_columns(arm=pl.lit(name)))
        score666.SCORES_DIR.parent.mkdir(parents=True, exist_ok=True)
    result = pl.concat(rows, how="diagonal").sort("arm", "farm", "sun_class")
    out = (
        common.DATA_DIR.parent
        / "00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero"
    )
    out.mkdir(exist_ok=True)
    result.write_csv(out / "solar_gate_scores.csv")
    print(result)


if __name__ == "__main__":
    main()
