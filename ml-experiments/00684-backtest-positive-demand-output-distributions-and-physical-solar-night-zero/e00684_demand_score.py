"""Experiment 684: score the demand arms against experiment 667's stored forecasts and its clamp.

Arms: `control_667` (experiment 667's stored NRA rows), `control_667_clamped` (the same rows with
every quantile of the series raised to at least zero, the post hoc clamp), and one arm per output
family (`gauss`, `tgauss`, `gamma`) read from this experiment's fits. Series: the four Primaries
with a positive family plus every other Primary as a check. Scored rows, observed power, capacity
division, and the four lead bands are experiment 667's (`e00667_score.score_frame`). Output:
`score/demand_arms.csv` with one row per arm, fold month, series, and lead band.

    PYTHONPATH=ml-experiments/00667-*:ml-experiments/00684-* uv run --no-sync python \
        ml-experiments/00684-*/e00684_demand_score.py gauss tgauss gamma
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00510_common as c510  # ty: ignore[unresolved-import]
import e00667_export as exp  # ty: ignore[unresolved-import]
import e00667_extend as ext  # ty: ignore[unresolved-import]
import e00667_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

MONTHS = tuple(os.environ.get("E00684_MONTHS", "2026-02 2026-05").split())
ROOT_667 = exp.ROOT
ROOT_684 = (
    ROOT_667.parent
    / "00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero"
)
TAUS = sc.TAUS
MVA_IDS = (3, 9, 11, 14, 15, 24, 25, 26)


def arm_frame(*, arm: str, month: str) -> pl.DataFrame:
    """Return an arm's contract rows (NRA output) for one fold month."""
    if arm.startswith("control_667"):
        frame = pl.read_parquet(ROOT_667 / "exports" / f"NRA_{month}.parquet")
        if arm.endswith("clamped"):
            frame = frame.with_columns(
                pl.when(pl.col("time_series_id").is_in(MVA_IDS))
                .then(pl.col(q).clip(lower_bound=0.0))
                .otherwise(pl.col(q))
                .alias(q)
                for q in exp.QCOLS
            )
        return frame
    exp.ROOT = ROOT_684 / f"demand_{arm}"
    try:
        return exp.nra_month(month=month)
    finally:
        exp.ROOT = ROOT_667


def main() -> None:
    """Score every arm named on the command line, plus the controls."""
    ext.redirect()
    grid_start, step = c510.prep422.GRID_START_US, c510.prep422.STEP_US
    clean = sc.observed_stack(grid_start_us=grid_start, step_us=step)
    arms = ["control_667", "control_667_clamped", *sys.argv[1:]]
    rows = []
    for arm in arms:
        for month in MONTHS:
            frame = arm_frame(arm=arm, month=month)
            s = sc.score_frame(frame=frame, clean=clean, grid_start_us=grid_start, step_us=step)
            q = frame.select(exp.QCOLS).to_numpy().astype(np.float64)
            capacity = frame.join(
                exp.capacity_mw().with_columns(pl.col("time_series_id").cast(pl.Int8)),
                on="time_series_id",
            )["capacity_mw"].to_numpy()
            y = s["observed"].to_numpy()
            qpu = q / capacity[:, None]
            d = y[:, None] - qpu
            pin = 100.0 * np.maximum(TAUS * d, (TAUS - 1.0) * d)
            tail = pl.DataFrame(
                {
                    "pin01": pin[:, 0],
                    "pin05": pin[:, 1],
                    "pin95": pin[:, 9],
                    "pin99": pin[:, 10],
                    "below01": (y < qpu[:, 0]).astype(float),
                    "below05": (y < qpu[:, 1]).astype(float),
                    "above95": (y > qpu[:, 9]).astype(float),
                    "above99": (y > qpu[:, 10]).astype(float),
                    "q01_nonpos": (q[:, 0] <= 0).astype(float),
                }
            )
            s = s.hstack(tail).with_columns(arm=pl.lit(arm), month=pl.lit(month))
            rows.append(s.filter(pl.col("observed").is_finite()))
    s = pl.concat(rows)
    band = (
        pl.when(pl.col("lead_days") < 1)
        .then(pl.lit("0-1 d"))
        .when(pl.col("lead_days") < 3)
        .then(pl.lit("1-3 d"))
        .when(pl.col("lead_days") < 7)
        .then(pl.lit("3-7 d"))
        .otherwise(pl.lit("7-14 d"))
        .alias("band")
    )
    agg = [
        pl.len().alias("rows"),
        *(
            pl.col(c).mean()
            for c in (
                "crps_pts",
                "abs_err_pts",
                "in90",
                "in50",
                "pin01",
                "pin05",
                "pin95",
                "pin99",
                "below01",
                "below05",
                "above95",
                "above99",
                "q01_nonpos",
            )
        ),
    ]
    table = (
        s.with_columns(band)
        .group_by("arm", "month", "time_series_id", "band")
        .agg(agg)
        .sort("arm", "month", "time_series_id", "band")
    )
    out = ROOT_684 / "score"
    out.mkdir(exist_ok=True)
    table.write_csv(out / "demand_arms.csv")
    focus = table.filter(pl.col("time_series_id").is_in([14, 15, 26, 17]))
    print(
        focus.group_by("arm", "time_series_id")
        .agg(pl.col("crps_pts").mean(), pl.col("in90").mean(), pl.col("q01_nonpos").mean())
        .sort("time_series_id", "arm")
    )
    _ = Path


if __name__ == "__main__":
    main()
