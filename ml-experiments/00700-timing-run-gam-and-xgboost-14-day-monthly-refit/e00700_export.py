# ruff: noqa: E501
"""Experiment 700: write the forecasts of origin months 2025-11 to 2026-06 in experiment 677's layout.

The GAM is Primary demand research default 3.

    PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility \
        uv run --no-sync python ml-experiments/00700-*/e00700_export.py gam

Run with `DATA_PATH_INTERNAL` set to the real data folder. Reads the sandbox outputs of
`e00700_gam_*.py` and experiment 378's `e00378_arms*.py` run in the sandbox, and writes
`backtests/gam_year_sp1.parquet` under this experiment's private folder (point `E00677_ROOT` at it
when scoring).

**The forecast is Primary demand research default 3 as experiment 378 assembles it**, with the
stage 2 stack of experiment 377 beyond 240 hours:

- up to 168 hours after the native origin: the solar summary S (`x1`, `e00378_arms_low.py`);
- above 168 and up to 240 hours: S and the daylight heuristic blend (`d3`, `e00378_arms.py`);
- above 240 hours: stage 2 without S (`stack_d`, `e00377s3_stack.py`).

**The native origin is 09:00 UTC and the contract origin is 12:00 UTC**, as in experiment 667's
`contract_rows`: `power_fcst_init_time` is moved to 12:00 UTC, `nwp_init_time` is the 00:00 UTC
ENS run of the same day, and valid times from 12:00 UTC up to but not including 12:00 UTC 14 days
later are kept. The forecast uses nothing observed after 09:00 UTC, so the relabel moves the
origin to a later time only. Quantiles are points of effective capacity; the file holds megawatts
(or megavolt-amperes at the seven MVA-only Primaries), as experiment 677's files do. D15 has no
stage 2 rows before 2026-03, because experiment 377 requires 12 months of history, so the file
holds 15 Primaries.
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00700-timing-run-gam-and-xgboost-14-day-monthly-refit"
ROOT: Path = ML_EXPERIMENTS_DATA_DIR / NAME
SANDBOX: Path = ROOT / "sandbox" / "ml-experiments"
Q = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
START = datetime(2025, 11, 1, tzinfo=UTC)
END = datetime(2026, 7, 1, tzinfo=UTC)
SOURCE_OF = {
    **dict.fromkeys(("D5", "D9", "D10", "D11", "D14", "D16", "D18"), "control"),
    **dict.fromkeys(("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"), "mva"),
}
CONTRACT_HOUR, NATIVE_HOUR = 12, 9


def gam_rows() -> pl.DataFrame:
    """Return default 3's quantiles at the 15 Primaries, in points, with leads from the native origin."""
    w377 = next(SANDBOX.glob("00377-*")) / "work"
    w378 = next(SANDBOX.glob("00378-*")) / "work" / "arms"
    frames = []
    for site, source in SOURCE_OF.items():
        name = "stack_d_donor_D8" if site == "D8" else f"stack_d_{site}"
        base = pl.read_parquet(w377 / source / "arms" / "a0" / "dp_preds_s2" / f"{name}.parquet")
        # A Primary has no forecast before its first month in experiment 377's stored file (12 months
        # of history are required), and experiment 378's solar summary files stop there too.
        first_init = pl.read_parquet(
            ML_EXPERIMENTS_DATA_DIR
            / w377.relative_to(SANDBOX).parts[0]
            / "work"
            / source
            / "arms"
            / "a0"
            / "dp_preds_s2"
            / f"{name}.parquet",
            columns=["power_fcst_init_time"],
        )["power_fcst_init_time"].min()
        assert isinstance(first_init, datetime), name
        first = first_init.replace(day=1, hour=0, minute=0)
        base = base.filter(
            (pl.col("power_fcst_init_time") >= first)
            & (pl.col("seed") == 0)
            & (pl.col("power_fcst_init_time") >= START)
            & (pl.col("power_fcst_init_time") < END)
        )
        keys = ["power_fcst_init_time", "valid_time"]
        low = pl.read_parquet(w378 / f"low_lead_{source}.parquet").filter(pl.col("site") == site)
        long = pl.read_parquet(w378 / f"long_lead_{source}.parquet").filter(pl.col("site") == site)
        joined = base.join(
            low.select(*keys, *[pl.col(f"x1_{q}").alias(f"low_{q}") for q in Q]),
            on=keys,
            how="left",
        ).join(
            long.select(*keys, *[pl.col(f"d3_{q}").alias(f"long_{q}") for q in Q]),
            on=keys,
            how="left",
        )
        short = pl.col("lead_hours") <= 168.0
        mid = pl.col("lead_hours") <= 240.0
        out = joined.select(
            pl.lit(site).alias("site"),
            *keys,
            "lead_hours",
            *[
                pl.when(short)
                .then(pl.col(f"low_{q}"))
                .when(mid)
                .then(pl.col(f"long_{q}"))
                .otherwise(pl.col(q))
                .alias(q)
                for q in Q
            ],
        )
        assert out.select(Q).null_count().sum_horizontal().item() == 0, site
        frames.append(out)
    return pl.concat(frames)


def to_contract(*, frame: pl.DataFrame, unit: str) -> pl.DataFrame:
    """Relabel the origin as 12:00 UTC, keep 14 days from it, and convert to megawatts."""
    series = adapter.series_table().select("time_series_id", "label", "capacity_mw")
    assert (frame["power_fcst_init_time"].dt.hour() == NATIVE_HOUR).all()
    scale = {"points": 0.01, "fraction": 1.0}[unit]
    origin = pl.col("power_fcst_init_time") + timedelta(hours=CONTRACT_HOUR - NATIVE_HOUR)
    out = (
        frame.join(series, left_on="site", right_on="label")
        .with_columns(contract_origin=origin)
        .filter(
            (pl.col("valid_time") >= pl.col("contract_origin"))
            & (pl.col("valid_time") < pl.col("contract_origin") + timedelta(days=14))
        )
        .select(
            pl.col("time_series_id").cast(pl.Int8),
            "valid_time",
            pl.col("contract_origin").dt.truncate("1d").alias("nwp_init_time"),
            pl.col("contract_origin").alias("power_fcst_init_time"),
            *[
                (pl.col(q).cast(pl.Float64) * pl.col("capacity_mw") * scale).cast(pl.Float32)
                for q in Q
            ],
        )
        .sort("time_series_id", "power_fcst_init_time", "valid_time")
    )
    q = out.select(Q).to_numpy()
    assert (np.diff(q, axis=1) >= -1e-6).all(), "quantiles cross"
    sorted_q = np.sort(q, axis=1)
    return out.with_columns([pl.Series(c, sorted_q[:, i]) for i, c in enumerate(Q)])


def write(*, frame: pl.DataFrame, name: str, step: str) -> None:
    """Write a file in experiment 677's layout (seed 0, `NRA` rows)."""
    out = frame.with_columns(
        seed=pl.lit(0, dtype=pl.Int64),
        step=pl.lit(step),
        version=pl.lit("research"),
        NRA_or_prevailing=pl.lit("NRA"),
    )
    path = ROOT / "backtests" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    per = out.group_by("time_series_id", "power_fcst_init_time").len()["len"].value_counts()
    print(
        name,
        out.height,
        "rows; Primaries",
        out["time_series_id"].n_unique(),
        "; rows per origin",
        per.to_dicts(),
    )


def xgb_rows(*, arm: str, root: Path) -> pl.DataFrame:
    """Return an arm of experiment 87 for the origins of 2026-01, with quantiles as fractions of capacity."""
    path = next(root.glob("00087-*")) / "work" / "arms" / f"{arm}.parquet"
    return (
        pl.scan_parquet(path)
        .filter(
            (pl.col("power_fcst_init_time") >= START)
            & (pl.col("power_fcst_init_time") < END)
            & (pl.col("group") == "Primary")
        )
        .select("site", "power_fcst_init_time", "valid_time", "lead_hours", *Q)
        .collect()
    )


def compare_with_stored(*, arm: str) -> float:
    """Return the largest absolute difference of q50 (fraction of capacity) from the stored arm."""
    new = xgb_rows(arm=arm, root=SANDBOX)
    old = xgb_rows(arm=arm, root=ML_EXPERIMENTS_DATA_DIR)
    keys = ["site", "power_fcst_init_time", "valid_time"]
    both = new.join(old, on=keys, suffix="_old")
    print(arm, "rows new", new.height, "stored", old.height, "joined", both.height)
    return float(np.nanmax(np.abs(both["q50"].to_numpy() - both["q50_old"].to_numpy())))


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "gam":
        write(frame=to_contract(frame=gam_rows(), unit="points"), name="gam_year_sp1", step="gam")
    elif which == "xgb":
        for arm, name in (
            ("p_s6_bg2", "xgb_bg2_year_sp1"),
            ("p_a3_monthly_in_weekly_slot", "xgb_monthly_year_sp1"),
        ):
            write(
                frame=to_contract(
                    frame=xgb_rows(arm=arm, root=ML_EXPERIMENTS_DATA_DIR), unit="fraction"
                ),
                name=name,
                step=arm,
            )
