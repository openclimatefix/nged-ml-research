# ruff: noqa: E501
"""Refit wind default 2's mixture-density head at day 0 and day 14 of each development month.

Wind default 2 is experiment 347's arm y2: the frozen Stage 3 plant run on the 51 ensemble members,
then one pooled three-component mixture-density head per refit, with effective capacity at the 99.9th
percentile of the farm's earlier power (experiment 461's rule, recomputed at each refit date).

For a refit at `R` the head trains on daily runs with `init < R - 62 days` and valid time before
that date, and stops early on the runs between that date and `R` (the same 62-day stopping window as
experiment 347's 2025-04-16 to 2025-06-17). It forecasts the runs from `R` to the next refit date.
The target and the plant summaries (except the saturation share) are divided by each farm's ratio of
new to old capacity, and the quantiles are stored in MW.

`--gate` refits experiment 347's own head (cutoff 2025-04-16, old capacity, seed 0) and compares
its scored quantiles with experiment 347's stored part file.

    source <env with PYTHONPATH of the experiments listed in e00615_wind_inputs.py>
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00615-*/e00615_wind_fit.py [--gate] [--seeds 0 1]
"""

import argparse
import json
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Final

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00347_s4_data as data347  # ty: ignore[unresolved-import]
import e00347_s4_run as run347  # ty: ignore[unresolved-import]
import e00461_common as common461  # ty: ignore[unresolved-import]
import e00615_common as common
import e00615_wind_inputs as inputs
import numpy as np
import polars as pl

OUT_DIR: Final = common.DATA_DIR / "wind_default2"
STOP_WINDOW_DAYS: Final[int] = 62
SCORING_CAPACITY_CUTOFF: Final[datetime] = datetime(2025, 10, 1, tzinfo=UTC)
LABEL: Final[dict[str, str]] = common461.WIND_LABEL
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]


def refit_dates() -> list[datetime]:
    """Return day 0 and day 14 of each development month (month start and the 15th)."""
    out = []
    for month in common.DEV_MONTHS:
        year, mon = (int(x) for x in month.split("-"))
        out += [datetime(year, mon, 1, tzinfo=UTC), datetime(year, mon, 15, tzinfo=UTC)]
    return out


def month_end(*, refit: datetime) -> datetime:
    """Return the first day of the month after `refit`."""
    return datetime(refit.year + (refit.month == 12), refit.month % 12 + 1, 1, tzinfo=UTC)


def all_cases() -> tuple[pl.DataFrame, np.ndarray]:
    """Return every daily case (hindcast, then the rebuilt runs) and its (n, 6) plant summary."""
    hind = data347.hindcast_frame().select("site", *KEYS, "lead_hours", "actual_norm", "masked")
    hind_summary = data347.summaries(arm_plant="y2", set_name="hindcast")
    new = pl.concat(
        [
            pl.read_parquet(common.DATA_DIR / "wind_inputs" / f"cases_{f}.parquet")
            for f in data347.FARMS
        ]
    )
    new_summary = new.select(inputs.SUMMARY_NAMES).to_numpy()
    new = new.select("site", *KEYS, "lead_hours", "actual_norm", "masked")
    frame = pl.concat([hind, new])
    summary = np.concatenate([hind_summary, new_summary]).astype(np.float64)
    assert len(frame) == len(summary)
    assert np.isfinite(summary).all(), "a plant summary is not finite"
    return frame, summary


def coverage_checks(*, frame: pl.DataFrame) -> None:
    """Print and assert the coverage of the idle mask and the capacity series."""
    from e00303_common import idle_mask, site_table  # ty: ignore[unresolved-import]

    mask_end = idle_mask()["valid_time"].max()
    series = pl.read_parquet(common461.SERIES_PATH)
    ends = series.group_by("site").agg(pl.col("time").max()).sort("site")
    print("idle mask last valid time:", mask_end, flush=True)
    print(ends, flush=True)
    assert mask_end > datetime(2026, 6, 15, tzinfo=UTC)
    assert (
        ends.filter(pl.col("site").is_in(list(LABEL.values())))["time"]
        > datetime(2026, 6, 15, tzinfo=UTC)
    ).all()
    sites = site_table()
    for farm in data347.FARMS:
        old = common461.capacity(site=LABEL[farm], cutoff=datetime(2025, 10, 1, tzinfo=UTC))[
            "old_mw"
        ]
        stored = float(sites.filter(pl.col("site") == farm)["effective_capacity_mw"][0])
        assert abs(old - stored) < 1e-3, (farm, old, stored)
    runs = frame["power_fcst_init_time"].dt.date().n_unique()
    print("distinct run days in the case table:", runs, flush=True)


def rescale(
    *, frame: pl.DataFrame, summary: np.ndarray, ratio: dict[str, float]
) -> tuple[pl.DataFrame, np.ndarray]:
    """Divide the target and the plant summaries (not the saturation share) by each farm's ratio."""
    per_row = frame["site"].replace_strict(ratio, return_dtype=pl.Float64).to_numpy()
    out = summary.copy()
    out[:, :5] /= per_row[:, None]
    return frame.with_columns(actual_norm=pl.col("actual_norm") / pl.Series(per_row)), out


def gate() -> None:
    """Refit experiment 347's head with its own inputs and compare with its stored part file."""
    hindcast = data347.hindcast_frame()
    scored = data347.scored_frame()
    summary_h = data347.summaries(arm_plant="y2", set_name="hindcast")
    summary_s = data347.summaries(arm_plant="y2", set_name="scored")
    out = run347.train_and_predict(
        hindcast=hindcast,
        summary_h=summary_h,
        rows=np.ones(len(hindcast), dtype=bool),
        cutoff=data347.MAIN_CUTOFF,
        valid_limit=None,
        kind="full",
        farm_columns=True,
        targets={"scored": (scored, summary_s)},
        seed=0,
    )["scored"]
    stored = np.load(data347.PARTS_DIR / "y2_h0.npy")
    diff = np.abs(out - stored)
    print(f"gate: max abs difference {diff.max():.5f}, mean {diff.mean():.6f}, rows {len(stored)}")


def main() -> None:
    """Fit every refit and seed, skipping finished parts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    if args.gate:
        gate()
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, summary = all_cases()
    coverage_checks(frame=frame)
    scoring_cap = {
        f: common461.capacity(site=LABEL[f], cutoff=SCORING_CAPACITY_CUTOFF) for f in data347.FARMS
    }
    (OUT_DIR / "scoring_capacity.json").write_text(json.dumps(scoring_cap, indent=1) + "\n")
    dates = refit_dates()
    started = time.time()
    for index, refit in enumerate(dates):
        upper = (
            dates[index + 1]
            if index + 1 < len(dates) and dates[index + 1].month == refit.month
            else month_end(refit=refit)
        )
        cap = {f: common461.capacity(site=LABEL[f], cutoff=refit) for f in data347.FARMS}
        ratio = {f: float(cap[f]["ratio"]) for f in data347.FARMS}
        before = (frame["power_fcst_init_time"] < refit).to_numpy()
        target = (
            (frame["power_fcst_init_time"] >= refit) & (frame["power_fcst_init_time"] < upper)
        ).to_numpy()
        train_frame, train_summary = rescale(
            frame=frame.filter(pl.Series(before)), summary=summary[before], ratio=ratio
        )
        usable = train_frame["actual_norm"].is_finite().fill_null(value=False).to_numpy()
        train_frame = train_frame.filter(pl.Series(usable))
        train_summary = train_summary[usable]
        target_frame, target_summary = rescale(
            frame=frame.filter(pl.Series(target)), summary=summary[target], ratio=ratio
        )
        for seed in args.seeds:
            path = OUT_DIR / f"refit={refit:%Y-%m-%d}_seed={seed}.parquet"
            if path.exists():
                continue
            out = run347.train_and_predict(
                hindcast=train_frame,
                summary_h=train_summary,
                rows=np.ones(len(train_frame), dtype=bool),
                cutoff=refit - timedelta(days=STOP_WINDOW_DAYS),
                valid_limit=refit,
                kind="full",
                farm_columns=True,
                targets={"fc": (target_frame, target_summary)},
                seed=seed,
            )["fc"]
            mw = (
                out
                * target_frame["site"]
                .replace_strict({f: float(cap[f]["new_mw"]) for f in cap}, return_dtype=pl.Float64)
                .to_numpy()[:, None]
            )
            table = target_frame.select("site", *KEYS, "lead_hours", "masked").with_columns(
                refit=pl.lit(refit),
                seed=pl.lit(seed),
                **{f"q{i}": pl.Series(mw[:, i]) for i in range(11)},
            )
            table.write_parquet(path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)
            print(
                f"refit {refit:%Y-%m-%d} seed {seed}: {table.height:,} rows, ratio {ratio} {time.time() - started:.0f} s",
                flush=True,
            )


if __name__ == "__main__":
    main()
