"""Experiment 401, Job 1, synthetic arm: matched-event injection into clean runs, b0 boosters.

Prediction only, as in experiment 395's Part E (copied: that experiment is unmerged). For each
Primary and each monthly b0 booster of experiment 391 (refit months October, December, February,
April, and June of the scored year), one clean forecast run (the 15th of the refit month) is
predicted again with a step of each real event's corrected signed depth at that Primary (as a
fraction of capacity) added to the `k` newest lagged-power inputs (k = 1, 2, 3). The actual reading
is stored unshifted; the observed share (`(q50 shifted - actual) / depth`) is the Part H share for
an actual reading shifted by the depth. Clean rows: finite actual, not flagged by experiment 391's
event masks, no lag flagged; lead under 240 hours. Output per Primary: one row per (event, k, month,
valid time) with the median shift and both share definitions.

```bash
PYTHONPATH=<as in e00401_score.py> DATA_PATH_INTERNAL=... \
    systemd-run --user --scope -p MemoryMax=6G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00401-*/e00401_sweep.py run [D5 ...]
```
"""
# ruff: noqa: E501

import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00391_fit as f391  # ty: ignore[unresolved-import]
import e00401_arms as arms
import e00401_events as ev
import numpy as np
import polars as pl
import xgboost as xgb

OUT: Final[Path] = arms.WORK / "sweep"
BOOSTERS: Final[Path] = f391.BOOSTER_DIR / "b0"
RUN_DAY: Final[int] = 15
MONTHS: Final[tuple[int, ...]] = (10, 12, 14, 16, 18)
NTHREAD: Final[int] = 2
MAX_LEAD_H: Final[float] = 240.0
Q50: Final[int] = common.LEVEL_COLUMNS.index("q50")


def predict_q50(*, model: xgb.Booster, matrix: np.ndarray, members: int) -> np.ndarray:
    """Return the mixture's median per valid time for a matrix of member rows."""
    raw = model.predict(xgb.DMatrix(matrix))
    cube = raw.reshape(-1, members, len(common.LEVELS))
    return common.mixture_quantiles(member_quantiles=cube)[:, Q50]


def run_site(*, site: str, events: pl.DataFrame) -> pl.DataFrame:
    """Return the sweep rows for one Primary."""
    series, array_by_site = fd._series_and_arrays()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    columns = f391.ARMS["b0"].columns()
    lag_idx = [columns.index(c) for c in common.LAG_COLUMNS]
    masks = f391.event_masks()
    power = array_by_site[site].power
    mine = events.filter(pl.col("label") == site)
    depths = {
        r["event_id"]: r["corrected_depth_mw"] / r["capacity_mw"]
        for r in mine.iter_rows(named=True)
    }
    out = []
    for month in MONTHS:
        year, mon = (2025, month) if month <= 12 else (2026, month - 12)
        refit = datetime(year, mon, 1, tzinfo=UTC)
        origin = datetime(year, mon, RUN_DAY, 9, tzinfo=UTC)
        model = xgb.Booster()
        model.load_model(BOOSTERS / f"{site}_{refit:%Y%m%d}.ubj")
        model.set_param({"nthread": NTHREAD, "device": "cpu"})
        frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        prepared = fd.prepared_run(
            run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
            row=row, arrays=array_by_site[site], a7=False,
        )  # fmt: skip
        keys = prepared.select(*common.KEYS).unique(maintain_order=True)
        members = prepared["ensemble_member"].n_unique()
        target = np.asarray(common.grid_index(times=keys["valid_time"]))
        origin_idx = np.asarray(common.grid_index(times=keys["power_fcst_init_time"]))
        actual = common.take(values=power, index=target)
        lead = (
            keys["valid_time"] - keys["power_fcst_init_time"]
        ).dt.total_minutes().to_numpy() / 60
        clean = (
            np.isfinite(actual)
            & ~masks[site][target]
            & ~f391.lag_flags(mask=masks[site], target=target, origin=origin_idx).any(axis=1)
            & (lead < MAX_LEAD_H)
        )
        matrix = prepared.select(columns).to_numpy().astype(np.float32)
        base = predict_q50(model=model, matrix=matrix, members=members)
        for event_id, depth in depths.items():
            for k in (0, 1, 2, 3):
                if k == 0:
                    shifted = base
                else:
                    moved = matrix.copy()
                    for i in range(k):
                        moved[:, lag_idx[i]] += depth
                    shifted = predict_q50(model=model, matrix=moved, members=members)
                out.append(
                    pl.DataFrame(
                        {
                            "label": site,
                            "event_id": event_id,
                            "k": k,
                            "month": mon + 12 * (year - 2025),
                            "valid_time": keys["valid_time"].filter(pl.Series(clean)),
                            "lead_hours": lead[clean],
                            "depth": depth,
                            "q50_base": base[clean],
                            "q50_shifted": shifted[clean],
                            "actual": actual[clean],
                        }
                    )
                )
        print(site, refit.date(), "done", flush=True)
    return pl.concat(out)


def run(*, sites: list[str]) -> None:
    """Run the sweep for the named Primaries, skipping those already written."""
    OUT.mkdir(parents=True, exist_ok=True)
    events = ev.events_with_depths()
    for site in sites or sorted(events["label"].unique().to_list()):
        target = OUT / f"{site}.parquet"
        if target.exists():
            continue
        start = time.time()
        run_site(site=site, events=events).write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(site, "written in", round(time.time() - start), "s", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    run(sites=sys.argv[2:])
