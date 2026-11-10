"""Experiment 677 (adapted from experiment 668): score one ladder step against observed power.

uv run --no-sync python ml-experiments/00677-*/e00677_score.py step8w_with_injected_name 0 1 2

Writes CSV tables to `<experiment folder>/scores/<name>_s<seeds>/` (private). Systems: the model
in the file, the manual heuristic, and climatology, all scored on the rows where every system has
a forecast and the observed power is a finite, unflagged reading.
"""

import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00668_score_lib as lib
import numpy as np
import polars as pl

NAME: Final = "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
ROOT: Final[Path] = (
    Path(os.environ["E00677_ROOT"]) if os.environ.get("E00677_ROOT") else lib.FOLDER.parent / NAME
)
FIRST_MONTH_INDEX: Final = 2025 * 12 + 10
"""2025-11, the first scored month and the first refit."""
QCOLS: Final = tuple(lib.QCOLS)


def seed_mean(*, name: str, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return the ladder step's rows with each quantile averaged over `seeds` and sorted within the row."""
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    raw = pl.read_parquet(ROOT / "backtests" / f"{name}.parquet").filter(
        pl.col("seed").is_in(list(seeds))
    )
    assert set(raw["seed"].unique().to_list()) == set(seeds), (name, seeds)
    assert raw.group_by(keys).len()["len"].unique().to_list() == [len(seeds)], (
        "a seed lacks some rows"
    )
    mean = raw.group_by(keys).agg(pl.col(list(QCOLS)).mean()).sort(keys)
    q = np.sort(mean.select(list(QCOLS)).to_numpy().astype(np.float32), axis=1)
    return mean.with_columns([pl.Series(c, q[:, i]) for i, c in enumerate(QCOLS)])


SYSTEMS = ("Model", "Manual heuristic", "Climatology")
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring"}
SEASONS |= {6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


def season_label(*, origin: pl.Series) -> np.ndarray:
    """Return the meteorological season of each origin, splitting autumn across the year ends."""
    month = origin.dt.month().to_numpy()
    year = origin.dt.year().to_numpy()
    base = np.array([SEASONS[m] for m in month], dtype=object)
    autumn = base == "autumn"
    base[autumn] = np.where(year[autumn] == 2025, "autumn (Nov 2025)", "autumn (Sep to Nov 2026)")
    return base


def main(name: str, kind: str | None, seeds: tuple[int, ...] = (0, 1, 2)) -> None:
    """Score one file."""
    forecast = seed_mean(name=name, seeds=seeds)
    capacity = pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
    ids = sorted(forecast["time_series_id"].unique().to_list())
    cap = capacity.filter(pl.col("time_series_id").is_in(ids)).sort("time_series_id")["capacity"]
    cap = cap.to_numpy().astype(np.float64)
    grid = lib.cleaned_grid(ids=ids)
    row = forecast["time_series_id"].replace_strict({t: i for i, t in enumerate(ids)}).to_numpy()
    vidx = lib.grid_index(times=forecast["valid_time"])
    oidx = lib.grid_index(times=forecast["power_fcst_init_time"])
    if vidx.max() >= grid.shape[1]:  # valid times run past the last observation
        grid = np.pad(
            grid, ((0, 0), (0, int(vidx.max()) + 1 - grid.shape[1])), constant_values=np.nan
        )
    lead_h = (vidx - oidx) / 2.0
    y = grid[row, vidx].copy()
    span = int(re.search(r"_sp(\d+)$", name).group(1))
    month_index = (
        forecast["power_fcst_init_time"].dt.year().to_numpy().astype(int) * 12
        + forecast["power_fcst_init_time"].dt.month().to_numpy().astype(int)
        - 1
    )
    refit_index = month_index - (month_index - FIRST_MONTH_INDEX) % span
    """The climatology trains on data before the refit month, like the networks (not before each origin's month)."""
    month_start = pl.Series(
        [datetime(int(i) // 12, int(i) % 12 + 1, 1, tzinfo=UTC) for i in refit_index]
    )
    cutoff = lib.grid_index(times=month_start)
    scale = 100.0 / cap[row]
    q_model = forecast.select(lib.QCOLS).to_numpy().astype(np.float64) * scale[:, None]
    q_heur = lib.manual_heuristic(grid=grid, row=row, vidx=vidx, lead_h=lead_h) * scale[:, None]
    q_clim = lib.climatology(grid=grid, row=row, vidx=vidx, cutoff_idx=cutoff) * scale[:, None]
    y = y * scale
    ok = np.isfinite(y)
    for q in (q_model, q_heur, q_clim):
        ok &= np.isfinite(q).all(axis=1)
    out_dir = ROOT / "scores" / f"{name}_s{''.join(map(str, seeds))}"
    out_dir.mkdir(parents=True, exist_ok=True)
    month = forecast["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    counts = pl.DataFrame({"origin_month": month, "scored": ok, "no_truth": ~np.isfinite(y)})
    counts.group_by("origin_month").agg(
        rows=pl.len(), scored=pl.col("scored").sum(), no_truth=pl.col("no_truth").sum()
    ).sort("origin_month").write_csv(out_dir / "row_counts.csv")
    print(name, kind, "rows", len(y), "scored", int(ok.sum()), flush=True)

    per_system = {
        "Model": lib.losses(y=y, q=q_model),
        "Manual heuristic": lib.losses(y=y, q=np.where(ok[:, None], q_heur, 0.0)),
        "Climatology": lib.losses(y=y, q=np.where(ok[:, None], q_clim, 0.0)),
    }
    day = oidx // 48
    pl.DataFrame(
        {
            "series_index": row.astype(np.int8),
            "time_series_id": np.asarray(ids, dtype=np.int8)[row],
            "valid_idx": vidx.astype(np.int32),
            "origin_idx": oidx.astype(np.int32),
            "lead_h": lead_h.astype(np.float32),
            "ok": ok,
            **{
                f"{tag}_{name}": arr.astype(np.float32)
                for tag, d in per_system.items()
                for name, arr in d.items()
                if name in ("crps", "mae")
            },
        }
    ).write_parquet(out_dir / "rows.parquet")
    band = np.select(
        [(lo <= lead_h) & (lead_h < hi) for lo, hi in lib.LEAD_BANDS.values()],
        list(lib.LEAD_BANDS),
        default="days 10 to 14",
    )
    season = season_label(origin=forecast["power_fcst_init_time"])
    rows: list[dict] = []

    def add(mask: np.ndarray, **label: str) -> None:
        rows.extend(
            lib.summarise(
                mask=ok & mask, day=day, per_system=per_system, reference="Model", label=label
            )
        )

    everything = np.ones(len(y), dtype=bool)
    add(everything, slice="all rows", group="all")
    for b in lib.LEAD_BANDS:
        add(band == b, slice="lead band", group=b)
    lead_day = np.minimum(np.floor(lead_h / 24.0), 13).astype(int)
    for k in range(14):
        add(lead_day == k, slice="lead day", group=f"{k:02d}")
    for m in sorted(set(month)):
        add(month == m, slice="origin month", group=m)
    for s in dict.fromkeys(season):
        add(season == s, slice="season", group=s)
        for b in lib.LEAD_BANDS:
            add((season == s) & (band == b), slice="season and lead band", group=f"{s} | {b}")
    for i, tid in enumerate(ids):
        add(row == i, slice="series", group=str(tid))
    pl.DataFrame(rows).write_csv(out_dir / "scores.csv")
    print("wrote", out_dir, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], None, tuple(int(v) for v in sys.argv[2:]) or (0, 1, 2))
