# ruff: noqa: E501
"""Experiment 702: re-run the read-only checks whose numbers the write-up quotes, and save the output.

    PYTHONPATH=ml-experiments/00668-documentation-of-the-one-year-backtest-files-models-training-and-performance \
        uv run --no-sync python ml-experiments/00702-*/e00702_evidence.py

Writes `checks/evidence.txt` in the experiment's private data folder. It fits nothing. It reads the
first run's files under `pilot_2026-05/`, experiment 677's stored forecasts and bank file, the phase
logs, and the cleaned-power Delta table, and prints:

- the refit match: CRPS of the pilot refit of the global arm (seed 0) and of experiment 677's stored
  seeds 0 to 2, on 2026-05 at D14, D2, and D15, leads of 0 to 240 hours;
- the bank match: the largest differences between this experiment's bank run on experiment 677's
  stored networks and experiment 677's own bank file;
- the pre-training row count of the global arm in each fold;
- the first kept reading of D15 and D18 in the cleaned-power table, and the drop reasons before it.
"""

import contextlib
import io
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / (
    "00702-per-substation-versus-global-training-of-the-modular-demand-network"
)
ROOT_677: Final[Path] = ML_EXPERIMENTS_DATA_DIR / (
    "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
)
SITE_MAP: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00521-*")) / "site_map.csv"
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
PILOT_MONTH: Final[str] = "2026-05"
PILOT_LABELS: Final[tuple[str, ...]] = ("D14", "D2", "D15")
FOLDS: Final[tuple[str, ...]] = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07",
)  # fmt: skip


def label_ids(*, labels: tuple[str, ...]) -> dict[str, int]:
    """Return the time-series identifier of each Primary label."""
    site = pl.read_csv(SITE_MAP).filter(pl.col("label").is_in(labels))
    return dict(zip(site["label"].to_list(), site["time_series_id"].to_list(), strict=True))


def refit_match() -> None:
    """Print the CRPS of the pilot refit (seed 0) and of the stored seeds 0 to 2 on the same rows."""
    ids = sorted(label_ids(labels=PILOT_LABELS).values())
    month = pl.col("power_fcst_init_time").dt.strftime("%Y-%m")
    frames = {"refit seed 0": pl.read_parquet(ROOT / "pilot_2026-05" / "backtests" / "global_pilot.parquet").filter(pl.col("seed") == 0)}  # fmt: skip
    stored = pl.read_parquet(ROOT_677 / "backtests" / "step5w_with_injected_sp1.parquet")
    for s in (0, 1, 2):
        frames[f"stored seed {s}"] = stored.filter(pl.col("seed") == s)
    frames = {
        n: f.filter(month == PILOT_MONTH, pl.col("time_series_id").is_in(ids))
        for n, f in frames.items()
    }
    keys = None
    for f in frames.values():
        keys = f.select(KEYS) if keys is None else keys.join(f.select(KEYS), on=KEYS)
    assert keys is not None
    keys = keys.sort(KEYS)
    grid = lib.cleaned_grid(ids=ids)
    capacity = (
        pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
        .filter(pl.col("time_series_id").is_in(ids))
        .sort("time_series_id")["capacity"]
        .to_numpy()
    )
    row = keys["time_series_id"].replace_strict({t: i for i, t in enumerate(ids)}).to_numpy()
    vidx = lib.grid_index(times=keys["valid_time"])
    lead = (vidx - lib.grid_index(times=keys["power_fcst_init_time"])) / 2.0
    scale = 100.0 / capacity[row]
    y = grid[row, vidx] * scale
    cut = lib.grid_index(times=pl.Series(keys["power_fcst_init_time"].dt.truncate("1mo")))
    heur = lib.manual_heuristic(grid=grid, row=row, vidx=vidx, lead_h=lead)
    clim = lib.climatology(grid=grid, row=row, vidx=vidx, cutoff_idx=cut)
    ok = (
        np.isfinite(y)
        & np.isfinite(heur).all(axis=1)
        & np.isfinite(clim).all(axis=1)
        & (lead < 240.0)
    )
    print(f"refit match, {PILOT_MONTH}, {', '.join(PILOT_LABELS)}, {int(ok.sum())} rows")
    for name, f in frames.items():
        q = keys.join(f, on=KEYS, how="left").select(lib.QCOLS).to_numpy() * scale[:, None]
        print(f"  {name}: CRPS {lib.losses(y=y[ok], q=q[ok])['crps'].mean():.5f}")


def bank_match() -> None:
    """Print the largest differences between this experiment's bank run and experiment 677's file."""
    name = f"{PILOT_MONTH}_d0_s0.parquet"
    mine = pl.read_parquet(
        ROOT
        / "pilot_2026-05"
        / "arms"
        / "global_stored"
        / PILOT_MONTH
        / "pooled"
        / "work"
        / "bank"
        / name
    )
    theirs = pl.read_parquet(ROOT_677 / f"P0_D50_{PILOT_MONTH}_5w_sp1" / "work" / "bank" / name)
    mine, theirs = (f.with_columns(pl.col("time_series_id").cast(pl.Int64)) for f in (mine, theirs))
    joined = mine.join(theirs, on=KEYS, how="inner", suffix="_677")
    print(
        f"bank match, {PILOT_MONTH}, seed 0: rows {mine.height} and {theirs.height}, joined {joined.height}"
    )
    for col in ("median_base", "median_bank", "lag_shift"):
        diff = joined.select((pl.col(col) - pl.col(f"{col}_677")).abs().max()).item()
        print(f"  largest difference in {col}: {diff}")


def pretraining_rows() -> None:
    """Print the pre-training row count of the global arm in each fold, from the fold's log."""
    print("pre-training rows of the global arm")
    for fold in FOLDS:
        text = (ROOT / "logs" / f"global_{fold}_pretrain.log").read_text()
        match = re.search(r"pretraining rows ([\d,]+)", text)
        print(f"  {fold}: {match.group(1) if match else 'not in the log'}")


def cleaned_starts() -> None:
    """Print the first kept reading of D15 and D18, and the drop reasons of the readings before it."""
    print("cleaned-power table: first kept reading, and the drop reasons before it")
    scan = pl.scan_delta(str(REPO_DATA_DIR / "NGED" / "cleaned_power_time_series.delta"))
    for label, tid in label_ids(labels=("D15", "D18")).items():
        part = scan.filter(pl.col("time_series_id") == tid).select("time", "drop_reason").collect()
        first = part.filter(pl.col("drop_reason").is_null())["time"].min()
        before = (
            part.filter(pl.col("time") < first, pl.col("time") >= datetime(2025, 3, 1, tzinfo=UTC))
            .group_by("drop_reason")
            .agg(rows=pl.len())
        )
        print(
            f"  {label}: first kept reading {first}; rows from 2025-03-01 before it by drop reason:"
        )
        print(before)


def main() -> None:
    """Run each check and write its output to `checks/evidence.txt`."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print(f"written {datetime.now(tz=UTC).isoformat(timespec='seconds')}")
        refit_match()
        bank_match()
        pretraining_rows()
        cleaned_starts()
    out = ROOT / "checks" / "evidence.txt"
    out.write_text(buffer.getvalue())
    print(buffer.getvalue())


if __name__ == "__main__":
    main()
