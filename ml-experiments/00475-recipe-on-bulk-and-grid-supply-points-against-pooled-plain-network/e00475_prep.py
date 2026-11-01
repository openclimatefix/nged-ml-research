"""Fold rows for experiment 475: experiment 425's arm A recipe on the 16 Primaries and 4 nodes.

Experiment 425's fine-tuning and ERA5-and-CAMS pretraining rows cover the 16 Primaries. Here the
4 nodes (D13 and D17 are bulk supply points; D1 and D6 are grid supply points) join them, with
experiment 415's node-type columns (three one-hot columns, `Primary`, `BSP`, `GSP`, appended and
left unstandardised). Sources:

- Primaries, fine-tuning rows: experiment 397's summary-scheme matrices, as in experiment 425.
- Nodes, fine-tuning rows: experiment 415's raw matrices (same feature builder, node ensemble
  summaries from `ens_summary_nodes`).
- Primaries, pretraining rows: experiment 418's `raw_pretrain` matrices.
- Nodes, pretraining rows: written here (`pretrain-raw`) by experiment 418's builder, from ERA5
  (`era5_features.parquet` holds the nodes) and CAMS (which holds D1, D6, D13, and D17).

The two cuts of experiment 425 (D15 and D18 start cut; flagged exact zeros dropped) apply to every
series. Commands (run with `PYTHONPATH` set as `e00475_fit.py` shows):

- `pretrain-raw`: write the four nodes' pretraining matrices.
- `check`: build one fold and print the row counts.
"""

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00407_fit as fit407  # ty: ignore[unresolved-import]
import e00415_prep as prep415  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR_NAME: Final[str] = (
    "00475-recipe-on-bulk-and-grid-supply-points-against-pooled-plain-network"
)
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_DIR_NAME / "work"
"""This experiment's private data folder; every other experiment's folder is only read."""
PRETRAIN_NODE_DIR: Final[Path] = WORK_DIR / "raw_pretrain_nodes"
NODES: Final[tuple[str, ...]] = prep415.NODES
TYPE_NAMES: Final[tuple[str, ...]] = prep415.TYPE_NAMES
SCREEN_MONTHS: Final[tuple[str, ...]] = prep415.SCREEN_MONTHS
US_PER_DAY: Final[int] = prep425.US_PER_DAY


def series20() -> pl.DataFrame:
    """Return the 16 Primaries and 4 nodes with a row index `primary` (the embedding index)."""
    return prep415.series20()


def _type_columns(*, substation_type: str, rows: int) -> np.ndarray:
    """Return the one-hot node-type columns for `rows` rows of one series."""
    out = np.zeros((rows, len(TYPE_NAMES)), dtype=np.float32)
    out[:, TYPE_NAMES.index(substation_type)] = 1.0
    return out


def fine_tune_path(*, site: str) -> Path:
    """Return the raw fine-tuning matrix of a series (Primaries: 397's; nodes: 415's)."""
    if site in NODES:
        return prep415.RAW_DIR / f"{site}.npz"
    return base.RAW_DIR / "summary" / f"{site}.npz"


def pretrain_path(*, site: str) -> Path:
    """Return the raw pretraining matrix of a series (Primaries: 418's; nodes: written here)."""
    if site in NODES:
        return PRETRAIN_NODE_DIR / f"{site}.npz"
    return prep418.RAW_DIR / f"{site}.npz"


def write_pretrain_raw(*, site: str) -> str:
    """Write a node's raw pretraining matrix (experiment 418's `write_raw`, for a node)."""
    target = pretrain_path(site=site)
    if target.exists():
        return f"{site}: exists"
    started = time.monotonic()
    table = series20().filter(pl.col("site") == site)
    row = table.row(0, named=True)
    power = common.load_power_arrays(series=table)[site].power
    weather = prep418.dense_weather(series_id=row["time_series_id"], site=site)
    frame, _ = prep418.pretrain_frame(row=row, power=power, weather=weather)
    x = base.feature_matrix(frame=frame, scheme="summary")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.npz")
    np.savez(
        temporary,
        x=x,
        y=power[np.asarray(common.grid_index(times=frame["valid_time"]))].astype(np.float32),
        valid=frame["valid_time"].dt.epoch("us").to_numpy(),
        origin=frame["power_fcst_init_time"].dt.epoch("us").to_numpy(),
        lead=frame["lead_hours"].to_numpy().astype(np.float32),
    )
    temporary.rename(target)
    return f"{site}: {len(x):,} rows, {time.monotonic() - started:.0f} s"


def build_ens_fold(*, refit: datetime) -> prep425.EnsFold:
    """Return the fine-tuning rows of the 20 series with the standing cuts applied.

    As `e00425_prep.build_ens_fold`, over 20 series, with the node-type columns appended. The
    standardiser and the per-series target statistics are fitted on the rows that remain.
    """
    _, mask, layout = base.feature_names(scheme="summary")
    mask = np.concatenate([mask, np.zeros(len(TYPE_NAMES), dtype=bool)])
    flagged = fit407.flagged_times()
    dropped = prep425.DroppedRows()
    parts = []
    for row in series20().iter_rows(named=True):
        z = np.load(fine_tune_path(site=row["site"]))
        keep = (
            (z["valid"] <= prep425._us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (z["origin"] <= prep425._us(t=refit))
            & (z["valid"] > prep425._us(t=refit - timedelta(days=base.WINDOW_DAYS)))
        )
        keep, counts = prep425.apply_cuts(
            keep=keep,
            valid=z["valid"],
            lead=z["lead"],
            site=row["site"],
            series_id=int(row["time_series_id"]),
            flagged=flagged,
        )
        dropped += counts
        if not keep.any():
            continue
        x = np.concatenate(
            [
                z["x"][keep],
                _type_columns(substation_type=row["substation_type"], rows=int(keep.sum())),
            ],
            axis=1,
        )
        parts.append(
            {"x": x}
            | {k: z[k][keep] for k in ("y", "origin", "valid", "lead")}
            | {"primary": np.full(int(keep.sum()), row["primary"], dtype=np.int64)}
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    origin_day = ((cat["origin"] - prep425._us(t=base.FIRST_RUN)) // US_PER_DAY).astype(np.int64)
    held, purged = base.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = base.Standardiser.fit(x=cat["x"][train], mask=mask)
    age_days = (prep425._us(t=refit) - cat["valid"]) / US_PER_DAY
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n_series = len(series20())
    target_mean, target_std = np.zeros(n_series), np.ones(n_series)
    for p in range(n_series):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep = train | held
    return prep425.EnsFold(
        x=standardiser.apply(x=cat["x"][keep]),
        y=cat["y"][keep],
        primary=cat["primary"][keep],
        weight=weight[keep].astype(np.float32),
        is_val=held[keep],
        lead=cat["lead"][keep],
        origin_day=origin_day[keep],
        standardiser=standardiser,
        target_mean=target_mean,
        target_std=target_std,
        layout=layout,
        valid=cat["valid"][keep],
        dropped=dropped,
    )


def fold_pretrain(
    *, fold: prep425.EnsFold, refit: datetime
) -> tuple[prep418.PretrainSet, prep425.DroppedRows]:
    """Return the ERA5-and-CAMS pretraining rows of the 20 series on the fold's input scale.

    As `e00425_prep.fold_pretrain`, over 20 series with the node-type columns appended.
    """
    table, _ = prep418.spread_tables(fold=fold)
    _, std_cols = prep418.column_indices()
    cutoff_valid = prep425._us(t=refit - regimes.REFIT_CUTOFF_AGE)
    cutoff_origin = prep425._us(t=refit)
    flagged = fit407.flagged_times()
    dropped = prep425.DroppedRows()
    files = []
    total = 0
    for row in series20().iter_rows(named=True):
        z = np.load(pretrain_path(site=row["site"]))
        origin = z["origin"]
        keep = (z["valid"] <= cutoff_valid) & (origin <= cutoff_origin)
        day = ((origin - prep425._us(t=base.FIRST_RUN)) // US_PER_DAY).astype(np.int64)
        held, purged = base.held_out_weeks(origin_day=day)
        keep &= ~purged
        keep, counts = prep425.apply_cuts(
            keep=keep,
            valid=z["valid"],
            lead=z["lead"],
            site=row["site"],
            series_id=int(row["time_series_id"]),
            flagged=flagged,
        )
        dropped += counts
        if keep.any():
            files.append((row, keep, held))
            total += int(keep.sum())
    out = prep418.PretrainSet(
        x=np.empty((total, fold.x.shape[1]), dtype=np.float32),
        y=np.empty(total, dtype=np.float32),
        primary=np.empty(total, dtype=np.int64),
        is_val=np.empty(total, dtype=bool),
        lead=np.empty(total, dtype=np.float32),
    )
    start = 0
    for row, keep, held in files:
        z = np.load(pretrain_path(site=row["site"]))
        n = int(keep.sum())
        raw = np.concatenate(
            [z["x"][keep], _type_columns(substation_type=row["substation_type"], rows=n)], axis=1
        )
        lead = z["lead"][keep]
        band = np.minimum((lead // prep418.LEAD_BAND_HOURS).astype(int), prep418.N_BANDS - 1)
        for a in range(0, n, prep418.CHUNK):
            b = min(a + prep418.CHUNK, n)
            chunk = fold.standardiser.apply(x=raw[a:b])
            chunk[:, std_cols] = table[band[a:b]]
            out.x[start + a : start + b] = chunk
        out.y[start : start + n] = z["y"][keep]
        out.primary[start : start + n] = row["primary"]
        out.is_val[start : start + n] = held[keep]
        out.lead[start : start + n] = lead
        start += n
    return out, dropped


def node_test_rows(*, month: str, fold: prep425.EnsFold) -> list[dict]:
    """Return, per node, the test month's keys, actuals (fraction of capacity), and inputs.

    The test rows are the node's raw rows whose run origin falls in the month, with the type
    columns appended and the fold's standardiser applied. No row is dropped here: the scorer
    drops flagged exact-zero actuals, as in experiment 425.
    """
    start = datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC)
    stop = (start + timedelta(days=32)).replace(day=1)
    out = []
    for row in series20().filter(pl.col("site").is_in(NODES)).iter_rows(named=True):
        z = np.load(fine_tune_path(site=row["site"]))
        keep = (z["origin"] >= prep425._us(t=start)) & (z["origin"] < prep425._us(t=stop))
        x = np.concatenate(
            [
                z["x"][keep],
                _type_columns(substation_type=row["substation_type"], rows=int(keep.sum())),
            ],
            axis=1,
        )
        out.append(
            {
                "row": row,
                "origin": z["origin"][keep],
                "valid": z["valid"][keep],
                "y": z["y"][keep],
                "x": fold.standardiser.apply(x=x),
            }
        )
    return out


def main() -> None:
    """Write the node pretraining matrices, or build one fold and print the row counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pretrain-raw", "check"))
    parser.add_argument("--month", default="2026-01")
    args = parser.parse_args()
    if args.command == "pretrain-raw":
        for site in NODES:
            print(write_pretrain_raw(site=site), flush=True)
        return
    refit = base.refit_of(month=args.month)
    fold = build_ens_fold(refit=refit)
    print(f"fold {args.month}: {len(fold.x):,} fine-tuning rows, {fold.x.shape[1]} inputs")
    print(f"fine-tuning rows dropped: {fold.dropped.as_dict()}")
    for p in range(len(series20())):
        n = int((fold.primary == p).sum())
        print(f"  series {p}: {n:,} rows")
    pre, pre_dropped = fold_pretrain(fold=fold, refit=refit)
    print(f"pretraining rows kept {len(pre.x):,}; dropped: {pre_dropped.as_dict()}")


if __name__ == "__main__":
    main()
