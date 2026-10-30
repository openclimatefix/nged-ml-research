"""Inputs, rows, and folds for the pooled network at the Primaries and four nodes (experiment 415).

The rows, the summary weather scheme, the 24-month window, the recency weights, and the held-out
weeks are experiment 397's (`e00397_prep`), extended from the 16 Primaries to 20 series: the 2 bulk
supply points (BSPs: D13, D17) and the 2 grid supply points (GSPs: D1, D6) join the Primaries.
Three things are new:

- the ensemble mean and standard deviation of each node, written to this experiment's folder;
- the short-history cut: every D15 and D18 row before 2025-03-01 is dropped, and a lag whose source
  half-hour lies before the cut is set missing;
- the node-type columns and the neighbour columns (the mean of the Primaries' three origin lags
  beneath each node, and the share of those Primaries with a reading), true and rewired.

Commands (run with `PYTHONPATH` set as `e00415_fit.py` shows):

- `summaries`: write the node ensemble summaries of every daily run.
- `raw`: write each series' raw input matrix, target, keys, and neighbour columns.
"""

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 415
WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00415-pooled-mixture-density-network-at-bulk-and-grid-supply-points"
    / "work"
)
NODE_SUMMARY_DIR: Final[Path] = WORK_DIR / "ens_summary_nodes"
RAW_DIR: Final[Path] = WORK_DIR / "raw"
HIERARCHY_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "substation_hierarchy_2026-10-15" / "Substations.csv"
)
METADATA_PATH: Final[Path] = ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "metadata.parquet"
NODES: Final[tuple[str, ...]] = ("D13", "D17", "D1", "D6")
CUT_SITES: Final[tuple[str, ...]] = ("D15", "D18")
CUT_TIME: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
"""Every D15 and D18 row before this time is dropped (the maintainer's cut, experiment 399)."""
REWIRE_SEED: Final[int] = 415
SCREEN_MONTHS: Final[tuple[str, ...]] = ("2026-06", "2026-01", "2025-10", "2026-03")
"""The four seasons of the few-fold screen, in the order the stages run them."""
TYPE_NAMES: Final[tuple[str, ...]] = ("Primary", "BSP", "GSP")
LAG_NAMES: Final[tuple[str, ...]] = ("lag_origin", "lag_day", "lag_week")
NEIGHBOUR_NAMES: Final[tuple[str, ...]] = (
    *[f"nb_{n}" for n in LAG_NAMES],
    *[f"nb_{n}_share" for n in LAG_NAMES],
)
ARMS: Final[tuple[str, ...]] = ("P0", "N1", "N2", "N2rw")
"""P0 fits at the Primaries only; N1 adds the nodes; N2 and N2rw add the neighbour columns."""


def series20() -> pl.DataFrame:
    """Return the 16 Primaries and 4 nodes of experiment 87's table, with a row index `primary`."""
    table = common.series_table()
    if table.height != 20:
        msg = f"expected 20 series, found {table.height}"
        raise ValueError(msg)
    return table.with_row_index("primary")


def children_of_nodes() -> dict[str, list[str]]:
    """Return, for each node, the labels of the Primaries beneath it in NGED's hierarchy file.

    A BSP's Primaries are the rows with its number in `bsp_substation_number`; a GSP's are the rows
    with its number in `gsp_substation_number`. Only series that are Primaries in the table count.
    The file is private NGED data, so only labels leave this function.
    """
    series = series20().select("site", "time_series_id", "substation_type")
    numbers = pl.read_parquet(METADATA_PATH, columns=["time_series_id", "substation_number"])
    labelled = series.join(numbers, on="time_series_id", how="left", validate="1:1")
    hierarchy = pl.read_csv(HIERARCHY_PATH).join(
        labelled.select("site", "substation_type", "substation_number"),
        on="substation_number",
        how="inner",
        validate="1:m",
    )
    out: dict[str, list[str]] = {}
    for row in labelled.filter(pl.col("site").is_in(NODES)).iter_rows(named=True):
        column = (
            "bsp_substation_number" if row["substation_type"] == "BSP" else "gsp_substation_number"
        )
        kids = hierarchy.filter(
            (pl.col(column) == row["substation_number"]) & (pl.col("substation_type") == "Primary")
        )["site"].to_list()
        out[row["site"]] = sorted(kids, key=lambda s: int(s[1:]))
    return out


def rewired_children(*, children: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return a random set of Primaries of the same size for each node, never the true set."""
    primaries = [s for s in series20()["site"].to_list() if s not in NODES]
    rng = np.random.default_rng(REWIRE_SEED)
    out: dict[str, list[str]] = {}
    for node in NODES:
        true = set(children[node])
        while True:
            pick = sorted(rng.choice(primaries, size=len(true), replace=False).tolist())
            if set(pick) != true:
                break
        out[node] = sorted(pick, key=lambda s: int(s[1:]))
    return out


# --- node ensemble summaries ----------------------------------------------------------------


def write_node_summary(*, run: datetime) -> str:
    """Write one run's node summaries if absent; return a progress line."""
    target = NODE_SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet"
    if target.exists():
        return f"{run:%Y-%m-%d}: exists"
    started = time.monotonic()
    targets = (
        series20()
        .filter(pl.col("site").is_in(NODES))
        .select("time_series_id", "h3_res_5", "latitude", "longitude")
    )
    table = base.summary_of_run(run=run, targets=targets).cast(
        dict.fromkeys([f"{k}_{w}" for k in ("mean", "std") for w in base.WEATHER], pl.Float32)
    )
    common.atomic_write(frame=table, target=target)
    return f"{run:%Y-%m-%d}: {table.height:,} rows, {time.monotonic() - started:.0f} s"


def _write_node_summary_kw(run: datetime) -> str:
    """Return `write_node_summary` as a picklable function of one argument."""
    return write_node_summary(run=run)


def load_summary(*, series_id: int, node_ids: set[int]) -> pl.DataFrame:
    """Return one series' ensemble summaries over all runs, from the Primary or the node folder."""
    folder = NODE_SUMMARY_DIR if series_id in node_ids else base.SUMMARY_DIR
    return (
        pl.scan_parquet(folder / "init=*.parquet")
        .filter(pl.col("time_series_id") == series_id)
        .collect()
    )


# --- raw matrices ---------------------------------------------------------------------------


def _us(*, t: datetime) -> int:
    """Return a UTC time as microseconds since the epoch."""
    return int(t.timestamp()) * 1_000_000


def series_rows(*, site: str) -> pl.DataFrame:
    """Return a series' control-member rows joined with its ensemble summary."""
    series, arrays = fd._series_and_arrays()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    node_ids = set(series.filter(pl.col("site").is_in(NODES))["time_series_id"].to_list())
    rows = fd.training_rows(row=row, arrays=arrays[site], frozen=False, a7=False)
    summary = load_summary(series_id=row["time_series_id"], node_ids=node_ids).drop(
        "time_series_id"
    )
    rows = rows.join(
        summary,
        on=["power_fcst_init_time", "valid_time"],
        how="inner",
        validate="m:1",
        maintain_order="left",
    )
    return rows.with_columns(capacity=pl.lit(float(row["effective_capacity_mw"])))


def apply_cut(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Drop the rows of a short-history series whose valid time precedes the cut."""
    return frame.filter(pl.col("valid_time") >= CUT_TIME)


def write_raw(*, site: str) -> str:
    """Write a series' raw matrix, target, keys, and lag columns."""
    target = RAW_DIR / f"{site}.npz"
    if target.exists():
        return f"{site}: exists"
    started = time.monotonic()
    rows = series_rows(site=site)
    if site in CUT_SITES:
        rows = apply_cut(frame=rows)
    x = base.feature_matrix(frame=rows, scheme="summary")
    names, _, _ = base.feature_names(scheme="summary")
    valid = rows["valid_time"].dt.epoch("us").to_numpy()
    if site in CUT_SITES:
        for tag in ("origin", "day", "week"):
            age_us = (
                x[:, names.index(f"lag_{tag}_age")].astype(np.float64)
                * base.LEAD_SCALE_HOURS
                * 3_600
                * 1_000_000
            )
            stale = (valid - age_us) < _us(t=CUT_TIME)
            x[stale, names.index(f"lag_{tag}")] = np.nan
            x[stale, names.index(f"lag_{tag}_missing")] = 1.0
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.npz")
    np.savez(
        temporary,
        x=x,
        y=(rows["power"] / rows["capacity"]).to_numpy().astype(np.float32),
        valid=valid,
        origin=rows["power_fcst_init_time"].dt.epoch("us").to_numpy(),
        lead=rows["lead_hours"].to_numpy().astype(np.float32),
    )
    temporary.rename(target)
    return f"{site}: {len(x):,} rows, {time.monotonic() - started:.0f} s"


def neighbour_columns(*, node: str, kids: list[str]) -> np.ndarray:
    """Return (rows, 6) neighbour columns for a node's raw rows: three means, three shares.

    The mean of lag `l` is the average over the Primaries in `kids` that have a reading for `l` at
    the same run and valid time (NaN where none has); the share is the fraction of `kids` with one.
    Every lag is a fraction of the Primary's own effective capacity, read as of the run's origin.
    """
    names, _, _ = base.feature_names(scheme="summary")
    lag_index = [names.index(n) for n in LAG_NAMES]
    pieces = []
    for kid in kids:
        z = np.load(RAW_DIR / f"{kid}.npz")
        frame = pl.DataFrame(
            {"origin": z["origin"], "valid": z["valid"]}
            | {f"l{i}": z["x"][:, j].astype(np.float64) for i, j in enumerate(lag_index)}
        ).with_columns(pl.col(f"l{i}").fill_nan(None) for i in range(3))
        pieces.append(frame)
    stacked = (
        pl.concat(pieces)
        .group_by("origin", "valid")
        .agg(
            *[pl.col(f"l{i}").mean().alias(f"m{i}") for i in range(3)],
            *[(pl.col(f"l{i}").count() / len(kids)).alias(f"s{i}") for i in range(3)],
        )
    )
    z = np.load(RAW_DIR / f"{node}.npz")
    keys = pl.DataFrame({"origin": z["origin"], "valid": z["valid"]})
    joined = keys.join(stacked, on=["origin", "valid"], how="left", maintain_order="left")
    out = np.empty((len(keys), 6), dtype=np.float32)
    for i in range(3):
        out[:, i] = joined[f"m{i}"].fill_null(float("nan")).to_numpy()
        out[:, 3 + i] = joined[f"s{i}"].fill_null(0.0).to_numpy()
    return out


def write_neighbours() -> list[str]:
    """Write, per node, the true and rewired neighbour columns beside its raw matrix."""
    children = children_of_nodes()
    wired = rewired_children(children=children)
    lines = []
    for node in NODES:
        target = RAW_DIR / f"{node}_neighbours.npz"
        true_cols = neighbour_columns(node=node, kids=children[node])
        rewired_cols = neighbour_columns(node=node, kids=wired[node])
        temporary = target.with_name(target.name + ".tmp.npz")
        np.savez(temporary, true=true_cols, rewired=rewired_cols)
        temporary.rename(target)
        overlap = len(set(children[node]) & set(wired[node]))
        lines.append(
            f"{node}: {len(children[node])} Primaries beneath, rewired set shares {overlap}"
        )
    return lines


# --- folds ----------------------------------------------------------------------------------


@dataclass
class Fold:
    """One fold's standardised training rows for an arm, and the test rows of the series."""

    train: base.FoldTrain
    refit: datetime
    fit_ids: list[int]
    extra_columns: int


def series_block(*, site: str, arm: str, series: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return a series' raw rows with the arm's extra columns appended."""
    z = np.load(RAW_DIR / f"{site}.npz")
    row = series.filter(pl.col("site") == site).row(0, named=True)
    parts = [z["x"]]
    if arm != "P0":
        kind = np.zeros((len(z["x"]), len(TYPE_NAMES)), dtype=np.float32)
        kind[:, TYPE_NAMES.index(row["substation_type"])] = 1.0
        parts.append(kind)
    if arm in ("N2", "N2rw"):
        if site in NODES:
            n = np.load(RAW_DIR / f"{site}_neighbours.npz")
            parts.append(n["true" if arm == "N2" else "rewired"])
        else:
            blank = np.full((len(z["x"]), 6), np.nan, dtype=np.float32)
            blank[:, 3:] = 0.0
            parts.append(blank)
    return {
        "x": np.concatenate(parts, axis=1),
        "y": z["y"],
        "valid": z["valid"],
        "origin": z["origin"],
        "lead": z["lead"],
    }


def extra_mask(*, arm: str) -> np.ndarray:
    """Return the standardise mask of the arm's extra columns (the neighbour means only)."""
    if arm == "P0":
        return np.zeros(0, dtype=bool)
    mask = [False] * len(TYPE_NAMES)
    if arm in ("N2", "N2rw"):
        mask += [True] * 3 + [False] * 3
    return np.array(mask)


def build_fold(*, arm: str, refit: datetime) -> Fold:
    """Return the standardised training rows of one arm at one fold.

    A fold sees a row only if its valid time is at least 3.5 hours before the refit and its run
    began at or before the refit, within the 24 months before the refit (experiment 397's rule).
    P0 trains on the 16 Primaries; the other arms train on the Primaries and the 4 nodes. Every
    series gets target statistics from its own training rows, so P0 can be applied to a node.
    """
    series = series20()
    _, mask, layout = base.feature_names(scheme="summary")
    mask = np.concatenate([mask, extra_mask(arm=arm)])
    fit_ids = [
        int(r["primary"])
        for r in series.iter_rows(named=True)
        if arm != "P0" or r["site"] not in NODES
    ]
    cutoff = _us(t=refit - regimes.REFIT_CUTOFF_AGE)
    parts = []
    for row in series.iter_rows(named=True):
        z = series_block(site=row["site"], arm=arm, series=series)
        keep = (
            (z["valid"] <= cutoff)
            & (z["origin"] <= _us(t=refit))
            & (z["valid"] > _us(t=refit - timedelta(days=base.WINDOW_DAYS)))
        )
        if not keep.any():
            continue
        parts.append(
            {k: v[keep] for k, v in z.items()}
            | {"primary": np.full(int(keep.sum()), row["primary"], dtype=np.int64)}
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    origin_day = ((cat["origin"] - _us(t=base.FIRST_RUN)) // (86_400 * 1_000_000)).astype(np.int64)
    held, purged = base.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    in_fit = np.isin(cat["primary"], fit_ids)
    standardiser = base.Standardiser.fit(x=cat["x"][train & in_fit], mask=mask)
    age_days = (_us(t=refit) - cat["valid"]) / (86_400 * 1_000_000)
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n_series = len(series)
    target_mean, target_std = np.zeros(n_series), np.ones(n_series)
    for p in range(n_series):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep = (train | held) & in_fit
    x_kept = standardiser.apply(x=cat["x"][keep])
    del cat["x"]
    return Fold(
        train=base.FoldTrain(
            x=x_kept,
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
        ),
        refit=refit,
        fit_ids=fit_ids,
        extra_columns=len(extra_mask(arm=arm)),
    )


def test_rows(*, arm: str, month: str, standardiser: base.Standardiser) -> list[dict]:
    """Return, per series, the test month's keys, actuals (fraction of capacity), and inputs.

    The test rows are the series' raw rows whose run origin falls in the month. The lags in them
    are read as of the origin, as in training.
    """
    series = series20()
    start = datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC)
    stop = (start + timedelta(days=32)).replace(day=1)
    out = []
    for row in series.iter_rows(named=True):
        z = series_block(site=row["site"], arm=arm, series=series)
        keep = (z["origin"] >= _us(t=start)) & (z["origin"] < _us(t=stop))
        out.append(
            {
                "row": row,
                "origin": z["origin"][keep],
                "valid": z["valid"][keep],
                "y": z["y"][keep],
                "x": standardiser.apply(x=z["x"][keep]),
            }
        )
    return out


def main() -> None:
    """Build the node summaries, the raw matrices, or the neighbour columns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("summaries", "raw", "neighbours"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="3 runs, or 2 series")
    args = parser.parse_args()
    if args.command == "summaries":
        runs = base.run_times()[:3] if args.smoke else base.run_times()
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for line in pool.map(_write_node_summary_kw, runs):
                print(line, flush=True)
    elif args.command == "raw":
        sites = series20()["site"].to_list()[: 2 if args.smoke else None]
        for site in sites:
            print(write_raw(site=site), flush=True)
    else:
        for line in write_neighbours():
            print(line, flush=True)


if __name__ == "__main__":
    main()
