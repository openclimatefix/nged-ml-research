"""Experiment 507: shared inputs (the real Primaries' spec, paths, and the screen months)."""

from functools import cache
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00402_worlds as worlds402  # ty: ignore[unresolved-import]
import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00428_data as data428  # ty: ignore[unresolved-import]
import e00404_lagless as lagless  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00507_generator as gen  # ty: ignore[unresolved-import]
import e00507_lagless as lab  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00507-history-encoder-first-test-oracle-repair-and-total-variation"
)
WORK_DIR: Final = EXPERIMENT_DIR / "work"
SCREEN_MONTHS: Final[tuple[str, ...]] = lab.SCREEN_MONTHS
SIGNED_SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
TRAIN_WORLDS: Final[tuple[int, ...]] = (100, 101, 102)
SCORE_WORLDS: Final[tuple[int, ...]] = (200, 201, 202)


def use_lab_tables() -> None:
    """Point experiment 404's and 472's table readers at the ERA5 lab tables."""
    lagless.table_path = lab.table_path  # type: ignore[invalid-assignment]


def sites() -> list[str]:
    """Return the Primaries' site labels in index order."""
    return prep.primaries()["site"].to_list()


@cache
def clean_world() -> np.ndarray:
    """Return the cleaned reading (primaries, grid) that experiment 428's worlds were built on."""
    stack, _ = data428.load_stack(include_eval=False)
    return stack[0].astype(np.float32)


@cache
def load_spec() -> gen.Spec:
    """Return the real Primaries' generator spec."""
    real = clean_world()
    series, _ = fd._series_and_arrays()
    cap = {r["site"]: float(r["effective_capacity_mw"]) for r in series.iter_rows(named=True)}
    labels = pl.read_csv(worlds402.LABELS)
    hours = labels["duration_hours"].drop_nulls().to_numpy()
    pool = np.maximum(1, np.rint(hours / 0.5)).astype(np.int64)
    table = pl.read_parquet(lab.table_path(month="2025-10", seed=0)).filter(
        pl.col("source") == "crossfit"
    )
    spread = np.zeros(len(sites()))
    for p in range(len(sites())):
        part = table.filter(pl.col("primary") == p)
        resid = real[p, part["grid"].to_numpy()] - part["q50"].to_numpy()
        resid = resid[np.isfinite(resid)]
        spread[p] = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    return gen.Spec(
        real=real,
        shape=gen.shape_of(real=real),
        spread=spread,
        capacity_mw=np.array([cap[s] for s in sites()]),
        mva_only=np.array([s in fit.MVA_ONLY for s in sites()]),
        pool_steps=pool,
    )


def group_of_primary() -> np.ndarray:
    """Return 'six' for the signed-MW six, 'eight' for the MVA-only eight, else 'other'."""
    out = []
    for s in sites():
        out.append("six" if s in SIGNED_SIX else "eight" if s in fit.MVA_ONLY else "other")
    return np.array(out)


GRID_START_US: Final[int] = int(common.GRID_START.timestamp() * 1_000_000)
