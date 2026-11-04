# ruff: noqa: E501
"""The extra input columns of experiment 562: real and shuffled-day placebo, for each arm.

Arms B1, B2, B3a, and B3b read columns of experiment 437's stored per-Primary tables, which are keyed
by `(origin, valid time)` and were built on the same half-hourly series as experiment 515's window
columns (the two agree on the shared columns). Arm B4 reads experiment 513's whole-day price shape.

A placebo takes the columns of a donor day (experiment 515's `DayStructure`: same calendar month, day
type, and length, at least 3 days away, on the same side of the refit): a row's origin and valid time
both move by the whole-day shift to the donor day, so the lead and the half-hour of day are kept.
"""

from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00437_features as f437  # ty: ignore[unresolved-import]
import e00513_data as d513  # ty: ignore[unresolved-import]
import e00513_shape as shape513  # ty: ignore[unresolved-import]
import e00515_placebo as placebo  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

KEY_SHIFT: Final[int] = 22
SCHOOL: Final[tuple[str, ...]] = (
    "school_term_flag", "school_term_start_days", "school_term_end_days", "school_return_days",
    "school_window_flag",
)  # fmt: skip
NATIONAL: Final[tuple[str, ...]] = (
    "price_prev_day_mean", "price_prev_day_max", "price_prev_day_evening", "dfs_prev_day",
    "national_demand_prev_day", "embedded_solar_prev_day", "embedded_wind_prev_day",
)  # fmt: skip
TABLE_ARMS: Final[dict[str, tuple[str, ...]]] = {
    "B1": (*f437.BUNDLES["T"], *f437.BUNDLES["S"]),
    "B2": f437.BUNDLES["F"],
    "B3a": SCHOOL,
    "B3b": NATIONAL,
}
"""Arm -> columns of experiment 437's tables."""
ENCODER: Final[dict[str, str]] = {
    "B1": "weather",
    "B2": "weather",
    "B3a": "datetime",
    "B3b": "datetime",
    "B4": "datetime",
}
"""Which encoder reads an arm's extra columns."""
PRICE_WIDTH: Final[int] = shape513.N_COLUMNS + 1
"""Ten shape features, the missing flag, and the presence flag."""
ALL_TABLE_COLUMNS: Final[tuple[str, ...]] = tuple(c for cols in TABLE_ARMS.values() for c in cols)
assert set(SCHOOL) <= set(f437.BUNDLES["C"])
assert set(NATIONAL) <= set(f437.BUNDLES["C"])


def width(*, arm: str) -> int:
    """Return the number of extra input columns of an arm (real and placebo arms alike)."""
    base = _base_arm(arm=arm)
    return PRICE_WIDTH if base == "B4" else len(TABLE_ARMS[base])


def _base_arm(*, arm: str) -> str:
    """Return the real arm of an arm name: `B3ap` is `B3a`, `B4-oracle` is `B4`, `B1p` is `B1`."""
    name = arm.removesuffix("-oracle")
    return name.removesuffix("p")


def is_placebo(*, arm: str) -> bool:
    """Return True for a shuffled-day placebo arm (`B1p`, `B3ap`, ...)."""
    return arm.removesuffix("-oracle").endswith("p")


class TableLookup:
    """Experiment 437's tables of all 16 Primaries, searchable by (primary, origin, valid)."""

    def __init__(self, *, columns: tuple[str, ...]) -> None:
        """Load the 16 tables' `columns`, sorted by key, as float32."""
        self.columns = columns
        self.keys: dict[int, np.ndarray] = {}
        self.values: dict[int, np.ndarray] = {}
        for row in prep.primaries().iter_rows(named=True):
            t = pl.read_parquet(f437.table_path(site=row["site"]))
            origin = (t["origin_us"].to_numpy() - prep422.GRID_START_US) // prep422.STEP_US
            valid = (t["valid_us"].to_numpy() - prep422.GRID_START_US) // prep422.STEP_US
            key = origin.astype(np.int64) * (1 << KEY_SHIFT) + valid.astype(np.int64)
            order = np.argsort(key)
            self.keys[int(row["primary"])] = key[order]
            self.values[int(row["primary"])] = (
                t.select(columns).to_numpy()[order].astype(np.float32)
            )

    def get(
        self, *, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the columns of each (primary, origin, valid) row (NaN where absent) and a found mask."""
        out = np.full((len(primary), len(self.columns)), np.nan, dtype=np.float32)
        found = np.zeros(len(primary), dtype=bool)
        want = origin_idx.astype(np.int64) * (1 << KEY_SHIFT) + valid_idx.astype(np.int64)
        for p in np.unique(primary):
            sel = np.nonzero(primary == p)[0]
            keys = self.keys[int(p)]
            pos = np.clip(np.searchsorted(keys, want[sel]), 0, len(keys) - 1)
            hit = keys[pos] == want[sel]
            out[sel[hit]] = self.values[int(p)][pos[hit]]
            found[sel[hit]] = True
        return out, found


class FoldColumns:
    """One fold's real and placebo extra columns for any rows, with the fold's standardiser."""

    def __init__(self, *, fit_until_idx: int) -> None:
        """Load the tables, the price-shape tables, and the placebo day draws (donors on the same side of the refit)."""
        self.table = TableLookup(columns=ALL_TABLE_COLUMNS)
        self.shapes = d513.ShapeTables(fit_until_idx=fit_until_idx)
        first_run = int(common.grid_index(times=prep.FIRST_RUN)) + 5 * placebo.STEPS_PER_DAY
        holdout = int(common.grid_index(times=common.HOLDOUT))
        self.days = placebo.DayStructure(fit_until_idx=fit_until_idx, allowed=(first_run, holdout))
        self.mean: dict[str, np.ndarray] = {}
        self.std: dict[str, np.ndarray] = {}
        self.unfilled = 0
        self.not_found_real = 0

    def _raw_real(
        self, *, arm: str, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> np.ndarray:
        """Return the arm's raw table columns at the rows' own keys; raise if any row has none."""
        index = [ALL_TABLE_COLUMNS.index(c) for c in TABLE_ARMS[arm]]
        values, found = self.table.get(primary=primary, origin_idx=origin_idx, valid_idx=valid_idx)
        if not found.all():
            msg = f"{int((~found).sum())} rows have no row in experiment 437's table"
            raise ValueError(msg)
        return values[:, index]

    def _raw_placebo(
        self, *, arm: str, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> np.ndarray:
        """Return donor-day columns, with up to three draws as fallbacks; rows with no donor stay NaN."""
        index = [ALL_TABLE_COLUMNS.index(c) for c in TABLE_ARMS[arm]]
        out = np.full((len(primary), len(index)), np.nan, dtype=np.float32)
        todo = np.ones(len(primary), dtype=bool)
        for attempt in range(len(placebo.SEEDS)):
            # B3b columns are constant within a run, so one donor shift per origin day keeps them so.
            shift = self.days.shifts(
                valid_idx=origin_idx if arm == "B3b" else valid_idx, attempt=attempt
            )
            usable = todo & np.isfinite(shift)
            if not usable.any():
                continue
            s = np.where(usable, shift, 0).astype(np.int64)
            o2, v2 = origin_idx + s, valid_idx + s
            inside = usable & (o2 >= 0) & (v2 < len(self.days.day_id))
            idx = np.nonzero(inside)[0]
            values, found = self.table.get(
                primary=primary[idx], origin_idx=o2[idx], valid_idx=v2[idx]
            )
            out[idx[found]] = values[found][:, index]
            todo[idx[found]] = False
        self.unfilled += int(todo.sum())
        return out

    def fit_standardiser(
        self,
        *,
        primary: np.ndarray,
        origin_idx: np.ndarray,
        valid_idx: np.ndarray,
        train: np.ndarray,
    ) -> None:
        """Set each table arm's column means and standard deviations from the real columns of training rows."""
        for arm in TABLE_ARMS:
            raw = self._raw_real(arm=arm, primary=primary[train], origin_idx=origin_idx[train], valid_idx=valid_idx[train])  # fmt: skip
            self.mean[arm] = raw.mean(axis=0)
            self.std[arm] = np.maximum(raw.std(axis=0), 1e-6)

    def table_columns(
        self, *, arm: str, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> np.ndarray:
        """Return the arm's standardised extra columns (float32); a missing placebo value becomes 0."""
        real_arm = _base_arm(arm=arm)
        getter = self._raw_placebo if is_placebo(arm=arm) else self._raw_real
        raw = getter(arm=real_arm, primary=primary, origin_idx=origin_idx, valid_idx=valid_idx)
        scaled = (raw - self.mean[real_arm]) / self.std[real_arm]
        return np.where(np.isfinite(scaled), scaled, 0.0).astype(np.float32)

    def price_columns(self, *, arm: str, target_idx: np.ndarray, present: np.ndarray) -> np.ndarray:
        """Return arm B4's 12 columns: the price shape (or its shuffled-day placebo) where `present`, then the flag.

        Rows where `present` is False have all 11 shape columns 0 and the presence flag 0.
        """
        source = "placebo" if is_placebo(arm=arm) else "price"
        shape = self.shapes.columns(source=source, target_idx=target_idx)
        shape = np.where(present[:, None], shape, 0.0).astype(np.float32)
        return np.concatenate([shape, present[:, None].astype(np.float32)], axis=1)
