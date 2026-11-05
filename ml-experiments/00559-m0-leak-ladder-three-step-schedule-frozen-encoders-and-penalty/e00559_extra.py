# ruff: noqa: E501
"""Extra weather columns for the weather encoder (and the spread input) of arm A11, real and placebo.

A source supplies, for ensemble rows keyed by (Primary, forecast origin, valid time) and for ERA5
rows keyed by (Primary, valid time), a block of columns: the first `n_weather` columns are ensemble
means and go to the weather encoder, the rest are ensemble spreads and go to the fine-tuning mixer.
The placebo is experiment 515's shuffled-day draw (`e00515_placebo.DayStructure`), applied to every
column of the block at once, so the real and placebo arms have the same number of columns and the
same missing-value pattern after the donor shift. Rows keep the origin-time rule: an ensemble row's
columns come from the run in the summary file keyed by that row's own origin, and a donor row moves
origin and valid time by the same whole-day shift.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00515_placebo as placebo  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

GRID = c.prep422.GRID_START_US
STEP = c.prep422.STEP_US
KEY_SHIFT: Final[int] = 22
UV_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00397-plain-and-attention-mixture-density-network-for-primary-demand/work/ens_summary_uv"
)
CLOUD_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00397-plain-and-attention-mixture-density-network-for-primary-demand/work/ens_summary_tcc"
)
ERA5_UV: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent
    / "studies/per_study/stage1_pilot/demand/inputs/era5_wind_components.parquet"
)
UV_NAMES: Final[tuple[str, ...]] = ("wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m")


def grid_index(*, us: np.ndarray) -> np.ndarray:
    """Return grid indices of microsecond timestamps."""
    return ((us - GRID) // STEP).astype(np.int64)


class UvSource:
    """Ensemble mean and standard deviation of u and v at 10 m and 100 m; ERA5 u and v for pretraining."""

    n_weather = 4
    n_spread = 4
    prescaled = False
    donor_start: int | None = None

    def __init__(self) -> None:
        """Load all summary files into per-Primary sorted key tables, and the ERA5 components by grid index."""
        table = prep.primaries().select("primary", "time_series_id")
        wanted = [
            "time_series_id",
            "power_fcst_init_time",
            "valid_time",
            *[f"mean_{n}" for n in UV_NAMES],
            *[f"std_{n}" for n in UV_NAMES],
        ]
        frames = [pl.read_parquet(p, columns=wanted) for p in sorted(UV_DIR.glob("init=*.parquet"))]
        ens = pl.concat(frames).join(table, on="time_series_id", how="inner")
        origin = grid_index(us=ens["power_fcst_init_time"].dt.epoch("us").to_numpy())
        valid = grid_index(us=ens["valid_time"].dt.epoch("us").to_numpy())
        key = origin * (1 << KEY_SHIFT) + valid
        primary = ens["primary"].to_numpy()
        values = ens.select(wanted[3:]).to_numpy().astype(np.float32)
        self.keys: dict[int, np.ndarray] = {}
        self.values: dict[int, np.ndarray] = {}
        for p in np.unique(primary):
            sel = np.nonzero(primary == p)[0]
            order = np.argsort(key[sel])
            self.keys[int(p)] = key[sel][order]
            self.values[int(p)] = values[sel][order]
        era = pl.read_parquet(ERA5_UV, columns=["time_series_id", "valid_time", *UV_NAMES]).join(
            table, on="time_series_id", how="inner"
        )
        e_valid = grid_index(us=era["valid_time"].dt.epoch("us").to_numpy())
        e_primary = era["primary"].to_numpy()
        e_values = era.select(UV_NAMES).to_numpy().astype(np.float32)
        self.era_keys: dict[int, np.ndarray] = {}
        self.era_values: dict[int, np.ndarray] = {}
        for p in np.unique(e_primary):
            sel = np.nonzero(e_primary == p)[0]
            order = np.argsort(e_valid[sel])
            self.era_keys[int(p)] = e_valid[sel][order]
            self.era_values[int(p)] = e_values[sel][order]

    def ens(
        self, *, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the 8 columns of each (Primary, origin, valid) row and a mask of rows found."""
        out = np.full((len(primary), 8), np.nan, dtype=np.float32)
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

    def era5(self, *, primary: np.ndarray, valid_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the 8 columns of ERA5 rows (the 4 spread columns are 0) and a mask of rows found."""
        out = np.full((len(primary), 8), np.nan, dtype=np.float32)
        found = np.zeros(len(primary), dtype=bool)
        for p in np.unique(primary):
            sel = np.nonzero(primary == p)[0]
            keys = self.era_keys[int(p)]
            pos = np.clip(np.searchsorted(keys, valid_idx[sel]), 0, len(keys) - 1)
            hit = keys[pos] == valid_idx[sel]
            out[sel[hit], :4] = self.era_values[int(p)][pos[hit]]
            out[sel[hit], 4:] = 0.0
            found[sel[hit]] = True
        return out, found


class CloudSource:
    """Total cloud cover (fraction 0-1): ensemble mean and standard deviation, and ERA5 for pretraining.

    The ensemble files hold percent (0-100) and start at the 2025-11-21 run; they are divided by 100 so
    both sources are fractions. Columns: the mean and a missing flag (weather encoder), then the standard
    deviation (fine-tuning mixer). A row whose run has no cloud file has mean 0, flag 1, and spread 0.
    The columns are already on a 0-1 scale and are not standardised.
    """

    n_weather = 2
    n_spread = 1
    prescaled = True

    def __init__(self) -> None:
        """Load the cloud summary files and the ERA5 cloud cover."""
        table = prep.primaries().select("primary", "time_series_id")
        wanted = [
            "time_series_id",
            "power_fcst_init_time",
            "valid_time",
            "mean_total_cloud_cover",
            "std_total_cloud_cover",
        ]
        frames = [
            pl.read_parquet(p, columns=wanted) for p in sorted(CLOUD_DIR.glob("init=*.parquet"))
        ]
        ens = pl.concat(frames).join(table, on="time_series_id", how="inner")
        origin = grid_index(us=ens["power_fcst_init_time"].dt.epoch("us").to_numpy())
        valid = grid_index(us=ens["valid_time"].dt.epoch("us").to_numpy())
        key = origin * (1 << KEY_SHIFT) + valid
        primary = ens["primary"].to_numpy()
        values = ens.select(wanted[3:]).to_numpy().astype(np.float32) / 100.0
        self.first_origin_idx = int(origin.min())
        self.donor_start = self.first_origin_idx
        self.keys: dict[int, np.ndarray] = {}
        self.values: dict[int, np.ndarray] = {}
        for p in np.unique(primary):
            sel = np.nonzero(primary == p)[0]
            order = np.argsort(key[sel])
            self.keys[int(p)] = key[sel][order]
            self.values[int(p)] = values[sel][order]
        era = pl.read_parquet(
            ERA5_UV, columns=["time_series_id", "valid_time", "total_cloud_cover"]
        ).join(table, on="time_series_id", how="inner")
        e_valid = grid_index(us=era["valid_time"].dt.epoch("us").to_numpy())
        e_primary = era["primary"].to_numpy()
        e_values = era["total_cloud_cover"].to_numpy().astype(np.float32)
        assert float(np.nanmax(e_values)) <= 1.0 + 1e-6, "ERA5 cloud cover should be a fraction"
        self.era_keys: dict[int, np.ndarray] = {}
        self.era_values: dict[int, np.ndarray] = {}
        for p in np.unique(e_primary):
            sel = np.nonzero(e_primary == p)[0]
            order = np.argsort(e_valid[sel])
            self.era_keys[int(p)] = e_valid[sel][order]
            self.era_values[int(p)] = e_values[sel][order]

    def ens_donor(
        self, *, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return [mean, 0, std] of each (Primary, origin, valid) row and a mask of rows that have cloud cover."""
        out = np.full((len(primary), 3), np.nan, dtype=np.float32)
        found = np.zeros(len(primary), dtype=bool)
        want = origin_idx.astype(np.int64) * (1 << KEY_SHIFT) + valid_idx.astype(np.int64)
        for p in np.unique(primary):
            sel = np.nonzero(primary == p)[0]
            keys = self.keys[int(p)]
            pos = np.clip(np.searchsorted(keys, want[sel]), 0, len(keys) - 1)
            hit = keys[pos] == want[sel]
            vals = self.values[int(p)][pos[hit]]
            out[sel[hit]] = np.stack(
                [vals[:, 0], np.zeros(len(vals), np.float32), vals[:, 1]], axis=1
            )
            found[sel[hit]] = True
        return out, found

    def ens(
        self, *, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the real columns: cloud where the run has it, otherwise [0, 1, 0]; every row counts as found."""
        out, found = self.ens_donor(primary=primary, origin_idx=origin_idx, valid_idx=valid_idx)
        out[~found] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return out, np.ones(len(primary), dtype=bool)

    def era5(self, *, primary: np.ndarray, valid_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return [cloud, 0, 0] of ERA5 rows and a mask of rows found."""
        out = np.full((len(primary), 3), np.nan, dtype=np.float32)
        found = np.zeros(len(primary), dtype=bool)
        for p in np.unique(primary):
            sel = np.nonzero(primary == p)[0]
            keys = self.era_keys[int(p)]
            pos = np.clip(np.searchsorted(keys, valid_idx[sel]), 0, len(keys) - 1)
            hit = keys[pos] == valid_idx[sel]
            out[sel[hit]] = np.stack(
                [
                    self.era_values[int(p)][pos[hit]],
                    np.zeros(hit.sum(), np.float32),
                    np.zeros(hit.sum(), np.float32),
                ],
                axis=1,
            )
            found[sel[hit]] = True
        return out, found

    @staticmethod
    def finish_null(*, real: np.ndarray, null: np.ndarray) -> np.ndarray:
        """Give the placebo the real columns' missing pattern (no cloud: mean 0, flag 1, spread 0)."""
        out = null.copy()
        missing = real[:, 1] == 1.0
        out[missing] = real[missing]
        out[~missing, 1] = 0.0
        return out

    @staticmethod
    def present(*, real: np.ndarray) -> np.ndarray:
        """Return which rows have cloud cover."""
        return real[:, 1] == 0.0


class ExtraColumns:
    """Standardised extra columns (real and placebo) for the month rows, the test rows, and pretraining rows."""

    def __init__(self, *, month: str, mr: c.MonthRows, source: UvSource | CloudSource) -> None:
        """Draw the placebo days, build the month's columns, and standardise on the training rows."""
        self.source = source
        refit = c.month_start(month=month)
        fit_until = int((c.us(t=refit) - GRID) // STEP)
        first_run = int(common.grid_index(times=prep.FIRST_RUN)) + 5 * placebo.STEPS_PER_DAY
        holdout = int(common.grid_index(times=common.HOLDOUT))
        self.days = {
            "era5": placebo.DayStructure(fit_until_idx=fit_until),
            "ens": placebo.DayStructure(
                fit_until_idx=fit_until,
                allowed=(max(first_run, source.donor_start or 0), holdout),
            ),
        }
        self.unfilled = 0
        origin_idx = grid_index(us=mr.origin_us)
        self.month_real = self._real(
            kind="ens", primary=mr.primary, origin_idx=origin_idx, valid_idx=mr.target_idx
        )
        self.month_null = self._shuffled(
            kind="ens", primary=mr.primary, origin_idx=origin_idx, valid_idx=mr.target_idx
        )
        if hasattr(source, "finish_null"):
            self.month_null = source.finish_null(real=self.month_real, null=self.month_null)  # ty: ignore[call-non-callable]
        train = ~mr.is_val[: mr.n_base]
        if source.prescaled:
            self.mean = np.zeros(self.month_real.shape[1], dtype=np.float32)
            self.std = np.ones(self.month_real.shape[1], dtype=np.float32)
        else:
            self.mean = self.month_real[: mr.n_base][train].mean(axis=0)
            self.std = np.maximum(self.month_real[: mr.n_base][train].std(axis=0), 1e-6)
        self.rows_with_value = None
        if hasattr(source, "present"):
            present = source.present(real=self.month_real)  # ty: ignore[call-non-callable]
            self.rows_with_value = int(present[: mr.n_base][train].sum())
        self.null_rows_without_donor = int(np.isnan(self.month_null).any(axis=1).sum())
        assert self.null_rows_without_donor < 0.01 * len(self.month_null), (
            self.null_rows_without_donor
        )
        i = len(origin_idx) // 2
        self.example = (
            f"origin {datetime.fromtimestamp(mr.origin_us[i] / 1e6, UTC)}, valid {datetime.fromtimestamp(mr.valid_us[i] / 1e6, UTC)}, "
            f"columns {np.round(self.month_real[i], 2).tolist()} from the summary file of the run keyed by this origin"
        )
        assert mr.valid_us[i] > mr.origin_us[i]

    def _real(
        self, *, kind: str, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> np.ndarray:
        """Return the raw columns of a row set; raises when a row has none."""
        if kind == "era5":
            values, found = self.source.era5(primary=primary, valid_idx=valid_idx)
        else:
            values, found = self.source.ens(
                primary=primary, origin_idx=origin_idx, valid_idx=valid_idx
            )
        if not found.all():
            msg = f"{int((~found).sum())} {kind} rows have no extra columns"
            raise ValueError(msg)
        return values

    def _shuffled(
        self, *, kind: str, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> np.ndarray:
        """Return the placebo columns: the donor day's, with fallback donors (NaN if none)."""
        out = np.full(
            (len(primary), self.source.n_weather + self.source.n_spread), np.nan, dtype=np.float32
        )
        todo = np.ones(len(primary), dtype=bool)
        donor = getattr(self.source, "ens_donor", self.source.ens)
        for attempt in range(len(placebo.SEEDS)):
            shift = self.days[kind].shifts(valid_idx=valid_idx, attempt=attempt)
            usable = todo & np.isfinite(shift)
            if not usable.any():
                continue
            s = np.where(usable, shift, 0).astype(np.int64)
            o2, v2 = origin_idx + s, valid_idx + s
            inside = usable & (o2 >= 0) & (v2 < len(self.days[kind].day_id))
            idx = np.nonzero(inside)[0]
            if kind == "era5":
                values, found = self.source.era5(primary=primary[idx], valid_idx=v2[idx])
            else:
                values, found = donor(primary=primary[idx], origin_idx=o2[idx], valid_idx=v2[idx])
            out[idx[found]] = values[found]
            todo[idx[found]] = False
        self.unfilled += int(todo.sum())
        return out

    def scale(self, *, raw: np.ndarray) -> np.ndarray:
        """Standardise raw columns with the fold's training statistics; missing values become 0."""
        out = (raw - self.mean) / self.std
        if self.source.prescaled:
            out = raw
        return np.where(np.isfinite(out), out, 0.0).astype(np.float32)

    def month(self, *, null: bool) -> np.ndarray:
        """Return the standardised columns of every month row."""
        return self.scale(raw=self.month_null if null else self.month_real)

    def rows(
        self,
        *,
        kind: str,
        primary: np.ndarray,
        origin_idx: np.ndarray,
        valid_idx: np.ndarray,
        null: bool,
    ) -> np.ndarray:
        """Return the standardised columns of another row set (`ens` for test rows, `era5` for pretraining rows)."""
        args = {"kind": kind, "primary": primary, "origin_idx": origin_idx, "valid_idx": valid_idx}
        raw = self._shuffled(**args) if null else self._real(**args)
        if null and hasattr(self.source, "finish_null") and kind == "ens":
            raw = self.source.finish_null(real=self._real(**args), null=raw)  # ty: ignore[call-non-callable]
        return self.scale(raw=raw)


class DriftColumns:
    """The knot position `tau` of each row (arm A10), with the same interface as `ExtraColumns`.

    `tau` is the number of months from 2019-09-01 to the row's valid time, fractional within the month.
    The fold has one knot at the first of every month from 2019-09 up to the fold's month (`n_knots`).
    """

    def __init__(self, *, month: str, mr: c.MonthRows) -> None:
        """Count the fold's knots and compute every month row's position."""
        year, mon = (int(part) for part in month.split("-"))
        self.n_knots = (year - 2019) * 12 + (mon - 9) + 1
        self.mr_valid = mr.valid_us
        self.null_rows_without_donor = 0
        i = len(mr.valid_us) // 2
        self.example = f"valid {datetime.fromtimestamp(mr.valid_us[i] / 1e6, UTC)} -> knot position {self.tau(valid_us=mr.valid_us[i : i + 1])[0]:.3f} of {self.n_knots - 1}"

    @staticmethod
    def tau(*, valid_us: np.ndarray) -> np.ndarray:
        """Return the knot position of valid times in microseconds."""
        t = valid_us.astype("datetime64[us]")
        start = t.astype("datetime64[M]")
        lo, hi = start.astype("datetime64[us]"), (start + 1).astype("datetime64[us]")
        frac = (t - lo) / (hi - lo)
        return ((start - np.datetime64("2019-09", "M")).astype(np.float64) + frac).astype(
            np.float32
        )

    def month(self, *, null: bool) -> np.ndarray:
        """Return the position column of every month row."""
        return self.tau(valid_us=self.mr_valid)[:, None]

    def rows(
        self,
        *,
        kind: str,
        primary: np.ndarray,
        origin_idx: np.ndarray,
        valid_idx: np.ndarray,
        null: bool,
    ) -> np.ndarray:
        """Return the position column of another row set from its valid-time grid indices."""
        return self.tau(valid_us=GRID + valid_idx.astype(np.int64) * STEP)[:, None]
