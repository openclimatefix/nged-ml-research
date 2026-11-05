# ruff: noqa: E501
"""The extra input columns of experiment 569, for pre-training rows (ERA5 and CAMS stand-ins) and fine-tuning rows.

One `Extras` object serves one arm (`B1`, `B1p`, `B2`, ..., `B3c`, `B4`, `B4p`; a `p` suffix is the
shuffled-day placebo of the same width). It builds each column group in both stages from the same
sources, and standardises with one mean and standard deviation per fold, fitted on the pre-training
rows' real columns, so a network pre-trained on the columns reads them at the same scale when it is
fine-tuned. Fine-tuning rows take their columns from experiment 562's `FoldColumns` (experiment 437's
stored tables of ensemble-run columns); pre-training rows have no ensemble run, so their columns
come from the dense ERA5 and CAMS weather (`e00437_features.dense_arrays`) or from the calendar and
the stored daily signals, exactly as experiment 437's `pretrain_extras` does for its T, S, and F bundles.

Presence flags: a school calendar exists from 2021-09, national signals from 2023-01, and the price
shape from 2023, but pre-training history starts in 2019-09. Pre-training rows without the source
get all-zero columns and a presence flag of 0 (a flag column is appended for B3a, B3b, and B3c).
"""

from typing import Final

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00437_features as f437  # ty: ignore[unresolved-import]
import e00562_columns as c562  # ty: ignore[unresolved-import]
import numpy as np

ARMS: Final[tuple[str, ...]] = ("B1", "B2", "B3a", "B3b", "B4", "B3c")
ROUTE: Final[dict[str, str]] = {
    "B1": "weather", "B2": "weather", "B3a": "datetime", "B3c": "datetime", "B3b": "parallel", "B4": "parallel",
}  # fmt: skip
SCHOOL: Final[tuple[str, ...]] = c562.SCHOOL
NATIONAL: Final[tuple[str, ...]] = c562.NATIONAL
COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "B1": c562.TABLE_ARMS["B1"], "B2": c562.TABLE_ARMS["B2"],
    "B3a": (*SCHOOL, "school_present"), "B3b": (*NATIONAL, "national_present"),
    "B3c": (*SCHOOL, "school_present", *[f"{c}_x_schools" for c in SCHOOL]),
}  # fmt: skip
PRICE_WIDTH: Final[int] = c562.PRICE_WIDTH
SCHOOL_RADIUS_KM: Final[str] = "schools_2km"


def base_arm(*, arm: str) -> str:
    """Return the real arm of an arm name: `B3ap` is `B3a`."""
    return arm.removesuffix("p") if arm not in ARMS and arm.endswith("p") else arm


def is_placebo(*, arm: str) -> bool:
    """Return True for a placebo arm name."""
    return arm not in ARMS and arm.endswith("p")


def width(*, arm: str) -> int:
    """Return the number of extra input columns of an arm."""
    base = base_arm(arm=arm)
    return PRICE_WIDTH if base == "B4" else len(COLUMNS[base])


class Extras:
    """Extra columns of one arm in both stages; holds per-process caches of the dense sources."""

    def __init__(self, *, arm: str, proximity: dict[int, float]) -> None:
        """Set the arm; `proximity` maps a Primary index to its share of the largest schools-within-2-km count (B3c)."""
        self.arm, self.base, self.placebo = arm, base_arm(arm=arm), is_placebo(arm=arm)
        self.proximity = proximity
        self.local_date = f437.local_date_codes()
        self.dense: dict[int, dict[str, np.ndarray]] = {}
        self.rows = {int(r["primary"]): r for r in prep.primaries().iter_rows(named=True)}
        self.school = f437.school_columns()
        self.national = f437.national_daily()
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    # --- raw columns ---------------------------------------------------------------------

    def _dense(self, *, primary: int) -> dict[str, np.ndarray]:
        if primary not in self.dense:
            self.dense[primary] = f437.dense_arrays(
                row=self.rows[primary], local_date=self.local_date
            )
        return self.dense[primary]

    def _gate(self, *, primary: np.ndarray) -> np.ndarray:
        return np.array([self.proximity[int(p)] for p in primary], dtype=np.float32)

    def raw_pretrain(
        self,
        *,
        primary: np.ndarray,
        origin_idx: np.ndarray,
        target_idx: np.ndarray,
        shift: np.ndarray | None,
    ) -> np.ndarray:
        """Return the arm's unstandardised columns for pre-training rows (NaN where the source has no value).

        `shift` is each row's donor-day shift in half-hours (NaN where there is no donor), or None for
        the real arm. A placebo moves the row's origin and valid time by the shift together, and
        B3b by the origin's shift alone, as experiment 562 did.
        """
        base, n = self.base, len(primary)
        if shift is not None:
            ok = np.isfinite(shift)
            s = np.where(ok, shift, 0).astype(np.int64)
        else:
            ok, s = np.ones(n, dtype=bool), np.zeros(n, dtype=np.int64)
        o2, v2 = origin_idx + s, target_idx + s
        grid = len(self.local_date)
        ok &= (o2 >= 0) & (v2 < grid) & (v2 >= 0)
        o2, v2 = np.clip(o2, 0, grid - 1), np.clip(v2, 0, grid - 1)
        if base in {"B1", "B2"}:
            out = np.full((n, len(COLUMNS[base])), np.nan, dtype=np.float32)
            for p in np.unique(primary):
                sel = np.nonzero((primary == p) & ok)[0]
                if not len(sel):
                    continue
                out[sel] = f437.pretrain_extras(
                    row=self.rows[int(p)], dense=self._dense(primary=int(p)), columns=COLUMNS[base],
                    valid_us=common_us(idx=v2[sel]), origin_us=common_us(idx=o2[sel]), max_steps=prep418.MAX_STEPS,
                )  # fmt: skip
            return out
        if base in {"B3a", "B3c"}:
            first, dense = self.school
            codes = self.local_date[v2]
            cols = [f437.lookup(first=first, dense=dense[c], codes=codes) for c in SCHOOL]
            out = np.stack(cols, axis=1).astype(np.float32)
            present = np.isfinite(out[:, 0]).astype(np.float32)
            out[~ok] = np.nan
            present[~ok] = 0.0
            if base == "B3a":
                return np.concatenate([out, present[:, None]], axis=1)
            gate = self._gate(primary=primary)[:, None]
            return np.concatenate([out, present[:, None], np.nan_to_num(out) * gate], axis=1)
        if base == "B3b":
            first, dense = self.national
            codes = self.local_date[o2] - 1
            out = np.stack(
                [f437.lookup(first=first, dense=dense[c], codes=codes) for c in NATIONAL], axis=1
            ).astype(np.float32)
            out[~ok] = np.nan
            present = np.isfinite(out[:, 0]).astype(np.float32)
            return np.concatenate([out, present[:, None]], axis=1)
        raise ValueError(base)

    # --- standardisation ----------------------------------------------------------------

    def fit(self, *, raw: np.ndarray) -> None:
        """Set the fold's mean and standard deviation from the real pre-training columns (presence flags stay 0 or 1)."""
        mean = np.nanmean(raw, axis=0)
        std = np.maximum(np.nanstd(raw, axis=0), 1e-6)
        for i, name in enumerate(COLUMNS[self.base]):
            if name.endswith("_present"):
                mean[i], std[i] = 0.0, 1.0
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)

    def scale(self, *, raw: np.ndarray) -> np.ndarray:
        """Return standardised columns, with a missing value as 0 (after standardising)."""
        assert self.mean is not None
        z = (raw - self.mean) / self.std
        return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


def common_us(*, idx: np.ndarray) -> np.ndarray:
    """Return the microsecond timestamp of grid half-hour indexes."""
    return prep422.GRID_START_US + idx.astype(np.int64) * prep422.STEP_US
