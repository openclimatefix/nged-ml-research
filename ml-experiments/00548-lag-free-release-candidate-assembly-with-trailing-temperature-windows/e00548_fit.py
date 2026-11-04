# ruff: noqa: E501
r"""Fit and forecast arm E of experiment 548: experiment 511's L-1d plus three trailing temperature windows.

Arm E is L-1d (daily refit at 00:00 UTC, 90-day half-life, ERA5 and CAMS pretraining, 2 seeds, five
folds of experiment 511) with the three W-T columns of experiment 515 appended after the 37 lag-free
columns: the means of the ensemble-mean 2 m temperature over the 6, 24, and 72 hours ending at the
valid time. The windows are experiment 515's `previous`-seam ensemble tables, which hold only the
run issued at the forecast origin and earlier runs. Pretraining weights are the W-T weights of
experiments 515 and 526 (2025-12, 2026-01, 2026-03) and this experiment's (2025-10, 2026-06).

Widening by 1.15 is applied at scoring, as in every 511 score. Everything is written under this
experiment's `work/` folder.

Commands: `fit` (daily fits and forecasts), `assemble` (one table per month and seed).
"""

import argparse
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f  # ty: ignore[unresolved-import]
import e00515_features as feat  # ty: ignore[unresolved-import]
import e00515_fit as f515  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00548-lag-free-release-candidate-assembly-with-trailing-temperature-windows"
)
W: Final[Path] = ROOT / "work"
E515_PRE: Final[Path] = f515.W / "pretrained" / "W-T"
COLUMNS: Final[tuple[str, ...]] = feat.ARM_COLUMNS["T"]
ARM_TAG: Final[str] = "E"
CHECK: dict = {"train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}}


def pretrained_file(*, arm: f.Arm, month: str, seed: int) -> Path:
    """Return the W-T pretrained weights of a fold: this experiment's, else experiment 515's."""
    _ = arm
    own = W / "pretrained" / "W-T" / f"{month}_s{seed}.pt"
    if own.exists():
        return own
    old = E515_PRE / f"{month}_s{seed}.pt"
    if old.exists():
        return old
    msg = f"no W-T pretrained weights for {month} seed {seed}"
    raise FileNotFoundError(msg)


class WindowColumns:
    """The three W-T columns of any row set, standardised with the fold's fine-tuning statistics."""

    def __init__(self, *, mr: c.MonthRows) -> None:
        """Load the 16 ensemble window tables and fit the mean and standard deviation on the fold's training rows."""
        self.lookup = f515.EnsembleLookup(seam="previous")
        base = mr.base
        raw = self.raw(primary=base.primary, origin_idx=base.origin_idx, valid_idx=base.target_idx)
        train = ~base.is_val
        self.index = [feat.ALL_COLUMNS.index(n) for n in COLUMNS]
        self.mean = raw[train].mean(axis=0)
        self.std = np.maximum(raw[train].std(axis=0), 1e-6)

    def raw(
        self, *, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> np.ndarray:
        """Return the raw W-T columns of each row, and fail when any row is missing from the tables."""
        values, found = self.lookup.get(primary=primary, origin_idx=origin_idx, valid_idx=valid_idx)
        if not found.all():
            msg = f"{int((~found).sum())} rows have no window columns"
            raise ValueError(msg)
        return values[:, [feat.ALL_COLUMNS.index(n) for n in COLUMNS]]

    def scaled(
        self, *, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> np.ndarray:
        """Return the standardised columns, with a non-finite entry set to 0 as experiment 515 does."""
        raw = self.raw(primary=primary, origin_idx=origin_idx, valid_idx=valid_idx)
        out = (raw - self.mean) / self.std
        return np.where(np.isfinite(out), out, 0.0).astype(np.float32)


def idx_of(*, us: np.ndarray) -> np.ndarray:
    """Return the half-hour grid index of microsecond timestamps."""
    return ((us.astype(np.int64) - c.prep422.GRID_START_US) // c.prep422.STEP_US).astype(np.int64)


def assert_origin_time(*, origin_us: np.ndarray, valid_us: np.ndarray, label: str) -> str:
    """Assert, row by row, that the weather run is the one issued 9 h before the origin.

    The target must also follow the origin. The window table row for (origin, valid) holds only the
    run issued at the origin (its own ensemble mean after the origin) and earlier runs (the analysis
    proxy before it), by construction of
    `e00515_features.ensemble_table` and its causality check.
    """
    pub_us = int(common.PUBLICATION_DELAY.total_seconds() * 1e6)
    assert (valid_us > origin_us).all(), f"{label}: target not after origin"
    runs = np.unique(origin_us)
    miss = [
        r
        for r in runs
        if not (
            prep.SUMMARY_DIR
            / f"init={datetime.fromtimestamp((r - pub_us) / 1e6, UTC):%Y-%m-%d}.parquet"
        ).exists()
    ]
    assert not miss, f"{label}: {len(miss)} origins have no weather-run summary file 9 h earlier"
    i = len(origin_us) // 2
    return (
        f"{label}: origin {datetime.fromtimestamp(origin_us[i] / 1e6, UTC)}, target "
        f"{datetime.fromtimestamp(valid_us[i] / 1e6, UTC)}, weather run "
        f"{datetime.fromtimestamp((origin_us[i] - pub_us) / 1e6, UTC)} (summary file exists for all {len(runs)} origins)"
    )


_build_month, _build_test = k.build_month, data.build_test
STATE: dict = {}


def build_month(*, month: str, clean: np.ndarray) -> c.MonthRows:
    """Return experiment 511's month rows with the three window columns appended and every row checked."""
    mr = _build_month(month=month, clean=clean)
    win = WindowColumns(mr=mr)
    STATE["win"] = win
    STATE["month"] = month
    extra = win.scaled(
        primary=mr.primary, origin_idx=idx_of(us=mr.origin_us), valid_idx=mr.target_idx
    )
    ex = assert_origin_time(origin_us=mr.origin_us, valid_us=mr.valid_us, label=f"{month} train")
    CHECK["train_rows"] += len(mr.y)
    CHECK["months"][month] = {
        "train_rows": len(mr.y),
        "train_example": ex,
        "window_mean": win.mean.tolist(),
        "window_std": win.std.tolist(),
    }
    print(month, "train checks passed;", ex, flush=True)
    return dataclasses.replace(mr, x=np.concatenate([mr.x, extra], axis=1))


def build_test(*, month: str, standardiser: object) -> data.TestRows:
    """Return the month's test rows with the three window columns appended and every test origin checked."""
    test = _build_test(month=month, standardiser=standardiser)
    assert STATE["month"] == month, "window statistics belong to another month"
    win: WindowColumns = STATE["win"]
    extra = win.scaled(primary=test.primary, origin_idx=test.origin_idx, valid_idx=test.target_idx)
    ex = assert_origin_time(origin_us=test.origin, valid_us=test.valid, label=f"{month} test")
    pub = common.PUBLICATION_DELAY
    for o in np.unique(test.origin):
        origin = datetime.fromtimestamp(o / 1e6, UTC)
        run = origin - pub
        fr = pl.read_parquet(common.demand_frame_path(run=run))
        assert (fr["nwp_init_time"] <= fr["power_fcst_init_time"]).all()
        assert (fr["power_fcst_init_time"] == origin).all()
        CHECK["origins_weather_checked"] += 1
    CHECK["test_rows"] += len(test.x)
    CHECK["months"][month]["test_example"] = ex
    print(month, "test checks passed;", ex, flush=True)
    return dataclasses.replace(test, x=np.concatenate([test.x, extra], axis=1))


k.build_month, data.build_test = build_month, build_test
f.W = W
f.pretrained_file = pretrained_file
f.model_file = lambda *, arm, month, day, seed, stress="": (
    W / "models" / arm.tag / f"{month}_d{day}_s{seed}{stress}.pt"
)


def assemble() -> None:
    """Concatenate each month's daily window forecasts into one table per month and seed."""
    for month in c.MONTHS:
        cuts = k.cutoffs(month=month, cadence="d")
        for seed in (0, 1):
            parts = [
                W / "fits" / ARM_TAG / f"{month}_d{f.day_of(month=month, cut=cut)}_s{seed}.parquet"
                for cut in cuts
            ]
            assert all(p.exists() or p.with_suffix(".empty").exists() for p in parts), (month, seed)
            frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
            f.atomic_frame(
                frame=frame, target=W / "seed_preds" / ARM_TAG / f"{month}_s{seed}.parquet"
            )
            print("wrote", ARM_TAG, month, seed, frame.height, flush=True)


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "assemble"))
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.command == "assemble":
        assemble()
        return
    arm = f.Arm(ARM_TAG, "d", 90.0)
    f.fit_arm(
        arm=arm,
        months=args.months,
        seeds=args.seeds,
        stress=args.stress,
        limit=args.limit,
        device="cuda",
    )
    out = ROOT / f"checks_{ARM_TAG}{'_stress' if args.stress else ''}_{'-'.join(args.months)}.json"
    out.write_text(json.dumps(CHECK, indent=1, default=str))
    print("CHECKS PASSED", {key: v for key, v in CHECK.items() if key != "months"})


if __name__ == "__main__":
    main()
