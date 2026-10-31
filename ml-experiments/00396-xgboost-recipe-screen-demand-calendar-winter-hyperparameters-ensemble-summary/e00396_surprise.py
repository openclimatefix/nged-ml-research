r"""Window-surprise arms WS A1 to A3 of experiment 396: experiment 391's stage 2 plus window means.

Experiment 391 fits a monthly XGBoost member with three lag-surprise columns (`u_lag_origin`,
`u_lag_day`, `u_lag_week`). Its control `b0` adds nothing. These arms add means of the same
surprise `u` and of the signed stage-1 residual `r_mw` over longer windows. Nothing else changes:
the stage-1 tables of experiment 391 are read, only stage 2 is fitted, at 150 rounds and the same
monthly refits, so the files land beside `b0` and `b0s1` and experiment 391's scorer can read them.

- `ws_a1`: the 391 `own` columns plus the mean of `u` over the previous 6, 12, 24, and 168 hours.
- `ws_a2`: the 391 `own` columns plus the mean of `r_mw` over the same four windows.
- `ws_a3`: the 391 `own` columns plus both sets of windows.

Every window ends at the newest telemetry the origin may read (the origin minus
`common.NEWEST_AGE` half-hours), as the 391 columns do, so no column reads a time after the origin.
A window mean is NaN where fewer than 75% of its half-hours are finite.

The fit machinery is experiment 391's `e00391_fit`: this module registers the arms there, wraps its
`add_columns`, and calls its `main`. The arms run on the seven subset Primaries of 391's secondary
arms.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00391-xgboost-lag-surprise-second-stage:\\
$(ls -d ml-experiments/00396-*) \\
    E391_DEVICE=cpu uv run --no-sync python \\
    ml-experiments/00396-*/e00396_surprise.py --arms ws_a1 ws_a2 ws_a3 --workers 1 --cores 4
```
"""

from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00391_fit as fit  # ty: ignore[unresolved-import]
import e00391_stage1 as stage1  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

WINDOW_HOURS: Final[tuple[int, ...]] = (6, 12, 24, 168)
"""The previous-window lengths, in hours, of the window means."""
MIN_SHARE: Final[float] = 0.75
"""The share of a window's half-hours that must be finite for its mean to be defined."""
U_COLUMNS: Final[tuple[str, ...]] = tuple(f"u_mean_{h}h" for h in WINDOW_HOURS)
R_COLUMNS: Final[tuple[str, ...]] = tuple(f"r_mean_{h}h" for h in WINDOW_HOURS)
GROUP_COLUMNS: Final[dict[str, tuple[str, ...]]] = {"window_u": U_COLUMNS, "window_r": R_COLUMNS}
ARM_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "ws_a1": ("window_u",),
    "ws_a2": ("window_r",),
    "ws_a3": ("window_u", "window_r"),
}

_RESIDUALS: dict[tuple[str, str], np.ndarray] = {}
_ORIGINAL_ADD_COLUMNS = fit.add_columns
_ORIGINAL_COLUMNS = fit.Arm.columns
_ORIGINAL_RUN_JOB = fit.run_job


def window_mean(*, values: np.ndarray, origins: np.ndarray, window: int) -> np.ndarray:
    """Return the mean of `values` over the `window` half-hours before each origin index.

    The window is `[o - window, o)`. The result is NaN where fewer than `MIN_SHARE` of the
    half-hours are finite, and for origins closer than `window` to the start of the grid.
    """
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values.astype(np.float64)))])
    ccount = np.concatenate([[0], np.cumsum(np.isfinite(values))])
    start = origins - window
    clipped = np.clip(start, 0, None)
    total = csum[origins] - csum[clipped]
    count = ccount[origins] - ccount[clipped]
    enough = (start >= 0) & (count >= MIN_SHARE * window)
    return np.where(enough, total / np.maximum(count, 1), np.nan)


def residuals(*, site: str, version: str) -> np.ndarray:
    """Return the stage-1 residual `r_mw` of a site on experiment 87's grid, NaN where absent."""
    key = (site, version)
    if key not in _RESIDUALS:
        table = pl.read_parquet(stage1.STAGE1_DIR / version / f"{site}.parquet")
        index = np.asarray(common.grid_index(times=table["valid_time"]))
        series = np.full(common.grid_times().len(), np.nan, np.float32)
        series[index] = table["r_mw"].to_numpy()
        _RESIDUALS.clear()
        _RESIDUALS[key] = series
    return _RESIDUALS[key]


def add_columns(
    *, frame: pl.DataFrame, site: str, arm: fit.Arm, ctx: fit.Context, mask: np.ndarray
) -> pl.DataFrame:
    """Add experiment 391's `own` columns, then this arm's window means, to a frame."""
    out = _ORIGINAL_ADD_COLUMNS(frame=frame, site=site, arm=arm, ctx=ctx, mask=mask)
    groups = [g for g in arm.groups if g in GROUP_COLUMNS]
    if not groups:
        return out
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    ends = origin - common.NEWEST_AGE + 1
    sources = {"window_u": ctx.u[site], "window_r": residuals(site=site, version=arm.version)}
    columns = {
        name: window_mean(values=sources[g], origins=ends, window=2 * hours)
        for g in groups
        for name, hours in zip(GROUP_COLUMNS[g], WINDOW_HOURS, strict=True)
    }
    return out.hstack(pl.DataFrame({k: pl.Series(v).cast(pl.Float32) for k, v in columns.items()}))


def columns_of(self: fit.Arm) -> list[str]:
    """Return experiment 391's columns for the groups it knows, then this arm's window columns."""
    base = _ORIGINAL_COLUMNS(
        fit.Arm(self.name, tuple(g for g in self.groups if g not in GROUP_COLUMNS), self.version)
    )
    return [*base, *(c for g in self.groups if g in GROUP_COLUMNS for c in GROUP_COLUMNS[g])]


def register() -> None:
    """Register the arms, the wrapped `add_columns`, and the column order in experiment 391."""
    for name, groups in ARM_GROUPS.items():
        fit.ARMS[name] = fit.Arm(name, ("own", *groups), subset=True)
    fit.add_columns = add_columns
    fit.Arm.columns = columns_of  # ty: ignore[invalid-assignment]


def run_job(**kwargs: object) -> list[str]:
    """Run experiment 391's `run_job` in a worker process that has registered the arms."""
    return _ORIGINAL_RUN_JOB(**kwargs)


def main() -> None:
    """Fit the window-surprise arms through experiment 391's command line."""
    fit.run_job = run_job
    fit.main()


register()

if __name__ == "__main__":
    main()
