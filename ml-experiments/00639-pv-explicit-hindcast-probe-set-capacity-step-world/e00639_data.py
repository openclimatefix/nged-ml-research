# ruff: noqa: E501
"""Rows of experiment 639: the training rows of experiment 634 and a new builder for the test window.

Per row the module adds the physics unit curve `u`, the inverse capacity `inc`, the R4 table value
`r4` in megawatts, and the step indicator `after`. A test row is an ERA5-and-CAMS row whose valid
time lies in the 60 days after the cutoff, with the first origin kept for each Primary and valid time.
"""

from dataclasses import dataclass

import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00634_common as k634  # ty: ignore[unresolved-import]
import e00638_common as k638  # ty: ignore[unresolved-import]
import e00639_common as k
import numpy as np
import polars as pl
from e00397_prep import primaries  # ty: ignore[unresolved-import]


@dataclass
class Block:
    """Rows with everything an explicit network needs. `y` is the net target in points, step included."""

    x: np.ndarray
    y: np.ndarray
    primary: np.ndarray
    target_idx: np.ndarray
    is_val: np.ndarray
    t_us: np.ndarray
    u: np.ndarray
    u_shuf: np.ndarray
    inc: np.ndarray
    r4: np.ndarray
    after: np.ndarray
    kt: np.ndarray
    hour: np.ndarray


def _decorate(
    *, cutoff: str, world: str, x: np.ndarray, y: np.ndarray, primary: np.ndarray,
    target_idx: np.ndarray, is_val: np.ndarray, inj: k638.Injector,
) -> Block:  # fmt: skip
    """Add the physics curve, the placebo curve, the table value, and the step indicator to rows."""
    t_us = inj.t_us
    u = k634.unit_curve(ghi=inj.ghi, x=x, times_us=t_us, temp=inj.temp)
    donor = _shuffle_donors(primary=primary, t_us=t_us)
    u_shuf = k634.unit_curve(ghi=inj.ghi[donor], x=x, times_us=t_us, temp=inj.temp[donor])
    cap = inj.cap
    inc = np.where(inj.in_six, 1.0 / cap[primary], 0.0)
    table = k.r4_table()
    sy = k.season_year(t_us=t_us, cutoff=cutoff)
    r4 = np.zeros(len(primary))
    years = np.arange(k.SEASON_YEARS[0], k.SEASON_YEARS[1] + 1)
    for site, idx in inj.six.items():
        by_year = np.array([table[(site, int(y_))] for y_ in years])
        sel = primary == idx
        r4[sel] = by_year[sy[sel] - years[0]]
    start = k638.parse(name="step_spring23_mw4").start_us
    after = ((t_us >= start) & (world == "step")).astype(np.float64)
    cz = x[:, 14].astype(np.float64)
    clear = k634.haurwitz_w_m2(cos_zenith=cz)
    kt = np.where(clear > 10.0, inj.ghi / np.maximum(clear, 1e-9), 0.0)
    hour = (t_us // 3_600_000_000) % 24
    return Block(x, y, primary, target_idx, is_val, t_us, u, u_shuf, inc, r4, after, kt, hour)


def _shuffle_donors(*, primary: np.ndarray, t_us: np.ndarray) -> np.ndarray:
    """Return, per row, the index of a row of another date.

    The donor has the same Primary, the same half-hour, and the same 21-day block of the year.
    """
    stamp = pl.Series(t_us).cast(pl.Datetime("us"))
    doy = stamp.dt.ordinal_day().to_numpy().astype(np.int64)
    half_hour = (stamp.dt.hour().to_numpy() * 2 + stamp.dt.minute().to_numpy() // 30).astype(
        np.int64
    )
    key = (primary * 100 + half_hour) * 100 + doy // k.SHUFFLE_BLOCK_DAYS
    order = np.argsort(key, kind="stable")
    rng = np.random.default_rng(639)
    donor = np.arange(len(key))
    bounds = np.flatnonzero(np.diff(key[order])) + 1
    for group in np.split(order, bounds):
        if len(group) > 1:
            donor[group] = rng.permutation(group)
    return donor


def build_train(*, cutoff: str, world: str) -> tuple[Block, data.Rows, np.ndarray]:
    """Return the training rows (farm-gap rows dropped), the fold's rows, and the clean stack."""
    clean = c.clean_stack()
    refit = c.month_start(month=cutoff)
    rows = data.build_fold(refit=refit, clean=clean)
    pre = r448.pretrain_rows(rows=rows, refit=refit, clean=clean)
    inj = k638.Injector(
        primary=pre.primary, target_idx=pre.target_idx, x=pre.x, standardiser=rows.standardiser
    )
    keep = ~inj.gap
    change = (
        inj.change_points_step(condition=k638.parse(name="step_spring23_mw4"))
        if world == "step"
        else 0.0
    )
    y = pre.y.astype(np.float64) + change
    block = _decorate(
        cutoff=cutoff, world=world, x=pre.x, y=y, primary=pre.primary, target_idx=pre.target_idx,
        is_val=pre.is_val, inj=inj,
    )  # fmt: skip
    return _select(block=block, keep=keep), rows, clean


def _select(*, block: Block, keep: np.ndarray) -> Block:
    """Return the rows of a block where `keep` holds."""
    return Block(*(getattr(block, f)[keep] for f in block.__dataclass_fields__))


def build_test(*, cutoff: str, world: str, rows: data.Rows, clean: np.ndarray) -> Block:
    """Return the test rows: valid time in the window after the cutoff, one row per Primary and valid time.

    Rows follow `pretrain_rows` (the fold's standardiser, and the median spread by lead band for the
    `std_*` columns) with the valid-time filter moved to the window and no purge. Targets are the
    cleaned power, finite only, plus the step in the step world. Farm-gap rows are dropped as in training.
    """
    start_us, stop_us = k.window_us(cutoff=cutoff)
    keep_cols, _, _ = data.non_lag_layout()
    table, std_cols = r448.spread_table(rows=rows)
    grid = data.grid_length()
    parts = []
    for row in primaries().iter_rows(named=True):
        z = np.load(prep418.RAW_DIR / f"{row['site']}.npz")
        sel = (z["valid"] >= start_us) & (z["valid"] < stop_us) & (z["origin"] <= z["valid"])
        target_idx = ((z["valid"] - prep422.GRID_START_US) // prep422.STEP_US).astype(np.int64)
        sel &= (target_idx >= 0) & (target_idx < grid)
        y = np.full(len(sel), np.nan)
        y[sel] = clean[row["primary"]][target_idx[sel]]
        sel &= np.isfinite(y)
        if not sel.any():
            continue
        _, first = np.unique(target_idx[sel], return_index=True)
        idx = np.flatnonzero(sel)[first]
        raw = z["x"][idx][:, keep_cols]
        band = np.minimum(
            (z["lead"][idx] // prep418.LEAD_BAND_HOURS).astype(int), prep418.N_BANDS - 1
        )
        x = rows.standardiser.apply(x=raw)
        x[:, std_cols] = table[band]
        parts.append(
            {
                "x": x.astype(np.float32), "y": y[idx], "target_idx": target_idx[idx],
                "primary": np.full(len(idx), row["primary"], dtype=np.int64),
            }
        )  # fmt: skip
    cat = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
    inj = k638.Injector(
        primary=cat["primary"],
        target_idx=cat["target_idx"],
        x=cat["x"],
        standardiser=rows.standardiser,
    )
    change = (
        inj.change_points_step(condition=k638.parse(name="step_spring23_mw4"))
        if world == "step"
        else 0.0
    )
    block = _decorate(
        cutoff=cutoff, world=world, x=cat["x"], y=cat["y"].astype(np.float64) + change,
        primary=cat["primary"], target_idx=cat["target_idx"],
        is_val=np.zeros(len(cat["y"]), dtype=bool), inj=inj,
    )  # fmt: skip
    return _select(block=block, keep=~inj.gap)
