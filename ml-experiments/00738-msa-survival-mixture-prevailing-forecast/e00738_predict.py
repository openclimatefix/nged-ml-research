# ruff: noqa: E501
"""Predict one fold and one seed with every arm: the 16 scoring worlds and the real record.

One job per (fold, seed) loads the stored `M1` state, the fitted W0 state, and the four head arms'
states, builds the fold's data once in the prediction phase, and for each arm and each panel
(16 scoring-world passes, the resampled-duration world, and the real record with no shift) writes
one parquet file under `predictions/<arm>/<month>_s<seed>/`, atomically. A check re-runs the stored
`M1` on the real record and compares it with experiment 713's stored forecasts to 1e-5.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import e00738_common as common
import e00738_panel as panel
import e00738_world as world
import numpy as np
import polars as pl

REPRODUCTION_TOLERANCE: Final[float] = 1e-5
PANELS_REAL: Final[str] = "real"


def panel_path(*, arm: str, month: str, seed: int, name: str, out: Path | None = None) -> Path:
    """Return the forecast file of one arm, fold, seed, and panel (`real`, `pass00`, or `resampled00`)."""
    root = out if out is not None else common.OUT
    return root / "predictions" / arm / f"{month}_s{seed}" / f"{name}.parquet"


def write_parquet_atomic(*, frame: pl.DataFrame, path: Path) -> None:
    """Write a parquet file under a temporary name and rename it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.write_parquet(tmp)
    tmp.rename(path)


def reproduce_stored_m1(*, frame: pl.DataFrame, month: str, seed: int) -> dict[str, float]:
    """Compare the stored-`M1` re-run on the real record with experiment 713's stored forecast.

    Args:
        frame: The re-run's forecast frame on the real record (native-origin keys).
        month: The fold month.
        seed: The seed.

    Returns:
        The largest absolute difference over the quantiles (capacity shares), and the row count.

    Raises:
        AssertionError: When the largest difference is above 1e-5.
    """
    stored = pl.read_parquet(common.ROOT_713 / "work" / "fits" / "M1" / f"{month}_s{seed}.parquet")
    keys = list(panel.KEYS)
    joined = frame.join(stored.select(*keys, *panel.QCOLS), on=keys, how="inner", suffix="_stored")
    assert joined.height > 0, "the re-run and the stored forecast share no key"
    worst = max(
        float((joined[c] - joined[f"{c}_stored"]).abs().max())  # ty: ignore[invalid-argument-type]
        for c in panel.QCOLS
    )
    assert worst <= REPRODUCTION_TOLERANCE, f"stored M1 is not reproduced: {worst}"
    return {"max_abs_difference": worst, "rows": float(joined.height)}


def all_panel_shifts(
    *, month: str, month_start: int, window_end: int, is_mw: np.ndarray, real_days: np.ndarray
) -> dict[str, list[world.Shift]]:
    """Return the 16 shifts of every scoring panel of a fold: 16 passes, then the resampled world.

    Args:
        month: The fold month.
        month_start: The grid index of the first day of the month.
        window_end: The first grid index that no shift may reach.
        is_mw: Per Primary, whether it reports signed megawatts.
        real_days: The durations of the 66 real transfers, in days.

    Returns:
        The shifts by panel name (`pass00` to `pass11`, `pass200` to `pass203`, `resampled00`).
    """
    key = month_idx_650(month=month)
    out: dict[str, list[world.Shift]] = {}
    for r in world.scoring_passes():
        out[f"pass{r:02d}"] = [
            world.draw_scoring_shift(
                month_idx=key,
                r=r,
                primary=p,
                month_start_idx=month_start,
                window_end_idx=window_end,
                is_mw=is_mw,
            )
            for p in range(16)
        ]
    out["resampled00"] = [
        world.draw_resampled_shift(
            month_idx=key,
            r=0,
            primary=p,
            month_start_idx=month_start,
            window_end_idx=window_end,
            is_mw=is_mw,
            real_days=real_days,
        )
        for p in range(16)
    ]
    return out


def run_job(
    *,
    month: str,
    seed: int,
    device: str,
    arms: tuple[str, ...] = common.ALL_ARMS,
    only: tuple[str, ...] | None = None,
) -> None:
    """Predict every arm and panel of one fold and seed, and write the files.

    Args:
        month: The fold month.
        seed: The seed.
        device: `cuda` or `cpu`.
        arms: The arms to predict.
        only: The panel names to predict; `None` predicts every panel.
    """
    os.environ["E00667_PHASE"] = "predict"
    panel.assert_713_code()
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00713_arms as arms713  # ty: ignore[unresolved-import]
    import e00713_inputs as inp  # ty: ignore[unresolved-import]
    import e00738_fit as fit
    import torch  # ty: ignore[unresolved-import]
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    pending = [
        a
        for a in arms
        if not all(
            panel_path(arm=a, month=month, seed=seed, name=n).exists()
            for n in panel_names()
            if only is None or n in only
        )
    ]
    if not pending:
        return
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    saved = (
        arms713.root(data_dir=ML_EXPERIMENTS_DATA_DIR) / "work" / "standardiser" / f"{month}.sha256"
    )
    assert saved.exists(), "the standardiser fingerprint is missing"
    assert saved.read_text() == data.fingerprint, "standardiser differs from the training phase"
    nb_of, capacity, is_mw = fit.fold_world()
    del nb_of, capacity
    sites = k578.sites().sort("primary")
    ids = sites["time_series_id"].to_numpy()
    native = data.test_origin_idx
    assert native is not None
    month_start = data.cutoff_idx
    window_end = fit.window_end_idx()
    clean = data.clean
    real_days = world.load_real_durations_days()
    shifts_by_panel = all_panel_shifts(
        month=month,
        month_start=month_start,
        window_end=window_end,
        is_mw=is_mw,
        real_days=real_days,
    )
    panels: dict[str, np.ndarray] = {PANELS_REAL: np.zeros_like(clean, dtype=np.float32)}
    for name, shifts in shifts_by_panel.items():
        panels[name] = world.build_own(clean=clean, shifts=shifts, is_mw=is_mw)
    if only is not None:
        panels = {k: v for k, v in panels.items() if k in only}
    for arm in pending:
        model, model_fn = fit.load_for_prediction(
            arm=arm, month=month, seed=seed, data=data, device=device
        )
        for name, own in panels.items():
            path = panel_path(arm=arm, month=month, seed=seed, name=name)
            if path.exists():
                continue
            src = data.source.with_power_in(
                power_in=torch.from_numpy(
                    (
                        (world.inject(clean=clean, own=own) - data.target_mean[:, None])
                        / data.target_std[:, None]
                    ).astype(np.float32)
                ).to(device)
            )
            if name != PANELS_REAL:
                src = type(src)(
                    power_z=src.power_in,
                    dense_weather=src.dense_weather,
                    dense_datetime=src.dense_datetime,
                    day_type=src.day_type,
                    ens=src.ens,
                    power_in=src.power_in,
                )
            frame, diag = panel.forecast_pass(
                model_fn=model_fn,
                src=src,
                data=data,
                primaries=np.arange(16),
                native=native,
                lead_days=panel.SCORED_LEAD_DAYS,
                ids=ids,
            )
            if name == PANELS_REAL and arm == common.ARM_M1:
                rep = reproduce_stored_m1(frame=frame, month=month, seed=seed)
                common.atomic_json(
                    obj=rep, path=common.OUT / "checks" / f"reproduce_m1_{month}_s{seed}.json"
                )
            write_parquet_atomic(frame=frame, path=path)
            if diag.height:
                write_parquet_atomic(frame=diag, path=path.with_name(f"{name}_diag.parquet"))
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def month_idx_650(*, month: str) -> int:
    """Return experiment 650's month key: the grid index of the month's start (it keys the draw)."""
    import e00650_common as c650  # ty: ignore[unresolved-import]

    return int(c650.month_start_idx(month=month))


def panel_names() -> list[str]:
    """Return the names of the panels of one job."""
    return [PANELS_REAL, *(f"pass{r:02d}" for r in world.scoring_passes()), "resampled00"]
