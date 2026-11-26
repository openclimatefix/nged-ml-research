# ruff: noqa: E501
"""Experiment 733: prediction with a fitted network, in a process apart from the fit (torch).

Real folds: `forecast_frame` is experiment 716's `forecast_frame` with the untangled source in place
of
the PV source. It forecasts every (Primary, scored origin, lead day) of a fold and returns experiment
713's columns, so the frame of F0r is compared with M*'s stored file row by row. A prediction with every
channel masked (reading R4) or with the stored outputs in place of the re-run (reading R12) is the same
function on a source built from other tensors.

Synthetic world: `predict_synthetic` forecasts every sample of a split and returns the probability of
reverse flow, the mixture's quantiles, and the sample's keys, which `write_synthetic` stores as one
compressed archive.

Every tensor leaves the device through `train.to_numpy` (`detach`, `cpu`, then `numpy`).
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import e00733_core as core
import e00733_model as model
import e00733_paths as paths  # noqa: F401
import e00733_synth as sy
import e00733_train as train
import numpy as np

# isort: split
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import polars as pl
import torch  # ty: ignore[unresolved-import]

QCOLS: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
PREDICT_BATCH: Final = 512
FRAME_COLUMNS: Final = (
    "time_series_id", "power_fcst_init_time", "valid_time", *QCOLS, "seed", "mean", "actual",
    "lead_hours", "nwp_init_time",
)  # fmt: skip


def forecast_frame(
    *,
    net: model.UntangledNet,
    test_origin_idx: np.ndarray,
    source: train.UntSource,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    seed: int,
    grid_start_us: int,
    time_series_ids: np.ndarray,
    holdout_start: object | None = None,
    n_primaries: int = 16,
) -> pl.DataFrame:  # fmt: skip
    """Return the forecast frame of all scored origins, every Primary, and 14 lead days (713's columns).

    Args:
        net: The fitted network.
        test_origin_idx: The grid index of each scored native origin (09:00 UTC).
        source: The untangled source of the fold (its tensors decide whether the channels are read).
        target_mean: `(16,)` each Primary's target mean.
        target_std: `(16,)` each Primary's target standard deviation.
        seed: The seed, written as a column.
        grid_start_us: The grid start of experiment 422 in microseconds.
        time_series_ids: `(16,)` the `time_series_id` of each Primary.
        holdout_start: Valid times at or after this datetime are dropped (default: the experiment store's).
        n_primaries: The number of Primaries.

    Returns:
        The frame, sorted by Primary, origin, and valid time.
    """
    if holdout_start is None:
        from studies.experiment_store import HOLDOUT_START

        holdout_start = HOLDOUT_START
    c_all = test_origin_idx + lay.NATIVE_OFFSET_STEPS
    prim, cc, lead = np.meshgrid(
        np.arange(n_primaries), c_all, np.arange(lay.N_LEAD_DAYS), indexing="ij"
    )
    prim, cc, lead = prim.ravel(), cc.ravel(), lead.ravel()
    samples = tr.SampleSet(primary=prim, c_idx=cc, is_val=np.zeros(len(prim), bool))
    net.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(prim), PREDICT_BATCH):
            sel = np.arange(start, min(start + PREDICT_BATCH, len(prim)))
            b = train.batch_for(
                source=source, samples=samples, index=sel, lead=lead[sel], cut=10**9
            )
            inp = b.inputs
            ens = source.ens
            assert ens is not None
            filled = ens.filled[  # ty: ignore[unresolved-attribute]
                inp.primary[:, None],
                (b.c_idx // lay.STEPS_PER_DAY - ens.first_day)[:, None],  # ty: ignore[unresolved-attribute]
                (lay.STEPS_PER_DAY * inp.lead_day[:, None] + source.steps),
            ]
            assert bool(filled.all()), "a scored row has no ensemble inputs"
            out = net(inp).out
            tstd = torch.from_numpy(target_std).to(out.device)[inp.primary]
            tmean = torch.from_numpy(target_mean).to(out.device)[inp.primary]
            mean = mdl.standardised_mixture_mean(out=out) * tstd[:, None] + tmean[:, None]
            quant = mdl.standardised_quantiles(out=out) * tstd[:, None, None] + tmean[:, None, None]
            quant = torch.sort(quant, dim=-1).values
            actual = source.power_z[inp.primary[:, None], b.target_idx]
            parts.append(
                (
                    sel, train.to_numpy(tensor=quant), train.to_numpy(tensor=b.target_idx),
                    train.to_numpy(tensor=actual), train.to_numpy(tensor=mean),
                )
            )  # fmt: skip
    return _assemble(
        parts=parts, prim=prim, cc=cc, seed=seed, grid_start_us=grid_start_us,
        ids=time_series_ids, target_mean=target_mean, target_std=target_std, holdout_start=holdout_start,
    )  # fmt: skip


def _assemble(
    *, parts: Sequence[tuple], prim: np.ndarray, cc: np.ndarray, seed: int, grid_start_us: int,
    ids: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray, holdout_start: object,
) -> pl.DataFrame:  # fmt: skip
    """Return the frame of the collected batches (the columns of experiment 713's stored files)."""
    rows = []
    for sel, q, tidx, actual, m in parts:
        pr = np.repeat(prim[sel], core.STEPS_PER_DAY)
        c = np.repeat(cc[sel], core.STEPS_PER_DAY)
        rows.append(
            pl.DataFrame({
                "time_series_id": ids[pr].astype(np.int32),
                "origin_us": grid_start_us + (c - lay.NATIVE_OFFSET_STEPS) * lay.HALF_HOUR_US,
                "valid_us": grid_start_us + tidx.reshape(-1) * lay.HALF_HOUR_US,
                **{name: q[:, :, i].reshape(-1).astype(np.float32) for i, name in enumerate(QCOLS)},
                "mean": m.reshape(-1).astype(np.float32),
                "actual": (actual.reshape(-1) * target_std[pr] + target_mean[pr]).astype(np.float32),
            })
        )  # fmt: skip
    utc = pl.Datetime("us", "UTC")
    return (
        pl.concat(rows)
        .with_columns(
            power_fcst_init_time=pl.col("origin_us").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
            valid_time=pl.col("valid_us").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
        )
        .with_columns(
            nwp_init_time=pl.col("power_fcst_init_time").dt.truncate("1d"),
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0,
            seed=pl.lit(seed, dtype=pl.Int64),
        )
        .filter(pl.col("valid_time") < pl.lit(holdout_start).cast(utc))
        .drop("origin_us", "valid_us")
        .select(*FRAME_COLUMNS)
        .sort("time_series_id", "power_fcst_init_time", "valid_time")
    )  # fmt: skip


def write_frame_atomic(*, frame: pl.DataFrame, path: Path) -> Path:
    """Write a forecast frame under a temporary name that carries the process id, then rename it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.write_parquet(tmp)
    tmp.rename(path)
    return path


# --------------------------------------------------------------------------- the synthetic world


def predict_synthetic(
    *,
    net: model.UntangledNet,
    segments: Sequence[sy.SegmentArrays],
    index: sy.SampleIndex,
    d8_index: int,
    target_std: float,
    mode: str,
    device: str,
    batch_size: int = PREDICT_BATCH,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Forecast every sample of a split.

    Args:
        net: The fitted synthetic network (with the sign head).
        segments: The split's segments.
        index: Its samples.
        d8_index: The Primary index of D8.
        target_std: D8's target standard deviation.
        mode: The arm's channel mode.
        device: "cuda" or "cpu".
        batch_size: The batch size.

    Returns:
        `p_neg` `(N, 48)`, `quantiles` `(N, 48, 11)` and `mean` `(N, 48)` in standardised units,
        `target_z` `(N, 48)`, `sign_truth` `(N, 48)`, and the keys `seg`, `origin_day`, `lead_index`.

    Raises:
        ValueError: When the network has no sign head (the synthetic arms all have one).
    """
    if not net.has_sign_head:
        msg = "the synthetic forecaster has no sign head"
        raise ValueError(msg)
    net.eval()
    p, q, m, tz, st = [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            rows = np.arange(start, min(start + batch_size, len(index)))
            b = sy.gather_batch(
                segments=segments, index=index, rows=rows, d8_index=d8_index, target_std=target_std, mode=mode
            )  # fmt: skip
            out = net(train.synthetic_arrays_to_inputs(batch=b, device=device))
            assert out.sign_logit is not None
            p.append(train.to_numpy(tensor=torch.sigmoid(out.sign_logit)))
            q.append(
                train.to_numpy(
                    tensor=torch.sort(mdl.standardised_quantiles(out=out.out), dim=-1).values
                )
            )
            m.append(train.to_numpy(tensor=mdl.standardised_mixture_mean(out=out.out)))
            tz.append(b.target_z)
            st.append(b.sign_truth)
    cat = np.concatenate
    return {
        "p_neg": cat(p).astype(np.float32), "quantiles": cat(q).astype(np.float32),
        "mean": cat(m).astype(np.float32), "target_z": cat(tz), "sign_truth": cat(st),
        "seg": index.seg, "origin_day": index.origin_day, "lead_index": index.lead_index,
    }  # fmt: skip


def write_synthetic(*, arrays: dict[str, np.ndarray], path: Path) -> Path:
    """Write the synthetic predictions as one compressed archive, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)  # ty: ignore[invalid-argument-type]
    tmp.rename(path)
    return path


def read_synthetic(*, path: Path) -> dict[str, np.ndarray]:
    """Read the archive written by `write_synthetic`."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def check_stored_synthetic(*, arrays: dict[str, np.ndarray], n_expected: int) -> None:
    """Fail when a stored synthetic prediction has the wrong number of rows.

    A probability outside 0 to 1, or a non-finite value.

    This is the read of the stored output that the smoke test and the scorer both make.
    """
    if len(arrays["p_neg"]) != n_expected or n_expected == 0:
        msg = "the stored synthetic predictions hold another number of samples than the split"
        raise ValueError(msg)
    p = arrays["p_neg"]
    if not np.isfinite(p).all() or p.min() < 0.0 or p.max() > 1.0:
        msg = "a stored probability of reverse flow is non-finite or outside 0 to 1"
        raise ValueError(msg)
    if not np.isfinite(arrays["quantiles"]).all():
        msg = "a stored quantile is non-finite"
        raise ValueError(msg)


def compare_forecasts(
    *, new: pl.DataFrame, stored: pl.DataFrame, tolerance: float = core.TOLERANCE_F0R
) -> str:
    """Check that two frames hold identical keys and that every quantile is within `tolerance` of capacity."""
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    assert new.height == stored.height, (new.height, stored.height)
    assert new.select(keys).equals(stored.select(keys)), "the forecast keys differ"
    worst = max(float((new[c] - stored[c]).abs().max()) for c in QCOLS)  # ty: ignore[invalid-argument-type]
    assert worst <= tolerance, f"the largest difference is {worst}"
    return f"{new.height} rows, identical keys, largest difference {worst:.2e} of capacity"
