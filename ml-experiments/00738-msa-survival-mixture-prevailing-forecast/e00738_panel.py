# ruff: noqa: E501
"""Run an injected panel through a network of experiment 713 or a level arm, and frame the forecast.

**Provenance.** The loop is a copy of `forecast_pass` of experiment 726's `e00726_msa_panel.py`
(itself a copy of `cmd_bank` of experiment 713's `e00713_era.py`). The original runs `argparse` and
`parse_args()` at import, so it cannot be imported; this copy takes a model function instead of
building the model, and has no import-time effect. Changes from the original: the model function
returns the quantiles and a dict of diagnostics (the level head's age and depth), the frame carries
the injected `actual`, the lead days are an argument, and every row whose valid time is on or
after 2026-07-01 (`common.HOLDOUT_START_US`) is dropped, as in experiment 713's own pass.

`assert_713_code` compares experiment 713's code hash with the hash that the stored `M1` fits carry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

import e00738_common as common
import numpy as np
import polars as pl

QCOLS: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
SCORED_LEAD_DAYS: Final[tuple[int, ...]] = tuple(range(1, 10))
"""Lead days 1 to 9 (zero-based) are the target rows 2 to 10: 24 h up to but not including 240 h."""
BATCH: Final[int] = 512
ModelFn = Callable[..., tuple[object, dict[str, np.ndarray]]]
"""Maps an assembled `MsaInputs` to (standardised quantiles (B, 48, 11) as a tensor, diagnostics)."""


class CodeHashError(AssertionError):
    """Experiment 713's code differs from the code that fitted the stored states."""


def assert_hash(*, actual: str, expected: str = common.CODE_HASH_713) -> None:
    """Raise unless the code hash equals the one the stored fits carry.

    Args:
        actual: The current hash of experiment 713's fit modules and recipe modules.
        expected: The hash on the stored fits.

    Raises:
        CodeHashError: When the two differ.
    """
    if actual != expected:
        msg = (
            f"experiment 713's code hash is {actual}, and the stored M1 fits carry {expected}. "
            "Put the frozen checkout of experiment 713 (`7c29114a8`) first on PYTHONPATH "
            "(E00738_713_DIR in run.sh) and rerun."
        )
        raise CodeHashError(msg)


def assert_713_code() -> str:
    """Assert experiment 713's code hash against the stored fits, and return it.

    Returns:
        The hash.
    """
    import e00713_arms as arms  # ty: ignore[unresolved-import]

    actual = arms.code_hash()
    assert_hash(actual=actual)
    return actual


def before_holdout(*, valid_us: Any) -> Any:
    """Return the mask of valid times that lie before 2026-07-01 (the rows a forecast may keep)."""
    return valid_us < common.HOLDOUT_START_US


def forecast_pass(
    *,
    model_fn: ModelFn,
    src: object,
    data: object,
    primaries: np.ndarray,
    native: np.ndarray,
    lead_days: tuple[int, ...],
    ids: np.ndarray,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the 11 quantiles (capacity shares) and the diagnostics at the native origins and leads.

    Args:
        model_fn: The model function.
        src: The `Source` whose history power and targets carry the panel (experiment 713's class).
        data: The fold's `FoldData`.
        primaries: The Primaries to forecast.
        native: The native origin indices (09:00 UTC), in grid steps.
        lead_days: The zero-based lead days.
        ids: The `time_series_id` of each Primary.

    Returns:
        The forecast frame (keys, `lead_day`, `actual`, quantiles) and the per-sample diagnostics
        frame (primary, origin, lead day, and whatever the model function returned).
    """
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00713_layout as lay  # ty: ignore[unresolved-import]
    import e00713_train as tr  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    c_all = native + lay.NATIVE_OFFSET_STEPS
    prim, cc = np.meshgrid(primaries, c_all, indexing="ij")
    prim, cc = prim.ravel(), cc.ravel()
    samples = tr.SampleSet(primary=prim, c_idx=cc, is_val=np.zeros(prim.size, bool))
    frames, diags = [], []
    mean, std = data.target_mean, data.target_std  # ty: ignore[unresolved-attribute]
    for lead in lead_days:
        for start in range(0, prim.size, BATCH):
            sel = np.arange(start, min(start + BATCH, prim.size))
            b = tr.batch_for(
                source=src,
                samples=samples,
                index=sel,
                lead=np.full(len(sel), lead),
                ensemble=True,
                cut=10**9,
                dropout=None,
            )
            assert src.ens is not None  # ty: ignore[unresolved-attribute]
            assert bool(
                src.ens.filled[  # ty: ignore[unresolved-attribute]
                    b.inputs.primary[:, None],
                    (b.c_idx // 48 - src.ens.first_day)[:, None],  # ty: ignore[unresolved-attribute]
                    (48 * b.inputs.lead_day[:, None] + src.steps),  # ty: ignore[unresolved-attribute]
                ].all()
            ), "a scored row has no ensemble inputs"
            with torch.no_grad():
                q_t, diag = model_fn(b.inputs)
            p = prim[sel]
            q = q_t.cpu().numpy() * std[p][:, None, None] + mean[p][:, None, None]  # ty: ignore[unresolved-attribute]
            q = np.sort(q, axis=-1)
            tidx = b.target_idx.cpu().numpy()
            actual = src.power_z[b.inputs.primary[:, None], b.target_idx].cpu().numpy()  # ty: ignore[unresolved-attribute]
            actual = actual * std[p][:, None] + mean[p][:, None]
            origin_us = (
                prep422.GRID_START_US + (cc[sel] - lay.NATIVE_OFFSET_STEPS) * prep422.STEP_US
            )
            frames.append(
                pl.DataFrame(
                    {
                        "time_series_id": np.repeat(ids[p], 48),
                        "origin_us": np.repeat(origin_us, 48),
                        "valid_us": prep422.GRID_START_US + tidx.reshape(-1) * prep422.STEP_US,
                        "lead_day": np.full(len(sel) * 48, lead),
                        "actual": actual.reshape(-1).astype(np.float32),
                        **{
                            name: q[:, :, i].reshape(-1).astype(np.float32)
                            for i, name in enumerate(QCOLS)
                        },
                    }
                )
            )
            if diag:
                diags.append(
                    pl.DataFrame(
                        {
                            "time_series_id": ids[p],
                            "origin_us": origin_us,
                            "lead_day": np.full(len(sel), lead),
                            **{
                                k: np.asarray(v).reshape(len(sel), -1)[:, 0]
                                for k, v in diag.items()
                            },
                        }
                    )
                )
    utc = pl.Datetime("us", "UTC")
    frame = (
        pl.concat(frames)
        # no forecast of the data from 2026-07-01 on, as in experiment 713's own forecast pass
        .filter(before_holdout(valid_us=pl.col("valid_us")))
        .with_columns(
            power_fcst_init_time=pl.col("origin_us")
            .cast(pl.Datetime("us"))
            .dt.replace_time_zone("UTC")
            .cast(utc),
            valid_time=pl.col("valid_us")
            .cast(pl.Datetime("us"))
            .dt.replace_time_zone("UTC")
            .cast(utc),
            time_series_id=pl.col("time_series_id").cast(pl.Int64),
        )
        .drop("origin_us", "valid_us")
        .select(*KEYS, "lead_day", "actual", *QCOLS)
        .sort(*KEYS)
    )
    assert (
        frame.select(pl.all_horizontal(pl.col(c).is_finite() for c in QCOLS)).to_series().all()
    ), "a forecast quantile is not finite"
    assert not frame.select(list(KEYS)).is_duplicated().any(), "a forecast key is repeated"
    diag_frame = pl.concat(diags) if diags else pl.DataFrame()
    return frame, diag_frame
