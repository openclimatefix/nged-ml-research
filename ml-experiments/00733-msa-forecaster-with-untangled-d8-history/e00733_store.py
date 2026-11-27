# ruff: noqa: E501
"""Experiment 733: the names, folders, and row checks of the stored prediction sets (no torch).

A real-fold fit writes up to three sets, each through `experiment_store.write_predictions` with the
row set `ROW_SET_ID_REAL`:

- `<job>`: the arm's own prediction (the channels of the arm, the re-run of days D - 2 and D - 1);
- `<job>_masked`: reading R4, every channel and flag zero at prediction (FU only);
- `<job>_stored`: reading R12, the stored untangled outputs of days D - 2 and D - 1 in place of the
  re-run (FU only).

`<job>` is `job_name(...).lower()`. A validation set carries the suffix `_val` and the row set
`ROW_SET_ID_REAL + "_validation"`: it holds the fold's held-out validation origins whose targets
lie before the fold's start, and it is the only set that the smoke test and the scratch runs score,
so no test month is read. The test rows' keys must equal the keys of M*'s stored file of the same
fold and seed. `e00733_predictions` writes the sets (torch); this module names and reads them.
"""

from pathlib import Path
from typing import Final

import e00733_baselines as baselines
import e00733_core as core
import polars as pl

KEYS: Final = ("time_series_id", "power_fcst_init_time", "valid_time")
VARIANTS: Final = (None, "masked", "stored")
"""The variants of a real-fold prediction: the arm itself, R4, and R12."""
ROW_SET_VALIDATION: Final = f"{core.ROW_SET_ID_REAL}_validation"
EXPERIMENT_ID: Final = 733


class PredictionError(RuntimeError):
    """A prediction cannot be written or its row keys do not match."""


def job_name(*, arm: str, fold: str | None, seed: int) -> str:
    """Return the name of one fit (`e00733_fit.job_name` without the torch import)."""
    safe = arm.replace("*", "star")
    return f"{safe}_{fold}_s{seed}" if fold else f"{safe}_s{seed}"


def set_name(
    *, arm: str, fold: str | None, seed: int, variant: str | None = None, rows: str = "test"
) -> str:
    """Return the name of a prediction set: `job_name().lower()` with the variant and row suffixes.

    Args:
        arm: The arm.
        fold: The fold, or None on the synthetic world.
        seed: The seed.
        variant: None, "masked" (R4), or "stored" (R12).
        rows: "test" or "validation".

    Returns:
        The set's name, which matches `experiment_store.SET_NAME_PATTERN`.

    Raises:
        PredictionError: When `variant` or `rows` is unknown.
    """
    if variant not in VARIANTS or rows not in ("test", "validation"):
        msg = f"unknown prediction variant {variant!r} or row kind {rows!r}"
        raise PredictionError(msg)
    name = job_name(arm=arm, fold=fold, seed=seed).lower()
    return name + (f"_{variant}" if variant else "") + ("_val" if rows == "validation" else "")


def row_set_id(*, rows: str) -> str:
    """Return the row-set id of the test rows or of the validation rows."""
    return core.ROW_SET_ID_REAL if rows == "test" else ROW_SET_VALIDATION


def data_dir(*, root: Path) -> Path:
    """Return the folder of experiments' data that holds `root` (the store's `data_dir`)."""
    return root.parent


def set_folder(*, root: Path, name: str) -> Path:
    """Return the folder of a prediction set."""
    return root / "predictions" / name


def read_set(*, root: Path, name: str, rows: str = "test") -> pl.DataFrame:
    """Read a stored prediction set through the experiment store, checking its row set.

    Args:
        root: This experiment's data folder.
        name: The set's name.
        rows: "test" or "validation".

    Returns:
        The set's frame.
    """
    from studies import experiment_store

    return experiment_store.read_predictions(
        experiment_id=EXPERIMENT_ID, name=name, row_set_id=row_set_id(rows=rows),
        data_dir=data_dir(root=root),
    )  # fmt: skip


def assert_keys_match_m_star(*, frame: pl.DataFrame, fold: str, seed: int) -> int:
    """Check that the frame holds exactly the keys of M*'s stored file of the same fold and seed.

    Args:
        frame: A prediction frame (all 16 Primaries, native origins).
        fold: The fold.
        seed: The seed.

    Returns:
        The number of rows compared.

    Raises:
        PredictionError: When the keys differ or no row was compared.
    """
    stored = (
        pl.read_parquet(baselines.m_star_file(month=fold, seed=seed), columns=list(KEYS))
        .sort(*KEYS)
    )  # fmt: skip
    mine = frame.select(*KEYS).sort(*KEYS)
    if stored.height == 0 or not mine.equals(stored):
        msg = f"the {fold} seed {seed} prediction holds other keys than M*'s stored file ({mine.height} against {stored.height} rows)"
        raise PredictionError(msg)
    return mine.height
