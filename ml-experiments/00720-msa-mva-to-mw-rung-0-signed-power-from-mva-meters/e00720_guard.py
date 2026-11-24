# ruff: noqa: E501, PLW0108
"""Experiment 720: the anonymisation guard that every figure and table script of this experiment calls.

The rule of `CLAUDE.md`: a metered generator's time series, an inferred generator state, a dated
generator value, or a unit size never leaves the private data folder. This module offers

- `private_path`: the only place a script writes a table or figure that holds anything dated;
- `refuse_dated_generator_values`: raises when a frame has a time column and a generator column;
- `public_aggregate`: returns a frame only if every column is on an allow-list of aggregate
  statistics and no column is a time axis;
- `hours_band` and `pool_bands`: the bands of hours run per month, for pooled publication only.

The guard works on column names and dtypes, so it cannot see inside a column. It stops a script from
writing a dated generator series by mistake, and it is not a substitute for review.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_FOLDER: Final[str] = "00720-msa-mva-to-mw-rung-0-signed-power-from-mva-meters"
PRIVATE_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_FOLDER / "private"
"""Where every dated table, posterior, and figure data of this experiment is written."""

GENERATOR_MARKERS: Final[tuple[str, ...]] = (
    "unit", "engine", "generator", "farm", "inject", "pv_hat", "pv_component", "n_run", "gas",
    "hours_run", "running", "p_any", "e_n", "flag", "signed_mean", "p_negative", "n_hat", "state", "gross", "rho",
)  # fmt: skip
"""Column-name fragments that mark a generator quantity."""
TIME_MARKERS: Final[tuple[str, ...]] = (
    "time",
    "date",
    "day",
    "month",
    "week",
    "stamp",
    "valid",
    "hour_of",
    "fold",
)
"""Column-name fragments that mark a time axis."""
ERA_LABELS: Final[frozenset[str]] = frozenset({"era_1", "era_2", "era_3", "era_4"})
"""The only values an `era` column may hold in a published table: opaque labels, never a date range."""
PUBLIC_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "primary", "level", "readout", "sky", "pass", "auc", "recall_at_5pct_false", "false_reverse_rate",
        "n_rows", "n_reverse", "n_import", "f1", "count_accuracy", "sign_accuracy", "brier",
        "reliability", "crps_signed", "crps_hard", "coverage_90", "band", "n_months", "era",
        "group", "n_negative", "share_negative", "min_reading_points", "rho_spearman", "p1_points", "gate_open",
    }
)  # fmt: skip
"""The columns a published aggregate may hold: scores and counts, never a dated or generator value."""
HOURS_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("none", 0.0, 0.5),
    ("1 to 50", 0.5, 50.5),
    ("51 to 150", 50.5, 150.5),
    ("151 to 300", 150.5, 300.5),
    ("over 300", 300.5, float("inf")),
)
"""Half-open bands `[lo, hi)` of hours run in a month. Every half-hourly count is a multiple of 0.5, so 0.5 hours is in
`1 to 50` and 50.5 hours is in `51 to 150`; no value falls between bands."""


def private_path(*, name: str) -> Path:
    """Return a path under the private folder for `name`, creating the folder."""
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    out = PRIVATE_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _has(*, columns: Iterable[str], markers: Sequence[str]) -> list[str]:
    return [c for c in columns if any(m in c.lower() for m in markers)]


def refuse_dated_generator_values(*, frame: pl.DataFrame) -> None:
    """Raise `ValueError` when `frame` has both a time axis and a generator column.

    A column counts as a time axis when its dtype is a date or datetime or its name holds a time
    marker, and as a generator column when its name holds a generator marker.
    """
    temporal = [
        c
        for c, t in frame.schema.items()
        if t in (pl.Date, pl.Datetime) or isinstance(t, pl.Datetime)
    ]
    temporal += _has(columns=frame.columns, markers=TIME_MARKERS)
    generator = _has(columns=frame.columns, markers=GENERATOR_MARKERS)
    if temporal and generator:
        msg = f"refusing to write a dated generator value: time columns {sorted(set(temporal))}, generator columns {generator}"
        raise ValueError(msg)


def public_aggregate(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return `frame` unchanged when every column is an allowed aggregate, else raise `ValueError`."""
    refuse_dated_generator_values(frame=frame)
    if "era" in frame.columns:
        bad = sorted({str(v) for v in frame["era"].to_list()} - ERA_LABELS)
        if bad:
            msg = f"era values {bad} are not opaque labels; use one of {sorted(ERA_LABELS)}"
            raise ValueError(msg)
    extra = [c for c in frame.columns if c not in PUBLIC_COLUMNS]
    if extra:
        msg = (
            f"columns {extra} are not on the public allow-list; write them under the private folder"
        )
        raise ValueError(msg)
    return frame


def hours_band(*, hours: float) -> str:
    """Return the band of hours run in a month."""
    for name, lo, hi in HOURS_BANDS:
        if lo <= hours < hi:
            return name
    return HOURS_BANDS[-1][0]


def pool_bands(*, monthly_hours: np.ndarray) -> pl.DataFrame:
    """Return the number of months in each band, pooled over the era with no month labels."""
    names = [hours_band(hours=float(h)) for h in monthly_hours]
    return pl.DataFrame({"band": [b[0] for b in HOURS_BANDS]}).with_columns(
        n_months=pl.col("band").map_elements(lambda b: names.count(b), return_dtype=pl.Int64)
    )
