# ruff: noqa: E501
"""Experiment 722: the parts that jobs 1, 2, and 3 share.

**A job refuses to start unless job 1's check file allows it.** `read_checks` raises when the
check file is missing, and the jobs read the generator-class table that job 1 wrote, so no job
chooses a generator's class a second time.
"""

import json
import os
from pathlib import Path
from typing import Any, Final

import e00722_core as core
import e00722_data as data
import e00722_guard as guard
import numpy as np
import polars as pl

CHECKS_NAME: Final[str] = "u0_checks.json"
"""Job 1's check file."""

CLASSES_NAME: Final[str] = "u0_generator_classes.parquet"
"""Job 1's generator-class table."""

LABELS_NAME: Final[str] = "u0_generator_labels.parquet"
"""Job 1's map from anonymous generator label to time series id; it lives under `PRIVATE/`."""

CLASS_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "bsp", "generator", "generator_type", "units", "primary", "era", "label", "coverage",
        "generator_class", "is_engine", "engine_like", "sign_ok", "subtract",
        "inside_polygon", "label_varies_by_era",
    }
)  # fmt: skip
"""The columns that jobs 2 and 3 read from job 1's class table."""

U2A_VERDICT_NAME: Final[str] = "u2a_verdict.json"
"""Job 2's verdict file, which job 3 reads."""

U2B_VERDICT_NAME: Final[str] = "u2b_verdict.json"
"""Job 3's verdict file."""

VARIANTS: Final[tuple[str, ...]] = ("primary", "with_unresolved")
"""Residual variants: the generators of class `outside_or_added_back` only, or those and the unresolved ones."""

ENV_DEFAULTS: Final[dict[str, str]] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}
"""The thread and GPU settings every job asserts (the launch wrapper sets them)."""


class JobRefusedError(RuntimeError):
    """Raised when an earlier job's output forbids this job."""


def assert_environment(*, env: dict[str, str]) -> None:
    """Raise unless every thread and GPU variable has the value of section 8.1."""
    wrong = {k: env.get(k) for k, v in ENV_DEFAULTS.items() if env.get(k) != v}
    if wrong:
        msg = f"launch through run.sh: these variables are wrong or unset: {sorted(wrong)}"
        raise JobRefusedError(msg)


def assert_one_core() -> None:
    """Raise unless this process may run on exactly one core (the wrapper pins it)."""
    allowed = os.sched_getaffinity(0)
    if len(allowed) != 1:
        msg = f"launch through run.sh: the process may use {len(allowed)} cores, not one"
        raise JobRefusedError(msg)


def read_checks(*, out_dir: Path = guard.OUT_DIR) -> dict[str, Any]:
    """Return job 1's check file, or raise when job 1 has not run."""
    path = out_dir / CHECKS_NAME
    if not path.is_file():
        msg = "run job 1 (e00722_u0.py) first: its check file is missing"
        raise JobRefusedError(msg)
    return json.loads(path.read_text())


def read_classes(
    *, out_dir: Path = guard.OUT_DIR, private_dir: Path = guard.PRIVATE_DIR
) -> pl.DataFrame:
    """Return job 1's class table with each generator's id joined from the private label map.

    Args:
        out_dir: The folder that holds the public class table.
        private_dir: The folder that holds the map from anonymous label to time series id.

    Returns:
        The class table with a `generator_id` column.

    Raises:
        JobRefusedError: When a column the jobs read is missing, so that drift between job 1 and
            the later jobs stops the job instead of changing a result.
    """
    classes = pl.read_parquet(out_dir / CLASSES_NAME)
    missing = CLASS_COLUMNS - set(classes.columns)
    if missing:
        msg = f"the class table lacks columns {sorted(missing)}; rerun job 1"
        raise JobRefusedError(msg)
    labels = pl.read_parquet(private_dir / LABELS_NAME)
    return classes.join(labels, on="generator", how="left", validate="m:1", maintain_order="left")


def require_class_audit(*, checks: dict[str, Any]) -> None:
    """Raise when the class audit failed (it stops U2b and U1)."""
    if not checks.get("class_audit_ok"):
        msg = "the class audit failed in job 1, which stops U2b and U1"
        raise JobRefusedError(msg)


def require_stamps(*, checks: dict[str, Any]) -> None:
    """Raise when the stamp alignment check of job 1 did not find lag 0."""
    if not checks.get("stamp_alignment_ok"):
        msg = "the stamp alignment check of job 1 did not peak at lag 0"
        raise JobRefusedError(msg)


def load_topology() -> data.Topology:
    """Join the site map, the metadata, and the hierarchy, and assert the design's membership."""
    site_map = pl.read_csv(data.SITE_MAP_PATH)
    metadata = pl.read_parquet(data.METADATA_PATH)
    hierarchy = pl.read_csv(data.HIERARCHY_PATH)
    topology = data.build_topology(site_map=site_map, metadata=metadata, hierarchy=hierarchy)
    data.assert_topology(topology=topology)
    return topology


def label_arrays(*, topology: data.Topology, power: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the Primaries' power by label."""
    rows = topology.primaries
    return {
        label: power[ts]
        for label, ts in zip(rows["label"].to_list(), rows["time_series_id"].to_list(), strict=True)
    }


def generator_ids_to_subtract(*, classes: pl.DataFrame, bsp: str, variant: str) -> list[int]:
    """Return the generators whose output enters BSP `bsp`'s residual under `variant`.

    The `primary` variant takes the generators that job 1 flagged `subtract`. The
    `with_unresolved` variant adds the generators of class `unresolved` that are metered in MW,
    cover enough of the span, and are not the engine's own series.
    """
    rows = classes.filter(pl.col("bsp") == bsp).unique(subset=["generator_id"])
    keep = pl.col("subtract")
    if variant == "with_unresolved":
        keep = keep | (
            (pl.col("generator_class") == "unresolved")
            & (pl.col("units") == "MW")
            & (pl.col("coverage") >= core.MIN_GENERATOR_COVERAGE)
            & ~pl.col("is_engine")
            & ~pl.col("engine_like")
            & pl.col("sign_ok")
        )
    return sorted(rows.filter(keep)["generator_id"].to_list())


def addback_candidates(*, classes: pl.DataFrame, primary: str) -> list[int]:
    """Return the ids of the generators that may be added back inside one Primary.

    A candidate is subtracted, lies inside the Primary's service-area polygon (column
    `inside_polygon`), and carries the label `added_back_or_not_below` at the Primary. The
    coefficient cannot tell an outside generator from an added-back one, and the polygon says
    which generators could sit below the Primary at all.
    """
    rows = classes.filter(
        (pl.col("primary") == primary)
        & (pl.col("label") == "added_back_or_not_below")
        & pl.col("subtract")
        & pl.col("inside_polygon")
    ).unique(subset=["generator_id"])
    return sorted(rows["generator_id"].to_list())


def both_start(*, series: dict[str, np.ndarray], transformers: tuple[str, ...]) -> int:
    """Return the first grid row at which every transformer has a reading, or the grid's length."""
    if not transformers:
        return 0
    have = np.all([np.isfinite(series[t]) for t in transformers], axis=0)
    rows = np.flatnonzero(have)
    return int(rows[0]) if rows.size else core.N_GRID


def spanned_residual(
    *,
    bsp: np.ndarray,
    series: dict[str, np.ndarray],
    children: tuple[str, ...],
    transformers: tuple[str, ...],
    leave_out: str | None,
    generators: list[np.ndarray],
    start: int,
) -> np.ndarray:
    """Return the residual that leaves one child out, with the two-transformer rule by span.

    Before row `start` the two transformers of 2025 have no readings and sit inside the
    remainder, so only the other children are subtracted. From row `start` every child is
    subtracted, and a half-hour where a transformer has no reading is NaN.
    """
    early_children = {c: series[c] for c in children if c not in transformers}
    late_children = {c: series[c] for c in children}
    early = core.leave_out_residual(
        bsp=bsp, children=early_children, leave_out=leave_out, generators=generators
    )
    late = core.leave_out_residual(
        bsp=bsp, children=late_children, leave_out=leave_out, generators=generators
    )
    rows = np.arange(len(bsp))
    return np.where(rows < start, early, late)
