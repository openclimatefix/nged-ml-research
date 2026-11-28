"""Experiment 732's transfers, re-drawn, and the worlds that carry them into D8's record.

A world is D8's power plus the segment's synthetic engine and solar output plus the transfers
(Design A of the plan), and the mirror at the partner Primary when experiment 732 mirrored the
transfer. World `k` merges draw `k` of every test month that lies inside experiment 653's record.
Every addition is written at the panel index of the bank half-hour plus 1. When experiment 734's
`apply_additions` or `WorldDecoder` cannot take the engine and solar arrays, the probe raises,
unless the caller passes `allow_fallback`, in which case the builder falls back to transfers
alone. `world_design.json` records which of the two ran.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import polars as pl
from e00736_common import (
    COUNTED_HOURS,
    DATA_732,
    HALF_HOURS_PER_DAY,
    INDEX_OFFSET,
    N_DRAWS,
    N_MONTHS,
    OUT,
    POINT_MW,
    STEP_UNITS_COUNTED,
    atomic_json,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

GRID_START: Final[np.datetime64] = np.datetime64("2022-01-01")
"""First day of the offset model's panel (experiment 653's record)."""
GRID_LENGTH: Final[int] = 78816
"""Half-hours of the panel."""
READ_PARTNERS: Final[tuple[str, ...]] = ("D3", "D5", "D14")
"""Neighbours that experiment 732 mirrors, and that the 15-Primary panel holds."""
D8: Final[str] = "D8"
DESIGN_FULL: Final[str] = "A_with_engine_and_solar"
DESIGN_TRANSFERS_ONLY: Final[str] = "A_transfers_only"
TOLERANCE_G2: Final[float] = 1e-6
"""Relative tolerance of the injection check."""
WORLDS_DIR: Final[str] = "worlds"
BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("under 6 hours", 0.0, 6.0),
    ("6 to under 24 hours", 6.0, 24.0),
    ("24 to 96 hours", 24.0, float("inf")),
)


@dataclass(frozen=True)
class Env:
    """Experiment 732's test bank and its regenerated overlays."""

    segments: list[Any]
    overlays: list[Any]
    noise_d8: float
    digest: str


def noise_unit_d8() -> float:
    """Return D8's step noise unit in points from experiment 732's `limits.json`.

    Returns:
        The noise unit, 3.4176 points.
    """
    import json

    return float(json.loads((DATA_732 / "limits.json").read_text())["noise_units"]["D8"])


def load_env() -> Env:
    """Regenerate experiment 732's test overlays from its stored bank and neighbours.

    Returns:
        The segments, the overlays, D8's noise unit, and the overlays' digest.
    """
    # `e00732_paths` puts the folders of experiments 729 and 730 on `sys.path`, so it must be
    # imported before their modules; `importlib` keeps the import sorter from reordering them.
    import e00732_paths as paths  # ty: ignore[unresolved-import]

    banks729 = importlib.import_module("e00729_banks")
    banks730 = importlib.import_module("e00730_banks")
    cli = importlib.import_module("e00732_cli")
    ov = importlib.import_module("e00732_overlays")
    ctx = cli.real_context()
    base = cli.real_base(ctx=ctx)
    env = cli.real_env(root=paths.data_root(), ctx=ctx, base=base, with_real=False)
    bank = banks729.load_bank(path=banks730.bank_path(root=paths.root_730(), kind="test"))
    overlays = ov.overlay_segments(segments=bank.segments, kind="test", neighbours=env.neighbours)
    return Env(
        segments=list(bank.segments),
        overlays=overlays,
        noise_d8=noise_unit_d8(),
        digest=ov.overlay_digest(overlays=overlays),
    )


def grid_day_of(*, seg: Any) -> int:
    """Return the panel day of a segment's first dense day.

    Args:
        seg: A segment with `dates` (contiguous calendar days, `datetime64[D]`).

    Returns:
        Days from the panel's first day to the segment's first dense day (negative if before).
    """
    dates = np.asarray(seg.dates).astype("datetime64[D]")
    if not np.all(np.diff(dates).astype(int) == 1):
        msg = "a segment's dense days are not consecutive calendar days"
        raise ValueError(msg)
    return int((dates[0] - GRID_START).astype(int))


def inside_record(*, seg: Any) -> bool:
    """Return whether a segment's first real day lies inside the panel's record.

    Args:
        seg: A segment with `dates`.

    Returns:
        True when the day after the padding day is on or after the panel's first day.
    """
    return bool(np.asarray(seg.dates).astype("datetime64[D]")[1] >= GRID_START)


def months_and_draws(*, segments: Sequence[Any]) -> pl.DataFrame:
    """Return, for each test segment inside the record, its month index and draw index.

    The draw of a segment is its rank among the segments of the same month, in bank order. The
    month index orders the months inside the record from the oldest.

    Args:
        segments: The test bank's segments.

    Returns:
        A frame with `seg`, `month`, `month_idx`, and `draw`, one row per segment inside the record.
    """
    rows = [
        {"seg": i, "month": str(s.month)} for i, s in enumerate(segments) if inside_record(seg=s)
    ]
    frame = pl.DataFrame(rows, schema={"seg": pl.Int64, "month": pl.String})
    months = sorted(frame["month"].unique().to_list())
    return frame.with_columns(
        month_idx=pl.col("month").replace_strict(
            months, list(range(len(months))), return_dtype=pl.Int64
        ),
        draw=pl.col("seg").rank("ordinal").over("month").cast(pl.Int64) - 1,
    ).sort("seg")


def band_of(*, hours: float) -> str:
    """Return the name of the duration band of a transfer.

    Args:
        hours: The transfer's duration in hours.

    Returns:
        The band's name.
    """
    for name, lo, hi in BANDS:
        if lo <= hours < hi:
            return name
    return BANDS[-1][0]


def transfer_table(
    *, segments: Sequence[Any], overlays: Sequence[Any], noise_d8: float
) -> pl.DataFrame:
    """Return one row per injected transfer of the segments inside the record.

    Times are on the bank's dense axis (`start_hh`, `end_hh`) and on the panel (`start_panel`,
    `end_panel`, which add the +1 index offset). Durations are in hours (a tick is 5 minutes).

    Args:
        segments: The test bank's segments.
        overlays: The transfer overlays, one per segment.
        noise_d8: D8's step noise unit in points.

    Returns:
        The table, sorted by segment and start.
    """
    layout = months_and_draws(segments=segments)
    rows: list[dict[str, Any]] = []
    for r in layout.iter_rows(named=True):
        seg, o = segments[r["seg"]], overlays[r["seg"]]
        day0 = grid_day_of(seg=seg)
        scored = scored_mask(seg=seg)
        has_arrays = bool((np.asarray(seg.engine) != 0).any() or (np.asarray(seg.solar) != 0).any())
        for k, t in enumerate(sorted(o.transfers, key=lambda x: x.start_tick)):
            hours = t.ticks / 12.0
            rows.append(
                {
                    "seg": r["seg"],
                    "month_idx": r["month_idx"],
                    "draw": r["draw"],
                    "k": k,
                    "start_hh": int(t.start_half_hour),
                    "end_hh": int(t.end_half_hour),
                    "start_panel": day0 * HALF_HOURS_PER_DAY
                    + int(t.start_half_hour)
                    + INDEX_OFFSET,
                    "end_panel": day0 * HALF_HOURS_PER_DAY + int(t.end_half_hour) + INDEX_OFFSET,
                    "size_points": float(t.size),
                    "sign": 1 if t.size >= 0 else -1,
                    "units": abs(float(t.size)) / noise_d8,
                    "hours": hours,
                    "partner": str(t.partner),
                    "mirrored": bool(t.mirrored),
                    "shape": str(t.shape),
                    "start_scored": bool(scored[t.start_half_hour]),
                    "end_scored": bool(scored[t.end_half_hour]),
                    "has_arrays": has_arrays,
                }
            )
    frame = pl.DataFrame(rows)
    return frame.with_columns(
        band=pl.col("hours").map_elements(lambda h: band_of(hours=h), return_dtype=pl.String),
        big=pl.col("units") >= STEP_UNITS_COUNTED,
        counted=(pl.col("units") >= STEP_UNITS_COUNTED)
        & (pl.col("hours") >= COUNTED_HOURS[0])
        & (pl.col("hours") <= COUNTED_HOURS[1]),
    ).sort("seg", "start_hh")


def scored_mask(*, seg: Any) -> np.ndarray:
    """Return the flat mask of scored half-hours: base days with a valid, finite truth.

    Args:
        seg: A segment with `cal_index`, `valid`, and `x`.

    Returns:
        A boolean array of length `(D + 2) * 48`.
    """
    on_base = np.repeat(np.asarray(seg.cal_index) >= 0, HALF_HOURS_PER_DAY)
    return on_base & (np.asarray(seg.valid, dtype=bool) & np.isfinite(seg.x)).reshape(-1)


def with_gaps(*, table: pl.DataFrame, d8_power: np.ndarray) -> pl.DataFrame:
    """Add `gap`, true when a transfer cannot be scored.

    A transfer has a gap when its start half-hour is not a scored half-hour, or when D8's power has
    no reading in the 6-hour block of the panel that holds the start.

    Args:
        table: The transfer table.
        d8_power: D8's power on the panel, NaN where missing.

    Returns:
        The table with a `gap` column.
    """
    block = 12
    has = np.array(
        [
            bool(np.isfinite(d8_power[(s // block) * block : (s // block) * block + block]).any())
            if 0 <= s < len(d8_power)
            else False
            for s in table["start_panel"].to_list()
        ],
        dtype=bool,
    )
    return table.with_columns(gap=~(pl.Series(has) & pl.col("start_scored")))


def probe_design_a(*, allow_fallback: bool = False) -> dict[str, Any]:
    """Check that experiment 734's code can take engine and solar arrays as additions.

    The probe calls `apply_additions` on a toy power series at an MVA Primary and at a signed
    Primary, and reads the signature of `WorldDecoder`. A failed check raises, unless
    `allow_fallback` is true, in which case the transfers-only design is selected.

    Args:
        allow_fallback: Whether a failed check selects the transfers-only design.

    Returns:
        A record with `design` (the design that will run) and what each check found.

    Raises:
        RuntimeError: When a check failed and `allow_fallback` is false.
    """
    record: dict[str, Any] = {"checks": {}}
    ok = True
    try:
        import e00734_decode as dec  # ty: ignore[unresolved-import]
        import e00734_layers as layers  # ty: ignore[unresolved-import]

        power = {
            "A": np.array([1.0, 2.0, np.nan, 1.0, 5.0]),
            "B": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        }
        add = {
            "A": np.array([-3.0, 0.5, 1.0, 0.0, -1.0]),
            "B": np.array([-3.0, 0.0, 0.0, 0.0, 0.0]),
        }
        out = layers.apply_additions(power=power, additions=add, is_mva={"A": True, "B": False})
        want_a = np.array([2.0, 2.5, np.nan, 1.0, 4.0])
        want_b = np.array([-2.0, 1.0, 1.0, 1.0, 1.0])
        record["checks"]["apply_additions_mva_folds"] = bool(
            np.allclose(out["A"], want_a, equal_nan=True)
        )
        record["checks"]["apply_additions_signed_keeps_sign"] = bool(np.allclose(out["B"], want_b))
        params = inspect.signature(dec.WorldDecoder.__init__).parameters
        record["checks"]["world_decoder_takes_power"] = "power" in params
        ok = all(record["checks"].values())
    except Exception as exc:  # noqa: BLE001 - any failure selects the fallback and is recorded
        record["error"] = f"{type(exc).__name__}: {exc}"
        ok = False
    if not ok and not allow_fallback:
        msg = f"the design probe failed and no fallback was allowed: {record}"
        raise RuntimeError(msg)
    record["design"] = DESIGN_FULL if ok else DESIGN_TRANSFERS_ONLY
    return record


def _check_array_written(
    *, name: str, raw: np.ndarray, written: np.ndarray, keep: np.ndarray, segment: int
) -> None:
    """Raise when a segment's engine or solar array is not all zero but its addition is.

    The message names the array and the segment's index in the bank, and no value.

    Args:
        name: `engine` or `solar`.
        raw: The segment's array, flat.
        written: The segment's D8 addition so far, flat, in points.
        keep: Which half-hours of the segment lie inside the panel.
        segment: The segment's index in the bank.

    Raises:
        ValueError: When the array has nonzero half-hours inside the panel but the addition is
            zero at every one of them.
    """
    marked = keep & (raw != 0.0)
    if marked.any() and not np.any(written[marked] != 0.0):
        msg = f"segment {segment}: the {name} array is nonzero but its addition is zero everywhere"
        raise ValueError(msg)


def world_additions(
    *,
    segments: Sequence[Any],
    overlays: Sequence[Any],
    draw: int,
    design: str,
    sites: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return the additions, in megawatts, that turn D8's record into world `draw`.

    For each segment of the draw inside the record, D8 gains the transfers' mean load, and with
    `DESIGN_FULL` also loses the segment's engine and solar output (the truth of experiment 732 is
    `base - engine - solar + transfer`). A mirrored partner loses the jitter factor times the
    transfer. Every value is written at the panel index of the bank half-hour plus 1.

    Args:
        segments: The test bank's segments.
        overlays: The transfer overlays.
        draw: The draw (world) index.
        design: `DESIGN_FULL` or `DESIGN_TRANSFERS_ONLY`.
        sites: Every Primary of the panel.

    Returns:
        One array of `GRID_LENGTH` per Primary, zero where nothing is added.
    """
    import e00732_overlays as ov  # ty: ignore[unresolved-import]

    add = {s: np.zeros(GRID_LENGTH) for s in sites}
    layout = months_and_draws(segments=segments).filter(pl.col("draw") == draw)
    for r in layout.iter_rows(named=True):
        seg, o = segments[r["seg"]], overlays[r["seg"]]
        n = seg.x.size
        idx = grid_day_of(seg=seg) * HALF_HOURS_PER_DAY + np.arange(n) + INDEX_OFFSET
        keep = (idx >= 0) & (idx < GRID_LENGTH)
        d8 = np.asarray(o.t, dtype=np.float64).reshape(-1).copy()
        if design == DESIGN_FULL:
            for name in ("engine", "solar"):
                raw = np.asarray(getattr(seg, name), dtype=np.float64).reshape(-1)
                d8 -= raw
                _check_array_written(
                    name=name, raw=raw, written=d8, keep=keep, segment=int(r["seg"])
                )
        add[D8][idx[keep]] += d8[keep] * POINT_MW
        mirrors = ov.transfer_arrays(transfers=o.transfers, shape=seg.x.shape).mirror
        for partner in READ_PARTNERS:
            m = np.asarray(mirrors[partner], dtype=np.float64).reshape(-1)
            add[partner][idx[keep]] -= m[keep] * POINT_MW
    return add


def world_path(*, draw: int) -> Any:
    """Return the file of a world's additions.

    Args:
        draw: The draw index.

    Returns:
        The path under `OUT`.
    """
    return OUT / WORLDS_DIR / f"world_{draw}.npz"


def write_world(*, additions: dict[str, np.ndarray], draw: int, record: dict[str, Any]) -> None:
    """Write a world's additions, atomically, with a JSON record beside them.

    Args:
        additions: The additions per Primary.
        draw: The draw index.
        record: The world's record (design, counts, and the checks).
    """
    import os

    path = world_path(draw=draw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **additions)  # ty: ignore[invalid-argument-type]
    tmp.rename(path)
    atomic_json(obj=record, path=path.with_suffix(".json"))


def read_world(*, draw: int) -> dict[str, np.ndarray]:
    """Read a world's additions.

    Args:
        draw: The draw index.

    Returns:
        The additions per Primary.
    """
    with np.load(world_path(draw=draw)) as z:
        return {k: z[k] for k in z.files}


def build_world_power(
    *, power: dict[str, np.ndarray], additions: dict[str, np.ndarray], is_mva: dict[str, bool]
) -> dict[str, np.ndarray]:
    """Return a world's power: experiment 734's `apply_additions` on the real record.

    Args:
        power: The real power of every Primary.
        additions: The world's additions in megawatts.
        is_mva: Whether each Primary is an MVA meter.

    Returns:
        The world's power per Primary.
    """
    import e00734_layers as layers  # ty: ignore[unresolved-import]

    full = {s: additions.get(s, np.zeros_like(v)) for s, v in power.items()}
    return layers.apply_additions(power=power, additions=full, is_mva=is_mva)


__all__ = [
    "BANDS",
    "D8",
    "DESIGN_FULL",
    "DESIGN_TRANSFERS_ONLY",
    "GRID_LENGTH",
    "N_DRAWS",
    "N_MONTHS",
    "TOLERANCE_G2",
    "Env",
    "band_of",
    "build_world_power",
    "grid_day_of",
    "inside_record",
    "load_env",
    "months_and_draws",
    "noise_unit_d8",
    "probe_design_a",
    "read_world",
    "scored_mask",
    "transfer_table",
    "with_gaps",
    "world_additions",
    "world_path",
    "write_world",
]
