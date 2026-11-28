"""Decoding of the worlds with experiment 734's offset model, one world and arm per task.

The tuned values are read, unchanged, from experiment 734's `params.json`, after checking them
against the hash pinned in `e00736_common.py` and the hash that experiment stored beside it.
Nothing is refitted. A task writes the called and graded events of every Primary to
`OUT/decode/`.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import polars as pl
from e00736_common import ARMS, DATA_734, OUT, PARAMS_734_SHA256, atomic_json, sha256_of, time_log
from e00736_world import build_world_power, read_world

if TYPE_CHECKING:
    from pathlib import Path

PARAMS_734: Final[Path] = DATA_734 / "params.json"
PARAMS_734_SHA: Final[Path] = DATA_734 / "params.sha256"
"""The hash file that experiment 734 stored beside `params.json`."""
DECODE_DIR: Final[str] = "decode"
VALUE_TOLERANCE: Final[float] = 1e-6
"""Largest relative difference of a reproduced event's amplitude or confidence (gate G3)."""


def load_tuned() -> dict[str, Any]:
    """Read experiment 734's tuned values, read only, after checking their hash.

    Returns:
        The parsed `params.json`.

    Raises:
        ValueError: When the file's SHA-256 differs from the one pinned in this experiment or from
            the one stored beside it.
    """
    digest = sha256_of(path=PARAMS_734)
    stored = PARAMS_734_SHA.read_text().split()[0]
    if digest != PARAMS_734_SHA256 or digest != stored:
        msg = (
            f"params.json has digest {digest}, but this experiment pins {PARAMS_734_SHA256} and "
            f"params.sha256 holds {stored}"
        )
        raise ValueError(msg)
    return json.loads(PARAMS_734.read_text())


def events_path(*, draw: int, arm: str) -> Path:
    """Return the file of a world's decoded events for one arm.

    Args:
        draw: The world's draw index.
        arm: `H-own` or `H-nbr`.

    Returns:
        The path under `OUT`.
    """
    return OUT / DECODE_DIR / f"world_{draw}_{arm}_events.parquet"


def assert_c_kernel() -> None:
    """Stop unless the decode ran on the compiled Viterbi kernel, which keeps the time estimate.

    Raises:
        RuntimeError: When the numpy fallback ran instead of the C kernel.
    """
    import e00734_hmm as hmm  # ty: ignore[unresolved-import]

    if set(hmm.KERNELS_USED) != {"c"}:
        msg = f"the decode did not use the C kernel (kernels used: {sorted(hmm.KERNELS_USED)})"
        raise RuntimeError(msg)


def decode_task(*, draw: int, arm: str) -> None:
    """Decode one world with one arm and write its events.

    Args:
        draw: The world's draw index (the world's additions must exist).
        arm: `H-own` or `H-nbr`.

    Raises:
        ValueError: When `arm` is not one of the experiment's arms.
    """
    import e00734_decode as dec  # ty: ignore[unresolved-import]

    if arm not in ARMS:
        msg = f"arm {arm} is not one of {ARMS}"
        raise ValueError(msg)
    start = time.time()
    (OUT / DECODE_DIR).mkdir(parents=True, exist_ok=True)
    tuned = load_tuned()
    ctx = dec.load_context()
    rig = ctx.rig
    power = build_world_power(
        power=rig.panel.power, additions=read_world(draw=draw), is_mva=rig.panel.is_mva
    )
    decoder = dec.WorldDecoder(ctx=ctx, power=power, cal_cm=ctx.cal_cm, cal_sh=ctx.cal_sh)
    params = dec.arm_params(tuned=tuned, arm=arm)
    result = decoder.run_arm(arm=arm, params=params)
    assert_c_kernel()
    called = decoder.final_events(result=result, tau=params.tau)
    frame = dec.events_frame(events=called)
    import e00734_store as store  # ty: ignore[unresolved-import]

    blocks = store.site_blocks(post=result.posts["D8"], sigma=result.blocks.sigma["D8"])
    post_path = OUT / DECODE_DIR / f"world_{draw}_{arm}_d8_posterior.npz"
    np.savez(
        post_path.with_name(f".{post_path.stem}.tmp.npz"),
        mean_mw=blocks.mean_mw,
        p_zero=blocks.p_zero,
        sigma=blocks.sigma,
    )
    post_path.with_name(f".{post_path.stem}.tmp.npz").rename(post_path)
    path = events_path(draw=draw, arm=arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    frame.write_parquet(tmp)
    tmp.rename(path)
    time_log(stage="decode", task=f"world_{draw}_{arm}", start=start)


def uninjected_frame(*, arm: str = "H-nbr") -> pl.DataFrame:
    """Decode the real record with no addition and return the called events' table (gate G3).

    Args:
        arm: The arm to decode.

    Returns:
        The events table of `dec.events_frame`, `end` of -1 for an open event.
    """
    import e00734_decode as dec  # ty: ignore[unresolved-import]

    tuned = load_tuned()
    ctx = dec.load_context()
    rig = ctx.rig
    decoder = dec.WorldDecoder(ctx=ctx, power=rig.panel.power, cal_cm=ctx.cal_cm, cal_sh=ctx.cal_sh)
    params = dec.arm_params(tuned=tuned, arm=arm)
    result = decoder.run_arm(arm=arm, params=params)
    assert_c_kernel()
    called = decoder.final_events(result=result, tau=params.tau)
    return dec.events_frame(events=called)


def event_keys(*, frame: pl.DataFrame) -> list[tuple[str, int, int]]:
    """Return `(primary, start, end)` of every event of a table.

    Args:
        frame: An events table.

    Returns:
        The keys, `end` of -1 for an open event.
    """
    return [
        (str(r["primary"]), int(r["start"]), int(r["end"])) for r in frame.iter_rows(named=True)
    ]


def compare_event_values(
    *, mine: pl.DataFrame, stored: pl.DataFrame, rel: float = VALUE_TOLERANCE
) -> dict[str, Any]:
    """Compare the amplitude, confidence, and grade of two tables with the same event keys.

    Args:
        mine: The reproduced events table.
        stored: The stored events table.
        rel: The largest relative difference of `amp_mw` and `confidence`.

    Returns:
        `equal` (true when every event of one table has a partner of the other table with the same
        key, grade, and values within `rel`), the worst relative differences, and the number of
        grade mismatches.
    """
    keys = ["primary", "start", "end"]
    cols = [*keys, "amp_mw", "confidence", "grade"]
    joined = mine.select(cols).join(stored.select(cols), on=keys, how="full", suffix="_stored")
    worst: dict[str, float] = {}
    for col in ("amp_mw", "confidence"):
        a, b = joined[col].to_numpy(), joined[f"{col}_stored"].to_numpy()
        diff = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
        worst[col] = float("inf") if np.isnan(diff).any() else float(np.max(diff, initial=0.0))
    n_grade = int((joined["grade"] != joined["grade_stored"]).fill_null(True).sum())
    return {
        "equal": bool(all(v <= rel for v in worst.values()) and n_grade == 0),
        "worst_relative_difference": worst,
        "n_grade_mismatches": n_grade,
    }


def uninjected_events(*, arm: str = "H-nbr") -> list[tuple[str, int, int]]:
    """Decode the real record with no addition and return the called events' keys.

    Args:
        arm: The arm to decode.

    Returns:
        `(primary, start, end)` of every called event, `end` of -1 for an open event.
    """
    return event_keys(frame=uninjected_frame(arm=arm))


def stored_uninjected(*, arm: str = "H-nbr") -> pl.DataFrame:
    """Read experiment 734's stored un-injected event table, read only.

    Args:
        arm: The arm.

    Returns:
        The stored table.
    """
    return pl.read_parquet(DATA_734 / "uninjected" / f"{arm}_events.parquet")


def read_events(*, draw: int, arm: str) -> pl.DataFrame:
    """Read a world's decoded events.

    Args:
        draw: The world's draw index.
        arm: The arm.

    Returns:
        The events table of `dec.events_frame`.
    """
    return pl.read_parquet(events_path(draw=draw, arm=arm))


def write_marker(*, name: str, record: dict[str, Any]) -> None:
    """Write a small JSON record under `OUT/decode/`.

    Args:
        name: File name.
        record: The record.
    """
    atomic_json(obj=record, path=OUT / DECODE_DIR / name)


__all__ = [
    "assert_c_kernel",
    "compare_event_values",
    "decode_task",
    "event_keys",
    "events_path",
    "load_tuned",
    "read_events",
    "stored_uninjected",
    "uninjected_events",
    "uninjected_frame",
    "write_marker",
]
