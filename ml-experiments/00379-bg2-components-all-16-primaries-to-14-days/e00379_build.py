"""Re-blend experiment 87's stored Primary members into BG2's component arms at all 16 Primaries.

Reads the stored refit predictions of experiment 87 (no fits), rebuilds BG2's five members for all
16 Primaries with experiment 87's own code (offset twins at D9, D10, D11, D14, and D16, spread
factor 1.15, 14-day correction on the stack member only), and blends every subset with experiment
87's rolling monthly weights (re-chosen on the grid for each subset). Writes each arm to
`work/arms/<arm>.parquet` (keys, 11 quantiles as fractions of effective capacity, `lead_hours`).
Arms: `full` (BG2), `drop_<member>`, and `only_<member>` for W, M, AN, PT, and ST. The staircase
arms and the no-offset and no-spread arms are read from experiment 87's stored arms by the scorer.
The gate compares `full` with the stored BG2 arm.

    PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
        OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00379-*/e00379_build.py
"""

import dataclasses
from pathlib import Path
from typing import Final

import e00087_assemble as asm  # ty: ignore[unresolved-import]
import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00379-bg2-components-all-16-primaries-to-14-days" / "work"
)
STORED_BG2: Final[Path] = common.WORK_DIR / "arms" / "p_s6_bg2.parquet"
MEMBERS: Final[tuple[str, ...]] = asm.BLEND_FIVE
GATE_POINTS: Final[float] = 0.001
"""The largest allowed gap between the re-blended and the stored BG2 quantile, in points."""


def write_arm(*, name: str, keys: pl.DataFrame, quantiles: np.ndarray) -> None:
    """Write one arm: keys, the 11 quantile columns (fractions of capacity), and the lead."""
    frame = keys.hstack(pl.DataFrame(quantiles, schema=list(common.LEVEL_COLUMNS))).with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    (OUT / "arms").mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUT / "arms" / f"{name}.parquet")
    print(f"wrote {name}: {frame.height:,} rows", flush=True)


def gate(*, keys: pl.DataFrame, full: np.ndarray) -> None:
    """Raise if the re-blended BG2 differs from the stored BG2 by more than `GATE_POINTS`."""
    stored = (
        pl.scan_parquet(STORED_BG2)
        .select(*asm.KEY_COLUMNS, *common.LEVEL_COLUMNS)
        .collect()
        .sort(*asm.KEY_COLUMNS)
    )
    mine = keys.hstack(pl.DataFrame(full, schema=list(common.LEVEL_COLUMNS))).sort(*asm.KEY_COLUMNS)
    joined = mine.join(stored, on=list(asm.KEY_COLUMNS), suffix="_stored", validate="1:1")
    gaps = np.stack(
        [100.0 * (joined[c] - joined[f"{c}_stored"]).abs().to_numpy() for c in common.LEVEL_COLUMNS]
    )
    print(
        f"gate: {joined.height:,} of {stored.height:,} stored rows matched; "
        f"max gap {np.nanmax(gaps):.5f} points, mean {np.nanmean(gaps):.6f}",
        flush=True,
    )
    if joined.height != stored.height or np.nanmax(gaps) > GATE_POINTS:
        msg = "the re-blended BG2 does not reproduce experiment 87's stored BG2"
        raise RuntimeError(msg)


def main() -> None:
    """Build every arm and run the gate."""
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = dataclasses.replace(
        asm.make_context(smoke=False, store=False), results_dir=OUT / "weights"
    )
    primary = ctx.sites_of(groups=("Primary",))
    print(f"{len(primary)} Primaries: {sorted(primary)}", flush=True)
    tables: dict[str, pl.DataFrame] = {}
    for member in MEMBERS:
        raw = asm.raw_member(ctx=ctx, member=member, offset="vintage", sites=primary)
        tables[member] = asm.finish_member(ctx=ctx, raw=raw, correct=member == "ST", spread=True)
        print(f"member {member}: {tables[member].height:,} rows", flush=True)
    keys, by_member = asm.align(tables=tables, strict=True)
    del tables
    actual = asm.actual_array(frame=keys, power_of=ctx.power_of)
    full, weights = asm.rolling_blend(
        keys=keys, actual=actual, members=[by_member[m] for m in MEMBERS], names=list(MEMBERS)
    )
    (OUT / "weights").mkdir(exist_ok=True)
    weights.write_parquet(OUT / "weights" / "full.parquet")
    gate(keys=keys, full=full)
    write_arm(name="full", keys=keys, quantiles=full)
    for dropped in MEMBERS:
        kept = [m for m in MEMBERS if m != dropped]
        blended, weights = asm.rolling_blend(
            keys=keys, actual=actual, members=[by_member[m] for m in kept], names=kept
        )
        weights.write_parquet(OUT / "weights" / f"drop_{dropped}.parquet")
        write_arm(name=f"drop_{dropped}", keys=keys, quantiles=blended)
    for member in MEMBERS:
        write_arm(name=f"only_{member}", keys=keys, quantiles=by_member[member])


if __name__ == "__main__":
    main()
