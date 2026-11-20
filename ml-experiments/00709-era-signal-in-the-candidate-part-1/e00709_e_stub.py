"""Write phase E's one-row stand-in demand frames under phase E's own folder.

Experiment 667's `fold.install` writes a stand-in frame for each run that has no demand frame into
experiment 667's `inputs/frames`. Phase E must not write there. `redirect_stub_frames` wraps
`demand_frame_path` before `fold.install` runs, so the wrapper writes the stand-ins under
`E00667_ROOT` and `fold.install` sees an existing path and writes nothing.
"""

import os
from pathlib import Path

import e00667_extend as ext  # ty: ignore[unresolved-import]
import polars as pl


def redirect_stub_frames(*, e: object) -> None:
    """Make `e.common.demand_frame_path` write its stand-in frames under `E00667_ROOT`.

    Args:
        e: Experiment 574's module, which holds `common.demand_frame_path`.
    """
    stub_dir = Path(os.environ["E00667_ROOT"]) / "inputs" / "frames"
    stub_dir.mkdir(parents=True, exist_ok=True)
    real_path = e.common.demand_frame_path  # ty: ignore[unresolved-attribute]

    def demand_frame_path(*, run: object) -> Path:
        path = real_path(run=run)
        if path.exists():
            return path
        stub = stub_dir / f"init={run:%Y-%m-%d}.parquet"  # ty: ignore[invalid-argument-type]
        if not stub.exists():
            summary = ext.E607 / "ens_summary" / f"init={run:%Y-%m-%d}.parquet"  # ty: ignore[invalid-argument-type]
            origin = (
                pl.scan_parquet(summary).select(pl.col("power_fcst_init_time").unique()).collect()
            )
            assert origin.height == 1, summary
            tmp = stub.with_suffix(f".{os.getpid()}.tmp")
            origin.with_columns(nwp_init_time=pl.lit(run)).write_parquet(tmp)
            tmp.rename(stub)
        return stub

    e.common.demand_frame_path = demand_frame_path  # ty: ignore[unresolved-attribute]
