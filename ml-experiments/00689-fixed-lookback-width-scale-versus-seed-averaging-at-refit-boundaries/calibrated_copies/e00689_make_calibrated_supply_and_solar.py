"""Write `bsps_calibrated`, `gsps_calibrated`, and `solar_calibrated` from experiment 683's arm 2.

Run from the repository root with the output folder as the first argument, then the file names:

    PYTHONPATH=ml-experiments/00668-* uv run --no-sync python \
        ml-experiments/00689-*/calibrated_copies/e00689_make_calibrated_supply_and_solar.py \
        <output-folder> bsps gsps solar

Reads the stored forecast files in the backtest folder and the calibrated quantiles of experiment
683's arm 2 (`predictions/arm2/<name>.parquet`, written by `e00683_calibrate.py`), so nothing is
refitted. The copy keeps the stored file's keys and row order, takes the 11 quantiles from arm 2 as
Float32, and checks that `q50` is unchanged. These are the copies described under "Calibrated
copies" in the backtest folder's README.
"""

import sys
from pathlib import Path
from typing import Final

import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARM2: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files"
    / "predictions"
    / "arm2"
)
KEYS: Final[list[str]] = [
    "time_series_id",
    "valid_time",
    "nwp_init_time",
    "power_fcst_init_time",
]
ROW_GROUPS: Final[dict[str, int]] = {"bsps": 6, "gsps": 6, "solar": 16}
"""Number of row groups of each stored file."""
Q50_TOLERANCE: Final[float] = 1e-4


def make_copy(*, name: str, out_dir: Path) -> None:
    """Write `<name>_calibrated.parquet` into `out_dir`."""
    original = pl.read_parquet(lib.FOLDER / f"{name}.parquet")
    arm = pl.read_parquet(ARM2 / f"{name}.parquet")
    if original.height != arm.height or not original.select(KEYS).equals(arm.select(KEYS)):
        msg = f"{name}: the arm 2 file does not have the stored file's keys in the same order"
        raise ValueError(msg)
    out = original.select(KEYS).with_columns([arm[c].cast(pl.Float32) for c in lib.QCOLS])
    shift = (out["q50"].cast(pl.Float64) - original["q50"].cast(pl.Float64)).abs().max()
    if shift > Q50_TOLERANCE * max(1.0, original["q50"].abs().max()):
        msg = f"{name}: q50 changed by {shift}"
        raise ValueError(msg)
    out.write_parquet(
        out_dir / f"{name}_calibrated.parquet",
        compression="zstd",
        row_group_size=original.height // ROW_GROUPS[name] + 1,
    )
    print(name, out.height, "written", flush=True)


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True, exist_ok=True)
    for file_name in sys.argv[2:]:
        make_copy(name=file_name, out_dir=folder)
