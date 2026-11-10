"""Experiment 668: recast the Float64 quantiles of the BSP and GSP backtest files to Float32.

Reads `pre_float32/<name>.parquet` in the backtest folder, casts every quantile column
(`q01` to `q99`) to Float32 with Arrow, and writes `<name>.parquet` with the settings of the stored
file: ZSTD compression, dictionary encoding, the same row-group size as the original, and the same
row order. Every Float32 value equals `float32` of the Float64 original.

By default the script writes to `<out-dir>/<name>.parquet` (a scratch folder, never the stored
file), then prints the SHA-256 prefix of the result next to the prefix the backtest README states.
With `pyarrow` 25.0.1 the written bytes match the stored files (`f8f4b3ca0123b367266d` for
`bsps.parquet`, `86111148ef1be55331c7` for `gsps.parquet`). A different `pyarrow` version can write
different bytes for the same values, and the script then reports a mismatch without failing.

Usage: `python e00668_recast_float32.py <out-dir>`.
"""

import hashlib
import sys
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq
from studies.sources import ML_EXPERIMENTS_DATA_DIR

BACKTEST_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "backtest_year_2025-11_to_2026-10"
EXPECTED_PREFIX: Final[dict[str, str]] = {
    "bsps": "f8f4b3ca0123b367266d",
    "gsps": "86111148ef1be55331c7",
}
PREFIX_LENGTH: Final[int] = 20


def recast(*, name: str, out_dir: Path) -> Path:
    """Write the Float32 copy of `pre_float32/<name>.parquet` into `out_dir` and return its path."""
    source = BACKTEST_DIR / "pre_float32" / f"{name}.parquet"
    original = pq.read_table(source)
    row_group_size = pq.ParquetFile(source).metadata.row_group(0).num_rows
    quantile_columns = {c for c in original.column_names if c.startswith("q")}
    schema = pa.schema(
        [
            pa.field(f.name, pa.float32() if f.name in quantile_columns else f.type)
            for f in original.schema
        ]
    )
    out_path = out_dir / f"{name}.parquet"
    pq.write_table(
        original.cast(schema), out_path, compression="zstd", row_group_size=row_group_size
    )
    return out_path


def main(out_dir: Path) -> None:
    """Recast the BSP and GSP files and report whether the bytes match the stored files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in EXPECTED_PREFIX.items():
        path = recast(name=name, out_dir=out_dir)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()[:PREFIX_LENGTH]
        verdict = "matches" if actual == expected else "DOES NOT match"
        print(f"{name}: {actual} {verdict} the stored prefix {expected}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
