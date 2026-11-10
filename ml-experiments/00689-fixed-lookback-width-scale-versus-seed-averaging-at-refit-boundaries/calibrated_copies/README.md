# Scripts that wrote the four calibrated backtest copies

These scripts write the four calibrated copies that sit beside the stored forecast files in
`data/ml-experiments/backtest_year_2025-11_to_2026-10/`. Each belongs to the "Calibrated copies"
section of that folder's README. Both scripts take an output folder, so they never overwrite the
stored files.

- **`e00689_make_calibrated_supply_and_solar.py`** writes `bsps_calibrated.parquet`,
  `gsps_calibrated.parquet`, and `solar_calibrated.parquet`. It reads the calibrated quantiles of
  experiment 683's arm 2 (written by `ml-experiments/00683-*/e00683_calibrate.py`), keeps the stored
  file's keys and row order, and checks that `q50` is unchanged. Nothing is refitted.
- **`e00689_make_calibrated_primaries.py`** writes `primaries_calibrated.parquet` and
  `primaries_calibrated_scales.parquet`. It refits experiment 683's arm 2 rule on the prevailing
  rows of the stored file, holds the 2026-06 scale of Primaries 13 and 16 from 2026-07, and leaves
  the normal-running rows and the 2025-11 origin month as stored.

**All four copies reproduce the stored files byte for byte.** The SHA-256 prefixes of the copies
written to a separate folder match the table in the backtest folder's README (`5f437ab019ce216fdb94`,
`9883e547d475e5030d99`, `82dd3e6812c715823552`, `10b95677d9c873ac3c5f`), and the frames compare
equal.

Run from the repository root with `DATA_PATH_INTERNAL` set. The first script needs
`PYTHONPATH=ml-experiments/00668-*`, and the second needs
`PYTHONPATH=ml-experiments/00668-*:ml-experiments/00683-*`.
