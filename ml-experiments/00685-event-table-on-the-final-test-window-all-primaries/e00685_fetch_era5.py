"""Fetch the stage-1 ERA5 box for 2026-08-01 to 2026-10-31, without editing the fetch script.

`fetch_era5_stage1_box_arco.py` fixes its span in module constants. This wrapper overrides
`FIRST_HOUR` and `LAST_HOUR` in the two modules that hold them and calls the script's own `main`,
so the variables, cells, checks, and checkpointing are the script's. The output goes to this
experiment's folder, so the existing 2019-09 to 2026-07 files are never rewritten.

    cd studies/weather_downloads
    PYTHONPATH=. uv run --no-sync python \
        ../../ml-experiments/00685-*/e00685_fetch_era5.py --output-dir <folder>
"""

from __future__ import annotations

from datetime import UTC, datetime

import fetch_era5_solar_arco as solar
import fetch_era5_stage1_box_arco as box

FIRST: datetime = datetime(2026, 8, 1, tzinfo=UTC)
LAST: datetime = datetime(2026, 10, 31, 23, tzinfo=UTC)

for module in (solar, box):
    module.FIRST_HOUR = FIRST
    module.LAST_HOUR = LAST

if __name__ == "__main__":
    box.main()
