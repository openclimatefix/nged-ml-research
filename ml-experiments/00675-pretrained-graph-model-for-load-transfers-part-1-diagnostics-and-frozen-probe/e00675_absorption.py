"""Does arm C's rolling 24-month window absorb a long real event? Compare residuals at D5 and D16.

For the verified D5 and D16 transfer, the mean residual in the event window and in the 28 days
before it, under arm C and under the year-cross-fitted copy (which never trains on the event's
year). A smaller event-window shift under arm C than under the cross-fit means arm C absorbs part
of the event.
"""

# ruff: noqa: E501  # research code run once; behaviour fixed by the results

import datetime as dt

import e00660_data as d  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR as DATA

ROOT = DATA / "00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe"


def main() -> None:
    """Print event-window shifts in MW and noise units."""
    panel, info = d.load_panel(), d.series_info()
    a = d.idx_of(when=dt.datetime(2025, 12, 29, tzinfo=dt.UTC))
    b = d.idx_of(when=dt.datetime(2026, 1, 13, tzinfo=dt.UTC))
    for s in ("D5", "D16"):
        p = panel.power[s]
        for name, f in (
            ("arm C", d.load_forecast(site=s)),
            ("cross-fit", np.load(ROOT / "forecast_loyo" / f"forecast_{info[s]['id']}.npy")),
            (
                "cross-fit, 45 pin",
                np.load(ROOT / "forecast_loyo_v45" / f"forecast_{info[s]['id']}.npy"),
            ),
        ):
            r = p - f
            shift = np.nanmean(r[a:b]) - np.nanmean(r[a - 28 * 48 : a])
            print(
                f"{s} {name}: event-window shift {shift:+.2f} MW ({shift / panel.noise[s]:+.1f} noise units)"
            )


if __name__ == "__main__":
    main()
