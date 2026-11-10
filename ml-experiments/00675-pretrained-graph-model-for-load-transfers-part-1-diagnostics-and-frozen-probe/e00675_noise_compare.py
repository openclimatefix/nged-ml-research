"""Step 0b of experiment 675: out-of-fold residual noise per Primary for arm C and the cross-fits.

Noise unit: 1.4826 times the median absolute deviation of successive 6-hour block-mean residuals
(the definition of experiment 653), in MW, over 2024-01-01 onward and over 2022 to 2023.
Run with `PYTHONPATH` holding experiment 660's folder: `uv run --no-sync python <this file>`.
"""

# ruff: noqa: B007, PERF102  # research code run once; behaviour fixed by the results

import datetime as dt
import sys

import e00660_data as d  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR as DATA

ROOT = DATA / "00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe"
ARMS = {
    "C": None,
    "loyo": "forecast_loyo",
    "p649": "forecast_loyo_p649",
    "v45": "forecast_loyo_v45",
}


def spread(r: np.ndarray, lo: int, hi: int) -> float:
    """Noise unit of the residual between half-hours `lo` and `hi`."""
    x = r[lo:hi]
    n = len(x) // 12 * 12
    b = np.nanmean(x[:n].reshape(-1, 12), axis=1)
    dd = np.diff(b)
    dd = dd[np.isfinite(dd)]
    return float(1.4826 * np.median(np.abs(dd - np.median(dd))))


def main() -> None:
    """Print the table."""
    panel, info = d.load_panel(), d.series_info()
    cut = d.idx_of(when=dt.datetime(2024, 1, 1, tzinfo=dt.UTC))
    print("site  " + "  ".join(f"{a:>12s}" for a in ARMS) + "   (2024+ | 2022-23)")
    wins = dict.fromkeys(ARMS, 0)
    for s, i in info.items():
        p = panel.power[s]
        row = []
        for a, folder in ARMS.items():
            f = (
                d.load_forecast(site=s)
                if folder is None
                else np.load(ROOT / folder / f"forecast_{i['id']}.npy")
            )
            r = p - f
            row.append((spread(r, cut, len(p)), spread(r, 0, cut)))
        best = min(range(len(row)), key=lambda k: row[k][0])
        wins[list(ARMS)[best]] += 1
        print(f"{s:5s} " + "  ".join(f"{a:.3f}|{b:.3f}" for a, b in row))
    print("lowest 2024+ noise:", wins)


if __name__ == "__main__":
    sys.exit(main())
