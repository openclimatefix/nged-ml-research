"""Print the reverse-flow rule's statistics at three MVA-only Primaries (experiment 381, arm A7).

The rule is experiment 366's, applied to the training rows of a refit (forecast shortwave of the
ENS control member, not ERA5 irradiance). The script prints, for the refits at 2025-07-01,
2025-12-01, and 2026-03-01, the number of sunny midday rows, the Pearson correlation of the readings
with shortwave, the 1st percentile of the readings (points of capacity), and the rule's flag.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack \
    uv run --no-sync python ml-experiments/00381-*/e00381_rf_check.py
```
"""

from datetime import UTC, datetime

import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00381_fit as fit  # ty: ignore[unresolved-import]
import numpy as np

SITES = ("D12", "D2", "D8")
REFITS = (
    datetime(2025, 7, 1, tzinfo=UTC),
    datetime(2025, 12, 1, tzinfo=UTC),
    datetime(2026, 3, 1, tzinfo=UTC),
)


def main() -> None:
    """Print the rule's statistics."""
    series, arrays = fd._series_and_arrays()  # noqa: SLF001
    config = fd.CONFIGS["n1h"]
    for site in SITES:
        row = series.filter(series["site"] == site).row(0, named=True)
        rows = fd._training_rows_cached(row=row, arrays=arrays[site], config=config)  # noqa: SLF001
        for refit in REFITS:
            known = fd.known_rows(rows=rows, refit=refit, config=config)
            flag, slope = fit.reverse_flow_rule(
                known=known, capacity_mw=row["effective_capacity_mw"]
            )
            cosine = np.cos(np.radians(known["solar_zenith"].to_numpy())).clip(min=0.0)
            clear = 1098.0 * cosine * np.exp(-0.057 / np.maximum(cosine, 0.01))
            shortwave = known["shortwave"].to_numpy().astype(float)
            hour = known["hour_local"].to_numpy()
            mask = (
                (clear > 1.0)
                & (shortwave > 0.6 * np.maximum(clear, 1.0))
                & (hour >= 10)
                & (hour <= 14)
            )
            reading = known["power"].to_numpy()[mask] / row["effective_capacity_mw"] * 100
            print(
                f"{site} refit {refit:%Y-%m-%d}: rows {mask.sum()}, "
                f"correlation {np.corrcoef(shortwave[mask], reading)[0, 1]:+.3f}, "
                f"1st percentile {np.percentile(reading, 1.0):.2f}, flag {flag}, slope {slope:.4f}"
            )


if __name__ == "__main__":
    main()
