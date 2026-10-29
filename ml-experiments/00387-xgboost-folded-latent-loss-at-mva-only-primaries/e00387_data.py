"""Series arrays for experiment 387: the metered series, or a synthetic unsigned (MVA) version.

The known-answer gate takes a signed-MW Primary, subtracts a synthetic photovoltaic output (a
multiple of the ERA5 shortwave, sized from the training window only) to make reverse flow, then
discards the sign: the synthetic apparent power is S = sqrt(P² + Q²), with Q set by a reactive
setting. The fit sees only S (labels and lags). The signed truth P stays for scoring.

Reactive settings: `none` (Q = 0), `pf97` (Q = P tan(arccos 0.97)), `q2` (Q = 2 points of
effective capacity), `pf97q2` (both: Q² = (P tan)² + 2²), `q4` (Q = 4 points).
"""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np

TRAIN_END: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
PEAK_FACTOR: Final[float] = 1.5
"""The injected output at 1000 W/m² is this multiple of the median sunny-midday training reading."""
SUNNY_SHORTWAVE: Final[float] = 600.0
MIDDAY_HOURS_UTC: Final[tuple[int, int]] = (10, 14)
TAN_PF97: Final[float] = float(np.tan(np.arccos(0.97)))
CONSTANT_Q_POINTS: Final[dict[str, float]] = {"q2": 2.0, "pf97q2": 2.0, "q4": 4.0}
REACTIVE_SETTINGS: Final[tuple[str, ...]] = ("none", "pf97", "q2", "pf97q2", "q4")
SIGNED_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
MVA_ONLY: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")


@dataclass(frozen=True)
class Synthetic:
    """A synthetic series: the fitted arrays and the signed truth, both over effective capacity."""

    arrays: common.SeriesArrays
    truth: np.ndarray
    injection_scale: float


def shortwave_of(*, arrays: common.SeriesArrays) -> np.ndarray:
    """Return the ERA5 shortwave of the grid (W/m²)."""
    return arrays.features[:, common.feature_columns().index("shortwave")].astype(float)


def synthesise(
    *, arrays: common.SeriesArrays, level: float, reactive: str
) -> Synthetic:
    """Inject photovoltaic output of size `level`, discard the sign, and add reactive power.

    The output at 1000 W/m² is `level` x `PEAK_FACTOR` x the median reading of the training
    window's sunny middays, so no value after 2025-07-01 sets the scale.
    """
    shortwave = shortwave_of(arrays=arrays)
    hours = arrays.valid_time.dt.hour().to_numpy()
    training = (arrays.valid_time < TRAIN_END).to_numpy()
    power = arrays.power.astype(float)
    sunny = (
        training
        & (shortwave > SUNNY_SHORTWAVE)
        & (hours >= MIDDAY_HOURS_UTC[0])
        & (hours <= MIDDAY_HOURS_UTC[1])
        & np.isfinite(power)
    )
    scale = level * PEAK_FACTOR * float(np.median(power[sunny])) / 1000.0
    truth = power - scale * np.nan_to_num(shortwave, nan=0.0)
    q_points = CONSTANT_Q_POINTS.get(reactive, 0.0) / 100.0
    q_tan = TAN_PF97 if reactive in ("pf97", "pf97q2") else 0.0
    apparent = np.sqrt(truth**2 + (truth * q_tan) ** 2 + q_points**2)
    return Synthetic(
        arrays=replace(arrays, power=apparent.astype(np.float32)),
        truth=truth.astype(np.float32),
        injection_scale=scale,
    )
