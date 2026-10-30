"""Surprise and detector columns of experiment 391, built from the stage 1 tables.

All arrays live on experiment 87's dense half-hourly grid (index 0 is 2019-09-01 00:00 UTC). A
detector origin is a grid index that is a multiple of 12 (00, 06, 12, and 18 UTC). The statistic Z
at detector origin o is the mean normal score `u` over the 24 half-hours before o (at least 18
present), and R is the mean residual in MW. Noise units standardise Z by the expanding median and
1.4826 times the expanding median absolute deviation of earlier origins (at least `MIN_ORIGINS`),
so no origin sees a later median.

Alarm rules follow `briefs/opus_detector_distillation_answer.md`, section 3: an alarm opens at the
second of two consecutive same-sign origins with |Z~| above k, ends when Z~ of that sign is below
k/2 at two consecutive origins, is capped at 14 days with re-opening, and merges with a same-sign
alarm that opens within 48 hours of its end. The neighbour veto drops an alarm that would open when
at least two eligible neighbours share its sign with |R_n| at least 0.25 |R_p| and the
opposite-sign neighbours' summed R is under 0.25 |R_p| (or under twice its noise).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

STEP_PER_ORIGIN: Final[int] = 12
WINDOW: Final[int] = 48
"""Half-hours in a window: 24 hours."""
MIN_PRESENT: Final[int] = 36
"""Present half-hours needed in a 24-hour window (18 hours of 24, 36 of 48 half-hours)."""
MIN_ORIGINS: Final[int] = 120
"""Earlier origins needed before a noise unit exists (30 days)."""
MAD_SCALE: Final[float] = 1.4826
CAP_ORIGINS: Final[int] = 56
"""A 14-day cap, in 6-hourly origins."""
MERGE_ORIGINS: Final[int] = 8
"""Same-sign alarms less than 48 hours apart merge."""
VETO_SHARE: Final[float] = 0.25

ADJACENCY: Final[dict[str, tuple[str, ...]]] = {
    "D2": ("D10", "D4", "D15", "D18", "D21"), "D3": ("D22", "D14", "D8"), "D4": ("D19", "D2"),
    "D5": ("D22", "D16", "D8"), "D8": ("D3", "D14", "D5"), "D9": ("D21",),
    "D10": ("D11", "D15", "D18", "D12", "D21", "D2"), "D11": ("D10", "D12", "D21"),
    "D12": ("D11", "D10", "D15", "D18", "D22", "D14"), "D14": ("D3", "D15", "D18", "D12", "D8"),
    "D15": ("D10", "D18", "D12", "D14", "D2"), "D16": ("D22", "D5"),
    "D18": ("D10", "D15", "D12", "D14", "D2"), "D19": ("D4", "D21"),
    "D21": ("D9", "D11", "D19", "D10", "D2"), "D22": ("D3", "D12", "D5", "D16"),
}  # fmt: skip
"""Experiment 298's static neighbour list in site codes (the veto's list, fixed in the plan)."""


@dataclass(frozen=True)
class Origins:
    """Per-site statistics at the 6-hourly detector origins.

    Attributes:
        z: Mean normal score over the window, NaN where fewer than `MIN_PRESENT` half-hours exist.
        r: Mean residual in MW over the same window.
        z_noise: Z in noise units (expanding standardisation), NaN before `MIN_ORIGINS`.
        r_sigma: Expanding robust scale of R in MW, NaN before `MIN_ORIGINS`.
    """

    z: np.ndarray
    r: np.ndarray
    z_noise: np.ndarray
    r_sigma: np.ndarray


def load_dense(*, directory: Path, site: str, size: int, first_time_index: np.ndarray) -> tuple:
    """Return `(u, r_mw)` on the dense grid for a site, NaN where the stage 1 table has no row.

    Args:
        directory: A version's folder of stage 1 tables.
        site: The Primary's label.
        size: Number of grid half-hours.
        first_time_index: Grid index of each stage 1 table row's valid time, in file order.

    Returns:
        The surprise `u` and the residual `r_mw`, each a float32 array of length `size`.
    """
    table = pl.read_parquet(directory / f"{site}.parquet")
    u, r = np.full(size, np.nan, np.float32), np.full(size, np.nan, np.float32)
    u[first_time_index] = table["u"].to_numpy()
    r[first_time_index] = table["r_mw"].to_numpy()
    return u, r


def window_means(*, values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Return the mean of `values` over the `WINDOW` half-hours before each origin index.

    NaN where fewer than `MIN_PRESENT` are finite. The window is `[o - WINDOW, o)`.
    """
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values.astype(np.float64)))])
    ccount = np.concatenate([[0], np.cumsum(np.isfinite(values))])
    start = np.clip(origins - WINDOW, 0, None)
    total = csum[origins] - csum[start]
    count = ccount[origins] - ccount[start]
    return np.where(count >= MIN_PRESENT, total / np.maximum(count, 1), np.nan)


def expanding_noise(*, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the expanding median and scale (1.4826 MAD) of the earlier finite values.

    The value at position j uses positions before j only; NaN until `MIN_ORIGINS` finite values.
    """
    median, scale = np.full(values.size, np.nan), np.full(values.size, np.nan)
    finite: list[float] = []
    for j, value in enumerate(values):
        if len(finite) >= MIN_ORIGINS:
            history = np.asarray(finite)
            centre = np.median(history)
            median[j], scale[j] = centre, MAD_SCALE * np.median(np.abs(history - centre))
        if np.isfinite(value):
            finite.append(float(value))
    return median, scale


def origin_statistics(*, u: np.ndarray, r: np.ndarray) -> Origins:
    """Return a site's `Origins` at every 6-hourly origin of the grid."""
    origins = np.arange(0, u.size, STEP_PER_ORIGIN)
    z, r_mean = window_means(values=u, origins=origins), window_means(values=r, origins=origins)
    centre, spread = expanding_noise(values=z)
    _, r_scale = expanding_noise(values=r_mean)
    return Origins(
        z=z, r=r_mean, z_noise=(z - centre) / np.where(spread > 0, spread, np.nan), r_sigma=r_scale
    )


def vetoed(
    *, p: int, j: int, sign: int, stats: dict[str, Origins], neighbours: tuple[str, ...], site: str
) -> bool:
    """Return whether the neighbour veto drops an alarm opening at origin `j` of site index `p`."""
    r_p = stats[site].r[j]
    eligible = [
        n
        for n in neighbours
        if n in stats and np.isfinite(stats[n].r[j]) and np.isfinite(stats[n].r_sigma[j])
    ]
    if len(eligible) < 1 or not np.isfinite(r_p):
        return False
    same = sum(
        1
        for n in eligible
        if np.sign(stats[n].r[j]) == sign and abs(stats[n].r[j]) >= VETO_SHARE * abs(r_p)
    )
    opposite = sum(-sign * stats[n].r[j] for n in eligible if np.sign(stats[n].r[j]) == -sign)
    noise = 2.0 * np.sqrt(sum(stats[n].r_sigma[j] ** 2 for n in eligible))
    covered = opposite >= VETO_SHARE * abs(r_p) and opposite > noise
    return same >= 2 and not covered


def alarm_state(
    *, stats: dict[str, Origins], k: float, veto: bool
) -> dict[str, dict[str, np.ndarray]]:
    """Return, per site and origin, the causal alarm state: open, sign, onset origin, veto tag.

    Arrays have one entry per detector origin. `onset` is the origin index of the first flagged
    window's start (NaN when closed); `tag` is 0 none, 1 alarm opened with no eligible neighbour,
    2 opened with at least one eligible neighbour. The end of an alarm is applied only from the
    origin at which it is known (two consecutive low origins), never earlier.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for p, (site, s) in enumerate(stats.items()):
        n = s.z_noise.size
        state = {
            "open": np.zeros(n, np.int8),
            "sign": np.zeros(n, np.int8),
            "onset": np.full(n, np.nan),
            "tag": np.zeros(n, np.int8),
        }
        is_open, sign, onset, opened_at, last_end = False, 0, 0, 0, -(10**9)
        low_run = 0
        for j in range(n):
            z = s.z_noise[j]
            if is_open:
                below = np.isfinite(z) and sign * z < k / 2
                low_run = low_run + 1 if below else 0
                if low_run >= 2 or j - opened_at >= CAP_ORIGINS:
                    is_open, last_end, low_run = False, j, 0
            if not is_open and j >= 1 and np.isfinite(z) and np.isfinite(s.z_noise[j - 1]):
                prev = s.z_noise[j - 1]
                if abs(z) > k and abs(prev) > k and np.sign(z) == np.sign(prev):
                    here = int(np.sign(z))
                    if not (
                        veto
                        and vetoed(
                            p=p,
                            j=j,
                            sign=here,
                            stats=stats,
                            neighbours=ADJACENCY.get(site, ()),
                            site=site,
                        )
                    ):
                        merge = (j - last_end < MERGE_ORIGINS) and here == sign and last_end > 0
                        is_open, sign, opened_at, low_run = True, here, j, 0
                        if not merge:
                            onset = (j - 1) * STEP_PER_ORIGIN - WINDOW
                        state["tag"][j] = (
                            2
                            if any(
                                nb in stats and np.isfinite(stats[nb].r[j])
                                for nb in ADJACENCY.get(site, ())
                            )
                            else 1
                        )
            if is_open:
                state["open"][j], state["sign"][j], state["onset"][j] = 1, sign, onset
        out[site] = state
    return out
