"""Injected copies of the target and lags for experiment 477, in fixed megawatts.

Experiment 409's helpers split a farm's output into a monthly daylight profile and a day-to-day
anomaly and shuffle days within a calendar month. This module injects, into the target and the
three power lags of every signed-direction Primary, a farm's output scaled to a fixed number of
megawatts. Conditions are named `control`, `held_<farm>_mw<N>`, `placebo_<farm>_mw<N>` (the farm's
daylight profile plus its anomaly shuffled by day within the calendar month), and
`profile_<farm>_mw<N>` (the daylight profile alone).

```bash
. <the environment script that sets DATA_PATH_INTERNAL and PYTHONPATH>
uv run --no-sync python ml-experiments/00477-*/e00477_inject.py
```
"""

import argparse
import re
from collections.abc import Callable
from pathlib import Path
from typing import Final

import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_prep as p5  # ty: ignore[unresolved-import]
import e00405_pv as pv  # ty: ignore[unresolved-import]
import e00409_prep as p9  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00477-neural-pv-module-placebo-gate-and-known-answer-on-stored-fits"
    / "work"
)
INJ_DIR: Final[Path] = WORK / "inj"
FARMS: Final[tuple[str, ...]] = ("A", "C")
"""Two of the metered solar farms A to F that experiments 405 and 409 could inject."""
CONDITIONS: Final[tuple[str, ...]] = (
    "control",
    "held_A_mw2",
    "held_A_mw4",
    "held_C_mw2",
    "held_C_mw4",
    "placebo_C_mw4",
    "profile_C_mw4",
)
PATTERN: Final = re.compile(r"(held|placebo|profile)_([A-F])_mw(\d+)")


def condition_function(
    *, condition: str
) -> tuple[Callable[[np.ndarray], np.ndarray] | None, float]:
    """Return the injected series as a function of times (microseconds), and its megawatts."""
    if condition == "control":
        return None, 0.0
    match = PATTERN.fullmatch(condition)
    assert match is not None, condition
    kind, farm, mw = match.group(1), match.group(2), float(match.group(3))
    series = p5.farm_norm_series(farm=farm)
    profile, anomaly = p9.split_profile_anomaly(series=series)

    def held(times: np.ndarray) -> np.ndarray:
        return p9.series_at(times_us=times, series=series, shuffle_salt=None)

    def profile_only(times: np.ndarray) -> np.ndarray:
        return p9.series_at(times_us=times, series=profile, shuffle_salt=None)

    def placebo(times: np.ndarray) -> np.ndarray:
        return profile_only(times) + p9.series_at(times_us=times, series=anomaly, shuffle_salt=0)

    return {"held": held, "profile": profile_only, "placebo": placebo}[kind], mw


def write_injection(*, site: str, condition: str) -> str:
    """Write an injected copy of a signed Primary's target and lags for a condition."""
    target = INJ_DIR / condition / f"{site}.npz"
    if target.exists():
        return f"{condition} {site}: exists"
    z = np.load(base.RAW_DIR / "summary" / f"{site}.npz")
    row = next(r for r in base.primaries().iter_rows(named=True) if r["site"] == site)
    function, mw = condition_function(condition=condition)
    increment = mw / float(row["effective_capacity_mw"])
    union = [p5.farm_norm_series(farm=f) for f in p5.UNION_FARMS]
    valid = z["valid"]

    def injected(*, times: np.ndarray) -> np.ndarray:
        """Return the series at the times, NaN wherever any union farm has a gap (as in 405)."""
        gap = np.zeros(len(times), dtype=bool)
        for other in union:
            gap |= np.isnan(p9.series_at(times_us=times, series=other, shuffle_salt=None))
            gap |= np.isnan(p9.series_at(times_us=times, series=other, shuffle_salt=0))
        if function is None:
            return np.where(gap, np.nan, 0.0)
        return np.where(gap, np.nan, function(times))

    unique, inverse = np.unique(valid, return_inverse=True)
    y = z["y"].astype(np.float64) - increment * injected(times=unique)[inverse]
    lags = np.empty((len(valid), 3), dtype=np.float32)
    for j, tag_lag in enumerate(("origin", "day", "week")):
        raw = z["x"][:, p5.col(name=f"lag_{tag_lag}")].astype(np.float64)
        age = (
            z["x"][:, p5.col(name=f"lag_{tag_lag}_age")].astype(np.float64) * base.LEAD_SCALE_HOURS
        )
        lag_time = valid - np.rint(age * 2.0).astype(np.int64) * (p5.US_PER_HOUR // 2)
        uniq, inv = np.unique(lag_time, return_inverse=True)
        lags[:, j] = raw - increment * injected(times=uniq)[inv]
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp.npz")
    np.savez(tmp, y=y.astype(np.float32), lags=lags, increment_points=increment)
    tmp.rename(target)
    return f"{condition} {site}: increment {increment:.4f}, {int(np.isnan(y).sum()):,} dropouts"


def main() -> None:
    """Write the injected copies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for cond in args.conditions:
        for s in pv.SIGNED_SITES[: 2 if args.smoke else None]:
            print(write_injection(site=s, condition=cond), flush=True)


if __name__ == "__main__":
    main()
