"""Data of experiment 00429: the 51-member weather of the hindcast and scored cases, cached.

Builds, per farm (W1 to W3) and case set (the daily hindcast and the 53 scored Tuesday runs), the
weather of every ensemble member as a float32 array `members_<set>_<farm>.npy` of shape
(cases, 51, 7) in the column order of `WEATHER`, and the stage-3 plant's power for every member as
`plant_<set>_<farm>.npy` of shape (cases, 51). Cases follow experiment 00347's order (hindcast
cases with an observed power, then the scored cases), so the arrays line up with
`e00347_s4_data.hindcast_frame()` and `scored_frame()`.

    PYTHONPATH=<347>:<230>:<303>:<310>:<326>:<336 folders> \
    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00429-*/e00429_data.py [W1 W2 W3]
"""

import os
import sys
import time
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_common as c303  # noqa: E402  # ty: ignore[unresolved-import]
import e00347_common as common  # noqa: E402  # ty: ignore[unresolved-import]
import e00347_plant as plant  # noqa: E402  # ty: ignore[unresolved-import]
import e00347_s4_inputs as inputs  # noqa: E402  # ty: ignore[unresolved-import]
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from studies.sources import ML_EXPERIMENTS_DATA_DIR  # noqa: E402

EXPERIMENT_ID: Final[int] = 429
SLUG: Final[str] = "unified-forecaster-wind-ablations"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
CACHE_DIR: Final[Path] = DATA_DIR / "cache"
FARMS: Final[tuple[str, ...]] = common.FARMS
WEATHER: Final[tuple[str, ...]] = c303.WEATHER_COLUMNS
"""Wind speed at 10 m and 100 m, direction sine and cosine at both heights, temperature."""
MEMBERS: Final[int] = c303.MEMBERS
PLANT_FILE: Final[str] = inputs.PLANT_FILES["y2"]
"""Stage 3's plant, the frozen plant of Wind farms research default 2."""


def member_array(*, rows: pl.DataFrame) -> np.ndarray:
    """Return the member weather of `rows` as (cases, 51, 7) float32."""
    flat = np.stack([rows[c].to_numpy().astype(np.float32) for c in WEATHER], axis=1)
    return flat.reshape(-1, MEMBERS, len(WEATHER))


def build_farm(*, farm: str) -> None:
    """Cache one farm's member weather and stage-3 plant powers for both case sets."""
    started = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sites = c303.site_table()
    series = int(sites.filter(pl.col("site") == farm)["time_series_id"][0])
    prior = common.prior_scales()[farm]
    parameters = plant.PlantParameters.from_dict(common.read_json(name=PLANT_FILE)[farm])
    sets = {
        "hindcast": inputs.hindcast_members(series=series, sites=sites),
        "scored": inputs.scored_members(series=series, sites=sites),
    }
    for set_name, (members, cases) in sets.items():
        usable = (
            cases["actual_norm"].is_finite().fill_null(value=False).to_numpy()
            if set_name == "hindcast"
            else np.ones(cases.height, dtype=bool)
        )
        power = inputs.expected_in_chunks(
            parameters=parameters, weather=inputs.member_weather(rows=members), prior_scale=prior
        ).reshape(-1, MEMBERS)
        array = member_array(rows=members)
        np.save(CACHE_DIR / f"members_{set_name}_{farm}.npy", array[usable])
        np.save(CACHE_DIR / f"plant_{set_name}_{farm}.npy", power[usable].astype(np.float32))
        elapsed = time.time() - started
        print(f"{farm} {set_name} {int(usable.sum())} cases {elapsed:.0f} s", flush=True)


def load(*, set_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (weather (cases, 51, 7), stage-3 plant power (cases, 51)), farms stacked W1 to W3."""
    weather = np.concatenate([np.load(CACHE_DIR / f"members_{set_name}_{f}.npy") for f in FARMS])
    power = np.concatenate([np.load(CACHE_DIR / f"plant_{set_name}_{f}.npy") for f in FARMS])
    return weather, power


def main() -> None:
    """Build the farms named on the command line (all three by default)."""
    for farm in sys.argv[1:] or list(FARMS):
        build_farm(farm=farm)


if __name__ == "__main__":
    main()
