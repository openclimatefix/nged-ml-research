# ruff: noqa: E501
"""Shared pieces of experiment 635: the metered neighbour column, the arms, and the condition names.

Experiment 621's injected worlds are reused unchanged (`e00621_common.Injector`). Each world gets a
column of the normalised output of one metered farm near each of the six MW Primaries (never the
farm injected at that Primary), which the arm-b and arm-c networks read as a second sky input.
"""

from pathlib import Path
from typing import Final

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00405_prep as p5  # ty: ignore[unresolved-import]
import e00409_prep as p9  # ty: ignore[unresolved-import]
import e00621_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies import pv_dataset
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00635-pv-probe-metered-neighbour-pv-as-sky-input"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
CUTOFF: Final = "2026-06"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
ARMS: Final[tuple[str, ...]] = ("b", "c")
CANDIDATES: Final[tuple[str, ...]] = ("A", "B", "C", "F")
"""Farms that can be an input: the four of experiment 621's row mask. D (3% to 6% gaps) and E (10% to 36%) are left out."""
RADIUS_KM: Final = 30.0
METERED_SCALE: Final = 3.0
"""The normalised output (about 0 to 1) is multiplied by this to sit on the scale of the standardised weather."""
SHUFFLE_SALT: Final = 1
CONDITIONS: Final[tuple[str, ...]] = (
    "control_r0", "control_r1",
    "farm_mw2_r0", "farm_mw4_r0", "farm_mw2_r1", "farm_mw4_r1",
    "phys_mw2", "phys_mw4",
    "profile_mw4_r0", "profile_mw4_r1", "shuffled_mw4_r0", "shuffled_mw4_r1",
    "cloudlin_weak", "cloudsq_weak", "cloudlin_strong", "cloudsq_strong",
    "profile_mw4_r0_mshuf", "control_r0_mshuf",
)  # fmt: skip


def split(*, name: str) -> tuple[k.Condition, int, bool]:
    """Return (the experiment 621 condition, the rotation, whether the metered column is shuffled)."""
    shuffled = name.endswith("_mshuf")
    base = name.removesuffix("_mshuf")
    if base.startswith("control_r"):
        return k.parse(name="control"), int(base[-1]), shuffled
    cond = k.parse(name=base)
    return cond, cond.rotation, shuffled


def twin(*, name: str) -> str:
    """Return the uninjected twin (same input column) of a condition."""
    _, rotation, shuffled = split(name=name)
    return f"control_r{rotation}" + ("_mshuf" if shuffled else "")


def km(*, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the distance in kilometres between two points (equirectangular)."""
    r = np.pi / 180
    x = (lon2 - lon1) * r * np.cos((lat1 + lat2) / 2 * r)
    return float(6371.0 * np.hypot(x, (lat2 - lat1) * r))


def distances() -> pl.DataFrame:
    """Return the distance from each of the six Primaries' centroids to each metered NGED farm."""
    import os

    md = pl.read_parquet(f"{os.environ['DATA_PATH_INTERNAL']}/NGED/metadata.parquet")
    prim = prep.primaries().join(
        md.select("time_series_id", "area_center_lat", "area_center_lon"), on="time_series_id"
    )
    farms = pv_dataset.pv_sites()
    rows = []
    for p in prim.filter(pl.col("site").is_in(list(k.SIX))).iter_rows(named=True):
        for f in farms.iter_rows(named=True):
            d = km(lat1=p["area_center_lat"], lon1=p["area_center_lon"], lat2=f["latitude"], lon2=f["longitude"])
            rows.append({"site": p["site"], "farm": f["site"], "km": d})  # fmt: skip
    return pl.DataFrame(rows)


def assignment(*, rotation: int) -> dict[str, str]:
    """Return the input farm of each MW Primary: the nearest candidate within the radius other than the injected farm."""
    dist = distances()
    out = {}
    for j, site in enumerate(k.SIX):
        injected = k.FARMS[(j + rotation) % len(k.FARMS)]
        near = dist.filter(
            (pl.col("site") == site)
            & pl.col("farm").is_in(list(CANDIDATES))
            & (pl.col("farm") != injected)
            & (pl.col("km") <= RADIUS_KM)
        ).sort("km")
        assert near.height > 0, (site, rotation)
        out[site] = near["farm"][0]
    return out


def metered_column(
    *, primary: np.ndarray, t_us: np.ndarray, rotation: int, shuffled: bool
) -> np.ndarray:
    """Return the scaled metered column for rows (0 outside the six Primaries and at gaps)."""
    six = k.six_indices()
    assign = assignment(rotation=rotation)
    out = np.zeros(len(primary), dtype=np.float32)
    for site, idx in six.items():
        sel = primary == idx
        if not sel.any():
            continue
        v = p9.series_at(
            times_us=t_us[sel], series=p5.farm_norm_series(farm=assign[site]),
            shuffle_salt=SHUFFLE_SALT if shuffled else None,
        )  # fmt: skip
        out[sel] = np.nan_to_num(v, nan=0.0) * METERED_SCALE
    return out


def weights_path(*, arm: str, condition: str, seed: int) -> Path:
    """Return where a pretrained network is stored."""
    return WORK / "pretrained" / arm / f"{condition}_s{seed}.pt"
