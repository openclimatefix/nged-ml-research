# ruff: noqa: E501
"""The fixed physical part of the solar term of experiment 405, and the capacity priors.

The unit shape is the AC output per megawatt of AC capacity of one plant with the population values
(tilt 35 degrees, azimuth 180, DC to AC ratio 1.15, the 0.4% per kelvin derating of
`studies.dp_gam`), driven by the ENS mean shortwave irradiance. The ENS spread enters through a
3-point Gauss-Hermite rule: the unit shape at the mean and at the mean plus and minus
`sqrt(3)` standard deviations. Nothing here is learned.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pvlib
from studies.pv_physics import (
    SunAndSky,
    ac_power_mw,
    plane_of_array_w_m2,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

TILT_DEG: Final[float] = 35.0
AZIMUTH_DEG: Final[float] = 180.0
DC_AC_RATIO: Final[float] = 1.15
DERATING_PER_KELVIN: Final[float] = 0.004
CELL_RISE_K: Final[float] = 25.0
REFERENCE_CELL_C: Final[float] = 25.0
GH_NODES: Final[tuple[float, float, float]] = (-(3.0**0.5), 0.0, 3.0**0.5)
GH_WEIGHTS: Final[tuple[float, float, float]] = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
MAX_ZENITH_DEG: Final[float] = 90.0

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00405-mdn-explicit-pv-term-learned-capacity-known-answer" / "work"
)
U99_PATH: Final[Path] = WORK_DIR / "u99.json"
STUDIES_TABLE: Final[Path] = Path(__file__).resolve().parents[2] / (
    "docs/studies/assets/embedded_pv_capacity_table.csv"
)
E269_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00269-pv-capacity-gain-decomposition" / "work" / "real_capacities.parquet"
)
SIGNED_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D15", "D16", "D18")
"""The Primaries metered in signed megawatts; the solar term runs only here."""
SIGNED_SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
"""The signed-MW six that every C-arm contrast uses; D15 and D18 are report-only."""
FIRST_VALID: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
"""The first month of the ENS summaries; month index 0."""


def _unit_ac(
    *,
    ghi: np.ndarray,
    cos_zenith: np.ndarray,
    azimuth_deg: np.ndarray,
    doy: np.ndarray,
    temperature_c: np.ndarray,
) -> np.ndarray:
    """Return the derated AC output per MW of AC capacity for each half-hour."""
    zenith = np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))
    ghi = np.where(cos_zenith > 0.0, np.clip(ghi, 0.0, None), 0.0)
    split = pvlib.irradiance.erbs(ghi=ghi, zenith=zenith, datetime_or_doy=doy)
    sky = SunAndSky(
        half_hour_end_time=np.zeros(len(ghi), dtype="datetime64[us]"),
        ghi_w_m2=ghi,
        dni_w_m2=np.nan_to_num(np.asarray(split["dni"], dtype=np.float64)),
        dhi_w_m2=np.nan_to_num(np.asarray(split["dhi"], dtype=np.float64)),
        zenith_deg=zenith,
        sun_azimuth_deg=azimuth_deg,
    )
    poa = plane_of_array_w_m2(sky=sky, tilt_deg=TILT_DEG, azimuth_deg=AZIMUTH_DEG, tracker=False)
    unit = ac_power_mw(poa_w_m2=poa, dc_capacity_mw=DC_AC_RATIO, ac_capacity_mw=1.0)
    cell = temperature_c + CELL_RISE_K * unit
    return unit * np.clip(1.0 - DERATING_PER_KELVIN * (cell - REFERENCE_CELL_C), 0.0, None)


def unit_shape(
    *,
    ghi_mean: np.ndarray,
    ghi_sd: np.ndarray,
    cos_zenith: np.ndarray,
    sin_azimuth: np.ndarray,
    cos_azimuth: np.ndarray,
    doy: np.ndarray,
    temperature_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and standard deviation of the unit shape, before the 99th-percentile scaling."""
    azimuth = np.degrees(np.arctan2(sin_azimuth, cos_azimuth)) % 360.0
    values = np.stack(
        [
            _unit_ac(
                ghi=ghi_mean + node * ghi_sd,
                cos_zenith=cos_zenith,
                azimuth_deg=azimuth,
                doy=doy,
                temperature_c=temperature_c,
            )
            for node in GH_NODES
        ]
    )
    weights = np.array(GH_WEIGHTS)[:, None]
    mean = (weights * values).sum(axis=0)
    variance = (weights * (values - mean) ** 2).sum(axis=0)
    return mean, np.sqrt(variance)


def u99() -> float:
    """Return the scaling constant: the 99th percentile of the mean shape over daylight rows.

    It is computed once from the ENS summaries of one Primary (D10) and stored, so that every Primary
    and every fold uses the same constant. A capacity of C megawatts means that the shape reaches C
    at its 99th percentile, the unit of `docs/studies/embedded-pv-capacity.md`.
    """
    if U99_PATH.exists():
        return float(json.loads(U99_PATH.read_text())["u99"])
    msg = "run `e00405_prep.py u99` first"
    raise FileNotFoundError(msg)


def month_index(*, valid_us: np.ndarray) -> np.ndarray:
    """Return months since 2024-04 for valid times in microseconds since the epoch."""
    t = pl.Series(valid_us).cast(pl.Datetime("us"))
    return ((t.dt.year() - 2024) * 12 + t.dt.month() - 4).to_numpy().astype(np.int64)


def season_year_index(*, valid_us: np.ndarray) -> np.ndarray:
    """Return the September-to-August season-year label minus 2023 (2023/24 is 0)."""
    t = pl.Series(valid_us).cast(pl.Datetime("us"))
    label = pl.select(
        pl.when(t.dt.month() >= 9).then(t.dt.year()).otherwise(t.dt.year() - 1)
    ).to_series()
    return (label - 2023).to_numpy().astype(np.int64)


def latest_complete_label(*, refit: datetime) -> int:
    """Return the latest September-to-August season-year label that ended before the refit."""
    return refit.year - 1 if refit.month >= 9 else refit.year - 2


def studies_prior(*, site: str, n_cells: int, refit: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Return the prior centre and standard deviation (MW) per season-year cell (2023/24 onward).

    The centre is the studies page's capacity and the standard deviation is a quarter of its 95%
    interval. For a flagged season-year the page gives no capacity, so the centre is the middle of
    the library range and the standard deviation is a quarter of the range. A cell whose season-year had not ended before the refit uses the latest completed
    season-year's value, so no prior or pinned capacity uses data on or after the refit.
    """
    table = pl.read_csv(STUDIES_TABLE).filter(pl.col("site") == site).sort("season_year")
    centre, sd = [], []
    cap = latest_complete_label(refit=refit)
    for label in range(2023, 2023 + n_cells):
        label = min(label, cap)
        years = table["season_year"].to_numpy()
        nearest = int(years[np.argmin(np.abs(years - label))])
        r = table.filter(pl.col("season_year") == nearest).row(0, named=True)
        if r["flagged"] and r["library_low_mw"] is not None:
            centre.append(0.5 * (r["library_low_mw"] + r["library_high_mw"]))
            sd.append(0.25 * (r["library_high_mw"] - r["library_low_mw"]))
        else:
            centre.append(r["capacity_mw"])
            sd.append(0.25 * (r["capacity_high_mw"] - r["capacity_low_mw"]))
    return np.array(centre), np.maximum(np.array(sd), 0.05)


def pinned_e269(*, site: str, n_cells: int, refit: datetime) -> np.ndarray:
    """Return experiment 269's physical-shape estimate (MW) per Sep-Aug season-year cell.

    Experiment 269 labels season-years from July. The cell for September to August year Y takes the
    value for the July-year Y (the first ten months of the cell fall in it).
    """
    table = (
        pl.read_parquet(E269_PATH)
        .filter(pl.col("site") == site, pl.col("shape") == "physical")
        .sort("season_year")
    )
    out = []
    cap = latest_complete_label(refit=refit)
    for label in range(2023, 2023 + n_cells):
        label = min(label, cap)
        years = table["season_year"].to_numpy()
        nearest = int(years[np.argmin(np.abs(years - label))])
        out.append(float(table.filter(pl.col("season_year") == nearest)["estimate_mw"][0]))
    return np.array(out)
