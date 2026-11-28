"""Tests of the real loader on a small fixture shaped like the real files, and of gate G4."""
# ruff: noqa: D103, E402

from __future__ import annotations

import dataclasses
import datetime
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _n in ("39", "40", "653", "660"):
    for _d in HERE.parent.glob(f"{_n.zfill(5)}-*"):
        sys.path.append(str(_d))

import e00039_common as c39  # ty: ignore[unresolved-import]
import e00653_common as c653  # ty: ignore[unresolved-import]
import e00737_gates as gates
import e00737_inputs as inputs
import polars as pl
from e00737_common import CUT, GRID_LENGTH, GRID_START
from e00737_forward import PRIOR_FLEET

SITE = "D5"
SERIES_ID = 5


def _write_fixture(*, root: Path) -> None:
    """Write the four files that `load_primary` reads, with the real files' columns and dtypes."""
    pl.DataFrame(
        {
            "time_series_id": [SERIES_ID],
            "site": [SITE],
            "latitude": [52.0],
            "longitude": [-1.0],
            "effective_capacity_mw": [10.0],
        }
    ).write_parquet(root / "series.parquet")
    (root / "power").mkdir()
    (root / "forecast" / "C").mkdir(parents=True)
    ramp = np.arange(GRID_LENGTH, dtype=np.float32)
    np.save(root / "power" / f"power_{SERIES_ID}.npy", ramp)
    np.save(root / "forecast" / "C" / f"forecast_{SERIES_ID}.npy", ramp + 0.5)
    labels = inputs.hour_labels()
    rng = np.random.default_rng(0)
    day_index = np.clip(rng.uniform(0.2, 1.0, size=GRID_LENGTH // 48), 0.0, 1.0)
    day = np.arange(len(labels)) // 24
    mid_hour = labels.dt.hour().to_numpy() - 0.5
    day_of_year = labels.dt.ordinal_day().to_numpy()
    declination = np.radians(23.44) * np.sin(2 * np.pi * (day_of_year - 81) / 365.0)
    latitude = np.radians(52.0)
    cos_zenith = np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(
        declination
    ) * np.cos(np.radians(15.0 * (mid_hour - 12.0)))
    clear = 1000.0 * np.maximum(cos_zenith, 0.0) ** 1.2
    pl.DataFrame(
        {
            "site": SITE,
            "time": labels,
            "ghi_w_m2": clear * day_index[day],
            "clear_sky_ghi_w_m2": clear,
            "reliability": np.ones(len(labels)),
        }
    ).write_parquet(root / "cams.parquet")
    times = pl.datetime_range(
        GRID_START - timedelta(days=10), CUT, interval="1h", time_zone="UTC", eager=True
    )
    pl.DataFrame(
        {
            "valid_time": times,
            "time_series_id": SERIES_ID,
            "temperature_2m": 15.0
            + 9.0 * np.sin(np.arange(len(times)) / 24.0 / 9.0)
            + 2.0 * np.sin(np.arange(len(times)) * np.pi / 12.0),
            "wind_speed_10m": np.full(len(times), 3.0),
            "is_holiday": np.zeros(len(times), dtype=bool),
        }
    ).write_parquet(root / "features.parquet")


@pytest.fixture(scope="module")
def fixture_inputs(tmp_path_factory: pytest.TempPathFactory) -> inputs.PrimaryInputs:
    root = tmp_path_factory.mktemp("e737_fixture")
    _write_fixture(root=root)
    patch = pytest.MonkeyPatch()
    patch.setattr(c653, "SERIES_PATH", root / "series.parquet")
    patch.setattr(c39, "CAMS_DEMAND_POINTS_PATH", root / "cams.parquet")
    patch.setattr(inputs, "E327", root)
    patch.setattr(inputs, "FEATURES_PATH", root / "features.parquet")
    inputs.load_primary.cache_clear()
    loaded = inputs.load_primary(site=SITE)
    inputs.load_primary.cache_clear()
    patch.undo()
    return loaded


def test_real_loader_on_a_fixture_shaped_like_the_real_files(
    fixture_inputs: inputs.PrimaryInputs,
) -> None:
    inp = fixture_inputs
    assert inp.series_id == SERIES_ID
    assert len(inp.power) == GRID_LENGTH
    assert len(inp.sky.ghi_w_m2) == GRID_LENGTH
    assert len(inp.hourly) == GRID_LENGTH // 2
    assert inp.reliable.all()
    assert str(inp.end_time[0]).startswith("2022-01-01T00:30")
    assert str(inp.end_time[-1]).startswith("2026-07-01T00:00")
    assert inp.weather.temperature_columns.shape == (GRID_LENGTH, 4)
    # The sun is up in midsummer daytime and down at midnight.
    assert inp.sun_up.any()
    assert not inp.sun_up[:2].any()
    assert float(inp.sky.ghi_w_m2[:2].max()) == 0.0
    assert inp.hourly["ghi_w_m2"].to_numpy().max() > 100.0


def test_the_store_index_shifts_by_one_half_hour_onto_the_grid(
    fixture_inputs: inputs.PrimaryInputs,
) -> None:
    # The fixture's store holds `i` at store index `i`, and `i + 0.5` in the forecast.
    inp = fixture_inputs
    assert inp.power[0] == 1.0
    assert inp.power[12345] == 12346.0
    assert inp.power[GRID_LENGTH - 2] == GRID_LENGTH - 1
    assert np.isnan(inp.power[-1])
    assert inp.forecast[0] == 1.5
    assert np.isnan(inp.forecast[-1])
    # Store index k ends at GRID_START + 30 min x k, which is the end stamp of grid index k - 1.
    start = np.datetime64(GRID_START.replace(tzinfo=None), "us")
    k = int(inp.power[777])
    assert inp.end_time[777] == start + k * np.timedelta64(30, "m")


def test_the_holiday_flag_follows_the_local_date_of_each_half_hours_start(
    tmp_path: Path,
) -> None:
    times = pl.datetime_range(
        GRID_START - timedelta(days=10), CUT, interval="30m", time_zone="UTC", eager=True
    )
    local_date = times.dt.convert_time_zone("Europe/London").dt.date()
    pl.DataFrame(
        {
            "valid_time": times,
            "time_series_id": SERIES_ID,
            "temperature_2m": 10.0 + 8.0 * np.sin(np.arange(len(times)) / 100.0),
            "wind_speed_10m": 3.0,
            "is_holiday": (local_date == datetime.date(2022, 1, 3)),
        }
    ).write_parquet(tmp_path / "features.parquet")
    patch = pytest.MonkeyPatch()
    patch.setattr(inputs, "FEATURES_PATH", tmp_path / "features.parquet")
    try:
        flags = inputs.load_weather(series_id=SERIES_ID).holiday
    finally:
        patch.undo()
    # Half-hours that start on 3 January (grid index 96 to 143) are on the holiday. The half-hour
    # that ends at its midnight starts on 2 January.
    assert np.flatnonzero(flags).tolist() == list(range(96, 144))


def _delay(*, curve: np.ndarray, half_hours: float) -> np.ndarray:
    return gates.delayed_curve(curve=curve, shift=half_hours)


def test_the_scan_interpolates_a_fractional_delay_linearly() -> None:
    curve = np.arange(48, dtype=float)
    assert _delay(curve=curve, half_hours=0.5)[10] == 9.5
    assert _delay(curve=curve, half_hours=-0.25)[10] == 10.25
    assert _delay(curve=curve, half_hours=1.0)[10] == 9.0


def _g4_with_power_delayed(*, fixture: inputs.PrimaryInputs, half_hours: int) -> dict[str, Any]:
    unit = inputs.unit_term(inputs=fixture, fleet=PRIOR_FLEET)
    power = -2.0 * np.roll(unit, half_hours)
    return gates.g4(inputs={SITE: dataclasses.replace(fixture, power=power)})


def test_g4_passes_when_the_real_power_follows_the_model(
    fixture_inputs: inputs.PrimaryInputs,
) -> None:
    record = _g4_with_power_delayed(fixture=fixture_inputs, half_hours=0)
    assert record["primaries"][SITE]["best_shift"] == pytest.approx(0.0)
    assert record["pooled_best_shift"] == pytest.approx(0.0)
    assert record["passed"] is True
    assert record["diagnostic_integer_shift"]["primaries"][SITE]["best_shift"] == 0


def test_g4_scan_recovers_the_delay_of_a_late_power_series(
    fixture_inputs: inputs.PrimaryInputs,
) -> None:
    # With no west lean, a power series one half-hour late shows a best shift of +1.0, which the
    # pooled range [-0.5, +1.5] allows (the real fleet's lean already gives about +1.1).
    one = _g4_with_power_delayed(fixture=fixture_inputs, half_hours=1)
    assert one["primaries"][SITE]["best_shift"] == pytest.approx(1.0, abs=0.25)
    # A series two half-hours late (the lean plus the old loader's extra half-hour) fails.
    two = _g4_with_power_delayed(fixture=fixture_inputs, half_hours=2)
    assert two["primaries"][SITE]["best_shift"] >= 1.75
    assert two["passed"] is False
    early = _g4_with_power_delayed(fixture=fixture_inputs, half_hours=-2)
    assert early["passed"] is False


def test_g4_fails_a_lean_series_that_is_one_half_hour_later_again(
    fixture_inputs: inputs.PrimaryInputs,
) -> None:
    unit = inputs.unit_term(inputs=fixture_inputs, fleet=PRIOR_FLEET)
    lean = -2.0 * (0.9 * np.roll(unit, 1) + 0.1 * np.roll(unit, 2))  # about 1.1 half-hours late
    corrected = dataclasses.replace(fixture_inputs, power=lean)
    assert gates.g4(inputs={SITE: corrected})["passed"] is True
    uncorrected = dataclasses.replace(fixture_inputs, power=np.roll(lean, 1))  # about 2.1
    record = gates.g4(inputs={SITE: uncorrected})
    assert record["pooled_best_shift"] >= 2.0 - 1e-9
    assert record["passed"] is False


def test_g4_ignores_a_winter_evening_demand_peak(fixture_inputs: inputs.PrimaryInputs) -> None:
    unit = inputs.unit_term(inputs=fixture_inputs, fleet=PRIOR_FLEET)
    slot = np.arange(GRID_LENGTH) % 48
    month = (fixture_inputs.end_time - np.timedelta64(30, "m")).astype("datetime64[M]").astype(
        int
    ) % 12 + 1
    evening = np.where(np.isin(month, gates.WINTER_MONTHS) & (slot >= 32) & (slot <= 38), 5.0, 0.0)
    power = -2.0 * unit + evening
    record = gates.g4(inputs={SITE: dataclasses.replace(fixture_inputs, power=power)})
    assert record["primaries"][SITE]["best_shift"] == 0
    assert record["diagnostic_summer_minus_winter"]["decides_gate"] is False


def test_unreadable_contrast_fails_the_gate(fixture_inputs: inputs.PrimaryInputs) -> None:
    empty = dataclasses.replace(fixture_inputs, power=np.full(GRID_LENGTH, np.nan))
    record = gates.g4(inputs={SITE: empty})
    assert record["primaries"][SITE]["readable"] is False
    assert record["passed"] is False
