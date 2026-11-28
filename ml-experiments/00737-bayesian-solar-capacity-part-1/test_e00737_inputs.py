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
import e00737_worlds as worlds
import polars as pl
from e00737_common import CUT, GRID_LENGTH, GRID_START
from e00737_forward import PRIOR_FLEET

SITE = "D5"
GATE_PRIMARIES = ("D5", "D9", "D10", "D11", "D14", "D16")
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


def _unit(*, fixture: inputs.PrimaryInputs) -> np.ndarray:
    return inputs.unit_term(inputs=fixture, fleet=PRIOR_FLEET)


def _g4(
    *,
    fixture: inputs.PrimaryInputs,
    power: np.ndarray,
    farm: np.ndarray | None,
    others: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run G4 on six Primaries: D5 reads `power`, the other five read `others` (default on time)."""
    farms = [] if farm is None else [farm]
    on_time = -2.0 * _unit(fixture=fixture)
    powers = [power, *(others if others is not None else [on_time] * 5)]
    assert len(powers) == len(GATE_PRIMARIES)
    sites = {
        site: dataclasses.replace(fixture, power=each)
        for site, each in zip(GATE_PRIMARIES, powers, strict=True)
    }
    return gates.g4(inputs=sites, farms=farms)


def test_g4_passes_when_the_power_is_on_time_relative_to_a_farm(
    fixture_inputs: inputs.PrimaryInputs,
) -> None:
    unit = _unit(fixture=fixture_inputs)
    record = _g4(fixture=fixture_inputs, power=-2.0 * unit, farm=unit)
    assert record["primaries"][SITE]["best_shift"] == pytest.approx(0.0)
    assert record["primaries"][SITE]["farm_reference_shift"] == pytest.approx(0.0)
    assert record["pooled_difference"] == pytest.approx(0.0)
    assert record["passed"] is True
    assert record["diagnostic_integer_shift"]["primaries"][SITE]["best_shift"] == 0


def test_g4_fails_a_primary_one_half_hour_later_than_a_farm_like_series(
    fixture_inputs: inputs.PrimaryInputs,
) -> None:
    unit = _unit(fixture=fixture_inputs)
    late = _g4(fixture=fixture_inputs, power=-2.0 * np.roll(unit, 1), farm=unit)
    assert late["primaries"][SITE]["difference"] == pytest.approx(1.0, abs=0.25)
    # One Primary outside the bound leaves five inside: the gate passes and excludes that Primary.
    assert late["primaries"][SITE]["within_bound"] is False
    assert late["excluded_primaries"] == [SITE]
    assert late["passed"] is True
    # Half a half-hour later is inside both ranges.
    half = _g4(fixture=fixture_inputs, power=-1.0 * (unit + np.roll(unit, 1)), farm=unit)
    assert half["primaries"][SITE]["difference"] == pytest.approx(0.5, abs=0.25)
    assert half["passed"] is True
    assert half["excluded_primaries"] == []
    two = _g4(fixture=fixture_inputs, power=-2.0 * np.roll(unit, 2), farm=unit)
    assert two["excluded_primaries"] == [SITE]
    early = _g4(fixture=fixture_inputs, power=-2.0 * np.roll(unit, -1), farm=unit)
    assert early["excluded_primaries"] == [SITE]


def test_g4_needs_four_of_six_primaries_within_the_bound(
    fixture_inputs: inputs.PrimaryInputs,
) -> None:
    unit = _unit(fixture=fixture_inputs)
    late = -2.0 * (0.4 * unit + 0.6 * np.roll(unit, 1))  # about 0.6 half-hours late: inside
    later = -2.0 * (0.1 * unit + 0.9 * np.roll(unit, 1))  # about 0.9 half-hours late: outside
    on_time = -2.0 * unit
    # Two outside: four inside passes, and the two are excluded.
    two = _g4(
        fixture=fixture_inputs,
        power=later,
        farm=unit,
        others=[later, late, on_time, on_time, on_time],
    )
    assert two["excluded_primaries"] == ["D5", "D9"]
    assert abs(two["pooled_difference"]) <= 0.5
    assert two["passed"] is True
    # Three outside: only three inside, so the gate fails although the pooled difference passes.
    three = _g4(
        fixture=fixture_inputs,
        power=later,
        farm=unit,
        others=[later, later, on_time, on_time, on_time],
    )
    assert three["excluded_primaries"] == ["D5", "D9", "D10"]
    assert abs(three["pooled_difference"]) <= 0.5
    assert three["passed"] is False
    # Every Primary 0.65 half-hours late is inside each Primary's bound but outside the pooled one.
    slow = -2.0 * (0.35 * unit + 0.65 * np.roll(unit, 1))
    pooled = _g4(fixture=fixture_inputs, power=slow, farm=unit, others=[slow] * 5)
    assert pooled["excluded_primaries"] == []
    assert pooled["pooled_difference"] > 0.5
    assert pooled["passed"] is False


def test_g4_reference_follows_a_farm_that_is_itself_late(
    fixture_inputs: inputs.PrimaryInputs,
) -> None:
    unit = _unit(fixture=fixture_inputs)
    lean = 0.9 * np.roll(unit, 1) + 0.1 * np.roll(unit, 2)  # about 1.1 half-hours late
    record = _g4(fixture=fixture_inputs, power=-2.0 * lean, farm=lean, others=[-2.0 * lean] * 5)
    assert record["primaries"][SITE]["best_shift"] >= 1.0
    assert record["passed"] is True
    record = _g4(
        fixture=fixture_inputs,
        power=-2.0 * np.roll(lean, 1),
        farm=lean,
        others=[-2.0 * np.roll(lean, 1)] * 5,
    )
    assert record["pooled_difference"] >= 0.75
    assert record["passed"] is False


def test_g4_ignores_a_winter_evening_demand_peak(fixture_inputs: inputs.PrimaryInputs) -> None:
    unit = _unit(fixture=fixture_inputs)
    slot = np.arange(GRID_LENGTH) % 48
    month = (fixture_inputs.end_time - np.timedelta64(30, "m")).astype("datetime64[M]").astype(
        int
    ) % 12 + 1
    evening = np.where(np.isin(month, gates.WINTER_MONTHS) & (slot >= 32) & (slot <= 38), 5.0, 0.0)
    record = _g4(fixture=fixture_inputs, power=-2.0 * unit + evening, farm=unit)
    assert record["primaries"][SITE]["best_shift"] == 0
    assert record["diagnostic_summer_minus_winter"]["decides_gate"] is False


def test_unreadable_contrast_or_missing_farms_fail_the_gate(
    fixture_inputs: inputs.PrimaryInputs,
) -> None:
    unit = _unit(fixture=fixture_inputs)
    empty = _g4(fixture=fixture_inputs, power=np.full(GRID_LENGTH, np.nan), farm=unit)
    assert empty["primaries"][SITE]["readable"] is False
    assert empty["excluded_primaries"] == [SITE]
    assert empty["passed"] is True  # the other five are inside
    no_farms = _g4(fixture=fixture_inputs, power=-2.0 * unit, farm=None)
    assert no_farms["primaries"][SITE]["readable"] is False
    assert no_farms["passed"] is False


def test_farm_to_grid_shifts_a_period_end_label_to_the_start_index() -> None:
    start = np.datetime64(GRID_START.replace(tzinfo=None), "us")
    step = np.timedelta64(30, "m")
    # A row stamped with the end of grid index 0, 5, and the last index lands on that index.
    times = np.array([start + step, start + 6 * step, start + GRID_LENGTH * step])
    out = inputs.farm_to_grid(times=times, power=np.array([2.0, 4.0, 6.0]), capacity_mw=2.0)
    assert len(out) == GRID_LENGTH
    assert out[0] == 1.0
    assert out[5] == 2.0
    assert out[GRID_LENGTH - 1] == 3.0
    assert np.isfinite(out).sum() == 3
    # A row stamped with the grid's own start (the end of index -1) and a NaN reading are dropped.
    dropped = inputs.farm_to_grid(
        times=np.array([start, start + 2 * step]), power=np.array([1.0, np.nan]), capacity_mw=1.0
    )
    assert not np.isfinite(dropped).any()


def test_the_cut_drops_rows_after_the_cut_in_the_real_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hour = timedelta(hours=1)
    cams_times = [CUT - hour, CUT, CUT + hour, CUT + 2 * hour]
    pl.DataFrame(
        {
            "site": SITE,
            "time": pl.Series(cams_times, dtype=pl.Datetime("us", "UTC")),
            "ghi_w_m2": [1.0, 2.0, 3.0, 4.0],
            "clear_sky_ghi_w_m2": [1.0, 2.0, 3.0, 4.0],
            "reliability": 1.0,
        }
    ).write_parquet(tmp_path / "cams.parquet")
    monkeypatch.setattr(c39, "CAMS_DEMAND_POINTS_PATH", tmp_path / "cams.parquet")
    cams = inputs.load_cams(site=SITE)
    assert cams["ghi_w_m2"].to_list() == [1.0, 2.0]
    farm_times = pl.Series(
        [CUT - timedelta(minutes=30), CUT, CUT + timedelta(minutes=30)],
        dtype=pl.Datetime("us", "UTC"),
    )
    pl.DataFrame(
        {"time_series_id": SERIES_ID, "time": farm_times, "power": [1.0, 2.0, 3.0]}
    ).write_parquet(tmp_path / "farms.parquet")
    pl.DataFrame({"time_series_id": [SERIES_ID], "effective_capacity_mw": [1.0]}).write_parquet(
        tmp_path / "sites.parquet"
    )
    monkeypatch.setattr(inputs, "FARMS_PATH", tmp_path / "farms.parquet")
    monkeypatch.setattr(inputs, "FARM_IDS", (SERIES_ID,))
    monkeypatch.setattr(inputs, "pv_sites", lambda: pl.read_parquet(tmp_path / "sites.parquet"))
    farm = inputs.load_farm_per_unit(farm=0)
    # The row stamped at the cut is the last half-hour; the row after it is never read.
    assert farm[GRID_LENGTH - 1] == 2.0
    assert farm[GRID_LENGTH - 2] == 1.0
    assert np.isfinite(farm).sum() == 2


def test_the_cut_check_rejects_a_timestamp_after_the_cut() -> None:
    late = np.array([np.datetime64(CUT.replace(tzinfo=None), "us") + np.timedelta64(1, "us")])
    with pytest.raises(ValueError, match="after the cut"):
        inputs.assert_cut(times=late, what="probe")


def test_a_world_lowers_the_net_power_by_day_and_leaves_the_night_alone(
    fixture_inputs: inputs.PrimaryInputs,
) -> None:
    spec = next(s for s in worlds.world_specs(primary=SITE) if s.name == "main_5_ramp_737101")
    world = worlds.build_world(inputs=fixture_inputs, spec=spec)
    assert (world.injected >= 0.0).all()
    assert world.injected.max() > 0.0
    assert np.array_equal(world.power, fixture_inputs.power - world.injected, equal_nan=True)
    assert (world.injected[~fixture_inputs.sun_up] == 0.0).all()
    lowered = world.power[fixture_inputs.sun_up] <= fixture_inputs.power[fixture_inputs.sun_up]
    assert lowered[np.isfinite(world.power[fixture_inputs.sun_up])].all()


def test_g2_and_g3_pass_on_the_fixture_and_fail_on_a_broken_injection(
    fixture_inputs: inputs.PrimaryInputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert gates.g2(inputs={SITE: fixture_inputs})["passed"] is True
    assert gates.g3(inputs={SITE: fixture_inputs})["passed"] is True
    spec = next(s for s in worlds.world_specs(primary=SITE) if s.name == "main_5_ramp_737101")
    world = worlds.build_world(inputs=fixture_inputs, spec=spec)
    broken = dataclasses.replace(world, injected=np.zeros_like(world.injected))
    assert worlds.check_injection(inputs=fixture_inputs, world=broken)["passed"] is False
    real_build = gates.build_world

    def leaky(*, inputs: inputs.PrimaryInputs, spec: worlds.WorldSpec) -> worlds.World:
        built = real_build(inputs=inputs, spec=spec)
        return dataclasses.replace(
            built, injected=built.injected + 1e-3 * np.nan_to_num(inputs.power)
        )

    monkeypatch.setattr(gates, "build_world", leaky)
    assert gates.g3(inputs={SITE: fixture_inputs})["passed"] is False
