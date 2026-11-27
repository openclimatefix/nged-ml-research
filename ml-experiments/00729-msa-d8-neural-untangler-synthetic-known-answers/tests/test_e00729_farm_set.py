# ruff: noqa: E501
"""Tests of the real `data.farm_set` on a small frame shaped like experiment 39's `frame.parquet`."""

from datetime import UTC, datetime, timedelta

import e00729_core as core
import e00729_data as data
import e00729_sim as sim
import numpy as np
import polars as pl
import pytest
from studies.pv_physics import solar_position

T0 = datetime(2023, 6, 13, tzinfo=UTC)
N_DAYS = 190
N_STEPS = N_DAYS * 48
DAYS = np.array(["2023-06-15", "2023-12-15"], dtype="datetime64[D]")
CSI = 0.6


def farm_frame(*, ghi_nan: bool = False) -> pl.DataFrame:
    """Return one farm's rows as `frame.parquet` holds them: period-end times, ghi_u zero at night."""
    ends = [T0 + timedelta(minutes=30 * (i + 1)) for i in range(N_STEPS)]
    mid = pl.Series([e - timedelta(minutes=15) for e in ends], dtype=pl.Datetime("us", "UTC"))
    zen, _ = solar_position(stamps=mid, latitude=52.0, longitude=-1.0)
    clear = np.where(zen < 90.0, 1000.0 * np.cos(np.radians(np.minimum(zen, 89.9))) ** 1.2, 0.0)
    ghi = np.where(clear > data.MIN_CLEAR_W_M2, CSI * clear, 0.0)
    return pl.DataFrame(
        {
            "site": "A", "kind": "farm", "time": pl.Series(ends, dtype=pl.Datetime("us", "UTC")),
            "power": np.clip(ghi / 1000.0, 0.0, 1.0).astype(np.float32),
            "ghi_u": np.full(N_STEPS, np.nan) if ghi_nan else ghi, "ghi_u_clear": clear,
            "zenith": zen, "temperature_2m": np.full(N_STEPS, 10.0, dtype=np.float32),
        }
    )  # fmt: skip


@pytest.fixture
def farms(monkeypatch: pytest.MonkeyPatch) -> sim.FarmSet:
    """Return the real `farm_set` output for a June and a December day."""
    sites = pl.DataFrame(
        {"site": ["A"], "latitude": [52.0], "longitude": [-1.0], "effective_capacity_mw": [1.0]}
    )
    monkeypatch.setattr("studies.pv_dataset.pv_sites", lambda: sites)
    return data.farm_set(dates=DAYS, frame=farm_frame(), t0=T0, n_steps=N_STEPS)


def test_the_sky_is_present_on_a_june_and_a_december_day(farms: sim.FarmSet) -> None:
    assert farms.sky_ok.tolist() == [[True, True]]
    assert farms.power_ok.tolist() == [[True, True]]


def test_the_old_whole_day_count_would_have_failed_this_fixture(farms: sim.FarmSet) -> None:
    finite = (farms.ghi > 0).sum(axis=2)
    assert (finite < core.MIN_HH_FARM_DAY).all()
    assert (finite > 0).all()


def test_the_sky_is_absent_when_the_irradiance_is_missing() -> None:
    sun_up = np.zeros((1, 2, 48), dtype=bool)
    sun_up[:, :, 14:34] = True
    ghi = np.where(sun_up, 500.0, np.nan)
    assert data.sky_present(ghi=ghi, sun_up=sun_up).tolist() == [[True, True]]
    ghi[0, 1, 14:30] = np.nan
    assert data.sky_present(ghi=ghi, sun_up=sun_up).tolist() == [[True, False]]
    assert not data.sky_present(ghi=ghi, sun_up=np.zeros_like(sun_up)).any()


def test_an_s2_solar_segment_has_output_on_a_clear_day(farms: sim.FarmSet) -> None:
    found = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        out, fallback, source, _ = sim.draw_solar(
            rng=rng, farms=farms, cal_days=np.array([0, 1]), record={}
        )
        if source != "S2":
            continue
        found += 1
        assert not fallback.any()
        assert (out.max(axis=1) > 0).all()
        assert out[:, :6].max() == 0.0
    assert found > 5
