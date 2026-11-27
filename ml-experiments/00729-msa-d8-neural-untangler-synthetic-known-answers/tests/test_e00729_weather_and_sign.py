# ruff: noqa: E501
"""Tests of the ERA5 wind and precipitation inputs, the sign-only arm's plan, and the ambiguity read-out."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import e00729_arms as arms
import e00729_banks as banks
import e00729_core as core
import e00729_data as data
import e00729_leakage as leak
import e00729_score as score
import numpy as np
import polars as pl
import pytest
from synthetic_world import make_calendar
from test_e00729_arms import STATS, seg
from test_e00729_score import make_rows


def test_the_arms_that_extend_arm_d_read_wind_and_precipitation() -> None:
    names_d = arms.stack_names(spec=arms.ARM_SPECS["D"])
    assert names_d[-3:] == ["t2m", "wind10", "precip"]
    for name in ("Dq", "D24", "E", "F", "G1", "G2", "Dsign"):
        assert arms.stack_names(spec=arms.ARM_SPECS[name]) == names_d
    for name in ("A", "B", "C", "H"):
        assert "wind10" not in arms.stack_names(spec=arms.ARM_SPECS[name])
    assert {"wind10", "precip"} <= leak.ALLOWED_INPUT_ROWS


def test_the_wind_and_precipitation_rows_are_standardised_and_logged() -> None:
    s, cal = seg()
    spec = arms.ARM_SPECS["D"]
    stack = arms.segment_stack(seg=s, spec=spec, holidays=cal.holidays, stats=STATS)
    names = arms.stack_names(spec=spec)
    wind = stack[names.index("wind10")]
    rain = stack[names.index("precip")]
    on = np.isfinite(s.wind10.reshape(-1))
    assert on.any()
    assert np.allclose(wind[on], (s.wind10.reshape(-1)[on] - 5.0) / 2.0, atol=1e-5)
    assert np.allclose(rain[on], np.log1p(s.precip.reshape(-1)[on]) / 0.5, atol=1e-5)
    assert (rain >= 0).all()
    assert (wind[~on] == 0).all()
    assert (rain[~on] == 0).all()


def test_input_stats_take_the_wind_mean_and_sd_over_the_training_days() -> None:
    cal = make_calendar()
    stats = arms.input_stats(cal=cal, train_days=np.arange(10))
    assert stats.w_mean == pytest.approx(float(cal.wind10[:10].mean()))
    assert stats.w_sd == pytest.approx(float(cal.wind10[:10].std()))


def test_the_weather_does_not_change_a_bank_or_its_digest() -> None:
    cal = make_calendar()
    use = {m: np.array([str(d)[:7] == m for d in cal.dates]) for m in ("2022-03",)}
    first = banks.build_bank(
        cal=cal,
        kind="test",
        seed=None,
        months=["2022-03"],
        month_use=use,  # ty: ignore[invalid-argument-type]
    )
    cal.wind10[:] = 0.0
    cal.precip[:] = 9.0
    second = banks.build_bank(
        cal=cal,
        kind="test",
        seed=None,
        months=["2022-03"],
        month_use=use,  # ty: ignore[invalid-argument-type]
    )
    assert first.digest == second.digest
    assert np.nanmax(second.segments[0].precip) == 9.0
    assert np.nanmax(first.segments[0].wind10) > 0.0


def test_the_era5_loader_aligns_wind_and_precipitation_with_the_half_hour_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    t0 = datetime(2022, 3, 1, tzinfo=UTC)
    n_rows = 97  # half-hours from the first instant to the end of the second day, inclusive
    stamps = [t0 + timedelta(minutes=30 * h) for h in range(n_rows)]
    whole = [h % 2 == 0 for h in range(n_rows)]
    frame = pl.DataFrame(
        {
            "time_series_id": [core.D8_SERIES_ID] * n_rows + [999] * n_rows,
            "valid_time": stamps + stamps,
            "wind_speed_10m": [float(h // 2) if w else -77.0 for h, w in enumerate(whole)]
            + [500.0] * n_rows,
            "precipitation": [(h // 2) / 3600.0 if w else 55.0 for h, w in enumerate(whole)]
            + [1.0] * n_rows,
        }
    )
    path = tmp_path / "era5.parquet"
    frame.write_parquet(path)
    monkeypatch.setattr(data, "ERA5_FEATURES", path)
    wind, rain = data.era5_wind_and_precipitation(t0=t0, n_steps=96)
    assert wind.shape == rain.shape == (96,)
    assert not np.any(
        wind < 0.0
    )  # the interpolated half-hour rows and the other series are ignored
    # wind at the whole hour k is k, and the pair of half-hours of hour k blends 0.25 of a neighbour
    assert wind[2 * 10] == pytest.approx(0.75 * 10 + 0.25 * 9)
    assert wind[2 * 10 + 1] == pytest.approx(0.75 * 10 + 0.25 * 11)
    # precipitation is the accumulation that ends at the next whole hour, in millimetres an hour
    assert rain[2 * 10] == rain[2 * 10 + 1] == pytest.approx(11.0)
    assert rain[-1] == pytest.approx(48.0)


def test_the_unit_of_the_precipitation_is_millimetres_an_hour() -> None:
    assert data.MM_PER_HOUR_PER_KG_M2_S == 3600.0


def test_the_sign_arm_is_the_13th_report_only_contrast_and_adds_three_fits() -> None:
    assert core.CONTRASTS["C13"] == ("Dsign", "D")
    assert core.REPORT_ONLY_CONTRASTS == ("C13",)
    assert len(core.CONTRASTS) - len(core.REPORT_ONLY_CONTRASTS) == core.N_CONTRASTS == 12
    assert pytest.approx(1.0 - 0.05 / 12) == core.LEVEL
    assert "Dsign" in core.ARMS
    assert "Dsign" in core.LADDER_ARMS
    assert core.N_FITS == 47 + 3
    assert "C13" not in core.BSP_CONTRASTS
    assert "C13" not in core.PINBALL_CONTRASTS


def test_the_ambiguity_bands_cut_where_the_constants_say() -> None:
    rows = make_rows()
    scale = core.POINTS / core.CAPACITY_MVA
    edges = [e * scale for e in core.READING_BAND_EDGES_MVA]
    pattern = np.array(
        [0.0, edges[0] - 1e-6, edges[0], edges[1] - 1e-6, edges[1], edges[2] - 1e-6, edges[2], 1e3]
    )
    rows = replace(rows, r=np.resize(pattern, rows.n))
    bands = score.ambiguity_strata(rows=rows)["reading"]
    assert list(bands) == ["MVA_0_to_0.3", "MVA_0.3_to_1", "MVA_1_to_3", "MVA_3_to_inf"]
    labels = np.array(list(bands))
    which = np.argmax(np.stack([bands[k] for k in labels]), axis=0)
    first = which[:8]
    assert first.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    for group in score.ambiguity_strata(rows=rows).values():
        total = np.sum(np.stack(list(group.values())), axis=0)
        assert (total == 1).all()


def test_the_truth_magnitude_bands_are_in_multiples_of_the_reactive_floor() -> None:
    rows = make_rows()
    q = core.QHAT_POINTS
    xs = np.array([0.0, 0.99 * q, -1.0 * q, 2.9 * q, -3.0 * q, 9.9 * q, 10.0 * q, -50.0])
    rows = replace(rows, x=np.resize(xs, rows.n))
    bands = score.ambiguity_strata(rows=rows)["truth_magnitude"]
    labels = list(bands)
    assert labels == ["QHAT_0_to_1", "QHAT_1_to_3", "QHAT_3_to_10", "QHAT_10_to_inf"]
    which = np.argmax(np.stack([bands[k] for k in labels]), axis=0)[:8]
    assert which.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_the_ambiguity_readout_reports_reliability_and_sharpness_per_band() -> None:
    rows = make_rows()
    q = core.QHAT_POINTS
    pattern = np.array([0.5 * q, -0.5 * q, 20.0, -20.0])
    rows = replace(rows, x=np.resize(pattern, rows.n))
    p = np.resize(np.array([0.5, 0.5, 0.0, 1.0]), rows.n)
    out = score.ambiguity_readout(rows=rows, p_neg=p)
    amb = out["truth_magnitude"]["QHAT_0_to_1"]
    far = out["truth_magnitude"]["QHAT_10_to_inf"]
    assert amb["mean_forecast"] == pytest.approx(0.5)
    assert amb["frequency"] == pytest.approx(0.5)
    assert amb["sharpness"] == pytest.approx(0.0)
    assert amb["reliability"] == pytest.approx(0.0)
    assert amb["confident_share"] == 0.0
    assert far["sharpness"] == pytest.approx(1.0)
    assert far["confident_share"] == 1.0
    assert far["frequency"] == pytest.approx(0.5)
    assert out["truth_magnitude"]["QHAT_1_to_3"] == {"n": 0.0}
    assert amb["n"] + far["n"] == rows.n


def test_the_wind_speed_follows_the_temperature_path_of_experiment_39s_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    t0 = datetime(2022, 3, 1, tzinfo=UTC)
    n_steps = 96
    hourly = np.linspace(0.0, 47.0, 49) ** 1.3  # a curved series, so a shift of a half-hour shows
    # the frame: one row per half-hour, labelled by its end, taking the hour stamped at (end + 29 min)
    # truncated to the hour; the loader then places the rows by their start and interpolates
    ends = [t0 + timedelta(minutes=30 * (p + 1)) for p in range(n_steps)]
    hour_of_row = [int((e + timedelta(minutes=29) - t0).total_seconds() // 3600) for e in ends]
    frame = pl.DataFrame(
        {"time": ends, "temperature_2m": [float(hourly[h]) for h in hour_of_row]}
    ).with_columns(start=pl.col("time") - pl.duration(minutes=30))
    temperature = data.hourly_to_half_hour(
        values=data._grid(frame=frame, t0=t0, n_steps=n_steps, column="temperature_2m")
    )
    stamps = [t0 + timedelta(minutes=30 * h) for h in range(n_steps + 1)]
    features = pl.DataFrame(
        {
            "time_series_id": [core.D8_SERIES_ID] * len(stamps),
            "valid_time": stamps,
            "wind_speed_10m": [
                float(hourly[h // 2]) if h % 2 == 0 else -1.0 for h in range(len(stamps))
            ],
            "precipitation": [0.0] * len(stamps),
        }
    )
    path = tmp_path / "era5.parquet"
    features.write_parquet(path)
    monkeypatch.setattr(data, "ERA5_FEATURES", path)
    wind, _ = data.era5_wind_and_precipitation(t0=t0, n_steps=n_steps)
    assert np.allclose(wind, temperature)
    # pin the alignment itself: the first half-hour of hour 10 leans on hour 9, the second on hour 11
    assert wind[20] == pytest.approx(0.75 * hourly[10] + 0.25 * hourly[9])
    assert wind[21] == pytest.approx(0.75 * hourly[10] + 0.25 * hourly[11])


def test_the_precipitation_of_a_half_hour_is_the_accumulation_of_the_hour_that_contains_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    t0 = datetime(2022, 3, 1, tzinfo=UTC)
    stamps = [t0 + timedelta(minutes=30 * h) for h in range(97)]
    features = pl.DataFrame(
        {
            "time_series_id": [core.D8_SERIES_ID] * 97,
            "valid_time": stamps,
            "wind_speed_10m": [1.0] * 97,
            "precipitation": [(h // 2) / 3600.0 if h % 2 == 0 else 9.0 for h in range(97)],
        }
    )
    path = tmp_path / "era5.parquet"
    features.write_parquet(path)
    monkeypatch.setattr(data, "ERA5_FEATURES", path)
    _, rain = data.era5_wind_and_precipitation(t0=t0, n_steps=96)
    # the half-hours 00:00 and 00:30 of hour 5 lie in the hour that ends at 06:00
    assert rain[10] == rain[11] == pytest.approx(6.0)
    assert rain[0] == rain[1] == pytest.approx(1.0)


def test_the_target_stack_holds_the_folded_reading_not_the_truth_magnitude() -> None:
    s, _ = seg(world="both")
    targets = arms.target_stack(seg=s)
    assert np.array_equal(targets[4], np.nan_to_num(s.r.reshape(-1)).astype(np.float32))
    on = np.isfinite(s.r.reshape(-1)) & (s.valid.reshape(-1))
    assert not np.allclose(targets[4][on], np.abs(s.x.reshape(-1))[on], atol=1e-4)
    assert (targets[4][on] >= np.abs(s.x.reshape(-1))[on] - 1e-4).all()
